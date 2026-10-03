use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine;
use playwright_rs::protocol::CDPSession;
use playwright_rs::server::channel_owner::ChannelOwner;
use playwright_rs::server::connection::{ConnectionExt, deserialize_arc_str};
use playwright_rs::{
    Browser, BrowserContext, BrowserContextOptions, ConnectOverCdpOptions, GotoOptions,
    LaunchOptions, Page, Playwright, ScreenshotOptions, SelectOption, Viewport, WaitForOptions,
    WaitUntil,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::browser::blockdetect::{detect_block_page, detect_block_page_with_url};
use crate::browser::blocklist;
use crate::error::CharlotteError;
use crate::ingest::markdown::body_html_to_markdown;

use super::backend::BrowserBackend;
use super::{AutomationProfile, BrowserConfig, NavigateOptions};

/// Minimal init script applied in [`AutomationProfile::Stealth`] mode (runs before page scripts).
const STEALTH_INIT_SCRIPT: &str = r#"
(() => {
  const patch = () => {
    try {
      Object.defineProperty(navigator, "webdriver", { get: () => undefined });
    } catch (_) {}
  };
  patch();
})();
"#;

const CONSOLE_CAPTURE_INIT_SCRIPT: &str = r#"
(() => {
  try {
    if (window.__charlotteConsoleBuffer) return;
    Object.defineProperty(window, "__charlotteConsoleBuffer", {
      value: [],
      configurable: false,
      enumerable: false,
      writable: false,
    });
    const push = (level, args) => {
      try {
        const line = args.map((a) => {
          if (typeof a === "string") return a;
          try { return JSON.stringify(a); } catch (_) { return String(a); }
        }).join(" ");
        const row = { level, text: line, ts: Date.now() };
        window.__charlotteConsoleBuffer.push(row);
        if (window.__charlotteConsoleBuffer.length > 500) {
          window.__charlotteConsoleBuffer.splice(0, window.__charlotteConsoleBuffer.length - 500);
        }
      } catch (_) {}
    };
    const levels = ["log","info","warn","error","debug"];
    for (const level of levels) {
      const orig = console[level];
      console[level] = (...args) => {
        push(level, args);
        return orig.apply(console, args);
      };
    }
  } catch (_) {}
})();
"#;

struct BrowserSession {
    playwright: Playwright,
    browser: Browser,
    #[allow(dead_code)]
    context: BrowserContext,
    page: Page,
    cdp_session: Option<CDPSession>,
    cdp_session_page_url: Option<String>,
    network_events: Arc<Mutex<VecDeque<Value>>>,
    config: BrowserConfig,
    last_web_search_at: Option<Instant>,
}

pub(crate) struct PlaywrightBackend {
    inner: Mutex<Option<BrowserSession>>,
    backend_id: &'static str,
}

impl PlaywrightBackend {
    pub(crate) async fn launch(config: BrowserConfig) -> Result<Self, CharlotteError> {
        let playwright = Playwright::launch().await?;
        let mut launch = LaunchOptions::new().headless(config.headless);
        if let Some(ref p) = config.executable_path {
            launch = launch.executable_path(p.clone());
        }
        if let Some(ref ch) = config.channel {
            launch = launch.channel(ch.clone());
        }
        let args = chromium_launch_args(&config);
        if !args.is_empty() {
            launch = launch.args(args);
        }
        let browser = playwright.chromium().launch_with_options(launch).await?;
        Self::attach(playwright, browser, config, "playwright").await
    }

    pub(crate) async fn connect_over_cdp(
        config: BrowserConfig,
        endpoint_url: &str,
    ) -> Result<Self, CharlotteError> {
        let playwright = Playwright::launch().await?;
        Self::attach_over_cdp(playwright, config, endpoint_url).await
    }

    async fn attach_over_cdp(
        playwright: Playwright,
        config: BrowserConfig,
        endpoint_url: &str,
    ) -> Result<Self, CharlotteError> {
        let browser_type = playwright.chromium();
        let options = ConnectOverCdpOptions::new();
        let mut params = serde_json::Map::new();
        params.insert("endpointURL".into(), json!(endpoint_url));
        params.insert(
            "timeout".into(),
            json!(options.timeout.unwrap_or(playwright_rs::DEFAULT_TIMEOUT_MS)),
        );
        if let Some(slow_mo) = options.slow_mo {
            params.insert("slowMo".into(), json!(slow_mo));
        }
        if let Some(headers) = options.headers {
            let entries: Vec<Value> = headers
                .into_iter()
                .map(|(name, value)| json!({ "name": name, "value": value }))
                .collect();
            params.insert("headers".into(), Value::Array(entries));
        }
        let params = Value::Object(params);

        #[derive(Deserialize)]
        struct GuidRef {
            #[serde(deserialize_with = "deserialize_arc_str")]
            guid: Arc<str>,
        }

        #[derive(Deserialize)]
        struct ConnectOverCdpResponse {
            browser: GuidRef,
            #[serde(rename = "defaultContext")]
            default_context: Option<GuidRef>,
        }

        let response: ConnectOverCdpResponse = browser_type
            .channel()
            .send("connectOverCDP", params)
            .await?;
        let browser: Browser = browser_type
            .connection()
            .get_typed(&response.browser.guid)
            .await?;

        let context: BrowserContext = if let Some(ctx_ref) = response.default_context {
            browser_type.connection().get_typed(&ctx_ref.guid).await?
        } else {
            browser.new_context().await?
        };

        if matches!(config.automation_profile, AutomationProfile::Stealth) {
            context.add_init_script(STEALTH_INIT_SCRIPT).await?;
        }
        context.add_init_script(CONSOLE_CAPTURE_INIT_SCRIPT).await?;
        install_network_policy(&context, config.block_trackers_and_ads, config.block_fonts).await?;

        let page = if let Some(existing) = context.pages().into_iter().next() {
            existing
        } else {
            context.new_page().await?
        };

        let network_events = Arc::new(Mutex::new(VecDeque::new()));
        install_network_observers(&page, Arc::clone(&network_events)).await?;
        let session = BrowserSession {
            playwright,
            browser,
            context,
            page,
            cdp_session: None,
            cdp_session_page_url: None,
            network_events,
            config,
            last_web_search_at: None,
        };
        Ok(Self {
            inner: Mutex::new(Some(session)),
            backend_id: "obscura-cdp",
        })
    }

    async fn attach(
        playwright: Playwright,
        browser: Browser,
        config: BrowserConfig,
        backend_id: &'static str,
    ) -> Result<Self, CharlotteError> {
        let context = if config.viewport.is_some()
            || config.user_agent.is_some()
            || config.storage_state_path.is_some()
        {
            let mut b = BrowserContextOptions::builder();
            if let Some((w, h)) = config.viewport {
                b = b.viewport(Viewport {
                    width: w,
                    height: h,
                });
            }
            if let Some(ref ua) = config.user_agent {
                b = b.user_agent(ua.clone());
            }
            if let Some(ref p) = config.storage_state_path {
                b = b.storage_state_path(p.display().to_string());
            }
            browser.new_context_with_options(b.build()).await?
        } else {
            browser.new_context().await?
        };
        if matches!(config.automation_profile, AutomationProfile::Stealth) {
            context.add_init_script(STEALTH_INIT_SCRIPT).await?;
        }
        context.add_init_script(CONSOLE_CAPTURE_INIT_SCRIPT).await?;
        install_network_policy(&context, config.block_trackers_and_ads, config.block_fonts).await?;
        let page = context.new_page().await?;
        let network_events = Arc::new(Mutex::new(VecDeque::new()));
        install_network_observers(&page, Arc::clone(&network_events)).await?;
        let session = BrowserSession {
            playwright,
            browser,
            context,
            page,
            cdp_session: None,
            cdp_session_page_url: None,
            network_events,
            config,
            last_web_search_at: None,
        };
        Ok(Self {
            inner: Mutex::new(Some(session)),
            backend_id,
        })
    }
}

#[async_trait]
impl BrowserBackend for PlaywrightBackend {
    fn backend_id(&self) -> &'static str {
        self.backend_id
    }

    async fn close(&self) -> Result<(), CharlotteError> {
        let mut guard = self.inner.lock().await;
        if let Some(s) = guard.take() {
            if self.backend_id == "obscura-cdp" {
                // Disconnect the Playwright driver only. Closing the default page/context
                // on a shared Obscura browser leaves later CDP sessions with a dead tab.
                let _ = s.playwright.shutdown().await;
            } else {
                s.browser.close().await?;
                s.playwright.shutdown().await?;
            }
        }
        Ok(())
    }

    async fn enforce_url_policy(&self, url: &str) -> Result<(), CharlotteError> {
        let guard = self.inner.lock().await;
        let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
        s.config.url_policy.check(url)
    }

    async fn navigate_with_options(
        &self,
        url: &str,
        options: NavigateOptions,
    ) -> Result<Value, CharlotteError> {
        self.enforce_url_policy(url).await?;
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let timeout = options.timeout.unwrap_or(s.config.navigation_timeout);
        let goto_opts = GotoOptions::new()
            .timeout(timeout)
            .wait_until(options.wait_until);
        let resp = s.page.goto(url, Some(goto_opts)).await?;
        if let Some(sel) = optional_css_selector(options.wait_for_selector.as_deref()) {
            let loc = s.page.locator(sel).await;
            loc.wait_for(Some(
                WaitForOptions::builder()
                    .timeout(options.wait_for_selector_timeout_ms)
                    .build(),
            ))
            .await?;
        }
        let (status, final_url, ok) = match &resp {
            Some(r) => (Some(r.status()), r.url().to_string(), r.ok()),
            None => (None, s.page.url(), true),
        };
        let redirected = redirect_changed(url, &final_url);
        Ok(json!({
            "requested_url": url,
            "url": final_url,
            "status": status,
            "ok": ok,
            "redirected": redirected,
        }))
    }

    async fn web_search(&self, query: &str, max_results: usize) -> Result<Value, CharlotteError> {
        let max_results = max_results.clamp(1, 20);
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let search_url = build_search_url(&s.config.search_url_template, query);
        drop(guard);
        self.enforce_url_policy(&search_url).await?;
        let cooldown_wait = {
            let guard = self.inner.lock().await;
            let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
            web_search_cooldown_remaining(s.last_web_search_at, s.config.web_search_cooldown)
        };
        if !cooldown_wait.is_zero() {
            tracing::debug!(wait_ms = cooldown_wait.as_millis(), "web_search cooldown");
            tokio::time::sleep(cooldown_wait).await;
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        s.last_web_search_at = Some(Instant::now());
        let opts = GotoOptions::new()
            .timeout(s.config.navigation_timeout)
            .wait_until(WaitUntil::Load);
        s.page.goto(&search_url, Some(opts)).await?;
        let outcome =
            parse_ddg_results(&s.page, max_results, s.config.web_search_action_timeout).await?;
        let mut out = json!({
            "search_url": search_url,
            "results": outcome.results,
        });
        if let Some(reason) = outcome.blocked_reason {
            out["blocked"] = json!(true);
            out["blocked_reason"] = json!(reason);
        }
        if let Some(warning) = outcome.warning {
            out["warning"] = json!(warning);
        }
        Ok(out)
    }

    async fn page_content_with_max_chars(
        &self,
        selector: Option<&str>,
        max_chars: usize,
    ) -> Result<Value, CharlotteError> {
        let selector = optional_css_selector(selector);
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let text = if let Some(sel) = selector {
            evaluate_selector_inner_text(&s.page, sel).await?
        } else {
            let body = s.page.locator("body").await;
            body.inner_text().await?
        };
        let page_title: Option<String> = s
            .page
            .evaluate("() => document.title || ''", None::<&()>)
            .await
            .ok();
        let truncated = truncate_chars(&text, max_chars);
        let mut out = json!({
            "text": truncated,
            "truncated": truncated.chars().count() < text.chars().count(),
            "max_chars": max_chars,
        });
        if let Some(reason) = detect_block_page_with_url(page_title.as_deref(), &text, None) {
            out["blocked"] = json!(true);
            out["blocked_reason"] = json!(reason);
        }
        if let Some(sel) = selector {
            out["selector"] = json!(sel);
        }
        Ok(out)
    }

    async fn page_html(&self) -> Result<String, CharlotteError> {
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        Ok(s.page.content().await?)
    }

    async fn page_markdown(
        &self,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError> {
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        markdown_from_page(&s.page, focus, max_html_bytes, max_markdown_chars).await
    }

    async fn list_links(
        &self,
        selector: Option<&str>,
        limit: usize,
        same_origin_only: bool,
    ) -> Result<Value, CharlotteError> {
        let limit = limit.clamp(1, 200);
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let page_url = s.page.url();
        let root = optional_css_selector(selector);
        let base_sel = root
            .map(|sel| format!("{sel} a[href]"))
            .unwrap_or_else(|| "a[href]".to_string());
        let loc = s.page.locator(&base_sel).await;
        let links = loc.all().await?;
        let mut out = Vec::new();
        for link in links.into_iter().take(limit) {
            let href = match link.get_attribute("href").await? {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };
            if same_origin_only {
                let full = resolve_href(&page_url, &href);
                if !same_origin(&page_url, &full) {
                    continue;
                }
            }
            let text = link
                .inner_text()
                .await
                .unwrap_or_default()
                .trim()
                .to_string();
            out.push(json!({ "href": href, "text": text }));
        }
        Ok(json!({ "links": out }))
    }

    async fn scroll_by(&self, delta_x: i32, delta_y: i32) -> Result<Value, CharlotteError> {
        let dx = clamp_scroll_delta(delta_x);
        let dy = clamp_scroll_delta(delta_y);
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let arg = json!({ "dx": dx, "dy": dy });
        let _: bool = s
            .page
            .evaluate(
                "(arg) => { window.scrollBy(arg.dx, arg.dy); return true; }",
                Some(&arg),
            )
            .await?;
        Ok(json!({
            "ok": true,
            "scrolled": { "delta_x": dx, "delta_y": dy }
        }))
    }

    async fn scroll_into_view(&self, selector: &str) -> Result<Value, CharlotteError> {
        let sel = selector.trim();
        if sel.is_empty() {
            return Err(CharlotteError::msg("selector must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let loc = s.page.locator(sel).await;
        loc.scroll_into_view_if_needed().await?;
        Ok(json!({
            "ok": true,
            "selector": sel,
        }))
    }

    async fn click(
        &self,
        selector: &str,
        timeout_ms: Option<f64>,
    ) -> Result<Value, CharlotteError> {
        let sel = selector.trim();
        if sel.is_empty() {
            return Err(CharlotteError::msg("selector must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let loc = s.page.locator(sel).await;
        if let Some(ms) = timeout_ms {
            loc.wait_for(Some(WaitForOptions::builder().timeout(ms).build()))
                .await?;
        }
        loc.click(None).await?;
        Ok(json!({ "ok": true, "selector": sel }))
    }

    async fn type_text(
        &self,
        selector: &str,
        text: &str,
        delay_ms: Option<f64>,
    ) -> Result<Value, CharlotteError> {
        let sel = selector.trim();
        if sel.is_empty() {
            return Err(CharlotteError::msg("selector must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let loc = s.page.locator(sel).await;
        loc.focus().await?;
        if let Some(delay) = delay_ms {
            let opts = playwright_rs::PressSequentiallyOptions::builder()
                .delay(delay.max(0.0))
                .build();
            loc.press_sequentially(text, Some(opts)).await?;
        } else {
            loc.press_sequentially(text, None).await?;
        }
        Ok(json!({
            "ok": true,
            "selector": sel,
            "chars": text.chars().count(),
        }))
    }

    async fn fill(&self, selector: &str, text: &str) -> Result<Value, CharlotteError> {
        let sel = selector.trim();
        if sel.is_empty() {
            return Err(CharlotteError::msg("selector must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let loc = s.page.locator(sel).await;
        loc.fill(text, None).await?;
        Ok(json!({
            "ok": true,
            "selector": sel,
            "chars": text.chars().count(),
        }))
    }

    async fn press_key(&self, key: &str) -> Result<Value, CharlotteError> {
        let k = key.trim();
        if k.is_empty() {
            return Err(CharlotteError::msg("key must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        s.page.keyboard().press(k, None).await?;
        Ok(json!({ "ok": true, "key": k }))
    }

    async fn select_option(
        &self,
        selector: &str,
        value: Option<&str>,
        label: Option<&str>,
        index: Option<usize>,
    ) -> Result<Value, CharlotteError> {
        let sel = selector.trim();
        if sel.is_empty() {
            return Err(CharlotteError::msg("selector must not be empty"));
        }
        let choice = if let Some(v) = value.map(str::trim).filter(|v| !v.is_empty()) {
            SelectOption::Value(v.to_string())
        } else if let Some(l) = label.map(str::trim).filter(|l| !l.is_empty()) {
            SelectOption::Label(l.to_string())
        } else if let Some(i) = index {
            SelectOption::Index(i)
        } else {
            return Err(CharlotteError::msg(
                "select_option requires one of value, label, or index",
            ));
        };
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let loc = s.page.locator(sel).await;
        let selected = loc.select_option(choice, None).await?;
        Ok(json!({
            "ok": true,
            "selector": sel,
            "selected": selected,
        }))
    }

    async fn wait_for_selector(
        &self,
        selector: &str,
        timeout_ms: Option<f64>,
    ) -> Result<Value, CharlotteError> {
        let sel = selector.trim();
        if sel.is_empty() {
            return Err(CharlotteError::msg("selector must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let loc = s.page.locator(sel).await;
        let opts = timeout_ms.map(|ms| WaitForOptions::builder().timeout(ms).build());
        loc.wait_for(opts).await?;
        Ok(json!({
            "ok": true,
            "selector": sel,
            "timeout_ms": timeout_ms,
        }))
    }

    async fn screenshot(&self, full_page: bool) -> Result<Value, CharlotteError> {
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        match s
            .page
            .screenshot(Some(
                ScreenshotOptions::builder().full_page(full_page).build(),
            ))
            .await
        {
            Ok(bytes) => Ok(screenshot_json_from_bytes(&bytes)),
            Err(e) if self.backend_id == "obscura-cdp" && screenshot_unsupported(&e) => {
                let page = s.page.clone();
                drop(guard);
                screenshot_via_dom_render(&page, full_page).await
            }
            Err(e) => Err(CharlotteError::Playwright(e)),
        }
    }

    async fn list_tabs(&self) -> Result<Value, CharlotteError> {
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let pages = s.context.pages();
        let mut tabs = Vec::with_capacity(pages.len());
        for (i, p) in pages.iter().enumerate() {
            let title = p.title().await.unwrap_or_default();
            tabs.push(json!({
                "index": i,
                "url": p.url(),
                "title": title,
                "active": p.url() == s.page.url(),
            }));
        }
        Ok(json!({ "tabs": tabs }))
    }

    async fn switch_tab(
        &self,
        index: Option<usize>,
        url_contains: Option<&str>,
    ) -> Result<Value, CharlotteError> {
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let pages = s.context.pages();
        let target = if let Some(i) = index {
            pages.get(i).cloned()
        } else if let Some(fragment) = url_contains.map(str::trim).filter(|f| !f.is_empty()) {
            pages.into_iter().find(|p| p.url().contains(fragment))
        } else {
            None
        }
        .ok_or_else(|| CharlotteError::msg("no tab matched index/url_contains"))?;
        set_active_page(s, target.clone());
        Ok(json!({
            "ok": true,
            "url": target.url(),
            "title": target.title().await.unwrap_or_default(),
        }))
    }

    async fn evaluate_js(&self, script: &str, arg: Option<Value>) -> Result<Value, CharlotteError> {
        let s = script.trim();
        if s.is_empty() {
            return Err(CharlotteError::msg("script must not be empty"));
        }
        let mut guard = self.inner.lock().await;
        let state = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let value = if let Some(arg) = arg {
            state.page.evaluate::<Value, Value>(s, Some(&arg)).await?
        } else {
            state.page.evaluate::<(), Value>(s, None).await?
        };
        Ok(json!({
            "ok": true,
            "value": value,
        }))
    }

    async fn cdp_call(&self, method: &str, params: Option<Value>) -> Result<Value, CharlotteError> {
        let m = method.trim();
        if m.is_empty() {
            return Err(CharlotteError::msg("method must not be empty"));
        }
        if self.backend_id == "obscura-cdp" {
            return Ok(json!({
                "ok": false,
                "error": "cdp_unsupported_backend",
                "backend": self.backend_id,
                "method": m,
                "reason": "Playwright newCDPSession over connect_over_cdp crashes for this backend (Duplicate target).",
                "hint": "Use eval_js for page-scoped reads/actions, or run cdp only with CHARLOTTE_BROWSER_BACKEND=playwright.",
                "advice": "use_eval_js_or_switch_backend",
            }));
        }
        if is_denied_cdp_method(m) {
            return Ok(json!({
                "ok": false,
                "error": "cdp_method_denied",
                "method": m,
                "reason": "method category denied by default safety policy",
            }));
        }

        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let session = ensure_cdp_session_for_active_page(s).await?;
        let result = session.send(m, params).await?;
        Ok(json!({
            "ok": true,
            "method": m,
            "result": result,
        }))
    }

    async fn network_requests(&self, limit: usize) -> Result<Value, CharlotteError> {
        let lim = limit.clamp(1, 500);
        let guard = self.inner.lock().await;
        let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
        let events = s.network_events.lock().await;
        let rows = events
            .iter()
            .rev()
            .take(lim)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        Ok(json!({
            "ok": true,
            "count": rows.len(),
            "events": rows,
        }))
    }

    async fn console_messages(&self, limit: usize) -> Result<Value, CharlotteError> {
        let lim = limit.clamp(1, 500);
        let mut guard = self.inner.lock().await;
        let s = guard.as_mut().ok_or(CharlotteError::Closed)?;
        let raw: Value = s
            .page
            .evaluate(
                "(arg) => (window.__charlotteConsoleBuffer || []).slice(-arg.limit)",
                Some(&json!({"limit": lim})),
            )
            .await
            .unwrap_or_else(|_| json!([]));
        Ok(json!({
            "ok": true,
            "count": raw.as_array().map_or(0, |a| a.len()),
            "messages": raw,
        }))
    }

    async fn max_text_chars_limit(&self) -> Result<usize, CharlotteError> {
        let guard = self.inner.lock().await;
        let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
        Ok(s.config.max_text_chars)
    }

    async fn max_markdown_chars_limit(&self) -> Result<usize, CharlotteError> {
        let guard = self.inner.lock().await;
        let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
        Ok(s.config.max_markdown_chars)
    }

    async fn max_html_bytes_limit(&self) -> Result<usize, CharlotteError> {
        let guard = self.inner.lock().await;
        let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
        Ok(s.config.max_html_bytes)
    }

    async fn scrape_isolated(
        &self,
        url: &str,
        options: NavigateOptions,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError> {
        self.enforce_url_policy(url).await?;
        let (context, default_timeout) = {
            let guard = self.inner.lock().await;
            let s = guard.as_ref().ok_or(CharlotteError::Closed)?;
            (s.context.clone(), s.config.navigation_timeout)
        };
        let page = context.new_page().await?;
        let timeout = options.timeout.unwrap_or(default_timeout);
        let wall = timeout.saturating_add(Duration::from_secs(8));
        let scraped = tokio::time::timeout(
            wall,
            scrape_page(
                &page,
                url,
                options,
                timeout,
                focus,
                max_html_bytes,
                max_markdown_chars,
            ),
        )
        .await;
        let _ = page.close().await;
        match scraped {
            Ok(result) => result,
            Err(_) => Err(CharlotteError::msg(format!(
                "scrape timed out after {}ms: {url}",
                wall.as_millis()
            ))),
        }
    }
}

pub struct PlaywrightLaunchedBackend;

impl PlaywrightLaunchedBackend {
    pub async fn create(config: BrowserConfig) -> Result<Arc<dyn BrowserBackend>, CharlotteError> {
        let backend = PlaywrightBackend::launch(config).await?;
        Ok(Arc::new(backend))
    }
}

fn screenshot_json_from_bytes(bytes: &[u8]) -> Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    json!({
        "ok": true,
        "format": "png",
        "bytes": bytes.len(),
        "base64": b64,
    })
}

fn screenshot_unsupported(err: &playwright_rs::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("capturescreenshot") || msg.contains("not supported")
}

/// DOM render fallback when Obscura rejects Playwright `Page.captureScreenshot`.
async fn screenshot_via_dom_render(page: &Page, full_page: bool) -> Result<Value, CharlotteError> {
    const SCRIPT: &str = r##"
async (opts) => {
  const fullPage = !!opts?.fullPage;
  const vw = Math.max(1, window.innerWidth || 1280);
  const vh = Math.max(1, window.innerHeight || 720);
  const sw = Math.max(document.documentElement.scrollWidth || vw, vw);
  const sh = Math.max(document.documentElement.scrollHeight || vh, vh);
  const w = Math.min(fullPage ? sw : vw, 8192);
  const h = Math.min(fullPage ? sh : vh, 8192);
  const html = document.documentElement.outerHTML;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
    <foreignObject width="100%" height="100%">
      <div xmlns="http://www.w3.org/1999/xhtml">${html}</div>
    </foreignObject>
  </svg>`;
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("svg render failed"));
      image.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2d unavailable");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    const dataUrl = canvas.toDataURL("image/png");
    const comma = dataUrl.indexOf(",");
    if (comma < 0) throw new Error("invalid data url");
    return { ok: true, base64: dataUrl.slice(comma + 1) };
  } finally {
    URL.revokeObjectURL(url);
  }
}
"##;

    let value: Value = page
        .evaluate(SCRIPT, Some(&json!({ "fullPage": full_page })))
        .await?;
    let b64 = value
        .get("base64")
        .and_then(Value::as_str)
        .ok_or_else(|| CharlotteError::msg("dom screenshot fallback returned no base64"))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| CharlotteError::msg(format!("dom screenshot invalid base64: {e}")))?;
    Ok(screenshot_json_from_bytes(&bytes))
}

/// Limits per-axis scroll distance from tool or API callers.
const SCROLL_DELTA_CLAMP: i32 = 50_000;

fn clamp_scroll_delta(n: i32) -> i32 {
    n.clamp(-SCROLL_DELTA_CLAMP, SCROLL_DELTA_CLAMP)
}

async fn markdown_from_page(
    page: &Page,
    focus: Option<&str>,
    max_html_bytes: usize,
    max_markdown_chars: usize,
) -> Result<Value, CharlotteError> {
    let url = page.url();
    let snapshot: serde_json::Value = page
        .evaluate(
            "() => ({ body: document.body ? document.body.outerHTML : '', title: document.title || null })",
            None::<&()>,
        )
        .await?;
    let body_html = snapshot.get("body").and_then(|v| v.as_str()).unwrap_or("");
    let title = snapshot.get("title").and_then(|v| v.as_str());
    let result =
        body_html_to_markdown(body_html, title, focus, max_html_bytes, max_markdown_chars)?;
    let mut out = json!({
        "url": url,
        "title": result.title,
        "markdown": result.markdown,
        "truncated": result.truncated,
        "max_markdown_chars": max_markdown_chars,
        "max_html_bytes": max_html_bytes,
        "cleanup": result.cleanup.to_json(),
    });
    let out_title = out.get("title").and_then(|v| v.as_str());
    let out_markdown = out.get("markdown").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(reason) = detect_block_page(out_title, out_markdown) {
        out["blocked"] = json!(true);
        out["blocked_reason"] = json!(reason);
    }
    Ok(out)
}

async fn scrape_page(
    page: &Page,
    url: &str,
    options: NavigateOptions,
    timeout: Duration,
    focus: Option<&str>,
    max_html_bytes: usize,
    max_markdown_chars: usize,
) -> Result<Value, CharlotteError> {
    let goto_opts = GotoOptions::new()
        .timeout(timeout)
        .wait_until(options.wait_until);
    let resp = page.goto(url, Some(goto_opts)).await?;
    if let Some(sel) = optional_css_selector(options.wait_for_selector.as_deref()) {
        let loc = page.locator(sel).await;
        loc.wait_for(Some(
            WaitForOptions::builder()
                .timeout(options.wait_for_selector_timeout_ms)
                .build(),
        ))
        .await?;
    }
    let (status, final_url, ok) = match &resp {
        Some(r) => (Some(r.status()), r.url().to_string(), r.ok()),
        None => (None, page.url(), true),
    };
    if !ok {
        return Ok(json!({
            "url": url,
            "ok": false,
            "error": "navigate_not_ok",
            "requested_url": url,
            "final_url": final_url,
            "status": status,
        }));
    }
    let page_md = markdown_from_page(page, focus, max_html_bytes, max_markdown_chars).await?;
    Ok(json!({
        "url": url,
        "ok": true,
        "requested_url": url,
        "final_url": final_url,
        "status": status,
        "title": page_md.get("title").cloned().unwrap_or(Value::Null),
        "markdown": page_md.get("markdown").cloned().unwrap_or_else(|| json!("")),
        "truncated": page_md.get("truncated").cloned().unwrap_or(json!(false)),
        "blocked": page_md.get("blocked").cloned().unwrap_or(json!(false)),
        "blocked_reason": page_md.get("blocked_reason").cloned().unwrap_or(Value::Null),
    }))
}

/// Returns `None` if `sel` is missing, empty, or whitespace-only so Playwright never receives `""`.
fn optional_css_selector(sel: Option<&str>) -> Option<&str> {
    let s = sel?;
    let t = s.trim();
    if t.is_empty() { None } else { Some(t) }
}

fn set_active_page(session: &mut BrowserSession, page: Page) {
    session.page = page;
    session.cdp_session = None;
    session.cdp_session_page_url = None;
}

fn cdp_session_matches_active_page(cached_page_url: Option<&str>, active_page_url: &str) -> bool {
    cached_page_url == Some(active_page_url)
}

async fn ensure_cdp_session_for_active_page(
    session: &mut BrowserSession,
) -> Result<CDPSession, CharlotteError> {
    let active_page_url = session.page.url();
    if cdp_session_matches_active_page(session.cdp_session_page_url.as_deref(), &active_page_url) {
        if let Some(existing) = session.cdp_session.clone() {
            return Ok(existing);
        }
    }
    session.cdp_session = None;
    session.cdp_session_page_url = None;
    let created = session.context.new_cdp_session(&session.page).await?;
    session.cdp_session = Some(created.clone());
    session.cdp_session_page_url = Some(active_page_url);
    Ok(created)
}

fn chromium_performance_args() -> Vec<String> {
    vec![
        "--disable-dev-shm-usage".to_string(),
        "--mute-audio".to_string(),
    ]
}

fn chromium_launch_args(config: &BrowserConfig) -> Vec<String> {
    let mut args = Vec::new();
    if config.performance_launch_args {
        args.extend(chromium_performance_args());
    }
    if matches!(config.automation_profile, AutomationProfile::Stealth) {
        args.push("--disable-blink-features=AutomationControlled".to_string());
    }
    args.extend(config.extra_chromium_args.iter().cloned());
    args
}

async fn install_network_policy(
    context: &BrowserContext,
    block_trackers: bool,
    block_fonts: bool,
) -> Result<(), CharlotteError> {
    if !block_trackers && !block_fonts {
        return Ok(());
    }
    context
        .route("**/*", move |route| {
            let block_trackers = block_trackers;
            let block_fonts = block_fonts;
            async move {
                let req = route.request();
                let url = req.url();
                let rt = req.resource_type();
                if block_trackers && blocklist::url_looks_like_tracker_or_ad(url) {
                    return route.abort(Some("blockedbyclient")).await;
                }
                if block_fonts && rt == "font" {
                    return route.abort(Some("blockedbyclient")).await;
                }
                route.continue_(None).await
            }
        })
        .await?;
    Ok(())
}

async fn install_network_observers(
    page: &Page,
    events: Arc<Mutex<VecDeque<Value>>>,
) -> Result<(), CharlotteError> {
    let req_events = Arc::clone(&events);
    page.on_request(move |req| {
        let req_events = Arc::clone(&req_events);
        async move {
            push_network_event(
                &req_events,
                json!({
                    "kind": "request",
                    "url": req.url(),
                    "method": req.method(),
                    "resource_type": req.resource_type(),
                    "failure": req.failure(),
                }),
            )
            .await;
            Ok(())
        }
    })
    .await?;

    let resp_events = Arc::clone(&events);
    page.on_response(move |resp| {
        let resp_events = Arc::clone(&resp_events);
        async move {
            push_network_event(
                &resp_events,
                json!({
                    "kind": "response",
                    "url": resp.url(),
                    "status": resp.status(),
                    "ok": (200..400).contains(&resp.status()),
                }),
            )
            .await;
            Ok(())
        }
    })
    .await?;

    let fail_events = Arc::clone(&events);
    page.on_request_failed(move |req| {
        let fail_events = Arc::clone(&fail_events);
        async move {
            push_network_event(
                &fail_events,
                json!({
                    "kind": "request_failed",
                    "url": req.url(),
                    "method": req.method(),
                    "failure": req.failure(),
                }),
            )
            .await;
            Ok(())
        }
    })
    .await?;
    Ok(())
}

async fn push_network_event(events: &Arc<Mutex<VecDeque<Value>>>, row: Value) {
    let mut guard = events.lock().await;
    push_bounded(&mut guard, row, 500);
}

fn push_bounded(buf: &mut VecDeque<Value>, row: Value, cap: usize) {
    buf.push_back(row);
    if cap == 0 {
        buf.clear();
        return;
    }
    if buf.len() > cap {
        let drain = buf.len() - cap;
        buf.drain(..drain);
    }
}

fn build_search_url(template: &str, query: &str) -> String {
    let enc = urlencoding::encode(query);
    template.replace("{query}", enc.as_ref())
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>()
}

/// Read visible text for `sel` via in-page `querySelector` (no Playwright 30s locator wait).
async fn evaluate_selector_inner_text(page: &Page, sel: &str) -> Result<String, CharlotteError> {
    let arg = json!({ "sel": sel });
    let value: Option<String> = page
        .evaluate(
            r#"(arg) => {
                const el = document.querySelector(arg.sel);
                if (!el) return null;
                return (el.innerText || "").trim();
            }"#,
            Some(&arg),
        )
        .await?;
    value.ok_or_else(|| CharlotteError::msg(format!("selector not found on page: {sel}")))
}

fn same_origin(a: &str, b: &str) -> bool {
    let Ok(ua) = url::Url::parse(a) else {
        return false;
    };
    let Ok(ub) = url::Url::parse(b) else {
        return false;
    };
    ua.scheme() == ub.scheme()
        && ua.host() == ub.host()
        && ua.port_or_known_default() == ub.port_or_known_default()
}

fn redirect_changed(requested_url: &str, final_url: &str) -> bool {
    let requested = normalize_redirect_compare_url(requested_url);
    let final_u = normalize_redirect_compare_url(final_url);
    match (requested, final_u) {
        (Some(a), Some(b)) => a != b,
        _ => requested_url != final_url,
    }
}

fn normalize_redirect_compare_url(raw: &str) -> Option<String> {
    let mut url = url::Url::parse(raw).ok()?;
    url.set_fragment(None);
    Some(url.to_string())
}

fn is_denied_cdp_method(method: &str) -> bool {
    const DENIED_PREFIXES: &[&str] = &[
        "Browser.close",
        "Browser.crash",
        "Browser.crashGpuProcess",
        "Target.",
        "Storage.",
        "Cookie.",
        "Network.setCookie",
        "Network.setCookies",
        "Network.deleteCookies",
        "Browser.setPermission",
        "Browser.grantPermissions",
        "Browser.resetPermissions",
        "Page.setDownloadBehavior",
        "DOM.setFileInputFiles",
        "SystemInfo.",
    ];
    DENIED_PREFIXES.iter().any(|p| method.starts_with(p))
}

pub(crate) fn resolve_href(base_url: &str, href: &str) -> String {
    url::Url::parse(base_url)
        .and_then(|b| b.join(href))
        .map(|u| u.to_string())
        .unwrap_or_else(|_| href.to_string())
}

/// Outcome of parsing a DuckDuckGo HTML search results page.
struct DdgParseOutcome {
    results: Vec<Value>,
    blocked_reason: Option<&'static str>,
    warning: Option<&'static str>,
}

const DDG_EVAL_TIMEOUT: Duration = Duration::from_secs(15);
const DDG_POLL: Duration = Duration::from_millis(400);
const DDG_NO_RESULT_ROWS_GIVE_UP: Duration = Duration::from_secs(45);
const DDG_STALE_ROWS_GIVE_UP: Duration = Duration::from_secs(45);

fn web_search_cooldown_remaining(last_at: Option<Instant>, cooldown: Duration) -> Duration {
    if cooldown.is_zero() {
        return Duration::ZERO;
    }
    let Some(last_at) = last_at else {
        return Duration::ZERO;
    };
    cooldown.saturating_sub(last_at.elapsed())
}

/// Parse DuckDuckGo HTML result rows (`div.result`, `a.result__a`, `a.result__snippet`).
///
/// Important: `playwright-rs` hardcodes a 30s timeout on locator helpers, so this scrapes in-page.
async fn parse_ddg_results(
    page: &Page,
    max: usize,
    wait_budget: Duration,
) -> Result<DdgParseOutcome, CharlotteError> {
    let deadline = Instant::now() + wait_budget;
    let no_rows_deadline = Instant::now() + DDG_NO_RESULT_ROWS_GIVE_UP.min(wait_budget);
    let mut stale_rows_since: Option<Instant> = None;

    loop {
        if Instant::now() > deadline {
            return Ok(DdgParseOutcome {
                results: Vec::new(),
                blocked_reason: None,
                warning: Some("ddg_parse_timeout"),
            });
        }

        let page_url = page.url();
        if let Some((title, text)) = ddg_page_snapshot(page).await {
            if let Some(reason) =
                detect_block_page_with_url(title.as_deref(), &text, Some(&page_url))
            {
                return Ok(DdgParseOutcome {
                    results: Vec::new(),
                    blocked_reason: Some(reason),
                    warning: None,
                });
            }
        }

        let n = ddg_result_row_count(page).await.unwrap_or(0);

        if n == 0 && Instant::now() > no_rows_deadline {
            return Ok(DdgParseOutcome {
                results: Vec::new(),
                blocked_reason: None,
                warning: Some("ddg_no_result_rows"),
            });
        }

        let rows = ddg_scrape_evaluate(page, max).await?;
        if !rows.is_empty() {
            return Ok(DdgParseOutcome {
                results: rows,
                blocked_reason: None,
                warning: None,
            });
        }

        if n > 0 {
            let started = stale_rows_since.get_or_insert_with(Instant::now);
            if started.elapsed() >= DDG_STALE_ROWS_GIVE_UP.min(wait_budget) {
                return Ok(DdgParseOutcome {
                    results: Vec::new(),
                    blocked_reason: None,
                    warning: Some("ddg_stale_result_rows"),
                });
            }
        } else {
            stale_rows_since = None;
        }

        tokio::time::sleep(DDG_POLL).await;
    }
}

async fn ddg_page_snapshot(page: &Page) -> Option<(Option<String>, String)> {
    let snapshot = tokio::time::timeout(
        DDG_EVAL_TIMEOUT,
        page.evaluate::<(), Value>(
            r#"() => ({
                title: document.title || '',
                text: (document.body && document.body.innerText) ? document.body.innerText : ''
            })"#,
            None,
        ),
    )
    .await
    .ok()?
    .ok()?;
    let title = snapshot
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    let text = snapshot
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    Some((title, text))
}

async fn ddg_result_row_count(page: &Page) -> Option<usize> {
    let value = tokio::time::timeout(
        DDG_EVAL_TIMEOUT,
        page.evaluate::<(), Value>("() => document.querySelectorAll('div.result').length", None),
    )
    .await
    .ok()?
    .ok()?;
    json_value_as_usize(&value)
}

/// Read DDG result rows in the browser context (no per-field Playwright 30s locator cap).
async fn ddg_scrape_evaluate(page: &Page, max: usize) -> Result<Vec<Value>, CharlotteError> {
    const SCRAPER: &str = r#"(max) => {
        const lim = Math.min(Math.max(1, Number(max) || 8), 20);
        const rows = [...document.querySelectorAll('div.result')].slice(0, lim);
        const out = [];
        for (const row of rows) {
            const a = row.querySelector('a.result__a');
            const sn = row.querySelector('a.result__snippet');
            const title = (a && a.innerText) ? a.innerText.trim() : '';
            const url = (a && a.getAttribute('href')) ? a.getAttribute('href') : '';
            const snippet = (sn && sn.textContent) ? sn.textContent.trim() : '';
            if (url || title) {
                out.push({ title, url: url || '', snippet: snippet || '' });
            }
        }
        return out;
    }"#;

    let raw = tokio::time::timeout(
        DDG_EVAL_TIMEOUT,
        page.evaluate::<_, Value>(SCRAPER, Some(&max)),
    )
    .await
    .map_err(|_| CharlotteError::msg("timed out scraping DuckDuckGo result rows"))?
    .unwrap_or(Value::Array(vec![]));
    Ok(raw.as_array().cloned().unwrap_or_default())
}

fn json_value_as_usize(v: &Value) -> Option<usize> {
    v.as_u64()
        .map(|n| n as usize)
        .or_else(|| v.as_i64().and_then(|n| (n >= 0).then_some(n as usize)))
        .or_else(|| v.as_f64().map(|n| n as usize))
        .or_else(|| v.get("n").and_then(|n| n.as_u64()).map(|n| n as usize))
}

#[cfg(test)]
mod optional_css_selector_tests {
    use super::{
        PlaywrightBackend, cdp_session_matches_active_page, is_denied_cdp_method,
        optional_css_selector, push_bounded, web_search_cooldown_remaining,
    };
    use crate::browser::backend::BrowserBackend;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};
    use tokio::sync::Mutex;

    #[test]
    fn none_empty_and_whitespace_are_omitted() {
        assert_eq!(optional_css_selector(None), None);
        assert_eq!(optional_css_selector(Some("")), None);
        assert_eq!(optional_css_selector(Some("   \t  ")), None);
    }

    #[test]
    fn trims_and_preserves_non_empty() {
        assert_eq!(optional_css_selector(Some("  body  ")), Some("body"));
        assert_eq!(optional_css_selector(Some(".item")), Some(".item"));
    }

    #[test]
    fn cdp_guard_denies_sensitive_methods() {
        assert!(is_denied_cdp_method("Target.attachToTarget"));
        assert!(is_denied_cdp_method("Storage.getCookies"));
        assert!(is_denied_cdp_method("Page.setDownloadBehavior"));
    }

    #[test]
    fn cdp_guard_allows_common_read_methods() {
        assert!(!is_denied_cdp_method("Runtime.evaluate"));
        assert!(!is_denied_cdp_method("DOM.getDocument"));
        assert!(!is_denied_cdp_method("Network.getResponseBody"));
    }

    #[test]
    fn cdp_session_reuse_requires_matching_page_marker() {
        assert!(cdp_session_matches_active_page(
            Some("https://example.com/a"),
            "https://example.com/a"
        ));
        assert!(!cdp_session_matches_active_page(
            Some("https://example.com/a"),
            "https://example.com/b"
        ));
        assert!(!cdp_session_matches_active_page(
            None,
            "https://example.com/a"
        ));
    }

    #[test]
    fn screenshot_unsupported_detects_capture_screenshot_errors() {
        let err = playwright_rs::Error::ProtocolError(
            "Protocol error (Page.captureScreenshot): not supported".into(),
        );
        assert!(super::screenshot_unsupported(&err));
    }

    #[tokio::test]
    async fn obscura_backend_cdp_returns_structured_error() {
        let backend = PlaywrightBackend {
            inner: Mutex::new(None),
            backend_id: "obscura-cdp",
        };
        let out = backend
            .cdp_call("Runtime.evaluate", None)
            .await
            .expect("tool should return structured error");
        assert_eq!(out.get("ok").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(
            out.get("error").and_then(|v| v.as_str()),
            Some("cdp_unsupported_backend")
        );
    }

    #[test]
    fn push_bounded_keeps_recent_entries() {
        let mut buf = VecDeque::new();
        for i in 0..5 {
            push_bounded(&mut buf, json!({"i": i}), 3);
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0]["i"], 2);
        assert_eq!(buf[2]["i"], 4);
    }

    #[test]
    fn web_search_cooldown_waits_only_when_needed() {
        assert_eq!(
            web_search_cooldown_remaining(None, Duration::from_secs(8)),
            Duration::ZERO
        );
        let just_now = web_search_cooldown_remaining(Some(Instant::now()), Duration::from_secs(8));
        assert!(just_now >= Duration::from_secs(7) && just_now <= Duration::from_secs(8));
        assert_eq!(
            web_search_cooldown_remaining(Some(Instant::now()), Duration::ZERO),
            Duration::ZERO
        );
        let five_secs_ago = Instant::now() - Duration::from_secs(5);
        let remaining = web_search_cooldown_remaining(Some(five_secs_ago), Duration::from_secs(8));
        assert!(remaining >= Duration::from_secs(2) && remaining <= Duration::from_secs(4));
    }
}
