//! Shared browser handle for Charlotte tools and crawlers.

pub mod backend;
pub(crate) mod blockdetect;
pub(crate) mod blocklist;
pub mod obscura;
pub mod playwright;
pub mod pool;
pub(crate) mod url_policy;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use playwright_rs::WaitUntil;
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::instrument;

use crate::browser::url_policy::UrlPolicy;
use crate::error::CharlotteError;
use crate::ingest::markdown::{AGENT_DEFAULT_MAX_HTML_BYTES, AGENT_DEFAULT_MAX_MARKDOWN_CHARS};

pub use backend::{BrowserBackend, BrowserBackendKind};
pub use obscura::ObscuraCdpBackend;
pub use playwright::PlaywrightLaunchedBackend;
pub(crate) use playwright::resolve_href;

/// Whether the playwright-rs Node driver is locatable on this host.
///
/// Needed for both launched Chromium and Obscura/`connect_over_cdp` — the driver
/// is the CDP client. Deployed harn servers should set `PLAYWRIGHT_DRIVER_PATH`.
#[must_use]
pub fn playwright_driver_available() -> bool {
    playwright_rs::server::driver::get_driver_executable().is_ok()
}

/// Best-effort mitigation of automation signals in Chromium (not a full “undetected” browser patch).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AutomationProfile {
    #[default]
    Default,
    /// Extra launch flags + context init script to reduce `navigator.webdriver` visibility.
    Stealth,
}

/// Per-navigation options (timeouts, load milestones, optional post-load selector wait).
#[derive(Debug, Clone)]
pub struct NavigateOptions {
    /// When Playwright considers `goto` finished (`load`, `domcontentloaded`, `networkidle`, …).
    pub wait_until: WaitUntil,
    /// When `None`, uses [`BrowserConfig::navigation_timeout`].
    pub timeout: Option<std::time::Duration>,
    /// After navigation, wait until this CSS selector matches (visible).
    pub wait_for_selector: Option<String>,
    pub wait_for_selector_timeout_ms: f64,
}

impl Default for NavigateOptions {
    fn default() -> Self {
        Self {
            wait_until: WaitUntil::Load,
            timeout: None,
            wait_for_selector: None,
            wait_for_selector_timeout_ms: 30_000.0,
        }
    }
}

impl NavigateOptions {
    /// Fast scrape: stop at DOM ready instead of full `load`, with a short wall timeout.
    #[must_use]
    pub fn for_scrape() -> Self {
        Self {
            wait_until: WaitUntil::DomContentLoaded,
            timeout: Some(Duration::from_secs(15)),
            wait_for_selector: None,
            wait_for_selector_timeout_ms: 5_000.0,
        }
    }
}

/// Configuration for launching and using the browser.
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// Run Chromium without a window (default true).
    pub headless: bool,
    /// Max time for navigations (`page.goto`).
    pub navigation_timeout: std::time::Duration,
    /// Wall-clock budget for Playwright actions while parsing HTML search results.
    pub web_search_action_timeout: Duration,
    /// Minimum time between consecutive [`BrowserHandle::web_search`] calls (rate-limit DDG).
    pub web_search_cooldown: Duration,
    /// URL template for web search; must contain `{query}` (percent-encoded substitution).
    pub search_url_template: String,
    /// Max characters returned by [`BrowserHandle::page_content`].
    pub max_text_chars: usize,
    /// Max Markdown characters returned by [`BrowserHandle::page_markdown`] and `page_markdown`.
    pub max_markdown_chars: usize,
    /// Max HTML bytes read before Markdown conversion in [`BrowserHandle::page_markdown`].
    pub max_html_bytes: usize,
    /// Abort requests whose URL matches a small built-in tracker/ads substring list.
    pub block_trackers_and_ads: bool,
    /// Abort font requests to reduce bandwidth.
    pub block_fonts: bool,
    /// Extra Chromium flags for typical automation workloads.
    pub performance_launch_args: bool,
    /// Viewport for the browser context (affects layout; `None` uses Playwright default).
    pub viewport: Option<(u32, u32)>,
    /// Overrides the default user agent for the context.
    pub user_agent: Option<String>,
    /// Automation fingerprint tweaks; see [`AutomationProfile`].
    pub automation_profile: AutomationProfile,
    /// Use a specific Chromium/Chrome binary instead of Playwright’s bundled Chromium.
    pub executable_path: Option<String>,
    /// Use an installed channel, e.g. `"chrome"` or `"msedge"`.
    pub channel: Option<String>,
    /// Extra flags appended after performance/stealth args.
    pub extra_chromium_args: Vec<String>,
    /// Load cookies / localStorage from a Playwright-exported storage state JSON.
    pub storage_state_path: Option<PathBuf>,
    /// Optional host allow-list and private-IP blocking for navigations.
    pub url_policy: UrlPolicy,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            headless: true,
            navigation_timeout: std::time::Duration::from_secs(60),
            web_search_action_timeout: Duration::from_secs(120),
            web_search_cooldown: Duration::from_secs(20),
            search_url_template: "https://html.duckduckgo.com/html/?q={query}".to_string(),
            max_text_chars: 48_000,
            max_markdown_chars: AGENT_DEFAULT_MAX_MARKDOWN_CHARS,
            max_html_bytes: AGENT_DEFAULT_MAX_HTML_BYTES,
            block_trackers_and_ads: true,
            block_fonts: true,
            performance_launch_args: true,
            viewport: None,
            user_agent: None,
            automation_profile: AutomationProfile::default(),
            executable_path: None,
            channel: None,
            extra_chromium_args: Vec::new(),
            storage_state_path: None,
            url_policy: UrlPolicy::default(),
        }
    }
}

/// Handle to a single browser page, shared across Charlotte tools (mutex-serialized).
#[derive(Clone)]
pub struct BrowserHandle {
    inner: Arc<Mutex<Option<Arc<dyn BrowserBackend>>>>,
}

impl BrowserHandle {
    /// Build a browser handle from env-selected backend settings.
    #[instrument(skip_all)]
    pub async fn from_config(config: BrowserConfig) -> Result<Self, CharlotteError> {
        let backend_kind = crate::config::browser_backend_kind_from_env()?;
        let endpoint = crate::config::cdp_endpoint_from_env();
        match backend_kind {
            BrowserBackendKind::Playwright => {
                if endpoint.is_some() {
                    tracing::warn!(
                        "CHARLOTTE_CDP_ENDPOINT is set but ignored because CHARLOTTE_BROWSER_BACKEND=playwright"
                    );
                }
                let handle = Self::new(config).await?;
                tracing::info!("Charlotte browser backend: {}", backend_kind.as_str());
                Ok(handle)
            }
            BrowserBackendKind::ObscuraCdp => {
                let endpoint = endpoint.ok_or_else(|| {
                    CharlotteError::msg(
                        "CHARLOTTE_CDP_ENDPOINT must be set when CHARLOTTE_BROWSER_BACKEND=obscura",
                    )
                })?;
                let handle = Self::connect_over_cdp(config, &endpoint).await?;
                tracing::info!(
                    "Charlotte browser backend: {} ({})",
                    backend_kind.as_str(),
                    endpoint
                );
                Ok(handle)
            }
        }
    }

    /// Launch Playwright, Chromium, and a single page.
    #[instrument(skip_all)]
    pub async fn new(config: BrowserConfig) -> Result<Self, CharlotteError> {
        let backend = PlaywrightLaunchedBackend::create(config).await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(backend))),
        })
    }

    /// Attach to an existing Chromium instance via the Chrome DevTools Protocol (CDP).
    #[instrument(skip_all)]
    pub async fn connect_over_cdp(
        config: BrowserConfig,
        endpoint_url: &str,
    ) -> Result<Self, CharlotteError> {
        let backend = ObscuraCdpBackend::create(config, endpoint_url).await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(backend))),
        })
    }

    pub async fn backend_id(&self) -> Result<&'static str, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        Ok(backend.backend_id())
    }

    /// Enforce [`BrowserConfig::url_policy`] before navigation (no-op when policy is permissive).
    pub async fn enforce_url_policy(&self, url: &str) -> Result<(), CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.enforce_url_policy(url).await
    }

    /// Scrape `url` on a new tab so parallel scrapes do not fight over the shared page.
    #[instrument(skip(self, options))]
    pub async fn scrape_isolated(
        &self,
        url: &str,
        options: NavigateOptions,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError> {
        let backend = {
            let guard = self.inner.lock().await;
            guard.as_ref().ok_or(CharlotteError::Closed)?.clone()
        };
        backend
            .scrape_isolated(url, options, focus, max_html_bytes, max_markdown_chars)
            .await
    }

    /// Close the browser and shut down the Playwright driver.
    #[instrument(skip_all)]
    pub async fn close(&self) -> Result<(), CharlotteError> {
        let backend = {
            let mut guard = self.inner.lock().await;
            guard.take()
        };
        if let Some(backend) = backend {
            backend.close().await?;
        }
        Ok(())
    }

    /// Navigate and return URL + HTTP status when available.
    #[instrument(skip(self))]
    pub async fn navigate(&self, url: &str) -> Result<Value, CharlotteError> {
        self.navigate_with_options(url, NavigateOptions::default())
            .await
    }

    /// Navigate with explicit load/wait behavior (used by crawls and advanced callers).
    #[instrument(skip(self, options))]
    pub async fn navigate_with_options(
        &self,
        url: &str,
        options: NavigateOptions,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.navigate_with_options(url, options).await
    }

    /// DuckDuckGo HTML search.
    #[instrument(skip(self))]
    pub async fn web_search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.web_search(query, max_results).await
    }

    /// Visible text for the full page or a CSS selector (truncated to config max).
    #[instrument(skip(self))]
    pub async fn page_content(&self, selector: Option<&str>) -> Result<Value, CharlotteError> {
        let max_chars = self.max_text_chars_limit().await?;
        self.page_content_with_max_chars(selector, max_chars).await
    }

    /// [`Self::page_content`] with an explicit character limit for this call (UTF-8 safe truncation).
    #[instrument(skip(self))]
    pub async fn page_content_with_max_chars(
        &self,
        selector: Option<&str>,
        max_chars: usize,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend
            .page_content_with_max_chars(selector, max_chars)
            .await
    }

    /// Current `max_text_chars` from [`BrowserConfig`] (used by [`Self::page_content`]).
    pub async fn max_text_chars_limit(&self) -> Result<usize, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.max_text_chars_limit().await
    }

    /// Current `max_markdown_chars` from [`BrowserConfig`] (used by `page_markdown`).
    pub async fn max_markdown_chars_limit(&self) -> Result<usize, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.max_markdown_chars_limit().await
    }

    /// Current `max_html_bytes` from [`BrowserConfig`] (used by `page_markdown`).
    pub async fn max_html_bytes_limit(&self) -> Result<usize, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.max_html_bytes_limit().await
    }

    /// Full HTML document (including doctype), as returned by Playwright [`Page::content`].
    #[instrument(skip(self))]
    pub async fn page_html(&self) -> Result<String, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.page_html().await
    }

    /// Snapshot `document.body`, convert to Markdown (MarkItDown-style), optional focus ranking.
    #[instrument(skip(self))]
    pub async fn page_markdown(
        &self,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend
            .page_markdown(focus, max_html_bytes, max_markdown_chars)
            .await
    }

    /// List links on the current page, optionally scoped to a root `selector`.
    #[instrument(skip(self))]
    pub async fn list_links(
        &self,
        selector: Option<&str>,
        limit: usize,
        same_origin_only: bool,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.list_links(selector, limit, same_origin_only).await
    }

    /// Scroll the main window using `window.scrollBy`.
    #[instrument(skip(self))]
    pub async fn scroll_by(&self, delta_x: i32, delta_y: i32) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.scroll_by(delta_x, delta_y).await
    }

    /// Scroll the first element matching `selector` into the viewport.
    #[instrument(skip(self))]
    pub async fn scroll_into_view(&self, selector: &str) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.scroll_into_view(selector).await
    }

    #[instrument(skip(self))]
    pub async fn click(
        &self,
        selector: &str,
        timeout_ms: Option<f64>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.click(selector, timeout_ms).await
    }

    #[instrument(skip(self, text))]
    pub async fn type_text(
        &self,
        selector: &str,
        text: &str,
        delay_ms: Option<f64>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.type_text(selector, text, delay_ms).await
    }

    #[instrument(skip(self, text))]
    pub async fn fill(&self, selector: &str, text: &str) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.fill(selector, text).await
    }

    #[instrument(skip(self))]
    pub async fn press_key(&self, key: &str) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.press_key(key).await
    }

    #[instrument(skip(self))]
    pub async fn select_option(
        &self,
        selector: &str,
        value: Option<&str>,
        label: Option<&str>,
        index: Option<usize>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.select_option(selector, value, label, index).await
    }

    #[instrument(skip(self))]
    pub async fn wait_for_selector(
        &self,
        selector: &str,
        timeout_ms: Option<f64>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.wait_for_selector(selector, timeout_ms).await
    }

    #[instrument(skip(self))]
    pub async fn screenshot(&self, full_page: bool) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.screenshot(full_page).await
    }

    #[instrument(skip(self))]
    pub async fn list_tabs(&self) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.list_tabs().await
    }

    #[instrument(skip(self))]
    pub async fn switch_tab(
        &self,
        index: Option<usize>,
        url_contains: Option<&str>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.switch_tab(index, url_contains).await
    }

    #[instrument(skip(self, arg))]
    pub async fn evaluate_js(
        &self,
        script: &str,
        arg: Option<Value>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.evaluate_js(script, arg).await
    }

    #[instrument(skip(self, params))]
    pub async fn cdp_call(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.cdp_call(method, params).await
    }

    #[instrument(skip(self))]
    pub async fn network_requests(&self, limit: usize) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.network_requests(limit).await
    }

    #[instrument(skip(self))]
    pub async fn console_messages(&self, limit: usize) -> Result<Value, CharlotteError> {
        let guard = self.inner.lock().await;
        let backend = guard.as_ref().ok_or(CharlotteError::Closed)?;
        backend.console_messages(limit).await
    }
}

pub(crate) fn tool_error(e: CharlotteError) -> superglue::tools::ToolInvokeError {
    superglue::tools::ToolInvokeError::handler(e.to_string(), Some("charlotte".to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[tokio::test]
    async fn from_config_uses_default_cdp_endpoint_for_obscura_backend() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: this test serializes env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_BROWSER_BACKEND", "obscura");
            std::env::remove_var("CHARLOTTE_CDP_ENDPOINT");
        }
        // This should no longer fail due to missing endpoint config.
        match BrowserHandle::from_config(BrowserConfig::default()).await {
            Ok(handle) => {
                let backend = handle.backend_id().await.expect("backend id");
                assert_eq!(backend, "obscura-cdp");
                handle.close().await.expect("close browser");
            }
            Err(err) => {
                assert!(
                    !err.to_string()
                        .contains("CHARLOTTE_CDP_ENDPOINT must be set"),
                    "missing-endpoint error should not happen anymore: {err}"
                );
            }
        }
    }
}
