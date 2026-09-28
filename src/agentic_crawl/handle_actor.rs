//! [`PageActor`] on the shared headed [`BrowserHandle`].
//!
//! Actions use the same page as `navigate` / `click`. `stop` does not close the browser.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use super::sidecar::PageActor;
use super::targets::ObservePage;
use crate::CharlotteError;
use crate::browser::BrowserHandle;

const SETTLE: Duration = Duration::from_millis(250);

const CLICK_TID_JS: &str = r#"
(id) => {
  const els = document.querySelectorAll('[data-charlotte-tid="' + id + '"]');
  if (!els.length) return { ok: false, error: "missing" };
  els[0].click();
  return { ok: true, count: els.length };
}
"#;

/// Same interactive-element script as the Browser Use sidecar (`t1`… ids).
const OBSERVE_JS: &str = r#"
() => {
  function charlotteRegion(el) {
    if (el.closest('nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"], .mw-indicators, .mw-indicator, .hatnote, #toc, .toc, .vector-toc, [role="note"], #leftbar, .leftbar, [id="leftbar"]')) {
      return "nav";
    }
    const box = el.getBoundingClientRect();
    const vw = window.innerWidth || 0;
    const vh = window.innerHeight || 0;
    if (vw > 0 && box.right > 0 && box.right < vw * 0.22) return "nav";
    if (vh > 0 && box.top > vh * 0.88) return "nav";
    if (el.closest('main, [role="main"], article, #center, .col, #table')) {
      return "content";
    }
    return "content";
  }
  function charlotteHeadingText(node) {
    const tag = (node.tagName || "").toUpperCase();
    if (/^H[2-6]$/.test(tag)) {
      return (node.innerText || "").replace(/\s+/g, " ").trim().slice(0, 60);
    }
    return "";
  }
  function charlotteSection(el) {
    let node = el;
    while (node) {
      let prev = node.previousElementSibling;
      while (prev) {
        const own = charlotteHeadingText(prev);
        if (own) return own;
        const heading = prev.querySelector ? prev.querySelector("h2,h3,h4,h5,h6") : null;
        if (heading) {
          const text = charlotteHeadingText(heading);
          if (text) return text;
        }
        prev = prev.previousElementSibling;
      }
      node = node.parentElement;
    }
    return "";
  }
  function charlotteInViewport(r) {
    const vh = window.innerHeight || 0;
    const vw = window.innerWidth || 0;
    return r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  }
  document.querySelectorAll("[data-charlotte-tid]").forEach((el) => {
    el.removeAttribute("data-charlotte-tid");
  });
  const out = [];
  const sel =
    'a[href], button, input, textarea, select, [role="button"], [role="link"]';
  const nodes = document.querySelectorAll(sel);
  let n = 0;
  for (const el of nodes) {
    if (n >= 40) break;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const style = window.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none" || style.opacity === "0") {
      continue;
    }
    const tag = (el.tagName || "").toUpperCase();
    const role = (el.getAttribute("role") || "").toLowerCase();
    let kind = "button";
    if (tag === "A" || role === "link") kind = "link";
    else if (tag === "INPUT" || tag === "TEXTAREA") kind = "input";
    else if (tag === "SELECT") kind = "select";
    else if (tag === "BUTTON" || role === "button") kind = "button";
    const label = (
      el.innerText ||
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.getAttribute("value") ||
      el.getAttribute("name") ||
      el.getAttribute("href") ||
      ""
    )
      .replace(/\s+/g, " ")
      .trim();
    if (!label) continue;
    n += 1;
    const id = "t" + n;
    el.setAttribute("data-charlotte-tid", id);
    const href = el.href || el.getAttribute("href") || null;
    out.push({
      id,
      kind,
      label,
      href,
      region: charlotteRegion(el),
      section: charlotteSection(el),
      in_viewport: charlotteInViewport(r),
    });
  }
  const body = document.body ? document.body.innerText || "" : "";
  const y = Math.round(window.scrollY || document.documentElement.scrollTop || 0);
  const vh = Math.round(window.innerHeight || 0);
  const h = Math.round(Math.max(
    document.documentElement.scrollHeight || 0,
    document.body ? document.body.scrollHeight : 0
  ));
  const slack = 8;
  return {
    title: document.title || "",
    url: location.href || "",
    markdown: body.slice(0, 8000),
    targets: out,
    scroll_y: y,
    viewport_h: vh,
    scroll_height: h,
    at_top: y <= slack,
    at_bottom: y + vh >= h - slack,
  };
}
"#;

/// Every distinct `a[href]` on the page, including off-screen links. Cap 1500.
const GET_LINKS_JS: &str = r#"
() => {
  function charlotteRegion(el) {
    if (el.closest('nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"], .mw-indicators, .mw-indicator, .hatnote, #toc, .toc, .vector-toc, [role="note"], #leftbar, .leftbar, [id="leftbar"]')) {
      return "nav";
    }
    const box = el.getBoundingClientRect();
    const vw = window.innerWidth || 0;
    const vh = window.innerHeight || 0;
    if (vw > 0 && box.right > 0 && box.right < vw * 0.22) return "nav";
    if (vh > 0 && box.top > vh * 0.88) return "nav";
    if (el.closest('main, [role="main"], article, #center, .col, #table')) {
      return "content";
    }
    return "content";
  }
  function charlotteHeadingText(node) {
    const tag = (node.tagName || "").toUpperCase();
    if (/^H[2-6]$/.test(tag)) {
      return (node.innerText || "").replace(/\s+/g, " ").trim().slice(0, 60);
    }
    return "";
  }
  function charlotteSection(el) {
    let node = el;
    while (node) {
      let prev = node.previousElementSibling;
      while (prev) {
        const own = charlotteHeadingText(prev);
        if (own) return own;
        const heading = prev.querySelector ? prev.querySelector("h2,h3,h4,h5,h6") : null;
        if (heading) {
          const text = charlotteHeadingText(heading);
          if (text) return text;
        }
        prev = prev.previousElementSibling;
      }
      node = node.parentElement;
    }
    return "";
  }
  function charlotteInViewport(r) {
    const vh = window.innerHeight || 0;
    const vw = window.innerWidth || 0;
    return r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  }
  document.querySelectorAll("[data-charlotte-tid]").forEach((el) => {
    el.removeAttribute("data-charlotte-tid");
  });
  const out = [];
  const seen = new Set();
  const nodes = document.querySelectorAll("a[href]");
  let n = 0;
  for (const el of nodes) {
    if (n >= 1500) break;
    const href = el.href || el.getAttribute("href") || "";
    if (!href || href.toLowerCase().startsWith("javascript:")) continue;
    if (seen.has(href)) continue;
    const label = (
      el.innerText ||
      el.getAttribute("aria-label") ||
      el.getAttribute("title") ||
      href ||
      ""
    )
      .replace(/\s+/g, " ")
      .trim();
    if (!label) continue;
    seen.add(href);
    n += 1;
    const id = "t" + n;
    el.setAttribute("data-charlotte-tid", id);
    const r = el.getBoundingClientRect();
    out.push({
      id,
      kind: "link",
      label,
      href,
      region: charlotteRegion(el),
      section: charlotteSection(el),
      in_viewport: charlotteInViewport(r),
    });
  }
  const body = document.body ? document.body.innerText || "" : "";
  const y = Math.round(window.scrollY || document.documentElement.scrollTop || 0);
  const vh = Math.round(window.innerHeight || 0);
  const h = Math.round(Math.max(
    document.documentElement.scrollHeight || 0,
    document.body ? document.body.scrollHeight : 0
  ));
  const slack = 8;
  return {
    title: document.title || "",
    url: location.href || "",
    markdown: body.slice(0, 8000),
    targets: out,
    scroll_y: y,
    viewport_h: vh,
    scroll_height: h,
    at_top: y <= slack,
    at_bottom: y + vh >= h - slack,
  };
}
"#;

/// Drives Jev actions on the headed Playwright / Obscura page.
pub struct HandlePageActor {
    handle: Arc<BrowserHandle>,
}

impl HandlePageActor {
    /// Wrap an existing handle. Does not launch or close the browser.
    #[must_use]
    pub fn new(handle: Arc<BrowserHandle>) -> Self {
        Self { handle }
    }

    async fn observe_now(&self) -> Result<ObservePage, CharlotteError> {
        let raw = self.handle.evaluate_js(OBSERVE_JS, None).await?;
        observe_page_from_eval(&raw)
    }

    async fn settle_observe(&self) -> Result<ObservePage, CharlotteError> {
        tokio::time::sleep(SETTLE).await;
        match self.observe_now().await {
            Ok(page) => Ok(page),
            Err(err) if is_destroyed_context(&err) => {
                tokio::time::sleep(SETTLE).await;
                self.observe_now().await
            }
            Err(err) => Err(err),
        }
    }

    /// Bring a target into view the way a person would before operating it.
    /// Best effort: a control outside the viewport still clicks without this.
    async fn reveal(&self, target_id: &str) {
        if let Ok(sel) = tid_selector(target_id) {
            let _ = self.handle.scroll_into_view(&sel).await;
        }
    }

    async fn click_tid(&self, target_id: &str) -> Result<ObservePage, CharlotteError> {
        let id = target_id.trim();
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(CharlotteError::msg(format!("invalid target id: {id}")));
        }
        self.reveal(id).await;
        let raw = self
            .handle
            .evaluate_js(CLICK_TID_JS, Some(serde_json::json!(id)))
            .await?;
        let inner = raw.get("value").cloned().unwrap_or(raw);
        let value = match inner {
            Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::Null),
            other => other,
        };
        if value.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            return Err(CharlotteError::msg(format!(
                "no element with data-charlotte-tid={id}"
            )));
        }
        self.settle_observe().await
    }
}

#[async_trait]
impl PageActor for HandlePageActor {
    async fn start(&mut self) -> Result<(), CharlotteError> {
        Ok(())
    }

    async fn navigate(&mut self, url: &str) -> Result<ObservePage, CharlotteError> {
        self.handle.navigate(url).await?;
        self.settle_observe().await
    }

    async fn observe(&mut self) -> Result<ObservePage, CharlotteError> {
        self.observe_now().await
    }

    async fn click(&mut self, target_id: &str) -> Result<ObservePage, CharlotteError> {
        self.click_tid(target_id).await
    }

    async fn type_text(
        &mut self,
        target_id: &str,
        text: &str,
    ) -> Result<ObservePage, CharlotteError> {
        let sel = tid_selector(target_id)?;
        self.reveal(target_id).await;
        self.handle.type_text(&sel, text, None).await?;
        self.settle_observe().await
    }

    async fn press(&mut self, key: &str) -> Result<ObservePage, CharlotteError> {
        self.handle.press_key(key).await?;
        self.settle_observe().await
    }

    async fn wait(&mut self, ms: u64) -> Result<ObservePage, CharlotteError> {
        let capped = ms.min(5_000);
        tokio::time::sleep(Duration::from_millis(capped)).await;
        self.observe_now().await
    }

    async fn scroll(&mut self, delta: i32) -> Result<ObservePage, CharlotteError> {
        self.handle.scroll_by(0, delta).await?;
        self.settle_observe().await
    }

    async fn follow(
        &mut self,
        target_id: &str,
        href: Option<&str>,
    ) -> Result<ObservePage, CharlotteError> {
        match href.map(str::trim).filter(|url| !url.is_empty()) {
            Some(url) => self.navigate(url).await,
            None => self.click_tid(target_id).await,
        }
    }

    async fn get_links(&mut self) -> Result<ObservePage, CharlotteError> {
        let raw = self.handle.evaluate_js(GET_LINKS_JS, None).await?;
        observe_page_from_eval(&raw)
    }

    async fn stop(&mut self) -> Result<(), CharlotteError> {
        Ok(())
    }
}

fn is_destroyed_context(err: &CharlotteError) -> bool {
    let text = err.to_string().to_ascii_lowercase();
    text.contains("execution context was destroyed")
        || text.contains("most likely because of a navigation")
}

/// CSS selector for a `data-charlotte-tid` id (`t1`, `t2`, …).
pub(crate) fn tid_selector(id: &str) -> Result<String, CharlotteError> {
    let id = id.trim();
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(CharlotteError::msg(format!("invalid target id: {id}")));
    }
    Ok(format!(r#"[data-charlotte-tid="{id}"]"#))
}

/// Parse `evaluate_js` output (`{ok, value}`) or a bare observe object.
pub(crate) fn observe_page_from_eval(raw: &Value) -> Result<ObservePage, CharlotteError> {
    let inner = raw.get("value").cloned().unwrap_or_else(|| raw.clone());
    let page_val = match inner {
        Value::String(s) => serde_json::from_str(&s)
            .map_err(|e| CharlotteError::msg(format!("observe JSON string: {e}")))?,
        other => other,
    };
    if page_val.as_object().is_none() {
        return Err(CharlotteError::msg("observe returned a non-object"));
    }
    serde_json::from_value(page_val)
        .map_err(|e| CharlotteError::msg(format!("observe page shape: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tid_selector_accepts_t_ids() {
        assert_eq!(
            tid_selector("t1").expect("sel"),
            r#"[data-charlotte-tid="t1"]"#
        );
        assert!(tid_selector("").is_err());
        assert!(tid_selector("t1]; attack").is_err());
    }

    #[test]
    fn observe_page_from_eval_unwraps_evaluate_js() {
        let raw = json!({
            "ok": true,
            "value": {
                "title": "Dog food",
                "url": "https://en.wikipedia.org/wiki/Dog_food",
                "markdown": "Dog food is food for dogs.",
                "targets": [
                    {"id": "t1", "kind": "link", "label": "Dog", "href": "/wiki/Dog"}
                ],
                "scroll_y": 2400,
                "viewport_h": 800,
                "scroll_height": 3200,
                "at_top": false,
                "at_bottom": true
            }
        });
        let page = observe_page_from_eval(&raw).expect("page");
        assert_eq!(page.title, "Dog food");
        assert_eq!(page.url, "https://en.wikipedia.org/wiki/Dog_food");
        assert!(page.at_page_bottom());
        assert_eq!(page.targets.len(), 1);
        assert_eq!(page.targets[0].id, "t1");
        assert_eq!(page.targets[0].kind, "link");
        assert_eq!(page.targets[0].region, "");
        assert!(!page.targets[0].in_viewport);
    }

    #[test]
    fn observe_page_from_eval_parses_json_string_value() {
        let inner = json!({
            "title": "Sun",
            "url": "https://en.wikipedia.org/wiki/Sun",
            "markdown": "The Sun",
            "targets": []
        });
        let raw = json!({ "ok": true, "value": inner.to_string() });
        let page = observe_page_from_eval(&raw).expect("page");
        assert_eq!(page.title, "Sun");
        assert!(page.targets.is_empty());
    }
}
