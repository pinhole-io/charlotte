//! Web search backends for Charlotte (`web_search` tool).

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use superglue::http::HttpClient;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::sleep;

use crate::browser::BrowserHandle;
use crate::error::CharlotteError;

pub const DEFAULT_SEARXNG_URL: &str = "http://127.0.0.1:18088";
const DEFAULT_WEB_SEARCH_COOLDOWN: Duration = Duration::from_secs(20);
const SEARXNG_CIRCUIT_OPEN: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchBackendKind {
    #[default]
    Searxng,
    Browser,
}

impl SearchBackendKind {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "searxng" | "searx" => Some(Self::Searxng),
            "browser" | "playwright" | "html" => Some(Self::Browser),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Searxng => "searxng",
            Self::Browser => "browser",
        }
    }
}

#[derive(Debug, Default)]
struct ThrottleState {
    last_completed: Option<Instant>,
    circuit_open_until: Option<Instant>,
}

#[derive(Debug)]
struct WebSearchThrottle {
    gate: Semaphore,
    state: Mutex<ThrottleState>,
}

impl WebSearchThrottle {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            gate: Semaphore::new(1),
            state: Mutex::new(ThrottleState::default()),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub backend: SearchBackendKind,
    pub searxng_base_url: String,
    /// Minimum gap between completed `web_search` calls on the SearXNG path.
    pub web_search_cooldown: Duration,
    throttle: Arc<WebSearchThrottle>,
}

impl SearchConfig {
    #[must_use]
    pub fn new(
        backend: SearchBackendKind,
        searxng_base_url: impl Into<String>,
        web_search_cooldown: Duration,
    ) -> Self {
        Self {
            backend,
            searxng_base_url: searxng_base_url.into(),
            web_search_cooldown,
            throttle: WebSearchThrottle::new(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self::new(
            SearchBackendKind::Searxng,
            DEFAULT_SEARXNG_URL,
            DEFAULT_WEB_SEARCH_COOLDOWN,
        )
    }
}

/// Dispatch `web_search` to SearXNG JSON API or the browser HTML scraper.
pub async fn web_search(
    http: &HttpClient,
    browser: &BrowserHandle,
    config: &SearchConfig,
    query: &str,
    max_results: usize,
) -> Result<Value, CharlotteError> {
    match config.backend {
        SearchBackendKind::Searxng => searxng_web_search(http, config, query, max_results).await,
        SearchBackendKind::Browser => browser.web_search(query, max_results).await,
    }
}

async fn searxng_web_search(
    http: &HttpClient,
    config: &SearchConfig,
    query: &str,
    max_results: usize,
) -> Result<Value, CharlotteError> {
    let max_results = max_results.clamp(1, 20);
    if query_contains_site_operator(query) {
        return Ok(site_operator_unsupported_response(query));
    }

    let _permit = config
        .throttle
        .gate
        .acquire()
        .await
        .map_err(|_| CharlotteError::msg("web_search throttle closed"))?;

    if let Some(retry_after) = circuit_open_remaining(&config.throttle).await {
        return Ok(json!({
            "ok": false,
            "error": "searxng_circuit_open",
            "backend": "searxng",
            "retry_after_secs": retry_after.as_secs(),
            "message": "SearXNG upstream engines are rate-limited or blocked; wait before retrying web_search.",
            "hint": "Use navigate/page_markdown/list_links on known URLs, or set CHARLOTTE_SEARCH_BACKEND=browser.",
        }));
    }

    await_web_search_cooldown(&config.throttle, config.web_search_cooldown).await;

    let search_url = build_searxng_search_url(&config.searxng_base_url, query);
    let bytes = match http.get(&search_url).await {
        Ok(b) => b,
        Err(e) => {
            mark_web_search_completed(&config.throttle).await;
            return Ok(json!({
                "ok": false,
                "error": "searxng_unreachable",
                "search_url": search_url,
                "message": e.to_string(),
                "hint": "Start SearXNG via `make infra-up` or set CHARLOTTE_SEARCH_BACKEND=browser",
            }));
        }
    };

    let (rows, unresponsive) = parse_searxng_response(&bytes, max_results)?;
    let empty = rows.is_empty();
    let has_unresponsive = !unresponsive.is_empty();
    if should_open_circuit(&rows, &unresponsive) {
        open_circuit(&config.throttle).await;
    }
    mark_web_search_completed(&config.throttle).await;

    let mut out = json!({
        "search_url": search_url,
        "results": rows,
        "backend": "searxng",
    });
    if has_unresponsive {
        out["unresponsive_engines"] = Value::Array(unresponsive);
    }
    if empty && has_unresponsive {
        out["warning"] = json!(
            "No web results; one or more SearXNG upstream engines are suspended or blocked. \
             Prefer browser tools (navigate, page_markdown, list_links) until search recovers."
        );
    }
    Ok(out)
}

fn query_contains_site_operator(query: &str) -> bool {
    query.to_ascii_lowercase().contains("site:")
}

fn site_operator_unsupported_response(query: &str) -> Value {
    json!({
        "ok": false,
        "error": "site_operator_unsupported",
        "backend": "searxng",
        "query": query,
        "results": [],
        "message": "SearXNG uses Bing, which does not support the site: search operator.",
        "hint": "For pages on a specific site, use navigate + page_markdown or list_links. For open-web search, omit site: and use plain keywords (e.g. \"Apple official Twitter\"). Set CHARLOTTE_SEARCH_BACKEND=browser for DuckDuckGo HTML search with site: support.",
    })
}

async fn circuit_open_remaining(throttle: &WebSearchThrottle) -> Option<Duration> {
    let state = throttle.state.lock().await;
    let until = state.circuit_open_until?;
    let now = Instant::now();
    if now < until {
        Some(until.saturating_duration_since(now))
    } else {
        None
    }
}

async fn await_web_search_cooldown(throttle: &WebSearchThrottle, cooldown: Duration) {
    if cooldown.is_zero() {
        return;
    }
    let wait = {
        let state = throttle.state.lock().await;
        state
            .last_completed
            .map(|last| cooldown.saturating_sub(last.elapsed()))
            .unwrap_or(Duration::ZERO)
    };
    if !wait.is_zero() {
        tracing::info!(
            wait_ms = wait.as_millis(),
            backend = "searxng",
            "web_search cooldown"
        );
        sleep(wait).await;
    }
}

async fn mark_web_search_completed(throttle: &WebSearchThrottle) {
    let mut state = throttle.state.lock().await;
    state.last_completed = Some(Instant::now());
}

async fn open_circuit(throttle: &WebSearchThrottle) {
    let mut state = throttle.state.lock().await;
    state.circuit_open_until = Some(Instant::now() + SEARXNG_CIRCUIT_OPEN);
    tracing::warn!(
        open_secs = SEARXNG_CIRCUIT_OPEN.as_secs(),
        "SearXNG circuit opened after upstream engine suspensions"
    );
}

fn should_open_circuit(rows: &[Value], unresponsive: &[Value]) -> bool {
    rows_empty(rows) && unresponsive.len() >= 3
}

fn rows_empty(rows: &[Value]) -> bool {
    rows.is_empty()
}

fn build_searxng_search_url(base: &str, query: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    let encoded = urlencoding::encode(query);
    // Pin DuckDuckGo. The instance default (Bing) treats words like "top"/"best"
    // as dictionary lookups and returns Cambridge/Merriam-Webster junk.
    format!("{base}/search?q={encoded}&format=json&engines=duckduckgo")
}

fn parse_searxng_response(
    body: &[u8],
    max: usize,
) -> Result<(Vec<Value>, Vec<Value>), CharlotteError> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|e| CharlotteError::msg(format!("invalid SearXNG JSON: {e}")))?;
    let unresponsive = payload
        .get("unresponsive_engines")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let results = payload
        .get("results")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let rows = results
        .into_iter()
        .filter_map(map_searxng_row)
        .take(max)
        .collect();
    Ok((rows, unresponsive))
}

fn map_searxng_row(row: Value) -> Option<Value> {
    let title = row
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim();
    let url = row
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim();
    let snippet = row
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim();
    if url.is_empty() && title.is_empty() {
        return None;
    }
    Some(json!({
        "title": title,
        "url": url,
        "snippet": snippet,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_operator_detected_case_insensitive() {
        assert!(query_contains_site_operator("site:apple.com foo"));
        assert!(query_contains_site_operator("SITE:Apple.com"));
        assert!(!query_contains_site_operator("apple official twitter"));
    }

    #[test]
    fn build_searxng_search_url_encodes_query() {
        let url = build_searxng_search_url("http://127.0.0.1:18088", "site:apple.com social");
        assert_eq!(
            url,
            "http://127.0.0.1:18088/search?q=site%3Aapple.com%20social&format=json&engines=duckduckgo"
        );
    }

    #[test]
    fn parse_searxng_response_maps_content_to_snippet() {
        let body = br#"{
            "results": [
                {"title": "Apple", "url": "https://www.apple.com/", "content": "Official site"},
                {"title": "", "url": "", "content": "skip me"}
            ],
            "unresponsive_engines": [["duckduckgo", "CAPTCHA"]]
        }"#;
        let (rows, unresponsive) = parse_searxng_response(body, 8).expect("parse");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["title"], "Apple");
        assert_eq!(rows[0]["url"], "https://www.apple.com/");
        assert_eq!(rows[0]["snippet"], "Official site");
        assert_eq!(unresponsive.len(), 1);
    }

    #[test]
    fn parse_searxng_response_empty_array() {
        let (rows, unresponsive) = parse_searxng_response(b"{\"results\":[]}", 5).expect("parse");
        assert!(rows.is_empty());
        assert!(unresponsive.is_empty());
    }

    #[test]
    fn circuit_opens_when_many_engines_unresponsive() {
        let unresponsive = vec![
            json!(["brave", "Suspended"]),
            json!(["duckduckgo", "CAPTCHA"]),
            json!(["google cse", "Suspended"]),
        ];
        assert!(should_open_circuit(&[], &unresponsive));
        assert!(!should_open_circuit(
            &[json!({"title": "x", "url": "https://example.com", "snippet": ""})],
            &unresponsive,
        ));
    }

    #[tokio::test]
    async fn searxng_cooldown_waits_between_completed_calls() {
        let throttle = WebSearchThrottle::new();
        let cooldown = Duration::from_millis(80);
        let start = Instant::now();
        mark_web_search_completed(&throttle).await;
        await_web_search_cooldown(&throttle, cooldown).await;
        assert!(start.elapsed() >= Duration::from_millis(70));
    }

    #[tokio::test]
    async fn searxng_gate_serializes_overlapping_calls() {
        let config = SearchConfig::new(
            SearchBackendKind::Searxng,
            DEFAULT_SEARXNG_URL,
            Duration::from_millis(50),
        );
        let t1 = Arc::clone(&config.throttle);
        let t2 = Arc::clone(&config.throttle);
        let start = Instant::now();
        let h1 = tokio::spawn(async move {
            let _permit = t1.gate.acquire().await.unwrap();
            sleep(Duration::from_millis(60)).await;
            mark_web_search_completed(&t1).await;
        });
        let h2 = tokio::spawn(async move {
            let _permit = t2.gate.acquire().await.unwrap();
            mark_web_search_completed(&t2).await;
        });
        let (_, _) = tokio::join!(h1, h2);
        assert!(start.elapsed() >= Duration::from_millis(55));
    }
}
