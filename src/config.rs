//! Build [`superglue::http::ClientConfig`] from environment variables (used by the CLI and examples).
//!
//! | Variable | Effect |
//! |----------|--------|
//! | `HTTP_MAX_RETRIES` | Max retries after a failed attempt (default: 3). |
//! | `HTTP_RETRY_INITIAL_MS` | First backoff delay in ms (default: 1000). |
//! | `HTTP_RETRY_MAX_INTERVAL_MS` | Backoff cap in ms (default: 6000). |
//! | `HTTP_RETRY_MULTIPLIER` | Exponential multiplier (default: 3.0). |
//! | `HTTP_QUOTA_PER_SECOND` | Optional global QPS limit across requests (integer ≥ 1). |
//!
//! Browser / policy (optional):
//!
//! | Variable | Effect |
//! |----------|--------|
//! | `CHARLOTTE_STORAGE_STATE_PATH` | Playwright storage state JSON for logged-in sessions. |
//! | `CHARLOTTE_SEARCH_BACKEND` | Search backend: `searxng` (default) or `browser` (Playwright HTML scrape). |
//! | `CHARLOTTE_SEARXNG_URL` | SearXNG base URL when using the `searxng` backend (default: `http://127.0.0.1:18088`). Queries use the instance's configured engines. |
//! | `CHARLOTTE_SEARCH_URL_TEMPLATE` | Browser search URL with `{query}` (only when `CHARLOTTE_SEARCH_BACKEND=browser`). |
//! | `CHARLOTTE_ALLOW_HOST_SUFFIXES` | Comma-separated host suffixes; when non-empty, only those hosts may be loaded. |
//! | `CHARLOTTE_BLOCK_PRIVATE_IPS` | Set to `1` or `true` to block loopback/private IPs in URLs. |
//! | `CHARLOTTE_WEB_SEARCH_ACTION_TIMEOUT_SECS` | Max seconds for Playwright actions while parsing HTML search results (default: 120). |
//! | `CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS` | Minimum seconds between completed `web_search` calls on SearXNG and browser backends (default: 20; set `0` to disable). |
//! | `CHARLOTTE_MAX_TEXT_CHARS` | Max visible text from `page_content` (default: 48000). |
//! | `CHARLOTTE_MAX_MARKDOWN_CHARS` | Default max Markdown from `page_markdown` (default: 32000). |
//! | `CHARLOTTE_MAX_HTML_BYTES` | Default HTML byte budget before Markdown conversion (default: 600000). |
//! | `CHARLOTTE_BROWSER_BACKEND` | Browser backend: `playwright` (default), `obscura`, or `cdp` (alias). |
//! | `CHARLOTTE_CDP_ENDPOINT` | CDP endpoint when using Obscura/CDP backend (default: `ws://127.0.0.1:9222/devtools/browser`). |

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use superglue::http::{ClientConfig, HttpClient};

use crate::browser::url_policy::UrlPolicy;
use crate::browser::{BrowserBackendKind, BrowserConfig};
use crate::error::CharlotteError;
use crate::search::{DEFAULT_SEARXNG_URL, SearchBackendKind, SearchConfig};

pub const DEFAULT_CDP_ENDPOINT: &str = "http://127.0.0.1:9222";

fn env_u32(key: &str) -> Option<u32> {
    std::env::var(key).ok()?.parse().ok()
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse().ok()
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok()?.parse().ok()
}

/// Merge [`ClientConfig::default`] with optional `HTTP_*` overrides.
#[must_use]
pub fn client_config_from_env() -> ClientConfig {
    let mut c = ClientConfig::default();
    if let Some(n) = env_u32("HTTP_MAX_RETRIES") {
        c.retry.max_retries = n;
    }
    if let Some(ms) = env_u64("HTTP_RETRY_INITIAL_MS") {
        c.retry.initial_interval_ms = ms;
    }
    if let Some(ms) = env_u64("HTTP_RETRY_MAX_INTERVAL_MS") {
        c.retry.max_interval_ms = ms;
    }
    if let Some(m) = env_f64("HTTP_RETRY_MULTIPLIER")
        && m > 0.0
    {
        c.retry.multiplier = m;
    }
    if let Some(q) = env_u32("HTTP_QUOTA_PER_SECOND")
        && let Some(nz) = NonZeroU32::new(q)
    {
        c.quota_per_second = Some(nz);
    }
    c
}

/// [`HttpClient::new`](superglue::http::HttpClient::new) using [`client_config_from_env`].
pub fn http_client_from_env() -> Result<HttpClient, superglue::http::Error> {
    HttpClient::new(client_config_from_env())
}

/// [`UrlPolicy`] from `CHARLOTTE_ALLOW_HOST_SUFFIXES` and `CHARLOTTE_BLOCK_PRIVATE_IPS`.
#[must_use]
pub fn url_policy_from_env() -> UrlPolicy {
    let mut p = UrlPolicy::default();
    if let Ok(s) = std::env::var("CHARLOTTE_ALLOW_HOST_SUFFIXES") {
        p.allow_host_suffixes = s
            .split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(String::from)
            .collect();
    }
    if matches!(
        std::env::var("CHARLOTTE_BLOCK_PRIVATE_IPS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true")),
        Ok(true)
    ) {
        p.block_private_ips = true;
    }
    p
}

/// Merge optional browser-related env into an existing [`BrowserConfig`].
#[must_use]
pub fn apply_browser_config_from_env(mut c: BrowserConfig) -> BrowserConfig {
    if let Ok(p) = std::env::var("CHARLOTTE_STORAGE_STATE_PATH") {
        let t = p.trim();
        if !t.is_empty() {
            c.storage_state_path = Some(PathBuf::from(t));
        }
    }
    if let Ok(t) = std::env::var("CHARLOTTE_SEARCH_URL_TEMPLATE") {
        let t = t.trim();
        if t.contains("{query}") {
            c.search_url_template = t.to_string();
        }
    }
    if let Some(secs) = env_u64("CHARLOTTE_WEB_SEARCH_ACTION_TIMEOUT_SECS") {
        c.web_search_action_timeout = Duration::from_secs(secs);
    }
    if let Some(secs) = env_u64("CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS") {
        c.web_search_cooldown = Duration::from_secs(secs);
    }
    if let Some(n) = env_u64("CHARLOTTE_MAX_TEXT_CHARS") {
        c.max_text_chars = n.max(1) as usize;
    }
    if let Some(n) = env_u64("CHARLOTTE_MAX_MARKDOWN_CHARS") {
        c.max_markdown_chars = n.max(1) as usize;
    }
    if let Some(n) = env_u64("CHARLOTTE_MAX_HTML_BYTES") {
        c.max_html_bytes = n.max(1) as usize;
    }
    c.url_policy = url_policy_from_env();
    c
}

/// Search backend config from `CHARLOTTE_SEARCH_BACKEND` and `CHARLOTTE_SEARXNG_URL`.
#[must_use]
pub fn search_config_from_env() -> Result<SearchConfig, CharlotteError> {
    let backend = match std::env::var("CHARLOTTE_SEARCH_BACKEND") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                SearchBackendKind::Searxng
            } else {
                SearchBackendKind::parse(trimmed).ok_or_else(|| {
                    CharlotteError::msg(format!(
                        "unsupported CHARLOTTE_SEARCH_BACKEND value: {trimmed} (expected searxng|browser)"
                    ))
                })?
            }
        }
        Err(_) => SearchBackendKind::Searxng,
    };
    let searxng_base_url = match std::env::var("CHARLOTTE_SEARXNG_URL") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                DEFAULT_SEARXNG_URL.to_string()
            } else {
                trimmed.to_string()
            }
        }
        Err(_) => DEFAULT_SEARXNG_URL.to_string(),
    };
    let mut web_search_cooldown = std::time::Duration::from_secs(20);
    if let Some(secs) = env_u64("CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS") {
        web_search_cooldown = std::time::Duration::from_secs(secs);
    }
    Ok(SearchConfig::new(
        backend,
        searxng_base_url,
        web_search_cooldown,
    ))
}

/// Parse browser backend from `CHARLOTTE_BROWSER_BACKEND` with default `playwright`.
pub fn browser_backend_kind_from_env() -> Result<BrowserBackendKind, CharlotteError> {
    let Ok(raw) = std::env::var("CHARLOTTE_BROWSER_BACKEND") else {
        return Ok(BrowserBackendKind::Playwright);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(BrowserBackendKind::Playwright);
    }
    BrowserBackendKind::parse(trimmed).ok_or_else(|| {
        CharlotteError::msg(format!(
            "unsupported CHARLOTTE_BROWSER_BACKEND value: {trimmed} (expected playwright|obscura|cdp)"
        ))
    })
}

/// CDP endpoint from `CHARLOTTE_CDP_ENDPOINT`, or Charlotte's default local endpoint.
#[must_use]
pub fn cdp_endpoint_from_env() -> Option<String> {
    let Ok(raw) = std::env::var("CHARLOTTE_CDP_ENDPOINT") else {
        return Some(DEFAULT_CDP_ENDPOINT.to_string());
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Some(DEFAULT_CDP_ENDPOINT.to_string())
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;
    use crate::search::{DEFAULT_SEARXNG_URL, SearchBackendKind};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn browser_backend_defaults_to_playwright() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::remove_var("CHARLOTTE_BROWSER_BACKEND");
        }
        let kind = browser_backend_kind_from_env().expect("default backend");
        assert_eq!(kind, BrowserBackendKind::Playwright);
    }

    #[test]
    fn browser_backend_parses_obscura_and_cdp_alias() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_BROWSER_BACKEND", "obscura");
        }
        assert_eq!(
            browser_backend_kind_from_env().expect("obscura backend"),
            BrowserBackendKind::ObscuraCdp
        );
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_BROWSER_BACKEND", "cdp");
        }
        assert_eq!(
            browser_backend_kind_from_env().expect("cdp backend"),
            BrowserBackendKind::ObscuraCdp
        );
    }

    #[test]
    fn browser_backend_rejects_unknown_value() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_BROWSER_BACKEND", "unknown");
        }
        let err = browser_backend_kind_from_env().expect_err("unknown backend should fail");
        assert!(
            err.to_string()
                .contains("unsupported CHARLOTTE_BROWSER_BACKEND"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn cdp_endpoint_trims_and_defaults_empty() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_CDP_ENDPOINT", "  http://127.0.0.1:9222  ");
        }
        assert_eq!(
            cdp_endpoint_from_env(),
            Some("http://127.0.0.1:9222".to_string())
        );
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_CDP_ENDPOINT", "   ");
        }
        assert_eq!(
            cdp_endpoint_from_env(),
            Some(DEFAULT_CDP_ENDPOINT.to_string())
        );
    }

    #[test]
    fn search_config_defaults_to_searxng() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::remove_var("CHARLOTTE_SEARCH_BACKEND");
            std::env::remove_var("CHARLOTTE_SEARXNG_URL");
        }
        let cfg = search_config_from_env().expect("search config");
        assert_eq!(cfg.backend, SearchBackendKind::Searxng);
        assert_eq!(cfg.searxng_base_url, DEFAULT_SEARXNG_URL);
    }

    #[test]
    fn search_config_parses_browser_backend() {
        let _guard = env_lock().lock().expect("env lock");
        // SAFETY: tests in this module guard process env mutation with a global lock.
        unsafe {
            std::env::set_var("CHARLOTTE_SEARCH_BACKEND", "browser");
        }
        let cfg = search_config_from_env().expect("search config");
        assert_eq!(cfg.backend, SearchBackendKind::Browser);
    }
}
