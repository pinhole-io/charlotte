use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

const MAX_DUPLICATE_CALLS: u32 = 2;
const MAX_URL_VISITS: u32 = 2;

const NAV: &str = "navigate";
const SWITCH_TAB: &str = "switch_tab";
const CONTENT: &str = "page_content";
const PAGE_MARKDOWN: &str = "page_markdown";
const SCROLL: &str = "scroll";

#[derive(Default)]
pub struct GuardState {
    recent: HashMap<String, u32>,
    visited_urls: HashMap<String, u32>,
    blocked_urls: HashSet<String>,
    blocked_hosts: HashSet<String>,
    current_url: Option<String>,
}

pub type SharedGuardState = Arc<Mutex<GuardState>>;

pub fn new_guard_state() -> SharedGuardState {
    Arc::new(Mutex::new(GuardState::default()))
}

pub fn wrap_tool_with_guard(tool: Arc<dyn Tool>, state: SharedGuardState) -> Arc<dyn Tool> {
    Arc::new(GuardedTool {
        spec: tool.spec(),
        inner: tool,
        state,
    })
}

pub async fn blocked_target_response(state: &SharedGuardState, raw_url: &str) -> Option<Value> {
    let target_url = normalize_url_key(raw_url)?;
    let target_host = url_host_key(&target_url);
    let state = state.lock().await;
    if state.blocked_urls.contains(&target_url) {
        return Some(json!({
            "ok": false,
            "error": "url_blocked",
            "url": target_url,
            "hint": "This URL was blocked earlier; do not retry it. Use another source or search snippets.",
            "advice": "finalize_or_change_strategy",
        }));
    }
    if let Some(host) = target_host
        && state.blocked_hosts.contains(&host)
    {
        return Some(json!({
            "ok": false,
            "error": "domain_blocked",
            "host": host,
            "hint": "This host was blocked earlier in this session; choose another source.",
            "advice": "finalize_or_change_strategy",
        }));
    }
    None
}

struct GuardedTool {
    spec: ToolSpec,
    inner: Arc<dyn Tool>,
    state: SharedGuardState,
}

#[async_trait]
impl Tool for GuardedTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let tool_name = self.spec.name.as_str();
        let target_url = match tool_name {
            NAV => arguments
                .get("url")
                .and_then(Value::as_str)
                .and_then(normalize_url_key),
            SWITCH_TAB => arguments
                .get("url_contains")
                .and_then(Value::as_str)
                .and_then(normalize_url_key),
            _ => None,
        };
        let target_host = target_url.as_deref().and_then(url_host_key);

        let (signature, is_blocked_page_call) = {
            let state = self.state.lock().await;
            let current_url = state.current_url.clone();
            let blocked_page = is_current_page_tool(tool_name)
                && current_url
                    .as_ref()
                    .is_some_and(|url| is_url_or_host_blocked(&state, url));
            let signature = if blocked_page {
                format!("{tool_name}:blocked:{}", current_url.unwrap_or_default())
            } else {
                make_call_signature(tool_name, &arguments)
            };
            (signature, blocked_page)
        };

        {
            let mut state = self.state.lock().await;
            let call_count = {
                let count = state.recent.entry(signature).or_insert(0);
                *count += 1;
                *count
            };

            // After the first blocked-page read, never retry page tools on that URL
            // (the model often changes `focus` to bypass duplicate_call).
            if is_blocked_page_call && call_count > 1 {
                return Ok(blocked_no_retry_response(state.current_url.as_ref()));
            }

            if let Some(url) = target_url.as_ref() {
                if state.blocked_urls.contains(url) {
                    return Ok(json!({
                        "ok": false,
                        "error": "url_blocked",
                        "url": url,
                        "hint": "This URL was blocked earlier; do not retry it. Use another source or search snippets.",
                        "advice": "finalize_or_change_strategy",
                    }));
                }
                if let Some(host) = target_host.as_ref()
                    && state.blocked_hosts.contains(host)
                {
                    return Ok(json!({
                        "ok": false,
                        "error": "domain_blocked",
                        "host": host,
                        "hint": "This host was blocked earlier in this session; choose another source.",
                        "advice": "finalize_or_change_strategy",
                    }));
                }
                let visits = state.visited_urls.get(url).copied().unwrap_or(0);
                if visits >= MAX_URL_VISITS {
                    return Ok(json!({
                        "ok": false,
                        "error": "url_repeated",
                        "url": url,
                        "hint": "Already visited this exact URL; extract from what you have or pick a different URL.",
                        "advice": "finalize_or_change_strategy",
                    }));
                }
            } else if is_current_page_tool(tool_name)
                && let Some(current_url) = state.current_url.as_ref()
            {
                if is_url_or_host_blocked(&state, current_url) {
                    return Ok(json!({
                        "ok": false,
                        "error": "url_blocked",
                        "url": current_url,
                        "hint": "The current page URL was already flagged blocked; navigate elsewhere or use web_search snippets.",
                        "advice": "finalize_or_change_strategy",
                    }));
                }
            }

            if call_count > MAX_DUPLICATE_CALLS {
                return Ok(json!({
                    "ok": false,
                    "error": "duplicate_call",
                    "hint": "This exact call was already tried and did not help. Change source/query or finalize your answer.",
                    "advice": "finalize_or_change_strategy",
                }));
            }
        }

        let result = self.inner.call(arguments).await?;
        let mut state = self.state.lock().await;

        if matches!(tool_name, NAV | SWITCH_TAB) {
            if is_result_ok(&result) {
                let effective = result
                    .get("url")
                    .and_then(Value::as_str)
                    .and_then(normalize_url_key)
                    .or(target_url.clone());
                if let Some(url) = effective {
                    state.current_url = Some(url.clone());
                    *state.visited_urls.entry(url).or_insert(0) += 1;
                }
            }
        }

        if is_result_blocked(&result) {
            let effective_blocked_url = if tool_name == NAV {
                result
                    .get("url")
                    .and_then(Value::as_str)
                    .and_then(normalize_url_key)
                    .or(target_url.clone())
            } else {
                state.current_url.clone()
            };

            if let Some(url) = effective_blocked_url {
                state.blocked_urls.insert(url.clone());
                if let Some(host) = url_host_key(&url) {
                    state.blocked_hosts.insert(host);
                }
            }
        }

        Ok(result)
    }
}

fn make_call_signature(tool_name: &str, arguments: &Value) -> String {
    format!("{tool_name}:{}", normalize_json_for_signature(arguments))
}

fn normalize_json_for_signature(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs = map.iter().collect::<Vec<_>>();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            let parts = pairs
                .into_iter()
                .map(|(k, v)| format!("{k}:{}", normalize_json_for_signature(v)))
                .collect::<Vec<_>>();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(items) => {
            let inner = items
                .iter()
                .map(normalize_json_for_signature)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        Value::String(s) => normalize_url_key(s).unwrap_or_else(|| s.trim().to_string()),
        _ => v.to_string(),
    }
}

fn normalize_url_key(raw: &str) -> Option<String> {
    let mut url = url::Url::parse(raw).ok()?;
    url.set_fragment(None);

    let scheme = url.scheme().to_ascii_lowercase();
    let host = url.host_str()?.to_ascii_lowercase();
    let port = url
        .port()
        .map(|p| format!(":{p}"))
        .unwrap_or_else(String::new);

    let mut path = url.path().to_string();
    if path.len() > 1 {
        path = path.trim_end_matches('/').to_string();
    }
    let query = url.query().map(|q| format!("?{q}")).unwrap_or_default();
    Some(format!("{scheme}://{host}{port}{path}{query}"))
}

fn url_host_key(raw: &str) -> Option<String> {
    url::Url::parse(raw)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_ascii_lowercase()))
}

fn is_url_or_host_blocked(state: &GuardState, url: &str) -> bool {
    state.blocked_urls.contains(url)
        || url_host_key(url).is_some_and(|host| state.blocked_hosts.contains(&host))
}

fn blocked_no_retry_response(url: Option<&String>) -> Value {
    json!({
        "ok": false,
        "error": "blocked_no_retry",
        "url": url,
        "hint": "This page is blocked. Do not call page tools on it again. Navigate to another source or use web_search snippets.",
        "advice": "finalize_or_change_strategy",
    })
}

fn is_current_page_tool(tool_name: &str) -> bool {
    matches!(tool_name, CONTENT | PAGE_MARKDOWN | SCROLL)
}

fn is_result_ok(result: &Value) -> bool {
    result.get("ok").and_then(Value::as_bool).unwrap_or(true)
}

fn is_result_blocked(result: &Value) -> bool {
    result
        .get("blocked")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeTool {
        spec: ToolSpec,
        value: Value,
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn spec(&self) -> ToolSpec {
            self.spec.clone()
        }

        async fn call(&self, _arguments: Value) -> Result<Value, ToolInvokeError> {
            Ok(self.value.clone())
        }
    }

    fn fake_tool(name: &str, value: Value) -> Arc<dyn Tool> {
        Arc::new(FakeTool {
            spec: ToolSpec {
                name: name.to_string(),
                parameters_schema: json!({ "type": "object" }),
                description: None,
                static_tool: false,
            },
            value,
        })
    }

    #[tokio::test]
    async fn duplicate_call_short_circuits_third_attempt() {
        let state = new_guard_state();
        let guarded = wrap_tool_with_guard(fake_tool("web_search", json!({"results": []})), state);
        let args = json!({"query":"fish recipe","max_results":10});

        assert!(guarded.call(args.clone()).await.unwrap()["error"].is_null());
        assert!(guarded.call(args.clone()).await.unwrap()["error"].is_null());
        let third = guarded.call(args).await.unwrap();
        assert_eq!(third["error"], "duplicate_call");
    }

    #[tokio::test]
    async fn url_repeated_short_circuits_after_two_visits() {
        let state = new_guard_state();
        let guarded = wrap_tool_with_guard(
            fake_tool(
                "navigate",
                json!({"ok": true, "url": "https://example.com/docs"}),
            ),
            state,
        );
        let args = json!({"url":"https://example.com/docs"});
        assert_eq!(guarded.call(args.clone()).await.unwrap()["ok"], true);
        assert_eq!(guarded.call(args.clone()).await.unwrap()["ok"], true);
        let third = guarded.call(args).await.unwrap();
        assert_eq!(third["error"], "url_repeated");
    }

    #[tokio::test]
    async fn blocked_url_short_circuits_follow_up_navigate() {
        let state = new_guard_state();
        let nav = wrap_tool_with_guard(
            fake_tool(
                "navigate",
                json!({"ok": true, "url": "https://blocked.example/page"}),
            ),
            state.clone(),
        );
        let page = wrap_tool_with_guard(
            fake_tool(
                "page_markdown",
                json!({"blocked": true, "blocked_reason": "cloudflare_challenge"}),
            ),
            state,
        );

        let nav_args = json!({"url":"https://blocked.example/page"});
        assert_eq!(nav.call(nav_args.clone()).await.unwrap()["ok"], true);
        assert_eq!(page.call(json!({})).await.unwrap()["blocked"], true);

        let retry = nav.call(nav_args).await.unwrap();
        assert_eq!(retry["error"], "url_blocked");
    }

    #[tokio::test]
    async fn switch_tab_url_guard_repeats_full_url_contains() {
        let state = new_guard_state();
        let switcher = wrap_tool_with_guard(
            fake_tool(
                "switch_tab",
                json!({"ok": true, "url": "https://example.com/docs"}),
            ),
            state,
        );
        let args = json!({"url_contains":"https://example.com/docs"});
        assert_eq!(switcher.call(args.clone()).await.unwrap()["ok"], true);
        assert_eq!(switcher.call(args.clone()).await.unwrap()["ok"], true);
        let third = switcher.call(args).await.unwrap();
        assert_eq!(third["error"], "url_repeated");
    }

    #[tokio::test]
    async fn blocked_page_markdown_stops_after_one_retry_even_with_different_focus() {
        let state = new_guard_state();
        let nav = wrap_tool_with_guard(
            fake_tool(
                "navigate",
                json!({"ok": true, "url": "https://www.angi.com/companylist/us/ca/fresno/plumbing.htm"}),
            ),
            state.clone(),
        );
        let page = wrap_tool_with_guard(
            fake_tool(
                "page_markdown",
                json!({
                    "blocked": true,
                    "blocked_reason": "empty_or_js_shell",
                    "url": "https://www.angi.com/companylist/us/ca/fresno/plumbing.htm"
                }),
            ),
            state,
        );

        let nav_args = json!({"url":"https://www.angi.com/companylist/us/ca/fresno/plumbing.htm"});
        assert_eq!(nav.call(nav_args).await.unwrap()["ok"], true);

        let first = page.call(json!({"focus":"phone numbers"})).await.unwrap();
        assert_eq!(first["blocked"], true);

        let second = page.call(json!({"focus":"reviews"})).await.unwrap();
        assert_eq!(second["error"], "url_blocked");

        let third = page.call(json!({"focus":"ratings"})).await.unwrap();
        assert_eq!(third["error"], "blocked_no_retry");
    }

    #[test]
    fn normalize_url_preserves_path_and_query() {
        let key = normalize_url_key("https://Example.com/path/?q=1#frag").unwrap();
        assert_eq!(key, "https://example.com/path?q=1");
    }
}
