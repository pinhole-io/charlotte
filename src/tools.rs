//! [`superglue::tools::Tool`] implementations backed by [`crate::BrowserHandle`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use playwright_rs::WaitUntil;
use serde_json::{Value, json};
use tokio::time::sleep;

use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::tools::{
    OnToolError, Tool, ToolInvokeError, ToolRegistry, ToolRetryPolicy, ToolSpec,
};

use crate::agentic_crawl::{
    AGENTIC_CRAWL_NO_LIMIT, AgenticCrawlConfig, AgenticLlm, CrawlInput,
    agentic_crawl_sidecar_enabled, agentic_crawl_with_handle, agentic_crawl_with_sidecar,
};
use crate::browser::{BrowserConfig, BrowserHandle, tool_error};
use crate::chunk::ChunkerConfig;
use crate::crawl::site_map;
use crate::crawl::{CrawlHooks, CrawlRunConfig, SiteCrawlOutputLimits, site_crawl_with_chunks};
use crate::extract::{extract_structured_json, summarize_markdown};
use crate::loopguard::{
    SharedGuardState, blocked_target_response, new_guard_state, wrap_tool_with_guard,
};
use crate::markdown::{
    AGENT_DEFAULT_MAX_HTML_BYTES, AGENT_DEFAULT_MAX_MARKDOWN_CHARS, clamp_agent_html_bytes,
    clamp_agent_markdown_chars,
};
use crate::parse_document::{DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT, parse_document_url};
use crate::scrape_many::{
    LazyBrowserPool, SCRAPE_MANY_DEFAULT_MAX_TOTAL_TEXT_CHARS, ScrapeManyOptions, ScrapeRequest,
    clamp_concurrency, scrape_many_on, validate_urls,
};
use crate::search::{SearchConfig, web_search as dispatch_web_search};

// Only read tools and `agentic_crawl` are registered. Jev owns every page
// action, so no click, type, scroll, navigate, or script tool is exposed.
const SEARCH: &str = "web_search";
const CONTENT: &str = "page_content";
const PAGE_MARKDOWN: &str = "page_markdown";
const LINKS: &str = "list_links";
const SITE_CRAWL_CHUNKS: &str = "site_crawl_chunks";
const AGENTIC_CRAWL: &str = "agentic_crawl";
const WAIT_FOR: &str = "wait_for";
const SCREENSHOT: &str = "screenshot";
const LIST_TABS: &str = "list_tabs";
const SCRAPE_MANY: &str = "scrape_many";
const MAP: &str = "map";
const NETWORK_REQUESTS: &str = "network_requests";
const CONSOLE_MESSAGES: &str = "console_messages";
const EXTRACT: &str = "extract";
const SUMMARIZE: &str = "summarize";
const PARSE_DOCUMENT: &str = "parse_document";

/// Which Charlotte browser tools to register on a registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CharlotteToolProfile {
    /// Snippet-first quick lookup: `web_search` + `scrape_many` only.
    Quick,
    /// Full browser research toolkit.
    #[default]
    Full,
}

/// Tool names registered for each profile (stable for tests and docs).
#[must_use]
pub fn charlotte_tool_names(profile: CharlotteToolProfile) -> &'static [&'static str] {
    match profile {
        CharlotteToolProfile::Quick => &[SEARCH, SCRAPE_MANY],
        CharlotteToolProfile::Full => &[
            SEARCH,
            CONTENT,
            PAGE_MARKDOWN,
            LINKS,
            WAIT_FOR,
            SCREENSHOT,
            LIST_TABS,
            MAP,
            NETWORK_REQUESTS,
            CONSOLE_MESSAGES,
            SCRAPE_MANY,
            EXTRACT,
            SUMMARIZE,
            PARSE_DOCUMENT,
            SITE_CRAWL_CHUNKS,
            AGENTIC_CRAWL,
        ],
    }
}

pub struct ToolLlmContext {
    pub http: HttpClient,
    pub hooks: Arc<HookRegistry>,
    pub guardrails: Arc<GuardrailRegistry>,
    pub options: ChatOptions,
}

/// Browser tools return errors to the model instead of aborting the chat loop.
fn charlotte_browser_tool_policy() -> ToolRetryPolicy {
    ToolRetryPolicy {
        on_error: OnToolError::Skip,
    }
}

async fn register_browser_tool(
    registry: &ToolRegistry,
    tool: Arc<dyn Tool>,
    guard_state: SharedGuardState,
) -> Result<(), ToolInvokeError> {
    let guarded = wrap_tool_with_guard(tool, guard_state);
    let logged = wrap_tool_with_logging(guarded);
    registry
        .register_with_policy(logged, charlotte_browser_tool_policy())
        .await
}

/// Wraps `tool` so every call (and its outcome) is written to the `tracing` log — useful for
/// diagnosing a stalled or slow research run when the interactive progress panel gives no
/// indication of what is actually happening or how long it has been running.
///
/// Emitted at `info` level under the `charlotte::tools` target: set `RUST_LOG=info` (or
/// `HARN_DEBUG=1`, which raises harn's default log level to `info`) to see these in the log file.
fn wrap_tool_with_logging(tool: Arc<dyn Tool>) -> Arc<dyn Tool> {
    Arc::new(LoggingTool { inner: tool })
}

struct LoggingTool {
    inner: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for LoggingTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let name = self.inner.spec().name;
        let args_preview = truncate_log_str(&arguments.to_string(), 300);
        tracing::info!(
            target: "charlotte::tools",
            tool = %name,
            args = %args_preview,
            "tool call started"
        );
        let start = Instant::now();
        let result = self.inner.call(arguments).await;
        let elapsed_ms = start.elapsed().as_millis();
        match &result {
            Ok(value) => {
                let outcome = classify_tool_outcome(value);
                let preview = truncate_log_str(&value.to_string(), 300);
                tracing::info!(
                    target: "charlotte::tools",
                    tool = %name,
                    elapsed_ms,
                    outcome = %outcome,
                    result = %preview,
                    "tool call finished"
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "charlotte::tools",
                    tool = %name,
                    elapsed_ms,
                    error = %e,
                    "tool call errored"
                );
            }
        }
        result
    }
}

/// Best-effort classification of a tool's JSON result for log scanning (`ok` / `blocked` / `error`
/// / `guarded` for loop-guard short-circuits like `duplicate_call` / `url_blocked` / `url_repeated`).
fn classify_tool_outcome(value: &Value) -> &'static str {
    if let Some(err) = value.get("error").and_then(Value::as_str) {
        return match err {
            "duplicate_call" | "url_blocked" | "domain_blocked" | "url_repeated"
            | "blocked_no_retry" => "guarded",
            _ => "error",
        };
    }
    if value.get("blocked").and_then(Value::as_bool) == Some(true) {
        return "blocked";
    }
    if value.get("ok").and_then(Value::as_bool) == Some(false) {
        return "error";
    }
    "ok"
}

fn truncate_log_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push('…');
    out
}

/// Register all Charlotte browser tools on `registry`.
///
/// Returns the lazily-initialized `scrape_many` browser pool (unused until something
/// still calls [`LazyBrowserPool::workers`]). Callers must call [`LazyBrowserPool::close`]
/// on it once done with the registry.
pub async fn register_charlotte_tools(
    registry: &ToolRegistry,
    handle: &BrowserHandle,
    config: &BrowserConfig,
    search_config: &SearchConfig,
    http: &HttpClient,
    llm: Option<ToolLlmContext>,
    profile: CharlotteToolProfile,
) -> Result<Arc<LazyBrowserPool>, ToolInvokeError> {
    let h = Arc::new(handle.clone());
    let guard_state = new_guard_state();
    let pool = Arc::new(LazyBrowserPool::new(config.clone(), None));
    let llm = llm.map(Arc::new);
    register_browser_tool(
        registry,
        Arc::new(WebSearchTool {
            browser: Arc::clone(&h),
            http: http.clone(),
            search: search_config.clone(),
        }),
        Arc::clone(&guard_state),
    )
    .await?;
    if profile == CharlotteToolProfile::Full {
        register_full_charlotte_tools(registry, &h, &guard_state, llm.clone(), config, http)
            .await?;
    }
    register_browser_tool(
        registry,
        Arc::new(ScrapeManyTool {
            browser: Arc::clone(&h),
            guard_state: Arc::clone(&guard_state),
        }),
        Arc::clone(&guard_state),
    )
    .await?;
    Ok(pool)
}

async fn register_full_charlotte_tools(
    registry: &ToolRegistry,
    h: &Arc<BrowserHandle>,
    guard_state: &SharedGuardState,
    llm: Option<Arc<ToolLlmContext>>,
    config: &BrowserConfig,
    http: &HttpClient,
) -> Result<(), ToolInvokeError> {
    register_browser_tool(
        registry,
        Arc::new(PageContentTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(PageMarkdownTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(ListLinksTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(WaitForTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(ScreenshotTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(ListTabsTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(MapTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(NetworkRequestsTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(ConsoleMessagesTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(ExtractTool {
            browser: Arc::clone(h),
            llm: llm.clone(),
        }),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(SummarizeTool {
            browser: Arc::clone(h),
            llm: llm.clone(),
        }),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(ParseDocumentTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(SiteCrawlChunksTool(Arc::clone(h))),
        Arc::clone(guard_state),
    )
    .await?;
    register_browser_tool(
        registry,
        Arc::new(AgenticCrawlTool {
            browser: Arc::clone(h),
            policy: config.url_policy.clone(),
            http: http.clone(),
            llm,
        }),
        Arc::clone(guard_state),
    )
    .await?;
    Ok(())
}

struct WebSearchTool {
    browser: Arc<BrowserHandle>,
    http: HttpClient,
    search: SearchConfig,
}
struct PageContentTool(Arc<BrowserHandle>);
struct PageMarkdownTool(Arc<BrowserHandle>);
struct ListLinksTool(Arc<BrowserHandle>);
struct WaitForTool(Arc<BrowserHandle>);
struct ScreenshotTool(Arc<BrowserHandle>);
struct ListTabsTool(Arc<BrowserHandle>);
struct MapTool(Arc<BrowserHandle>);
struct NetworkRequestsTool(Arc<BrowserHandle>);
struct ConsoleMessagesTool(Arc<BrowserHandle>);
struct ScrapeManyTool {
    browser: Arc<BrowserHandle>,
    guard_state: SharedGuardState,
}
struct ExtractTool {
    browser: Arc<BrowserHandle>,
    llm: Option<Arc<ToolLlmContext>>,
}
struct SummarizeTool {
    browser: Arc<BrowserHandle>,
    llm: Option<Arc<ToolLlmContext>>,
}
struct ParseDocumentTool(Arc<BrowserHandle>);
struct SiteCrawlChunksTool(Arc<BrowserHandle>);
struct AgenticCrawlTool {
    browser: Arc<BrowserHandle>,
    policy: crate::url_policy::UrlPolicy,
    http: HttpClient,
    llm: Option<Arc<ToolLlmContext>>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SEARCH.to_string(),
            description: Some(
                "Run a web search via SearXNG JSON API (default, Bing) or DuckDuckGo HTML when CHARLOTTE_SEARCH_BACKEND=browser. Returns { title, url, snippet } rows. \
                 Prefer answering from titles/snippets when they already contain the fact. When page text is needed, use scrape_many on result URLs (not serial navigate). \
                 Default SearXNG/Bing does NOT support `site:` — use plain keywords for open-web search, or navigate + page_markdown/list_links for on-site discovery when those tools are available. \
                 With browser backend, `site:` root domain only (`site:example.com topic`); never `site:domain/path`."
                    .to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. On default SearXNG/Bing, do not use `site:` (returns no results). Use plain keywords, or browser tools for on-site discovery. Browser backend only: `site:example.com topic` with root domain, not paths in the operator."
                    },
                    "max_results": { "type": "integer", "description": "Max results (1–20)", "default": 8 }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: query", None))?;
        let max_results = arguments
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        dispatch_web_search(&self.http, &self.browser, &self.search, query, max_results)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for PageContentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: CONTENT.to_string(),
            description: Some(
                "Read visible text from the current page, optionally limited to a CSS selector. Missing selectors fail immediately (no 30s wait). Long text is truncated to the configured max.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "Optional CSS selector; omit for full page body text" }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let selector = arguments
            .get("selector")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        self.0.page_content(selector).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for PageMarkdownTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: PAGE_MARKDOWN.to_string(),
            description: Some(
                "Read the current page as Markdown: body HTML snapshot converted MarkItDown-style (strip script/style, ATX headings). Optional focus string prioritizes sections matching what you are looking for. Defaults keep responses within model context (~32k chars); check truncated and cleanup metadata.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "focus": { "type": "string", "description": "Optional: what you are trying to learn from this page (e.g. product price, release date); ranks relevant markdown sections" },
                    "max_html_bytes": { "type": "integer", "description": "Truncate HTML before conversion (default from browser config, typically 600000; hard cap 2000000)", "default": AGENT_DEFAULT_MAX_HTML_BYTES },
                    "max_markdown_chars": { "type": "integer", "description": "Truncate Markdown returned to the model (default from browser config, typically 32000; hard cap 100000)", "default": AGENT_DEFAULT_MAX_MARKDOWN_CHARS },
                    "post_wait_ms": { "type": "integer", "description": "Pause before snapshot (helps client-rendered pages); default 0", "default": 0 }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let focus = arguments
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let default_html = self.0.max_html_bytes_limit().await.map_err(tool_error)?;
        let default_md = self
            .0
            .max_markdown_chars_limit()
            .await
            .map_err(tool_error)?;
        let max_html_bytes = clamp_agent_html_bytes(
            arguments
                .get("max_html_bytes")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_html),
        );
        let max_markdown_chars = clamp_agent_markdown_chars(
            arguments
                .get("max_markdown_chars")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_md),
        );
        let post_wait_ms = arguments
            .get("post_wait_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if post_wait_ms > 0 {
            sleep(Duration::from_millis(post_wait_ms)).await;
        }
        self.0
            .page_markdown(focus, max_html_bytes, max_markdown_chars)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ListLinksTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: LINKS.to_string(),
            description: Some(
                "List links (`href` + visible text) on the current page, optionally under a root selector. Can filter to same-origin only.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "Optional CSS selector for a subtree" },
                    "limit": { "type": "integer", "description": "Max links (1–200)", "default": 50 },
                    "same_origin_only": { "type": "boolean", "description": "Only links with same scheme/host/port as the page", "default": false }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let selector = arguments
            .get("selector")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;
        let same_origin_only = arguments
            .get("same_origin_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        self.0
            .list_links(selector, limit, same_origin_only)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for WaitForTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: WAIT_FOR.to_string(),
            description: Some("Wait for a selector to appear/be actionable.".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "Required CSS selector" },
                    "timeout_ms": { "type": "integer", "description": "Optional wait timeout in milliseconds", "default": 30000 }
                },
                "required": ["selector"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let selector = arguments
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: selector", None))?;
        let timeout_ms = arguments
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .map(|v| v as f64);
        self.0
            .wait_for_selector(selector, timeout_ms)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ScreenshotTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SCREENSHOT.to_string(),
            description: Some("Capture a PNG screenshot (base64 payload).".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "full_page": { "type": "boolean", "description": "Capture full scrollable page", "default": false }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let full_page = arguments
            .get("full_page")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        self.0.screenshot(full_page).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ListTabsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: LIST_TABS.to_string(),
            description: Some("List open tabs for the current browser context.".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _arguments: Value) -> Result<Value, ToolInvokeError> {
        self.0.list_tabs().await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for MapTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: MAP.to_string(),
            description: Some(
                "Discover reachable URLs from a seed URL using bounded same-origin BFS without returning full page content."
                    .to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "seed_url": { "type": "string", "description": "HTTP/HTTPS seed URL" },
                    "max_depth": { "type": "integer", "default": 2 },
                    "max_pages": { "type": "integer", "default": 25 },
                    "same_origin_only": { "type": "boolean", "default": true },
                    "link_limit_per_page": { "type": "integer", "default": 80 },
                    "delay_between_pages_ms": { "type": "integer", "default": 0 },
                    "preserve_url_fragments": { "type": "boolean", "default": true },
                    "post_navigation_wait_ms": { "type": "integer", "default": 2000 },
                    "respect_robots": { "type": "boolean", "default": false }
                },
                "required": ["seed_url"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let seed_url = arguments
            .get("seed_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: seed_url", None))?;
        let run = CrawlRunConfig {
            max_depth: arguments
                .get("max_depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            max_pages: arguments
                .get("max_pages")
                .and_then(|v| v.as_u64())
                .unwrap_or(25)
                .max(1) as usize,
            same_origin_only: arguments
                .get("same_origin_only")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            delay_between_pages: Duration::from_millis(
                arguments
                    .get("delay_between_pages_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            ),
            cache_ttl: None,
            page_cache: None,
            navigate: {
                let mut n = crate::browser::NavigateOptions::default();
                n.wait_until = WaitUntil::NetworkIdle;
                n
            },
            link_limit_per_page: arguments
                .get("link_limit_per_page")
                .and_then(|v| v.as_u64())
                .unwrap_or(80)
                .clamp(1, 200) as usize,
            include_page_markdown: false,
            max_html_bytes: 0,
            max_markdown_chars: 0,
            strip_url_fragments: !arguments
                .get("preserve_url_fragments")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            after_navigation_delay: Duration::from_millis(
                arguments
                    .get("post_navigation_wait_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2000),
            ),
            respect_robots: arguments
                .get("respect_robots")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        };
        site_map(&self.0, seed_url, run, &CrawlHooks::default())
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for NetworkRequestsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: NETWORK_REQUESTS.to_string(),
            description: Some(
                "List recent network events from the active page session.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 100, "description": "Max events to return (1-500)" }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        self.0.network_requests(limit).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ConsoleMessagesTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: CONSOLE_MESSAGES.to_string(),
            description: Some("List recent console messages from the active page.".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 100, "description": "Max messages to return (1-500)" }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        self.0.console_messages(limit).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ScrapeManyTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SCRAPE_MANY.to_string(),
            description: Some(
                "Scrape multiple URLs in parallel on isolated tabs of the shared browser. Returns per-URL markdown/title/error rows."
                    .to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 20,
                        "description": "Absolute URLs to scrape in parallel"
                    },
                    "focus": { "type": "string", "description": "Optional extraction focus string passed to page_markdown for each URL" },
                    "max_concurrency": { "type": "integer", "description": "Parallel tabs to run at once (default 3, max 4)" },
                    "max_markdown_chars": { "type": "integer", "description": "Per-URL markdown cap (default from browser config)" },
                    "max_total_text_chars": { "type": "integer", "description": "Combined markdown cap across all results (default 100000; 0 disables cap)" },
                },
                "required": ["urls"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let urls = arguments
            .get("urls")
            .and_then(Value::as_array)
            .ok_or_else(|| ToolInvokeError::handler("missing array field: urls", None))?
            .iter()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let focus = arguments
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string);
        let max_concurrency_arg = arguments
            .get("max_concurrency")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let max_total_text_chars = arguments
            .get("max_total_text_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        let requests = validate_urls(urls).map_err(tool_error)?;
        let mut blocked_rows = Vec::<Value>::new();
        let mut allowed = Vec::<ScrapeRequest>::new();
        for req in requests {
            if let Some(blocked) = blocked_target_response(&self.guard_state, &req.url).await {
                blocked_rows.push(json!({
                    "url": req.url,
                    "ok": false,
                    "error": blocked.get("error").cloned().unwrap_or_else(|| json!("url_blocked")),
                    "message": blocked.get("hint").cloned().unwrap_or(Value::Null),
                }));
                continue;
            }
            allowed.push(req);
        }
        if allowed.is_empty() {
            return Ok(json!({
                "ok": true,
                "max_concurrency": 0,
                "results": blocked_rows,
                "output": {
                    "truncated": false,
                    "max_total_text_chars": max_total_text_chars.unwrap_or(SCRAPE_MANY_DEFAULT_MAX_TOTAL_TEXT_CHARS),
                }
            }));
        }

        let pool_size = 1;
        let max_concurrency = clamp_concurrency(max_concurrency_arg, pool_size);
        let default_html = self
            .browser
            .max_html_bytes_limit()
            .await
            .map_err(tool_error)?;
        let default_markdown = self
            .browser
            .max_markdown_chars_limit()
            .await
            .map_err(tool_error)?;
        let max_markdown_chars = clamp_agent_markdown_chars(
            arguments
                .get("max_markdown_chars")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_markdown),
        );
        let options = ScrapeManyOptions::with_defaults(
            focus,
            Some(max_concurrency),
            default_html,
            max_markdown_chars,
            max_total_text_chars,
            pool_size,
        );
        let mut result = scrape_many_on(&self.browser, allowed, options)
            .await
            .map_err(tool_error)?;
        if !blocked_rows.is_empty()
            && let Some(rows) = result.get_mut("results").and_then(Value::as_array_mut)
        {
            rows.splice(0..0, blocked_rows);
        }
        Ok(result)
    }
}

#[async_trait]
impl Tool for ExtractTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: EXTRACT.to_string(),
            description: Some(
                "Extract structured JSON from the current page markdown using an LLM prompt/schema."
                    .to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "focus": { "type": "string", "description": "Optional extraction focus" },
                    "prompt": { "type": "string", "description": "Extraction instruction" },
                    "schema": { "type": "object", "description": "Optional JSON schema to enforce" },
                    "max_html_bytes": { "type": "integer", "default": AGENT_DEFAULT_MAX_HTML_BYTES },
                    "max_markdown_chars": { "type": "integer", "default": AGENT_DEFAULT_MAX_MARKDOWN_CHARS }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let Some(llm) = self.llm.as_ref() else {
            return Err(ToolInvokeError::handler(
                "extract is unavailable: missing LLM context",
                None,
            ));
        };
        let focus = arguments
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let prompt = arguments
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let schema = arguments.get("schema");
        let default_html = self
            .browser
            .max_html_bytes_limit()
            .await
            .map_err(tool_error)?;
        let default_md = self
            .browser
            .max_markdown_chars_limit()
            .await
            .map_err(tool_error)?;
        let max_html_bytes = clamp_agent_html_bytes(
            arguments
                .get("max_html_bytes")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_html),
        );
        let max_markdown_chars = clamp_agent_markdown_chars(
            arguments
                .get("max_markdown_chars")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_md),
        );
        let page = self
            .browser
            .page_markdown(focus, max_html_bytes, max_markdown_chars)
            .await
            .map_err(tool_error)?;
        let markdown = page.get("markdown").and_then(Value::as_str).unwrap_or("");
        extract_structured_json(
            &llm.http,
            llm.hooks.as_ref(),
            llm.guardrails.as_ref(),
            &llm.options,
            markdown,
            focus,
            prompt,
            schema,
        )
        .await
        .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for SummarizeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SUMMARIZE.to_string(),
            description: Some(
                "Summarize the current page markdown using the configured LLM.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "focus": { "type": "string", "description": "Optional summary focus" },
                    "max_sentences": { "type": "integer", "default": 5 },
                    "max_html_bytes": { "type": "integer", "default": AGENT_DEFAULT_MAX_HTML_BYTES },
                    "max_markdown_chars": { "type": "integer", "default": AGENT_DEFAULT_MAX_MARKDOWN_CHARS }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let Some(llm) = self.llm.as_ref() else {
            return Err(ToolInvokeError::handler(
                "summarize is unavailable: missing LLM context",
                None,
            ));
        };
        let focus = arguments
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let max_sentences = arguments
            .get("max_sentences")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .clamp(1, 12) as usize;
        let default_html = self
            .browser
            .max_html_bytes_limit()
            .await
            .map_err(tool_error)?;
        let default_md = self
            .browser
            .max_markdown_chars_limit()
            .await
            .map_err(tool_error)?;
        let max_html_bytes = clamp_agent_html_bytes(
            arguments
                .get("max_html_bytes")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_html),
        );
        let max_markdown_chars = clamp_agent_markdown_chars(
            arguments
                .get("max_markdown_chars")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(default_md),
        );
        let page = self
            .browser
            .page_markdown(focus, max_html_bytes, max_markdown_chars)
            .await
            .map_err(tool_error)?;
        let markdown = page.get("markdown").and_then(Value::as_str).unwrap_or("");
        summarize_markdown(
            &llm.http,
            llm.hooks.as_ref(),
            llm.guardrails.as_ref(),
            &llm.options,
            markdown,
            focus,
            max_sentences,
        )
        .await
        .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ParseDocumentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: PARSE_DOCUMENT.to_string(),
            description: Some(
                "Fetch and parse a document URL (PDF currently) into markdown-like text."
                    .to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Document URL (PDF)" },
                    "max_markdown_chars": { "type": "integer", "default": DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT }
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let url = arguments
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: url", None))?;
        self.0.enforce_url_policy(url).await.map_err(tool_error)?;
        let max_markdown_chars = arguments
            .get("max_markdown_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT);
        parse_document_url(url, max_markdown_chars)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for SiteCrawlChunksTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SITE_CRAWL_CHUNKS.to_string(),
            description: Some(
                "Bounded same-origin crawl from seed_url: loads pages, converts HTML to Markdown (htmd), then splits each page with hierarchical Markdown headings. Uses networkidle and optional post-load wait for JS docs. Default preserves URL #fragments (SPAs); set preserve_url_fragments false for classic sites. Returns chunks with id, source_url, text plus crawl stats and output (truncation metadata)—responses are capped by max_chunks_returned / max_total_text_chars so tool payloads stay within model limits.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "seed_url": { "type": "string", "description": "HTTP/HTTPS seed URL" },
                    "max_depth": { "type": "integer", "description": "Max link hops from seed (0 = seed only)", "default": 2 },
                    "max_pages": { "type": "integer", "description": "Stop after this many pages (minimum 1)", "default": 25 },
                    "same_origin_only": { "type": "boolean", "description": "Only enqueue links same origin as each page", "default": true },
                    "link_limit_per_page": { "type": "integer", "description": "Max anchor hrefs collected per page (1–200)", "default": 80 },
                    "delay_between_pages_ms": { "type": "integer", "description": "Pause between pages (rate limit)", "default": 0 },
                    "preserve_url_fragments": { "type": "boolean", "description": "If true (default), keep #fragments when deduping URLs—required for many SPAs. If false, strip fragments like classic crawlers.", "default": true },
                    "post_navigation_wait_ms": { "type": "integer", "description": "After each goto, wait before collecting links/HTML (helps client-rendered docs). Default 2000ms; use 0 if pages are static.", "default": 2000 },
                    "max_html_bytes": { "type": "integer", "description": "Truncate HTML before Markdown conversion", "default": 2000000 },
                    "max_markdown_chars": { "type": "integer", "description": "Truncate Markdown per page", "default": 500000 },
                    "max_section_chars": { "type": "integer", "description": "Max chars per hierarchical section before sub-chunking", "default": 8000 },
                    "section_overlap": { "type": "integer", "description": "Overlap when sub-chunking oversized sections", "default": 200 },
                    "max_chunks_returned": { "type": "integer", "description": "Max chunk rows in the tool JSON (limits LLM context). Default 120; use 0 for no limit.", "default": 120 },
                    "max_total_text_chars": { "type": "integer", "description": "Max combined characters of all chunk text fields in the response. Default 100000 (~25k tokens rough); use 0 for no limit.", "default": 100000 },
                    "respect_robots": { "type": "boolean", "description": "If true, fetch robots.txt per origin and skip disallowed paths (default false).", "default": false }
                },
                "required": ["seed_url"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let seed_url = arguments
            .get("seed_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: seed_url", None))?;
        let max_depth = arguments
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let max_pages = arguments
            .get("max_pages")
            .and_then(|v| v.as_u64())
            .unwrap_or(25)
            .max(1) as usize;
        let same_origin_only = arguments
            .get("same_origin_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let link_limit_per_page = arguments
            .get("link_limit_per_page")
            .and_then(|v| v.as_u64())
            .unwrap_or(80)
            .clamp(1, 200) as usize;
        let delay_ms = arguments
            .get("delay_between_pages_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let max_html_bytes = arguments
            .get("max_html_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(2_000_000) as usize;
        let max_markdown_chars = arguments
            .get("max_markdown_chars")
            .and_then(|v| v.as_u64())
            .unwrap_or(500_000) as usize;
        let max_section_chars = arguments
            .get("max_section_chars")
            .and_then(|v| v.as_u64())
            .unwrap_or(8000)
            .max(1) as usize;
        let section_overlap = arguments
            .get("section_overlap")
            .and_then(|v| v.as_u64())
            .unwrap_or(200) as usize;
        let preserve_url_fragments = arguments
            .get("preserve_url_fragments")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let post_navigation_wait_ms = arguments
            .get("post_navigation_wait_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(2000);
        let respect_robots = arguments
            .get("respect_robots")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let tool_default = SiteCrawlOutputLimits::agent_tool_default();
        let max_chunks_returned = arguments
            .get("max_chunks_returned")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(tool_default.max_chunks);
        let max_total_text_chars = arguments
            .get("max_total_text_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(tool_default.max_total_text_chars);
        let output_limits = SiteCrawlOutputLimits {
            max_chunks: max_chunks_returned,
            max_total_text_chars,
        };

        let mut navigate = crate::browser::NavigateOptions::default();
        navigate.wait_until = WaitUntil::NetworkIdle;

        let run = CrawlRunConfig {
            max_depth,
            max_pages,
            same_origin_only,
            delay_between_pages: Duration::from_millis(delay_ms),
            cache_ttl: None,
            page_cache: None,
            navigate,
            link_limit_per_page,
            include_page_markdown: true,
            max_html_bytes,
            max_markdown_chars,
            strip_url_fragments: !preserve_url_fragments,
            after_navigation_delay: Duration::from_millis(post_navigation_wait_ms),
            respect_robots,
        };

        let chunker = ChunkerConfig::Hierarchical {
            max_section_chars,
            subchunk_overlap: section_overlap,
        };

        site_crawl_with_chunks(
            &self.0,
            seed_url,
            run,
            &CrawlHooks::default(),
            &chunker,
            &output_limits,
        )
        .await
        .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for AgenticCrawlTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: AGENTIC_CRAWL.to_string(),
            description: Some(
                "The only page-action tool. Jev picks every click, type, follow, back, scroll, and stop. Use this once to walk, fill a form, or complete a task on the headed page. Pass user-supplied values in `inputs` (named fields). Do not put secrets in `goal`. You have no click, type, fill, scroll, or navigate tool. site_crawl_chunks is a BFS dump only.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "seed_url": { "type": "string", "description": "HTTP/HTTPS seed URL" },
                    "goal": { "type": "string", "description": "What the crawl should find or complete" },
                    "max_pages": { "type": "integer", "description": "Ignored. The crawl uses 9999999 (no practical page cap)." },
                    "max_steps": { "type": "integer", "description": "Ignored. The crawl uses 9999999 (no practical step cap)." },
                    "same_origin_only": { "type": "boolean", "description": "If true, only follow same-origin URLs. Default false: Jev may follow any in-page href that UrlPolicy allows.", "default": false },
                    "input_text": { "type": "string", "description": "A single value to type when Jev picks type (search box). Use `inputs` for a form with more than one field." },
                    "inputs": { "type": "object", "description": "Named values Jev may type, for example {\"email\": \"a@b.com\", \"password\": \"...\"}. Jev picks the field and the name; it never sees the values. Pass credentials here, never in `goal`.", "additionalProperties": { "type": "string" } },
                    "extract_schema": { "type": "object", "description": "Optional JSON schema for chat extract when Jev sets need_extract" },
                    "respect_robots": { "type": "boolean", "description": "Ignored. agentic_crawl never consults robots.txt." }
                },
                "required": ["seed_url", "goal"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let seed_url = arguments
            .get("seed_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: seed_url", None))?;
        let goal = arguments
            .get("goal")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: goal", None))?;
        let mut cfg = AgenticCrawlConfig::new(seed_url, goal);
        cfg.url_policy = self.policy.clone();
        cfg.max_pages = AGENTIC_CRAWL_NO_LIMIT;
        cfg.max_steps = AGENTIC_CRAWL_NO_LIMIT;
        cfg.same_origin_only = arguments
            .get("same_origin_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        cfg.input_text = arguments
            .get("input_text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.trim().is_empty());
        cfg.inputs = arguments
            .get("inputs")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .filter_map(|(name, value)| {
                        let value = value.as_str()?;
                        (!name.trim().is_empty() && !value.is_empty())
                            .then(|| CrawlInput::new(name.trim(), value))
                    })
                    .collect()
            })
            .unwrap_or_default();
        cfg.extract_schema = arguments.get("extract_schema").cloned();
        cfg.respect_robots = false;

        let llm = self.llm.as_ref().map(|ctx| AgenticLlm {
            http: ctx.http.clone(),
            hooks: Arc::clone(&ctx.hooks),
            guardrails: Arc::clone(&ctx.guardrails),
            options: ctx.options.clone(),
        });
        if agentic_crawl_sidecar_enabled() {
            agentic_crawl_with_sidecar(&cfg, &self.http, llm.as_ref())
                .await
                .map_err(tool_error)
        } else {
            agentic_crawl_with_handle(Arc::clone(&self.browser), &cfg, &self.http, llm.as_ref())
                .await
                .map_err(tool_error)
        }
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn quick_profile_exposes_only_search_and_scrape_many() {
        assert_eq!(
            charlotte_tool_names(CharlotteToolProfile::Quick),
            &["web_search", "scrape_many"]
        );
    }

    #[test]
    fn full_profile_exposes_reads_and_agentic_crawl_only() {
        let names = charlotte_tool_names(CharlotteToolProfile::Full);
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"scrape_many"));
        assert!(names.contains(&"site_crawl_chunks"));
        assert!(names.contains(&"agentic_crawl"));
        for banned in [
            "navigate",
            "click",
            "type",
            "fill",
            "press_key",
            "scroll",
            "select_option",
            "eval_js",
            "cdp",
            "switch_tab",
        ] {
            assert!(
                !names.contains(&banned),
                "Full profile must not expose {banned}; Jev owns page actions"
            );
        }
    }
}
