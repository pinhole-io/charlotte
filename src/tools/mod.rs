//! [`superglue::tools::Tool`] implementations backed by [`crate::BrowserHandle`].

mod agentic;
mod llm;
mod logging;
mod map;
mod names;
mod page;
#[cfg(feature = "documents")]
mod parse;
mod scrape;
mod search;
mod site_crawl;

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolRetryPolicy, ToolSpec};

use crate::browser::{BrowserConfig, BrowserHandle};
use crate::crawl::agentic::CrawlInput;
use crate::crawl::loopguard::{SharedGuardState, new_guard_state};
use crate::crawl::scrape_many::LazyBrowserPool;
use crate::search::SearchConfig;

use agentic::AgenticCrawlTool;
use llm::{ExtractTool, SummarizeTool};
use logging::{charlotte_browser_tool_policy, register_browser_tool, wrap_browser_tool};
use map::MapTool;
#[cfg(feature = "documents")]
use names::PARSE_DOCUMENT;
use names::{
    AGENTIC_CRAWL, CONSOLE_MESSAGES, CONTENT, EXTRACT, LINKS, LIST_TABS, MAP, NETWORK_REQUESTS,
    PAGE_MARKDOWN, SCRAPE_MANY, SCREENSHOT, SEARCH, SITE_CRAWL_CHUNKS, SUMMARIZE, WAIT_FOR,
};
use page::{
    ConsoleMessagesTool, ListLinksTool, ListTabsTool, NetworkRequestsTool, PageContentTool,
    PageMarkdownTool, ScreenshotTool, WaitForTool,
};
#[cfg(feature = "documents")]
use parse::ParseDocumentTool;
use scrape::ScrapeManyTool;
use search::WebSearchTool;
use site_crawl::SiteCrawlChunksTool;

// Only read tools and `agentic_crawl` are registered. Jev owns every page
// action, so no click, type, scroll, navigate, or script tool is exposed.

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
        CharlotteToolProfile::Full => {
            #[cfg(feature = "documents")]
            {
                &[
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
                ]
            }
            #[cfg(not(feature = "documents"))]
            {
                &[
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
                    SITE_CRAWL_CHUNKS,
                    AGENTIC_CRAWL,
                ]
            }
        }
    }
}

#[derive(Clone)]
pub struct ToolLlmContext {
    pub http: HttpClient,
    pub hooks: Arc<HookRegistry>,
    pub guardrails: Arc<GuardrailRegistry>,
    pub options: ChatOptions,
}

/// Secret values resolved by a trusted host after tool approval.
#[derive(Debug, Clone)]
pub struct ResolvedCredential {
    /// Exact HTTPS origins where these values may be typed.
    pub origins: Vec<String>,
    /// Named values. The model sees each name, but never its value.
    pub inputs: Vec<CrawlInput>,
}

/// Per-call model and cancellation state supplied by a Charlotte host.
pub trait ToolRuntimeContext: Send + Sync {
    /// Build the current model context. `None` disables optional extraction.
    fn llm_context(&self) -> Result<Option<ToolLlmContext>, ToolInvokeError>;

    /// Resolve when the active host turn is cancelled.
    fn cancelled(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(std::future::pending())
    }

    /// Resolve a host-owned credential alias for the requested seed URL.
    ///
    /// Hosts must enforce the alias-to-origin binding before they read a secret.
    fn resolve_credential<'a>(
        &'a self,
        _alias: &'a str,
        _seed_url: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedCredential, ToolInvokeError>> + Send + 'a>>
    {
        Box::pin(async {
            Err(ToolInvokeError::handler(
                "credentials are unavailable",
                Some("credential_unavailable".to_string()),
            ))
        })
    }
}

struct StaticToolRuntimeContext(ToolLlmContext);

impl ToolRuntimeContext for StaticToolRuntimeContext {
    fn llm_context(&self) -> Result<Option<ToolLlmContext>, ToolInvokeError> {
        Ok(Some(self.0.clone()))
    }
}

/// One Charlotte tool that a host can select explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CharlotteTool {
    WebSearch,
    AgenticCrawl,
}

impl CharlotteTool {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::WebSearch => SEARCH,
            Self::AgenticCrawl => AGENTIC_CRAWL,
        }
    }
}

/// Fixed resource caps for host-owned Charlotte tools.
#[derive(Clone)]
pub struct CharlotteToolLimits {
    pub max_search_results: usize,
    pub max_crawl_pages: usize,
    pub max_crawl_steps: usize,
    pub crawl_timeout: std::time::Duration,
    pub respect_robots: bool,
}

fn validate_selected_tools(
    selected: &[CharlotteTool],
    limits: &CharlotteToolLimits,
) -> Result<(), ToolInvokeError> {
    if limits.max_search_results == 0
        || limits.max_search_results > 20
        || limits.max_crawl_pages == 0
        || limits.max_crawl_pages > 50
        || limits.max_crawl_steps == 0
        || limits.max_crawl_steps > 200
        || limits.crawl_timeout.is_zero()
        || limits.crawl_timeout > std::time::Duration::from_secs(300)
    {
        return Err(ToolInvokeError::handler(
            "Charlotte tool limits are invalid",
            Some("invalid_limits".to_string()),
        ));
    }

    let mut names = HashSet::with_capacity(selected.len());
    for tool in selected {
        if !names.insert(tool.name()) {
            return Err(ToolInvokeError::DuplicateRegistration {
                name: tool.name().to_string(),
            });
        }
    }
    Ok(())
}

/// Build exact schemas for selected Charlotte tools without browser resources.
pub fn charlotte_tool_specs(
    selected: &[CharlotteTool],
    limits: &CharlotteToolLimits,
) -> Result<Vec<ToolSpec>, ToolInvokeError> {
    validate_selected_tools(selected, limits)?;
    Ok(selected
        .iter()
        .map(|tool| match tool {
            CharlotteTool::WebSearch => search::web_search_spec(limits.max_search_results),
            CharlotteTool::AgenticCrawl => agentic::agentic_crawl_spec(
                limits.max_crawl_pages,
                limits.max_crawl_steps,
                limits.respect_robots,
            ),
        })
        .collect())
}

impl Default for CharlotteToolLimits {
    fn default() -> Self {
        Self {
            max_search_results: 20,
            max_crawl_pages: 50,
            max_crawl_steps: 200,
            crawl_timeout: std::time::Duration::from_secs(300),
            respect_robots: true,
        }
    }
}

/// A tool paired with the error policy Charlotte requires.
#[derive(Clone)]
pub struct RegisteredCharlotteTool {
    pub tool: Arc<dyn Tool>,
    pub retry_policy: ToolRetryPolicy,
}

/// Runtime-owned Charlotte tools and their isolated browser context.
pub struct CharlotteToolSet {
    browser: Arc<BrowserHandle>,
    tools: Vec<RegisteredCharlotteTool>,
}

impl CharlotteToolSet {
    /// Build an explicit host-selected tool set.
    pub fn build(
        browser: BrowserHandle,
        config: &BrowserConfig,
        search_config: &SearchConfig,
        http: &HttpClient,
        runtime: Option<Arc<dyn ToolRuntimeContext>>,
        selected: &[CharlotteTool],
    ) -> Result<Self, ToolInvokeError> {
        Self::build_with_limits(
            browser,
            config,
            search_config,
            http,
            runtime,
            selected,
            CharlotteToolLimits::default(),
        )
    }

    /// Build an explicit host-selected tool set with stricter host caps.
    pub fn build_with_limits(
        browser: BrowserHandle,
        config: &BrowserConfig,
        search_config: &SearchConfig,
        http: &HttpClient,
        runtime: Option<Arc<dyn ToolRuntimeContext>>,
        selected: &[CharlotteTool],
        limits: CharlotteToolLimits,
    ) -> Result<Self, ToolInvokeError> {
        validate_selected_tools(selected, &limits)?;

        let browser = Arc::new(browser);
        let guard_state = new_guard_state();
        let mut tools = Vec::with_capacity(selected.len());
        for selected_tool in selected {
            let tool: Arc<dyn Tool> = match selected_tool {
                CharlotteTool::WebSearch => Arc::new(WebSearchTool {
                    browser: Arc::clone(&browser),
                    http: http.clone(),
                    search: search_config.clone(),
                    max_results: limits.max_search_results,
                }),
                CharlotteTool::AgenticCrawl => Arc::new(AgenticCrawlTool {
                    browser: Arc::clone(&browser),
                    policy: config.url_policy.clone(),
                    http: http.clone(),
                    runtime: runtime.clone(),
                    max_pages: limits.max_crawl_pages,
                    max_steps: limits.max_crawl_steps,
                    timeout: limits.crawl_timeout,
                    respect_robots: limits.respect_robots,
                }),
            };
            tools.push(RegisteredCharlotteTool {
                tool: wrap_browser_tool(tool, Arc::clone(&guard_state)),
                retry_policy: charlotte_browser_tool_policy(),
            });
        }
        Ok(Self { browser, tools })
    }

    /// Return registrations in the caller's selected order.
    #[must_use]
    pub fn tools(&self) -> Vec<RegisteredCharlotteTool> {
        self.tools.clone()
    }

    /// Close this runtime's isolated browser context.
    pub async fn close(&self) -> Result<(), crate::CharlotteError> {
        self.browser.close().await
    }
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
            max_results: 20,
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
    #[cfg(feature = "documents")]
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
            runtime: llm.map(|context| {
                Arc::new(StaticToolRuntimeContext(context.as_ref().clone()))
                    as Arc<dyn ToolRuntimeContext>
            }),
            max_pages: 50,
            max_steps: 200,
            timeout: std::time::Duration::from_secs(300),
            respect_robots: true,
        }),
        Arc::clone(guard_state),
    )
    .await?;
    Ok(())
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

    #[test]
    fn selected_tool_specs_apply_host_limits_without_browser_resources() {
        let limits = CharlotteToolLimits {
            max_search_results: 7,
            max_crawl_pages: 9,
            max_crawl_steps: 11,
            crawl_timeout: std::time::Duration::from_secs(30),
            respect_robots: true,
        };
        let specs = charlotte_tool_specs(
            &[CharlotteTool::WebSearch, CharlotteTool::AgenticCrawl],
            &limits,
        )
        .expect("valid selected schemas");

        assert_eq!(specs[0].name, "web_search");
        assert_eq!(
            specs[0].parameters_schema["properties"]["max_results"]["maximum"],
            7
        );
        assert_eq!(
            specs[0].parameters_schema["properties"]["max_results"]["default"],
            7
        );
        assert_eq!(specs[1].name, "agentic_crawl");
        assert_eq!(
            specs[1].parameters_schema["properties"]["max_pages"]["maximum"],
            9
        );
        assert_eq!(
            specs[1].parameters_schema["properties"]["max_pages"]["default"],
            9
        );
        assert_eq!(
            specs[1].parameters_schema["properties"]["max_steps"]["maximum"],
            11
        );
        assert_eq!(
            specs[1].parameters_schema["properties"]["respect_robots"]["default"],
            true
        );

        let default_specs = charlotte_tool_specs(
            &[CharlotteTool::AgenticCrawl],
            &CharlotteToolLimits::default(),
        )
        .expect("valid default schema");
        assert_eq!(
            default_specs[0].parameters_schema["properties"]["max_pages"]["default"],
            50
        );
    }
}
