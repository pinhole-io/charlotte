//! [`superglue::tools::Tool`] implementations backed by [`crate::BrowserHandle`].

mod agentic;
mod llm;
mod logging;
mod map;
mod names;
mod page;
mod parse;
mod scrape;
mod search;
mod site_crawl;

use std::sync::Arc;

use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::tools::{ToolInvokeError, ToolRegistry};

use crate::browser::{BrowserConfig, BrowserHandle};
use crate::crawl::loopguard::{SharedGuardState, new_guard_state};
use crate::crawl::scrape_many::LazyBrowserPool;
use crate::search::SearchConfig;

use agentic::AgenticCrawlTool;
use llm::{ExtractTool, SummarizeTool};
use logging::register_browser_tool;
use map::MapTool;
use names::{
    AGENTIC_CRAWL, CONSOLE_MESSAGES, CONTENT, EXTRACT, LINKS, LIST_TABS, MAP, NETWORK_REQUESTS,
    PAGE_MARKDOWN, PARSE_DOCUMENT, SCRAPE_MANY, SCREENSHOT, SEARCH, SITE_CRAWL_CHUNKS, SUMMARIZE,
    WAIT_FOR,
};
use page::{
    ConsoleMessagesTool, ListLinksTool, ListTabsTool, NetworkRequestsTool, PageContentTool,
    PageMarkdownTool, ScreenshotTool, WaitForTool,
};
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
