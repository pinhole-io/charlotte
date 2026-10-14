use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use superglue::http::HttpClient;
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::search::{SearchConfig, web_search as dispatch_web_search};
use crate::tools::names::SEARCH;

const MAX_SEARCH_QUERY_CHARS: usize = 2_000;

pub(crate) fn web_search_spec(max_results: usize) -> ToolSpec {
    ToolSpec {
        name: SEARCH.to_string(),
        description: Some(
            "Run a web search via SearXNG JSON API (default, Bing) or DuckDuckGo HTML when CHARLOTTE_SEARCH_BACKEND=browser. Returns { title, url, snippet } rows. \
             Use the returned pages. A page about the same term in another field is a source. Do not drop the results and answer from memory. \
             Prefer answering from titles/snippets when they already contain the fact. When page text is needed, use fetch on a result URL (or scrape_many when that tool is available). \
             Default SearXNG/Bing does NOT support `site:`. With the browser backend, `site:` is a root domain only (`site:example.com topic`); never `site:domain/path`."
                .to_string(),
        ),
        static_tool: false,
        parameters_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SEARCH_QUERY_CHARS,
                    "description": "Search query. On SearXNG/Bing, do not use `site:`. A result about the same term in another field is still a source."
                },
                "max_results": { "type": "integer", "minimum": 1, "maximum": max_results, "description": "Max results", "default": 8.min(max_results) }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
    }
}

pub(crate) struct WebSearchTool {
    pub(crate) browser: Arc<BrowserHandle>,
    pub(crate) http: HttpClient,
    pub(crate) search: SearchConfig,
    pub(crate) max_results: usize,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        web_search_spec(self.max_results)
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: query", None))?;
        if query.trim().is_empty() || query.chars().count() > MAX_SEARCH_QUERY_CHARS {
            return Err(ToolInvokeError::handler(
                "search query length is invalid",
                None,
            ));
        }
        let max_results = arguments
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(8)
            .clamp(1, self.max_results as u64) as usize;
        dispatch_web_search(
            &self.http,
            Some(&self.browser),
            &self.search,
            query,
            max_results,
        )
        .await
        .map_err(tool_error)
    }
}
