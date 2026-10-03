use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::crawl::loopguard::{SharedGuardState, blocked_target_response};
use crate::crawl::scrape_many::{
    SCRAPE_MANY_DEFAULT_MAX_TOTAL_TEXT_CHARS, ScrapeManyOptions, ScrapeRequest, clamp_concurrency,
    scrape_many_on, validate_urls,
};
use crate::ingest::markdown::clamp_agent_markdown_chars;
use crate::tools::names::SCRAPE_MANY;

pub(crate) struct ScrapeManyTool {
    pub(crate) browser: Arc<BrowserHandle>,
    pub(crate) guard_state: SharedGuardState,
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
