use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use playwright_rs::WaitUntil;
use serde_json::{Value, json};

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::crawl::{CrawlHooks, CrawlRunConfig, site_map};
use crate::tools::names::MAP;

pub(crate) struct MapTool(pub(crate) Arc<BrowserHandle>);

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
