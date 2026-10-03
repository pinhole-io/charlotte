use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use playwright_rs::WaitUntil;
use serde_json::{Value, json};

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::crawl::{CrawlHooks, CrawlRunConfig, SiteCrawlOutputLimits, site_crawl_with_chunks};
use crate::ingest::chunk::ChunkerConfig;
use crate::tools::names::SITE_CRAWL_CHUNKS;

pub(crate) struct SiteCrawlChunksTool(pub(crate) Arc<BrowserHandle>);

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
