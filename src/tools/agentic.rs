use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use superglue::http::HttpClient;
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::url_policy::UrlPolicy;
use crate::browser::{BrowserHandle, tool_error};
use crate::crawl::agentic::{
    AGENTIC_CRAWL_NO_LIMIT, AgenticCrawlConfig, AgenticLlm, CrawlInput,
    agentic_crawl_sidecar_enabled, agentic_crawl_with_handle, agentic_crawl_with_sidecar,
};
use crate::tools::ToolLlmContext;
use crate::tools::names::AGENTIC_CRAWL;

pub(crate) struct AgenticCrawlTool {
    pub(crate) browser: Arc<BrowserHandle>,
    pub(crate) policy: UrlPolicy,
    pub(crate) http: HttpClient,
    pub(crate) llm: Option<Arc<ToolLlmContext>>,
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
