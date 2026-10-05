use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use superglue::http::HttpClient;
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::url_policy::UrlPolicy;
use crate::browser::{BrowserHandle, tool_error};
use crate::crawl::agentic::{
    AgenticCrawlConfig, AgenticLlm, CrawlInput, agentic_crawl_with_handle,
};
use crate::tools::ToolRuntimeContext;
use crate::tools::names::AGENTIC_CRAWL;

const DEFAULT_MAX_PAGES: usize = 50;
const DEFAULT_MAX_STEPS: usize = 50;
const MAX_GOAL_CHARS: usize = 4_000;
const MAX_INPUTS: usize = 32;
const MAX_INPUT_NAME_CHARS: usize = 128;
const MAX_INPUT_CHARS: usize = 16_384;
const MAX_SCHEMA_BYTES: usize = 64 * 1024;
const MAX_URL_CHARS: usize = 4_096;
const AGENTIC_CRAWL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

pub(crate) fn agentic_crawl_spec(
    max_pages: usize,
    max_steps: usize,
    respect_robots: bool,
) -> ToolSpec {
    ToolSpec {
        name: AGENTIC_CRAWL.to_string(),
        description: Some(
            "The only page-action tool. Jev picks every click, type, follow, back, scroll, and stop. Start from a direct content URL, preferably from web_search; do not use a search-engine results page because search engines commonly block browser automation. Use this once to walk, fill a form, or complete a task on the headed page. Pass user-supplied values in `inputs` (named fields). Do not put secrets in `goal`. You have no click, type, fill, scroll, or navigate tool. site_crawl_chunks is a BFS dump only.".to_string(),
        ),
        static_tool: false,
        parameters_schema: json!({
            "type": "object",
            "properties": {
                "seed_url": { "type": "string", "maxLength": MAX_URL_CHARS, "description": "HTTP/HTTPS seed URL" },
                "goal": { "type": "string", "maxLength": MAX_GOAL_CHARS, "description": "What the crawl should find or complete" },
                "max_pages": { "type": "integer", "minimum": 1, "maximum": max_pages, "default": DEFAULT_MAX_PAGES.min(max_pages) },
                "max_steps": { "type": "integer", "minimum": 1, "maximum": max_steps, "default": DEFAULT_MAX_STEPS.min(max_steps) },
                "same_origin_only": { "type": "boolean", "description": "If true, only follow same-origin URLs. Default false: Jev may follow any in-page href that UrlPolicy allows.", "default": false },
                "input_text": { "type": "string", "maxLength": MAX_INPUT_CHARS, "description": "A single value to type when Jev picks type (search box). Use `inputs` for a form with more than one field." },
                "inputs": { "type": "object", "maxProperties": MAX_INPUTS, "description": "Named values Jev may type, for example {\"email\": \"a@b.com\", \"password\": \"...\"}. Jev picks the field and the name; it never sees the values. Pass credentials here, never in `goal`.", "additionalProperties": { "type": "string", "maxLength": MAX_INPUT_CHARS } },
                "extract_schema": { "type": "object", "description": "Optional JSON schema for chat extract when Jev sets need_extract" },
                "respect_robots": { "type": "boolean", "default": respect_robots, "description": "Fetch and enforce robots.txt for each origin." }
            },
            "required": ["seed_url", "goal"],
            "additionalProperties": false
        }),
    }
}

pub(crate) struct AgenticCrawlTool {
    pub(crate) browser: Arc<BrowserHandle>,
    pub(crate) policy: UrlPolicy,
    pub(crate) http: HttpClient,
    pub(crate) runtime: Option<Arc<dyn ToolRuntimeContext>>,
    pub(crate) max_pages: usize,
    pub(crate) max_steps: usize,
    pub(crate) timeout: std::time::Duration,
    pub(crate) respect_robots: bool,
}

#[async_trait]
impl Tool for AgenticCrawlTool {
    fn spec(&self) -> ToolSpec {
        agentic_crawl_spec(self.max_pages, self.max_steps, self.respect_robots)
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let seed_url = arguments
            .get("seed_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: seed_url", None))?;
        if seed_url.chars().count() > MAX_URL_CHARS {
            return Err(ToolInvokeError::handler("seed URL is too long", None));
        }
        let goal = arguments
            .get("goal")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: goal", None))?;
        if goal.chars().count() > MAX_GOAL_CHARS {
            return Err(ToolInvokeError::handler("goal is too long", None));
        }
        if arguments
            .get("input_text")
            .and_then(Value::as_str)
            .is_some_and(|value| value.chars().count() > MAX_INPUT_CHARS)
        {
            return Err(ToolInvokeError::handler("input text is too long", None));
        }
        if let Some(inputs) = arguments.get("inputs").and_then(Value::as_object)
            && (inputs.len() > MAX_INPUTS
                || inputs.iter().any(|(name, value)| {
                    name.chars().count() > MAX_INPUT_NAME_CHARS
                        || value
                            .as_str()
                            .is_none_or(|value| value.chars().count() > MAX_INPUT_CHARS)
                }))
        {
            return Err(ToolInvokeError::handler("crawl inputs are invalid", None));
        }
        if arguments
            .get("extract_schema")
            .is_some_and(|schema| schema.to_string().len() > MAX_SCHEMA_BYTES)
        {
            return Err(ToolInvokeError::handler(
                "extraction schema is too large",
                None,
            ));
        }
        let mut cfg = AgenticCrawlConfig::new(seed_url, goal);
        cfg.url_policy = self.policy.clone();
        cfg.max_pages = arguments
            .get("max_pages")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_MAX_PAGES.min(self.max_pages) as u64)
            .clamp(1, self.max_pages as u64) as usize;
        cfg.max_steps = arguments
            .get("max_steps")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_MAX_STEPS.min(self.max_steps) as u64)
            .clamp(1, self.max_steps as u64) as usize;
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
        cfg.respect_robots = self.respect_robots
            || arguments
                .get("respect_robots")
                .and_then(Value::as_bool)
                .unwrap_or(false);

        let llm_context = match &self.runtime {
            Some(runtime) => runtime.llm_context()?,
            None => None,
        };
        let llm = llm_context.as_ref().map(|ctx| AgenticLlm {
            http: ctx.http.clone(),
            hooks: Arc::clone(&ctx.hooks),
            guardrails: Arc::clone(&ctx.guardrails),
            options: ctx.options.clone(),
        });
        let crawl = tokio::time::timeout(
            self.timeout.min(AGENTIC_CRAWL_TIMEOUT),
            agentic_crawl_with_handle(Arc::clone(&self.browser), &cfg, &self.http, llm.as_ref()),
        );
        let result = if let Some(runtime) = &self.runtime {
            tokio::select! {
                biased;
                () = runtime.cancelled() => {
                    return Err(ToolInvokeError::handler(
                        "agentic crawl cancelled",
                        Some("cancelled".to_string()),
                    ));
                }
                result = crawl => result,
            }
        } else {
            crawl.await
        };
        result
            .map_err(|_| {
                ToolInvokeError::handler("agentic crawl timed out", Some("timeout".to_string()))
            })?
            .map_err(tool_error)
    }
}
