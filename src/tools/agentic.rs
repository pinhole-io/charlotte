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
const MAX_CREDENTIAL_ALIAS_CHARS: usize = 32;
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
            "The only page-action tool. System One picks every click, type, follow, back, scroll, and stop. Start from a direct content URL, preferably from web_search; do not use a search-engine results page because search engines commonly block browser automation. Use this once to walk, fill a form, or complete a task on the headed page. Pass user-supplied values in `inputs` (named fields). Do not put secrets in `goal`. You have no click, type, fill, scroll, or navigate tool. site_crawl_chunks is a BFS dump only.".to_string(),
        ),
        static_tool: false,
        parameters_schema: json!({
            "type": "object",
            "properties": {
                "seed_url": { "type": "string", "maxLength": MAX_URL_CHARS, "description": "HTTP/HTTPS seed URL" },
                "goal": { "type": "string", "maxLength": MAX_GOAL_CHARS, "description": "What the crawl should find or complete" },
                "max_pages": { "type": "integer", "minimum": 1, "maximum": max_pages, "default": DEFAULT_MAX_PAGES.min(max_pages) },
                "max_steps": { "type": "integer", "minimum": 1, "maximum": max_steps, "default": DEFAULT_MAX_STEPS.min(max_steps) },
                "same_origin_only": { "type": "boolean", "description": "If true, only follow same-origin URLs. Default false: System One may follow any in-page href that UrlPolicy allows.", "default": false },
                "input_text": { "type": "string", "maxLength": MAX_INPUT_CHARS, "description": "A single value to type when System One picks type (search box). Use `inputs` for a form with more than one field." },
                "inputs": { "type": "object", "maxProperties": MAX_INPUTS, "description": "Named non-secret values System One may type. System One picks the field and the name. Use credential for passwords or other secrets.", "additionalProperties": { "type": "string", "maxLength": MAX_INPUT_CHARS } },
                "credential": { "type": "string", "maxLength": MAX_CREDENTIAL_ALIAS_CHARS, "pattern": "^[a-z0-9-]+$", "description": "A host-owned credential profile alias. The host resolves its values after approval; the model never sees them. Do not combine this with input_text or inputs." },
                "extract_schema": { "type": "object", "description": "Optional JSON schema. When set, each kept page is extracted into this schema. When omitted, System One decides whether to summarize the page body." },
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
        let credential = credential_alias(&arguments)?;
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
        let operation = async {
            if let Some(alias) = credential.as_deref() {
                let runtime = self.runtime.as_ref().ok_or_else(|| {
                    ToolInvokeError::handler(
                        "credentials are unavailable",
                        Some("credential_unavailable".to_string()),
                    )
                })?;
                let resolved = runtime.resolve_credential(alias, &cfg.seed_url).await?;
                if resolved.origins.is_empty() || resolved.inputs.is_empty() {
                    return Err(ToolInvokeError::handler(
                        "credentials are unavailable",
                        Some("credential_unavailable".to_string()),
                    ));
                }
                cfg.inputs = resolved.inputs;
                cfg.credential_origins = Some(resolved.origins);
            }
            agentic_crawl_with_handle(Arc::clone(&self.browser), &cfg, &self.http, llm.as_ref())
                .await
                .map_err(tool_error)
        };
        let crawl = tokio::time::timeout(self.timeout.min(AGENTIC_CRAWL_TIMEOUT), operation);
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
        match result {
            Ok(result) => result,
            Err(_) => Err(ToolInvokeError::handler(
                "agentic crawl timed out",
                Some("timeout".to_string()),
            )),
        }
    }
}

fn credential_alias(arguments: &Value) -> Result<Option<String>, ToolInvokeError> {
    let alias = match arguments.get("credential") {
        None => Ok(None),
        Some(Value::String(alias))
            if !alias.is_empty()
                && alias.len() <= MAX_CREDENTIAL_ALIAS_CHARS
                && alias.bytes().all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                }) =>
        {
            Ok(Some(alias.clone()))
        }
        Some(_) => Err(ToolInvokeError::handler(
            "credential alias is invalid",
            None,
        )),
    }?;
    if alias.is_some()
        && (arguments.get("input_text").is_some() || arguments.get("inputs").is_some())
    {
        return Err(ToolInvokeError::handler(
            "credential cannot be combined with plaintext inputs",
            None,
        ));
    }
    Ok(alias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_alias_is_bounded_and_cannot_mix_with_plaintext() {
        assert_eq!(
            credential_alias(&json!({"credential": "example-login"}))
                .unwrap()
                .as_deref(),
            Some("example-login")
        );
        assert!(credential_alias(&json!({"credential": "Example"})).is_err());

        let arguments = json!({
            "credential": "example-login",
            "inputs": {"password": "marker-password"}
        });
        assert!(credential_alias(&arguments).is_err());
    }

    #[test]
    fn schema_exposes_reference_only_credential_field() {
        let spec = agentic_crawl_spec(5, 5, true);
        let credential = &spec.parameters_schema["properties"]["credential"];
        assert_eq!(credential["pattern"], "^[a-z0-9-]+$");
        let schema = spec.parameters_schema.to_string();
        assert!(!schema.contains("marker-password"));
    }
}
