use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::ingest::extract::{extract_structured_json, summarize_markdown};
use crate::ingest::markdown::{
    AGENT_DEFAULT_MAX_HTML_BYTES, AGENT_DEFAULT_MAX_MARKDOWN_CHARS, clamp_agent_html_bytes,
    clamp_agent_markdown_chars,
};
use crate::tools::ToolLlmContext;
use crate::tools::names::{EXTRACT, SUMMARIZE};

pub(crate) struct ExtractTool {
    pub(crate) browser: Arc<BrowserHandle>,
    pub(crate) llm: Option<Arc<ToolLlmContext>>,
}
pub(crate) struct SummarizeTool {
    pub(crate) browser: Arc<BrowserHandle>,
    pub(crate) llm: Option<Arc<ToolLlmContext>>,
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
