use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::ingest::parse_document::{DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT, parse_document_url};
use crate::tools::names::PARSE_DOCUMENT;

pub(crate) struct ParseDocumentTool(pub(crate) Arc<BrowserHandle>);

#[async_trait]
impl Tool for ParseDocumentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: PARSE_DOCUMENT.to_string(),
            description: Some(
                "Fetch and parse a document URL (PDF currently) into markdown-like text."
                    .to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Document URL (PDF)" },
                    "max_markdown_chars": { "type": "integer", "default": DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT }
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let url = arguments
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: url", None))?;
        self.0.enforce_url_policy(url).await.map_err(tool_error)?;
        let max_markdown_chars = arguments
            .get("max_markdown_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT);
        parse_document_url(url, max_markdown_chars)
            .await
            .map_err(tool_error)
    }
}
