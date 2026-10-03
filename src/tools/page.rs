use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::time::sleep;

use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::{BrowserHandle, tool_error};
use crate::ingest::markdown::{
    AGENT_DEFAULT_MAX_HTML_BYTES, AGENT_DEFAULT_MAX_MARKDOWN_CHARS, clamp_agent_html_bytes,
    clamp_agent_markdown_chars,
};
use crate::tools::names::{
    CONSOLE_MESSAGES, CONTENT, LINKS, LIST_TABS, NETWORK_REQUESTS, PAGE_MARKDOWN, SCREENSHOT,
    WAIT_FOR,
};

pub(crate) struct PageContentTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct PageMarkdownTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct ListLinksTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct WaitForTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct ScreenshotTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct ListTabsTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct NetworkRequestsTool(pub(crate) Arc<BrowserHandle>);
pub(crate) struct ConsoleMessagesTool(pub(crate) Arc<BrowserHandle>);

#[async_trait]
impl Tool for PageContentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: CONTENT.to_string(),
            description: Some(
                "Read visible text from the current page, optionally limited to a CSS selector. Missing selectors fail immediately (no 30s wait). Long text is truncated to the configured max.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "Optional CSS selector; omit for full page body text" }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let selector = arguments
            .get("selector")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        self.0.page_content(selector).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for PageMarkdownTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: PAGE_MARKDOWN.to_string(),
            description: Some(
                "Read the current page as Markdown: body HTML snapshot converted MarkItDown-style (strip script/style, ATX headings). Optional focus string prioritizes sections matching what you are looking for. Defaults keep responses within model context (~32k chars); check truncated and cleanup metadata.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "focus": { "type": "string", "description": "Optional: what you are trying to learn from this page (e.g. product price, release date); ranks relevant markdown sections" },
                    "max_html_bytes": { "type": "integer", "description": "Truncate HTML before conversion (default from browser config, typically 600000; hard cap 2000000)", "default": AGENT_DEFAULT_MAX_HTML_BYTES },
                    "max_markdown_chars": { "type": "integer", "description": "Truncate Markdown returned to the model (default from browser config, typically 32000; hard cap 100000)", "default": AGENT_DEFAULT_MAX_MARKDOWN_CHARS },
                    "post_wait_ms": { "type": "integer", "description": "Pause before snapshot (helps client-rendered pages); default 0", "default": 0 }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let focus = arguments
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let default_html = self.0.max_html_bytes_limit().await.map_err(tool_error)?;
        let default_md = self
            .0
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
        let post_wait_ms = arguments
            .get("post_wait_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if post_wait_ms > 0 {
            sleep(Duration::from_millis(post_wait_ms)).await;
        }
        self.0
            .page_markdown(focus, max_html_bytes, max_markdown_chars)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ListLinksTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: LINKS.to_string(),
            description: Some(
                "List links (`href` + visible text) on the current page, optionally under a root selector. Can filter to same-origin only.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "Optional CSS selector for a subtree" },
                    "limit": { "type": "integer", "description": "Max links (1–200)", "default": 50 },
                    "same_origin_only": { "type": "boolean", "description": "Only links with same scheme/host/port as the page", "default": false }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let selector = arguments
            .get("selector")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;
        let same_origin_only = arguments
            .get("same_origin_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        self.0
            .list_links(selector, limit, same_origin_only)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for WaitForTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: WAIT_FOR.to_string(),
            description: Some("Wait for a selector to appear/be actionable.".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "selector": { "type": "string", "description": "Required CSS selector" },
                    "timeout_ms": { "type": "integer", "description": "Optional wait timeout in milliseconds", "default": 30000 }
                },
                "required": ["selector"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let selector = arguments
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: selector", None))?;
        let timeout_ms = arguments
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .map(|v| v as f64);
        self.0
            .wait_for_selector(selector, timeout_ms)
            .await
            .map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ScreenshotTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SCREENSHOT.to_string(),
            description: Some("Capture a PNG screenshot (base64 payload).".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "full_page": { "type": "boolean", "description": "Capture full scrollable page", "default": false }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let full_page = arguments
            .get("full_page")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        self.0.screenshot(full_page).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ListTabsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: LIST_TABS.to_string(),
            description: Some("List open tabs for the current browser context.".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _arguments: Value) -> Result<Value, ToolInvokeError> {
        self.0.list_tabs().await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for NetworkRequestsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: NETWORK_REQUESTS.to_string(),
            description: Some(
                "List recent network events from the active page session.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 100, "description": "Max events to return (1-500)" }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        self.0.network_requests(limit).await.map_err(tool_error)
    }
}

#[async_trait]
impl Tool for ConsoleMessagesTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: CONSOLE_MESSAGES.to_string(),
            description: Some("List recent console messages from the active page.".to_string()),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 100, "description": "Max messages to return (1-500)" }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        self.0.console_messages(limit).await.map_err(tool_error)
    }
}
