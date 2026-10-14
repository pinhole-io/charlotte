//! HTTP `fetch` with Markdown extraction for host-selected Charlotte tools.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use url::Url;

use superglue::http::HttpClient;
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

use crate::browser::tool_error;
use crate::browser::url_policy::UrlPolicy;
use crate::error::CharlotteError;
use crate::ingest::markdown::{AGENT_DEFAULT_MAX_MARKDOWN_CHARS, html_to_markdown_limited};
use crate::tools::names::FETCH;

const MAX_URL_BYTES: usize = 8 * 1024;
const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) fn fetch_spec() -> ToolSpec {
    ToolSpec {
        name: FETCH.to_string(),
        description: Some(
            "Fetch an HTTP or HTTPS URL and return extracted Markdown for HTML, or capped plain text. \
             Prefer web_search snippets when they already answer the ask. Use this when a page body is needed. \
             Do not fetch many full pages when a few snippets are enough."
                .to_string(),
        ),
        static_tool: false,
        parameters_schema: json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_URL_BYTES,
                    "description": "Absolute http or https URL"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        }),
    }
}

pub(crate) struct FetchTool {
    pub(crate) http: HttpClient,
    pub(crate) policy: UrlPolicy,
}

#[async_trait]
impl Tool for FetchTool {
    fn spec(&self) -> ToolSpec {
        fetch_spec()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let Some(url) = arguments.get("url").and_then(Value::as_str) else {
            return Ok(failure("invalid arguments", None));
        };
        fetch_url(&self.http, &self.policy, url)
            .await
            .map_err(tool_error)
    }
}

/// Fetch a public URL and return Markdown or capped text for the model.
pub async fn fetch_url(
    http: &HttpClient,
    policy: &UrlPolicy,
    raw_url: &str,
) -> Result<Value, CharlotteError> {
    let url = match parse_fetch_url(raw_url) {
        Ok(url) => url,
        Err(error) => return Ok(failure(error, None)),
    };
    if let Err(error) = policy.check_resolved(url.as_str()).await {
        return Ok(failure(&error.to_string(), None));
    }

    let client = match http.clone_with_timeouts(REQUEST_TIMEOUT, CONNECT_TIMEOUT) {
        Ok(client) => client,
        Err(_) => return Ok(failure("request failed", None)),
    };
    let body = match client.get(url.as_str()).await {
        Ok(body) => body,
        Err(error) => {
            let status = error.status().map(|code| code.as_u16());
            return Ok(failure("request failed", status));
        }
    };
    if body.len() > MAX_BODY_BYTES {
        return Ok(failure("response too large", None));
    }
    let raw = String::from_utf8_lossy(&body);
    let content_type = if looks_like_html(None, &raw) {
        Some("text/html")
    } else {
        Some("text/plain")
    };
    let (text, truncated) = match extract_body(&body, content_type) {
        Ok(extracted) => extracted,
        Err(error) => return Ok(failure(&error.to_string(), None)),
    };
    Ok(json!({
        "ok": true,
        "status": 200,
        "url": url.as_str(),
        "content_type": content_type,
        "text": text,
        "truncated": truncated,
    }))
}

fn parse_fetch_url(raw: &str) -> Result<Url, &'static str> {
    if raw.is_empty() || raw.len() > MAX_URL_BYTES || raw.chars().any(char::is_control) {
        return Err("invalid URL");
    }
    let url = Url::parse(raw).map_err(|_| "invalid URL")?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("URL must use http or https");
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URL credentials are not allowed");
    }
    Ok(url)
}

fn extract_body(body: &[u8], content_type: Option<&str>) -> Result<(String, bool), CharlotteError> {
    let raw = String::from_utf8_lossy(body);
    if looks_like_html(content_type, &raw) {
        return html_to_markdown_limited(&raw, AGENT_DEFAULT_MAX_MARKDOWN_CHARS);
    }
    let truncated = raw.chars().count() > AGENT_DEFAULT_MAX_MARKDOWN_CHARS;
    let text = if truncated {
        raw.chars().take(AGENT_DEFAULT_MAX_MARKDOWN_CHARS).collect()
    } else {
        raw.into_owned()
    };
    Ok((text, truncated))
}

fn looks_like_html(content_type: Option<&str>, body: &str) -> bool {
    if content_type.is_some_and(|value| {
        let lower = value.to_ascii_lowercase();
        lower.contains("text/html") || lower.contains("application/xhtml")
    }) {
        return true;
    }
    let head = body
        .chars()
        .take(256)
        .collect::<String>()
        .to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html")
}

fn failure(error: &str, status: Option<u16>) -> Value {
    match status {
        Some(status) => json!({
            "ok": false,
            "error": error,
            "status": status,
        }),
        None => json!({
            "ok": false,
            "error": error
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_html_drops_script_and_returns_markdown() {
        let html = r#"<!DOCTYPE html>
<html><head><title>T</title><script>evil()</script></head>
<body><h1>Hello</h1><p>World</p></body></html>"#;
        let (text, truncated) =
            extract_body(html.as_bytes(), Some("text/html; charset=utf-8")).expect("extract");
        assert!(!truncated);
        assert!(!text.contains("evil"));
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn extract_long_html_sets_truncated() {
        let mut body = String::from("<!DOCTYPE html><html><body>");
        body.push_str(&"word ".repeat(AGENT_DEFAULT_MAX_MARKDOWN_CHARS));
        body.push_str("</body></html>");
        let (text, truncated) = extract_body(body.as_bytes(), Some("text/html")).expect("extract");
        assert!(truncated);
        assert!(text.chars().count() <= AGENT_DEFAULT_MAX_MARKDOWN_CHARS);
    }

    #[test]
    fn extract_plain_text_caps_without_html_conversion() {
        let body = "a".repeat(AGENT_DEFAULT_MAX_MARKDOWN_CHARS + 10);
        let (text, truncated) = extract_body(body.as_bytes(), Some("text/plain")).expect("extract");
        assert!(truncated);
        assert_eq!(text.chars().count(), AGENT_DEFAULT_MAX_MARKDOWN_CHARS);
        assert_eq!(text, "a".repeat(AGENT_DEFAULT_MAX_MARKDOWN_CHARS));
    }

    #[test]
    fn rejects_credentials_and_schemes() {
        assert!(parse_fetch_url("https://user:pass@example.com/").is_err());
        assert!(parse_fetch_url("file:///etc/passwd").is_err());
        assert!(parse_fetch_url("https://example.com/ok").is_ok());
    }
}
