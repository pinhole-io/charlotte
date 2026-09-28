use serde_json::{Value, json};
use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;

use crate::error::CharlotteError;

fn extract_json_from_text(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    if let Some(start) = trimmed.find("```json") {
        let rest = &trimmed[start + "```json".len()..];
        if let Some(end) = rest.find("```") {
            let body = rest[..end].trim();
            if let Ok(v) = serde_json::from_str::<Value>(body) {
                return Some(v);
            }
        }
    }
    None
}

pub async fn summarize_markdown(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_options: &ChatOptions,
    markdown: &str,
    focus: Option<&str>,
    max_sentences: usize,
) -> Result<Value, CharlotteError> {
    let mut opts = base_options.clone();
    opts.max_tool_rounds = 1;
    opts.system_prompt = Some(
        "You summarize web page markdown. Return plain text only (no code fences).".to_string(),
    );
    let prompt = format!(
        "Summarize the page in at most {max_sentences} sentence(s). Focus: {}.\n\nMarkdown:\n{}",
        focus.unwrap_or("main points"),
        markdown
    );
    let messages = vec![ChatMessage::text("user", prompt)];
    let registry = ToolRegistry::new();
    let outcome = complete_with_tools(http, &registry, hooks, guardrails, messages, &opts).await?;
    let summary = outcome.content.unwrap_or_default().trim().to_string();
    Ok(json!({
        "ok": true,
        "summary": summary,
    }))
}

pub async fn extract_structured_json(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_options: &ChatOptions,
    markdown: &str,
    focus: Option<&str>,
    prompt: Option<&str>,
    schema: Option<&Value>,
) -> Result<Value, CharlotteError> {
    let mut opts = base_options.clone();
    opts.max_tool_rounds = 1;
    opts.system_prompt = Some(
        "Extract structured JSON from markdown. Return ONLY valid JSON; no prose, no code fences."
            .to_string(),
    );
    let extraction_prompt = prompt.unwrap_or("Extract the key structured fields from the page.");
    let schema_block = schema
        .map(|s| format!("\nRequired JSON schema:\n{}", s))
        .unwrap_or_default();
    let focus_line = focus.unwrap_or("general page facts");
    let user = format!(
        "Task: {extraction_prompt}\nFocus: {focus_line}{schema_block}\n\nMarkdown:\n{markdown}"
    );
    let messages = vec![ChatMessage::text("user", user)];
    let registry = ToolRegistry::new();
    let outcome = complete_with_tools(http, &registry, hooks, guardrails, messages, &opts).await?;
    let raw = outcome.content.unwrap_or_default();
    if let Some(data) = extract_json_from_text(&raw) {
        return Ok(json!({
            "ok": true,
            "data": data,
        }));
    }
    Ok(json!({
        "ok": false,
        "error": "extract_parse_failed",
        "raw": raw,
    }))
}

#[cfg(test)]
mod tests {
    use super::extract_json_from_text;
    use serde_json::json;

    #[test]
    fn extract_json_parses_plain_json() {
        let out = extract_json_from_text(r#"{"a":1}"#);
        assert_eq!(out, Some(json!({"a": 1})));
    }

    #[test]
    fn extract_json_parses_fenced_json() {
        let out = extract_json_from_text("```json\n{\"a\":2}\n```");
        assert_eq!(out, Some(json!({"a": 2})));
    }

    #[test]
    fn extract_json_returns_none_on_invalid() {
        assert!(extract_json_from_text("not json").is_none());
    }
}
