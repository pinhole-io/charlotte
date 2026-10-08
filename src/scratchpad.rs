//! Session scratchpad integration via [notepad](https://github.com/Bioto/thestack/tree/main/projects/notepad).
//!
//! Charlotte browser tools can produce large payloads (especially
//! `site_crawl_chunks`). Pair them with notepad tools so the model
//! stores distilled findings and optional offload hooks keep the active context small.

use std::sync::Arc;

use notepad::register_offload_hook;
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::tools::{ToolInvokeError, ToolRegistry};
use tokio::sync::RwLock;

use crate::browser::{BrowserConfig, BrowserHandle};
use crate::search::SearchConfig;
use crate::tools::{CharlotteToolProfile, ToolLlmContext, register_charlotte_tools};

/// Shared session notepad handle for Charlotte + notepad tool registration.
pub type SharedNotepad = Arc<RwLock<Notepad>>;

/// Create a new in-process session notepad.
#[must_use]
pub fn new_notepad() -> SharedNotepad {
    Arc::new(RwLock::new(Notepad::new()))
}

/// Default offload config tuned for Charlotte (`site_crawl_chunks` always offloads when auto mode is on).
#[must_use]
pub fn charlotte_offload_config() -> notepad::OffloadConfig {
    let mut cfg = notepad::OffloadConfig::default();
    if !cfg
        .never_offload_tools
        .iter()
        .any(|name| name == "web_search")
    {
        cfg.never_offload_tools.push("web_search".to_string());
    }
    if !cfg
        .always_offload_tools
        .iter()
        .any(|name| name == "site_crawl_chunks")
    {
        cfg.always_offload_tools
            .push("site_crawl_chunks".to_string());
    }
    // Crawl control (stopped_reason, trace, acts) must stay inline.
    // Offloading replaces it with Wikipedia chrome and hides why System One stopped.
    cfg.always_offload_tools
        .retain(|name| name != "agentic_crawl");
    if !cfg
        .never_offload_tools
        .iter()
        .any(|name| name == "agentic_crawl")
    {
        cfg.never_offload_tools.push("agentic_crawl".to_string());
    }
    cfg
}

/// Register Charlotte browser tools and notepad scratchpad tools on one registry.
pub async fn register_charlotte_with_notepad(
    registry: &ToolRegistry,
    browser: &BrowserHandle,
    browser_config: &BrowserConfig,
    search_config: &SearchConfig,
    http: &HttpClient,
    llm_context: Option<ToolLlmContext>,
    notepad: SharedNotepad,
) -> Result<(), ToolInvokeError> {
    register_charlotte_tools(
        registry,
        browser,
        browser_config,
        search_config,
        http,
        llm_context,
        CharlotteToolProfile::Full,
    )
    .await?;
    notepad::register_notepad_tools(registry, notepad).await
}

pub fn make_tool_llm_context(
    http: &HttpClient,
    hooks: &Arc<HookRegistry>,
    guardrails: &Arc<GuardrailRegistry>,
    options: &ChatOptions,
) -> ToolLlmContext {
    ToolLlmContext {
        http: http.clone(),
        hooks: Arc::clone(hooks),
        guardrails: Arc::clone(guardrails),
        options: options.clone(),
    }
}

/// Register the notepad PostTool offload hook (optional automatic spill for large tool JSON).
pub async fn register_charlotte_notepad_offload(
    hooks: &HookRegistry,
    notepad: SharedNotepad,
    config: notepad::OffloadConfig,
) {
    register_offload_hook(hooks, notepad, config).await;
}

pub use notepad::{
    Entry, EntryId, Notepad, OffloadConfig, OffloadHook, StoredPayload, compact_tool_messages,
    is_notepad_stub, register_notepad_tools,
};

#[cfg(feature = "synthesize")]
pub use notepad::{NotepadSynthesizer, SynthesisConfig, SynthesisError, SynthesisOutcome};

#[cfg(test)]
mod tests {
    use super::*;
    use superglue::openai::ChatMessage;

    #[test]
    fn charlotte_offload_config_includes_site_crawl() {
        let cfg = charlotte_offload_config();
        assert!(
            cfg.always_offload_tools
                .iter()
                .any(|name| name == "site_crawl_chunks"),
            "expected site_crawl_chunks in always_offload_tools"
        );
        assert!(
            cfg.never_offload_tools
                .iter()
                .any(|name| name == "agentic_crawl"),
            "expected agentic_crawl in never_offload_tools"
        );
        assert!(
            !cfg.always_offload_tools
                .iter()
                .any(|name| name == "agentic_crawl"),
            "agentic_crawl must not always-offload"
        );
        assert!(
            cfg.never_offload_tools
                .iter()
                .any(|name| name == "web_search"),
            "expected web_search in never_offload_tools"
        );
    }

    #[test]
    fn agentic_crawl_result_stays_inline() {
        let cfg = charlotte_offload_config();
        let hook = OffloadHook::new(new_notepad(), cfg);
        let body = r#"{"ok":true,"stopped_reason":"system_one_stop","trace":[{"act":"follow"}],"pages":[{"url":"https://en.wikipedia.org/wiki/Sun","title":"Sun","excerpt":"Jump to content"}]}"#;
        assert!(
            !hook.should_offload("agentic_crawl", body, None),
            "agentic_crawl envelope must stay inline"
        );
    }

    #[test]
    fn compact_tool_messages_stores_and_stubs_large_tool_output() {
        let mut cfg = charlotte_offload_config();
        cfg.auto_offload = true;
        cfg.min_chars = 64;

        let mut pad = Notepad::new();
        let large_payload = format!("{{\"text\":\"{}\"}}", "x".repeat(2048));
        let mut msg = ChatMessage::text("tool", large_payload);
        msg.name = Some("page_markdown".to_string());
        let mut messages = vec![msg];

        compact_tool_messages(&mut messages, &mut pad, &cfg);

        let stub = messages[0]
            .content
            .as_ref()
            .and_then(|c| c.as_text())
            .expect("tool message should stay text");
        assert!(is_notepad_stub(stub), "expected notepad stub, got: {stub}");
        assert_eq!(pad.entries().len(), 1, "expected one notepad entry");
    }
}
