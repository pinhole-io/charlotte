//! Babe agent example: Qdrant tools only (no browser).
//!
//! Requires a running Qdrant instance and `OPENAI_API_KEY`.
//!
//! ```text
//! export OPENAI_API_KEY=sk-...
//! export QDRANT_URL=http://localhost:6334   # optional
//! cargo run --example babe_chat
//! ```

use std::io::{self, Write};

use charlotte::{
    agents::babe::babe_agent_spec, http_client_from_env, register_charlotte_qdrant_tools,
};
use qdrant_client::Qdrant;
use superglue::chat::{ChatOptions, Conversation};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("Set OPENAI_API_KEY to run this example.");
            return Ok(());
        }
    };

    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let api_key = std::env::var("QDRANT_API_KEY")
        .ok()
        .filter(|s| !s.is_empty());
    let client = Qdrant::from_url(&qdrant_url).api_key(api_key).build()?;

    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());
    let model = std::env::var("OPENAI_MODEL")
        .unwrap_or_else(|_| "gpt-5.4-nano-2026-03-17-mini".to_string());

    let http = http_client_from_env()?;
    let tools = superglue::tools::ToolRegistry::new();
    register_charlotte_qdrant_tools(&tools, &client).await?;

    let spec = babe_agent_spec();
    let mut opts = ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from(key),
        model,
        max_tool_rounds: spec.max_tool_rounds,
        temperature: Some(0.2),
        ..Default::default()
    };
    opts.system_prompt = Some(spec.compile_system_prompt());
    if !spec.model.is_empty() {
        opts.model = spec.model.clone();
    }

    let hooks = HookRegistry::new();
    let guardrails = GuardrailRegistry::new();

    println!(
        "Babe (example) — Qdrant tools only. Type exit or quit.\n\
         Qdrant: {qdrant_url}\n"
    );

    let mut conv = Conversation::new();
    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.eq_ignore_ascii_case("exit") || line.eq_ignore_ascii_case("quit") {
            break;
        }

        conv.push_user(line.to_string());
        let outcome = conv
            .complete(&http, &tools, &hooks, &guardrails, &opts)
            .await?;
        println!(
            "\nBabe: {}\n",
            outcome.content.unwrap_or_else(|| "(no text)".to_string())
        );
    }

    Ok(())
}
