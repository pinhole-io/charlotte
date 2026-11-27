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
use std::sync::Arc;

use charlotte::agents::babe::babe_agent_spec;
use charlotte::agents::charlotte::prepare_completer;
use charlotte::{http_client_from_env, register_charlotte_qdrant_tools};
use jig::{AgentRunner, StreamPiece, TurnInput, VerifierRegistry, WorkflowRegistry};
use qdrant_client::Qdrant;
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::openai::ChatMessage;

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
    let opts = ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from(key),
        model,
        reasoning_effort: Some("high".to_string()),
        temperature: Some(0.2),
        ..Default::default()
    };
    let hooks = Arc::new(HookRegistry::new());
    let guardrails = Arc::new(GuardrailRegistry::new());
    let completer = prepare_completer(http, Arc::clone(&hooks), Arc::clone(&guardrails), &opts);
    let drivers = charlotte::agents::babe::babe_drivers();
    let workflows = WorkflowRegistry::with_chat(20);
    let verifiers = VerifierRegistry::with_builtins();

    println!(
        "Babe (example) — Qdrant tools only. Type exit or quit.\n\
         Qdrant: {qdrant_url}\n"
    );

    let mut transcript: Vec<ChatMessage> = Vec::new();
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

        let thinking = opts
            .reasoning_effort
            .clone()
            .unwrap_or_else(|| "none".to_string());
        print!("\nBabe: ");
        io::stdout().flush().ok();
        let outcome = AgentRunner::run(
            &completer,
            TurnInput {
                spec,
                drivers: &drivers,
                workflows: &workflows,
                verifiers: &verifiers,
                tools: &tools,
                history: &[],
                transcript: &transcript,
                user_text: line,
                volatile: &[],
                cancel: superglue::cancel::CancellationToken::new(),
                single: true,
                unattended: false,
                server_rounds: 20,
                swarm_steps: 1,
                model: &opts.model,
                thinking: &thinking,
                temperature: opts.temperature,
                steering: None,
            },
            |piece| {
                if let StreamPiece::Text(text) = piece {
                    print!("{text}");
                    let _ = io::stdout().flush();
                }
            },
        )
        .await?;
        println!();
        transcript = outcome.transcript;
        if outcome.reply.is_empty() {
            println!("(no text)\n");
        }
    }

    Ok(())
}
