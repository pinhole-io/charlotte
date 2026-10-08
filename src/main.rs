//! CLI: interactive chat with the Charlotte (web) and Babe (Qdrant) agents.

use std::io::{self, Write};
use std::sync::Arc;

use anyhow::Context;
use charlotte::{
    AutomationProfile, BriefMode, BrowserConfig, BrowserHandle, SearchBackendKind, agents,
    agents::babe::babe_agent_spec, apply_browser_config_from_env, charlotte_offload_config,
    compact_tool_messages, expand_user_request_to_research_brief_shaped, http_client_from_env,
    judge_brief_gate, make_tool_llm_context, new_notepad, register_charlotte_notepad_offload,
    register_charlotte_qdrant_tools, register_charlotte_with_notepad, search_config_from_env,
    should_expand_brief,
};
use clap::{Parser, Subcommand};
use qdrant_client::Qdrant;
use superglue::agents::{AgentEngine, AgentSpec};
use superglue::chat::{ChatOptions, Conversation};
use superglue::events::{ProcessEvent, ProcessEventKind, StatusEmitter, StatusSubscriber};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::providers::{ProviderCredentials, ProviderId};
use tokio::sync::Mutex;

#[derive(Parser)]
#[command(
    name = "charlotte",
    version,
    about = "Interactive chat with Charlotte (web) or Babe (Qdrant) agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive REPL (stdin). Subcommands pick the agent.
    Chat {
        #[command(subcommand)]
        agent: ChatAgent,
    },
}

#[derive(Subcommand)]
enum ChatAgent {
    /// Web research: browser tools (navigate, web_search, page_markdown, etc.). Brief expansion is routed by `CHARLOTTE_BRIEF_MODE`.
    Charlotte {
        /// Show the Chromium window (default is headless).
        #[arg(long)]
        headed: bool,
        /// Best-effort stealth automation profile.
        #[arg(long)]
        stealth: bool,
        /// Print debug telemetry (tool calls + reasoning summaries) while chatting.
        #[arg(long)]
        debug_chat: bool,
    },
    /// Qdrant: vector search and upsert (`qdrant_*`).
    Babe {
        /// Qdrant gRPC URL (also `QDRANT_URL`).
        #[arg(long, default_value = "http://localhost:6334", env = "QDRANT_URL")]
        qdrant_url: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let key = std::env::var("OPENAI_API_KEY").context("set OPENAI_API_KEY for the LLM")?;
    if key.is_empty() {
        anyhow::bail!("OPENAI_API_KEY is empty");
    }

    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.6-luna".to_string());
    let reasoning_effort = model
        .to_ascii_lowercase()
        .contains("luna")
        .then(|| "none".to_string());

    let mut base_opts = ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from(key),
        model,
        // Luna's Chat Completions endpoint rejects function tools unless this
        // field is explicitly set to `none`; reasoning is available via Responses.
        reasoning_effort,
        max_tool_rounds: 20,
        temperature: Some(0.2),
        ..Default::default()
    };
    let env_creds = ProviderCredentials::from_env();
    if env_creds.has_key(ProviderId::TypeSafe) {
        base_opts.provider_credentials = Some(Arc::new(env_creds));
    }

    let http = http_client_from_env()?;
    let hooks = Arc::new(HookRegistry::new());
    let guardrails = Arc::new(GuardrailRegistry::new());

    match cli.command {
        Command::Chat { agent } => match agent {
            ChatAgent::Charlotte {
                headed,
                stealth,
                debug_chat,
            } => {
                run_charlotte(
                    &http,
                    &hooks,
                    &guardrails,
                    &base_opts,
                    debug_chat,
                    apply_browser_config_from_env(BrowserConfig {
                        headless: !headed,
                        automation_profile: if stealth {
                            AutomationProfile::Stealth
                        } else {
                            AutomationProfile::default()
                        },
                        ..Default::default()
                    }),
                )
                .await?;
            }
            ChatAgent::Babe { qdrant_url } => {
                run_babe(
                    &http,
                    hooks.as_ref(),
                    guardrails.as_ref(),
                    &base_opts,
                    &qdrant_url,
                )
                .await?;
            }
        },
    }

    Ok(())
}

fn chat_options_for_agent(spec: &AgentSpec, mut base: ChatOptions) -> ChatOptions {
    base.system_prompt = Some(spec.compile_system_prompt());
    if !spec.model.is_empty() {
        base.model = spec.model.clone();
    }
    base.max_tool_rounds = spec.max_tool_rounds;
    base
}

async fn should_expand_brief_auto(
    mode: BriefMode,
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_opts: &ChatOptions,
    user_text: &str,
) -> (bool, &'static str, Option<charlotte::BriefTaskShape>) {
    match mode {
        BriefMode::Always => (true, "always", None),
        BriefMode::Never => (false, "never", None),
        BriefMode::Heuristic => (should_expand_brief(user_text), "heuristic", None),
        BriefMode::Llm => {
            match judge_brief_gate(http, hooks, guardrails, base_opts, user_text).await {
                Ok(decision) => {
                    let source = if decision.via_system_one { "system_one" } else { "llm" };
                    (decision.expand, source, decision.shape)
                }
                Err(_) => (should_expand_brief(user_text), "llm->heuristic", None),
            }
        }
    }
}

async fn run_charlotte(
    http: &HttpClient,
    hooks: &Arc<HookRegistry>,
    guardrails: &Arc<GuardrailRegistry>,
    base_opts: &ChatOptions,
    debug_chat: bool,
    browser_config: BrowserConfig,
) -> anyhow::Result<()> {
    let browser = BrowserHandle::from_config(browser_config.clone())
        .await
        .map_err(|e| anyhow::anyhow!(e))?;
    let search_config = search_config_from_env().map_err(|e| anyhow::anyhow!(e))?;
    let backend_id = browser.backend_id().await.map_err(|e| anyhow::anyhow!(e))?;
    let cdp_endpoint = std::env::var("CHARLOTTE_CDP_ENDPOINT").ok();
    let tools = superglue::tools::ToolRegistry::new();
    let notepad = new_notepad();
    let llm_context = make_tool_llm_context(http, hooks, guardrails, base_opts);
    register_charlotte_with_notepad(
        &tools,
        &browser,
        &browser_config,
        &search_config,
        http,
        Some(llm_context),
        notepad.clone(),
    )
    .await
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut offload_cfg = charlotte_offload_config();
    offload_cfg.auto_offload = true;
    register_charlotte_notepad_offload(hooks.as_ref(), notepad.clone(), offload_cfg.clone()).await;

    let engine = AgentEngine::new(agents::charlotte::charlotte_agent_spec());
    let mut opts = chat_options_for_agent(&engine.spec, base_opts.clone());
    let brief_mode_env = std::env::var("CHARLOTTE_BRIEF_MODE").ok();
    let brief_mode = BriefMode::parse_env(brief_mode_env.as_deref());
    let status_emitter = {
        let emitter = Arc::new(StatusEmitter::new());
        emitter
            .subscribe(Arc::new(DebugChatSubscriber {
                verbose: debug_chat,
                ..Default::default()
            }) as Arc<dyn StatusSubscriber>)
            .await;
        opts.status_emitter = Some(Arc::clone(&emitter));
        Some(emitter)
    };

    println!(
        "Charlotte — web research. Type exit or quit to leave.\n\
         Environment: OPENAI_BASE_URL, OPENAI_MODEL (optional; default gpt-5.6-luna).\n\
         HTTP: HTTP_MAX_RETRIES, HTTP_RETRY_* , HTTP_QUOTA_PER_SECOND (optional).\n\
         Brief mode: CHARLOTTE_BRIEF_MODE=llm|heuristic|always|never (default llm).\n\
         Browser: CHARLOTTE_STORAGE_STATE_PATH, CHARLOTTE_SEARCH_URL_TEMPLATE, CHARLOTTE_WEB_SEARCH_ACTION_TIMEOUT_SECS, CHARLOTTE_ALLOW_HOST_SUFFIXES, CHARLOTTE_BLOCK_PRIVATE_IPS, CHARLOTTE_BROWSER_BACKEND, CHARLOTTE_CDP_ENDPOINT (optional).\n\
         Search: CHARLOTTE_SEARCH_BACKEND=searxng|browser, CHARLOTTE_SEARXNG_URL (default http://127.0.0.1:18088), CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS (default 20).\n\
         Notepad: notepad_* tools enabled; large tool output auto-offloads to stubs.\n\
         Parallel: scrape_many opens isolated tabs on the shared browser (default 3 at a time).\n\
         Active search backend: {search_backend}{search_detail}\n\
         Active browser backend: {backend_id}{}\n\
         Active model: {model}\n",
        cdp_endpoint
            .as_ref()
            .map(|e| format!(" ({e})"))
            .unwrap_or_default(),
        search_backend = search_config.backend.as_str(),
        search_detail = match search_config.backend {
            SearchBackendKind::Searxng => {
                format!(" ({})", search_config.searxng_base_url)
            }
            SearchBackendKind::Browser => String::new(),
        },
        model = base_opts.model,
    );
    eprintln!("[trace] tool calls print here. Pass --debug-chat for model reasoning.");
    if debug_chat {
        eprintln!("[debug] enabled: reasoning summaries");
    }

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

        let (should_expand, gate_source, brief_shape) = should_expand_brief_auto(
            brief_mode,
            http,
            hooks.as_ref(),
            guardrails.as_ref(),
            base_opts,
            line,
        )
        .await;
        eprintln!(
            "brief_mode={} ({gate_source})",
            if should_expand { "expanded" } else { "direct" }
        );
        let user_message = if should_expand {
            let mut brief_opts = base_opts.clone();
            if let Some(emitter) = status_emitter.as_ref() {
                brief_opts.status_emitter = Some(Arc::clone(emitter));
            }
            let brief = expand_user_request_to_research_brief_shaped(
                http,
                hooks.as_ref(),
                guardrails.as_ref(),
                &brief_opts,
                &search_config,
                line,
                brief_shape,
            )
            .await?;
            eprintln!("\n--- Research brief ---\n{brief}\n");
            format!(
                "**Original request:** {line}\n\n\
                 **Research brief (execute with the registered browser tools until success criteria are met):**\n\n\
                 {brief}",
                line = line,
                brief = brief
            )
        } else {
            line.to_string()
        };
        conv.push_user(user_message);
        let outcome = conv
            .complete(http, &tools, hooks.as_ref(), guardrails.as_ref(), &opts)
            .await?;
        {
            let mut np = notepad.write().await;
            compact_tool_messages(&mut conv.messages, &mut np, &offload_cfg);
        }
        println!(
            "\nCharlotte: {}\n",
            outcome.content.unwrap_or_else(|| "(no text)".to_string())
        );
        eprintln!(
            "[rounds: {}, finish: {:?}]",
            outcome.rounds, outcome.finish_reason
        );
    }

    browser.close().await.map_err(|e| anyhow::anyhow!(e))?;
    Ok(())
}

#[derive(Default)]
struct DebugChatSubscriber {
    verbose: bool,
    reasoning_by_round: Mutex<std::collections::HashMap<u32, String>>,
}

#[async_trait::async_trait]
impl StatusSubscriber for DebugChatSubscriber {
    async fn on_event(&self, event: ProcessEvent) {
        match event.kind {
            ProcessEventKind::ToolCallStart => {
                let name = event
                    .metadata
                    .get("tool_name")
                    .map_or("unknown", String::as_str);
                let args = event
                    .metadata
                    .get("arguments")
                    .map(|s| preview(s, 180))
                    .unwrap_or_else(|| "{}".to_string());
                eprintln!("[tool:call  r{}] calling {}({})", event.round, name, args);
            }
            ProcessEventKind::ToolCallEnd => {
                let name = event
                    .metadata
                    .get("tool_name")
                    .map_or("unknown", String::as_str);
                let result = event
                    .metadata
                    .get("result")
                    .map(|s| preview(s, 180))
                    .unwrap_or_else(|| "(no result metadata)".to_string());
                eprintln!("[tool:done  r{}] {} returned {}", event.round, name, result);
            }
            ProcessEventKind::ReasoningDelta => {
                if let Some(delta) = event.metadata.get("delta") {
                    if self.verbose {
                        let trimmed = delta.trim();
                        if !trimmed.is_empty() {
                            eprintln!("[thinking  r{}] {}", event.round, preview(trimmed, 200));
                        }
                    }
                    let mut by_round = self.reasoning_by_round.lock().await;
                    by_round
                        .entry(event.round)
                        .and_modify(|s| s.push_str(delta))
                        .or_insert_with(|| delta.clone());
                }
            }
            ProcessEventKind::LlmCallStart => {
                if self.verbose {
                    eprintln!(
                        "[llm:think r{}] model={} thinking...",
                        event.round, event.model
                    );
                }
            }
            ProcessEventKind::LlmCallEnd => {
                if event.tool_call_count > 0 {
                    eprintln!(
                        "[llm:done  r{}] model={} decided {} tool call(s)",
                        event.round, event.model, event.tool_call_count
                    );
                } else if self.verbose {
                    if let Some(usage) = event.usage.as_ref() {
                        eprintln!(
                            "[llm:done  r{}] model={} tokens(p/c/t)={}/{}/{}",
                            event.round,
                            event.model,
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens
                        );
                    } else {
                        eprintln!("[llm:done  r{}] model={}", event.round, event.model);
                    }
                }
                let summary = {
                    let mut by_round = self.reasoning_by_round.lock().await;
                    by_round.remove(&event.round)
                };
                if self.verbose {
                    if let Some(text) = summary {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            eprintln!("[reasoning r{}] {}", event.round, preview(trimmed, 300));
                        }
                    }
                }
            }
            ProcessEventKind::LlmCallError => {
                eprintln!(
                    "[llm:error r{}] model={} error={}",
                    event.round,
                    event.model,
                    event.error_type.unwrap_or_else(|| "unknown".to_string())
                );
            }
            ProcessEventKind::ToolRoute => {}
            ProcessEventKind::VoiceSessionStart => {
                eprintln!("[voice:start r{}] model={}", event.round, event.model);
            }
            ProcessEventKind::VoiceSessionEnd => {
                eprintln!("[voice:end   r{}] model={}", event.round, event.model);
            }
            ProcessEventKind::VoiceFunctionCall => {
                let name = event
                    .metadata
                    .get("tool_name")
                    .or_else(|| event.metadata.get("name"))
                    .map_or("unknown", String::as_str);
                eprintln!(
                    "[voice:fn    r{}] model={} name={}",
                    event.round, event.model, name
                );
            }
        }
    }
}

fn preview(s: &str, max_chars: usize) -> String {
    let compact = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        compact
    } else {
        format!("{}...", compact.chars().take(max_chars).collect::<String>())
    }
}

async fn run_babe(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_opts: &ChatOptions,
    qdrant_url: &str,
) -> anyhow::Result<()> {
    let api_key = std::env::var("QDRANT_API_KEY")
        .ok()
        .filter(|s| !s.is_empty());
    let client = Qdrant::from_url(qdrant_url)
        .api_key(api_key)
        .build()
        .map_err(|e| anyhow::anyhow!(e))?;

    let tools = superglue::tools::ToolRegistry::new();
    register_charlotte_qdrant_tools(&tools, &client)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let spec = babe_agent_spec();
    let engine = AgentEngine::new(spec.clone());
    let opts = chat_options_for_agent(&engine.spec, base_opts.clone());

    println!(
        "Babe — Qdrant. Type exit or quit to leave.\n\
         Qdrant: {qdrant_url} (optional QDRANT_API_KEY).\n"
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
            .complete(http, &tools, hooks, guardrails, &opts)
            .await?;
        println!(
            "\nBabe: {}\n",
            outcome.content.unwrap_or_else(|| "(no text)".to_string())
        );
        eprintln!(
            "[rounds: {}, finish: {:?}]",
            outcome.rounds, outcome.finish_reason
        );
    }

    Ok(())
}
