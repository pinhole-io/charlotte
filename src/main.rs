//! CLI: interactive chat with the Charlotte (web) and Babe (Qdrant) agents.

use std::io::{self, Write};
use std::sync::Arc;

use anyhow::Context;
use charlotte::{
    AutomationProfile, BriefMode, BrowserConfig, BrowserHandle, SearchBackendKind, agents,
    apply_browser_config_from_env, charlotte_offload_config, compact_tool_messages,
    expand_user_request_to_research_brief_shaped, http_client_from_env, judge_brief_gate,
    make_tool_llm_context, new_notepad, register_charlotte_notepad_offload,
    register_charlotte_qdrant_tools, register_charlotte_with_notepad, search_config_from_env,
    should_expand_brief,
};
use clap::{Parser, Subcommand};
use jig::{
    AgentRunner, AgentSpec, EngineDriver, StreamPiece, ToolPoint, TurnInput, VerifierRegistry,
    WorkflowRegistry,
};
use qdrant_client::Qdrant;
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::openai::ChatMessage;
use superglue::providers::{ProviderCredentials, ProviderId};

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
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-6-luna".to_string());

    let mut base_opts = ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from(key),
        model,
        // Chat Completions rejects Luna function tools unless this is `none`.
        // The chat loops below use the Responses API, which keeps `high` and
        // still returns tool-call arguments.
        reasoning_effort: Some("high".to_string()),
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
                run_babe(&http, &hooks, &guardrails, &base_opts, &qdrant_url).await?;
            }
        },
    }

    Ok(())
}

async fn should_expand_brief_auto(
    mode: BriefMode,
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_opts: &ChatOptions,
    user_text: &str,
) -> anyhow::Result<(bool, &'static str, Option<charlotte::BriefTaskShape>)> {
    match mode {
        BriefMode::Always => Ok((true, "always", None)),
        BriefMode::Never => Ok((false, "never", None)),
        BriefMode::Heuristic => Ok((should_expand_brief(user_text), "heuristic", None)),
        BriefMode::Llm => {
            let decision = judge_brief_gate(http, hooks, guardrails, base_opts, user_text).await?;
            let source = if decision.via_system_one {
                "system_one"
            } else {
                "llm"
            };
            Ok((decision.expand, source, decision.shape))
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

    let spec = agents::charlotte::charlotte_agent_spec();
    let completer = agents::charlotte::prepare_completer(
        http.clone(),
        Arc::clone(hooks),
        Arc::clone(guardrails),
        base_opts,
    );
    let drivers = agents::charlotte::charlotte_drivers();
    let workflows = WorkflowRegistry::with_chat(500);
    let verifiers = VerifierRegistry::with_builtins();
    let brief_mode_env = std::env::var("CHARLOTTE_BRIEF_MODE").ok();
    let brief_mode = BriefMode::parse_env(brief_mode_env.as_deref());

    println!(
        "Charlotte — web research. Type exit or quit to leave.\n\
         Environment: OPENAI_BASE_URL, OPENAI_MODEL (optional; default gpt-6-luna).\n\
         HTTP: HTTP_MAX_RETRIES, HTTP_RETRY_* , HTTP_QUOTA_PER_SECOND (optional).\n\
         Brief mode: CHARLOTTE_BRIEF_MODE=llm|heuristic|always|never (default llm).\n\
         Browser: CHARLOTTE_STORAGE_STATE_PATH, CHARLOTTE_SEARCH_URL_TEMPLATE, CHARLOTTE_WEB_SEARCH_ACTION_TIMEOUT_SECS, CHARLOTTE_ALLOW_HOST_SUFFIXES, CHARLOTTE_BLOCK_PRIVATE_IPS, CHARLOTTE_BROWSER_BACKEND, CHARLOTTE_CDP_ENDPOINT (optional).\n\
         Search: CHARLOTTE_SEARCH_BACKEND=searxng|browser, CHARLOTTE_SEARXNG_URL (default http://127.0.0.1:18088), CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS (default 20).\n\
         Notepad: notepad_* tools enabled; large tool output auto-offloads to stubs.\n\
         Parallel: scrape_many opens isolated tabs on the shared browser (default 3 at a time).\n\
         Active search backend: {search_backend}{search_detail}\n\
         Active browser backend: {backend_id}{}\n\
         Active model: {model}\n\
         Active reasoning: high (Responses API)\n",
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

        let (should_expand, gate_source, brief_shape) = should_expand_brief_auto(
            brief_mode,
            http,
            hooks.as_ref(),
            guardrails.as_ref(),
            base_opts,
            line,
        )
        .await?;
        eprintln!(
            "brief_mode={} ({gate_source})",
            if should_expand { "expanded" } else { "direct" }
        );
        let mut volatile = Vec::new();
        if should_expand {
            let brief = expand_user_request_to_research_brief_shaped(
                http,
                hooks.as_ref(),
                guardrails.as_ref(),
                base_opts,
                &search_config,
                line,
                brief_shape,
            )
            .await?;
            eprintln!("\n--- Research brief ---\n{brief}\n");
            volatile.push(format!(
                "Research brief (execute with the registered browser tools until success criteria are met):\n\n{brief}"
            ));
        }
        print!("\nCharlotte: ");
        io::stdout().flush().ok();
        let outcome = run_turn(
            &completer,
            spec,
            &drivers,
            &workflows,
            &verifiers,
            &tools,
            base_opts,
            &transcript,
            line,
            &volatile,
            debug_chat,
        )
        .await?;
        println!();
        {
            let mut stored = outcome.transcript;
            let mut np = notepad.write().await;
            compact_tool_messages(&mut stored, &mut np, &offload_cfg);
            transcript = stored;
        }
        if outcome.reply.is_empty() {
            println!("(no text)\n");
        }
    }

    browser.close().await.map_err(|e| anyhow::anyhow!(e))?;
    Ok(())
}

fn render_piece(piece: StreamPiece, debug_chat: bool) {
    match piece {
        StreamPiece::Text(text) => {
            print!("{text}");
            io::stdout().flush().ok();
        }
        StreamPiece::Reasoning(text) if debug_chat => {
            eprint!("{text}");
        }
        StreamPiece::Reasoning(_) => {}
        StreamPiece::Tool(tool) => match tool.point {
            ToolPoint::Call => eprintln!(
                "[tool:call] {}({})",
                tool.name,
                preview(&tool.arguments, 180)
            ),
            ToolPoint::Result => eprintln!(
                "[tool:done] {} {}",
                tool.name,
                if tool.ok { "ok" } else { tool.error.as_str() }
            ),
            ToolPoint::Blocked => eprintln!("[tool:block] {} {}", tool.name, tool.error),
        },
    }
}

async fn run_turn(
    completer: &jig::SuperGlueCompleter,
    spec: &'static AgentSpec,
    drivers: &[Arc<dyn EngineDriver>],
    workflows: &WorkflowRegistry,
    verifiers: &VerifierRegistry,
    tools: &superglue::tools::ToolRegistry,
    options: &ChatOptions,
    transcript: &[ChatMessage],
    user_text: &str,
    volatile: &[String],
    debug_chat: bool,
) -> anyhow::Result<jig::TurnOutcome> {
    let thinking = options
        .reasoning_effort
        .clone()
        .unwrap_or_else(|| "none".to_string());
    let cancel = superglue::cancel::CancellationToken::new();
    AgentRunner::run(
        completer,
        TurnInput {
            spec,
            drivers,
            workflows,
            verifiers,
            tools,
            history: &[],
            transcript,
            user_text,
            volatile,
            cancel,
            single: true,
            unattended: false,
            server_rounds: 500,
            swarm_steps: 1,
            model: &options.model,
            thinking: &thinking,
            temperature: options.temperature,
        },
        |piece| render_piece(piece, debug_chat),
    )
    .await
    .map_err(|err| anyhow::anyhow!(err.to_string()))
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
    hooks: &Arc<HookRegistry>,
    guardrails: &Arc<GuardrailRegistry>,
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

    let spec = agents::babe::babe_agent_spec();
    let completer = agents::charlotte::prepare_completer(
        http.clone(),
        Arc::clone(hooks),
        Arc::clone(guardrails),
        base_opts,
    );
    let drivers = agents::babe::babe_drivers();
    let workflows = WorkflowRegistry::with_chat(20);
    let verifiers = VerifierRegistry::with_builtins();

    println!(
        "Babe — Qdrant. Type exit or quit to leave.\n\
         Qdrant: {qdrant_url} (optional QDRANT_API_KEY).\n"
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

        print!("\nBabe: ");
        io::stdout().flush().ok();
        let outcome = run_turn(
            &completer,
            spec,
            &drivers,
            &workflows,
            &verifiers,
            &tools,
            base_opts,
            &transcript,
            line,
            &[],
            false,
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
