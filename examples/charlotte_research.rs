//! Live demo: Charlotte agent with browser tools against an OpenAI-compatible API.
//!
//! Prerequisites:
//! - Chromium installed for the Playwright version bundled with `playwright-rs`
//!   (see README).
//! - `OPENAI_API_KEY` set.
//!
//! Single question (default prompt if no args):
//!
//! ```text
//! export OPENAI_API_KEY=sk-...
//! cargo run --example charlotte_research
//! cargo run --example charlotte_research -- "Your question here"
//! ```
//!
//! Interactive multi-turn chat (`exit` or `quit` to stop):
//!
//! ```text
//! cargo run --example charlotte_research -- --interactive
//! cargo run --example charlotte_research -- --interactive --headed
//! ```
//!
//! `--headed` opens a visible Chromium window (default is headless).
//!
//! Optional: `--browsers N` or `--browsers auto` (CPU-based default), `--stealth` for best-effort
//! automation mitigation. The agent still uses a single shared tab on the first browser instance.
//!
//! Brief expansion is routed by `CHARLOTTE_BRIEF_MODE` (default: `llm`; fallback to heuristic on gate errors).

use std::io::{self, Write};
use std::sync::Arc;

use charlotte::{
    AutomationProfile, BriefMode, BrowserConfig, BrowserHandle, BrowserPool, SharedNotepad,
    apply_browser_config_from_env, charlotte_agent_spec, charlotte_offload_config,
    compact_tool_messages, default_browser_count, expand_user_request_to_research_brief_shaped,
    http_client_from_env, judge_brief_gate, make_tool_llm_context, new_notepad,
    register_charlotte_notepad_offload, register_charlotte_with_notepad, search_config_from_env,
    should_expand_brief,
};
use superglue::agents::{AgentEngine, AgentSpec};
use superglue::chat::{ChatOptions, Conversation};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parsed = parse_args();

    let key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!(
                "Set OPENAI_API_KEY to run this example. Optional: OPENAI_BASE_URL, OPENAI_MODEL."
            );
            return Ok(());
        }
    };
    if parsed.interactive && !parsed.prompt_words.is_empty() {
        eprintln!("Note: extra arguments after --interactive are ignored in REPL mode.");
    }

    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());
    let model = std::env::var("OPENAI_MODEL")
        .unwrap_or_else(|_| "gpt-5.4-nano-2026-03-17-mini".to_string());

    let browser_config = apply_browser_config_from_env(BrowserConfig {
        headless: !parsed.headed,
        automation_profile: if parsed.stealth {
            AutomationProfile::Stealth
        } else {
            AutomationProfile::default()
        },
        ..Default::default()
    });
    let browser_count = parsed.browser_count.unwrap_or(1).max(1);
    eprintln!(
        "Browser: {} ({}{}).",
        if parsed.headed {
            "headed (visible window)"
        } else {
            "headless"
        },
        browser_count,
        if browser_count == 1 {
            " instance"
        } else {
            " instances"
        }
    );
    if parsed.stealth {
        eprintln!("Automation profile: stealth (best-effort).");
    }
    if browser_count > 1 {
        eprintln!(
            "Note: browser tools use the first instance only; extra browsers stay open for your own code or future tooling."
        );
    }

    let (browser, pool_for_close) = if browser_count == 1 {
        (
            BrowserHandle::from_config(browser_config.clone()).await?,
            None::<BrowserPool>,
        )
    } else {
        let pool = BrowserPool::from_config(browser_config.clone(), Some(browser_count)).await?;
        let first = pool.handles()[0].clone();
        (first, Some(pool))
    };
    let http = http_client_from_env()?;
    let tools = superglue::tools::ToolRegistry::new();
    let notepad = new_notepad();
    let hooks = Arc::new(HookRegistry::new());
    let guardrails = Arc::new(GuardrailRegistry::new());

    let spec = charlotte_agent_spec();
    let engine = AgentEngine::new(spec);

    let base_opts = ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from(key),
        model,
        max_tool_rounds: 20,
        temperature: Some(0.2),
        ..Default::default()
    };
    let opts = options_for_agent(&engine.spec, base_opts.clone());
    let brief_mode_env = std::env::var("CHARLOTTE_BRIEF_MODE").ok();
    let brief_mode = BriefMode::parse_env(brief_mode_env.as_deref());

    let llm_context = make_tool_llm_context(&http, &hooks, &guardrails, &base_opts);
    let search_config = search_config_from_env()?;
    register_charlotte_with_notepad(
        &tools,
        &browser,
        &browser_config,
        &search_config,
        &http,
        Some(llm_context),
        notepad.clone(),
    )
    .await?;
    let mut offload_cfg = charlotte_offload_config();
    offload_cfg.auto_offload = true;
    register_charlotte_notepad_offload(hooks.as_ref(), notepad.clone(), offload_cfg.clone()).await;

    if parsed.interactive {
        run_interactive(
            &http,
            &tools,
            hooks.as_ref(),
            guardrails.as_ref(),
            &opts,
            &base_opts,
            &search_config,
            brief_mode,
            notepad,
            offload_cfg,
        )
        .await?;
    } else {
        let prompt = if parsed.prompt_words.is_empty() {
            "What is the current stable Rust version? Use web search, cite URLs.".to_string()
        } else {
            parsed.prompt_words.join(" ")
        };
        let (should_expand, gate_source, brief_shape) = should_expand_brief_auto(
            brief_mode,
            &http,
            hooks.as_ref(),
            guardrails.as_ref(),
            &base_opts,
            &prompt,
        )
        .await;
        eprintln!(
            "brief_mode={} ({gate_source})",
            if should_expand { "expanded" } else { "direct" }
        );
        let user_prompt = if should_expand {
            let brief = expand_user_request_to_research_brief_shaped(
                &http,
                hooks.as_ref(),
                guardrails.as_ref(),
                &base_opts,
                &search_config,
                &prompt,
                brief_shape,
            )
            .await?;
            eprintln!("\n--- Research brief ---\n{brief}\n");
            format!(
                "**Original request:** {prompt}\n\n\
                 **Research brief (execute with the registered browser tools until success criteria are met):**\n\n\
                 {brief}",
                prompt = prompt,
                brief = brief
            )
        } else {
            prompt
        };
        run_single(&engine, &http, &tools, &user_prompt, &base_opts).await?;
    }

    if let Some(pool) = pool_for_close {
        pool.close_all().await?;
    } else {
        browser.close().await?;
    }

    Ok(())
}

struct ParsedArgs {
    interactive: bool,
    /// When true, launch Chromium with a visible window (`headless: false`).
    headed: bool,
    /// `None` → 1 browser; `Some(n)` → n instances (`--browsers auto` is resolved before this).
    browser_count: Option<usize>,
    stealth: bool,
    prompt_words: Vec<String>,
}

fn parse_args() -> ParsedArgs {
    let mut interactive = false;
    let mut headed = false;
    let mut browser_count: Option<usize> = None;
    let mut stealth = false;
    let mut rest = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-i" | "--interactive" => interactive = true,
            "--headed" => headed = true,
            "--stealth" => stealth = true,
            "--browsers" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("error: --browsers requires a value (number or \"auto\")");
                    std::process::exit(2);
                });
                if v.eq_ignore_ascii_case("auto") {
                    browser_count = Some(0);
                } else {
                    let n: usize = v.parse().unwrap_or_else(|_| {
                        eprintln!("error: --browsers must be a positive integer or \"auto\"");
                        std::process::exit(2);
                    });
                    browser_count = Some(n);
                }
            }
            "-h" | "--help" => {
                let auto = default_browser_count();
                println!(
                    "\
Usage:
  charlotte_research [OPTIONS] [PROMPT...]

Options:
  -i, --interactive   Multi-turn chat on stdin (type 'exit' or 'quit' to stop)
      --headed        Show the browser window (default is headless)
      --browsers N      Chromium instances: integer >= 1, or \"auto\" (currently {auto} on this machine)
      --stealth         Best-effort automation fingerprint mitigation (see charlotte docs)
  -h, --help          Show this help

Environment:
  OPENAI_API_KEY      Required
  OPENAI_BASE_URL     Optional (default https://api.openai.com)
  OPENAI_MODEL        Optional (default gpt-5.4-nano-2026-03-17-mini)
"
                );
                std::process::exit(0);
            }
            _ => rest.push(a),
        }
    }
    let browser_count = browser_count.map(|n| if n == 0 { default_browser_count() } else { n });
    ParsedArgs {
        interactive,
        headed,
        browser_count,
        stealth,
        prompt_words: rest,
    }
}

/// Mirror [`AgentEngine`]'s injection of system prompt, model override, and tool rounds.
fn options_for_agent(spec: &AgentSpec, mut base: ChatOptions) -> ChatOptions {
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
                    let source = if decision.via_jev { "jev" } else { "llm" };
                    (decision.expand, source, decision.shape)
                }
                Err(_) => (should_expand_brief(user_text), "llm->heuristic", None),
            }
        }
    }
}

async fn run_single(
    engine: &AgentEngine,
    http: &HttpClient,
    tools: &superglue::tools::ToolRegistry,
    prompt: &str,
    base_opts: &ChatOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let outcome = engine.run(http, tools, prompt, base_opts).await?;
    println!("rounds:  {}", outcome.rounds);
    println!("content: {:?}", outcome.content);
    println!("finish:  {:?}", outcome.finish_reason);
    Ok(())
}

async fn run_interactive(
    http: &HttpClient,
    tools: &superglue::tools::ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    opts: &ChatOptions,
    base_opts: &ChatOptions,
    search_config: &charlotte::SearchConfig,
    brief_mode: BriefMode,
    notepad: SharedNotepad,
    offload_cfg: charlotte::OffloadConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Charlotte interactive mode — brief routing via CHARLOTTE_BRIEF_MODE=llm|heuristic|always|never (default llm). Commands: exit, quit, help"
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
        if line.eq_ignore_ascii_case("help") {
            println!("Type a message to send. Use exit or quit to leave.\n");
            continue;
        }

        let (should_expand, gate_source, brief_shape) =
            should_expand_brief_auto(brief_mode, http, hooks, guardrails, base_opts, line).await;
        eprintln!(
            "brief_mode={} ({gate_source})",
            if should_expand { "expanded" } else { "direct" }
        );
        let user_message = if should_expand {
            let brief = expand_user_request_to_research_brief_shaped(
                http,
                hooks,
                guardrails,
                base_opts,
                search_config,
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
        let outcome = conv.complete(http, tools, hooks, guardrails, opts).await?;
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
    Ok(())
}
