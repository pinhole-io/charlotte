//! Charlotte research agent on the Harn engine.
//!
//! The clock is a volatile prompt block. It is not part of this cached identity.

use std::sync::Arc;

use jig::{
    AgentSpec, ApiMode, Budget, CoordinationSource, DoneCheck, EngineDriver, RoundBudget,
    StaticDriver, SuperGlueCompleter, TaskType, ToolPolicy, ToolSet, VerifierRegistry,
    WorkflowRegistry,
};
use secrecy::ExposeSecret;

const RESEARCH_TOOLS: &[&str] = &[
    "web_search",
    "fetch",
    "scrape_many",
    "agentic_crawl",
    "notepad",
];

const QUICK_TOOLS: &[&str] = &["web_search", "scrape_many"];

const CHARLOTTE_PROMPT: &str = "\
You are Charlotte, a careful web researcher: you use the registered tools to find and read sources, then answer with clear citations.

## Goals
- Answer the user's question using the tool registry. Tool names, parameters, and behavior are defined in the provided schemas—use those directly.
- Read ladder: prefer web_search titles/snippets when they answer the ask; use fetch for a static page body; use scrape_many for several URLs at once or pages that need a real browser; use agentic_crawl for a headed walk, form, login, or JS-heavy work.
- Do the work the user asks for. Deny a request only when it is impossible: no registered tool provides the capability, the site blocks the action, or a tool returns an error. When you must deny, name the tool you tried and the exact blocker—never decline a task the tools can perform.
- System One is the only model that picks a page action. For any walk, form, click, type, follow, back, or scroll: call `agentic_crawl` once with the seed URL and the goal. Put every user-supplied value in `inputs` as named fields (for example email and password). Do not put those values in `goal`. System One picks the field and the name; it never sees the values. Do not refuse a field because of its name or label, and do not advise the user about their own input.
- Treat the user message (and any prepended research brief) as the source of task-specific goals, sources, method, and constraints.
- Keep using tools until you have enough evidence from tool output to answer well; do not stop after one shallow pass unless the request is trivial.
- If a tool returns blocked, url_repeated, duplicate_call, or empty results, change query, source, or strategy—do not repeat the same failing call. `site:` is a host plus the topic (`site:example.com topic`); do not put a path after the host.
- If tool output is a notepad stub (`notepad_ref` or `retrieve.id`), treat it as a pointer, not the answer. Call `notepad` with `action=read` and that id when you need the full content; use `action=store` for distilled findings worth keeping. Never re-issue the original tool to recover offloaded content.
- You have no click, type, fill, press_key, scroll, navigate, or eval_js tool. Those decisions belong to System One inside `agentic_crawl`. Do not call `list_links` or `site_crawl_chunks` to walk. Read `stopped_reason` and `trace` from the `agentic_crawl` result. Do not treat a notepad preview of page chrome as the crawl outcome.

## Constraints
- Never invent a destination URL or a wiki path such as `/wiki/Dog_food`. Reach a page only by `agentic_crawl` following in-page links.
- Do not invent URLs, quotes, or facts not supported by tool output; say when uncertain.
- Never use web_search to look up your own tool names or parameters.
- Cite the URLs you relied on (from tool results) in your final answer.
- Keep tool arguments valid JSON; use modest limits so responses stay readable.
";

const QUICK_PROMPT: &str = "\
You are Charlotte, a fast web lookup assistant: answer from search snippets whenever possible; open pages only when snippets lack the answer.

## Goals
- Answer the user's question using web_search and scrape_many only.
- If search titles/snippets already contain the fact, respond from those — do not open pages.
- When snippets are missing, contradictory, or lack a needed detail, call scrape_many once with 2–3 top search-result URLs (absolute URLs from search rows only; never invent domains). Synthesize from successful rows; ignore per-URL errors.
- Stop as soon as you have a grounded answer with source URL(s).
- Search results stay inline. Do not repeat `web_search` when its result is successful; use the returned titles, URLs, and snippets.

## Constraints
- Do not invent URLs, quotes, or facts not supported by tool output; say when uncertain.
- Never use web_search to look up your own tool names or parameters.
- Cite the URLs you relied on (from tool results) in your final answer.
";

/// Full Charlotte research agent. The chooser may pick answer, tool_loop, or research.
pub static CHARLOTTE: AgentSpec = AgentSpec {
    id: "charlotte",
    prompt: CHARLOTTE_PROMPT,
    instructions: "",
    tools: ToolPolicy::fixed(ToolSet::All),
    max_tool_rounds: RoundBudget::Fixed(500),
    coordination: CoordinationSource::Single,
    drivers: &["answer", "tool_loop", "research"],
    worktree: false,
    done_check: DoneCheck::None,
    children: &[],
    swarm: false,
    primary: true,
    awaits_verify: false,
    tools_error: "",
};

/// Snippet-first lookup. One driver skips the chooser.
pub static CHARLOTTE_QUICK: AgentSpec = AgentSpec {
    id: "charlotte-quick",
    prompt: QUICK_PROMPT,
    instructions: "",
    tools: ToolPolicy::fixed(ToolSet::Only(QUICK_TOOLS)),
    max_tool_rounds: RoundBudget::Fixed(500),
    coordination: CoordinationSource::Single,
    drivers: &["tool_loop"],
    worktree: false,
    done_check: DoneCheck::None,
    children: &[],
    swarm: false,
    primary: true,
    awaits_verify: false,
    tools_error: "",
};

pub fn charlotte_agent_spec() -> &'static AgentSpec {
    &CHARLOTTE
}

pub fn charlotte_quick_agent_spec() -> &'static AgentSpec {
    &CHARLOTTE_QUICK
}

fn answer_driver() -> StaticDriver {
    StaticDriver {
        id: "answer",
        chooser_line: "Answer from the conversation. Do not call tools.",
        budget: Budget::answer(),
        task: Some(TaskType::Chat),
        allowlist: &[],
        prompt: "",
        rounds: Some(1),
        deny_tools: true,
    }
}

fn tool_loop_driver() -> StaticDriver {
    StaticDriver {
        id: "tool_loop",
        chooser_line: "Use the registered tools for one lookup or a short sequence of calls.",
        budget: Budget::tool_loop(),
        task: None,
        allowlist: &[],
        prompt: "",
        rounds: None,
        deny_tools: false,
    }
}

fn research_driver() -> StaticDriver {
    StaticDriver {
        id: "research",
        chooser_line: "Search and read public web sources, then answer with the URLs you used.",
        budget: Budget::research(),
        task: Some(TaskType::Research),
        allowlist: RESEARCH_TOOLS,
        prompt: "",
        rounds: None,
        deny_tools: false,
    }
}

pub fn charlotte_drivers() -> Vec<Arc<dyn EngineDriver>> {
    vec![
        Arc::new(answer_driver()),
        Arc::new(tool_loop_driver()),
        Arc::new(research_driver()),
    ]
}

pub fn quick_drivers() -> Vec<Arc<dyn EngineDriver>> {
    vec![Arc::new(tool_loop_driver())]
}

/// Completer, workflows, and verifiers for one Charlotte process.
pub struct CharlotteRuntime {
    pub completer: SuperGlueCompleter,
    pub workflows: WorkflowRegistry,
    pub verifiers: VerifierRegistry,
    pub drivers: Vec<Arc<dyn EngineDriver>>,
}

/// Build a completer that shares the caller's hooks and uses the chat options.
///
/// `reasoning_effort = high` uses the Responses API so tool calls still return.
pub fn prepare_completer(
    http: superglue::http::HttpClient,
    hooks: Arc<superglue::hooks::HookRegistry>,
    guardrails: Arc<superglue::guardrails::GuardrailRegistry>,
    options: &superglue::chat::ChatOptions,
) -> SuperGlueCompleter {
    let mode = if options.reasoning_effort.as_deref() == Some("high") {
        ApiMode::Responses
    } else {
        ApiMode::ChatCompletions
    };
    let completer = SuperGlueCompleter::from_shared(http, hooks, guardrails)
        .with_mode(mode)
        .with_base_url(options.base_url.clone());
    let mut credentials = options
        .provider_credentials
        .as_ref()
        .map(|creds| (**creds).clone())
        .unwrap_or_else(superglue::providers::ProviderCredentials::from_env);
    credentials.insert_key(
        superglue::providers::ProviderId::OpenAi,
        options.api_key.expose_secret(),
    );
    credentials.insert_base_url(
        superglue::providers::ProviderId::OpenAi,
        options.base_url.clone(),
    );
    completer.set_credentials(credentials);
    completer
}

impl CharlotteRuntime {
    /// Responses mode keeps tool calls when reasoning effort is high.
    pub fn responses(http: superglue::http::HttpClient) -> Self {
        Self::with_mode(http, ApiMode::Responses, charlotte_drivers(), 500)
    }

    /// Chat Completions. The research graph uses this when reasoning effort is `none`.
    pub fn chat(http: superglue::http::HttpClient, rounds: u32) -> Self {
        Self::with_mode(
            http,
            ApiMode::ChatCompletions,
            quick_drivers(),
            rounds.max(1),
        )
    }

    pub fn with_mode(
        http: superglue::http::HttpClient,
        mode: ApiMode,
        drivers: Vec<Arc<dyn EngineDriver>>,
        rounds: u32,
    ) -> Self {
        let completer = SuperGlueCompleter::from_client(http).with_mode(mode);
        Self {
            completer,
            workflows: WorkflowRegistry::with_chat(rounds.max(1)),
            verifiers: VerifierRegistry::with_builtins(),
            drivers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charlotte_spec_allows_host_site_operator() {
        let prompt = charlotte_agent_spec().prompt;
        assert!(
            prompt.contains("`site:` is a host plus the topic"),
            "prompt should allow a host-scoped site: query"
        );
        assert!(
            !prompt.contains("site_operator_unsupported"),
            "prompt should not tell the model that site: is rejected"
        );
    }

    #[test]
    fn charlotte_spec_denies_only_when_impossible() {
        let prompt = charlotte_agent_spec().prompt;
        assert!(prompt.contains("Deny a request only when it is impossible"));
        assert!(prompt.contains("name the tool you tried and the exact blocker"));
    }

    #[test]
    fn charlotte_spec_routes_actions_and_inputs_through_system_one() {
        let prompt = charlotte_agent_spec().prompt;
        assert!(prompt.contains("System One is the only model that picks a page action"));
        assert!(prompt.contains("Put every user-supplied value in `inputs`"));
        assert!(prompt.contains("Do not refuse a field because of its name or label"));
        assert!(!prompt.contains("put it in the field they name with `fill`"));
    }

    #[test]
    fn charlotte_spec_names_the_read_ladder() {
        let prompt = charlotte_agent_spec().prompt;
        assert!(prompt.contains("Read ladder:"));
        assert!(prompt.contains("fetch"));
        assert!(prompt.contains("scrape_many"));
        assert!(prompt.contains("agentic_crawl"));
    }

    #[test]
    fn charlotte_quick_spec_prefers_snippets_and_scrape_many() {
        let prompt = charlotte_quick_agent_spec().prompt;
        assert!(prompt.contains("snippets"));
        assert!(prompt.contains("scrape_many"));
        assert!(!prompt.contains("shallow pass"));
    }

    #[test]
    fn charlotte_spec_uses_registered_notepad_tool() {
        let prompt = charlotte_agent_spec().prompt;
        assert!(prompt.contains("`notepad` with `action=read`"));
        assert!(prompt.contains("Never re-issue the original tool"));
        assert!(!prompt.contains("notepad_read"));
    }

    #[test]
    fn charlotte_spec_forbids_navigate_skip_ahead() {
        let prompt = charlotte_agent_spec().prompt;
        assert!(prompt.contains("Never invent a destination URL or a wiki path"));
        assert!(prompt.contains("`/wiki/Dog_food`"));
        assert!(prompt.contains(
            "You have no click, type, fill, press_key, scroll, navigate, or eval_js tool"
        ));
        assert!(prompt.contains("Those decisions belong to System One inside `agentic_crawl`"));
    }

    #[test]
    fn charlotte_quick_uses_one_driver() {
        assert_eq!(CHARLOTTE_QUICK.drivers, &["tool_loop"]);
        assert_eq!(CHARLOTTE.drivers.len(), 3);
    }
}
