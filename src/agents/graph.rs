//! Fixed research graph.
//!
//! The runner writes a plan first. When that plan needs facts that are not in the task,
//! it gathers those facts and revises the plan. The investigation then searches, reads,
//! and writes. The model does not pick the next node.
//! A thin research read may return to search one time.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};
use superglue::agents::{AgentEngine, AgentSpec};
use superglue::cancel::CancellationToken;
use superglue::chat::ChatOptions;
use superglue::http::HttpClient;
use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

use crate::CharlotteError;
use crate::browser::{BrowserConfig, BrowserHandle};
use crate::config::{
    apply_browser_config_from_env, http_client_from_env, search_config_from_env,
    url_policy_from_env,
};
use crate::search::{SearchBackendKind, SearchConfig, web_search};
use crate::tools::research_read_tools;

const PLAN_ROUNDS: u32 = 1;
const DEFAULT_TOOL_ROUNDS: u32 = 250;
const WRITE_ROUNDS: u32 = 2;

const PLAN_INSTRUCTION: &str = "\
Write a research plan for the task.
Reply with JSON only: {\"needs_context\":false,\"plan\":\"...\"}.
Set needs_context to true when the plan needs facts that are not in the task.
Name those gaps in the plan.
When the notes above already contain those facts, set needs_context to false.
Do not call tools.";

/// One node in the research graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResearchNode {
    Plan,
    ContextSearch,
    ContextRead,
    Search,
    Read,
    Write,
}

impl ResearchNode {
    /// Sidebar label for this node.
    #[must_use]
    pub const fn phase(self) -> &'static str {
        match self {
            Self::Plan => "Planning",
            Self::ContextSearch => "Gathering",
            Self::ContextRead => "Context",
            Self::Search => "Searching",
            Self::Read => "Reading",
            Self::Write => "Writing",
        }
    }
}

/// Text from one node.
///
/// `thin` is set only by the research read node.
/// `needs_context` is set only by the plan node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeOutput {
    pub text: String,
    pub thin: bool,
    pub needs_context: bool,
}

/// Position in the fixed graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkState {
    pub node: ResearchNode,
    /// True after the context pass, or when the plan did not ask for one.
    pub context_done: bool,
    /// True after a thin research read has returned to search.
    pub read_retried: bool,
}

impl WalkState {
    /// The plan node, before any context pass.
    #[must_use]
    pub const fn start() -> Self {
        Self {
            node: ResearchNode::Plan,
            context_done: false,
            read_retried: false,
        }
    }
}

/// Chat settings from `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.
///
/// This path stays for Charlotte's own callers. The app passes
/// [`research_chat_options_from`] when Settings hold the values.
pub fn research_chat_options(
    cancel: Option<CancellationToken>,
) -> Result<ChatOptions, CharlotteError> {
    let key =
        std::env::var("OPENAI_API_KEY").map_err(|_| CharlotteError::msg("Set OPENAI_API_KEY."))?;
    let base_url = std::env::var("OPENAI_BASE_URL").unwrap_or_default();
    let model = std::env::var("OPENAI_MODEL").unwrap_or_default();
    research_chat_options_from(&key, &base_url, &model, 4, cancel)
}

/// Chat settings from explicit values. An empty host or model uses the default.
///
/// An empty key returns an error. `tool_rounds` below 1 becomes 250.
pub fn research_chat_options_from(
    api_key: &str,
    base_url: &str,
    model: &str,
    tool_rounds: u32,
    cancel: Option<CancellationToken>,
) -> Result<ChatOptions, CharlotteError> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(CharlotteError::msg("Set OPENAI_API_KEY."));
    }
    let base_url = filled(base_url, "https://api.openai.com");
    let model = filled(model, "gpt-5.6-luna");
    let tool_rounds = if tool_rounds < 1 {
        DEFAULT_TOOL_ROUNDS
    } else {
        tool_rounds
    };
    let reasoning_effort = model
        .to_ascii_lowercase()
        .contains("luna")
        .then(|| "none".to_string());
    Ok(ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from(api_key),
        model,
        reasoning_effort,
        max_tool_rounds: tool_rounds,
        temperature: Some(0.2),
        cancel,
        ..Default::default()
    })
}

fn filled(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_string()
    } else {
        value.to_string()
    }
}

/// Headless browser settings. Private hosts stay blocked.
#[must_use]
pub fn research_browser_config() -> BrowserConfig {
    let mut config = apply_browser_config_from_env(BrowserConfig::default());
    config.headless = true;
    let mut policy = url_policy_from_env();
    policy.block_private_ips = true;
    config.url_policy = policy;
    config
}

/// Read `{"thin":bool,"notes":"..."}` from a read node.
///
/// A bad payload is not thin. The notes are the raw text.
#[must_use]
pub fn parse_read_node(text: &str) -> (String, bool) {
    let Some(value) = read_json(text) else {
        return (text.trim().to_string(), false);
    };
    let Some(thin) = value.get("thin").and_then(Value::as_bool) else {
        return (text.trim().to_string(), false);
    };
    let notes = value
        .get("notes")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if notes.is_empty() {
        (text.trim().to_string(), thin)
    } else {
        (notes, thin)
    }
}

/// Read `{"needs_context":bool,"plan":"..."}` from a plan node.
///
/// A bad payload does not ask for context. The plan text is the raw text.
#[must_use]
pub fn parse_plan_node(text: &str) -> (String, bool) {
    let Some(value) = read_json(text) else {
        return (text.trim().to_string(), false);
    };
    let needs_context = value
        .get("needs_context")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let plan = value
        .get("plan")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if plan.is_empty() {
        (text.trim().to_string(), needs_context)
    } else {
        (plan, needs_context)
    }
}

fn read_json(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    serde_json::from_str(&trimmed[start..=end]).ok()
}

/// Next node.
///
/// `needs_context` applies only to the plan, and only before the context pass.
/// `thin` applies only to the research read. That read may return to search once.
#[must_use]
pub fn next_research_node(state: WalkState, needs_context: bool, thin: bool) -> Option<WalkState> {
    Some(match state.node {
        ResearchNode::Plan if needs_context && !state.context_done => WalkState {
            node: ResearchNode::ContextSearch,
            context_done: false,
            read_retried: state.read_retried,
        },
        ResearchNode::Plan => WalkState {
            node: ResearchNode::Search,
            context_done: true,
            read_retried: state.read_retried,
        },
        ResearchNode::ContextSearch => WalkState {
            node: ResearchNode::ContextRead,
            ..state
        },
        ResearchNode::ContextRead => WalkState {
            node: ResearchNode::Plan,
            context_done: true,
            read_retried: state.read_retried,
        },
        ResearchNode::Search => WalkState {
            node: ResearchNode::Read,
            ..state
        },
        ResearchNode::Read if thin && !state.read_retried => WalkState {
            node: ResearchNode::Search,
            context_done: state.context_done,
            read_retried: true,
        },
        ResearchNode::Read => WalkState {
            node: ResearchNode::Write,
            ..state
        },
        ResearchNode::Write => return None,
    })
}

/// Walk the graph with `step`. `step` does the work for one node.
pub async fn run_graph<F, Fut>(
    cancel: &CancellationToken,
    mut step: F,
) -> Result<String, CharlotteError>
where
    F: FnMut(ResearchNode, &str) -> Fut,
    Fut: Future<Output = Result<NodeOutput, CharlotteError>>,
{
    let mut state = WalkState::start();
    let mut prior = String::new();
    loop {
        if cancel.is_cancelled() {
            return Err(CharlotteError::msg("cancelled"));
        }
        let output = step(state.node, &prior).await?;
        prior = join_prior(&prior, state.node, &output.text);
        match next_research_node(state, output.needs_context, output.thin) {
            Some(next) => state = next,
            None => return Ok(output.text),
        }
    }
}

fn join_prior(prior: &str, node: ResearchNode, text: &str) -> String {
    let title = node.phase();
    if prior.is_empty() {
        format!("## {title}\n{text}")
    } else {
        format!("{prior}\n\n## {title}\n{text}")
    }
}

/// Ask the clarifying questions for a Research rune. No browser tools are registered.
pub async fn ask_clarifying_questions(
    query: &str,
    options: &ChatOptions,
) -> Result<String, CharlotteError> {
    let http = http_client_from_env().map_err(|err| CharlotteError::msg(err.to_string()))?;
    crate::agents::brief::clarifying_questions(
        &http,
        &superglue::hooks::HookRegistry::new(),
        &superglue::guardrails::GuardrailRegistry::new(),
        options,
        query,
    )
    .await
}

/// Run the live graph for one query and its clarifying answers.
///
/// `tool_rounds` caps search and read, including the context pass.
/// Plan stays at 1 round. Write stays at 2 rounds.
pub async fn run_research_graph(
    query: &str,
    answers: &[(String, String)],
    cancel: CancellationToken,
    options: ChatOptions,
    tool_rounds: u32,
    on_node: impl Fn(ResearchNode),
    on_plan: impl Fn(&str),
) -> Result<String, CharlotteError> {
    let http = http_client_from_env().map_err(|err| CharlotteError::msg(err.to_string()))?;
    let search = search_config_from_env()?;
    let browser_config = research_browser_config();
    let task = task_text(query, answers);
    let tool_rounds = if tool_rounds < 1 {
        DEFAULT_TOOL_ROUNDS
    } else {
        tool_rounds
    };
    let mut browser = None;
    let mut live = Live {
        cancel: &cancel,
        task: &task,
        http: &http,
        options: &options,
        tool_rounds,
        search: &search,
        browser_config: &browser_config,
        browser: &mut browser,
    };
    let result = walk_live(&mut live, &on_node, &on_plan).await;
    if let Some(browser) = browser {
        let _ = browser.close().await;
    }
    result
}

struct Live<'a> {
    cancel: &'a CancellationToken,
    task: &'a str,
    http: &'a HttpClient,
    options: &'a ChatOptions,
    tool_rounds: u32,
    search: &'a SearchConfig,
    browser_config: &'a BrowserConfig,
    browser: &'a mut Option<BrowserHandle>,
}

async fn walk_live(
    live: &mut Live<'_>,
    on_node: &impl Fn(ResearchNode),
    on_plan: &impl Fn(&str),
) -> Result<String, CharlotteError> {
    let mut state = WalkState::start();
    let mut prior = String::new();
    loop {
        if live.cancel.is_cancelled() {
            return Err(CharlotteError::msg("cancelled"));
        }
        on_node(state.node);
        let output = live_step(live, state.node, &prior).await?;
        if state.node == ResearchNode::Plan && !output.text.trim().is_empty() {
            on_plan(&output.text);
        }
        prior = join_prior(&prior, state.node, &output.text);
        match next_research_node(state, output.needs_context, output.thin) {
            Some(next) => state = next,
            None => return Ok(output.text),
        }
    }
}

fn task_text(query: &str, answers: &[(String, String)]) -> String {
    let mut text = query.trim().to_string();
    if answers.is_empty() {
        return text;
    }
    text.push_str("\n\nClarifying answers:\n");
    for (question, answer) in answers {
        text.push_str(&format!("- {question}\n  {answer}\n"));
    }
    text
}

async fn live_step(
    live: &mut Live<'_>,
    node: ResearchNode,
    prior: &str,
) -> Result<NodeOutput, CharlotteError> {
    let user = format!("{}\n\n{prior}", live.task);
    match node {
        ResearchNode::Plan => {
            let registry = ToolRegistry::new();
            let text = run_node(
                PLAN_INSTRUCTION,
                PLAN_ROUNDS,
                &registry,
                live.http,
                live.options,
                &user,
            )
            .await?;
            let (plan, needs_context) = parse_plan_node(&text);
            Ok(NodeOutput {
                text: plan,
                thin: false,
                needs_context,
            })
        }
        ResearchNode::ContextSearch => {
            search_node(
                live,
                "The plan lists missing facts. Search only for those facts. Call web_search. Report the titles, URLs, and snippets. Do not answer the research question.",
                &user,
            )
            .await
        }
        ResearchNode::ContextRead => {
            read_node(
                live,
                "Read the sources that fill the gaps in the plan. Summarize only those facts. Do not answer the research question.",
                &user,
                false,
            )
            .await
        }
        ResearchNode::Search => {
            search_node(
                live,
                "Follow the plan. Search the web for the investigation. Call web_search. Report the titles, URLs, and snippets you will use.",
                &user,
            )
            .await
        }
        ResearchNode::Read => {
            read_node(
                live,
                "Read the sources. Reply with JSON only: {\"thin\":false,\"notes\":\"...\"}. Set thin to true when the sources cannot support an answer.",
                &user,
                true,
            )
            .await
        }
        ResearchNode::Write => {
            let registry = ToolRegistry::new();
            let text = run_node(
                "Write the findings in Markdown. Cite URLs from the notes. Do not call tools.",
                WRITE_ROUNDS,
                &registry,
                live.http,
                live.options,
                &user,
            )
            .await?;
            Ok(NodeOutput {
                text,
                thin: false,
                needs_context: false,
            })
        }
    }
}

async fn search_node(
    live: &mut Live<'_>,
    instruction: &str,
    user: &str,
) -> Result<NodeOutput, CharlotteError> {
    let handle = ensure_browser(live.search, live.browser_config, live.browser).await?;
    let registry = ToolRegistry::new();
    registry
        .register(Arc::new(StopRepeat::new(Arc::new(SearchTool {
            http: live.http.clone(),
            search: live.search.clone(),
            browser: handle,
        }))))
        .await
        .map_err(|err| CharlotteError::msg(err.to_string()))?;
    let text = run_node(
        instruction,
        live.tool_rounds,
        &registry,
        live.http,
        live.options,
        user,
    )
    .await?;
    Ok(NodeOutput {
        text,
        thin: false,
        needs_context: false,
    })
}

async fn read_node(
    live: &mut Live<'_>,
    instruction: &str,
    user: &str,
    record_thin: bool,
) -> Result<NodeOutput, CharlotteError> {
    let handle = open_browser(live.browser_config, live.browser).await?;
    let registry = ToolRegistry::new();
    for tool in research_read_tools(&handle, live.browser_config, live.http) {
        registry
            .register(Arc::new(StopRepeat::new(tool)))
            .await
            .map_err(|err| CharlotteError::msg(err.to_string()))?;
    }
    let text = run_node(
        instruction,
        live.tool_rounds,
        &registry,
        live.http,
        live.options,
        user,
    )
    .await?;
    if record_thin {
        let (notes, thin) = parse_read_node(&text);
        Ok(NodeOutput {
            text: notes,
            thin,
            needs_context: false,
        })
    } else {
        Ok(NodeOutput {
            text,
            thin: false,
            needs_context: false,
        })
    }
}

async fn ensure_browser(
    search: &SearchConfig,
    config: &BrowserConfig,
    browser: &mut Option<BrowserHandle>,
) -> Result<Option<BrowserHandle>, CharlotteError> {
    if search.backend == SearchBackendKind::Searxng {
        return Ok(None);
    }
    open_browser(config, browser).await.map(Some)
}

async fn open_browser(
    config: &BrowserConfig,
    browser: &mut Option<BrowserHandle>,
) -> Result<BrowserHandle, CharlotteError> {
    if let Some(browser) = browser {
        return Ok(browser.clone());
    }
    let opened = BrowserHandle::from_config(config.clone()).await?;
    *browser = Some(opened.clone());
    Ok(opened)
}

async fn run_node(
    instruction: &str,
    rounds: u32,
    registry: &ToolRegistry,
    http: &HttpClient,
    options: &ChatOptions,
    user: &str,
) -> Result<String, CharlotteError> {
    let spec = AgentSpec::new("Charlotte", "a careful web researcher.")
        .with_instructions(instruction)
        .with_constraint("Do not invent URLs, quotes, or facts.")
        .with_max_tool_rounds(rounds);
    let outcome = AgentEngine::new(spec)
        .run(http, registry, user, options)
        .await?;
    outcome
        .content
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .ok_or(CharlotteError::BriefEmpty)
}

struct SearchTool {
    http: HttpClient,
    search: SearchConfig,
    browser: Option<BrowserHandle>,
}

#[async_trait]
impl Tool for SearchTool {
    fn spec(&self) -> ToolSpec {
        crate::tools::web_search_tool_spec()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolInvokeError::handler("missing string field: query", None))?;
        web_search(&self.http, self.browser.as_ref(), &self.search, query, 8)
            .await
            .map_err(|err| ToolInvokeError::handler(err.to_string(), None))
    }
}

struct StopRepeat {
    inner: Arc<dyn Tool>,
    fails: AtomicU32,
}

impl StopRepeat {
    fn new(inner: Arc<dyn Tool>) -> Self {
        Self {
            inner,
            fails: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Tool for StopRepeat {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        match self.inner.call(arguments).await {
            Ok(value) => {
                self.fails.store(0, Ordering::Relaxed);
                Ok(value)
            }
            Err(_) if self.fails.fetch_add(1, Ordering::Relaxed) == 0 => {
                Ok(json!({"ok": false, "error": "the tool failed"}))
            }
            Err(_) => Err(ToolInvokeError::handler(
                "the same tool failed twice",
                Some("repeat".to_string()),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_phase_is_the_node() {
        assert_eq!(ResearchNode::Plan.phase(), "Planning");
        assert_eq!(ResearchNode::ContextSearch.phase(), "Gathering");
        assert_eq!(ResearchNode::ContextRead.phase(), "Context");
        assert_eq!(ResearchNode::Search.phase(), "Searching");
        assert_eq!(ResearchNode::Read.phase(), "Reading");
        assert_eq!(ResearchNode::Write.phase(), "Writing");
    }

    #[test]
    fn a_plan_that_needs_context_gathers_before_research() {
        let plan = WalkState::start();
        let gather = next_research_node(plan, true, false).expect("gather");
        assert_eq!(gather.node, ResearchNode::ContextSearch);
        let read = next_research_node(gather, false, false).expect("context");
        assert_eq!(read.node, ResearchNode::ContextRead);
        let revised = next_research_node(read, false, false).expect("plan");
        assert_eq!(revised.node, ResearchNode::Plan);
        assert!(revised.context_done);
        let search = next_research_node(revised, true, false).expect("search");
        assert_eq!(search.node, ResearchNode::Search);
    }

    #[test]
    fn a_ready_plan_starts_the_research() {
        let search = next_research_node(WalkState::start(), false, false).expect("search");
        assert_eq!(search.node, ResearchNode::Search);
        assert!(search.context_done);
    }

    #[test]
    fn a_thin_read_returns_to_search_once() {
        let read = WalkState {
            node: ResearchNode::Read,
            context_done: true,
            read_retried: false,
        };
        let again = next_research_node(read, false, true).expect("search");
        assert_eq!(again.node, ResearchNode::Search);
        assert!(again.read_retried);
        let reread = WalkState {
            node: ResearchNode::Read,
            ..again
        };
        let write = next_research_node(reread, false, true).expect("write");
        assert_eq!(write.node, ResearchNode::Write);
    }

    #[test]
    fn plan_json_sets_needs_context_and_a_bad_payload_does_not() {
        let (plan, needs_context) =
            parse_plan_node(r#"{"needs_context":true,"plan":"name the port"}"#);
        assert!(needs_context);
        assert_eq!(plan, "name the port");
        let (plan, needs_context) = parse_plan_node("plain text");
        assert!(!needs_context);
        assert_eq!(plan, "plain text");
    }

    #[test]
    fn read_json_sets_thin_and_a_bad_payload_does_not() {
        let (notes, thin) = parse_read_node(r#"{"thin":true,"notes":"too few"}"#);
        assert!(thin);
        assert_eq!(notes, "too few");
        let (notes, thin) = parse_read_node("plain text");
        assert!(!thin);
        assert_eq!(notes, "plain text");
    }

    #[tokio::test]
    async fn scripted_graph_walks_plan_search_read_write() {
        let cancel = CancellationToken::new();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let text = run_graph(&cancel, move |node, _prior| {
            let log = Arc::clone(&log);
            async move {
                log.lock().expect("log").push(node);
                Ok(NodeOutput {
                    text: node.phase().to_string(),
                    thin: false,
                    needs_context: false,
                })
            }
        })
        .await
        .expect("graph");
        assert_eq!(
            *seen.lock().expect("log"),
            [
                ResearchNode::Plan,
                ResearchNode::Search,
                ResearchNode::Read,
                ResearchNode::Write,
            ]
        );
        assert_eq!(text, "Writing");
    }

    #[tokio::test]
    async fn a_plan_gathers_context_once_then_researches() {
        let cancel = CancellationToken::new();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let text = run_graph(&cancel, move |node, _prior| {
            let log = Arc::clone(&log);
            async move {
                log.lock().expect("log").push(node);
                Ok(NodeOutput {
                    text: node.phase().to_string(),
                    thin: false,
                    needs_context: node == ResearchNode::Plan,
                })
            }
        })
        .await
        .expect("graph");
        assert_eq!(
            *seen.lock().expect("log"),
            [
                ResearchNode::Plan,
                ResearchNode::ContextSearch,
                ResearchNode::ContextRead,
                ResearchNode::Plan,
                ResearchNode::Search,
                ResearchNode::Read,
                ResearchNode::Write,
            ]
        );
        assert_eq!(text, "Writing");
    }

    #[tokio::test]
    async fn a_thin_read_searches_once_more_then_writes() {
        let cancel = CancellationToken::new();
        let reads = Arc::new(AtomicU32::new(0));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let read_count = Arc::clone(&reads);
        let text = run_graph(&cancel, move |node, _prior| {
            let log = Arc::clone(&log);
            let read_count = Arc::clone(&read_count);
            async move {
                log.lock().expect("log").push(node);
                let thin = if node == ResearchNode::Read {
                    read_count.fetch_add(1, Ordering::Relaxed) == 0
                } else {
                    false
                };
                Ok(NodeOutput {
                    text: node.phase().to_string(),
                    thin,
                    needs_context: false,
                })
            }
        })
        .await
        .expect("graph");
        assert_eq!(
            *seen.lock().expect("log"),
            [
                ResearchNode::Plan,
                ResearchNode::Search,
                ResearchNode::Read,
                ResearchNode::Search,
                ResearchNode::Read,
                ResearchNode::Write,
            ]
        );
        assert_eq!(text, "Writing");
    }
}
