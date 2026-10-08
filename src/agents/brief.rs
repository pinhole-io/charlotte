//! Turn a short user request into a structured research brief (one LLM step, no browser tools).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::openai::ChatMessage;
use superglue::providers::{ProviderCredentials, ProviderId};
use superglue::systemone::{
    Choice, Questions, SystemOneRequest, choice_if_confident, clip_state, system_one,
};
use superglue::tools::ToolRegistry;

use crate::CharlotteError;
use crate::agents::context::utc_now_context_line;
use crate::search::{SearchBackendKind, SearchConfig};

const BRIEF_EXPANDER_SYSTEM: &str = r#"You plan tasks for a web-research agent with browser and search tools registered at runtime (web_search, navigate, page_markdown, list_links, notepad_*, site_crawl_chunks, agentic_crawl, etc.).

The user message includes:
- **Search stack** — active backend and search rules for THIS session (read at runtime; do not assume a different backend).
- **Process patterns** — task-shaped workflow hints; adapt them to the user task.
- **User task** — what to deliver.

Produce a substantive, execution-focused brief the agent will follow. Reflect what the user asked—do not add verification rules, source policies, or constraints they did not imply or ask for.

Required sections (use all that apply; **Process** is always required):
1. **Goal** — What to find or deliver, in the user's terms.
2. **Success criteria** — What "done" looks like, only if the user implied it.
3. **Sources** — Preferences or avoids, only if the user implied them.
4. **Process** — Numbered steps (typically 5–8) tailored to THIS task. Each step names concrete tools and order of operations. Must follow the Search stack (backend, site: rules, cooldown). When the task names a domain or company site, start with navigate → page_markdown / list_links on that site before open-web web_search. Plan 1–2 web_search calls per agent round, not parallel query spam. Include when to pivot if a step returns empty, blocked, or a notepad stub.
5. **Method** — One short paragraph summarizing the Process (optional if Process is already clear).
6. **Constraints** — Scope limits the user stated or clearly implied only.

Optional: a fenced `json` citations block if structured cites are needed.

Do not perform the research yourself and do not invent facts, URLs, or listings."#;

const BRIEF_GATE_SYSTEM: &str = r#"You are a router for a web-research assistant.
Decide whether the user message should be expanded into a research brief first.

Output exactly one token:
- BRIEF: if the request is complex (multi-part constraints, broad crawl/research, comparisons, schemas/structured output requirements, many entities, or explicit planning needed)
- DIRECT: if the request is simple/direct and can be answered immediately with normal tool use.

No punctuation. No explanation. Output only BRIEF or DIRECT."#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BriefMode {
    Llm,
    Heuristic,
    Always,
    Never,
}

const BRIEF_SYSTEM_ONE_TIMEOUT: Duration = Duration::from_millis(400);
const BRIEF_CHOICE_MIN_CONFIDENCE: f64 = 0.6;

/// Closed task shape for brief process hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BriefTaskShape {
    Crawl,
    Compare,
    Listing,
    OpenWeb,
}

impl BriefTaskShape {
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "crawl" => Some(Self::Crawl),
            "compare" => Some(Self::Compare),
            "listing" => Some(Self::Listing),
            "open-web" | "open_web" | "openweb" => Some(Self::OpenWeb),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Crawl => "crawl",
            Self::Compare => "compare",
            Self::Listing => "listing",
            Self::OpenWeb => "open-web",
        }
    }
}

/// Result of the brief gate, including an optional System One task shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BriefGateDecision {
    pub expand: bool,
    pub shape: Option<BriefTaskShape>,
    pub via_system_one: bool,
}

impl BriefMode {
    #[must_use]
    pub fn parse_env(raw: Option<&str>) -> Self {
        match raw
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
            .unwrap_or("llm")
        {
            "heuristic" => Self::Heuristic,
            "always" => Self::Always,
            "never" => Self::Never,
            _ => Self::Llm,
        }
    }
}

/// One URL (and optional title / quote) for structured citation extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub url: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub quote: Option<String>,
}

/// Operational search guidance for the brief expander, matching the active [`SearchConfig`].
#[must_use]
pub fn brief_search_context(config: &SearchConfig) -> String {
    let cooldown_secs = config.web_search_cooldown.as_secs();
    let backend_id = config.backend.as_str();
    match config.backend {
        SearchBackendKind::Searxng => format!(
            "### Active configuration (runtime)\n\
             - `CHARLOTTE_SEARCH_BACKEND`: {backend_id}\n\
             - SearXNG URL: {}\n\
             - General web engine: Bing (via SearXNG JSON API)\n\
             - `web_search` cooldown: ~{cooldown_secs}s between completed calls (serialized; parallel calls queue)\n\
             \n\
             ### web_search on this backend\n\
             - Use plain keywords (entity + topic). Example: `Apple official Twitter Instagram`.\n\
             - Do NOT plan `site:` queries — Charlotte rejects them or Bing returns empty.\n\
             - Plan 1–2 `web_search` calls per agent round, not 3–4 parallel variants.\n\
             - Empty results, `unresponsive_engines`, or `searxng_circuit_open`: stop retrying the same query; pivot to browser tools.\n\
             \n\
             ### On-site discovery (preferred when user names a domain)\n\
             - `navigate` → `page_markdown` (optional `focus`) and/or `list_links` on homepage, footer, contact, newsroom, about.\n\
             - Follow same-origin links that plausibly hold the answer before open-web search.\n\
             \n\
             ### Notepad\n\
             - Large tool output may return `notepad_ref` stubs — plan `notepad_read` when full content is needed; `notepad_store` for distilled facts.",
            config.searxng_base_url.trim_end_matches('/')
        ),
        SearchBackendKind::Browser => format!(
            "### Active configuration (runtime)\n\
             - `CHARLOTTE_SEARCH_BACKEND`: {backend_id}\n\
             - Search path: browser HTML scrape (DuckDuckGo by default via `CHARLOTTE_SEARCH_URL_TEMPLATE`)\n\
             - `web_search` cooldown: ~{cooldown_secs}s between completed calls (serialized)\n\
             \n\
             ### web_search on this backend\n\
             - `site:` allowed with **root domain only** (`site:example.com topic`); never `site:domain/path`.\n\
             - Higher CAPTCHA/block risk — prefer `navigate` when a URL is known; fewer searches.\n\
             - Plan 1–2 `web_search` calls per round. If blocked or empty, pivot to `navigate` / `page_markdown`.\n\
             \n\
             ### On-site discovery\n\
             - `navigate` → `page_markdown` / `list_links` when the user names a site or you have a canonical URL.\n\
             \n\
             ### Notepad\n\
             - Large tool output may return `notepad_ref` stubs — plan `notepad_read` when needed."
        ),
    }
}

/// Task-shaped process hints for the brief expander (adapt to the user query; not hard constraints).
#[must_use]
pub fn brief_task_process_hints(user_query: &str, config: &SearchConfig) -> String {
    brief_task_process_hints_shaped(user_query, config, None)
}

/// Process hints with an optional closed shape from System One.
#[must_use]
pub fn brief_task_process_hints_shaped(
    user_query: &str,
    config: &SearchConfig,
    shape: Option<BriefTaskShape>,
) -> String {
    let q = user_query.trim();
    let lower = q.to_ascii_lowercase();
    let mentions_domain = match shape {
        Some(BriefTaskShape::OpenWeb) => false,
        Some(_) => false,
        None => query_mentions_domain(&lower),
    };
    let is_comparison = match shape {
        Some(BriefTaskShape::Compare) => true,
        Some(_) => false,
        None => ["compare", "versus", " vs ", "across", "difference between"]
            .iter()
            .any(|m| lower.contains(m)),
    };
    let is_crawl = match shape {
        Some(BriefTaskShape::Crawl) => true,
        Some(_) => false,
        None => ["crawl", "documentation", "docs site", "site map", "sitemap"]
            .iter()
            .any(|m| lower.contains(m)),
    };
    let is_listing = match shape {
        Some(BriefTaskShape::Listing) => true,
        Some(BriefTaskShape::OpenWeb) => false,
        Some(_) => false,
        None => ["list ", "find all", "enumerate", "every ", "all official"]
            .iter()
            .any(|m| lower.contains(m)),
    };

    let search_note = match config.backend {
        SearchBackendKind::Searxng => "Use plain-keyword web_search only (no site:).",
        SearchBackendKind::Browser => "web_search may use site:example.com with root domain only.",
    };

    if is_crawl {
        return format!(
            "Task shape: **site crawl / docs coverage**.\n\
             Suggested Process skeleton:\n\
             1. Confirm seed URL(s) from the user task.\n\
             2. `map` or `site_crawl_chunks` for same-origin discovery; use `agentic_crawl` when the site is JS-heavy and needs click/scroll/type (pass a goal).\n\
             3. `navigate` + `page_markdown` on highest-value pages; use `focus` for the topic.\n\
             4. One focused `web_search` only if crawl misses a section ({search_note}).\n\
             5. `notepad_store` distilled findings; answer with cited URLs."
        );
    }

    if is_comparison {
        return format!(
            "Task shape: **comparison across entities**.\n\
             Suggested Process skeleton:\n\
             1. List entities from the user task.\n\
             2. Per entity: one `web_search` OR `navigate` to primary source ({search_note}).\n\
             3. `page_markdown` with `focus` on comparison-relevant facts.\n\
             4. `notepad_store` per entity before synthesizing.\n\
             5. Final answer structured per user format; cite URLs from tool output."
        );
    }

    if mentions_domain || is_listing {
        return format!(
            "Task shape: **on-site + targeted discovery** (domain or listing in query).\n\
             Suggested Process skeleton:\n\
             1. `navigate` to the site root or page the user named.\n\
             2. `page_markdown` with `focus` on footer, contact, about, newsroom, or section matching the ask.\n\
             3. `list_links` (same-origin) to find pages likely containing the answer.\n\
             4. `navigate` to 1–2 best same-origin candidates; `page_markdown` again.\n\
             5. If still thin: one `web_search` with plain keywords ({search_note}) — not parallel variants.\n\
             6. `notepad_read` any stubs; `notepad_store` verified facts; answer only from tool output with URLs."
        );
    }

    format!(
        "Task shape: **open-web research**.\n\
         Suggested Process skeleton:\n\
         1. One focused `web_search` with keywords from the user task ({search_note}).\n\
         2. If snippets are insufficient, `scrape_many` on the 1–3 most authoritative URLs from results (parallel); otherwise answer from snippets.\n\
         3. If gap remains: one different `web_search` query (do not repeat the first).\n\
         4. `notepad_store` key facts; final answer cites URLs from tool output."
    )
}

fn query_mentions_domain(lower: &str) -> bool {
    [
        ".com", ".org", ".net", ".io", ".gov", ".edu", ".co/", "www.",
    ]
    .iter()
    .any(|tld| lower.contains(tld))
}

fn brief_expander_user_message(
    search_config: &SearchConfig,
    user_query: &str,
    shape: Option<BriefTaskShape>,
) -> String {
    format!(
        "{}\n\n\
         ## Search stack (active for this session)\n\
         {}\n\n\
         ## Process patterns (adapt to user task)\n\
         {}\n\n\
         ## User task\n\
         {}",
        utc_now_context_line(),
        brief_search_context(search_config),
        brief_task_process_hints_shaped(user_query, search_config, shape),
        user_query.trim()
    )
}

/// Parse a ` ```json ` … ` ``` ` fenced block containing a JSON array of [`Citation`] (if present).
#[must_use]
pub fn parse_citations_fenced_json(text: &str) -> Option<Vec<Citation>> {
    let start = text.find("```json")?;
    let after = start + "```json".len();
    let rest = &text[after..];
    let end = rest.find("```")?;
    let body = rest[..end].trim();
    serde_json::from_str::<Vec<Citation>>(body).ok()
}

/// Heuristic gate for whether a user request should be expanded into a research brief.
///
/// Returns `false` for short, direct asks so Charlotte can execute immediately.
#[must_use]
pub fn should_expand_brief(user_text: &str) -> bool {
    let raw = user_text.trim();
    if raw.is_empty() {
        return false;
    }
    let lower = raw.to_ascii_lowercase();
    let word_count = lower.split_whitespace().count();

    let complex_markers = [
        "compare",
        "versus",
        " vs ",
        "across",
        "analyze",
        "investigate",
        "research",
        "crawl",
        "documentation",
        "docs",
        "at least",
        "timeframe",
        "last week",
        "last month",
        "before ",
        "after ",
        "sources",
        "citations",
        "json",
        "schema",
        "constraints",
        "success criteria",
    ];
    let marker_hits = complex_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .count();
    let has_structure = raw.contains('\n') || raw.contains("1.") || raw.contains("- ");
    let has_many_separators = [',', ';', ':']
        .iter()
        .map(|c| raw.matches(*c).count())
        .sum::<usize>()
        >= 3;

    if marker_hits >= 2 || has_structure {
        return true;
    }
    if marker_hits >= 1 && (word_count > 16 || has_many_separators) {
        return true;
    }
    word_count > 30
}

/// LLM or System One gate for whether a user request should be expanded into a research brief.
///
/// Returns `true` for BRIEF and `false` for DIRECT.
pub async fn should_expand_brief_llm(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_options: &ChatOptions,
    user_text: &str,
) -> Result<bool, CharlotteError> {
    Ok(
        judge_brief_gate(http, hooks, guardrails, base_options, user_text)
            .await?
            .expand,
    )
}

/// Brief gate with optional System One task shape. TypeSafe Choice when credentials exist.
pub async fn judge_brief_gate(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_options: &ChatOptions,
    user_text: &str,
) -> Result<BriefGateDecision, CharlotteError> {
    if let Some(creds) = typesafe_credentials(base_options) {
        match judge_brief_system_one(http, &creds, user_text).await {
            Some(decision) => return Ok(decision),
            None => {
                return Ok(BriefGateDecision {
                    expand: should_expand_brief(user_text),
                    shape: None,
                    via_system_one: false,
                });
            }
        }
    }

    let registry = ToolRegistry::new();
    let mut opts = base_options.clone();
    opts.system_prompt = Some(BRIEF_GATE_SYSTEM.to_string());
    opts.max_tool_rounds = 1;

    let user_message = format!("{}\n\n{}", utc_now_context_line(), user_text.trim());
    let messages = vec![ChatMessage::text("user", user_message)];
    let outcome = complete_with_tools(http, &registry, hooks, guardrails, messages, &opts).await?;
    let token = outcome
        .content
        .unwrap_or_default()
        .trim()
        .to_ascii_uppercase();
    if token.contains("BRIEF") {
        return Ok(BriefGateDecision {
            expand: true,
            shape: None,
            via_system_one: false,
        });
    }
    if token.contains("DIRECT") {
        return Ok(BriefGateDecision {
            expand: false,
            shape: None,
            via_system_one: false,
        });
    }
    Err(CharlotteError::msg(format!(
        "brief gate returned unexpected value: {token}"
    )))
}

fn typesafe_credentials(options: &ChatOptions) -> Option<ProviderCredentials> {
    if let Some(creds) = options.provider_credentials.as_deref() {
        if creds.has_key(ProviderId::TypeSafe) {
            return Some(creds.clone());
        }
    }
    let env = ProviderCredentials::from_env();
    env.has_key(ProviderId::TypeSafe).then_some(env)
}

async fn judge_brief_system_one(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    user_text: &str,
) -> Option<BriefGateDecision> {
    let mut questions = Questions::new();
    questions.insert(
        "gate",
        Choice::new("Should this web-research request be expanded into a research brief first?")
            .option(
                "brief",
                "Complex: multi-part constraints, crawl, comparison, schema, or explicit planning.",
            )
            .option(
                "direct",
                "Simple. Answer with normal tool use. No brief needed.",
            ),
    );
    questions.insert(
        "shape",
        Choice::new("What is the task shape for the research process?")
            .option("crawl", "Site crawl or documentation coverage.")
            .option("compare", "Compare entities or options.")
            .option("listing", "On-site listing or find-all discovery.")
            .option("open-web", "Open-web research without a named site."),
    );
    let request = SystemOneRequest::new(clip_state(user_text, 0), questions)
        .with_model("typesafe:jev-latest");
    let work = system_one(http, credentials, request);
    let response = tokio::time::timeout(BRIEF_SYSTEM_ONE_TIMEOUT, work)
        .await
        .ok()?
        .ok()?;
    let gate = response.answer("gate")?;
    let choice = choice_if_confident(gate, BRIEF_CHOICE_MIN_CONFIDENCE)?;
    let expand = match choice {
        "brief" => true,
        "direct" => false,
        _ => return None,
    };
    let shape = response
        .answer("shape")
        .and_then(|answer| choice_if_confident(answer, BRIEF_CHOICE_MIN_CONFIDENCE))
        .and_then(BriefTaskShape::parse);
    Some(BriefGateDecision {
        expand,
        shape,
        via_system_one: true,
    })
}

/// Runs one chat completion (no tools) that rewrites `user_query` into a research brief.
///
/// `base_options` should be the same API settings you use for the main agent (`base_url`, `api_key`,
/// `model`, `temperature`, …). This call overrides `system_prompt` and uses an empty tool list.
pub async fn expand_user_request_to_research_brief(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_options: &ChatOptions,
    search_config: &SearchConfig,
    user_query: &str,
) -> Result<String, CharlotteError> {
    expand_user_request_to_research_brief_shaped(
        http,
        hooks,
        guardrails,
        base_options,
        search_config,
        user_query,
        None,
    )
    .await
}

/// Expand a user request, using an optional System One task shape for process hints.
pub async fn expand_user_request_to_research_brief_shaped(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    base_options: &ChatOptions,
    search_config: &SearchConfig,
    user_query: &str,
    shape: Option<BriefTaskShape>,
) -> Result<String, CharlotteError> {
    let registry = ToolRegistry::new();
    let mut opts = base_options.clone();
    opts.system_prompt = Some(BRIEF_EXPANDER_SYSTEM.to_string());
    opts.max_tool_rounds = 4;

    let user_message = brief_expander_user_message(search_config, user_query, shape);
    let messages = vec![ChatMessage::text("user", user_message)];
    let outcome = complete_with_tools(http, &registry, hooks, guardrails, messages, &opts).await?;

    let text = outcome
        .content
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or(CharlotteError::BriefEmpty)?;

    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchBackendKind;

    #[test]
    fn brief_search_context_searxng_includes_runtime_config() {
        let ctx = brief_search_context(&SearchConfig::default());
        assert!(ctx.contains("CHARLOTTE_SEARCH_BACKEND"));
        assert!(ctx.contains("searxng"));
        assert!(ctx.contains("SearXNG URL"));
        assert!(ctx.contains("site:"));
        assert!(ctx.contains("cooldown"));
    }

    #[test]
    fn brief_search_context_browser_allows_root_site_operator() {
        let ctx = brief_search_context(&SearchConfig::new(
            SearchBackendKind::Browser,
            "http://example.invalid",
            SearchConfig::default().web_search_cooldown,
        ));
        assert!(ctx.contains("browser HTML"));
        assert!(ctx.contains("site:example.com"));
        assert!(ctx.contains("never `site:domain/path`"));
    }

    #[test]
    fn brief_expander_user_message_includes_search_stack_and_process_patterns() {
        let msg = brief_expander_user_message(
            &SearchConfig::default(),
            "find apple.com social accounts",
            None,
        );
        assert!(msg.contains("## Search stack (active for this session)"));
        assert!(msg.contains("## Process patterns (adapt to user task)"));
        assert!(msg.contains("on-site"));
        assert!(msg.contains("## User task"));
        assert!(msg.contains("find apple.com social accounts"));
    }

    #[test]
    fn brief_task_process_hints_open_web_for_generic_query() {
        let hints = brief_task_process_hints(
            "what is the latest stable rust version",
            &SearchConfig::default(),
        );
        assert!(hints.contains("open-web"));
        assert!(hints.contains("scrape_many"));
    }

    #[test]
    fn brief_task_process_hints_comparison() {
        let hints = brief_task_process_hints(
            "compare postgres vs mysql for analytics",
            &SearchConfig::default(),
        );
        assert!(hints.contains("comparison"));
    }

    #[test]
    fn parses_citation_fence() {
        let t = r#"Intro text
```json
[{"url":"https://example.com/a","title":"Doc"}]
```
"#;
        let c = parse_citations_fenced_json(t).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].url, "https://example.com/a");
        assert_eq!(c[0].title.as_deref(), Some("Doc"));
    }

    #[test]
    fn brief_gate_skips_simple_direct_request() {
        assert!(!should_expand_brief("find a recipe for fish"));
        assert!(!should_expand_brief("what is the latest rust version"));
    }

    #[test]
    fn brief_gate_expands_complex_request() {
        assert!(should_expand_brief(
            "Compare three hosted vector DBs, include citations and JSON output schema."
        ));
        assert!(should_expand_brief(
            "Crawl this docs site and summarize key auth changes from last month."
        ));
    }

    #[test]
    fn brief_mode_parse_env_defaults_to_llm() {
        assert_eq!(BriefMode::parse_env(None), BriefMode::Llm);
        assert_eq!(BriefMode::parse_env(Some("llm")), BriefMode::Llm);
    }

    #[test]
    fn brief_mode_parse_env_supports_all_values() {
        assert_eq!(
            BriefMode::parse_env(Some("heuristic")),
            BriefMode::Heuristic
        );
        assert_eq!(BriefMode::parse_env(Some("always")), BriefMode::Always);
        assert_eq!(BriefMode::parse_env(Some("never")), BriefMode::Never);
    }
}
