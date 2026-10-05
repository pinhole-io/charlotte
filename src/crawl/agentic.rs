//! Jev-guided crawl on a [`PageActor`].
//!
//! Default actor is the headed [`crate::BrowserHandle`]. The uv sidecar is opt-in.
//! Jev picks keep / act / target / extract. Chat runs only when Jev asks.

mod handle_actor;
mod jev;
mod sidecar;
mod targets;

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::HttpClient;
use superglue::providers::{ProviderCredentials, ProviderId};

use crate::CharlotteError;
use crate::browser::BrowserHandle;
use crate::browser::url_policy::UrlPolicy;
use crate::crawl::robots::RobotsFetcher;
use crate::ingest::extract::{extract_structured_json, summarize_markdown};

pub use handle_actor::HandlePageActor;
pub use sidecar::{PageActor, ScriptedPageActor, UvSidecar};
pub use targets::{
    ObservePage, RawTarget, canonical_page_url, clip_chars, page_matches_goal, same_document,
    select_jev_targets,
};

use jev::{CrawlAct, PathHop, StepCtx, TriedHop, decide_step};
use targets::{filter_follow_candidates, follow_menu, group_link_sections, without_tried_follows};

/// Window scroll distance for Jev `scroll_down` / `scroll_up`.
const SCROLL_DELTA: i32 = 600;
/// Path hops kept in Jev state.
const PATH_KEEP: usize = 20;

/// Page body kept in the tool result. Long Wikipedia text must not bury the trace.
const PAGE_RESULT_CLIP: usize = 400;

/// Practical "no limit" for Jev steps and kept pages. The chat model must not lower this.
pub const AGENTIC_CRAWL_NO_LIMIT: usize = 9_999_999;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ActionStateKey {
    page_fingerprint: u64,
    act: &'static str,
    target_id: Option<String>,
    input_name: Option<String>,
}

/// One value the crawl may type into a field.
///
/// Jev chooses an input by `name`. The value never enters a prompt, a log, or
/// the tool result, so a password stays inside the process that types it.
pub struct CrawlInput {
    pub name: String,
    value: SecretString,
}

impl CrawlInput {
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: SecretString::from(value.into()),
        }
    }

    /// The text to type. Never log or serialize the result.
    #[must_use]
    pub fn value(&self) -> &str {
        self.value.expose_secret()
    }
}

impl Clone for CrawlInput {
    fn clone(&self) -> Self {
        Self::new(self.name.clone(), self.value.expose_secret())
    }
}

impl std::fmt::Debug for CrawlInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrawlInput")
            .field("name", &self.name)
            .field("value", &"[redacted]")
            .finish()
    }
}

/// Inputs for one `agentic_crawl` run.
#[derive(Debug, Clone)]
pub struct AgenticCrawlConfig {
    pub seed_url: String,
    pub goal: String,
    pub max_pages: usize,
    pub max_steps: usize,
    pub same_origin_only: bool,
    /// Single unnamed value. `inputs` covers a form with more than one field.
    pub input_text: Option<String>,
    pub inputs: Vec<CrawlInput>,
    pub extract_schema: Option<Value>,
    pub respect_robots: bool,
    pub url_policy: UrlPolicy,
    pub delay_between_steps: Duration,
    /// When false, ignore `TYPESAFE_API_KEY` from the process environment (tests).
    pub use_env_jev: bool,
}

impl AgenticCrawlConfig {
    #[must_use]
    pub fn new(seed_url: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            seed_url: seed_url.into(),
            goal: goal.into(),
            max_pages: AGENTIC_CRAWL_NO_LIMIT,
            max_steps: AGENTIC_CRAWL_NO_LIMIT,
            same_origin_only: false,
            input_text: None,
            inputs: Vec::new(),
            extract_schema: None,
            respect_robots: false,
            url_policy: UrlPolicy::permissive(),
            delay_between_steps: Duration::ZERO,
            use_env_jev: true,
        }
    }

    /// Names Jev may choose between when it types. Empty means one value or none.
    #[must_use]
    pub fn input_names(&self) -> Vec<String> {
        self.inputs.iter().map(|i| i.name.clone()).collect()
    }

    /// True when the run carries anything to type.
    #[must_use]
    pub fn can_type(&self) -> bool {
        !self.inputs.is_empty() || self.input_text.is_some()
    }

    /// The text for Jev's chosen input name, or the single unnamed value.
    fn input_value(&self, name: Option<&str>) -> Option<&str> {
        if let Some(name) = name
            && let Some(found) = self.inputs.iter().find(|i| i.name == name)
        {
            return Some(found.value());
        }
        if let [only] = self.inputs.as_slice() {
            return Some(only.value());
        }
        self.input_text.as_deref()
    }
}

/// Chat client used only when Jev sets `need_extract`.
#[derive(Clone)]
pub struct AgenticLlm {
    pub http: HttpClient,
    pub hooks: Arc<HookRegistry>,
    pub guardrails: Arc<GuardrailRegistry>,
    pub options: ChatOptions,
}

/// Run a bounded Jev-guided crawl on `actor`.
pub async fn agentic_crawl(
    actor: &mut dyn PageActor,
    config: &AgenticCrawlConfig,
    http: &HttpClient,
    llm: Option<&AgenticLlm>,
) -> Result<Value, CharlotteError> {
    config.url_policy.check_resolved(&config.seed_url).await?;
    let robots = if config.respect_robots {
        Some(RobotsFetcher::with_policy(config.url_policy.clone())?)
    } else {
        None
    };
    if let Some(r) = &robots
        && !r.is_allowed(&config.seed_url).await?
    {
        return Err(CharlotteError::msg("seed_url blocked by robots.txt"));
    }

    actor.start().await?;
    let mut page = actor.navigate(&config.seed_url).await?;
    if page.url.trim().is_empty() {
        page.url = config.seed_url.clone();
    }

    let creds = typesafe_credentials(llm.map(|l| &l.options), config.use_env_jev);
    let input_names = config.input_names();
    let mut visited = HashSet::new();
    visited.insert(page.url.clone());
    let mut link_inventories: HashMap<String, Vec<RawTarget>> = HashMap::new();
    let mut exclude_hrefs: HashSet<String> = HashSet::new();
    let mut path = vec![PathHop {
        url: canonical_page_url(&page.url),
        title: page.title.clone(),
    }];
    let mut tried: HashMap<String, Vec<TriedHop>> = HashMap::new();
    let mut visit_counts: HashMap<String, u32> = HashMap::new();
    visit_counts.insert(canonical_page_url(&page.url), 1);
    let mut scrolls_on: HashMap<String, u32> = HashMap::new();
    let mut seen_hrefs = target_href_set(&page.targets);
    let mut new_targets_since_scroll = 0u32;
    let mut attempted_actions = HashSet::new();

    let mut kept_pages = Vec::new();
    let mut trace = Vec::new();
    let mut pages_kept = 0usize;
    let mut last_act = None;
    let mut jev_steps = 0u32;
    let mut heuristic_steps = 0u32;
    let mut extracts = 0u32;
    let mut stopped_reason = "max_steps".to_string();

    for step in 1..=config.max_steps {
        if !config.delay_between_steps.is_zero() && step > 1 {
            tokio::time::sleep(config.delay_between_steps).await;
        }
        if creds.is_none() {
            if page_matches_goal(&page.url, &page.title, &config.goal) {
                tracing::info!(
                    target: "charlotte::agentic_crawl",
                    step,
                    "crawl goal reached"
                );
                if !already_kept(&kept_pages, &page.url) {
                    kept_pages.push(kept_page_row(&page));
                    pages_kept += 1;
                }
                trace.push(json!({
                    "step": step,
                    "url": page.url,
                    "act": "stop",
                    "target": Value::Null,
                    "keep": true,
                    "need_extract": false,
                    "via_jev": false,
                    "goal_reached": true,
                }));
                stopped_reason = "goal_reached".into();
                break;
            }
            return Err(CharlotteError::msg(
                "Jev credentials missing. Set TYPESAFE_API_KEY so agentic_crawl can pick actions.",
            ));
        }
        let page_url = if page.url.trim().is_empty() {
            config.seed_url.as_str()
        } else {
            page.url.as_str()
        };
        let page_key = canonical_page_url(page_url);
        let inventory_live = link_inventories.contains_key(&page_key);
        let raw_targets = link_inventories
            .get(&page_key)
            .map(Vec::as_slice)
            .unwrap_or(page.targets.as_slice());
        let observed = select_jev_targets(
            &page.targets,
            &config.goal,
            &config.url_policy,
            page_url,
            config.same_origin_only,
            &exclude_hrefs,
        );
        let empty_tried = Vec::new();
        let tried_here = tried.get(&page_key).unwrap_or(&empty_tried);
        let follow_candidates = filter_follow_candidates(
            raw_targets,
            &config.url_policy,
            page_url,
            config.same_origin_only,
            &exclude_hrefs,
        );
        let tried_hrefs: Vec<String> = tried_here.iter().map(|hop| hop.href.clone()).collect();
        let sections = group_link_sections(without_tried_follows(follow_candidates, &tried_hrefs));
        let follow_targets = follow_menu(&sections);
        let kept: Vec<PathHop> = kept_pages
            .iter()
            .filter_map(|row| {
                Some(PathHop {
                    url: row.get("url")?.as_str()?.to_string(),
                    title: row.get("title")?.as_str()?.to_string(),
                })
            })
            .collect();
        let opened: Vec<TriedHop> = tried.values().flatten().cloned().collect();
        let ctx = StepCtx {
            goal: &config.goal,
            page: &page,
            observed_targets: &observed,
            sections: &sections,
            follow_targets: &follow_targets,
            step,
            max_steps: config.max_steps,
            pages_kept,
            max_pages: config.max_pages,
            can_type: config.can_type(),
            input_names: &input_names,
            last_act,
            has_link_inventory: inventory_live,
            path: &path,
            tried_here,
            kept: &kept,
            opened: &opened,
            visits_here: visit_counts.get(&page_key).copied().unwrap_or(1),
            scrolls_here: scrolls_on.get(&page_key).copied().unwrap_or(0),
            new_targets_since_scroll,
        };
        let decision = decide_step(http, creds.as_ref(), &ctx).await?;
        if decision.via_jev {
            jev_steps += 1;
        } else {
            heuristic_steps += 1;
        }
        let source = if decision.via_jev { "jev" } else { "no_jev" };
        tracing::info!(
            target: "charlotte::agentic_crawl",
            step,
            source,
            keep = decision.keep,
            act = decision.act.as_str(),
            need_extract = decision.need_extract,
            "crawl step decided"
        );

        if decision.act != CrawlAct::Stop
            && !attempted_actions.insert(action_state_key(&page, &decision))
        {
            trace.push(json!({
                "step": step,
                "url": page.url,
                "act": decision.act.as_str(),
                "section": decision.section_key,
                "target": decision.target_id,
                "keep": decision.keep,
                "need_extract": decision.need_extract,
                "via_jev": decision.via_jev,
                "blocked": "repeated_action",
            }));
            stopped_reason = "repeated_action".into();
            break;
        }

        if decision.keep && !already_kept(&kept_pages, &page.url) {
            let mut row = kept_page_row(&page);
            if decision.need_extract
                && let Some(llm) = llm
            {
                match extract_or_summarize(
                    llm,
                    &page.markdown,
                    &config.goal,
                    config.extract_schema.as_ref(),
                )
                .await
                {
                    Ok(extracted) => {
                        row["extract"] = extracted;
                        extracts += 1;
                    }
                    Err(err) => {
                        row["extract_error"] = json!(err.to_string());
                    }
                }
            }
            kept_pages.push(row);
            pages_kept += 1;
        }

        trace.push(json!({
            "step": step,
            "url": page.url,
            "act": decision.act.as_str(),
            "section": decision.section_key,
            "target": decision.target_id,
            "keep": decision.keep,
            "need_extract": decision.need_extract,
            "via_jev": decision.via_jev,
        }));

        if pages_kept >= config.max_pages {
            stopped_reason = "max_pages".into();
            break;
        }
        if decision.act == CrawlAct::Stop {
            stopped_reason = if decision.via_jev {
                "jev_stop".into()
            } else {
                "jev_unavailable".into()
            };
            break;
        }

        let action_targets: &[RawTarget] = match decision.act {
            CrawlAct::Follow => &follow_targets,
            CrawlAct::Click | CrawlAct::Type => &observed,
            _ => &[],
        };
        let followed_href = if decision.act == CrawlAct::Follow {
            decision
                .target_id
                .as_deref()
                .and_then(|id| target_href(action_targets, id))
                .map(ToOwned::to_owned)
        } else {
            None
        };
        // `back` walks the crawl's own trail. Session history would step off the
        // crawl entirely, into the blank tab the browser started on.
        let back_target = if decision.act == CrawlAct::Back {
            path.iter().rev().nth(1).map(|hop| hop.url.clone())
        } else {
            None
        };
        let next = dispatch_act(
            actor,
            &decision,
            config,
            action_targets,
            &robots,
            page_url,
            back_target.as_deref(),
        )
        .await?;
        if let Some(mut next) = next {
            if !next.url.is_empty() && next.url != page.url {
                visited.insert(next.url.clone());
            }
            if decision.act == CrawlAct::GetLinks {
                link_inventories.insert(canonical_page_url(&next.url), next.targets.clone());
                let _ = count_new_hrefs(&mut seen_hrefs, &next.targets);
            }
            if matches!(decision.act, CrawlAct::ScrollDown | CrawlAct::ScrollUp) {
                *scrolls_on.entry(page_key.clone()).or_insert(0) += 1;
                new_targets_since_scroll = count_new_hrefs(&mut seen_hrefs, &next.targets);
            }
            if decision.act == CrawlAct::Follow {
                if let Some(href) = &followed_href {
                    tried.entry(page_key.clone()).or_default().push(TriedHop {
                        href: canonical_page_url(href),
                        landed_title: next.title.clone(),
                    });
                }
                if same_document(&page.url, &next.url) {
                    if let Some(href) = followed_href {
                        exclude_hrefs.insert(canonical_page_url(&href));
                    }
                } else {
                    exclude_hrefs.clear();
                    remember_new_document(
                        &mut path,
                        &mut visit_counts,
                        &mut seen_hrefs,
                        &mut new_targets_since_scroll,
                        &link_inventories,
                        &next,
                    );
                }
                if let Some(inv) = link_inventories.get(&canonical_page_url(&next.url)) {
                    next.targets = inv.clone();
                }
            }
            if matches!(decision.act, CrawlAct::Click | CrawlAct::Type)
                && !same_document(&page.url, &next.url)
            {
                exclude_hrefs.clear();
                remember_new_document(
                    &mut path,
                    &mut visit_counts,
                    &mut seen_hrefs,
                    &mut new_targets_since_scroll,
                    &link_inventories,
                    &next,
                );
            }
            if decision.act == CrawlAct::Back {
                exclude_hrefs.clear();
                walk_back_path(&mut path, &next);
                enter_document(
                    &mut visit_counts,
                    &mut seen_hrefs,
                    &mut new_targets_since_scroll,
                    &link_inventories,
                    &next,
                );
                if let Some(inv) = link_inventories.get(&canonical_page_url(&next.url)) {
                    next.targets = inv.clone();
                }
            }
            page = next;
            last_act = Some(decision.act);
        } else {
            stopped_reason = "dispatch_failed".into();
            break;
        }
    }

    let _ = actor.stop().await;
    Ok(json!({
        "ok": true,
        "pages": kept_pages,
        "trace": trace,
        "stopped_reason": stopped_reason,
        "via_jev": { "steps": jev_steps, "heuristic_steps": heuristic_steps, "extracts": extracts },
        "pages_kept": pages_kept,
        "visited_count": visited.len(),
    }))
}

fn target_href_set(targets: &[RawTarget]) -> HashSet<String> {
    targets
        .iter()
        .filter_map(|t| t.href.as_deref().map(canonical_page_url))
        .collect()
}

fn count_new_hrefs(seen: &mut HashSet<String>, targets: &[RawTarget]) -> u32 {
    let mut added = 0u32;
    for t in targets {
        if let Some(href) = t.href.as_deref()
            && seen.insert(canonical_page_url(href))
        {
            added += 1;
        }
    }
    added
}

fn action_state_key(page: &ObservePage, decision: &jev::StepDecision) -> ActionStateKey {
    let mut hasher = DefaultHasher::new();
    canonical_page_url(&page.url).hash(&mut hasher);
    page.title.hash(&mut hasher);
    page.markdown.hash(&mut hasher);
    page.scroll_y.hash(&mut hasher);
    page.viewport_h.hash(&mut hasher);
    page.scroll_height.hash(&mut hasher);
    page.at_page_top().hash(&mut hasher);
    page.at_page_bottom().hash(&mut hasher);
    for target in &page.targets {
        target.id.hash(&mut hasher);
        target.kind.hash(&mut hasher);
        target.label.hash(&mut hasher);
        target.href.hash(&mut hasher);
        target.region.hash(&mut hasher);
        target.section.hash(&mut hasher);
        target.in_viewport.hash(&mut hasher);
    }
    ActionStateKey {
        page_fingerprint: hasher.finish(),
        act: decision.act.as_str(),
        target_id: decision.target_id.clone(),
        input_name: decision.input_name.clone(),
    }
}

fn push_path(path: &mut Vec<PathHop>, url: String, title: String) {
    path.push(PathHop { url, title });
    if path.len() > PATH_KEEP {
        let drain = path.len() - PATH_KEEP;
        path.drain(..drain);
    }
}

fn remember_new_document(
    path: &mut Vec<PathHop>,
    visit_counts: &mut HashMap<String, u32>,
    seen_hrefs: &mut HashSet<String>,
    new_targets_since_scroll: &mut u32,
    link_inventories: &HashMap<String, Vec<RawTarget>>,
    next: &ObservePage,
) {
    push_path(path, canonical_page_url(&next.url), next.title.clone());
    enter_document(
        visit_counts,
        seen_hrefs,
        new_targets_since_scroll,
        link_inventories,
        next,
    );
}

/// `back` unwinds the trail: drop the page we left, then land on the one before it.
/// A redirect can land elsewhere, so push whatever the browser actually reached.
fn walk_back_path(path: &mut Vec<PathHop>, next: &ObservePage) {
    let next_key = canonical_page_url(&next.url);
    path.pop();
    if path.last().map(|hop| hop.url.as_str()) != Some(next_key.as_str()) {
        push_path(path, next_key, next.title.clone());
    }
}

/// Reset the per-document counters after the crawl reaches a new document.
fn enter_document(
    visit_counts: &mut HashMap<String, u32>,
    seen_hrefs: &mut HashSet<String>,
    new_targets_since_scroll: &mut u32,
    link_inventories: &HashMap<String, Vec<RawTarget>>,
    next: &ObservePage,
) {
    let next_key = canonical_page_url(&next.url);
    *visit_counts.entry(next_key.clone()).or_insert(0) += 1;
    *seen_hrefs = target_href_set(&next.targets);
    if let Some(inv) = link_inventories.get(&next_key) {
        seen_hrefs.extend(target_href_set(inv));
    }
    *new_targets_since_scroll = 0;
}

fn already_kept(kept: &[Value], url: &str) -> bool {
    let key = canonical_page_url(url);
    kept.iter().any(|row| {
        row.get("url")
            .and_then(Value::as_str)
            .is_some_and(|u| canonical_page_url(u) == key)
    })
}

fn kept_page_row(page: &ObservePage) -> Value {
    json!({
        "url": page.url,
        "title": page.title,
        "excerpt": clip_chars(&page.markdown, PAGE_RESULT_CLIP),
    })
}

async fn dispatch_act(
    actor: &mut dyn PageActor,
    decision: &jev::StepDecision,
    config: &AgenticCrawlConfig,
    targets: &[RawTarget],
    robots: &Option<RobotsFetcher>,
    page_url: &str,
    back_target: Option<&str>,
) -> Result<Option<ObservePage>, CharlotteError> {
    match decision.act {
        CrawlAct::Stop => Ok(None),
        CrawlAct::Click => {
            let id = decision
                .target_id
                .as_deref()
                .ok_or_else(|| CharlotteError::msg("click missing target"))?;
            ensure_known_target(targets, id)?;
            if let Some(href) = target_href(targets, id) {
                check_nav(&config.url_policy, robots, config, page_url, href).await?;
            }
            actor.click(id).await.map(Some)
        }
        CrawlAct::Follow => {
            let id = decision
                .target_id
                .as_deref()
                .ok_or_else(|| CharlotteError::msg("follow missing target"))?;
            ensure_known_target(targets, id)?;
            let href = target_href(targets, id)
                .ok_or_else(|| CharlotteError::msg("follow target has no href"))?;
            check_nav(&config.url_policy, robots, config, page_url, href).await?;
            actor.follow(id, Some(href)).await.map(Some)
        }
        CrawlAct::Back => {
            let url = back_target
                .ok_or_else(|| CharlotteError::msg("back has no previous page in the trail"))?;
            check_nav(&config.url_policy, robots, config, page_url, url).await?;
            actor.navigate(url).await.map(Some)
        }
        CrawlAct::GetLinks => actor.get_links().await.map(Some),
        CrawlAct::ScrollDown => actor.scroll(SCROLL_DELTA).await.map(Some),
        CrawlAct::ScrollUp => actor.scroll(-SCROLL_DELTA).await.map(Some),
        CrawlAct::Type => {
            let id = decision
                .target_id
                .as_deref()
                .ok_or_else(|| CharlotteError::msg("type missing target"))?;
            ensure_known_target(targets, id)?;
            let text = config
                .input_value(decision.input_name.as_deref())
                .ok_or_else(|| CharlotteError::msg("type has no value to enter"))?;
            actor.type_text(id, text).await.map(Some)
        }
    }
}

fn ensure_known_target(targets: &[RawTarget], id: &str) -> Result<(), CharlotteError> {
    if targets.iter().any(|t| t.id == id) {
        Ok(())
    } else {
        Err(CharlotteError::msg(format!(
            "rejected unknown target id: {id}"
        )))
    }
}

fn target_href<'a>(targets: &'a [RawTarget], id: &str) -> Option<&'a str> {
    targets
        .iter()
        .find(|t| t.id == id)
        .and_then(|t| t.href.as_deref())
}

async fn check_nav(
    policy: &UrlPolicy,
    robots: &Option<RobotsFetcher>,
    config: &AgenticCrawlConfig,
    page_url: &str,
    url: &str,
) -> Result<(), CharlotteError> {
    policy.check_resolved(url).await?;
    if config.same_origin_only && !targets::same_origin(page_url, url) {
        return Err(CharlotteError::msg("navigation left page origin"));
    }
    if let Some(r) = robots
        && !r.is_allowed(url).await?
    {
        return Err(CharlotteError::msg("url blocked by robots.txt"));
    }
    Ok(())
}

async fn extract_or_summarize(
    llm: &AgenticLlm,
    markdown: &str,
    goal: &str,
    schema: Option<&Value>,
) -> Result<Value, CharlotteError> {
    if schema.is_some() {
        extract_structured_json(
            &llm.http,
            &llm.hooks,
            &llm.guardrails,
            &llm.options,
            markdown,
            Some(goal),
            Some(goal),
            schema,
        )
        .await
    } else {
        summarize_markdown(
            &llm.http,
            &llm.hooks,
            &llm.guardrails,
            &llm.options,
            markdown,
            Some(goal),
            4,
        )
        .await
    }
}

fn typesafe_credentials(
    options: Option<&ChatOptions>,
    use_env: bool,
) -> Option<ProviderCredentials> {
    if let Some(opts) = options
        && let Some(creds) = opts.provider_credentials.as_deref()
        && creds.has_key(ProviderId::TypeSafe)
    {
        return Some(creds.clone());
    }
    if !use_env {
        return None;
    }
    let env = ProviderCredentials::from_env();
    env.has_key(ProviderId::TypeSafe).then_some(env)
}

/// Run [`agentic_crawl`] on the shared headed [`BrowserHandle`].
pub async fn agentic_crawl_with_handle(
    handle: Arc<BrowserHandle>,
    config: &AgenticCrawlConfig,
    http: &HttpClient,
    llm: Option<&AgenticLlm>,
) -> Result<Value, CharlotteError> {
    let mut actor = HandlePageActor::new(handle);
    agentic_crawl(&mut actor, config, http, llm).await
}

/// Spawn the Browser Use sidecar and run [`agentic_crawl`].
///
/// Opt-in only (`CHARLOTTE_AGENTIC_CRAWL_SIDECAR=1`). Default path is
/// [`agentic_crawl_with_handle`].
pub async fn agentic_crawl_with_sidecar(
    config: &AgenticCrawlConfig,
    http: &HttpClient,
    llm: Option<&AgenticLlm>,
) -> Result<Value, CharlotteError> {
    let mut sidecar = UvSidecar::spawn()?;
    let result = agentic_crawl(&mut sidecar, config, http, llm).await;
    let _ = sidecar.stop().await;
    result
}

/// True when the uv / Browser Use sidecar should run instead of [`HandlePageActor`].
#[must_use]
pub fn agentic_crawl_sidecar_enabled() -> bool {
    match std::env::var("CHARLOTTE_AGENTIC_CRAWL_SIDECAR") {
        Ok(v) => matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => false,
    }
}
