//! System One (System One) step decisions. Every crawl action comes from System One.

use std::time::Duration;

use serde_json::{Value, json};
use superglue::http::HttpClient;
use superglue::providers::ProviderCredentials;
use superglue::systemone::{Choice, Questions, SystemOneRequest, choice_if_confident, system_one};

use super::targets::{
    EXCERPT_CLIP, GOAL_CLIP, LinkSection, ObservePage, RawTarget, TITLE_CLIP, canonical_page_url,
    clip_chars,
};
use crate::CharlotteError;

const SYSTEM_ONE_TIMEOUT: Duration = Duration::from_secs(15);
const SYSTEM_ONE_RETRIES: u8 = 2;
const SYSTEM_ONE_MIN_CONFIDENCE: f64 = 0.4;
/// One Choice per request. 40 rich options stay inside System One's 32k token window.
const SYSTEM_ONE_BATCH_CAP: usize = 40;

/// Closed actions System One may pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrawlAct {
    Click,
    Type,
    Follow,
    Back,
    GetLinks,
    ScrollDown,
    ScrollUp,
    Stop,
}

impl CrawlAct {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Click => "click",
            Self::Type => "type",
            Self::Follow => "follow",
            Self::Back => "back",
            Self::GetLinks => "get_links",
            Self::ScrollDown => "scroll_down",
            Self::ScrollUp => "scroll_up",
            Self::Stop => "stop",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "click" => Some(Self::Click),
            "type" => Some(Self::Type),
            "follow" => Some(Self::Follow),
            "back" => Some(Self::Back),
            "get_links" => Some(Self::GetLinks),
            "scroll_down" => Some(Self::ScrollDown),
            "scroll_up" => Some(Self::ScrollUp),
            "stop" => Some(Self::Stop),
            _ => None,
        }
    }

    fn keeps_page(self) -> bool {
        !matches!(self, Self::GetLinks | Self::ScrollDown | Self::ScrollUp)
    }
}

/// One System One (or heuristic) decision for a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepDecision {
    pub keep: bool,
    pub act: CrawlAct,
    pub target_id: Option<String>,
    pub section_key: Option<String>,
    /// Which named input to type. `None` means the single unnamed value.
    pub input_name: Option<String>,
    pub need_extract: bool,
    pub via_system_one: bool,
}

/// One hop already walked, oldest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathHop {
    pub url: String,
    pub title: String,
}

/// A follow already tried from the current URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriedHop {
    pub href: String,
    pub landed_title: String,
}

/// Inputs for one System One/heuristic step (keeps call sites under the argument lint).
pub struct StepCtx<'a> {
    pub goal: &'a str,
    pub page: &'a ObservePage,
    pub observed_targets: &'a [RawTarget],
    pub sections: &'a [LinkSection],
    /// Remaining follow links, content first. System One pages these across requests.
    pub follow_targets: &'a [RawTarget],
    pub step: usize,
    pub max_steps: usize,
    pub pages_kept: usize,
    pub max_pages: usize,
    /// True when the run carries any value to type.
    pub can_type: bool,
    /// Names of the values available to type. Values never reach System One.
    pub input_names: &'a [String],
    pub last_act: Option<CrawlAct>,
    /// True when `get_links` already ran on this URL. Do not collect again.
    pub has_link_inventory: bool,
    pub path: &'a [PathHop],
    pub tried_here: &'a [TriedHop],
    /// Pages already kept. System One uses this to count results.
    pub kept: &'a [PathHop],
    /// Every follow already opened, from every page.
    pub opened: &'a [TriedHop],
    pub visits_here: u32,
    pub scrolls_here: u32,
    pub new_targets_since_scroll: u32,
}

/// Ask System One. `Err` on timeout, HTTP failure, or a missing act.
pub async fn judge_step(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    ctx: &StepCtx<'_>,
) -> Result<StepDecision, String> {
    let state = step_state(ctx);
    if ask_done(http, credentials, &state).await? {
        return Ok(StepDecision {
            keep: true,
            act: CrawlAct::Stop,
            target_id: None,
            section_key: None,
            input_name: None,
            need_extract: false,
            via_system_one: true,
        });
    }
    let act = ask_act(http, credentials, ctx, &state).await?;
    match act {
        CrawlAct::Follow => {
            let links = ask_sliced_targets(http, credentials, &state, ctx.follow_targets).await?;
            let target_id = ask_target(
                http,
                credentials,
                &state,
                CrawlAct::Follow,
                links,
                ctx.tried_here,
            )
            .await?;
            if let Some(href) = target_id
                .as_deref()
                .and_then(|id| ctx.follow_targets.iter().find(|t| t.id == id))
                .and_then(|t| t.href.as_deref())
                && tried_count(Some(href), ctx.tried_here) > 0
            {
                return Err(format!("system_one follow already tried from here: {href}"));
            }
            Ok(StepDecision {
                keep: true,
                act,
                target_id: target_id.clone(),
                section_key: target_id
                    .as_deref()
                    .and_then(|id| section_key_for(id, ctx.sections)),
                input_name: None,
                need_extract: false,
                via_system_one: true,
            })
        }
        CrawlAct::Click | CrawlAct::Type => {
            let controls =
                ask_sliced_targets(http, credentials, &state, ctx.observed_targets).await?;
            let target_id =
                ask_target(http, credentials, &state, act, controls, ctx.tried_here).await?;
            let input_name = if act == CrawlAct::Type {
                ask_input(http, credentials, ctx, &state).await?
            } else {
                None
            };
            Ok(StepDecision {
                keep: true,
                act,
                target_id,
                section_key: None,
                input_name,
                need_extract: false,
                via_system_one: true,
            })
        }
        other => Ok(StepDecision {
            keep: other.keeps_page(),
            act: other,
            target_id: None,
            section_key: None,
            input_name: None,
            need_extract: false,
            via_system_one: true,
        }),
    }
}

/// Ask which named value belongs in the field System One just chose.
///
/// System One sees names only. The values stay in the crawl config.
async fn ask_input(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    ctx: &StepCtx<'_>,
    state: &Value,
) -> Result<Option<String>, String> {
    if ctx.input_names.len() < 2 {
        return Ok(ctx.input_names.first().cloned());
    }
    let names = ask_sliced_names(http, credentials, state, ctx.input_names).await?;
    if names.len() == 1 {
        return Ok(Some(names[0].clone()));
    }
    let mut question = Choice::new(
        "Which value belongs in the field you chose? Match the value's name to the field's label.",
    );
    for name in names {
        question = question.option(name, format!("Type the value named {name}."));
    }
    let raw = ask_choice(http, credentials, state, "input", question).await?;
    names
        .iter()
        .find(|name| *name == &raw)
        .cloned()
        .map(Some)
        .ok_or_else(|| format!("system_one input not offered: {raw}"))
}

fn step_state(ctx: &StepCtx<'_>) -> Value {
    json!({
        "goal": clip_chars(ctx.goal, GOAL_CLIP),
        "url": ctx.page.url,
        "title": clip_chars(&ctx.page.title, TITLE_CLIP),
        "excerpt": clip_chars(&ctx.page.markdown, EXCERPT_CLIP),
        "step": ctx.step,
        "max_steps": ctx.max_steps,
        "pages_kept": ctx.pages_kept,
        "max_pages": ctx.max_pages,
        "can_type": ctx.can_type,
        "input_names": ctx.input_names,
        "last_act": ctx.last_act.map(CrawlAct::as_str),
        "has_link_inventory": ctx.has_link_inventory,
        "path": ctx.path.iter().map(|hop| json!({
            "url": hop.url,
            "title": hop.title,
        })).collect::<Vec<_>>(),
        "kept": ctx.kept.iter().take(SYSTEM_ONE_BATCH_CAP).map(|hop| json!({
            "url": hop.url,
            "title": clip_chars(&hop.title, TITLE_CLIP),
        })).collect::<Vec<_>>(),
        "opened": ctx.opened.iter().take(SYSTEM_ONE_BATCH_CAP).map(|hop| json!({
            "href": hop.href,
            "landed_title": clip_chars(&hop.landed_title, TITLE_CLIP),
        })).collect::<Vec<_>>(),
        "tried_here": ctx.tried_here.iter().map(|hop| json!({
            "href": hop.href,
            "landed_title": hop.landed_title,
        })).collect::<Vec<_>>(),
        "visits_here": ctx.visits_here,
        "scrolls_here": ctx.scrolls_here,
        "new_targets_since_scroll": ctx.new_targets_since_scroll,
        "at_top": ctx.page.at_page_top(),
        "at_bottom": ctx.page.at_page_bottom(),
        "previous_page": ctx.path.iter().rev().nth(1).map(|hop| json!({
            "url": hop.url,
            "title": hop.title,
        })),
        "follow_offered": ctx.follow_targets.len(),
        "follow_batches": follow_batch_count(ctx.follow_targets.len()),
        "sections": ctx.sections.iter().map(|s| json!({
            "id": s.key,
            "label": s.label,
            "link_count": s.links.len(),
            "already_opened": s.links.iter()
                .filter(|t| tried_count(t.href.as_deref(), ctx.tried_here) > 0)
                .count(),
        })).collect::<Vec<_>>(),
    })
}

async fn ask_done(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
) -> Result<bool, String> {
    let question = Choice::new(
        "Has the crawl already found what was asked? \
         Look at this page, `kept`, and `opened`. This page counts. \
         If the goal asked for several results, answer yes only when that many matching results are in hand. \
         The crawl finds pages. It does not write the final report. \
         If you still need another page, answer no.",
    )
    .option(
        "yes",
        "The crawl has the result, or enough results. Keep this page and stop.",
    )
    .option("no", "Need another page. Pick a next action.");
    let raw = ask_choice(http, credentials, state, "done", question).await?;
    parse_done(&raw)
}

fn parse_done(raw: &str) -> Result<bool, String> {
    match raw {
        "yes" => Ok(true),
        "no" => Ok(false),
        other => Err(format!("system_one done not offered: {other}")),
    }
}

async fn ask_act(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    ctx: &StepCtx<'_>,
    state: &Value,
) -> Result<CrawlAct, String> {
    let after_links = ctx.last_act == Some(CrawlAct::GetLinks);
    let mut act = Choice::new(act_instructions(ctx));
    let mut offered: Vec<&str> = Vec::new();
    let mut offer = |act: Choice, key: &'static str, help: &str| {
        offered.push(key);
        act.option(key, help)
    };
    if !ctx.follow_targets.is_empty() {
        act = offer(act, "follow", "Open one collected link on this page.");
    }
    act = offer(
        act,
        "stop",
        "End the crawl. This page already answers the goal.",
    );
    if ctx.path.len() > 1 {
        act = offer(
            act,
            "back",
            "Return to the previous page to try a different link.",
        );
    }
    if !ctx.observed_targets.is_empty() {
        act = offer(
            act,
            "click",
            "Operate a control: sort, filter, search button, or submit.",
        );
    }
    if !after_links && !ctx.has_link_inventory {
        act = offer(act, "get_links", "Read every link on this page.");
    }
    if ctx.can_type && !ctx.observed_targets.is_empty() {
        act = offer(
            act,
            "type",
            "Type one of the provided values into a field on this page.",
        );
    }
    // Scroll only serves the click path. `get_links` already returns off-screen
    // links, so scrolling cannot add a follow candidate once an inventory exists.
    if !ctx.has_link_inventory {
        if !ctx.page.at_page_bottom() {
            act = offer(
                act,
                "scroll_down",
                "Scroll down to see more controls on this page.",
            );
        }
        if !ctx.page.at_page_top() {
            act = offer(act, "scroll_up", "Scroll up to see earlier controls.");
        }
    }
    let raw = ask_choice(http, credentials, state, "act", act).await?;
    // System One output is untrusted. An act outside the menu has no valid dispatch.
    if !offered.contains(&raw.as_str()) {
        return Err(format!("system_one act not offered: {raw}"));
    }
    CrawlAct::parse(&raw).ok_or_else(|| format!("system_one act unknown: {raw}"))
}

/// System One types values it never sees, so the guide names the fields, not the text.
fn type_note(ctx: &StepCtx<'_>) -> String {
    if ctx.input_names.is_empty() {
        return String::new();
    }
    format!(
        "\nValues available to type, by name: {}. You choose the field and the name; \
         the text itself is hidden from you.",
        ctx.input_names.join(", ")
    )
}

fn follow_batch_count(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    len.div_ceil(SYSTEM_ONE_BATCH_CAP)
}

async fn ask_sliced_targets<'a>(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    targets: &'a [RawTarget],
) -> Result<&'a [RawTarget], String> {
    if targets.is_empty() {
        return Err("system_one needs targets, none available".into());
    }
    let mut cur = targets;
    while cur.len() > SYSTEM_ONE_BATCH_CAP {
        let batches: Vec<&[RawTarget]> = cur.chunks(SYSTEM_ONE_BATCH_CAP).collect();
        cur = ask_target_batch(http, credentials, state, &batches).await?;
    }
    Ok(cur)
}

async fn ask_target_batch<'a>(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    batches: &[&'a [RawTarget]],
) -> Result<&'a [RawTarget], String> {
    if batches.is_empty() {
        return Err("system_one batch needs slices, none available".into());
    }
    let window = if batches.len() > SYSTEM_ONE_BATCH_CAP {
        ask_batch_window(http, credentials, state, batches).await?
    } else {
        batches
    };
    if window.len() == 1 {
        return Ok(window[0]);
    }
    ask_which_target_batch(http, credentials, state, window).await
}

async fn ask_batch_window<'a, 'b>(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    batches: &'b [&'a [RawTarget]],
) -> Result<&'b [&'a [RawTarget]], String> {
    let window_count = follow_batch_count(batches.len());
    let mut question = Choice::new(
        "There are many slices. Each option is a group of slices. \
         Read the sample names. Pick the group that contains the target you need.",
    );
    for index in 0..window_count {
        let start = index * SYSTEM_ONE_BATCH_CAP;
        let end = (start + SYSTEM_ONE_BATCH_CAP).min(batches.len());
        let window = &batches[start..end];
        let first = window
            .first()
            .and_then(|b| b.first())
            .map(|t| t.label.as_str());
        let last = window
            .last()
            .and_then(|b| b.last())
            .map(|t| t.label.as_str());
        let span = match (first, last) {
            (Some(a), Some(b)) => format!(" · {a} … {b}"),
            _ => String::new(),
        };
        question = question.option(
            batch_option_key(index),
            clip_chars(
                &format!(
                    "group {}/{} · {} slices{span}",
                    index + 1,
                    window_count,
                    window.len()
                ),
                200,
            ),
        );
    }
    let raw = ask_choice(http, credentials, state, "batch_group", question).await?;
    let index = resolve_index(&raw).ok_or_else(|| format!("system_one batch group unknown: {raw}"))?;
    let start = index.saturating_mul(SYSTEM_ONE_BATCH_CAP);
    if start >= batches.len() {
        return Err(format!("system_one batch group unknown: {raw}"));
    }
    let end = (start + SYSTEM_ONE_BATCH_CAP).min(batches.len());
    Ok(&batches[start..end])
}

async fn ask_which_target_batch<'a>(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    batches: &[&'a [RawTarget]],
) -> Result<&'a [RawTarget], String> {
    let mut question = Choice::new(
        "This list does not fit one request. Each option is one slice. \
         Read the sample names. Pick the slice that contains the target you need. \
         Do not pick the first slice only because it is first. \
         On a sorted result list, the first slice is the top of the list.",
    );
    for (index, batch) in batches.iter().enumerate() {
        question = question.option(
            batch_option_key(index),
            batch_option_desc(index, batches, batch),
        );
    }
    let raw = ask_choice(http, credentials, state, "batch", question).await?;
    resolve_batch(&raw, batches).ok_or_else(|| format!("system_one batch unknown: {raw}"))
}

async fn ask_sliced_names<'a>(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    names: &'a [String],
) -> Result<&'a [String], String> {
    if names.is_empty() {
        return Err("system_one input needs names, none available".into());
    }
    let mut cur = names;
    while cur.len() > SYSTEM_ONE_BATCH_CAP {
        let batches: Vec<&[String]> = cur.chunks(SYSTEM_ONE_BATCH_CAP).collect();
        cur = ask_name_batch(http, credentials, state, &batches).await?;
    }
    Ok(cur)
}

async fn ask_name_batch<'a>(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    batches: &[&'a [String]],
) -> Result<&'a [String], String> {
    let mut question = Choice::new(
        "There are more named values than one request can hold. \
         Read the sample names. Pick the slice that contains the value you need.",
    );
    for (index, batch) in batches.iter().enumerate() {
        let samples = batch
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        question = question.option(
            batch_option_key(index),
            clip_chars(
                &format!(
                    "slice {}/{} · {} names · {samples}",
                    index + 1,
                    batches.len(),
                    batch.len()
                ),
                200,
            ),
        );
    }
    let raw = ask_choice(http, credentials, state, "input_batch", question).await?;
    resolve_name_batch(&raw, batches).ok_or_else(|| format!("system_one input batch unknown: {raw}"))
}

fn resolve_name_batch<'a>(raw: &str, batches: &[&'a [String]]) -> Option<&'a [String]> {
    resolve_index(raw).and_then(|i| batches.get(i).copied())
}

fn resolve_index(raw: &str) -> Option<usize> {
    let rest = raw.strip_prefix('b').or_else(|| raw.strip_prefix('p'))?;
    let n: usize = rest.parse().ok()?;
    n.checked_sub(1)
}

fn batch_option_key(index: usize) -> String {
    format!("b{:02}", index + 1)
}

fn batch_option_desc(index: usize, batches: &[&[RawTarget]], batch: &[RawTarget]) -> String {
    let samples: Vec<&str> = batch
        .iter()
        .map(|t| t.label.as_str())
        .filter(|s| !s.is_empty())
        .take(12)
        .collect();
    let sample = if samples.is_empty() {
        String::new()
    } else {
        format!(" · {}", samples.join(", "))
    };
    let nav = batch
        .iter()
        .filter(|t| t.region == "nav" || t.region == "other")
        .count();
    let kind = if nav == batch.len() {
        "nav"
    } else if nav == 0 {
        "body"
    } else {
        "mixed"
    };
    clip_chars(
        &format!(
            "slice {}/{} · {kind} · {} links{sample}",
            index + 1,
            batches.len(),
            batch.len()
        ),
        200,
    )
}

fn resolve_batch<'a>(raw: &str, batches: &[&'a [RawTarget]]) -> Option<&'a [RawTarget]> {
    resolve_index(raw).and_then(|i| batches.get(i).copied())
}

async fn ask_target(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    act: CrawlAct,
    targets: &[RawTarget],
    tried_here: &[TriedHop],
) -> Result<Option<String>, String> {
    if targets.is_empty() {
        return fallback_target(act, targets);
    }
    let offered = targets;
    let mut with_targets = state.clone();
    if let Some(obj) = with_targets.as_object_mut() {
        obj.insert(
            "targets".into(),
            Value::Array(
                offered
                    .iter()
                    .enumerate()
                    .map(|(rank, t)| {
                        json!({
                            "rank": rank + 1,
                            "id": t.id,
                            "kind": t.kind,
                            "label": t.label,
                            "href": t.href,
                            "region": t.region,
                            "section": t.section,
                            "in_viewport": t.in_viewport,
                            "opened_from_here": tried_count(t.href.as_deref(), tried_here),
                            "landed_on": tried_landing(t.href.as_deref(), tried_here),
                        })
                    })
                    .collect(),
            ),
        );
    }
    let mut question = Choice::new(
        "Which target should the action use? This list is the remaining page, content first. \
         Lower list numbers are higher on the page. Prefer content over nav. \
         On a results list, pick a top, in-view listing after sort is correct. \
         A target marked as already opened led to the page named; choose it again only if you have a reason.",
    );
    if act != CrawlAct::Follow {
        question = question.option("none", "No target. Use get_links, scroll, or stop.");
    }
    for (rank, t) in offered.iter().enumerate() {
        let loc = if t.region.is_empty() {
            t.kind.clone()
        } else {
            format!("{} {}", t.region, t.kind)
        };
        let desc = match &t.href {
            Some(href) => format!("{loc} {} {href}", t.label),
            None => format!("{loc} {}", t.label),
        };
        let place = if rank == 0 {
            " [top of this list]"
        } else if t.in_viewport {
            " [in view]"
        } else {
            ""
        };
        let note = match tried_landing(t.href.as_deref(), tried_here) {
            Some(title) => format!(
                " [opened {}x from here, led to {}]",
                tried_count(t.href.as_deref(), tried_here),
                clip_chars(&title, 60)
            ),
            None => String::new(),
        };
        question = question.option(
            rank_option_key(rank),
            format!("{}{place}{note}", clip_chars(&desc, 160)),
        );
    }
    let raw = ask_choice(http, credentials, &with_targets, "target", question).await?;
    resolve_ranked_target(act, Some(&raw), offered)
}

async fn ask_choice(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    state: &Value,
    id: &str,
    question: Choice,
) -> Result<String, String> {
    let mut questions = Questions::new();
    questions.insert(id, question);
    let request = SystemOneRequest::new(state.clone(), questions).with_model("typesafe:jev-latest");
    let work = system_one(http, credentials, request);
    let response = match tokio::time::timeout(SYSTEM_ONE_TIMEOUT, work).await {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => return Err(format!("system_one: {err}")),
        Err(_) => return Err("system_one timed out".into()),
    };
    response
        .answer(id)
        .and_then(|a| choice_if_confident(a, SYSTEM_ONE_MIN_CONFIDENCE).or_else(|| a.choice()))
        .map(str::to_string)
        .ok_or_else(|| format!("system_one {id} missing"))
}

/// Teach the browser, not a rule list. System One sees this on every act question.
fn act_instructions(ctx: &StepCtx<'_>) -> String {
    let mut guide = String::from(
        "You drive a real web browser to finish the goal. Pick the next action.\n\
         Actions:\n\
         - get_links reads every link on this page, on-screen and off-screen. Use it once per page.\n\
         - follow opens one collected link. Lists that do not fit one request are split: first pick a slice by its sample names, then pick an item in that slice.\n\
         - back returns to the previous page in `path` when this page is a dead end.\n\
         - click operates a control: sort, filter, tab, or submit. Long control lists use the same slices.\n\
         - type fills a field with a provided named value. Long name lists use the same slices.\n\
         - scroll_down and scroll_up reveal more controls. Links never need scrolling.\n\
         - stop ends the crawl.\n\
         Workflows:\n\
         - Link walk: follow a body link toward the topic. The main page grid is in content sections. Help, about, safety, and best-of links in the left rail are nav. Do not wander into help, talk, template, or file pages unless the goal names them.\n\
         - Search: type the query, click search or the search control, then read the results page.\n\
         - Result list (classifieds, search hits, tables): before you follow an item, look at the sort and filter controls in the excerpt and observed targets. \
         If Newest, Recent, or Date descending matches the goal, keep it. If the list is oldest-first or an irrelevant sort (random, distance, price) and the goal wants a current or typical item, click the sort control first. \
         After sort is right, follow an item from the top of the list (low list number, in view). Do not skip to a match buried at the bottom.\n\
         - Form: click or type the named field, then submit. One pass per field.\n\
         Evidence: `path` is the trail from the seed, oldest first. `tried_here` is what you already opened from this page.",
    );
    if ctx.visits_here > 1 {
        guide.push_str(&format!(
            "\nYou have been on this page {} times, and the links you opened from here did not reach the goal. \
             Go back, or open a link you have not tried.",
            ctx.visits_here
        ));
    }
    if repeats_in_path(ctx.path) {
        guide.push_str(
            "\nThe pages in `path` repeat, so you are circling. Break the circle: go back, or open a link you have not tried.",
        );
    }
    guide.push_str(&type_note(ctx));
    guide
}

/// True when any URL appears more than once in the walked path.
fn repeats_in_path(path: &[PathHop]) -> bool {
    let mut seen = std::collections::HashSet::new();
    path.iter().any(|hop| !seen.insert(hop.url.as_str()))
}

fn tried_count(href: Option<&str>, tried: &[TriedHop]) -> usize {
    let Some(href) = href else {
        return 0;
    };
    let key = canonical_page_url(href);
    tried.iter().filter(|hop| hop.href == key).count()
}

fn tried_landing(href: Option<&str>, tried: &[TriedHop]) -> Option<String> {
    let href = href?;
    let key = canonical_page_url(href);
    tried
        .iter()
        .rev()
        .find(|hop| hop.href == key)
        .map(|hop| hop.landed_title.clone())
}

fn rank_option_key(rank: usize) -> String {
    format!("t{:03}", rank + 1)
}

fn parse_rank_key(raw: &str) -> Option<usize> {
    let rest = raw.strip_prefix('t')?;
    if rest.len() != 3 || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: usize = rest.parse().ok()?;
    if n == 0 {
        return None;
    }
    Some(n - 1)
}

fn section_key_for(target_id: &str, sections: &[LinkSection]) -> Option<String> {
    sections
        .iter()
        .find(|s| s.links.iter().any(|t| t.id == target_id))
        .map(|s| s.key.clone())
}

fn resolve_ranked_target(
    act: CrawlAct,
    raw: Option<&str>,
    targets: &[RawTarget],
) -> Result<Option<String>, String> {
    let named = raw.filter(|id| *id != "none").map(str::trim);
    if let Some(id) = named {
        if let Some(idx) = parse_rank_key(id)
            && let Some(t) = targets.get(idx)
        {
            if act == CrawlAct::Follow && t.href.is_none() {
                return fallback_target(act, targets);
            }
            return Ok(Some(t.id.clone()));
        }
        if targets.iter().any(|t| t.id == id) {
            if act == CrawlAct::Follow
                && targets
                    .iter()
                    .find(|t| t.id == id)
                    .and_then(|t| t.href.as_ref())
                    .is_none()
            {
                return fallback_target(act, targets);
            }
            return Ok(Some(id.to_string()));
        }
    }
    if matches!(act, CrawlAct::Click | CrawlAct::Type | CrawlAct::Follow) {
        return fallback_target(act, targets);
    }
    Ok(None)
}

fn fallback_target(act: CrawlAct, targets: &[RawTarget]) -> Result<Option<String>, String> {
    let picked = match act {
        CrawlAct::Follow => targets
            .iter()
            .find(|t| t.kind == "link" && t.href.is_some()),
        CrawlAct::Click | CrawlAct::Type => targets.first(),
        _ => None,
    };
    match picked {
        Some(t) => Ok(Some(t.id.clone())),
        None => Err(format!(
            "system_one {act} needs a target, none available",
            act = act.as_str()
        )),
    }
}

fn system_one_unavailable_stop() -> StepDecision {
    StepDecision {
        keep: false,
        act: CrawlAct::Stop,
        target_id: None,
        section_key: None,
        input_name: None,
        need_extract: false,
        via_system_one: false,
    }
}

/// Ask System One for the step. Stop when credentials are missing or System One does not answer.
pub async fn decide_step(
    http: &HttpClient,
    credentials: Option<&ProviderCredentials>,
    ctx: &StepCtx<'_>,
) -> Result<StepDecision, CharlotteError> {
    let Some(creds) = credentials else {
        return Ok(system_one_unavailable_stop());
    };
    for _ in 1..=SYSTEM_ONE_RETRIES {
        match judge_step(http, creds, ctx).await {
            Ok(decision) => return Ok(decision),
            Err(_) => {}
        }
    }
    tracing::warn!(target: "charlotte::agentic_crawl", "System One unavailable");
    Ok(system_one_unavailable_stop())
}

#[cfg(test)]
mod tests {
    use super::*;
    use superglue::http::ClientConfig;

    fn sample_page() -> ObservePage {
        ObservePage {
            title: "Home".into(),
            url: "https://example.com/".into(),
            markdown: "home".into(),
            targets: vec![],
            ..Default::default()
        }
    }

    fn sample_ctx<'a>(
        page: &'a ObservePage,
        observed: &'a [RawTarget],
        sections: &'a [LinkSection],
    ) -> StepCtx<'a> {
        StepCtx {
            goal: "docs",
            page,
            observed_targets: observed,
            sections,
            follow_targets: observed,
            step: 1,
            max_steps: 8,
            pages_kept: 0,
            max_pages: 8,
            can_type: false,
            input_names: &[],
            last_act: None,
            has_link_inventory: false,
            path: &[],
            tried_here: &[],
            kept: &[],
            opened: &[],
            visits_here: 1,
            scrolls_here: 0,
            new_targets_since_scroll: 0,
        }
    }

    #[tokio::test]
    async fn decide_step_without_credentials_stops() {
        let http = HttpClient::new(ClientConfig::default()).expect("http");
        let page = sample_page();
        let targets = vec![RawTarget {
            id: "t1".into(),
            kind: "link".into(),
            label: "Next".into(),
            href: Some("https://example.com/n".into()),
            ..Default::default()
        }];
        let ctx = sample_ctx(&page, &targets, &[]);
        let d = decide_step(&http, None, &ctx).await.expect("decide");
        assert_eq!(d.act, CrawlAct::Stop);
        assert!(!d.via_system_one);
        assert!(!d.keep);
    }

    #[test]
    fn parse_done_accepts_only_yes_and_no() {
        assert_eq!(parse_done("yes").expect("yes"), true);
        assert_eq!(parse_done("no").expect("no"), false);
        assert!(parse_done("stop").is_err());
    }

    #[test]
    fn rank_option_keys_are_zero_padded() {
        assert_eq!(rank_option_key(0), "t001");
        assert_eq!(rank_option_key(9), "t010");
        assert_eq!(parse_rank_key("t001"), Some(0));
        assert_eq!(parse_rank_key("t010"), Some(9));
        assert_eq!(parse_rank_key("t1"), None);
    }

    #[test]
    fn resolve_ranked_target_maps_padded_key_to_dom_id() {
        let targets = vec![
            RawTarget {
                id: "t47".into(),
                kind: "link".into(),
                label: "Earth".into(),
                href: Some("https://en.wikipedia.org/wiki/Earth".into()),
                ..Default::default()
            },
            RawTarget {
                id: "t2".into(),
                kind: "link".into(),
                label: "Help".into(),
                href: Some("https://en.wikipedia.org/wiki/Help:Contents".into()),
                ..Default::default()
            },
        ];
        let id = resolve_ranked_target(CrawlAct::Follow, Some("t001"), &targets)
            .expect("resolve")
            .expect("id");
        assert_eq!(id, "t47");
        let id = resolve_ranked_target(CrawlAct::Follow, Some("t2"), &targets)
            .expect("resolve")
            .expect("id");
        assert_eq!(id, "t2");
    }

    fn test_ctx<'a>(page: &'a ObservePage, path: &'a [PathHop], visits_here: u32) -> StepCtx<'a> {
        StepCtx {
            goal: "reach the Dog page",
            page,
            observed_targets: &[],
            sections: &[],
            follow_targets: &[],
            step: 1,
            max_steps: 10,
            pages_kept: 0,
            max_pages: 10,
            can_type: false,
            input_names: &[],
            last_act: None,
            has_link_inventory: false,
            path,
            tried_here: &[],
            kept: &[],
            opened: &[],
            visits_here,
            scrolls_here: 0,
            new_targets_since_scroll: 0,
        }
    }

    fn hop(url: &str) -> PathHop {
        PathHop {
            url: url.into(),
            title: url.into(),
        }
    }

    #[test]
    fn act_instructions_teach_every_action_and_ban_nothing() {
        let page = ObservePage::default();
        let path = [hop("https://en.wikipedia.org/wiki/Sun")];
        let text = act_instructions(&test_ctx(&page, &path, 1));
        for action in [
            "get_links",
            "follow",
            "back",
            "click",
            "scroll_down",
            "stop",
        ] {
            assert!(text.contains(action), "guide omits {action}");
        }
        assert!(text.contains("Result list"));
        assert!(text.contains("sort"));
        assert!(text.contains("top of the list"));
        assert!(text.contains("slice"));
        assert!(text.contains("left rail"));
        assert!(!text.contains("Do not follow"));
        assert!(!text.contains("circling"));
    }

    #[test]
    fn act_instructions_report_a_repeated_path() {
        let page = ObservePage::default();
        let path = [
            hop("https://en.wikipedia.org/wiki/Main_Page"),
            hop("https://en.wikipedia.org/wiki/Wikipedia:Contents"),
            hop("https://en.wikipedia.org/wiki/Main_Page"),
        ];
        let text = act_instructions(&test_ctx(&page, &path, 2));
        assert!(text.contains("circling"));
        assert!(text.contains("been on this page 2 times"));
    }

    #[test]
    fn tried_labels_match_on_canonical_href() {
        let tried = [TriedHop {
            href: "https://en.wikipedia.org/wiki/Wikipedia:Contents".into(),
            landed_title: "Wikipedia:Contents - Wikipedia".into(),
        }];
        let href = Some("https://en.wikipedia.org/wiki/Wikipedia:Contents#Overviews");
        assert_eq!(tried_count(href, &tried), 1);
        assert_eq!(
            tried_landing(href, &tried).as_deref(),
            Some("Wikipedia:Contents - Wikipedia")
        );
        assert_eq!(
            tried_count(Some("https://en.wikipedia.org/wiki/Dog"), &tried),
            0
        );
        assert_eq!(tried_count(None, &tried), 0);
    }

    #[test]
    fn parse_scroll_and_get_links_acts() {
        assert_eq!(CrawlAct::parse("get_links"), Some(CrawlAct::GetLinks));
        assert_eq!(CrawlAct::GetLinks.as_str(), "get_links");
        assert_eq!(CrawlAct::parse("scroll_down"), Some(CrawlAct::ScrollDown));
        assert_eq!(CrawlAct::parse("scroll_up"), Some(CrawlAct::ScrollUp));
        assert_eq!(CrawlAct::ScrollDown.as_str(), "scroll_down");
    }

    #[test]
    fn follow_batches_split_for_the_32k_window() {
        let targets: Vec<RawTarget> = (1..=85)
            .map(|i| RawTarget {
                id: format!("t{i}"),
                kind: "link".into(),
                label: format!("Link {i}"),
                href: Some(format!("https://example.com/{i}")),
                ..Default::default()
            })
            .collect();
        let batches: Vec<&[RawTarget]> = targets.chunks(SYSTEM_ONE_BATCH_CAP).collect();
        assert_eq!(follow_batch_count(targets.len()), 3);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 40);
        assert_eq!(batches[1].len(), 40);
        assert_eq!(batches[2].len(), 5);
        assert!(batch_option_desc(1, &batches, batches[1]).contains("Link 41"));
        assert_eq!(
            resolve_batch("b02", &batches).map(|b| b[0].label.as_str()),
            Some("Link 41")
        );
        assert_eq!(
            resolve_batch("p2", &batches).map(|b| b[0].label.as_str()),
            Some("Link 41")
        );
        assert_eq!(follow_batch_count(300), 8);
        assert_eq!(follow_batch_count(1600), 40);
        assert_eq!(follow_batch_count(1601), 41);
    }

    #[test]
    fn name_batches_split_the_same_way() {
        let names: Vec<String> = (1..=85).map(|i| format!("field_{i}")).collect();
        let batches: Vec<&[String]> = names.chunks(SYSTEM_ONE_BATCH_CAP).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(
            resolve_name_batch("b02", &batches)
                .and_then(|b| b.first())
                .map(String::as_str),
            Some("field_41")
        );
    }

    #[test]
    fn section_key_for_finds_the_group_that_owns_the_target() {
        let sections = vec![
            LinkSection {
                key: "s01".into(),
                label: "content".into(),
                links: vec![RawTarget {
                    id: "t40".into(),
                    kind: "link".into(),
                    label: "computers".into(),
                    ..Default::default()
                }],
            },
            LinkSection {
                key: "s02".into(),
                label: "nav".into(),
                links: vec![RawTarget {
                    id: "t4".into(),
                    kind: "link".into(),
                    label: "help".into(),
                    ..Default::default()
                }],
            },
        ];
        assert_eq!(section_key_for("t40", &sections).as_deref(), Some("s01"));
        assert_eq!(section_key_for("t4", &sections).as_deref(), Some("s02"));
        assert_eq!(section_key_for("missing", &sections), None);
    }
}
