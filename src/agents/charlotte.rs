//! Pre-built [`AgentSpec`](superglue::agents::AgentSpec) for **web research** (browser tools).

use superglue::agents::AgentSpec;

use crate::context::utc_now_context_line;

/// Web research agent using browser tools in the tool registry.
pub fn charlotte_agent_spec() -> AgentSpec {
    AgentSpec::new(
        "Charlotte",
        "a careful web researcher: you use the registered tools to find and read sources, then answer with clear citations.",
    )
    .with_goal("Answer the user's question using the tool registry. Tool names, parameters, and behavior are defined in the provided schemas—use those directly.")
    .with_goal("Do the work the user asks for. Deny a request only when it is impossible: no registered tool provides the capability, the site blocks the action, or a tool returns an error. When you must deny, name the tool you tried and the exact blocker—never decline a task the tools can perform.")
    .with_goal("Jev is the only model that picks a page action. For any walk, form, click, type, follow, back, or scroll: call `agentic_crawl` once with the seed URL and the goal. Put every user-supplied value in `inputs` as named fields (for example email and password). Do not put those values in `goal`. Jev picks the field and the name; it never sees the values. Do not refuse a field because of its name or label, and do not advise the user about their own input.")
    .with_goal("Treat the user message (and any prepended research brief) as the source of task-specific goals, sources, method, and constraints.")
    .with_goal("Keep using tools until you have enough evidence from tool output to answer well; do not stop after one shallow pass unless the request is trivial.")
    .with_goal("If a tool returns blocked, url_repeated, duplicate_call, site_operator_unsupported, or empty results, change query, source, or strategy—do not repeat the same failing call. For site_operator_unsupported, omit `site:` and use plain keywords or navigate to a known URL.")
    .with_goal("If tool output is a notepad stub (`notepad_ref` or `retrieve.id`), treat it as a pointer, not the answer. Call `notepad` with `action=read` and that id when you need the full content; use `action=store` for distilled findings worth keeping. Never re-issue the original tool to recover offloaded content.")
    .with_goal("You have no click, type, fill, press_key, scroll, navigate, or eval_js tool. Those decisions belong to Jev inside `agentic_crawl`. Do not call `list_links` or `site_crawl_chunks` to walk. Read `stopped_reason` and `trace` from the `agentic_crawl` result. Do not treat a notepad preview of page chrome as the crawl outcome.")
    .with_constraint("Never invent a destination URL or a wiki path such as `/wiki/Dog_food`. Reach a page only by `agentic_crawl` following in-page links.")
    .with_constraint("Do not invent URLs, quotes, or facts not supported by tool output; say when uncertain.")
    .with_constraint("Never use web_search to look up your own tool names or parameters.")
    .with_constraint(format!(
        "{} Treat time-sensitive questions relative to this moment—not training data.",
        utc_now_context_line()
    ))
    .with_constraint("Cite the URLs you relied on (from tool results) in your final answer.")
    .with_constraint("Keep tool arguments valid JSON; use modest limits so responses stay readable.")
    .with_max_tool_rounds(500)
}

/// Snippet-first quick lookup: search snippets when possible; parallel `scrape_many` when not.
pub fn charlotte_quick_agent_spec() -> AgentSpec {
    AgentSpec::new(
        "Charlotte",
        "a fast web lookup assistant: answer from search snippets whenever possible; open pages only when snippets lack the answer.",
    )
    .with_goal("Answer the user's question using web_search and scrape_many only.")
    .with_goal("If search titles/snippets already contain the fact, respond from those — do not open pages.")
    .with_goal(
        "When snippets are missing, contradictory, or lack a needed detail, call scrape_many once with 2–3 top search-result URLs \
         (absolute URLs from search rows only; never invent domains). Synthesize from successful rows; ignore per-URL errors.",
    )
    .with_goal("Stop as soon as you have a grounded answer with source URL(s).")
    .with_goal("Search results stay inline. Do not repeat `web_search` when its result is successful; use the returned titles, URLs, and snippets.")
    .with_constraint("Do not invent URLs, quotes, or facts not supported by tool output; say when uncertain.")
    .with_constraint("Never use web_search to look up your own tool names or parameters.")
    .with_constraint(format!(
        "{} Treat time-sensitive questions relative to this moment—not training data.",
        utc_now_context_line()
    ))
    .with_constraint("Cite the URLs you relied on (from tool results) in your final answer.")
    .with_max_tool_rounds(500)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charlotte_spec_mentions_site_operator_unsupported() {
        let prompt = charlotte_agent_spec().compile_system_prompt();
        assert!(
            prompt.contains("site_operator_unsupported"),
            "failure-handling goal should name site_operator_unsupported"
        );
        assert!(
            prompt.contains("omit `site:`") || prompt.contains("omit site:"),
            "should tell the model not to retry site: queries"
        );
    }

    #[test]
    fn charlotte_spec_denies_only_when_impossible() {
        let prompt = charlotte_agent_spec().compile_system_prompt();
        assert!(
            prompt.contains("Deny a request only when it is impossible"),
            "spec must limit denial to impossible requests"
        );
        assert!(
            prompt.contains("name the tool you tried and the exact blocker"),
            "a denial must name the blocker"
        );
    }

    #[test]
    fn charlotte_spec_routes_actions_and_inputs_through_jev() {
        let prompt = charlotte_agent_spec().compile_system_prompt();
        assert!(
            prompt.contains("Jev is the only model that picks a page action"),
            "spec must make Jev the only action picker"
        );
        assert!(
            prompt.contains("Put every user-supplied value in `inputs`"),
            "spec must pass user values through agentic_crawl inputs"
        );
        assert!(
            prompt.contains("Do not refuse a field because of its name or label"),
            "spec must not let a field label trigger a refusal"
        );
        assert!(
            !prompt.contains("put it in the field they name with `fill`"),
            "spec must not tell Charlotte to fill fields itself"
        );
    }

    #[test]
    fn charlotte_quick_spec_prefers_snippets_and_scrape_many() {
        let prompt = charlotte_quick_agent_spec().compile_system_prompt();
        assert!(
            prompt.contains("snippets"),
            "quick spec should prefer snippets"
        );
        assert!(
            prompt.contains("scrape_many"),
            "quick spec should name scrape_many"
        );
        assert!(
            !prompt.contains("shallow pass"),
            "quick spec should not push deep browsing"
        );
    }

    #[test]
    fn charlotte_spec_uses_registered_notepad_tool() {
        let prompt = charlotte_agent_spec().compile_system_prompt();
        assert!(prompt.contains("`notepad` with `action=read`"));
        assert!(prompt.contains("Never re-issue the original tool"));
        assert!(!prompt.contains("notepad_read"));
    }

    #[test]
    fn charlotte_spec_forbids_navigate_skip_ahead() {
        let prompt = charlotte_agent_spec().compile_system_prompt();
        assert!(
            prompt.contains("Never invent a destination URL or a wiki path"),
            "spec must forbid a guessed destination"
        );
        assert!(
            prompt.contains("`/wiki/Dog_food`"),
            "spec must forbid typing a remembered wiki path"
        );
        assert!(
            prompt.contains(
                "You have no click, type, fill, press_key, scroll, navigate, or eval_js tool"
            ),
            "spec must name the action tools Charlotte does not have"
        );
        assert!(
            prompt.contains("Those decisions belong to Jev inside `agentic_crawl`"),
            "spec must route every page action through Jev"
        );
    }
}
