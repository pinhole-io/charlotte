//! Babe vector-store agent on the Harn engine.

use std::sync::Arc;

use jig::{
    AgentSpec, Budget, CoordinationSource, DoneCheck, EngineDriver, RoundBudget, StaticDriver,
    ToolPolicy, ToolSet,
};

const BABE_TOOLS: &[&str] = &["qdrant_search", "qdrant_upsert"];

const BABE_PROMPT: &str = "\
You are Babe, a precise assistant for Qdrant vector collections: you similarity-search with query vectors and upsert points with embeddings and JSON payloads when asked.

## Goals
- Answer using qdrant_search and qdrant_upsert as needed: search returns nearest neighbors with scores and payloads; upsert inserts or replaces points by id.
- Name the collection explicitly in every tool call; pass dense vectors as JSON arrays of numbers matching the collection dimension.

## Constraints
- Do not invent point ids, scores, or payload fields; only report tool output.
- Keep tool arguments valid JSON; prefer modest limits (e.g. limit ≤ 50) so responses stay readable.
";

pub static BABE: AgentSpec = AgentSpec {
    id: "babe",
    prompt: BABE_PROMPT,
    instructions: "",
    tools: ToolPolicy::fixed(ToolSet::Only(BABE_TOOLS)),
    max_tool_rounds: RoundBudget::Fixed(20),
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

pub fn babe_agent_spec() -> &'static AgentSpec {
    &BABE
}

pub fn babe_drivers() -> Vec<Arc<dyn EngineDriver>> {
    vec![Arc::new(StaticDriver {
        id: "tool_loop",
        chooser_line: "Search or upsert Qdrant points.",
        budget: Budget::tool_loop(),
        task: None,
        allowlist: BABE_TOOLS,
        prompt: "",
        rounds: Some(20),
        deny_tools: false,
    })]
}
