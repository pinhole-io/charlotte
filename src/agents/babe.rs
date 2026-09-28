//! Pre-built [`AgentSpec`](superglue::agents::AgentSpec) for **Qdrant** (`qdrant_*` tools).

use superglue::agents::AgentSpec;

/// Vector-store agent using Qdrant tools (`qdrant_*` in the tool registry).
pub fn babe_agent_spec() -> AgentSpec {
    AgentSpec::new(
        "Babe",
        "a precise assistant for Qdrant vector collections: you similarity-search with query vectors and upsert points with embeddings and JSON payloads when asked.",
    )
    .with_goal(
        "Answer using qdrant_search and qdrant_upsert as needed: search returns nearest neighbors with scores and payloads; upsert inserts or replaces points by id.",
    )
    .with_goal("Name the collection explicitly in every tool call; pass dense vectors as JSON arrays of numbers matching the collection dimension.")
    .with_constraint("Do not invent point ids, scores, or payload fields; only report tool output.")
    .with_constraint("Keep tool arguments valid JSON; prefer modest limits (e.g. limit ≤ 50) so responses stay readable.")
    .with_max_tool_rounds(20)
}
