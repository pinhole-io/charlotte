//! Charlotte — a web research agent built on [superglue] and Playwright ([playwright-rs]).
//!
//! Ingest pipeline (optional): [`ingest`] — extract text → chunk → validate → dedupe; then embed and upsert via [`utils::qdrant_ingest`].
//!
//! [superglue]: https://github.com/ (path dependency)
//! [playwright-rs]: https://crates.io/crates/playwright-rs

mod config;
mod error;

pub mod agents;
pub mod browser;
pub mod crawl;
pub mod ingest;
#[cfg(feature = "notepad")]
pub mod scratchpad;
pub mod search;
pub mod tools;
#[cfg(feature = "qdrant")]
pub mod utils;

pub use agents::brief;
pub use browser::pool;
pub use crawl::agentic as agentic_crawl;

pub use agentic_crawl::{
    AGENTIC_CRAWL_NO_LIMIT, AgenticCrawlConfig, AgenticLlm, CrawlInput, HandlePageActor,
    ObservePage, RawTarget, ScriptedPageActor, agentic_crawl, agentic_crawl_sidecar_enabled,
    agentic_crawl_with_handle, agentic_crawl_with_sidecar, page_matches_goal,
};
pub use agents::babe::babe_agent_spec;
pub use agents::charlotte::{charlotte_agent_spec, charlotte_quick_agent_spec};
pub use agents::graph::{
    NodeOutput, ResearchNode, WalkState, ask_clarifying_questions, next_research_node,
    parse_plan_node, parse_read_node, research_browser_config, research_chat_options,
    research_chat_options_from, run_graph, run_research_graph,
};
pub use brief::{
    BriefGateDecision, BriefMode, BriefTaskShape, Citation, brief_search_context,
    brief_task_process_hints, brief_task_process_hints_shaped, clarifying_questions,
    expand_user_request_to_research_brief, expand_user_request_to_research_brief_shaped,
    judge_brief_gate, parse_citations_fenced_json, parse_clarifying_questions, should_expand_brief,
    should_expand_brief_llm,
};
pub use browser::host::{BrowserHost, BrowserHostConfig};
pub use browser::url_policy::UrlPolicy;
pub use browser::{
    AutomationProfile, BrowserBackend, BrowserBackendKind, BrowserConfig, BrowserHandle,
    NavigateOptions, playwright_driver_available,
};
pub use config::{
    apply_browser_config_from_env, browser_backend_kind_from_env, cdp_endpoint_from_env,
    client_config_from_env, http_client_from_env, search_config_from_env, url_policy_from_env,
};
pub use crawl::scrape_many::{
    LazyBrowserPool, SCRAPE_MANY_DEFAULT_MAX_TOTAL_TEXT_CHARS, SCRAPE_MANY_MAX_URLS,
    ScrapeManyOptions, ScrapeRequest, apply_total_text_cap, clamp_concurrency, scrape_many,
    scrape_many_on, validate_urls,
};
pub use crawl::{
    CrawlHooks, CrawlRunConfig, SharedCrawlCache, SiteCrawlOutputLimits,
    apply_site_crawl_output_limits, deep_crawl, deep_crawl_parallel, normalize_crawl_url,
    normalize_crawl_url_ex, site_crawl_with_chunks, site_map,
};
pub use error::CharlotteError;
pub use ingest::chunk::{ChunkDraft, Chunker, ChunkerConfig};
pub use ingest::dedup::{
    DedupConfig, deduplicate_batch, deduplicate_with_meta, normalize_for_exact_dedup, simhash64,
};
pub use ingest::extract::{extract_structured_json, summarize_markdown};
pub use ingest::markdown::{
    AGENT_DEFAULT_MAX_HTML_BYTES, AGENT_DEFAULT_MAX_MARKDOWN_CHARS, AGENT_MAX_HTML_BYTES_CAP,
    AGENT_MAX_MARKDOWN_CHARS_CAP, body_html_to_markdown, clamp_agent_html_bytes,
    clamp_agent_markdown_chars, document_html_to_markdown, html_to_markdown_limited,
    split_markdown_sections, truncate_str_by_bytes,
};
#[cfg(feature = "documents")]
pub use ingest::parse_document::{
    DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT, DocumentFormat, parse_document_bytes, parse_document_url,
};
pub use ingest::validate::{ValidatedChunk, ValidationPipeline, ValidatorConfig};
pub use ingest::{
    IngestChunk, IngestHooks, IngestInput, IngestOptions, IngestResult, IngestStats, SourcePage,
    ingest,
};
pub use pool::{BrowserPool, DEFAULT_BROWSER_POOL_CAP, default_browser_count};
#[cfg(feature = "qdrant")]
pub use qdrant_client::Qdrant;
#[cfg(feature = "notepad")]
pub use scratchpad::{
    OffloadConfig, SharedNotepad, charlotte_offload_config, compact_tool_messages,
    make_tool_llm_context, new_notepad, register_charlotte_notepad_offload,
    register_charlotte_with_notepad,
};
pub use search::{SearchBackendKind, SearchConfig, web_search};
pub use tools::fetch_url;
pub use tools::{
    CharlotteTool, CharlotteToolLimits, CharlotteToolProfile, CharlotteToolSet,
    RegisteredCharlotteTool, ResolvedCredential, ToolLlmContext, ToolRuntimeContext,
    charlotte_tool_names, charlotte_tool_specs, register_charlotte_tools,
};
#[cfg(feature = "qdrant")]
pub use utils::qdrant::register_charlotte_qdrant_tools;
#[cfg(feature = "qdrant")]
pub use utils::qdrant_crud::{
    PointIdArg, QdrantCrudError, QueryNearestOpts, ScrollOpts, UpsertRow, count_points,
    delete_points_by_ids, get_points, query_nearest, scroll_points, upsert_points,
};
#[cfg(feature = "qdrant")]
pub use utils::qdrant_ingest::{ingest_chunk_payload, ingest_chunks_to_upsert_rows};
