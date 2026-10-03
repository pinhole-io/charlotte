//! Unified ingest: fetch or accept text → chunk → validate → dedupe.
//!
//! Order: **extract text → chunk → validate → deduplicate** (Spiderweb-style).
//! Embeddings and Qdrant upsert are left to the caller ([`crate::utils::qdrant_ingest`]).
//!
//! ## Pipeline
//!
//! 1. Resolve input to one or more [`SourcePage`] records (body text).
//! 2. Chunk each page with [`ChunkerConfig`](crate::ingest::chunk::ChunkerConfig).
//! 3. [`ValidationPipeline`](crate::ingest::validate::ValidationPipeline) per page.
//! 4. [`deduplicate_with_meta`](crate::ingest::dedup::deduplicate_with_meta) across the flattened stream.

pub(crate) mod chunk;
pub(crate) mod dedup;
pub(crate) mod extract;
pub(crate) mod markdown;
pub(crate) mod parse_document;
pub(crate) mod validate;

use std::sync::Arc;

use crate::browser::{BrowserHandle, NavigateOptions};
use crate::error::CharlotteError;
use crate::ingest::chunk::ChunkerConfig;
use crate::ingest::chunk::{ChunkDraft, Chunker};
use crate::ingest::dedup::{DedupConfig, deduplicate_with_meta};
use crate::ingest::validate::{ValidatedChunk, ValidationPipeline, ValidatorConfig};

/// Where text for ingestion comes from.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum IngestInput {
    /// Open `url` in the shared browser and read visible body text.
    SingleUrl { url: String },
    /// Skip the browser; useful for tests and future file readers.
    RawText { source_id: String, text: String },
}

/// One retained chunk with stable metadata for storage or citation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestChunk {
    pub id: usize,
    pub text: String,
    pub source_id: String,
    pub source_url: Option<String>,
}

/// Counters for observability and tuning.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestStats {
    pub pages_read: usize,
    pub chunks_produced: usize,
    pub chunks_rejected: usize,
    pub duplicates_removed: usize,
}

/// Successful ingest output.
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub chunks: Vec<IngestChunk>,
    pub stats: IngestStats,
    pub warnings: Vec<String>,
}

/// Pipeline configuration (chunking, validation, dedup, navigation).
#[derive(Clone, Default)]
pub struct IngestOptions {
    pub chunker: ChunkerConfig,
    pub validator: ValidatorConfig,
    pub dedup: DedupConfig,
    /// Override max characters when reading a URL (`None` = use the handle’s configured max).
    pub max_text_chars: Option<usize>,
    pub navigate: NavigateOptions,
    pub hooks: IngestHooks,
}

/// Optional callbacks (same spirit as [`crate::crawl::CrawlHooks`]).
#[derive(Clone, Default)]
pub struct IngestHooks {
    /// Called once per source page after text is available (before chunking).
    pub after_extract: Option<Arc<dyn Fn(&SourcePage) + Send + Sync>>,
}

/// One logical document (one URL or one raw blob).
#[derive(Debug, Clone)]
pub struct SourcePage {
    pub source_id: String,
    pub url: Option<String>,
    pub text: String,
}

/// Run the full pipeline.
///
/// - [`IngestInput::SingleUrl`] requires `handle: Some(&BrowserHandle)`.
/// - [`IngestInput::RawText`] ignores `handle`.
pub async fn ingest(
    handle: Option<&BrowserHandle>,
    input: IngestInput,
    options: IngestOptions,
) -> Result<IngestResult, CharlotteError> {
    let chunker = options.chunker;
    let validator = ValidationPipeline::new(options.validator.clone())?;
    let mut warnings = Vec::new();

    let pages = match input {
        IngestInput::SingleUrl { url } => {
            let h = handle.ok_or_else(|| {
                CharlotteError::msg("ingest: IngestInput::SingleUrl requires a BrowserHandle")
            })?;
            let max_chars = match options.max_text_chars {
                Some(n) => n,
                None => h.max_text_chars_limit().await?,
            };
            let nav = h
                .navigate_with_options(&url, options.navigate.clone())
                .await?;
            let final_url = nav
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or(&url)
                .to_string();

            let content = h.page_content_with_max_chars(None, max_chars).await?;
            let text = content
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            if content.get("truncated").and_then(|t| t.as_bool()) == Some(true) {
                warnings.push(format!(
                    "page text truncated to max_text_chars ({max_chars}) for {final_url}"
                ));
            }

            let page = SourcePage {
                source_id: final_url.clone(),
                url: Some(final_url),
                text,
            };
            if let Some(ref hook) = options.hooks.after_extract {
                hook(&page);
            }
            vec![page]
        }
        IngestInput::RawText { source_id, text } => {
            let page = SourcePage {
                source_id: source_id.clone(),
                url: None,
                text,
            };
            if let Some(ref hook) = options.hooks.after_extract {
                hook(&page);
            }
            vec![page]
        }
    };

    let mut stats = IngestStats {
        pages_read: pages.len(),
        ..Default::default()
    };

    let mut rejected = 0usize;
    let mut flat: Vec<(ChunkDraft, String, Option<String>)> = Vec::new();

    for page in pages {
        let drafts = chunker.chunk(&page.text);
        let validated = validator.validate_source(drafts);
        for v in validated {
            match v {
                ValidatedChunk::Accepted(d) => {
                    flat.push((d, page.source_id.clone(), page.url.clone()));
                }
                ValidatedChunk::Rejected { .. } => {
                    rejected += 1;
                }
            }
        }
    }

    stats.chunks_rejected = rejected;

    let (kept_meta, dup_removed) = deduplicate_with_meta(flat, &options.dedup);
    stats.duplicates_removed = dup_removed;

    let chunks: Vec<IngestChunk> = kept_meta
        .into_iter()
        .enumerate()
        .map(|(i, (draft, source_id, source_url))| IngestChunk {
            id: i,
            text: draft.text,
            source_id,
            source_url,
        })
        .collect();

    stats.chunks_produced = chunks.len();

    Ok(IngestResult {
        chunks,
        stats,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ingest_raw_text_pipeline() {
        let r = ingest(
            None,
            IngestInput::RawText {
                source_id: "doc1".to_string(),
                text: "Hello world.\n\nSecond paragraph here with more text.".to_string(),
            },
            IngestOptions {
                chunker: ChunkerConfig::Paragraph {
                    merge_under_chars: 0,
                },
                ..Default::default()
            },
        )
        .await
        .expect("ingest");
        assert!(!r.chunks.is_empty());
        assert_eq!(r.stats.pages_read, 1);
        assert_eq!(r.chunks[0].source_id, "doc1");
    }
}
