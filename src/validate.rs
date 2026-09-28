//! Deterministic chunk validation (length, emptiness, optional regex deny rules).

use regex::RegexSet;

use crate::chunk::ChunkDraft;

/// Rules applied to each chunk after chunking and before deduplication.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Minimum character count (Unicode scalar values).
    pub min_chars: usize,
    /// Maximum character count.
    pub max_chars: usize,
    /// If a chunk body matches any of these patterns, it is rejected.
    pub deny_patterns: Vec<String>,
    /// Maximum chunks to keep per source document (`None` = unlimited).
    pub max_chunks_per_source: Option<usize>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            min_chars: 1,
            max_chars: 32_000,
            deny_patterns: Vec::new(),
            max_chunks_per_source: None,
        }
    }
}

/// Outcome of validating one chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedChunk {
    Accepted(ChunkDraft),
    Rejected { reason: String },
}

pub struct ValidationPipeline {
    config: ValidatorConfig,
    deny: Option<RegexSet>,
}

impl ValidationPipeline {
    pub fn new(config: ValidatorConfig) -> Result<Self, regex::Error> {
        let deny = if config.deny_patterns.is_empty() {
            None
        } else {
            Some(RegexSet::new(&config.deny_patterns)?)
        };
        Ok(Self { config, deny })
    }

    /// Validate drafts from a single source; may truncate to `max_chunks_per_source`.
    pub fn validate_source(&self, drafts: Vec<ChunkDraft>) -> Vec<ValidatedChunk> {
        let limit = self.config.max_chunks_per_source;
        let take_n = limit.unwrap_or(drafts.len()).min(drafts.len());
        let iter = drafts.into_iter().take(take_n);

        let mut out = Vec::new();
        for d in iter {
            let count = d.text.chars().count();
            if d.text.trim().is_empty() {
                out.push(ValidatedChunk::Rejected {
                    reason: "whitespace_only".to_string(),
                });
                continue;
            }
            if count < self.config.min_chars {
                out.push(ValidatedChunk::Rejected {
                    reason: format!("too_short:{count}"),
                });
                continue;
            }
            if count > self.config.max_chars {
                out.push(ValidatedChunk::Rejected {
                    reason: format!("too_long:{count}"),
                });
                continue;
            }
            if let Some(ref set) = self.deny
                && set.is_match(&d.text)
            {
                out.push(ValidatedChunk::Rejected {
                    reason: "deny_pattern".to_string(),
                });
                continue;
            }
            out.push(ValidatedChunk::Accepted(d));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        let p = ValidationPipeline::new(ValidatorConfig::default()).unwrap();
        let r = p.validate_source(vec![ChunkDraft {
            text: "   ".to_string(),
            start_byte: None,
        }]);
        assert!(matches!(&r[0], ValidatedChunk::Rejected { .. }));
    }

    #[test]
    fn deny_pattern() {
        let p = ValidationPipeline::new(ValidatorConfig {
            deny_patterns: vec![r"click here now".to_string()],
            ..Default::default()
        })
        .unwrap();
        let r = p.validate_source(vec![ChunkDraft {
            text: "click here now!!!".to_string(),
            start_byte: None,
        }]);
        assert!(matches!(&r[0], ValidatedChunk::Rejected { reason } if reason == "deny_pattern"));
    }
}
