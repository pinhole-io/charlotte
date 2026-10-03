//! Exact and SimHash near-duplicate detection within a batch of chunks.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::ingest::chunk::ChunkDraft;

/// Deduplication settings applied after validation.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Drop chunks with identical normalized text (whitespace collapsed).
    pub exact: bool,
    /// Enable SimHash near-duplicate removal.
    pub near_duplicate: bool,
    /// Max Hamming distance (0–64) for treating two SimHashes as duplicates.
    pub simhash_max_distance: u32,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            exact: true,
            near_duplicate: true,
            simhash_max_distance: 3,
        }
    }
}

/// Normalize text for exact dedup: trim, Unicode NFKC not required per plan — collapse runs of whitespace.
pub fn normalize_for_exact_dedup(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 64-bit SimHash for `text` (word tokens = whitespace split).
pub fn simhash64(text: &str) -> u64 {
    let mut weights = [0i32; 64];
    for token in text.split_whitespace() {
        if token.is_empty() {
            continue;
        }
        let h = hash_string(token);
        for (i, w) in weights.iter_mut().enumerate() {
            if (h >> i) & 1 == 1 {
                *w += 1;
            } else {
                *w -= 1;
            }
        }
    }
    let mut out = 0u64;
    for (i, w) in weights.iter().enumerate() {
        if *w > 0 {
            out |= 1u64 << i;
        }
    }
    out
}

fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Remove duplicates from `chunks` in order, preferring earlier entries.
pub fn deduplicate_batch(
    chunks: Vec<ChunkDraft>,
    config: &DedupConfig,
) -> (Vec<ChunkDraft>, usize) {
    if chunks.is_empty() {
        return (chunks, 0);
    }
    let mut removed = 0usize;
    let mut kept: Vec<ChunkDraft> = Vec::with_capacity(chunks.len());
    let mut seen_exact: HashSet<String> = HashSet::new();
    let mut simhashes: Vec<u64> = Vec::new();

    for c in chunks {
        let key = normalize_for_exact_dedup(&c.text);

        if config.exact && seen_exact.contains(&key) {
            removed += 1;
            continue;
        }

        if config.near_duplicate {
            let h = simhash64(&c.text);
            if simhashes
                .iter()
                .any(|&prev| hamming64(prev, h) <= config.simhash_max_distance)
            {
                removed += 1;
                continue;
            }
        }

        if config.exact {
            seen_exact.insert(key);
        }
        if config.near_duplicate {
            simhashes.push(simhash64(&c.text));
        }
        kept.push(c);
    }

    (kept, removed)
}

/// Like [`deduplicate_batch`] but preserves per-chunk metadata (first occurrence wins).
pub fn deduplicate_with_meta(
    items: Vec<(ChunkDraft, String, Option<String>)>,
    config: &DedupConfig,
) -> (Vec<(ChunkDraft, String, Option<String>)>, usize) {
    if items.is_empty() {
        return (items, 0);
    }
    let mut removed = 0usize;
    let mut kept: Vec<(ChunkDraft, String, Option<String>)> = Vec::with_capacity(items.len());
    let mut seen_exact: HashSet<String> = HashSet::new();
    let mut simhashes: Vec<u64> = Vec::new();

    for (draft, source_id, source_url) in items {
        let key = normalize_for_exact_dedup(&draft.text);

        if config.exact && seen_exact.contains(&key) {
            removed += 1;
            continue;
        }

        if config.near_duplicate {
            let h = simhash64(&draft.text);
            if simhashes
                .iter()
                .any(|&prev| hamming64(prev, h) <= config.simhash_max_distance)
            {
                removed += 1;
                continue;
            }
        }

        if config.exact {
            seen_exact.insert(key);
        }
        if config.near_duplicate {
            simhashes.push(simhash64(&draft.text));
        }
        kept.push((draft, source_id, source_url));
    }

    (kept, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_collapses_whitespace() {
        let a = ChunkDraft {
            text: "hello  world".to_string(),
            start_byte: None,
        };
        let b = ChunkDraft {
            text: "hello world".to_string(),
            start_byte: None,
        };
        let cfg = DedupConfig {
            exact: true,
            near_duplicate: false,
            ..Default::default()
        };
        let (out, n) = deduplicate_batch(vec![a, b], &cfg);
        assert_eq!(out.len(), 1);
        assert_eq!(n, 1);
    }

    #[test]
    fn hamming_counts_bits() {
        assert_eq!(hamming64(0, 0), 0);
        assert_eq!(hamming64(1, 0), 1);
    }
}
