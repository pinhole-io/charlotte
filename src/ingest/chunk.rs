//! Text chunking strategies for the ingest pipeline (character windows and paragraphs).

/// One segment of text before validation and deduplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkDraft {
    pub text: String,
    /// Byte offset of `text` start in the original string (best-effort for citations).
    pub start_byte: Option<usize>,
}

/// Selects how to split document text into [`ChunkDraft`]s.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ChunkerConfig {
    /// Fixed-size character windows with optional overlap (UTF-8 safe, by Unicode scalar value).
    SlidingWindow { chunk_size: usize, overlap: usize },
    /// Split on blank lines; merge runs shorter than `merge_under_chars` into neighbors.
    Paragraph { merge_under_chars: usize },
    /// Split Markdown on ATX headings (`#` … `######`); each section runs until the next heading
    /// of equal or higher outline level (smaller `#` count = higher level). Oversized sections are
    /// split with [`sliding_window`].
    Hierarchical {
        max_section_chars: usize,
        subchunk_overlap: usize,
    },
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        ChunkerConfig::SlidingWindow {
            chunk_size: 1200,
            overlap: 150,
        }
    }
}

/// Splits full document text into chunk drafts.
pub trait Chunker {
    fn chunk(&self, text: &str) -> Vec<ChunkDraft>;
}

impl Chunker for ChunkerConfig {
    fn chunk(&self, text: &str) -> Vec<ChunkDraft> {
        match self {
            ChunkerConfig::SlidingWindow {
                chunk_size,
                overlap,
            } => sliding_window(text, *chunk_size, *overlap),
            ChunkerConfig::Paragraph { merge_under_chars } => {
                paragraph_chunks(text, *merge_under_chars)
            }
            ChunkerConfig::Hierarchical {
                max_section_chars,
                subchunk_overlap,
            } => hierarchical_markdown(text, *max_section_chars, *subchunk_overlap),
        }
    }
}

fn sliding_window(text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkDraft> {
    if chunk_size == 0 {
        return Vec::new();
    }
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    let overlap = overlap.min(chunk_size.saturating_sub(1));
    let step = chunk_size.saturating_sub(overlap).max(1);

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let end = (i + chunk_size).min(n);
        let start_byte = chars[i].0;
        let end_byte = if end < n { chars[end].0 } else { text.len() };
        let slice = &text[start_byte..end_byte];
        out.push(ChunkDraft {
            text: slice.to_string(),
            start_byte: Some(start_byte),
        });
        if end >= n {
            break;
        }
        i += step;
    }
    out
}

fn paragraph_chunks(text: &str, merge_under: usize) -> Vec<ChunkDraft> {
    let normalized = text.replace("\r\n", "\n");
    let raw: Vec<&str> = normalized
        .split("\n\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if raw.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<String> = Vec::new();
    for p in raw {
        if merge_under == 0 {
            merged.push(p.to_string());
            continue;
        }
        let count = p.chars().count();
        if count < merge_under {
            match merged.pop() {
                Some(mut prev) => {
                    if !prev.is_empty() {
                        prev.push('\n');
                    }
                    prev.push_str(p);
                    merged.push(prev);
                }
                None => merged.push(p.to_string()),
            }
        } else {
            merged.push(p.to_string());
        }
    }

    merged
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|text| ChunkDraft {
            text,
            start_byte: None,
        })
        .collect()
}

fn hierarchical_markdown(text: &str, max_section_chars: usize, overlap: usize) -> Vec<ChunkDraft> {
    let sections = crate::ingest::markdown::split_markdown_sections(text);
    if sections.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for sec in sections {
        out.extend(subchunk_section(&sec, max_section_chars, overlap));
    }
    out
}

fn subchunk_section(text: &str, max_section_chars: usize, overlap: usize) -> Vec<ChunkDraft> {
    if max_section_chars == 0 {
        return Vec::new();
    }
    let n = text.chars().count();
    if n <= max_section_chars {
        if text.trim().is_empty() {
            return Vec::new();
        }
        return vec![ChunkDraft {
            text: text.to_string(),
            start_byte: None,
        }];
    }
    sliding_window(text, max_section_chars, overlap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_respects_utf8() {
        let t = "αβγδε"; // 5 chars
        let c = ChunkerConfig::SlidingWindow {
            chunk_size: 2,
            overlap: 0,
        };
        let v = c.chunk(t);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].text, "αβ");
        assert_eq!(v[1].text, "γδ");
        assert_eq!(v[2].text, "ε");
    }

    #[test]
    fn sliding_overlap() {
        let t = "abcdefghij";
        let c = ChunkerConfig::SlidingWindow {
            chunk_size: 4,
            overlap: 2,
        };
        let v = c.chunk(t);
        assert!(v.len() >= 2);
        assert_eq!(v[0].text, "abcd");
        assert_eq!(v[1].text, "cdef");
    }

    #[test]
    fn paragraph_split() {
        let t = "First para.\n\nSecond para here.\n\nThird.";
        let c = ChunkerConfig::Paragraph {
            merge_under_chars: 0,
        };
        let v = c.chunk(t);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn paragraph_merges_short() {
        let t = "A\n\nB\n\nLong paragraph text that is not merged.";
        let c = ChunkerConfig::Paragraph {
            merge_under_chars: 10,
        };
        let v = c.chunk(t);
        assert!(v.len() < 3);
    }

    #[test]
    fn hierarchical_splits_on_headings() {
        let t = r"# Title
intro
## A
text a
## B
text b
# Next
tail
";
        let c = ChunkerConfig::Hierarchical {
            max_section_chars: 10_000,
            subchunk_overlap: 0,
        };
        let v = c.chunk(t);
        assert!(v.len() >= 2, "expected multiple sections, got {}", v.len());
        assert!(v[0].text.contains("Title") || v[0].text.contains("intro"));
    }

    #[test]
    fn hierarchical_subchunks_oversized() {
        let mut big = "# H\n".to_string();
        big.push_str(&"x".repeat(500));
        let c = ChunkerConfig::Hierarchical {
            max_section_chars: 120,
            subchunk_overlap: 10,
        };
        let v = c.chunk(&big);
        assert!(v.len() > 1);
    }
}
