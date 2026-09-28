//! Map [`crate::ingest::IngestChunk`] rows to Qdrant upserts after you compute embeddings elsewhere.
//!
//! v1 does not call embedding APIs; pass one vector per chunk in order.

use serde_json::{Value, json};

use super::qdrant_crud::{PointIdArg, UpsertRow};

/// Build JSON payloads suitable for Qdrant `payload` (text + provenance).
pub fn ingest_chunk_payload(chunk: &crate::ingest::IngestChunk) -> Value {
    json!({
        "text": chunk.text,
        "source_id": chunk.source_id,
        "source_url": chunk.source_url,
        "ingest_chunk_id": chunk.id,
    })
}

/// Zip ingest chunks with caller-supplied ids and vectors into [`UpsertRow`]s.
///
/// Returns an error if the three slices have different lengths.
pub fn ingest_chunks_to_upsert_rows(
    chunks: &[crate::ingest::IngestChunk],
    ids: Vec<PointIdArg>,
    vectors: Vec<Vec<f32>>,
) -> Result<Vec<UpsertRow>, &'static str> {
    if chunks.len() != ids.len() || chunks.len() != vectors.len() {
        return Err("chunks, ids, and vectors must have the same length");
    }
    let mut out = Vec::with_capacity(chunks.len());
    for ((chunk, id), vector) in chunks.iter().zip(ids.into_iter()).zip(vectors.into_iter()) {
        out.push(UpsertRow {
            id,
            vector,
            payload: Some(ingest_chunk_payload(chunk)),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_rows_align_lengths() {
        let chunks = vec![
            crate::ingest::IngestChunk {
                id: 0,
                text: "a".to_string(),
                source_id: "s".to_string(),
                source_url: None,
            },
            crate::ingest::IngestChunk {
                id: 1,
                text: "b".to_string(),
                source_id: "s".to_string(),
                source_url: None,
            },
        ];
        let ids = vec![PointIdArg::Unsigned(1), PointIdArg::Unsigned(2)];
        let vectors = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let rows = ingest_chunks_to_upsert_rows(&chunks, ids, vectors).expect("rows");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn upsert_rows_length_mismatch_errors() {
        let chunks = vec![crate::ingest::IngestChunk {
            id: 0,
            text: "a".to_string(),
            source_id: "s".to_string(),
            source_url: None,
        }];
        let err = ingest_chunks_to_upsert_rows(&chunks, vec![], vec![]);
        assert!(err.is_err());
    }
}
