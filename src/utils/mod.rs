//! Shared helpers and optional integrations (blocklists, vector store tools, etc.).

pub(crate) mod blocklist;
pub mod qdrant;
/// Direct Qdrant CRUD-style helpers for scripts (outside the agent).
pub mod qdrant_crud;
/// Helpers to turn [`crate::ingest::IngestChunk`] into Qdrant upserts after embedding.
pub mod qdrant_ingest;
