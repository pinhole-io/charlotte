//! Direct Qdrant point operations for scripts and batch jobs — no LLM agent.
//!
//! These wrap [`Qdrant`] with small, typed inputs. For full control, use [`Qdrant`] and the
//! `qdrant_client::qdrant::*` builders directly.

use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CountPointsBuilder, DeletePointsBuilder, GetPointsBuilder, PointId, PointStruct, PointsIdsList,
    QueryPointsBuilder, ScrollPointsBuilder, UpsertPointsBuilder,
};
use serde_json::Value;
use thiserror::Error;

/// Errors from high-level CRUD helpers (invalid JSON payload or Qdrant client errors).
#[derive(Debug, Error)]
pub enum QdrantCrudError {
    #[error(transparent)]
    Client(#[from] qdrant_client::QdrantError),
    #[error("payload must be a JSON object: {0}")]
    PayloadNotObject(String),
}

/// Point id for script APIs (unsigned id or UUID string).
#[derive(Debug, Clone)]
pub enum PointIdArg {
    Unsigned(u64),
    Key(String),
}

impl From<PointIdArg> for PointId {
    fn from(a: PointIdArg) -> Self {
        match a {
            PointIdArg::Unsigned(n) => n.into(),
            PointIdArg::Key(s) => s.into(),
        }
    }
}

/// One row to upsert: dense vector, optional JSON object payload.
#[derive(Debug, Clone)]
pub struct UpsertRow {
    pub id: PointIdArg,
    pub vector: Vec<f32>,
    pub payload: Option<Value>,
}

impl UpsertRow {
    fn into_point_struct(self) -> Result<PointStruct, QdrantCrudError> {
        let id: PointId = self.id.into();
        let payload_val = match self.payload {
            None | Some(Value::Null) => Value::Object(Default::default()),
            Some(o @ Value::Object(_)) => o,
            Some(other) => return Err(QdrantCrudError::PayloadNotObject(other.to_string())),
        };
        let payload = Payload::try_from(payload_val)
            .map_err(|e| QdrantCrudError::PayloadNotObject(e.to_string()))?;
        Ok(PointStruct::new(id, self.vector, payload))
    }
}

/// Options for [`query_nearest`].
#[derive(Debug, Clone)]
pub struct QueryNearestOpts {
    pub limit: u64,
    pub offset: u64,
    pub with_payload: bool,
    pub with_vectors: bool,
    /// Named vector in multi-vector collections (`None` = default).
    pub using_vector: Option<String>,
}

impl Default for QueryNearestOpts {
    fn default() -> Self {
        Self {
            limit: 10,
            offset: 0,
            with_payload: true,
            with_vectors: false,
            using_vector: None,
        }
    }
}

/// Vector similarity search (same semantics as the agent’s `qdrant_search` tool).
pub async fn query_nearest(
    client: &Qdrant,
    collection: &str,
    query_vector: Vec<f32>,
    opts: QueryNearestOpts,
) -> Result<qdrant_client::qdrant::QueryResponse, qdrant_client::QdrantError> {
    let limit = opts.limit.clamp(1, 10_000);
    let mut b = QueryPointsBuilder::new(collection)
        .query(query_vector)
        .limit(limit)
        .offset(opts.offset)
        .with_payload(opts.with_payload)
        .with_vectors(opts.with_vectors);
    if let Some(name) = opts.using_vector {
        b = b.using(name);
    }
    client.query(b).await
}

/// Insert or replace points. Waits for visibility when `wait` is true.
pub async fn upsert_points(
    client: &Qdrant,
    collection: &str,
    rows: Vec<UpsertRow>,
    wait: bool,
) -> Result<qdrant_client::qdrant::PointsOperationResponse, QdrantCrudError> {
    let points: Result<Vec<PointStruct>, QdrantCrudError> =
        rows.into_iter().map(|r| r.into_point_struct()).collect();
    let points = points?;
    let mut req = UpsertPointsBuilder::new(collection, points);
    if wait {
        req = req.wait(true);
    }
    Ok(client.upsert_points(req).await?)
}

/// Fetch points by id.
pub async fn get_points(
    client: &Qdrant,
    collection: &str,
    ids: Vec<PointIdArg>,
    with_vectors: bool,
    with_payload: bool,
) -> Result<qdrant_client::qdrant::GetResponse, qdrant_client::QdrantError> {
    let ids: Vec<PointId> = ids.into_iter().map(Into::into).collect();
    client
        .get_points(
            GetPointsBuilder::new(collection, ids)
                .with_vectors(with_vectors)
                .with_payload(with_payload),
        )
        .await
}

/// Delete points by id.
pub async fn delete_points_by_ids(
    client: &Qdrant,
    collection: &str,
    ids: Vec<PointIdArg>,
    wait: bool,
) -> Result<qdrant_client::qdrant::PointsOperationResponse, qdrant_client::QdrantError> {
    let ids: Vec<PointId> = ids.into_iter().map(Into::into).collect();
    let list = PointsIdsList { ids };
    let mut req = DeletePointsBuilder::new(collection).points(list);
    if wait {
        req = req.wait(true);
    }
    client.delete_points(req).await
}

/// Options for [`scroll_points`].
#[derive(Debug, Clone)]
pub struct ScrollOpts {
    pub limit: u32,
    /// Pagination: start after this point id (Qdrant scroll cursor).
    pub offset: Option<PointIdArg>,
    pub with_payload: bool,
    pub with_vectors: bool,
}

impl Default for ScrollOpts {
    fn default() -> Self {
        Self {
            limit: 64,
            offset: None,
            with_payload: true,
            with_vectors: false,
        }
    }
}

/// Iterate points in a collection (no vector query — use for export / batch reads).
pub async fn scroll_points(
    client: &Qdrant,
    collection: &str,
    opts: ScrollOpts,
) -> Result<qdrant_client::qdrant::ScrollResponse, qdrant_client::QdrantError> {
    let mut b = ScrollPointsBuilder::new(collection)
        .limit(opts.limit)
        .with_payload(opts.with_payload)
        .with_vectors(opts.with_vectors);
    if let Some(off) = opts.offset {
        b = b.offset(PointId::from(off));
    }
    client.scroll(b).await
}

/// Count points (optionally exact — slower but precise when filters are used later).
pub async fn count_points(
    client: &Qdrant,
    collection: &str,
    exact: bool,
) -> Result<qdrant_client::qdrant::CountResponse, qdrant_client::QdrantError> {
    client
        .count(CountPointsBuilder::new(collection).exact(exact))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_row_empty_payload_ok() {
        let row = UpsertRow {
            id: PointIdArg::Unsigned(1),
            vector: vec![0.0_f32, 1.0],
            payload: None,
        };
        row.into_point_struct().unwrap();
    }

    #[test]
    fn upsert_row_rejects_non_object_payload() {
        let row = UpsertRow {
            id: PointIdArg::Unsigned(1),
            vector: vec![0.0_f32],
            payload: Some(Value::String("nope".into())),
        };
        assert!(row.into_point_struct().is_err());
    }
}
