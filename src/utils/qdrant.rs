//! [`superglue::tools::Tool`] implementations backed by [`qdrant_client::Qdrant`].

use std::sync::Arc;

use async_trait::async_trait;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors_output::VectorsOptions as OutputVectorsOptions;
use qdrant_client::qdrant::{PointStruct, QueryPointsBuilder, UpsertPointsBuilder, VectorsOutput};
use serde_json::{Value, json};

use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

const SEARCH: &str = "qdrant_search";
const UPSERT: &str = "qdrant_upsert";

/// Register Qdrant tools (`qdrant_*`) on `registry`.
///
/// Uses the same [`Qdrant`] client for all tool calls (cloneable, connection-pooled).
pub async fn register_charlotte_qdrant_tools(
    registry: &ToolRegistry,
    client: &Qdrant,
) -> Result<(), ToolInvokeError> {
    let c = Arc::new(client.clone());
    registry
        .register(Arc::new(QdrantSearchTool(Arc::clone(&c))))
        .await?;
    registry.register(Arc::new(QdrantUpsertTool(c))).await?;
    Ok(())
}

struct QdrantSearchTool(Arc<Qdrant>);
struct QdrantUpsertTool(Arc<Qdrant>);

#[async_trait]
impl Tool for QdrantSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: SEARCH.to_string(),
            description: Some(
                "Vector similarity search in a Qdrant collection. Returns scored points (id, score, payload, optional vectors). Uses the collection's default vector unless `using` names a vector.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "query_vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Dense query vector (f32)"
                    },
                    "limit": { "type": "integer", "description": "Max hits (1–100)", "default": 10 },
                    "using": { "type": "string", "description": "Optional vector name for named-vector collections" },
                    "with_payload": { "type": "boolean", "description": "Include payload (default true)", "default": true },
                    "with_vectors": { "type": "boolean", "description": "Include embedding vectors in results (default false)", "default": false },
                    "offset": { "type": "integer", "description": "Skip this many hits (default 0)" }
                },
                "required": ["collection", "query_vector"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let collection = arguments
            .get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: collection", None))?;
        let query_vector = parse_f32_vector(
            arguments
                .get("query_vector")
                .ok_or_else(|| ToolInvokeError::handler("missing field: query_vector", None))?,
        )
        .map_err(|m| ToolInvokeError::handler(m, None))?;
        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(10)
            .clamp(1, 100);
        let offset = arguments
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let with_payload = arguments
            .get("with_payload")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let with_vectors = arguments
            .get("with_vectors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let using = arguments
            .get("using")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());

        let mut b = QueryPointsBuilder::new(collection)
            .query(query_vector)
            .limit(limit)
            .offset(offset)
            .with_payload(with_payload)
            .with_vectors(with_vectors);
        if let Some(name) = using {
            b = b.using(name.to_string());
        }

        let response = self
            .0
            .query(b)
            .await
            .map_err(|e| ToolInvokeError::handler(e.to_string(), None))?;

        let points: Vec<Value> = response
            .result
            .into_iter()
            .map(scored_point_to_json)
            .collect();

        Ok(json!({
            "points": points,
            "time": response.time,
        }))
    }
}

#[async_trait]
impl Tool for QdrantUpsertTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: UPSERT.to_string(),
            description: Some(
                "Insert or update points in a Qdrant collection. Each point needs an id (unsigned integer or UUID string), a dense `vector`, and optional JSON `payload` object. Waits for indexing by default.".to_string(),
            ),
            static_tool: false,
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "points": {
                        "type": "array",
                        "description": "Points to upsert",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "description": "Numeric id (u64) or UUID string" },
                                "vector": {
                                    "type": "array",
                                    "items": { "type": "number" },
                                    "description": "Dense vector for the default vector name"
                                },
                                "payload": { "type": "object", "description": "Optional JSON payload" }
                            },
                            "required": ["id", "vector"]
                        }
                    },
                    "wait": { "type": "boolean", "description": "Wait until visible for search (default true)", "default": true }
                },
                "required": ["collection", "points"]
            }),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let collection = arguments
            .get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolInvokeError::handler("missing string field: collection", None))?;
        let wait = arguments
            .get("wait")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let points_val = arguments
            .get("points")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ToolInvokeError::handler("missing array field: points", None))?;
        if points_val.is_empty() {
            return Err(ToolInvokeError::handler(
                "points must be a non-empty array",
                None,
            ));
        }
        if points_val.len() > 256 {
            return Err(ToolInvokeError::handler(
                "points: at most 256 points per call",
                None,
            ));
        }

        let mut points = Vec::with_capacity(points_val.len());
        for (i, p) in points_val.iter().enumerate() {
            let id = p.get("id").ok_or_else(|| {
                ToolInvokeError::handler(format!("points[{i}]: missing id"), None)
            })?;
            let point_id = parse_point_id(id).map_err(|m| ToolInvokeError::handler(m, None))?;
            let vec_json = p.get("vector").ok_or_else(|| {
                ToolInvokeError::handler(format!("points[{i}]: missing vector"), None)
            })?;
            let vector =
                parse_f32_vector(vec_json).map_err(|m| ToolInvokeError::handler(m, None))?;
            let payload: Payload = match p.get("payload") {
                None | Some(Value::Null) => Payload::try_from(json!({}))
                    .map_err(|e| ToolInvokeError::handler(format!("payload default: {e}"), None))?,
                Some(v) if v.as_object().is_some() => {
                    Payload::try_from(v.clone()).map_err(|e| {
                        ToolInvokeError::handler(format!("points[{i}] payload: {e}"), None)
                    })?
                }
                Some(_) => {
                    return Err(ToolInvokeError::handler(
                        format!("points[{i}]: payload must be a JSON object or omitted"),
                        None,
                    ));
                }
            };
            points.push(PointStruct::new(point_id, vector, payload));
        }

        let req = UpsertPointsBuilder::new(collection, points).wait(wait);

        let response = self
            .0
            .upsert_points(req)
            .await
            .map_err(|e| ToolInvokeError::handler(e.to_string(), None))?;

        let op = response.result;
        Ok(json!({
            "time": response.time,
            "operation_id": op.as_ref().and_then(|r| r.operation_id),
            "status": op.map(|r| r.status),
        }))
    }
}

fn parse_f32_vector(v: &Value) -> Result<Vec<f32>, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "expected JSON array of numbers".to_string())?;
    if arr.is_empty() {
        return Err("vector must be non-empty".to_string());
    }
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let n = x
            .as_f64()
            .ok_or_else(|| "vector elements must be numbers".to_string())?;
        if !n.is_finite() {
            return Err("vector elements must be finite".to_string());
        }
        out.push(n as f32);
    }
    Ok(out)
}

fn parse_point_id(v: &Value) -> Result<qdrant_client::qdrant::PointId, String> {
    match v {
        Value::Number(n) => {
            let u = n
                .as_u64()
                .ok_or_else(|| "point id number must be a non-negative integer".to_string())?;
            Ok(u.into())
        }
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                return Err("point id string must be non-empty".to_string());
            }
            Ok(t.into())
        }
        _ => Err("point id must be a JSON number or string".to_string()),
    }
}

fn scored_point_to_json(p: qdrant_client::qdrant::ScoredPoint) -> Value {
    let id = match p.id.and_then(|pid| pid.point_id_options) {
        Some(PointIdOptions::Num(n)) => json!(n),
        Some(PointIdOptions::Uuid(s)) => json!(s),
        None => Value::Null,
    };
    let payload: serde_json::Map<String, Value> = p
        .payload
        .into_iter()
        .map(|(k, v)| (k, v.into_json()))
        .collect();
    json!({
        "id": id,
        "score": p.score,
        "version": p.version,
        "payload": Value::Object(payload),
        "vectors": vectors_output_to_json(p.vectors),
    })
}

fn vectors_output_to_json(vo: Option<VectorsOutput>) -> Value {
    let Some(vo) = vo else {
        return Value::Null;
    };
    match vo.vectors_options {
        Some(OutputVectorsOptions::Vector(v)) => query_vector_to_json(v.into_vector()),
        Some(OutputVectorsOptions::Vectors(named)) => {
            let m: serde_json::Map<String, Value> = named
                .vectors
                .into_iter()
                .map(|(k, vo)| (k, query_vector_to_json(vo.into_vector())))
                .collect();
            Value::Object(m)
        }
        None => Value::Null,
    }
}

fn query_vector_to_json(v: qdrant_client::qdrant::vector_output::Vector) -> Value {
    use qdrant_client::qdrant::vector_output::Vector::*;
    match v {
        Dense(d) => json!(d.data),
        Sparse(s) => json!({ "indices": s.indices, "values": s.values }),
        MultiDense(m) => json!(m.vectors.iter().map(|d| &d.data).collect::<Vec<_>>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vector_ok() {
        let v = json!([1.0, 2.5, -3.0]);
        let out = parse_f32_vector(&v).unwrap();
        assert_eq!(out, vec![1.0f32, 2.5, -3.0]);
    }

    #[test]
    fn parse_point_id_num_and_str() {
        assert!(parse_point_id(&json!(42)).is_ok());
        assert!(parse_point_id(&json!("550e8400-e29b-41d4-a716-446655440000")).is_ok());
    }
}
