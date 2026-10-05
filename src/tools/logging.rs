use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::Value;

use superglue::tools::{
    OnToolError, Tool, ToolInvokeError, ToolRegistry, ToolRetryPolicy, ToolSpec,
};

use crate::crawl::loopguard::{SharedGuardState, wrap_tool_with_guard};

/// Browser tools return errors to the model instead of aborting the chat loop.
pub(crate) fn charlotte_browser_tool_policy() -> ToolRetryPolicy {
    ToolRetryPolicy {
        on_error: OnToolError::Skip,
    }
}

pub(crate) async fn register_browser_tool(
    registry: &ToolRegistry,
    tool: Arc<dyn Tool>,
    guard_state: SharedGuardState,
) -> Result<(), ToolInvokeError> {
    let logged = wrap_browser_tool(tool, guard_state);
    registry
        .register_with_policy(logged, charlotte_browser_tool_policy())
        .await
}

pub(crate) fn wrap_browser_tool(
    tool: Arc<dyn Tool>,
    guard_state: SharedGuardState,
) -> Arc<dyn Tool> {
    let guarded = wrap_tool_with_guard(tool, guard_state);
    wrap_tool_with_logging(guarded)
}

/// Wraps `tool` so every call (and its outcome) is written to the `tracing` log — useful for
/// diagnosing a stalled or slow research run when the interactive progress panel gives no
/// indication of what is actually happening or how long it has been running.
///
/// Emitted at `info` level under the `charlotte::tools` target: set `RUST_LOG=info` (or
/// `HARN_DEBUG=1`, which raises harn's default log level to `info`) to see these in the log file.
fn wrap_tool_with_logging(tool: Arc<dyn Tool>) -> Arc<dyn Tool> {
    Arc::new(LoggingTool { inner: tool })
}

struct LoggingTool {
    inner: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for LoggingTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let name = self.inner.spec().name;
        tracing::info!(
            target: "charlotte::tools",
            tool = %name,
            "tool call started"
        );
        let start = Instant::now();
        let result = self.inner.call(arguments).await;
        let elapsed_ms = start.elapsed().as_millis();
        match &result {
            Ok(value) => {
                let outcome = classify_tool_outcome(value);
                tracing::info!(
                    target: "charlotte::tools",
                    tool = %name,
                    elapsed_ms,
                    outcome = %outcome,
                    "tool call finished"
                );
            }
            Err(error) => {
                tracing::warn!(
                    target: "charlotte::tools",
                    tool = %name,
                    elapsed_ms,
                    error = tool_error_class(error),
                    "tool call errored"
                );
            }
        }
        result
    }
}

/// Best-effort classification of a tool's JSON result for log scanning (`ok` / `blocked` / `error`
/// / `guarded` for loop-guard short-circuits like `duplicate_call` / `url_blocked` / `url_repeated`).
fn classify_tool_outcome(value: &Value) -> &'static str {
    if let Some(err) = value.get("error").and_then(Value::as_str) {
        return match err {
            "duplicate_call" | "url_blocked" | "domain_blocked" | "url_repeated"
            | "blocked_no_retry" => "guarded",
            _ => "error",
        };
    }
    if value.get("blocked").and_then(Value::as_bool) == Some(true) {
        return "blocked";
    }
    if value.get("ok").and_then(Value::as_bool) == Some(false) {
        return "error";
    }
    "ok"
}

fn tool_error_class(error: &ToolInvokeError) -> &'static str {
    match error {
        ToolInvokeError::HandlerFailed { .. } => "handler_failed",
        ToolInvokeError::UnknownTool { .. } => "unknown_tool",
        ToolInvokeError::DuplicateRegistration { .. } => "duplicate_registration",
    }
}
