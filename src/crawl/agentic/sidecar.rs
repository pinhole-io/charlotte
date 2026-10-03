//! Browser Use sidecar process (JSON lines over stdin/stdout).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::targets::ObservePage;
use crate::CharlotteError;
use crate::config::cdp_endpoint_from_env;

const DEFAULT_CDP: &str = "http://127.0.0.1:9222";
const ROUNDTRIP_TIMEOUT: Duration = Duration::from_secs(45);
const FIRST_RPC_TIMEOUT: Duration = Duration::from_secs(120);
const STDERR_TAIL_MAX: usize = 8_192;

/// Page actions the crawl controller may send.
#[async_trait]
pub trait PageActor: Send {
    async fn start(&mut self) -> Result<(), CharlotteError>;
    async fn navigate(&mut self, url: &str) -> Result<ObservePage, CharlotteError>;
    async fn observe(&mut self) -> Result<ObservePage, CharlotteError>;
    async fn click(&mut self, target_id: &str) -> Result<ObservePage, CharlotteError>;
    async fn type_text(
        &mut self,
        target_id: &str,
        text: &str,
    ) -> Result<ObservePage, CharlotteError>;
    async fn press(&mut self, key: &str) -> Result<ObservePage, CharlotteError>;
    async fn wait(&mut self, ms: u64) -> Result<ObservePage, CharlotteError>;
    async fn scroll(&mut self, delta: i32) -> Result<ObservePage, CharlotteError>;
    async fn follow(
        &mut self,
        target_id: &str,
        href: Option<&str>,
    ) -> Result<ObservePage, CharlotteError>;
    /// Collect every `a[href]` on the page (not only the first visible 40).
    async fn get_links(&mut self) -> Result<ObservePage, CharlotteError>;
    async fn stop(&mut self) -> Result<(), CharlotteError>;
}

/// Spawn `uv run` on the PEP 723 sidecar script.
pub struct UvSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    cdp_url: String,
    started: bool,
    stderr_tail: Arc<Mutex<String>>,
}

impl UvSidecar {
    /// Spawn the sidecar. Does not connect until [`PageActor::start`].
    pub fn spawn() -> Result<Self, CharlotteError> {
        let script = sidecar_script_path();
        if !script.is_file() {
            return Err(CharlotteError::msg(format!(
                "browser-use sidecar script missing: {}",
                script.display()
            )));
        }
        let mut cmd = Command::new("uv");
        cmd.arg("run")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env_clear();
        for (key, value) in sidecar_env() {
            cmd.env(key, value);
        }
        let mut child = cmd.spawn().map_err(|e| {
            CharlotteError::msg(format!(
                "failed to spawn uv sidecar ({e}). Install uv and run the sidecar once to fetch browser-use."
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CharlotteError::msg("sidecar stdin missing"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CharlotteError::msg("sidecar stdout missing"))?;
        let stderr_tail = Arc::new(Mutex::new(String::new()));
        if let Some(stderr) = child.stderr.take() {
            drain_sidecar_stderr(stderr, Arc::clone(&stderr_tail));
        }
        let cdp_url = cdp_url();
        tracing::info!(
            target: "charlotte::agentic_crawl",
            cdp_url = %cdp_url,
            "sidecar spawned; will connect to explicit CDP URL"
        );
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            cdp_url,
            started: false,
            stderr_tail,
        })
    }

    fn stderr_snapshot(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    fn fail(&self, msg: impl Into<String>) -> CharlotteError {
        let msg = msg.into();
        let stderr = self.stderr_snapshot();
        let trimmed = stderr.trim();
        if trimmed.is_empty() {
            tracing::warn!(target: "charlotte::agentic_crawl", "{msg}");
            CharlotteError::msg(msg)
        } else {
            tracing::warn!(
                target: "charlotte::agentic_crawl",
                stderr = %trimmed,
                "{msg}"
            );
            eprintln!("[crawl:sidecar] {msg}; stderr: {trimmed}");
            CharlotteError::msg(format!("{msg}; sidecar stderr: {trimmed}"))
        }
    }

    async fn rpc(&mut self, body: Value) -> Result<Value, CharlotteError> {
        self.rpc_with_timeout(body, ROUNDTRIP_TIMEOUT).await
    }

    async fn rpc_with_timeout(
        &mut self,
        mut body: Value,
        timeout: Duration,
    ) -> Result<Value, CharlotteError> {
        let id = self.next_id;
        self.next_id += 1;
        body["id"] = json!(id.to_string());
        let line = serde_json::to_string(&body)
            .map_err(|e| CharlotteError::msg(format!("sidecar encode: {e}")))?;
        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| self.fail(format!("sidecar write: {e}")))?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| self.fail(format!("sidecar write: {e}")))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| self.fail(format!("sidecar flush: {e}")))?;

        let mut response = String::new();
        let read = self.stdout.read_line(&mut response);
        tokio::time::timeout(timeout, read)
            .await
            .map_err(|_| self.fail("sidecar timed out"))?
            .map_err(|e| self.fail(format!("sidecar read: {e}")))?;
        if response.trim().is_empty() {
            return Err(self.fail("sidecar closed stdout"));
        }
        let value: Value = serde_json::from_str(response.trim())
            .map_err(|e| self.fail(format!("sidecar decode: {e}")))?;
        if value.get("ok").and_then(Value::as_bool) != Some(true) {
            let err = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("sidecar error");
            return Err(self.fail(err));
        }
        Ok(value)
    }

    async fn rpc_page(&mut self, body: Value) -> Result<ObservePage, CharlotteError> {
        let value = self.rpc(body).await?;
        page_from_value(&value)
    }
}

#[async_trait]
impl PageActor for UvSidecar {
    async fn start(&mut self) -> Result<(), CharlotteError> {
        self.rpc_with_timeout(
            json!({ "cmd": "start", "cdp_url": self.cdp_url }),
            FIRST_RPC_TIMEOUT,
        )
        .await?;
        self.started = true;
        Ok(())
    }

    async fn navigate(&mut self, url: &str) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "navigate", "url": url }))
            .await
    }

    async fn observe(&mut self) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "observe" })).await
    }

    async fn click(&mut self, target_id: &str) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "click", "target_id": target_id }))
            .await
    }

    async fn type_text(
        &mut self,
        target_id: &str,
        text: &str,
    ) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "type", "target_id": target_id, "text": text }))
            .await
    }

    async fn press(&mut self, key: &str) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "press", "key": key })).await
    }

    async fn wait(&mut self, ms: u64) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "wait", "ms": ms })).await
    }

    async fn scroll(&mut self, delta: i32) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "scroll", "delta": delta }))
            .await
    }

    async fn follow(
        &mut self,
        target_id: &str,
        href: Option<&str>,
    ) -> Result<ObservePage, CharlotteError> {
        if let Some(url) = href.map(str::trim).filter(|u| !u.is_empty()) {
            return self.navigate(url).await;
        }
        self.rpc_page(json!({ "cmd": "follow", "target_id": target_id }))
            .await
    }

    async fn get_links(&mut self) -> Result<ObservePage, CharlotteError> {
        self.rpc_page(json!({ "cmd": "get_links" })).await
    }

    async fn stop(&mut self) -> Result<(), CharlotteError> {
        if self.started {
            let _ = self.rpc(json!({ "cmd": "stop" })).await;
            self.started = false;
        }
        let _ = self.child.start_kill();
        Ok(())
    }
}

impl Drop for UvSidecar {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn page_from_value(value: &Value) -> Result<ObservePage, CharlotteError> {
    serde_json::from_value(value.clone())
        .map_err(|e| CharlotteError::msg(format!("sidecar page: {e}")))
}

fn drain_sidecar_stderr(stderr: tokio::process::ChildStderr, tail: Arc<Mutex<String>>) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break,
                        Ok(_) => append_stderr_tail(&tail, &line),
                        Err(_) => break,
                    }
                }
            });
        }
        Err(_) => {
            let _ = stderr;
        }
    }
}

fn append_stderr_tail(tail: &Mutex<String>, line: &str) {
    let Ok(mut guard) = tail.lock() else {
        return;
    };
    guard.push_str(line);
    if guard.len() > STDERR_TAIL_MAX {
        let drain = guard.len() - STDERR_TAIL_MAX;
        guard.drain(..drain);
    }
}

fn sidecar_script_path() -> PathBuf {
    if let Ok(raw) = std::env::var("CHARLOTTE_BROWSER_USE_SIDECAR") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("python/browser_use_sidecar.py")
}

fn cdp_url() -> String {
    cdp_endpoint_from_env()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_CDP.to_string())
}

/// Inherit PATH-style vars. Drop secrets so the sidecar cannot read provider keys.
fn sidecar_env() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    std::env::vars_os()
        .filter(|(key, _)| {
            let name = key.to_string_lossy().to_ascii_uppercase();
            !name.contains("API_KEY")
                && !name.contains("TOKEN")
                && !name.contains("SECRET")
                && !name.contains("PASSWORD")
                && !name.contains("COOKIE")
                && name != "TYPESAFE_BASE_URL"
        })
        .collect()
}

/// In-memory actor for tests. Each mutating call advances through scripted pages.
#[derive(Debug)]
pub struct ScriptedPageActor {
    pages: Vec<ObservePage>,
    index: usize,
    pub actions: Vec<String>,
}

impl ScriptedPageActor {
    #[must_use]
    pub fn new(pages: Vec<ObservePage>) -> Self {
        Self {
            pages,
            index: 0,
            actions: Vec::new(),
        }
    }

    fn current(&self) -> Result<ObservePage, CharlotteError> {
        self.pages
            .get(self.index)
            .cloned()
            .ok_or_else(|| CharlotteError::msg("scripted actor has no pages"))
    }

    fn advance(&mut self) -> Result<ObservePage, CharlotteError> {
        if self.index + 1 < self.pages.len() {
            self.index += 1;
        }
        self.current()
    }

    /// Jump to the scripted page with this URL. Stays put when none matches.
    fn seek(&mut self, url: &str) -> Result<ObservePage, CharlotteError> {
        if let Some(found) = self.pages.iter().position(|p| p.url == url) {
            self.index = found;
        }
        self.current()
    }
}

#[async_trait]
impl PageActor for ScriptedPageActor {
    async fn start(&mut self) -> Result<(), CharlotteError> {
        self.actions.push("start".into());
        Ok(())
    }

    async fn navigate(&mut self, url: &str) -> Result<ObservePage, CharlotteError> {
        self.actions.push(format!("navigate:{url}"));
        self.seek(url)
    }

    async fn observe(&mut self) -> Result<ObservePage, CharlotteError> {
        self.actions.push("observe".into());
        self.current()
    }

    async fn click(&mut self, target_id: &str) -> Result<ObservePage, CharlotteError> {
        self.actions.push(format!("click:{target_id}"));
        self.advance()
    }

    async fn type_text(
        &mut self,
        target_id: &str,
        text: &str,
    ) -> Result<ObservePage, CharlotteError> {
        let _ = text;
        self.actions.push(format!("type:{target_id}"));
        self.advance()
    }

    async fn press(&mut self, key: &str) -> Result<ObservePage, CharlotteError> {
        self.actions.push(format!("press:{key}"));
        self.advance()
    }

    async fn wait(&mut self, ms: u64) -> Result<ObservePage, CharlotteError> {
        self.actions.push(format!("wait:{ms}"));
        self.current()
    }

    async fn scroll(&mut self, delta: i32) -> Result<ObservePage, CharlotteError> {
        self.actions.push(format!("scroll:{delta}"));
        self.current()
    }

    async fn follow(
        &mut self,
        target_id: &str,
        href: Option<&str>,
    ) -> Result<ObservePage, CharlotteError> {
        self.actions.push(format!("follow:{target_id}"));
        match href.map(str::trim).filter(|url| !url.is_empty()) {
            Some(url) => self.seek(url),
            None => self.advance(),
        }
    }

    async fn get_links(&mut self) -> Result<ObservePage, CharlotteError> {
        self.actions.push("get_links".into());
        self.current()
    }

    async fn stop(&mut self) -> Result<(), CharlotteError> {
        self.actions.push("stop".into());
        Ok(())
    }
}
