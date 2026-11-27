//! Host integration primitives for embedding Charlotte tools with approvals and scoped credentials.

// std
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// external crates
use async_trait::async_trait;
use lockbox::providers::OnePasswordCli;
use lockbox::{Lockbox, Origin, ProviderRegistry};
use secrecy::ExposeSecret;
use serde_json::Value;
use superglue::tools::{
    OnToolError, Tool, ToolContextPolicy, ToolInvokeError, ToolRetryPolicy, ToolSpec,
};
use url::Url;

// local crate
use crate::crawl::agentic::CrawlInput;
use crate::tools::{CharlotteTool, ResolvedCredential};

const MAX_CREDENTIAL_ALIAS_BYTES: usize = 32;

/// Alias for [`crate::CharlotteTool`] used in host tool selection lists.
pub type CharlotteToolId = CharlotteTool;

/// Standard bounded Charlotte tool set exposed to host runtimes.
pub const SELECTED_TOOLS: &[CharlotteToolId] = &[
    CharlotteTool::WebSearch,
    CharlotteTool::Fetch,
    CharlotteTool::AgenticCrawl,
];

/// Approval resource string shared by bounded public-web read/search tools.
pub const PUBLIC_WEB_RESOURCE: &str = "search://public-web";

/// Interactive approval explanation for `web_search`.
pub const WEB_SEARCH_APPROVAL_DETAIL: &str = "This tool can search the public web.";

/// Interactive approval explanation for `fetch`.
pub const FETCH_APPROVAL_DETAIL: &str =
    "This tool can fetch a public web page and return Markdown.";

/// Interactive approval explanation for `agentic_crawl`.
pub const AGENTIC_CRAWL_APPROVAL_DETAIL: &str = "This tool can browse one public web origin and can use the named credential profile. \
The first call opens one visible browser for this conversation.";

/// Base system-prompt guidance for hosts exposing [`SELECTED_TOOLS`].
pub const READY_PROMPT: &str = "\
## Charlotte web tools
- Use `web_search` for bounded public-web result snippets.
- Use the pages `web_search` returns. A page about the same term in another field is a source.
- Prefer snippets when they answer the ask. Use `fetch` when a page body is needed; it returns Markdown for HTML, or capped plain text.
- Prefer `fetch` before `agentic_crawl` for a static page body.
- Write the report from the snippets and fetched text you already have. Do not fetch many full pages when a few snippets are enough.
- Use `agentic_crawl` for a headed walk, form fill, login, or JS-heavy page on a direct result URL from `web_search`. \
`agentic_crawl` opens one visible browser window for this conversation.
- Do not use a search-engine results page as the `agentic_crawl` seed. Search engines commonly block browser automation.
- Do not spawn agents for public-web search; keep those calls on this turn.
- Do not put secrets in a search query, URL, goal, or extraction schema.
- Put non-secret form values in `agentic_crawl.inputs`.
- Use a listed `credential` profile for passwords or other secrets.";

/// Per-agent scope restricting which [`Lockbox`] credential profile aliases are visible and usable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialScope {
    /// Permit every origin-bound profile loaded in the [`Lockbox`].
    All,
    /// Permit only the explicitly listed profile aliases.
    Allowlist(BTreeSet<String>),
}

impl CredentialScope {
    /// Scope that denies all credential aliases.
    #[must_use]
    pub fn none() -> Self {
        Self::Allowlist(BTreeSet::new())
    }

    /// Build an allowlist scope from an iterator of profile aliases.
    #[must_use]
    pub fn from_aliases<I, S>(aliases: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Allowlist(aliases.into_iter().map(Into::into).collect())
    }

    /// Return whether `alias` is permitted by this scope.
    #[must_use]
    pub fn allows(&self, alias: &str) -> bool {
        match self {
            Self::All => true,
            Self::Allowlist(allowed) => allowed.contains(alias),
        }
    }
}

/// Structured host approval request produced before invoking a Charlotte tool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostApprovalRequest {
    /// Canonical Charlotte tool name (`web_search`, `fetch`, `agentic_crawl`).
    pub tool_name: String,
    /// Bounded approval resource identifier derived from untrusted tool arguments.
    pub resource: String,
    /// Human-readable capability explanation for interactive prompts.
    pub detail: Option<&'static str>,
    /// Redacted, secret-free argument projection suitable for audit logs.
    pub audit_arguments: Value,
    /// Whether unattended execution requires an exact resource rule rather than a wildcard.
    pub requires_exact_rule: bool,
}

impl HostApprovalRequest {
    /// Build a redacted approval request for a Charlotte tool call.
    #[must_use]
    pub fn for_tool(tool_name: &str, arguments: &Value) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            resource: approval_resource_for_tool(tool_name, arguments),
            detail: approval_detail_for_tool(tool_name),
            audit_arguments: audit_arguments_for_tool(tool_name, arguments),
            requires_exact_rule: true,
        }
    }
}

/// Host callback that authorizes or rejects a Charlotte tool call before execution.
#[async_trait]
pub trait HostApproval: Send + Sync {
    /// Return `Ok(true)` to allow execution, `Ok(false)` to deny, or `Err(reason)` when blocked.
    async fn approve(&self, request: HostApprovalRequest) -> Result<bool, String>;
}

/// [`superglue::tools::Tool`] wrapper enforcing credential preflight and [`HostApproval`].
///
/// Approval runs outside [`crate::tools::CharlotteTool::AgenticCrawl`]'s wall-clock crawl timeout,
/// and credential scope / origin preflight runs before prompting the user.
pub struct GatedCharlotteTool {
    inner: Arc<dyn Tool>,
    approval: Arc<dyn HostApproval>,
    lockbox: Option<Arc<Lockbox>>,
    scope: CredentialScope,
}

impl GatedCharlotteTool {
    /// Wrap a Charlotte tool with scoped credential preflight and host approval.
    #[must_use]
    pub fn new(
        inner: Arc<dyn Tool>,
        approval: Arc<dyn HostApproval>,
        lockbox: Option<Arc<Lockbox>>,
        scope: CredentialScope,
    ) -> Self {
        Self {
            inner,
            approval,
            lockbox,
            scope,
        }
    }

    /// Convert this gated wrapper into a shared [`Tool`] trait object.
    #[must_use]
    pub fn into_arc(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }

    fn preflight(&self, tool_name: &str, arguments: &Value) -> Result<(), ToolInvokeError> {
        if tool_name != CharlotteTool::AgenticCrawl.name() {
            return Ok(());
        }
        let Some(raw_alias) = arguments.get("credential") else {
            return Ok(());
        };
        let Some(alias) = raw_alias
            .as_str()
            .filter(|alias| is_valid_credential_alias(alias))
        else {
            return Err(ToolInvokeError::handler(
                "credential alias is invalid",
                None,
            ));
        };
        if arguments.get("input_text").is_some() || arguments.get("inputs").is_some() {
            return Err(ToolInvokeError::handler(
                "credential cannot be combined with plaintext inputs",
                None,
            ));
        }
        if !self.scope.allows(alias) {
            return Err(credential_unavailable());
        }
        let lockbox = self.lockbox.as_deref().ok_or_else(credential_unavailable)?;
        let seed_url = arguments
            .get("seed_url")
            .and_then(Value::as_str)
            .ok_or_else(credential_unavailable)?;
        if sanitized_web_origin(seed_url).is_none() {
            return Err(credential_unavailable());
        }
        let requested_origin = Origin::from_url(seed_url).map_err(|_| credential_unavailable())?;
        let matches_profile = lockbox.aliases().into_iter().any(|summary| {
            summary.alias.as_str() == alias && summary.origins.contains(&requested_origin)
        });
        if !matches_profile {
            return Err(credential_unavailable());
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for GatedCharlotteTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn context_policy(&self) -> ToolContextPolicy {
        self.inner.context_policy()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let tool_name = self.inner.spec().name;
        self.preflight(&tool_name, &arguments)?;

        let request = HostApprovalRequest::for_tool(&tool_name, &arguments);
        match self.approval.approve(request).await {
            Ok(true) => self.inner.call(arguments).await,
            Ok(false) => Err(ToolInvokeError::handler(
                "tool call denied",
                Some("approval".to_string()),
            )),
            Err(reason) => Err(ToolInvokeError::handler(
                sanitize_error_reason(&reason),
                Some("approval".to_string()),
            )),
        }
    }
}

/// Standard SuperGlue retry policy for host-registered Charlotte tools (`OnToolError::Skip`).
#[must_use]
pub fn charlotte_retry_policy() -> ToolRetryPolicy {
    ToolRetryPolicy {
        on_error: OnToolError::Skip,
    }
}

/// Return the interactive approval explanation for a Charlotte tool name, if known.
#[must_use]
pub fn approval_detail_for_tool(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "web_search" => Some(WEB_SEARCH_APPROVAL_DETAIL),
        "fetch" => Some(FETCH_APPROVAL_DETAIL),
        "agentic_crawl" => Some(AGENTIC_CRAWL_APPROVAL_DETAIL),
        _ => None,
    }
}

/// Derive the bounded approval resource string for a Charlotte tool call.
#[must_use]
pub fn approval_resource_for_tool(tool_name: &str, arguments: &Value) -> String {
    match tool_name {
        "web_search" => web_search_approval_resource(arguments),
        "fetch" => fetch_approval_resource(arguments),
        "agentic_crawl" => agentic_approval_resource(arguments),
        _ => "tool://default".to_string(),
    }
}

/// Project non-sensitive audit arguments for a Charlotte tool call.
#[must_use]
pub fn audit_arguments_for_tool(tool_name: &str, arguments: &Value) -> Value {
    match tool_name {
        "web_search" => web_search_audit_arguments(arguments),
        "fetch" => fetch_audit_arguments(arguments),
        "agentic_crawl" => agentic_audit_arguments(arguments),
        _ => Value::Object(serde_json::Map::new()),
    }
}

/// Approval resource for `web_search` (`"search://public-web"`).
#[must_use]
pub fn web_search_approval_resource(_arguments: &Value) -> String {
    PUBLIC_WEB_RESOURCE.to_string()
}

/// Redacted audit arguments for `web_search` (`{"max_results": ...}`, omitting `query`).
#[must_use]
pub fn web_search_audit_arguments(arguments: &Value) -> Value {
    project_keys(arguments, &["max_results"])
}

/// Approval resource for `fetch` (`"search://public-web"`).
#[must_use]
pub fn fetch_approval_resource(_arguments: &Value) -> String {
    PUBLIC_WEB_RESOURCE.to_string()
}

/// Redacted audit arguments for `fetch` (`{"host": ...}`, omitting path, query, and userinfo).
#[must_use]
pub fn fetch_audit_arguments(arguments: &Value) -> Value {
    let mut projected = serde_json::Map::new();
    if let Some(raw) = arguments.get("url").and_then(Value::as_str)
        && let Ok(parsed) = Url::parse(raw)
        && let Some(host) = parsed.host_str()
    {
        projected.insert("host".to_string(), Value::String(host.to_string()));
    }
    Value::Object(projected)
}

/// Approval resource for `agentic_crawl`.
///
/// Produces `credential://<alias>/<origin>` when `"credential"` is present and valid,
/// `"credential://invalid"` when `"credential"` is malformed, or `<origin>` otherwise
/// (`"tool://default"` if `seed_url` is missing or invalid).
#[must_use]
pub fn agentic_approval_resource(arguments: &Value) -> String {
    let origin = arguments
        .get("seed_url")
        .and_then(Value::as_str)
        .and_then(sanitized_web_origin)
        .unwrap_or_else(|| "tool://default".to_string());
    let Some(raw_alias) = arguments.get("credential") else {
        return origin;
    };
    let Some(alias) = raw_alias
        .as_str()
        .filter(|alias| is_valid_credential_alias(alias))
    else {
        return "credential://invalid".to_string();
    };
    format!("credential://{alias}/{origin}")
}

/// Redacted audit arguments for `agentic_crawl` (omitting `goal`, `input_text`, `inputs`, and `extract_schema`).
#[must_use]
pub fn agentic_audit_arguments(arguments: &Value) -> Value {
    let mut projected = project_keys(
        arguments,
        &[
            "max_pages",
            "max_steps",
            "same_origin_only",
            "respect_robots",
        ],
    );
    if let Some(projected_map) = projected.as_object_mut() {
        if let Some(alias) = arguments
            .get("credential")
            .and_then(Value::as_str)
            .filter(|alias| is_valid_credential_alias(alias))
        {
            projected_map.insert("credential".to_string(), Value::String(alias.to_string()));
        }
        if let Some(origin) = arguments
            .get("seed_url")
            .and_then(Value::as_str)
            .and_then(sanitized_web_origin)
        {
            projected_map.insert("seed_origin".to_string(), Value::String(origin));
        }
    }
    projected
}

/// Extract an ASCII HTTP/HTTPS origin without userinfo, preserving URL origin serialization
/// (including trailing-dot hosts such as `"https://example.com."`).
#[must_use]
pub fn sanitized_web_origin(value: &str) -> Option<String> {
    let parsed = Url::parse(value).ok()?;
    if !matches!(parsed.scheme(), "http" | "https")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return None;
    }
    parsed.host_str()?;
    Some(parsed.origin().ascii_serialization())
}

/// Parse and validate a comma-separated host suffix allowlist (at most 32 hosts, no wildcards).
pub fn parse_allowlist(value: &str) -> Result<Vec<String>, &'static str> {
    let mut hosts = Vec::new();
    for raw in value.split(',') {
        let host = raw.trim().trim_start_matches('.').to_ascii_lowercase();
        if host.is_empty() {
            continue;
        }
        if hosts.len() >= 32
            || host.len() > 253
            || host.starts_with('-')
            || host.ends_with('-')
            || host.chars().any(|character| {
                !character.is_ascii_alphanumeric() && character != '.' && character != '-'
            })
        {
            return Err("Charlotte host allow-list is invalid");
        }
        if !hosts.contains(&host) {
            hosts.push(host);
        }
    }
    Ok(hosts)
}

/// Read an optional environment variable bounded by byte length and free of control characters.
pub fn bounded_environment(name: &str, max_bytes: usize) -> Result<Option<String>, &'static str> {
    match std::env::var(name) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) if value.len() <= max_bytes && !value.chars().any(char::is_control) => {
            Ok(Some(value))
        }
        Ok(_) => Err("Charlotte environment setting is invalid"),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err("Charlotte environment setting is invalid"),
    }
}

/// Load the optional [`Lockbox`] configured by `CHARLOTTE_CREDENTIALS_FILE` using [`OnePasswordCli`].
pub fn load_lockbox() -> Result<Option<Arc<Lockbox>>, &'static str> {
    let Some(path) = bounded_environment("CHARLOTTE_CREDENTIALS_FILE", 4_096)? else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    load_lockbox_from_path(&path).map(Some)
}

/// Load a [`Lockbox`] from `path` registered with the default [`OnePasswordCli`] provider.
pub fn load_lockbox_from_path(path: &Path) -> Result<Arc<Lockbox>, &'static str> {
    let mut providers = ProviderRegistry::new();
    providers
        .register(Arc::new(OnePasswordCli::default()))
        .map_err(|_| "Charlotte credential provider is unavailable")?;
    let lockbox =
        Lockbox::load(path, providers).map_err(|_| "Charlotte credential profiles are invalid")?;
    Ok(Arc::new(lockbox))
}

/// Resolve an origin-bound [`Lockbox`] credential profile with [`CredentialScope::All`].
///
/// All [`lockbox::LockboxError`] details are redacted to `"credential_unavailable"`.
pub async fn resolve_lockbox_credential(
    lockbox: Option<&Lockbox>,
    alias: &str,
    seed_url: &str,
) -> Result<ResolvedCredential, ToolInvokeError> {
    resolve_scoped_credential(lockbox, &CredentialScope::All, alias, seed_url).await
}

/// Resolve an origin-bound [`Lockbox`] credential profile gated by `scope`.
///
/// Checks `scope` before calling [`Lockbox::resolve`] and redacts all failure details
/// to `"credential_unavailable"`.
pub async fn resolve_scoped_credential(
    lockbox: Option<&Lockbox>,
    scope: &CredentialScope,
    alias: &str,
    seed_url: &str,
) -> Result<ResolvedCredential, ToolInvokeError> {
    if !scope.allows(alias) {
        return Err(credential_unavailable());
    }
    let lockbox = lockbox.ok_or_else(credential_unavailable)?;
    let requested_origin = sanitized_web_origin(seed_url).ok_or_else(credential_unavailable)?;
    let resolved = lockbox
        .resolve(alias, seed_url)
        .await
        .map_err(|_| credential_unavailable())?;
    Ok(ResolvedCredential {
        origins: vec![requested_origin],
        inputs: resolved
            .fields
            .into_iter()
            .map(|(name, value)| {
                CrawlInput::new(name.to_string(), value.expose_secret().to_string())
            })
            .collect(),
    })
}

/// Build the Charlotte system-prompt block listing only credential aliases permitted by `scope`.
#[must_use]
pub fn ready_prompt(lockbox: Option<&Lockbox>, scope: &CredentialScope) -> String {
    let mut prompt = READY_PROMPT.to_string();
    let Some(lockbox) = lockbox else {
        return prompt;
    };
    let aliases: Vec<_> = lockbox
        .aliases()
        .into_iter()
        .filter(|summary| !summary.origins.is_empty() && scope.allows(summary.alias.as_str()))
        .collect();
    if aliases.is_empty() {
        return prompt;
    }
    prompt.push_str("\n- Available credential profiles:");
    for summary in aliases {
        prompt.push_str("\n  - `");
        prompt.push_str(summary.alias.as_str());
        prompt.push_str("`: ");
        for (index, origin) in summary.origins.iter().enumerate() {
            if index > 0 {
                prompt.push_str(", ");
            }
            prompt.push_str(origin.as_str());
        }
    }
    prompt
}

fn is_valid_credential_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias.len() <= MAX_CREDENTIAL_ALIAS_BYTES
        && alias
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn project_keys(arguments: &Value, keys: &[&str]) -> Value {
    let mut projected = serde_json::Map::new();
    if let Some(arguments) = arguments.as_object() {
        for key in keys {
            if let Some(value) = arguments.get(*key) {
                projected.insert((*key).to_string(), value.clone());
            }
        }
    }
    Value::Object(projected)
}

fn credential_unavailable() -> ToolInvokeError {
    ToolInvokeError::handler(
        "credentials are unavailable",
        Some("credential_unavailable".to_string()),
    )
}

fn sanitize_error_reason(reason: &str) -> String {
    let trimmed = reason.trim();
    if trimmed.is_empty() || trimmed.len() > 256 || trimmed.chars().any(char::is_control) {
        "tool call blocked by host approval".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use lockbox::MemoryProvider;
    use serde_json::json;

    use super::*;

    fn test_lockbox() -> (PathBuf, Arc<Lockbox>, MemoryProvider) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);

        let directory = std::env::temp_dir().join(format!(
            "charlotte-host-lockbox-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(fs::create_dir_all(&directory).is_ok());
        let path = directory.join("credentials.toml");
        assert!(
            fs::write(
                &path,
                r#"
[[profile]]
alias = "github"
provider = "onepassword"
origins = ["https://github.com", "https://gist.github.com"]
[profile.fields]
password = "gh-password-ref"
username = "gh-username-ref"

[[profile]]
alias = "stripe"
provider = "onepassword"
origins = ["https://dashboard.stripe.com"]
[profile.fields]
password = "stripe-password-ref"
username = "stripe-username-ref"
"#,
            )
            .is_ok()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert!(fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).is_ok());
        }
        let provider = MemoryProvider::new();
        provider.insert("gh-password-ref", "marker-gh-pass");
        provider.insert("gh-username-ref", "marker-gh-user");
        provider.insert("stripe-password-ref", "marker-stripe-pass");
        provider.insert("stripe-username-ref", "marker-stripe-user");
        let mut providers = ProviderRegistry::new();
        assert!(providers.register(Arc::new(provider.clone())).is_ok());
        let lockbox = Lockbox::load(&path, providers).expect("load test lockbox");
        (directory, Arc::new(lockbox), provider)
    }

    struct RecordingTool {
        name: &'static str,
        calls: AtomicUsize,
        available: AtomicBool,
    }

    impl RecordingTool {
        fn new(name: &'static str) -> Arc<Self> {
            Arc::new(Self {
                name,
                calls: AtomicUsize::new(0),
                available: AtomicBool::new(true),
            })
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Tool for RecordingTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(self.name, json!({"type": "object"}))
        }

        fn is_available(&self) -> bool {
            self.available.load(Ordering::Acquire)
        }

        async fn call(&self, _arguments: Value) -> Result<Value, ToolInvokeError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"ok": true}))
        }
    }

    struct RecordingApproval {
        decision: Result<bool, String>,
        requests: Mutex<Vec<HostApprovalRequest>>,
    }

    impl RecordingApproval {
        fn new(decision: Result<bool, String>) -> Arc<Self> {
            Arc::new(Self {
                decision,
                requests: Mutex::new(Vec::new()),
            })
        }

        fn recorded(&self) -> Vec<HostApprovalRequest> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    #[async_trait]
    impl HostApproval for RecordingApproval {
        async fn approve(&self, request: HostApprovalRequest) -> Result<bool, String> {
            self.requests.lock().expect("requests lock").push(request);
            self.decision.clone()
        }
    }

    #[test]
    fn selected_tools_exposes_search_fetch_and_agentic_crawl() {
        let names: Vec<_> = SELECTED_TOOLS.iter().map(|tool| tool.name()).collect();
        assert_eq!(names, vec!["web_search", "fetch", "agentic_crawl"]);
    }

    #[test]
    fn allowlist_scope_filters_ready_prompt_aliases() {
        let (directory, lockbox, _) = test_lockbox();

        let all_prompt = ready_prompt(Some(&lockbox), &CredentialScope::All);
        assert!(all_prompt.contains("`github`: https://gist.github.com, https://github.com"));
        assert!(all_prompt.contains("`stripe`: https://dashboard.stripe.com"));
        assert!(!all_prompt.contains("gh-password-ref"));
        assert!(!all_prompt.contains("onepassword"));

        let github_only = CredentialScope::from_aliases(["github"]);
        let scoped_prompt = ready_prompt(Some(&lockbox), &github_only);
        assert!(scoped_prompt.contains("`github`: https://gist.github.com, https://github.com"));
        assert!(!scoped_prompt.contains("stripe"));
        assert!(!scoped_prompt.contains("dashboard.stripe.com"));

        let empty_prompt = ready_prompt(Some(&lockbox), &CredentialScope::none());
        assert_eq!(empty_prompt, READY_PROMPT);
        assert!(!empty_prompt.contains("Available credential profiles"));

        let _ = fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn scoped_credential_rejects_unassigned_alias_before_reading_provider() {
        let (directory, lockbox, provider) = test_lockbox();
        let github_only = CredentialScope::from_aliases(["github"]);

        let denied = resolve_scoped_credential(
            Some(&lockbox),
            &github_only,
            "stripe",
            "https://dashboard.stripe.com/login",
        )
        .await
        .expect_err("unassigned alias must fail");
        match denied {
            ToolInvokeError::HandlerFailed { message, code } => {
                assert_eq!(message, "credentials are unavailable");
                assert_eq!(code.as_deref(), Some("credential_unavailable"));
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(provider.read_count(), 0);

        let allowed = resolve_scoped_credential(
            Some(&lockbox),
            &github_only,
            "github",
            "https://github.com./login",
        )
        .await
        .expect("assigned alias resolves on trailing-dot origin");
        assert_eq!(allowed.origins, vec!["https://github.com."]);
        assert_eq!(allowed.inputs.len(), 2);
        let debug = format!("{allowed:?}");
        assert!(!debug.contains("marker-gh-pass"));
        assert!(!debug.contains("marker-gh-user"));
        assert_eq!(provider.read_count(), 2);

        let wrong_origin =
            resolve_lockbox_credential(Some(&lockbox), "github", "https://evil.example/login")
                .await
                .expect_err("wrong origin must fail");
        assert!(matches!(
            wrong_origin,
            ToolInvokeError::HandlerFailed { ref code, .. }
                if code.as_deref() == Some("credential_unavailable")
        ));
        assert_eq!(provider.read_count(), 2);

        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn agentic_approval_resource_handles_public_trailing_dot_credential_and_invalid_urls() {
        assert_eq!(
            agentic_approval_resource(&json!({
                "seed_url": "https://example.com/private?token=marker"
            })),
            "https://example.com"
        );
        assert_eq!(
            agentic_approval_resource(&json!({
                "seed_url": "https://example.com./login?token=marker"
            })),
            "https://example.com."
        );
        assert_eq!(
            agentic_approval_resource(&json!({
                "seed_url": "https://github.com/login?token=marker",
                "credential": "github"
            })),
            "credential://github/https://github.com"
        );
        assert_eq!(
            agentic_approval_resource(&json!({
                "seed_url": "https://github.com/login",
                "credential": "Invalid_Alias!"
            })),
            "credential://invalid"
        );
        assert_eq!(
            agentic_approval_resource(&json!({
                "seed_url": "ftp://example.com/file"
            })),
            "tool://default"
        );
        assert_eq!(
            agentic_approval_resource(&json!({
                "seed_url": "https://user:secret@example.com/path"
            })),
            "tool://default"
        );
        assert_eq!(
            agentic_approval_resource(&json!({"seed_url": "not a url"})),
            "tool://default"
        );
    }

    #[test]
    fn audit_argument_projectors_strip_queries_goals_inputs_and_schemas() {
        assert_eq!(
            web_search_approval_resource(&json!({"query": "marker-query"})),
            "search://public-web"
        );
        let search_audit = web_search_audit_arguments(&json!({
            "query": "marker-secret-query",
            "max_results": 5
        }));
        assert_eq!(search_audit, json!({"max_results": 5}));
        assert!(!search_audit.to_string().contains("marker"));

        assert_eq!(
            fetch_approval_resource(&json!({"url": "https://example.com/path?token=marker"})),
            "search://public-web"
        );
        let fetch_audit = fetch_audit_arguments(&json!({
            "url": "https://example.com/secret-path?token=marker#frag"
        }));
        assert_eq!(fetch_audit, json!({"host": "example.com"}));
        assert!(!fetch_audit.to_string().contains("marker"));
        assert!(!fetch_audit.to_string().contains("secret-path"));

        let crawl_audit = agentic_audit_arguments(&json!({
            "seed_url": "https://github.com/login?token=marker-token",
            "goal": "marker-goal",
            "input_text": "marker-input-text",
            "inputs": {"password": "marker-password"},
            "extract_schema": {"properties": {"secret": {"description": "marker-schema"}}},
            "credential": "github",
            "max_pages": 4,
            "max_steps": 12,
            "same_origin_only": true,
            "respect_robots": true
        }));
        assert_eq!(
            crawl_audit,
            json!({
                "max_pages": 4,
                "max_steps": 12,
                "same_origin_only": true,
                "respect_robots": true,
                "credential": "github",
                "seed_origin": "https://github.com"
            })
        );
        assert!(!crawl_audit.to_string().contains("marker"));

        let invalid_cred_audit = agentic_audit_arguments(&json!({
            "seed_url": "https://github.com/login",
            "goal": "sign in",
            "credential": "MarkerSecretStuffedIntoAlias!"
        }));
        assert!(!invalid_cred_audit.to_string().contains("MarkerSecret"));
    }

    #[test]
    fn parse_allowlist_validates_and_normalizes_hosts() {
        assert_eq!(
            parse_allowlist(" .Example.COM , docs.example.com, example.com ").unwrap(),
            vec!["example.com".to_string(), "docs.example.com".to_string()]
        );
        assert!(parse_allowlist("*.example.com").is_err());
        assert!(parse_allowlist("-bad.example.com").is_err());
        assert!(parse_allowlist(&"a".repeat(254)).is_err());
    }

    #[tokio::test]
    async fn gated_tool_preflight_rejects_unassigned_or_mismatched_credential_without_prompting() {
        let (directory, lockbox, provider) = test_lockbox();
        let inner = RecordingTool::new("agentic_crawl");
        let approval = RecordingApproval::new(Ok(true));
        let gated = GatedCharlotteTool::new(
            Arc::clone(&inner) as Arc<dyn Tool>,
            Arc::clone(&approval) as Arc<dyn HostApproval>,
            Some(Arc::clone(&lockbox)),
            CredentialScope::from_aliases(["github"]),
        );

        let unauthorized = gated
            .call(json!({
                "seed_url": "https://dashboard.stripe.com/login",
                "goal": "check payouts",
                "credential": "stripe"
            }))
            .await
            .expect_err("unassigned alias must fail preflight");
        assert!(matches!(
            unauthorized,
            ToolInvokeError::HandlerFailed { ref code, .. }
                if code.as_deref() == Some("credential_unavailable")
        ));
        assert!(approval.recorded().is_empty());
        assert_eq!(inner.call_count(), 0);
        assert_eq!(provider.read_count(), 0);

        let hallucinated = gated
            .call(json!({
                "seed_url": "https://github.com/login",
                "goal": "sign in",
                "credential": "nonexistent-profile"
            }))
            .await
            .expect_err("unknown alias must fail preflight");
        assert!(matches!(
            hallucinated,
            ToolInvokeError::HandlerFailed { ref code, .. }
                if code.as_deref() == Some("credential_unavailable")
        ));
        assert!(approval.recorded().is_empty());
        assert_eq!(inner.call_count(), 0);

        let wrong_origin = gated
            .call(json!({
                "seed_url": "https://evil.example/login",
                "goal": "sign in",
                "credential": "github"
            }))
            .await
            .expect_err("wrong origin must fail preflight");
        assert!(matches!(
            wrong_origin,
            ToolInvokeError::HandlerFailed { ref code, .. }
                if code.as_deref() == Some("credential_unavailable")
        ));
        assert!(approval.recorded().is_empty());
        assert_eq!(inner.call_count(), 0);

        let _ = fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn gated_tool_enforces_host_approval_allow_and_deny_before_execution() {
        let (directory, lockbox, _) = test_lockbox();
        let inner = RecordingTool::new("agentic_crawl");
        let allow = RecordingApproval::new(Ok(true));
        let gated_allow = GatedCharlotteTool::new(
            Arc::clone(&inner) as Arc<dyn Tool>,
            Arc::clone(&allow) as Arc<dyn HostApproval>,
            Some(Arc::clone(&lockbox)),
            CredentialScope::from_aliases(["github"]),
        );

        let ok = gated_allow
            .call(json!({
                "seed_url": "https://github.com/login?token=marker",
                "goal": "marker-goal",
                "credential": "github",
                "max_pages": 3
            }))
            .await
            .expect("approved call succeeds");
        assert_eq!(ok, json!({"ok": true}));
        assert_eq!(inner.call_count(), 1);
        let recorded = allow.recorded();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].tool_name, "agentic_crawl");
        assert_eq!(
            recorded[0].resource,
            "credential://github/https://github.com"
        );
        assert!(recorded[0].requires_exact_rule);
        assert_eq!(recorded[0].detail, Some(AGENTIC_CRAWL_APPROVAL_DETAIL));
        assert!(!recorded[0].audit_arguments.to_string().contains("marker"));

        let deny = RecordingApproval::new(Ok(false));
        let gated_deny = GatedCharlotteTool::new(
            Arc::clone(&inner) as Arc<dyn Tool>,
            Arc::clone(&deny) as Arc<dyn HostApproval>,
            Some(Arc::clone(&lockbox)),
            CredentialScope::from_aliases(["github"]),
        );
        let denied = gated_deny
            .call(json!({
                "seed_url": "https://github.com/login",
                "goal": "sign in",
                "credential": "github"
            }))
            .await
            .expect_err("denied approval must block execution");
        assert!(matches!(
            denied,
            ToolInvokeError::HandlerFailed { ref code, .. }
                if code.as_deref() == Some("approval")
        ));
        assert_eq!(inner.call_count(), 1);
        assert_eq!(deny.recorded().len(), 1);

        let err_approval = RecordingApproval::new(Err("unattended rule missing".to_string()));
        let fetch_inner = RecordingTool::new("fetch");
        let gated_fetch = GatedCharlotteTool::new(
            Arc::clone(&fetch_inner) as Arc<dyn Tool>,
            Arc::clone(&err_approval) as Arc<dyn HostApproval>,
            None,
            CredentialScope::none(),
        );
        let blocked = gated_fetch
            .call(json!({"url": "https://example.com/docs?secret=marker"}))
            .await
            .expect_err("errored approval must block execution");
        match blocked {
            ToolInvokeError::HandlerFailed { message, code } => {
                assert_eq!(message, "unattended rule missing");
                assert_eq!(code.as_deref(), Some("approval"));
            }
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(fetch_inner.call_count(), 0);
        let fetch_requests = err_approval.recorded();
        assert_eq!(fetch_requests.len(), 1);
        assert_eq!(fetch_requests[0].resource, "search://public-web");
        assert_eq!(
            fetch_requests[0].audit_arguments,
            json!({"host": "example.com"})
        );

        let _ = fs::remove_dir_all(directory);
    }
}
