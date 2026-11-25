//! Process-local gates for crawl load.
//!
//! Admission fails at once when every slot is busy. A deadline stops the caller
//! from waiting, and the slot stays held until the crawl task ends. Memory and
//! task soft limits refuse new work. Each host has its own concurrency cap.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use url::Url;

use crate::browser::resilient_pool::ResilientBrowserPool;
use crate::browser::url_policy::UrlPolicy;
use crate::browser::{BrowserConfig, NavigateOptions};
use crate::error::{CharlotteError, OverloadReason};

/// Default concurrent browser pages in one process.
pub const DEFAULT_MAX_PAGES: usize = 8;
/// Default Chromium processes for an agent tool pool.
pub const AGENT_BROWSER_POOL_SIZE: usize = 1;
/// Default Chromium processes for the crawl service.
pub const SERVER_BROWSER_POOL_SIZE: usize = 2;
/// Default in-flight crawls for one host.
pub const DEFAULT_PER_DOMAIN_MAX_ACTIVE: usize = 2;
/// Default time a caller waits for one crawl.
pub const DEFAULT_CRAWL_DEADLINE: Duration = Duration::from_secs(80);
/// Subtract this from a client timeout before waiting.
pub const DEFAULT_CANCEL_MARGIN: Duration = Duration::from_secs(5);
/// Soft memory limit before new crawls are refused.
pub const DEFAULT_MEMORY_SOFT_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Soft cgroup task count before new crawls are refused.
pub const DEFAULT_TASK_SOFT_LIMIT: u64 = 2_500;
/// Suggested delay on an overload response.
pub const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(15);
const MAX_TRACKED_DOMAINS: usize = 1_024;

/// Limits for [`CrawlRuntime`].
#[derive(Debug, Clone)]
pub struct CrawlCapacityConfig {
    /// Chromium processes in the pool.
    pub browser_pool_size: usize,
    /// Browser pages open at once, across the pool.
    pub max_pages: usize,
    /// Crawls that may hold a slot at once.
    pub admission_max_active: usize,
    /// Crawls that may run at once for one host.
    pub per_domain_max_active: usize,
    /// How long a caller waits.
    pub crawl_deadline: Duration,
    /// Time reserved so a client can observe a timeout.
    pub cancel_margin: Duration,
    /// Refuse new crawls at or above this memory use.
    pub memory_soft_limit: u64,
    /// Refuse new crawls at or above this cgroup task count.
    pub task_soft_limit: u64,
    /// Suggested wait after a refusal.
    pub retry_after: Duration,
}

impl CrawlCapacityConfig {
    /// Agent tool defaults. The pool size is one browser.
    #[must_use]
    pub fn agent() -> Self {
        Self::with_pool_size(AGENT_BROWSER_POOL_SIZE)
    }

    /// Crawl service defaults. The pool size is two browsers.
    #[must_use]
    pub fn server() -> Self {
        Self::with_pool_size(SERVER_BROWSER_POOL_SIZE)
    }

    fn with_pool_size(browser_pool_size: usize) -> Self {
        Self {
            browser_pool_size,
            max_pages: DEFAULT_MAX_PAGES,
            admission_max_active: DEFAULT_MAX_PAGES,
            per_domain_max_active: DEFAULT_PER_DOMAIN_MAX_ACTIVE,
            crawl_deadline: DEFAULT_CRAWL_DEADLINE,
            cancel_margin: DEFAULT_CANCEL_MARGIN,
            memory_soft_limit: DEFAULT_MEMORY_SOFT_LIMIT_BYTES,
            task_soft_limit: DEFAULT_TASK_SOFT_LIMIT,
            retry_after: DEFAULT_RETRY_AFTER,
        }
    }

    /// Replace set limits with `CHARLOTTE_*` environment variables.
    pub fn apply_env(mut self) -> Result<Self, CharlotteError> {
        if let Some(value) = env_usize("CHARLOTTE_BROWSER_POOL_SIZE")? {
            self.browser_pool_size = value;
        }
        let max_pages = env_usize("CHARLOTTE_MAX_PAGES")?;
        if let Some(value) = max_pages {
            self.max_pages = value;
        }
        match env_usize("CHARLOTTE_ADMISSION_MAX_ACTIVE")? {
            Some(value) => self.admission_max_active = value,
            None if max_pages.is_some() => self.admission_max_active = self.max_pages,
            None => {}
        }
        if let Some(value) = env_usize("CHARLOTTE_PER_DOMAIN_MAX_ACTIVE")? {
            self.per_domain_max_active = value;
        }
        if let Some(secs) = env_u64("CHARLOTTE_CRAWL_DEADLINE_SECS")? {
            self.crawl_deadline = Duration::from_secs(secs);
        }
        if let Some(secs) = env_u64("CHARLOTTE_CRAWL_CANCEL_MARGIN_SECS")? {
            self.cancel_margin = Duration::from_secs(secs);
        }
        if let Some(gib) = env_u64("CHARLOTTE_MEMORY_SOFT_LIMIT_GIB")? {
            self.memory_soft_limit = gib.saturating_mul(1024 * 1024 * 1024);
        }
        if let Some(value) = env_u64("CHARLOTTE_TASK_SOFT_LIMIT")? {
            self.task_soft_limit = value;
        }
        self.validate()?;
        Ok(self)
    }

    /// Reject a pool larger than the page cap, or a zero limit.
    pub fn validate(&self) -> Result<(), CharlotteError> {
        if self.browser_pool_size == 0
            || self.max_pages == 0
            || self.admission_max_active == 0
            || self.per_domain_max_active == 0
            || self.task_soft_limit == 0
            || self.memory_soft_limit == 0
        {
            return Err(CharlotteError::msg(
                "crawl capacity limits must be greater than zero",
            ));
        }
        if self.browser_pool_size > self.max_pages {
            return Err(CharlotteError::msg(
                "browser pool size cannot exceed the page limit",
            ));
        }
        if self.crawl_deadline.is_zero() {
            return Err(CharlotteError::msg(
                "crawl deadline must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// One fetched page for the crawl service.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "serve"), allow(dead_code))]
pub(crate) struct CrawlPage {
    /// URL after redirects when the browser reports one.
    pub url: String,
    /// Page body as Markdown.
    pub markdown: Option<String>,
    /// The navigation finished with a usable body.
    pub success: bool,
    /// HTTP status from the navigation, when the browser reported one.
    pub status_code: Option<u16>,
    /// Why the page was not usable.
    pub error_message: Option<String>,
}

#[cfg_attr(not(feature = "serve"), allow(dead_code))]
impl CrawlPage {
    fn from_value(requested_url: &str, value: &Value) -> Self {
        let success = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
        let url = value
            .get("final_url")
            .and_then(Value::as_str)
            .unwrap_or(requested_url)
            .to_string();
        let markdown = value
            .get("markdown")
            .and_then(Value::as_str)
            .map(str::to_string);
        let status_code = value
            .get("status")
            .and_then(Value::as_u64)
            .and_then(|code| u16::try_from(code).ok());
        let error_message = if success {
            None
        } else {
            Some(
                value
                    .get("message")
                    .or_else(|| value.get("error"))
                    .and_then(Value::as_str)
                    .unwrap_or("scrape failed")
                    .to_string(),
            )
        };
        Self {
            url,
            markdown,
            success,
            status_code,
            error_message,
        }
    }
}

/// Options for one browser scrape through [`CrawlRuntime`].
#[derive(Debug, Clone)]
pub(crate) struct CrawlScrapeOptions {
    /// Optional section focus passed to Markdown conversion.
    pub focus: Option<String>,
    /// HTML byte budget before conversion.
    pub max_html_bytes: usize,
    /// Markdown character budget.
    pub max_markdown_chars: usize,
    /// Navigation timeouts.
    pub navigate: NavigateOptions,
}

impl CrawlScrapeOptions {
    /// Scrape options taken from a browser config.
    #[must_use]
    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    pub fn from_browser(config: &BrowserConfig) -> Self {
        Self {
            focus: None,
            max_html_bytes: config.max_html_bytes,
            max_markdown_chars: config.max_markdown_chars,
            navigate: NavigateOptions::for_scrape(),
        }
    }
}

/// What a caller uses to run one admitted crawl.
#[async_trait]
#[cfg_attr(not(feature = "serve"), allow(dead_code))]
pub(crate) trait CrawlExecutor: Send + Sync {
    /// Browsers that can accept a page.
    fn ready_browser_count(&self) -> usize;
    /// Configured Chromium process count.
    fn browser_pool_size(&self) -> usize;
    /// Caller wait limit before a client timeout is applied.
    fn server_deadline(&self) -> Duration;
    /// Reserved time subtracted from a client timeout.
    fn cancel_margin(&self) -> Duration;
    /// Fetch one URL and return Markdown.
    async fn scrape(&self, url: &str, deadline: Duration) -> Result<CrawlPage, CharlotteError>;
}

/// Admission, resource limits, a per-host cap, and a restartable browser pool.
pub struct CrawlRuntime {
    config: CrawlCapacityConfig,
    admission: AdmissionController,
    domains: DomainLimiter,
    probe: Arc<dyn ResourceProbe>,
    policy: UrlPolicy,
    pool: ResilientBrowserPool,
    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    browser: BrowserConfig,
}

impl CrawlRuntime {
    /// Launch the browser pool and the load gates.
    pub async fn start(
        config: CrawlCapacityConfig,
        browser: BrowserConfig,
    ) -> Result<Self, CharlotteError> {
        config.validate()?;
        let pool = ResilientBrowserPool::start(
            config.browser_pool_size,
            config.max_pages,
            browser.clone(),
        )
        .await?;
        Ok(Self::from_parts(
            config,
            browser,
            Arc::new(CgroupProbe),
            pool,
        ))
    }

    fn from_parts(
        config: CrawlCapacityConfig,
        browser: BrowserConfig,
        probe: Arc<dyn ResourceProbe>,
        pool: ResilientBrowserPool,
    ) -> Self {
        let admission = AdmissionController::new(config.admission_max_active, config.retry_after);
        let domains = DomainLimiter::new(config.per_domain_max_active);
        Self {
            policy: browser.url_policy.clone(),
            config,
            admission,
            domains,
            probe,
            pool,
            browser,
        }
    }

    /// Scrape one URL. The returned JSON matches `scrape_many` rows.
    pub(crate) async fn scrape_value(
        &self,
        url: &str,
        deadline: Duration,
        options: CrawlScrapeOptions,
    ) -> Result<Value, CharlotteError> {
        ensure_headroom(self.probe.as_ref(), &self.config)?;
        let policy = self.policy.clone();
        let domains = self.domains.clone();
        let pool = self.pool.clone();
        let url = url.to_string();
        run_admitted(&self.admission, deadline, async move {
            tracing::info!(url = %url, "crawl admitted");
            policy.check_resolved(&url).await?;
            tracing::info!(url = %url, "crawl waiting for a domain slot");
            let _domain = domains.acquire(&url).await?;
            tracing::info!(url = %url, "crawl starting browser scrape");
            let scraped = pool
                .scrape(
                    &url,
                    options.navigate,
                    options.focus.as_deref(),
                    options.max_html_bytes,
                    options.max_markdown_chars,
                )
                .await;
            match &scraped {
                Ok(_) => tracing::info!(url = %url, "browser scrape finished"),
                Err(error) => tracing::warn!(url = %url, %error, "browser scrape failed"),
            }
            scraped
        })
        .await
    }

    /// Close every Chromium process and the Playwright driver.
    pub async fn close(&self) -> Result<(), CharlotteError> {
        self.pool.close().await
    }

    /// Caller wait configured for this runtime.
    #[must_use]
    pub(crate) fn deadline(&self) -> Duration {
        self.config.crawl_deadline
    }
}

#[cfg_attr(not(feature = "serve"), allow(dead_code))]
#[async_trait]
impl CrawlExecutor for CrawlRuntime {
    fn ready_browser_count(&self) -> usize {
        self.pool.ready_browser_count()
    }

    fn browser_pool_size(&self) -> usize {
        self.config.browser_pool_size
    }

    fn server_deadline(&self) -> Duration {
        self.config.crawl_deadline
    }

    fn cancel_margin(&self) -> Duration {
        self.config.cancel_margin
    }

    async fn scrape(&self, url: &str, deadline: Duration) -> Result<CrawlPage, CharlotteError> {
        let value = self
            .scrape_value(
                url,
                deadline,
                CrawlScrapeOptions::from_browser(&self.browser),
            )
            .await?;
        Ok(CrawlPage::from_value(url, &value))
    }
}

/// The shorter of the server deadline and the client timeout minus `margin`.
#[must_use]
#[cfg_attr(not(any(test, feature = "serve")), allow(dead_code))]
pub(crate) fn effective_deadline(
    server: Duration,
    client_timeout_s: Option<f64>,
    margin: Duration,
) -> Duration {
    let Some(secs) = client_timeout_s else {
        return server;
    };
    if !secs.is_finite() || secs <= 0.0 {
        return Duration::ZERO;
    }
    let client = Duration::from_secs_f64(secs);
    server.min(client.saturating_sub(margin))
}

/// Memory and task counts from the host. Missing files mean the limit is unknown.
pub(crate) trait ResourceProbe: Send + Sync {
    /// Current sample. `None` fields mean the probe has no reading.
    fn sample(&self) -> ResourceSample;
}

/// One reading from a [`ResourceProbe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceSample {
    /// Current memory use in bytes.
    pub memory_bytes: Option<u64>,
    /// Current task count.
    pub task_count: Option<u64>,
}

/// Read cgroup v2 files, then cgroup v1 paths.
#[derive(Debug, Default)]
pub(crate) struct CgroupProbe;

impl ResourceProbe for CgroupProbe {
    fn sample(&self) -> ResourceSample {
        ResourceSample {
            memory_bytes: read_cgroup_counter(&[
                "/sys/fs/cgroup/memory.current",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            ]),
            task_count: read_cgroup_counter(&[
                "/sys/fs/cgroup/pids.current",
                "/sys/fs/cgroup/pids/pids.current",
            ]),
        }
    }
}

/// Refuse the crawl when a soft limit is already reached.
pub(crate) fn ensure_headroom(
    probe: &dyn ResourceProbe,
    config: &CrawlCapacityConfig,
) -> Result<(), CharlotteError> {
    let sample = probe.sample();
    if let Some(bytes) = sample.memory_bytes
        && bytes >= config.memory_soft_limit
    {
        return Err(CharlotteError::Overloaded {
            reason: OverloadReason::Memory,
            retry_after: config.retry_after,
        });
    }
    if let Some(tasks) = sample.task_count
        && tasks >= config.task_soft_limit
    {
        return Err(CharlotteError::Overloaded {
            reason: OverloadReason::Tasks,
            retry_after: config.retry_after,
        });
    }
    Ok(())
}

pub(crate) fn read_cgroup_counter(paths: &[&str]) -> Option<u64> {
    for path in paths {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let text = text.trim();
        if text == "max" {
            return None;
        }
        if let Ok(value) = text.parse::<u64>() {
            return Some(value);
        }
    }
    None
}

#[derive(Clone)]
pub(crate) struct AdmissionController {
    inner: Arc<AdmissionInner>,
}

struct AdmissionInner {
    free: Mutex<Vec<usize>>,
    retry_after: Duration,
}

pub(crate) struct AdmissionLease {
    slot: usize,
    inner: Arc<AdmissionInner>,
    released: bool,
}

impl AdmissionController {
    pub(crate) fn new(max_active: usize, retry_after: Duration) -> Self {
        Self {
            inner: Arc::new(AdmissionInner {
                free: Mutex::new((0..max_active).collect()),
                retry_after,
            }),
        }
    }

    pub(crate) fn try_acquire(&self) -> Result<AdmissionLease, CharlotteError> {
        let mut free = lock_mutex(&self.inner.free);
        let Some(slot) = free.pop() else {
            return Err(CharlotteError::Overloaded {
                reason: OverloadReason::Admission,
                retry_after: self.inner.retry_after,
            });
        };
        Ok(AdmissionLease {
            slot,
            inner: Arc::clone(&self.inner),
            released: false,
        })
    }

    #[cfg(test)]
    pub(crate) fn available(&self) -> usize {
        lock_mutex(&self.inner.free).len()
    }
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let mut free = lock_mutex(&self.inner.free);
        if !free.contains(&self.slot) {
            free.push(self.slot);
        }
    }
}

/// Run `work` on a slot. On timeout the slot stays held until `work` ends.
pub(crate) async fn run_admitted<T, Fut>(
    admission: &AdmissionController,
    deadline: Duration,
    work: Fut,
) -> Result<T, CharlotteError>
where
    T: Send + 'static,
    Fut: Future<Output = Result<T, CharlotteError>> + Send + 'static,
{
    let lease = admission.try_acquire()?;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = work.await;
        drop(lease);
        let _ = sender.send(result);
    });
    match tokio::time::timeout(deadline, receiver).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(CharlotteError::msg("crawl task ended without a result")),
        Err(_) => Err(CharlotteError::DeadlineExceeded { deadline }),
    }
}

#[derive(Clone)]
struct DomainLimiter {
    inner: Arc<DomainInner>,
}

struct DomainInner {
    max_active: usize,
    hosts: Mutex<HashMap<String, Arc<Semaphore>>>,
}

struct DomainPermit {
    _permit: OwnedSemaphorePermit,
}

impl DomainLimiter {
    fn new(max_active: usize) -> Self {
        Self {
            inner: Arc::new(DomainInner {
                max_active,
                hosts: Mutex::new(HashMap::new()),
            }),
        }
    }

    async fn acquire(&self, url: &str) -> Result<DomainPermit, CharlotteError> {
        let semaphore = self.semaphore_for(url)?;
        let permit = semaphore
            .acquire_owned()
            .await
            .map_err(|_| CharlotteError::msg("domain limiter is closed"))?;
        Ok(DomainPermit { _permit: permit })
    }

    #[cfg(test)]
    fn try_acquire(&self, url: &str) -> Result<Option<DomainPermit>, CharlotteError> {
        let semaphore = self.semaphore_for(url)?;
        match semaphore.try_acquire_owned() {
            Ok(permit) => Ok(Some(DomainPermit { _permit: permit })),
            Err(_) => Ok(None),
        }
    }

    fn semaphore_for(&self, url: &str) -> Result<Arc<Semaphore>, CharlotteError> {
        let host = host_key(url)?;
        let mut hosts = lock_mutex(&self.inner.hosts);
        if let Some(existing) = hosts.get(&host) {
            return Ok(Arc::clone(existing));
        }
        if hosts.len() >= MAX_TRACKED_DOMAINS {
            hosts.retain(|_, semaphore| semaphore.available_permits() < self.inner.max_active);
            if hosts.len() >= MAX_TRACKED_DOMAINS {
                return Err(CharlotteError::msg("too many domains in flight"));
            }
        }
        let semaphore = Arc::new(Semaphore::new(self.inner.max_active));
        hosts.insert(host, Arc::clone(&semaphore));
        Ok(semaphore)
    }
}

fn host_key(url: &str) -> Result<String, CharlotteError> {
    let parsed = Url::parse(url).map_err(|_| CharlotteError::msg("invalid URL"))?;
    parsed
        .host_str()
        .map(|host| host.to_ascii_lowercase())
        .ok_or_else(|| CharlotteError::msg("URL has no host"))
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn env_usize(key: &str) -> Result<Option<usize>, CharlotteError> {
    let Some(raw) = env_raw(key) else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .map_err(|_| CharlotteError::msg(format!("{key} must be an integer greater than zero")))?;
    if value == 0 {
        return Err(CharlotteError::msg(format!(
            "{key} must be an integer greater than zero"
        )));
    }
    Ok(Some(value))
}

fn env_u64(key: &str) -> Result<Option<u64>, CharlotteError> {
    let Some(raw) = env_raw(key) else {
        return Ok(None);
    };
    raw.parse::<u64>()
        .map(Some)
        .map_err(|_| CharlotteError::msg(format!("{key} must be an integer")))
}

fn env_raw(key: &str) -> Option<String> {
    let raw = std::env::var(key).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// True when `error` is a URL policy rejection the caller can treat as a bad request.
#[must_use]
#[cfg_attr(not(feature = "serve"), allow(dead_code))]
pub(crate) fn is_url_rejection(error: &CharlotteError) -> bool {
    matches!(
        error.to_string().as_str(),
        "invalid URL"
            | "URL scheme is not allowed"
            | "URL user information is not allowed"
            | "URL has no host"
            | "URL has no network port"
            | "URL host could not be resolved"
            | "URL host resolved to no addresses"
            | "blocked non-public network address"
            | "blocked metadata-style host"
            | "host is not in the URL allow-list"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::Notify;

    struct FakeProbe {
        memory_bytes: Option<u64>,
        task_count: Option<u64>,
    }

    impl ResourceProbe for FakeProbe {
        fn sample(&self) -> ResourceSample {
            ResourceSample {
                memory_bytes: self.memory_bytes,
                task_count: self.task_count,
            }
        }
    }

    #[test]
    fn pool_larger_than_the_page_cap_is_rejected() {
        let mut config = CrawlCapacityConfig::agent();
        config.browser_pool_size = config.max_pages + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn missing_cgroup_file_is_unknown() {
        assert!(read_cgroup_counter(&["/no/such/charlotte/cgroup"]).is_none());
    }

    #[test]
    fn resource_probe_trips_memory_and_task_limits() {
        let config = CrawlCapacityConfig::agent();
        let under = FakeProbe {
            memory_bytes: Some(1),
            task_count: Some(1),
        };
        assert!(ensure_headroom(&under, &config).is_ok());

        let memory = FakeProbe {
            memory_bytes: Some(config.memory_soft_limit),
            task_count: None,
        };
        let memory_error = ensure_headroom(&memory, &config).unwrap_err();
        assert!(matches!(
            memory_error,
            CharlotteError::Overloaded {
                reason: OverloadReason::Memory,
                ..
            }
        ));

        let tasks = FakeProbe {
            memory_bytes: None,
            task_count: Some(config.task_soft_limit),
        };
        let task_error = ensure_headroom(&tasks, &config).unwrap_err();
        assert!(matches!(
            task_error,
            CharlotteError::Overloaded {
                reason: OverloadReason::Tasks,
                ..
            }
        ));
    }

    #[test]
    fn admission_rejects_when_every_slot_is_taken() {
        let admission = AdmissionController::new(1, DEFAULT_RETRY_AFTER);
        let held = admission.try_acquire().expect("first slot");
        let rejected = match admission.try_acquire() {
            Err(error) => error,
            Ok(_lease) => panic!("second acquire should fail"),
        };
        assert!(matches!(
            rejected,
            CharlotteError::Overloaded {
                reason: OverloadReason::Admission,
                ..
            }
        ));
        drop(held);
        assert!(admission.try_acquire().is_ok());
    }

    #[test]
    fn domain_limiter_caps_one_host() {
        let limiter = DomainLimiter::new(2);
        let first = limiter
            .try_acquire("https://Example.com/a")
            .expect("host")
            .expect("first permit");
        let _second = limiter
            .try_acquire("https://example.com/b")
            .expect("host")
            .expect("second permit");
        assert!(
            limiter
                .try_acquire("https://example.com/c")
                .expect("host")
                .is_none()
        );
        assert!(
            limiter
                .try_acquire("https://other.test/")
                .expect("other host")
                .is_some()
        );
        drop(first);
        assert!(
            limiter
                .try_acquire("https://example.com/d")
                .expect("host")
                .is_some()
        );
    }

    #[tokio::test]
    async fn deadline_returns_while_the_lease_stays_held() {
        let admission = AdmissionController::new(1, DEFAULT_RETRY_AFTER);
        let release = Arc::new(Notify::new());
        let entered = Arc::new(Notify::new());
        let release_task = Arc::clone(&release);
        let entered_task = Arc::clone(&entered);
        let pending = tokio::spawn({
            let admission = admission.clone();
            async move {
                run_admitted(&admission, Duration::from_millis(40), async move {
                    entered_task.notify_one();
                    release_task.notified().await;
                    Ok::<(), CharlotteError>(())
                })
                .await
            }
        });
        entered.notified().await;
        let error = pending.await.expect("task").unwrap_err();
        assert!(matches!(error, CharlotteError::DeadlineExceeded { .. }));
        assert_eq!(admission.available(), 0);
        release.notify_one();
        let released = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&released);
        tokio::time::timeout(Duration::from_secs(1), async move {
            loop {
                if admission.available() == 1 {
                    flag.store(true, Ordering::SeqCst);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("slot released");
        assert!(released.load(Ordering::SeqCst));
    }

    #[test]
    fn client_timeout_shortens_the_server_deadline() {
        let server = Duration::from_secs(80);
        let margin = Duration::from_secs(5);
        assert_eq!(
            effective_deadline(server, Some(30.0), margin),
            Duration::from_secs(25)
        );
        assert_eq!(effective_deadline(server, None, margin), server);
        assert_eq!(effective_deadline(server, Some(90.0), margin), server);
    }
}
