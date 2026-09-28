//! Crawl orchestration: bounded deep crawl, optional shared caching, and hooks.
//!
//! “Undetected” Chromium patches (as in some Python stacks) are not bundled; use
//! [`crate::browser::AutomationProfile::Stealth`] for best-effort mitigation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinSet;
use tokio::time::sleep;
use tracing::instrument;

use crate::browser::{BrowserHandle, NavigateOptions, resolve_href};
use crate::chunk::Chunker;
use crate::error::CharlotteError;
use crate::markdown::document_html_to_markdown;
use crate::pool::BrowserPool;
use crate::robots::RobotsFetcher;

/// Limits and timing for a [`deep_crawl`].
#[derive(Clone)]
pub struct CrawlRunConfig {
    /// Maximum link hops from the seed URL (0 = only the seed page).
    pub max_depth: usize,
    /// Stop after this many successfully loaded pages.
    pub max_pages: usize,
    /// Passed through to [`BrowserHandle::list_links`] (filters by current page origin).
    pub same_origin_only: bool,
    /// Pause between processing each queued URL (rate limiting).
    pub delay_between_pages: Duration,
    /// When set with [`Self::page_cache`], skip `goto` if a fresh cached payload exists.
    pub cache_ttl: Option<Duration>,
    /// Shared JSON cache (see [`SharedCrawlCache`]); requires [`Self::cache_ttl`].
    pub page_cache: Option<Arc<SharedCrawlCache>>,
    /// Navigation behavior for every page in the crawl.
    pub navigate: NavigateOptions,
    /// Max anchors collected per page for queue expansion.
    pub link_limit_per_page: usize,
    /// After navigation, fetch full HTML and convert to Markdown ([`htmd`]) for each page JSON `"markdown"` field.
    pub include_page_markdown: bool,
    /// Truncate raw HTML to this many bytes (UTF-8 safe) before conversion.
    pub max_html_bytes: usize,
    /// Truncate Markdown to this many Unicode scalar values after conversion.
    pub max_markdown_chars: usize,
    /// When true (default), strip `#fragments` when deduplicating URLs. Set **false** for SPAs that
    /// encode client routes in the fragment (otherwise every `#/…` route collapses to one page).
    pub strip_url_fragments: bool,
    /// After `goto` succeeds, wait this long before collecting links and HTML (helps JS-heavy docs).
    pub after_navigation_delay: Duration,
    /// When true, fetch `robots.txt` per origin and skip URLs disallowed for `User-agent: *`.
    /// Default false (local research / controlled targets).
    pub respect_robots: bool,
}

impl std::fmt::Debug for CrawlRunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrawlRunConfig")
            .field("max_depth", &self.max_depth)
            .field("max_pages", &self.max_pages)
            .field("same_origin_only", &self.same_origin_only)
            .field("delay_between_pages", &self.delay_between_pages)
            .field("cache_ttl", &self.cache_ttl)
            .field("page_cache", &self.page_cache.is_some())
            .field("navigate", &self.navigate)
            .field("link_limit_per_page", &self.link_limit_per_page)
            .field("include_page_markdown", &self.include_page_markdown)
            .field("max_html_bytes", &self.max_html_bytes)
            .field("max_markdown_chars", &self.max_markdown_chars)
            .field("strip_url_fragments", &self.strip_url_fragments)
            .field("after_navigation_delay", &self.after_navigation_delay)
            .field("respect_robots", &self.respect_robots)
            .finish()
    }
}

impl Default for CrawlRunConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_pages: 25,
            same_origin_only: true,
            delay_between_pages: Duration::ZERO,
            cache_ttl: None,
            page_cache: None,
            navigate: NavigateOptions::default(),
            link_limit_per_page: 80,
            include_page_markdown: false,
            max_html_bytes: 2_000_000,
            max_markdown_chars: 500_000,
            strip_url_fragments: true,
            after_navigation_delay: Duration::ZERO,
            respect_robots: false,
        }
    }
}

/// Optional synchronous hooks for observability (metrics, tracing, custom filtering).
#[derive(Clone, Default)]
pub struct CrawlHooks {
    pub before_navigate: Option<Arc<dyn Fn(&str, usize) + Send + Sync>>,
    pub after_page: Option<Arc<dyn Fn(&str, usize, &Value) + Send + Sync>>,
}

/// Share crawl payloads across [`deep_crawl`] runs or with other tasks (`nav` + `links` JSON).
#[derive(Clone, Default)]
pub struct SharedCrawlCache {
    inner: Arc<Mutex<HashMap<String, (Instant, Value)>>>,
}

impl SharedCrawlCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a cached page value (typically `{ "nav": …, "links": … }` as produced by a crawl step).
    pub async fn insert(&self, key: String, value: Value) {
        let mut g = self.inner.lock().await;
        g.insert(key, (Instant::now(), value));
    }

    /// Return cached JSON if present and younger than `ttl`.
    pub async fn get_if_fresh(&self, key: &str, ttl: Duration) -> Option<Value> {
        let g = self.inner.lock().await;
        let (t, v) = g.get(key)?;
        if t.elapsed() > ttl {
            return None;
        }
        Some(v.clone())
    }
}

/// Normalize for crawl deduplication. By default strips `#fragment` (see [`normalize_crawl_url_ex`]).
pub fn normalize_crawl_url(url: &str) -> Result<String, CharlotteError> {
    normalize_crawl_url_ex(url, true)
}

/// Normalize for crawl queue keys and visited-set deduplication.
///
/// - `strip_fragment: true` — same as historical behavior: remove `#…` (good for classic sites).
/// - `strip_fragment: false` — keep the fragment so `…/doc#/a` and `…/doc#/b` stay distinct (SPAs).
pub fn normalize_crawl_url_ex(url: &str, strip_fragment: bool) -> Result<String, CharlotteError> {
    let mut u =
        url::Url::parse(url).map_err(|e| CharlotteError::msg(format!("invalid URL: {e}")))?;
    if strip_fragment {
        u.set_fragment(None);
    }
    Ok(u.to_string())
}

/// Breadth-first crawl from `seed_url` using the shared browser tab.
///
/// Returns JSON: `{ "pages": [ { "url", "depth", "nav", "links", "from_cache", "markdown" } ], "stopped_reason" }`.
/// When [`CrawlRunConfig::include_page_markdown`] is true, `"markdown"` is a string (possibly empty); otherwise `null`.
#[instrument(skip_all, fields(seed_url = %seed_url, respect_robots = run.respect_robots))]
pub async fn deep_crawl(
    handle: &BrowserHandle,
    seed_url: &str,
    run: &CrawlRunConfig,
    hooks: &CrawlHooks,
) -> Result<Value, CharlotteError> {
    handle.enforce_url_policy(seed_url).await?;
    let robots = if run.respect_robots {
        Some(Arc::new(RobotsFetcher::new()?))
    } else {
        None
    };
    let seed_norm = normalize_crawl_url_ex(seed_url, run.strip_url_fragments)?;
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    queue.push_back((seed_norm.clone(), 0));
    let mut visited: HashSet<String> = HashSet::new();
    let mut pages_out: Vec<Value> = Vec::new();

    let mut stopped_reason = "complete";

    while let Some((url, depth)) = queue.pop_front() {
        if pages_out.len() >= run.max_pages {
            stopped_reason = "max_pages";
            break;
        }
        if !visited.insert(url.clone()) {
            continue;
        }

        let (page, outlink_norms) =
            crawl_url_once(handle, run, hooks, url, depth, robots.as_ref()).await?;
        pages_out.push(page);
        if depth < run.max_depth {
            enqueue_from_hrefs(&mut queue, &visited, &outlink_norms, depth + 1);
        }

        if !run.delay_between_pages.is_zero() {
            sleep(run.delay_between_pages).await;
        }
    }

    Ok(json!({
        "pages": pages_out,
        "stopped_reason": stopped_reason,
        "visited_count": visited.len(),
    }))
}

/// Discover URLs reachable from `seed_url` using bounded same-origin BFS without returning page content.
///
/// Returns `{ "urls": [...], "count", "stopped_reason", "visited_count" }`.
pub async fn site_map(
    handle: &BrowserHandle,
    seed_url: &str,
    mut run: CrawlRunConfig,
    hooks: &CrawlHooks,
) -> Result<Value, CharlotteError> {
    run.include_page_markdown = false;
    let crawled = deep_crawl(handle, seed_url, &run, hooks).await?;
    let urls = map_urls_from_pages(crawled.get("pages"));
    Ok(json!({
        "urls": urls,
        "count": urls.len(),
        "stopped_reason": crawled.get("stopped_reason").cloned().unwrap_or(Value::Null),
        "visited_count": crawled.get("visited_count").cloned().unwrap_or(Value::Null),
    }))
}

fn map_urls_from_pages(pages: Option<&Value>) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for u in pages
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("url").and_then(Value::as_str))
    {
        if seen.insert(u.to_string()) {
            out.push(u.to_string());
        }
    }
    out
}

/// Caps how much chunk data is returned to callers (e.g. LLM tool messages). Similar in spirit to
/// Crawl4AI’s `chunk_token_threshold` for keeping each downstream payload bounded.
///
/// - `max_chunks == 0` — no chunk-count limit.
/// - `max_total_text_chars == 0` — no total character limit.
#[derive(Debug, Clone)]
pub struct SiteCrawlOutputLimits {
    pub max_chunks: usize,
    pub max_total_text_chars: usize,
}

impl Default for SiteCrawlOutputLimits {
    fn default() -> Self {
        Self {
            max_chunks: 0,
            max_total_text_chars: 0,
        }
    }
}

impl SiteCrawlOutputLimits {
    /// Defaults suitable for agent tool calls: avoids multi-hundred-thousand-token tool payloads.
    pub fn agent_tool_default() -> Self {
        Self {
            max_chunks: 120,
            max_total_text_chars: 100_000,
        }
    }
}

/// Apply [`SiteCrawlOutputLimits`] to full chunk rows. May truncate the last included chunk’s `text`
/// to stay within `max_total_text_chars`.
pub fn apply_site_crawl_output_limits(
    mut chunks: Vec<Value>,
    limits: &SiteCrawlOutputLimits,
) -> (Vec<Value>, Value) {
    let chunks_total = chunks.len();
    if limits.max_chunks == 0 && limits.max_total_text_chars == 0 {
        let chars: usize = chunks
            .iter()
            .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
            .map(|s| s.chars().count())
            .sum();
        let stats = json!({
            "chunks_total": chunks_total,
            "chunks_returned": chunks_total,
            "chars_returned": chars,
            "truncated": false,
        });
        return (chunks, stats);
    }

    let mut out: Vec<Value> = Vec::new();
    let mut chars_used = 0usize;
    let mut truncated = false;

    for mut chunk in chunks.drain(..) {
        if limits.max_chunks > 0 && out.len() >= limits.max_chunks {
            truncated = true;
            break;
        }
        let text = chunk
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let n = text.chars().count();

        if limits.max_total_text_chars > 0 && chars_used + n > limits.max_total_text_chars {
            let remaining = limits.max_total_text_chars.saturating_sub(chars_used);
            if remaining == 0 {
                truncated = true;
                break;
            }
            let shortened: String = text.chars().take(remaining).collect();
            chars_used += shortened.chars().count();
            if let Some(obj) = chunk.as_object_mut() {
                obj.insert("text".to_string(), json!(shortened));
                obj.insert("text_truncated".to_string(), json!(true));
            }
            out.push(chunk);
            truncated = true;
            break;
        }

        chars_used += n;
        out.push(chunk);

        if limits.max_chunks > 0 && out.len() >= limits.max_chunks {
            truncated = truncated || chunks_total > out.len();
            break;
        }
    }

    if !truncated && chunks_total > out.len() {
        truncated = true;
    }

    let stats = json!({
        "chunks_total": chunks_total,
        "chunks_returned": out.len(),
        "chars_returned": chars_used,
        "truncated": truncated,
    });
    (out, stats)
}

/// Runs [`deep_crawl`] with [`CrawlRunConfig::include_page_markdown`] set to `true`, then applies `chunker` to each page’s Markdown.
///
/// `output_limits`: pass [`SiteCrawlOutputLimits::default()`] for no cap; use [`SiteCrawlOutputLimits::agent_tool_default()`]
/// for bounded tool-sized payloads.
///
/// Returns `{ stopped_reason, visited_count, pages_in_result, chunks, output }` where `output` describes truncation.
pub async fn site_crawl_with_chunks(
    handle: &BrowserHandle,
    seed_url: &str,
    mut run: CrawlRunConfig,
    hooks: &CrawlHooks,
    chunker: &crate::chunk::ChunkerConfig,
    output_limits: &SiteCrawlOutputLimits,
) -> Result<Value, CharlotteError> {
    run.include_page_markdown = true;
    let crawl_json = deep_crawl(handle, seed_url, &run, hooks).await?;
    let pages = crawl_json
        .get("pages")
        .and_then(|p| p.as_array())
        .ok_or_else(|| CharlotteError::msg("deep_crawl: missing pages array"))?;

    let mut chunks_out = Vec::new();
    let mut id = 0usize;
    for p in pages {
        let url = p.get("url").and_then(|u| u.as_str()).unwrap_or("");
        let md = p.get("markdown").and_then(|m| m.as_str()).unwrap_or("");
        if md.trim().is_empty() {
            continue;
        }
        for d in chunker.chunk(md) {
            if d.text.trim().is_empty() {
                continue;
            }
            chunks_out.push(json!({
                "id": id,
                "source_url": url,
                "text": d.text,
            }));
            id += 1;
        }
    }

    let (chunks_limited, output_stats) = apply_site_crawl_output_limits(chunks_out, output_limits);

    Ok(json!({
        "stopped_reason": crawl_json.get("stopped_reason"),
        "visited_count": crawl_json.get("visited_count"),
        "pages_in_result": pages.len(),
        "chunks": chunks_limited,
        "output": output_stats,
    }))
}

/// Same as [`deep_crawl`], but uses one browser per pool entry and processes the frontier concurrently.
///
/// Page order in `"pages"` is **not** strictly breadth-first; use [`deep_crawl`] when you need strict BFS ordering.
/// Requires a non-empty [`BrowserPool`].
pub async fn deep_crawl_parallel(
    pool: &BrowserPool,
    seed_url: &str,
    run: &CrawlRunConfig,
    hooks: &CrawlHooks,
) -> Result<Value, CharlotteError> {
    if pool.is_empty() {
        return Err(CharlotteError::msg("BrowserPool is empty"));
    }
    pool.handles()[0].enforce_url_policy(seed_url).await?;
    let robots = if run.respect_robots {
        Some(Arc::new(RobotsFetcher::new()?))
    } else {
        None
    };
    let robots_workers = robots.clone();
    let seed_norm = normalize_crawl_url_ex(seed_url, run.strip_url_fragments)?;
    let frontier = Arc::new(Mutex::new(VecDeque::from([(seed_norm, 0)])));
    let visited = Arc::new(Mutex::new(HashSet::new()));
    let results = Arc::new(Mutex::new(Vec::<Value>::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let work = Arc::new(Notify::new());

    let run_a = Arc::new(run.clone());
    let hooks_a = Arc::new(hooks.clone());

    let n_workers = pool.handles().len();
    let mut join_set = JoinSet::new();

    for wid in 0..n_workers {
        let handle = pool.handles()[wid].clone();
        let frontier = Arc::clone(&frontier);
        let visited = Arc::clone(&visited);
        let results = Arc::clone(&results);
        let stop = Arc::clone(&stop);
        let in_flight = Arc::clone(&in_flight);
        let work = Arc::clone(&work);
        let run_c = Arc::clone(&run_a);
        let hooks_c = Arc::clone(&hooks_a);
        let robots_w = robots_workers.clone();

        join_set.spawn(async move {
            parallel_worker_loop(
                handle, run_c, hooks_c, frontier, visited, results, stop, in_flight, work, robots_w,
            )
            .await
        });
    }

    while let Some(joined) = join_set.join_next().await {
        joined.map_err(|e| CharlotteError::msg(format!("crawl worker panicked: {e}")))??;
    }

    let pages = results.lock().await.clone();
    let visited_n = visited.lock().await.len();
    let stopped_reason = if pages.len() >= run.max_pages {
        "max_pages"
    } else {
        "complete"
    };

    Ok(json!({
        "pages": pages,
        "stopped_reason": stopped_reason,
        "visited_count": visited_n,
        "parallel_workers": n_workers,
    }))
}

async fn parallel_worker_loop(
    handle: BrowserHandle,
    run: Arc<CrawlRunConfig>,
    hooks: Arc<CrawlHooks>,
    frontier: Arc<Mutex<VecDeque<(String, usize)>>>,
    visited: Arc<Mutex<HashSet<String>>>,
    results: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    in_flight: Arc<AtomicUsize>,
    work: Arc<Notify>,
    robots: Option<Arc<RobotsFetcher>>,
) -> Result<(), CharlotteError> {
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }

        let job = {
            let mut q = frontier.lock().await;
            q.pop_front()
        };

        if job.is_none() {
            let queue_empty = {
                let q = frontier.lock().await;
                q.is_empty()
            };
            let infl = in_flight.load(Ordering::Acquire);
            if queue_empty && infl == 0 {
                stop.store(true, Ordering::Release);
                work.notify_waiters();
                break;
            }
            tokio::select! {
                _ = work.notified() => {}
                _ = sleep(Duration::from_millis(500)) => {}
            }
            continue;
        }

        let (url, depth) = job.unwrap();

        // Reserve work immediately so other workers do not treat the queue as globally idle.
        in_flight.fetch_add(1, Ordering::SeqCst);

        {
            let r = results.lock().await;
            if r.len() >= run.max_pages {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                stop.store(true, Ordering::Release);
                work.notify_waiters();
                break;
            }
        }

        {
            let mut v = visited.lock().await;
            if !v.insert(url.clone()) {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                work.notify_waiters();
                continue;
            }
        }

        let page_res =
            crawl_url_once(&handle, &run, &hooks, url.clone(), depth, robots.as_ref()).await;

        in_flight.fetch_sub(1, Ordering::SeqCst);
        work.notify_waiters();

        let (page, outlink_norms) = match page_res {
            Ok(p) => p,
            Err(e) => {
                stop.store(true, Ordering::Release);
                return Err(e);
            }
        };

        let mut r = results.lock().await;
        if r.len() >= run.max_pages {
            stop.store(true, Ordering::Release);
            break;
        }
        r.push(page);

        if depth < run.max_depth {
            let mut q = frontier.lock().await;
            for u in &outlink_norms {
                q.push_back((u.clone(), depth + 1));
            }
        }

        if !run.delay_between_pages.is_zero() {
            sleep(run.delay_between_pages).await;
        }

        work.notify_waiters();
    }

    Ok(())
}

/// Fetch one URL: navigation, optional cache, hooks, and a page JSON value (same shape as [`deep_crawl`]).
#[instrument(skip(handle, run, hooks, robots), fields(url = %url, depth))]
async fn crawl_url_once(
    handle: &BrowserHandle,
    run: &CrawlRunConfig,
    hooks: &CrawlHooks,
    url: String,
    depth: usize,
    robots: Option<&Arc<RobotsFetcher>>,
) -> Result<(Value, Vec<String>), CharlotteError> {
    handle.enforce_url_policy(&url).await?;
    if let Some(r) = robots
        && !r.is_allowed(&url).await?
    {
        let page = json!({
            "url": url,
            "depth": depth,
            "skipped": "robots_txt",
            "nav": Value::Null,
            "links": json!({ "links": [] }),
            "from_cache": false,
            "markdown": Value::Null,
        });
        return Ok((page, Vec::new()));
    }

    let use_cache = run.page_cache.is_some() && run.cache_ttl.is_some();

    let mut cached_blob = match (&run.page_cache, run.cache_ttl) {
        (Some(cache), Some(ttl)) => cache.get_if_fresh(&url, ttl).await,
        _ => None,
    };

    if run.include_page_markdown {
        cached_blob = cached_blob.filter(|b| b.get("markdown").is_some());
    }

    let from_cache = cached_blob.is_some();

    if !from_cache {
        if let Some(ref h) = hooks.before_navigate {
            h(&url, depth);
        }
    }

    let (nav, links_val, outlink_norms, markdown_json) = if let Some(blob) = cached_blob {
        let nav = blob
            .get("nav")
            .cloned()
            .unwrap_or_else(|| json!({ "url": url.clone() }));
        let links_val = blob
            .get("links")
            .cloned()
            .unwrap_or_else(|| json!({ "links": [] }));
        let norms = blob
            .get("outlinks_normalized")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let markdown_json = blob.get("markdown").cloned().unwrap_or(Value::Null);
        (nav, links_val, norms, markdown_json)
    } else {
        let nav = handle
            .navigate_with_options(&url, run.navigate.clone())
            .await?;
        if !run.after_navigation_delay.is_zero() {
            sleep(run.after_navigation_delay).await;
        }
        let final_url = nav
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or(&url)
            .to_string();
        let links_val = handle
            .list_links(None, run.link_limit_per_page, run.same_origin_only)
            .await?;
        let hrefs = extract_hrefs(&links_val);
        let mut normalized_out = Vec::new();
        for h in &hrefs {
            let resolved = resolve_href(&final_url, h);
            if let Ok(n) = normalize_crawl_url_ex(&resolved, run.strip_url_fragments) {
                normalized_out.push(n);
            }
        }

        let markdown_json = if run.include_page_markdown {
            let raw_html = handle.page_html().await?;
            let result = document_html_to_markdown(
                &raw_html,
                None,
                run.max_html_bytes,
                run.max_markdown_chars,
            )?;
            json!(result.markdown)
        } else {
            Value::Null
        };

        if use_cache {
            let cache = run.page_cache.as_ref().unwrap();
            let blob = json!({
                "nav": nav.clone(),
                "links": links_val.clone(),
                "outlinks_normalized": normalized_out.clone(),
                "markdown": markdown_json.clone(),
            });
            cache.insert(url.clone(), blob).await;
        }

        (nav, links_val, normalized_out, markdown_json)
    };

    if let Some(ref h) = hooks.after_page {
        h(&url, depth, &nav);
    }

    let page = json!({
        "url": url,
        "depth": depth,
        "nav": nav,
        "links": links_val,
        "from_cache": from_cache,
        "markdown": markdown_json,
    });

    Ok((page, outlink_norms))
}

fn extract_hrefs(links_val: &Value) -> Vec<String> {
    let Some(arr) = links_val.get("links").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|o| o.get("href").and_then(|v| v.as_str()).map(String::from))
        .collect()
}

fn enqueue_from_hrefs(
    queue: &mut VecDeque<(String, usize)>,
    visited: &HashSet<String>,
    normalized_hrefs: &[String],
    next_depth: usize,
) {
    for u in normalized_hrefs {
        if visited.contains(u) {
            continue;
        }
        queue.push_back((u.clone(), next_depth));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_fragment() {
        let n = normalize_crawl_url("https://example.com/a#x").unwrap();
        assert_eq!(n, "https://example.com/a");
    }

    #[test]
    fn normalize_preserves_fragment_when_disabled() {
        let n = normalize_crawl_url_ex("https://example.com/a#x", false).unwrap();
        assert_eq!(n, "https://example.com/a#x");
    }

    #[test]
    fn output_limits_no_cap_includes_all_and_stats() {
        let chunks = vec![
            json!({"id": 0, "source_url": "u", "text": "ab"}),
            json!({"id": 1, "source_url": "u", "text": "cd"}),
        ];
        let (out, stats) =
            apply_site_crawl_output_limits(chunks, &SiteCrawlOutputLimits::default());
        assert_eq!(out.len(), 2);
        assert_eq!(stats["chunks_total"], 2);
        assert_eq!(stats["chunks_returned"], 2);
        assert_eq!(stats["chars_returned"], 4);
        assert_eq!(stats["truncated"], false);
    }

    #[test]
    fn output_limits_max_chunks() {
        let chunks: Vec<Value> = (0..5)
            .map(|i| json!({"id": i, "source_url": "u", "text": "x"}))
            .collect();
        let limits = SiteCrawlOutputLimits {
            max_chunks: 2,
            max_total_text_chars: 0,
        };
        let (out, stats) = apply_site_crawl_output_limits(chunks, &limits);
        assert_eq!(out.len(), 2);
        assert_eq!(stats["truncated"], true);
        assert_eq!(stats["chunks_total"], 5);
    }

    #[test]
    fn output_limits_chars_truncates_last_chunk() {
        let chunks = vec![json!({"id": 0, "source_url": "u", "text": "abcdefghij"})];
        let limits = SiteCrawlOutputLimits {
            max_chunks: 0,
            max_total_text_chars: 4,
        };
        let (out, stats) = apply_site_crawl_output_limits(chunks, &limits);
        assert_eq!(out[0]["text"], "abcd");
        assert_eq!(out[0]["text_truncated"], true);
        assert_eq!(stats["truncated"], true);
        assert_eq!(stats["chars_returned"], 4);
    }

    #[test]
    fn map_urls_from_pages_deduplicates_and_preserves_order() {
        let pages = json!([
            {"url": "https://a.example/"},
            {"url": "https://a.example/docs"},
            {"url": "https://a.example/"},
            {"x": 1}
        ]);
        let urls = map_urls_from_pages(Some(&pages));
        assert_eq!(
            urls,
            vec![
                "https://a.example/".to_string(),
                "https://a.example/docs".to_string()
            ]
        );
    }
}
