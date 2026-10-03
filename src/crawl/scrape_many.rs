use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;

use crate::browser::pool::{BrowserPool, DEFAULT_BROWSER_POOL_CAP};
use crate::browser::{BrowserConfig, BrowserHandle, NavigateOptions};
use crate::error::CharlotteError;

pub const SCRAPE_MANY_MAX_URLS: usize = 20;
pub const SCRAPE_MANY_DEFAULT_MAX_TOTAL_TEXT_CHARS: usize = 100_000;
pub const SCRAPE_DEFAULT_CONCURRENCY: usize = 3;
pub const SCRAPE_MAX_CONCURRENCY: usize = 4;

#[derive(Debug, Clone)]
pub struct ScrapeRequest {
    pub url: String,
}

#[derive(Debug, Clone)]
pub struct ScrapeManyOptions {
    pub focus: Option<String>,
    pub max_concurrency: usize,
    pub max_html_bytes: usize,
    pub max_markdown_chars: usize,
    pub max_total_text_chars: usize,
    pub navigate_options: NavigateOptions,
}

impl ScrapeManyOptions {
    pub fn with_defaults(
        focus: Option<String>,
        max_concurrency: Option<usize>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
        max_total_text_chars: Option<usize>,
        pool_size: usize,
    ) -> Self {
        Self {
            focus,
            max_concurrency: clamp_concurrency(max_concurrency, pool_size),
            max_html_bytes,
            max_markdown_chars,
            max_total_text_chars: max_total_text_chars
                .unwrap_or(SCRAPE_MANY_DEFAULT_MAX_TOTAL_TEXT_CHARS),
            navigate_options: NavigateOptions::for_scrape(),
        }
    }
}

#[derive(Clone)]
struct PoolWorker {
    handle: BrowserHandle,
}

struct PoolRuntime {
    workers: Arc<Vec<PoolWorker>>,
}

#[derive(Clone)]
pub struct LazyBrowserPool {
    config: BrowserConfig,
    size: usize,
    runtime: Arc<Mutex<Option<PoolRuntime>>>,
}

impl LazyBrowserPool {
    pub fn new(config: BrowserConfig, size: Option<usize>) -> Self {
        let size = size.unwrap_or(1).max(1);
        Self {
            config,
            size,
            runtime: Arc::new(Mutex::new(None)),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    async fn workers(&self) -> Result<Arc<Vec<PoolWorker>>, CharlotteError> {
        let mut guard = self.runtime.lock().await;
        if guard.is_none() {
            let pool = BrowserPool::from_config(self.config.clone(), Some(self.size)).await?;
            let workers = pool
                .handles()
                .iter()
                .cloned()
                .map(|handle| PoolWorker { handle })
                .collect::<Vec<_>>();
            *guard = Some(PoolRuntime {
                workers: Arc::new(workers),
            });
        }
        // Safe: initialized just above when absent.
        Ok(guard
            .as_ref()
            .map(|r| Arc::clone(&r.workers))
            .unwrap_or_else(|| Arc::new(Vec::new())))
    }

    pub async fn default_limits(&self) -> Result<(usize, usize), CharlotteError> {
        let workers = self.workers().await?;
        let Some(first) = workers.first() else {
            return Err(CharlotteError::msg("parallel scrape pool has no workers"));
        };
        let max_html = first.handle.max_html_bytes_limit().await?;
        let max_markdown = first.handle.max_markdown_chars_limit().await?;
        Ok((max_html, max_markdown))
    }

    /// Close every browser worker if the pool was ever initialized (idempotent no-op otherwise).
    ///
    /// Must be called once the pool's owner is done with it — the pool otherwise holds one
    /// CDP connection (and, for the Playwright backend, one Node driver process) per worker
    /// for as long as the process runs.
    pub async fn close(&self) -> Result<(), CharlotteError> {
        let runtime = self.runtime.lock().await.take();
        let Some(runtime) = runtime else {
            return Ok(());
        };
        let mut first_err = None;
        for worker in runtime.workers.iter() {
            if let Err(e) = worker.handle.close().await {
                tracing::warn!(target: "charlotte::scrape_many", error = %e, "lazy browser pool: worker close failed");
                first_err.get_or_insert(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

pub fn validate_urls(urls: Vec<String>) -> Result<Vec<ScrapeRequest>, CharlotteError> {
    if urls.is_empty() {
        return Err(CharlotteError::msg("scrape_many requires at least one URL"));
    }
    if urls.len() > SCRAPE_MANY_MAX_URLS {
        return Err(CharlotteError::msg(format!(
            "scrape_many supports at most {SCRAPE_MANY_MAX_URLS} URLs per call"
        )));
    }
    let mut requests = Vec::with_capacity(urls.len());
    for raw in urls {
        let url = raw.trim();
        if url.is_empty() {
            return Err(CharlotteError::msg("scrape_many received an empty URL"));
        }
        requests.push(ScrapeRequest {
            url: url.to_string(),
        });
    }
    Ok(requests)
}

pub fn clamp_concurrency(requested: Option<usize>, _pool_size: usize) -> usize {
    let requested = requested.unwrap_or(SCRAPE_DEFAULT_CONCURRENCY).max(1);
    requested
        .min(DEFAULT_BROWSER_POOL_CAP)
        .min(SCRAPE_MAX_CONCURRENCY)
}

pub fn apply_total_text_cap(results: &mut [Value], max_total_text_chars: usize) -> bool {
    if max_total_text_chars == 0 {
        return false;
    }
    let mut used = 0usize;
    let mut truncated_any = false;
    for row in results.iter_mut() {
        let Some(markdown) = row.get("markdown").and_then(Value::as_str) else {
            continue;
        };
        let markdown_chars = markdown.chars().count();
        if used >= max_total_text_chars {
            row["markdown"] = json!("");
            row["truncated_total"] = json!(true);
            truncated_any = true;
            continue;
        }
        let remaining = max_total_text_chars - used;
        if markdown_chars > remaining {
            row["markdown"] = json!(markdown.chars().take(remaining).collect::<String>());
            row["truncated_total"] = json!(true);
            used = max_total_text_chars;
            truncated_any = true;
            continue;
        }
        used += markdown_chars;
    }
    truncated_any
}

pub async fn scrape_many(
    pool: &LazyBrowserPool,
    requests: Vec<ScrapeRequest>,
    options: ScrapeManyOptions,
) -> Result<Value, CharlotteError> {
    let workers = pool.workers().await?;
    let Some(worker) = workers.first() else {
        return Err(CharlotteError::msg("parallel scrape pool has no workers"));
    };
    scrape_many_on(&worker.handle, requests, options).await
}

/// Fan-out scrapes across isolated tabs on a single browser connection.
pub async fn scrape_many_on(
    handle: &BrowserHandle,
    requests: Vec<ScrapeRequest>,
    options: ScrapeManyOptions,
) -> Result<Value, CharlotteError> {
    let used_concurrency = options.max_concurrency.max(1).min(SCRAPE_MAX_CONCURRENCY);
    let permit_limit = Arc::new(Semaphore::new(used_concurrency));
    let mut jobs = JoinSet::new();

    for (idx, request) in requests.into_iter().enumerate() {
        let Some(permit) = permit_limit.clone().acquire_owned().await.ok() else {
            return Err(CharlotteError::msg(
                "failed to acquire scrape concurrency slot",
            ));
        };
        let handle = handle.clone();
        let opts = options.clone();
        jobs.spawn(async move {
            let _permit = permit;
            (idx, scrape_one(handle, request, opts).await)
        });
    }

    let mut ordered = Vec::new();
    while let Some(joined) = jobs.join_next().await {
        match joined {
            Ok((idx, row)) => ordered.push((idx, row)),
            Err(err) => ordered.push((
                usize::MAX,
                json!({
                    "ok": false,
                    "error": "task_join_error",
                    "message": err.to_string(),
                }),
            )),
        }
    }
    ordered.sort_by_key(|(idx, _)| *idx);
    let mut results = ordered.into_iter().map(|(_, row)| row).collect::<Vec<_>>();
    let truncated_total = apply_total_text_cap(&mut results, options.max_total_text_chars);

    Ok(json!({
        "ok": true,
        "max_concurrency": used_concurrency,
        "results": results,
        "output": {
            "truncated": truncated_total,
            "max_total_text_chars": options.max_total_text_chars,
        }
    }))
}

async fn scrape_one(
    handle: BrowserHandle,
    request: ScrapeRequest,
    options: ScrapeManyOptions,
) -> Value {
    let started = std::time::Instant::now();
    let row = match handle
        .scrape_isolated(
            &request.url,
            options.navigate_options,
            options.focus.as_deref(),
            options.max_html_bytes,
            options.max_markdown_chars,
        )
        .await
    {
        Ok(value) => {
            if value.get("ok").and_then(Value::as_bool) == Some(false) {
                value
            } else {
                json!({
                    "url": request.url,
                    "ok": true,
                    "requested_url": value.get("requested_url").cloned().unwrap_or(Value::Null),
                    "final_url": value.get("final_url").cloned().unwrap_or(Value::Null),
                    "status": value.get("status").cloned().unwrap_or(Value::Null),
                    "title": value.get("title").cloned().unwrap_or(Value::Null),
                    "markdown": value.get("markdown").cloned().unwrap_or_else(|| json!("")),
                    "truncated": value.get("truncated").cloned().unwrap_or(json!(false)),
                    "blocked": value.get("blocked").cloned().unwrap_or(json!(false)),
                    "blocked_reason": value.get("blocked_reason").cloned().unwrap_or(Value::Null),
                })
            }
        }
        Err(err) => json!({
            "url": request.url,
            "ok": false,
            "error": "scrape_failed",
            "message": err.to_string(),
        }),
    };
    let ok = row.get("ok").and_then(Value::as_bool).unwrap_or(false);
    tracing::info!(
        target: "charlotte::scrape_many",
        url = %request.url,
        ok,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "scrape url finished"
    );
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_urls_rejects_empty_and_oversized() {
        assert!(validate_urls(vec![]).is_err());
        let many = (0..(SCRAPE_MANY_MAX_URLS + 1))
            .map(|i| format!("https://example.com/{i}"))
            .collect::<Vec<_>>();
        assert!(validate_urls(many).is_err());
    }

    #[test]
    fn clamp_concurrency_stays_in_bounds() {
        assert_eq!(clamp_concurrency(Some(0), 4), 1);
        assert_eq!(clamp_concurrency(Some(99), 4), SCRAPE_MAX_CONCURRENCY);
        assert_eq!(clamp_concurrency(None, 1), SCRAPE_DEFAULT_CONCURRENCY);
    }

    #[test]
    fn apply_total_text_cap_truncates_following_rows() {
        let mut rows = vec![
            json!({"markdown":"abcdef"}),
            json!({"markdown":"ghijk"}),
            json!({"markdown":"lmnop"}),
        ];
        let truncated = apply_total_text_cap(&mut rows, 8);
        assert!(truncated);
        assert_eq!(rows[0]["markdown"], "abcdef");
        assert_eq!(rows[1]["markdown"], "gh");
        assert_eq!(rows[2]["markdown"], "");
        assert_eq!(rows[1]["truncated_total"], true);
        assert_eq!(rows[2]["truncated_total"], true);
    }
}
