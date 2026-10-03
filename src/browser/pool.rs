//! Round-robin access to several [`BrowserHandle`]s (e.g. one browser per worker).

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::browser::{BrowserConfig, BrowserHandle};
use crate::error::CharlotteError;

/// Maximum browsers used by [`default_browser_count`] (each Chromium is memory-heavy).
pub const DEFAULT_BROWSER_POOL_CAP: usize = 8;

/// Suggested pool size: about half of logical CPUs, at least one, at most [`DEFAULT_BROWSER_POOL_CAP`].
///
/// Override via [`BrowserPool::launch`]'s `count` argument when you need a fixed size.
pub fn default_browser_count() -> usize {
    let p = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(2);
    let n = (p / 2).max(1);
    n.min(DEFAULT_BROWSER_POOL_CAP)
}

/// Distributes work across multiple browser instances.
pub struct BrowserPool {
    handles: Vec<BrowserHandle>,
    next: AtomicUsize,
}

impl BrowserPool {
    /// Returns `None` when `handles` is empty.
    pub fn new(handles: Vec<BrowserHandle>) -> Option<Self> {
        if handles.is_empty() {
            return None;
        }
        Some(Self {
            handles,
            next: AtomicUsize::new(0),
        })
    }

    /// Build several browser handles with env-selected backend settings.
    ///
    /// `count`: number of browser handles (`None` → [`default_browser_count`]`). Values are clamped to at least 1.
    pub async fn from_config(
        config: BrowserConfig,
        count: Option<usize>,
    ) -> Result<Self, CharlotteError> {
        let n = count.unwrap_or_else(default_browser_count).max(1);
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            handles.push(BrowserHandle::from_config(config.clone()).await?);
        }
        Ok(Self {
            handles,
            next: AtomicUsize::new(0),
        })
    }

    /// Launch several isolated Chromium instances with the same [`BrowserConfig`].
    ///
    /// `count`: number of browsers (`None` → [`default_browser_count`]`). Values are clamped to at least 1.
    pub async fn launch(
        config: BrowserConfig,
        count: Option<usize>,
    ) -> Result<Self, CharlotteError> {
        Self::from_config(config, count).await
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// All handles (e.g. pin one worker per index for parallel crawls).
    pub fn handles(&self) -> &[BrowserHandle] {
        &self.handles
    }

    /// Next handle in round-robin order (each handle remains internally mutex-serialized).
    pub fn next(&self) -> &BrowserHandle {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.handles.len();
        &self.handles[i]
    }

    /// Close every browser in the pool.
    pub async fn close_all(&self) -> Result<(), CharlotteError> {
        for h in &self.handles {
            h.close().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_browser_count_in_range() {
        let n = default_browser_count();
        assert!(n >= 1);
        assert!(n <= DEFAULT_BROWSER_POOL_CAP);
    }
}
