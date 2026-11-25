//! Several Chromium processes on one Playwright driver.
//!
//! A dead process is marked down, closed, and launched again. The other
//! processes keep serving.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use playwright_rs::{Browser, Playwright};
use serde_json::Value;
use tokio::sync::{Notify, Semaphore};

use crate::browser::playwright::{self, FreshScrapeError};
use crate::browser::{BrowserConfig, NavigateOptions};
use crate::error::CharlotteError;

struct SlotState {
    ready: bool,
    active_pages: usize,
    restart_count: u32,
    restarting: bool,
}

/// Round-robin board for browser slots. Tests drive this without Chromium.
pub(crate) struct BrowserSlotBoard {
    states: Mutex<Vec<SlotState>>,
    next: AtomicUsize,
    idle: Vec<Notify>,
}

impl BrowserSlotBoard {
    pub(crate) fn new(size: usize) -> Self {
        let states = (0..size)
            .map(|_| SlotState {
                ready: false,
                active_pages: 0,
                restart_count: 0,
                restarting: false,
            })
            .collect();
        let idle = (0..size).map(|_| Notify::new()).collect();
        Self {
            states: Mutex::new(states),
            next: AtomicUsize::new(0),
            idle,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Next ready slot, skipping `excluded` and slots that are down.
    pub(crate) fn select(&self, excluded: &HashSet<usize>) -> Option<usize> {
        let states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let len = states.len();
        if len == 0 {
            return None;
        }
        let start = self.next.load(Ordering::Relaxed);
        for offset in 0..len {
            let index = (start + offset) % len;
            if excluded.contains(&index) || !states[index].ready {
                continue;
            }
            self.next.store((index + 1) % len, Ordering::Relaxed);
            return Some(index);
        }
        None
    }

    pub(crate) fn mark_ready(&self, index: usize) {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(state) = states.get_mut(index) {
            state.ready = true;
            state.restarting = false;
        }
    }

    /// Mark a slot down and count one restart. Returns false when a restart is already recorded.
    pub(crate) fn begin_restart(&self, index: usize) -> bool {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(state) = states.get_mut(index) else {
            return false;
        };
        if state.restarting {
            return false;
        }
        state.ready = false;
        state.restarting = true;
        state.restart_count = state.restart_count.saturating_add(1);
        true
    }

    #[cfg(test)]
    pub(crate) fn restart_count(&self, index: usize) -> u32 {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(index)
            .map(|state| state.restart_count)
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn is_ready(&self, index: usize) -> bool {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(index)
            .is_some_and(|state| state.ready)
    }

    fn acquire_page(&self, index: usize) -> bool {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(state) = states.get_mut(index) else {
            return false;
        };
        if !state.ready {
            return false;
        }
        state.active_pages = state.active_pages.saturating_add(1);
        true
    }

    fn release_page(&self, index: usize) {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(state) = states.get_mut(index) else {
            return;
        };
        state.active_pages = state.active_pages.saturating_sub(1);
        if state.active_pages == 0 {
            self.idle[index].notify_waiters();
        }
    }

    fn active_pages(&self, index: usize) -> usize {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(index)
            .map(|state| state.active_pages)
            .unwrap_or(0)
    }

    async fn wait_until_idle(&self, index: usize) {
        loop {
            let notified = self.idle[index].notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.active_pages(index) == 0 {
                return;
            }
            notified.await;
        }
    }

    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    pub(crate) fn ready_count(&self) -> usize {
        self.states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|state| state.ready)
            .count()
    }

    fn finish_failed_restart(&self, index: usize) {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(state) = states.get_mut(index) {
            state.restarting = false;
            state.ready = false;
        }
    }
}

struct PoolInner {
    playwright: Playwright,
    config: BrowserConfig,
    board: BrowserSlotBoard,
    browsers: Mutex<Vec<Option<Browser>>>,
    pages: Semaphore,
    closing: AtomicBool,
}

/// One Playwright driver and several Chromium processes.
#[derive(Clone)]
pub(crate) struct ResilientBrowserPool {
    inner: Arc<PoolInner>,
}

impl ResilientBrowserPool {
    pub(crate) async fn start(
        pool_size: usize,
        max_pages: usize,
        config: BrowserConfig,
    ) -> Result<Self, CharlotteError> {
        let playwright = Playwright::launch().await?;
        let mut launched = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            match playwright::launch_chromium(&playwright, &config).await {
                Ok(browser) => launched.push(browser),
                Err(error) => {
                    for browser in launched {
                        let _ = browser.close().await;
                    }
                    let _ = playwright.shutdown().await;
                    return Err(error);
                }
            }
        }
        let board = BrowserSlotBoard::new(pool_size);
        for index in 0..pool_size {
            board.mark_ready(index);
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                playwright,
                config,
                board,
                browsers: Mutex::new(launched.into_iter().map(Some).collect()),
                pages: Semaphore::new(max_pages),
                closing: AtomicBool::new(false),
            }),
        })
    }

    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    pub(crate) fn ready_browser_count(&self) -> usize {
        if self.inner.closing.load(Ordering::Relaxed) {
            return 0;
        }
        self.inner.board.ready_count()
    }

    pub(crate) async fn scrape(
        &self,
        url: &str,
        options: NavigateOptions,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError> {
        if self.inner.closing.load(Ordering::Relaxed) {
            return Err(CharlotteError::Closed);
        }
        if self.inner.pages.available_permits() == 0 {
            tracing::info!(url, "waiting for a browser page slot");
        }
        let _page_permit = self
            .inner
            .pages
            .acquire()
            .await
            .map_err(|_| CharlotteError::msg("browser page slots are closed"))?;
        tracing::info!(url, "browser page slot held");
        let mut attempted = HashSet::new();
        let slot_count = self.inner.board.len();
        while attempted.len() < slot_count {
            let Some(index) = self.inner.board.select(&attempted) else {
                break;
            };
            attempted.insert(index);
            tracing::info!(url, slot = index, "selected browser slot");
            let Some(browser) = self.acquire_browser(index) else {
                tracing::warn!(url, slot = index, "browser slot unavailable; restarting");
                self.schedule_restart(index);
                continue;
            };
            let scraped = playwright::scrape_in_fresh_context(
                &browser,
                &self.inner.config,
                url,
                options.clone(),
                focus,
                max_html_bytes,
                max_markdown_chars,
            )
            .await;
            self.inner.board.release_page(index);
            match scraped {
                Ok(value) => return Ok(value),
                Err(FreshScrapeError::Disconnected) => {
                    tracing::warn!(url, slot = index, "browser disconnected; restarting");
                    self.schedule_restart(index);
                }
                Err(FreshScrapeError::Failed(error)) => return Err(error),
            }
        }
        Err(CharlotteError::msg("no browser slots are available"))
    }

    pub(crate) async fn close(&self) -> Result<(), CharlotteError> {
        if self.inner.closing.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.inner.pages.close();
        let browsers = {
            let mut guard = self
                .inner
                .browsers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard
                .iter_mut()
                .filter_map(Option::take)
                .collect::<Vec<_>>()
        };
        let mut first_error = None;
        for browser in browsers {
            if let Err(error) = browser.close().await {
                first_error.get_or_insert(CharlotteError::from(error));
            }
        }
        if let Err(error) = self.inner.playwright.shutdown().await {
            first_error.get_or_insert(CharlotteError::from(error));
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn acquire_browser(&self, index: usize) -> Option<Browser> {
        let browser = {
            let guard = self
                .inner
                .browsers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.get(index).and_then(Clone::clone)
        }?;
        if !browser.is_connected() || !self.inner.board.acquire_page(index) {
            return None;
        }
        Some(browser)
    }

    fn schedule_restart(&self, index: usize) {
        if self.inner.closing.load(Ordering::Relaxed) || !self.inner.board.begin_restart(index) {
            return;
        }
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            restart_slot(inner, index).await;
        });
    }
}

async fn restart_slot(inner: Arc<PoolInner>, index: usize) {
    if inner.closing.load(Ordering::Relaxed) {
        return;
    }
    inner.board.wait_until_idle(index).await;
    if inner.closing.load(Ordering::Relaxed) {
        return;
    }
    let old = {
        let mut guard = inner
            .browsers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get_mut(index).and_then(Option::take)
    };
    if let Some(browser) = old {
        let _ = browser.close().await;
    }
    if inner.closing.load(Ordering::Relaxed) {
        return;
    }
    match playwright::launch_chromium(&inner.playwright, &inner.config).await {
        Ok(browser) => {
            {
                let mut guard = inner
                    .browsers
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(slot) = guard.get_mut(index) {
                    *slot = Some(browser);
                }
            }
            if !inner.closing.load(Ordering::Relaxed) {
                inner.board.mark_ready(index);
                tracing::info!(index, "browser slot restarted");
            }
        }
        Err(error) => {
            tracing::warn!(index, %error, "browser slot restart failed");
            inner.board.finish_failed_restart(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_skips_a_down_slot_and_records_a_restart() {
        let board = BrowserSlotBoard::new(2);
        board.mark_ready(0);
        board.mark_ready(1);

        assert_eq!(board.select(&HashSet::new()), Some(0));
        assert!(board.begin_restart(0));
        assert_eq!(board.restart_count(0), 1);
        assert!(!board.is_ready(0));
        assert!(!board.begin_restart(0));
        assert_eq!(board.restart_count(0), 1);
        assert_eq!(board.select(&HashSet::new()), Some(1));
    }
}
