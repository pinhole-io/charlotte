//! Error types for Charlotte browser and tools.

use std::time::Duration;

use thiserror::Error;

/// Why [`CharlotteError::Overloaded`] refused a crawl.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverloadReason {
    /// Every admission slot is in use.
    Admission,
    /// Process memory is at or above the soft limit.
    Memory,
    /// The cgroup task count is at or above the soft limit.
    Tasks,
}

impl OverloadReason {
    /// Stable token for logs and HTTP bodies.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admission => "admission",
            Self::Memory => "memory",
            Self::Tasks => "tasks",
        }
    }
}

impl std::fmt::Display for OverloadReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors from Playwright, LLM brief step, and Charlotte session management.
#[derive(Debug, Error)]
pub enum CharlotteError {
    #[error("browser session is closed")]
    Closed,
    #[error(transparent)]
    Playwright(#[from] playwright_rs::Error),
    #[error(transparent)]
    Chat(#[from] superglue::chat::ChatError),
    #[error("research brief step returned no assistant text")]
    BriefEmpty,
    /// The process refused the crawl. The caller should retry after `retry_after`.
    #[error("crawl overloaded: {reason}")]
    Overloaded {
        /// Which gate refused the crawl.
        reason: OverloadReason,
        /// Suggested delay before another attempt.
        retry_after: Duration,
    },
    /// The caller stopped waiting. The crawl may still be running.
    #[error("crawl exceeded the deadline of {} seconds", deadline.as_secs_f64())]
    DeadlineExceeded {
        /// Deadline the caller used.
        deadline: Duration,
    },
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Regex(#[from] regex::Error),
}

impl CharlotteError {
    pub fn msg(s: impl Into<String>) -> Self {
        CharlotteError::Message(s.into())
    }
}
