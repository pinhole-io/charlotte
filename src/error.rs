//! Error types for Charlotte browser and tools.

use thiserror::Error;

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
