use async_trait::async_trait;
use serde_json::Value;

use crate::error::CharlotteError;

use super::NavigateOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrowserBackendKind {
    #[default]
    Playwright,
    ObscuraCdp,
}

impl BrowserBackendKind {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "playwright" => Some(Self::Playwright),
            "obscura" | "cdp" | "obscura-cdp" => Some(Self::ObscuraCdp),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Playwright => "playwright",
            Self::ObscuraCdp => "obscura-cdp",
        }
    }
}

#[async_trait]
pub trait BrowserBackend: Send + Sync {
    fn backend_id(&self) -> &'static str;

    async fn close(&self) -> Result<(), CharlotteError>;
    async fn enforce_url_policy(&self, url: &str) -> Result<(), CharlotteError>;
    async fn navigate_with_options(
        &self,
        url: &str,
        options: NavigateOptions,
    ) -> Result<Value, CharlotteError>;
    async fn web_search(&self, query: &str, max_results: usize) -> Result<Value, CharlotteError>;
    async fn page_content_with_max_chars(
        &self,
        selector: Option<&str>,
        max_chars: usize,
    ) -> Result<Value, CharlotteError>;
    async fn page_html(&self) -> Result<String, CharlotteError>;
    async fn page_markdown(
        &self,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError>;
    async fn list_links(
        &self,
        selector: Option<&str>,
        limit: usize,
        same_origin_only: bool,
    ) -> Result<Value, CharlotteError>;
    async fn scroll_by(&self, delta_x: i32, delta_y: i32) -> Result<Value, CharlotteError>;
    async fn scroll_into_view(&self, selector: &str) -> Result<Value, CharlotteError>;
    async fn click(&self, selector: &str, timeout_ms: Option<f64>)
    -> Result<Value, CharlotteError>;
    async fn type_text(
        &self,
        selector: &str,
        text: &str,
        delay_ms: Option<f64>,
    ) -> Result<Value, CharlotteError>;
    async fn fill(&self, selector: &str, text: &str) -> Result<Value, CharlotteError>;
    async fn press_key(&self, key: &str) -> Result<Value, CharlotteError>;
    async fn select_option(
        &self,
        selector: &str,
        value: Option<&str>,
        label: Option<&str>,
        index: Option<usize>,
    ) -> Result<Value, CharlotteError>;
    async fn wait_for_selector(
        &self,
        selector: &str,
        timeout_ms: Option<f64>,
    ) -> Result<Value, CharlotteError>;
    async fn screenshot(&self, full_page: bool) -> Result<Value, CharlotteError>;
    async fn list_tabs(&self) -> Result<Value, CharlotteError>;
    async fn switch_tab(
        &self,
        index: Option<usize>,
        url_contains: Option<&str>,
    ) -> Result<Value, CharlotteError>;
    async fn evaluate_js(&self, script: &str, arg: Option<Value>) -> Result<Value, CharlotteError>;
    async fn cdp_call(&self, method: &str, params: Option<Value>) -> Result<Value, CharlotteError>;
    async fn network_requests(&self, limit: usize) -> Result<Value, CharlotteError>;
    async fn console_messages(&self, limit: usize) -> Result<Value, CharlotteError>;
    async fn max_text_chars_limit(&self) -> Result<usize, CharlotteError>;
    async fn max_markdown_chars_limit(&self) -> Result<usize, CharlotteError>;
    async fn max_html_bytes_limit(&self) -> Result<usize, CharlotteError>;
    /// Open a fresh tab, navigate, extract markdown, then close the tab (does not touch the shared page).
    async fn scrape_isolated(
        &self,
        url: &str,
        options: NavigateOptions,
        focus: Option<&str>,
        max_html_bytes: usize,
        max_markdown_chars: usize,
    ) -> Result<Value, CharlotteError>;
}
