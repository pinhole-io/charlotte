//! Shared browser process with one isolated context per Charlotte session.

use std::sync::Arc;

use crate::CharlotteError;
use secrecy::{ExposeSecret, SecretString};

use super::playwright::PlaywrightBrowserHost;
use super::{BrowserBackendKind, BrowserConfig, BrowserHandle};

/// Programmatic configuration for a shared browser host.
#[derive(Clone)]
pub struct BrowserHostConfig {
    /// Browser backend to start.
    pub backend: BrowserBackendKind,
    /// Browser and per-context limits.
    pub browser: BrowserConfig,
    /// CDP endpoint for [`BrowserBackendKind::ObscuraCdp`].
    pub cdp_endpoint: Option<SecretString>,
}

impl BrowserHostConfig {
    /// Validate settings before the browser process starts.
    pub fn validate(&self) -> Result<(), CharlotteError> {
        if self.browser.navigation_timeout.is_zero() {
            return Err(CharlotteError::msg(
                "browser navigation timeout must be greater than zero",
            ));
        }
        if self.browser.max_text_chars == 0
            || self.browser.max_markdown_chars == 0
            || self.browser.max_html_bytes == 0
        {
            return Err(CharlotteError::msg(
                "browser content limits must be greater than zero",
            ));
        }
        if !self.browser.search_url_template.contains("{query}") {
            return Err(CharlotteError::msg(
                "browser search URL template must contain {query}",
            ));
        }
        if self.backend == BrowserBackendKind::ObscuraCdp
            && self
                .cdp_endpoint
                .as_ref()
                .is_none_or(|endpoint| endpoint.expose_secret().trim().is_empty())
        {
            return Err(CharlotteError::msg(
                "CDP endpoint is required for the CDP browser backend",
            ));
        }
        Ok(())
    }
}

/// One shared Playwright driver and browser process.
#[derive(Clone)]
pub struct BrowserHost {
    inner: Arc<PlaywrightBrowserHost>,
}

impl BrowserHost {
    /// Start the configured browser backend.
    pub async fn start(config: BrowserHostConfig) -> Result<Self, CharlotteError> {
        config.validate()?;
        let inner = PlaywrightBrowserHost::start(
            config.backend,
            config.browser,
            config
                .cdp_endpoint
                .as_ref()
                .map(ExposeSecret::expose_secret),
        )
        .await?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Open an isolated browser context for one host runtime.
    pub async fn open_session(&self) -> Result<BrowserHandle, CharlotteError> {
        let backend = self.inner.open_backend().await?;
        Ok(BrowserHandle::from_backend(backend))
    }

    /// Close the browser process and Playwright driver.
    ///
    /// Close every session before this method. Repeated calls are no-ops.
    pub async fn close(&self) -> Result<(), CharlotteError> {
        self.inner.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdp_host_requires_an_endpoint() {
        let config = BrowserHostConfig {
            backend: BrowserBackendKind::ObscuraCdp,
            browser: BrowserConfig::default(),
            cdp_endpoint: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn browser_limits_must_be_nonzero() {
        let mut browser = BrowserConfig::default();
        browser.max_html_bytes = 0;
        let config = BrowserHostConfig {
            backend: BrowserBackendKind::Playwright,
            browser,
            cdp_endpoint: None,
        };
        assert!(config.validate().is_err());
    }
}
