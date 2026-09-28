use std::sync::Arc;

use crate::error::CharlotteError;

use super::BrowserConfig;
use super::backend::BrowserBackend;
use super::playwright::PlaywrightBackend;

pub struct ObscuraCdpBackend;

impl ObscuraCdpBackend {
    pub async fn create(
        config: BrowserConfig,
        endpoint_url: &str,
    ) -> Result<Arc<dyn BrowserBackend>, CharlotteError> {
        let backend = PlaywrightBackend::connect_over_cdp(config, endpoint_url).await?;
        Ok(Arc::new(backend))
    }
}
