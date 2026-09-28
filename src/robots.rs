//! Optional `robots.txt` checks (default off). See [`crate::CrawlRunConfig::respect_robots`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::instrument;
use url::Url;

use crate::CharlotteError;

/// Fetches and caches `robots.txt` per origin (scheme + host + port).
pub struct RobotsFetcher {
    client: reqwest::Client,
    cache: Mutex<HashMap<String, Arc<SimpleRobots>>>,
}

impl RobotsFetcher {
    /// New HTTP client for robots.txt only (short timeout).
    pub fn new() -> Result<Self, CharlotteError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("Charlotte/0.1")
            .build()
            .map_err(|e| CharlotteError::msg(format!("robots HTTP client: {e}")))?;
        Ok(Self {
            client,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Returns `false` when `Disallow` rules for `User-agent: *` block `url`'s path.
    #[instrument(skip(self), fields(url = %page_url))]
    pub async fn is_allowed(&self, page_url: &str) -> Result<bool, CharlotteError> {
        let u = Url::parse(page_url).map_err(|e| CharlotteError::msg(e.to_string()))?;
        let robots_url = robots_txt_url(&u)?;
        let key = origin_key(&u);

        {
            let g = self.cache.lock().await;
            if let Some(parsed) = g.get(&key) {
                return Ok(parsed_allows(parsed, &u));
            }
        }

        let parsed = self.fetch_parse(&robots_url).await?;
        let parsed = Arc::new(parsed);
        {
            let mut g = self.cache.lock().await;
            g.insert(key, Arc::clone(&parsed));
        }
        Ok(parsed_allows(&parsed, &u))
    }

    async fn fetch_parse(&self, robots_url: &Url) -> Result<SimpleRobots, CharlotteError> {
        let resp = self
            .client
            .get(robots_url.as_str())
            .send()
            .await
            .map_err(|e| CharlotteError::msg(format!("robots.txt fetch: {e}")))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(SimpleRobots::default());
        }
        if !status.is_success() {
            tracing::warn!(%status, url = %robots_url, "robots.txt non-success; treating as empty");
            return Ok(SimpleRobots::default());
        }
        let text = resp
            .text()
            .await
            .map_err(|e| CharlotteError::msg(format!("robots.txt body: {e}")))?;
        Ok(SimpleRobots::parse(&text))
    }
}

fn origin_key(u: &Url) -> String {
    let mut o = format!("{}://{}", u.scheme(), u.host_str().unwrap_or(""));
    if let Some(p) = u.port() {
        o.push(':');
        o.push_str(&p.to_string());
    }
    o
}

fn robots_txt_url(page: &Url) -> Result<Url, CharlotteError> {
    let mut s = origin_key(page);
    s.push_str("/robots.txt");
    Url::parse(&s).map_err(|e| CharlotteError::msg(e.to_string()))
}

fn parsed_allows(rules: &SimpleRobots, page: &Url) -> bool {
    let path = page.path();
    let path = if path.is_empty() { "/" } else { path };
    rules.allows(path)
}

#[derive(Debug, Default)]
struct SimpleRobots {
    /// Longest-prefix wins handled by checking every disallow as prefix (naive).
    disallow_prefixes: Vec<String>,
}

impl SimpleRobots {
    fn parse(text: &str) -> Self {
        let mut ua_star = false;
        let mut disallow_prefixes = Vec::new();
        for line in text.lines() {
            let t = line.split('#').next().unwrap_or("").trim();
            if t.is_empty() {
                continue;
            }
            let tl = t.to_ascii_lowercase();
            if tl.starts_with("user-agent:") {
                let ua = t.split(':').nth(1).map(str::trim).unwrap_or("");
                ua_star = ua == "*";
                continue;
            }
            if ua_star && tl.starts_with("disallow:") {
                let p = t.split(':').nth(1).map(str::trim).unwrap_or("").to_string();
                disallow_prefixes.push(p);
            }
        }
        Self { disallow_prefixes }
    }

    fn allows(&self, path: &str) -> bool {
        for p in &self.disallow_prefixes {
            if p.is_empty() {
                continue;
            }
            if path.starts_with(p) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_disallow_star() {
        let body = r#"
User-agent: *
Disallow: /private/
"#;
        let r = SimpleRobots::parse(body);
        assert!(!r.allows("/private/x"));
        assert!(r.allows("/public/"));
    }
}
