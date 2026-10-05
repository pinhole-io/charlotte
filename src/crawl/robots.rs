//! Optional `robots.txt` checks (default off). See [`crate::CrawlRunConfig::respect_robots`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::instrument;
use url::Url;

use crate::CharlotteError;
use crate::UrlPolicy;

const MAX_ROBOTS_BYTES: usize = 256 * 1024;
const MAX_REDIRECTS: usize = 5;

/// Fetches and caches `robots.txt` per origin (scheme + host + port).
pub struct RobotsFetcher {
    client: reqwest::Client,
    policy: UrlPolicy,
    cache: Mutex<HashMap<String, Arc<SimpleRobots>>>,
}

impl RobotsFetcher {
    /// New HTTP client for robots.txt only (short timeout).
    pub fn new() -> Result<Self, CharlotteError> {
        Self::with_policy(UrlPolicy::permissive())
    }

    /// New HTTP client that applies `policy` to every request and redirect.
    pub fn with_policy(policy: UrlPolicy) -> Result<Self, CharlotteError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("Charlotte/0.1")
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| CharlotteError::msg("failed to build robots HTTP client"))?;
        Ok(Self {
            client,
            policy,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Returns `false` when `Disallow` rules for `User-agent: *` block `url`'s path.
    #[instrument(skip_all)]
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
        let mut current = robots_url.clone();
        let mut redirects = 0usize;
        let mut resp = loop {
            self.policy.check_resolved(current.as_str()).await?;
            let response = self
                .client
                .get(current.as_str())
                .send()
                .await
                .map_err(|_| CharlotteError::msg("robots.txt fetch failed"))?;
            if !response.status().is_redirection() {
                break response;
            }
            if redirects >= MAX_REDIRECTS {
                return Err(CharlotteError::msg("robots.txt redirect limit exceeded"));
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| CharlotteError::msg("robots.txt redirect has no location"))?;
            current = current
                .join(location)
                .map_err(|_| CharlotteError::msg("robots.txt redirect URL is invalid"))?;
            redirects += 1;
        };
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(SimpleRobots::default());
        }
        if !status.is_success() {
            tracing::warn!(%status, "robots.txt non-success; treating as empty");
            return Ok(SimpleRobots::default());
        }
        if resp
            .content_length()
            .is_some_and(|length| length > MAX_ROBOTS_BYTES as u64)
        {
            return Err(CharlotteError::msg("robots.txt body is too large"));
        }
        let mut body = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|_| CharlotteError::msg("robots.txt body read failed"))?
        {
            if body.len().saturating_add(chunk.len()) > MAX_ROBOTS_BYTES {
                return Err(CharlotteError::msg("robots.txt body is too large"));
            }
            body.extend_from_slice(&chunk);
        }
        let text = std::str::from_utf8(&body)
            .map_err(|_| CharlotteError::msg("robots.txt body is not UTF-8"))?;
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
