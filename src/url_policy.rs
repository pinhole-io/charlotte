//! Optional URL policy for mitigating SSRF-style abuse when navigated URLs are user-controlled.
//!
//! When [`UrlPolicy::is_restricted`] is false (default), every `http`/`https` URL is allowed.
//! Otherwise, host suffix rules and private IP blocking apply.

use std::net::IpAddr;

use url::Url;

use crate::CharlotteError;

/// Host allow-list and private-IP controls for outbound navigations.
#[derive(Debug, Clone, Default)]
pub struct UrlPolicy {
    /// When non-empty, the URL host must equal one of these entries or end with `.{suffix}`.
    /// Example: `example.com` matches `https://a.example.com/path`.
    pub allow_host_suffixes: Vec<String>,
    /// Reject hostnames that parse as loopback, private, or link-local addresses.
    pub block_private_ips: bool,
}

impl UrlPolicy {
    /// No restrictions (legacy behaviour).
    #[must_use]
    pub fn permissive() -> Self {
        Self::default()
    }

    /// True when any enforcement is active.
    #[must_use]
    pub fn is_restricted(&self) -> bool {
        !self.allow_host_suffixes.is_empty() || self.block_private_ips
    }

    /// Validate `url_str` before navigation or crawl seeding.
    pub fn check(&self, url_str: &str) -> Result<(), CharlotteError> {
        if !self.is_restricted() {
            return Ok(());
        }
        let u =
            Url::parse(url_str).map_err(|e| CharlotteError::msg(format!("invalid URL: {e}")))?;
        let scheme = u.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(CharlotteError::msg(format!(
                "URL scheme not allowed: {scheme}"
            )));
        }
        let host = u
            .host_str()
            .ok_or_else(|| CharlotteError::msg("URL has no host"))?;

        if self.block_private_ips {
            if let Ok(ip) = host.parse::<IpAddr>() {
                if is_non_public_ip(ip) {
                    return Err(CharlotteError::msg(format!(
                        "blocked non-public IP host: {host}"
                    )));
                }
            } else if is_metadata_hostname(host) {
                return Err(CharlotteError::msg(format!(
                    "blocked metadata-style host: {host}"
                )));
            }
        }

        if !self.allow_host_suffixes.is_empty()
            && !host_matches_allowlist(host, &self.allow_host_suffixes)
        {
            return Err(CharlotteError::msg(format!(
                "host not in allow-list: {host}"
            )));
        }

        Ok(())
    }
}

fn is_metadata_hostname(host: &str) -> bool {
    let h = host.to_ascii_lowercase();
    h.contains("metadata.google.internal") || h == "metadata" || h.ends_with(".internal")
}

fn host_matches_allowlist(host: &str, suffixes: &[String]) -> bool {
    let h = host.to_ascii_lowercase();
    for s in suffixes {
        let t = s.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        if h == t || h.ends_with(&format!(".{t}")) {
            return true;
        }
    }
    false
}

fn is_non_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.octets()[0] == 0
        }
        IpAddr::V6(v6) => v6.is_loopback() || v6.is_unique_local() || v6.is_unicast_link_local(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permissive_allows_https() {
        let p = UrlPolicy::permissive();
        p.check("https://example.com/a").unwrap();
    }

    #[test]
    fn blocks_loopback_when_private_blocked() {
        let p = UrlPolicy {
            allow_host_suffixes: vec![],
            block_private_ips: true,
        };
        assert!(p.check("http://127.0.0.1/").is_err());
    }

    #[test]
    fn allowlist_blocks_unknown_host() {
        let p = UrlPolicy {
            allow_host_suffixes: vec!["example.com".to_string()],
            block_private_ips: false,
        };
        p.check("https://example.com/").unwrap();
        assert!(p.check("https://evil.com/").is_err());
    }

    #[test]
    fn allowlist_subdomain() {
        let p = UrlPolicy {
            allow_host_suffixes: vec!["example.com".to_string()],
            block_private_ips: false,
        };
        p.check("https://docs.example.com/x").unwrap();
    }
}
