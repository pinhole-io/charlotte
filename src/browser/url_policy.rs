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
        let u = self.parse(url_str)?;
        let host = u
            .host_str()
            .ok_or_else(|| CharlotteError::msg("URL has no host"))?;

        if self.block_private_ips {
            if let Ok(ip) = host.parse::<IpAddr>() {
                reject_non_public_ip(ip)?;
            } else if is_metadata_hostname(host) {
                return Err(CharlotteError::msg("blocked metadata-style host"));
            }
        }

        if !self.allow_host_suffixes.is_empty()
            && !host_matches_allowlist(host, &self.allow_host_suffixes)
        {
            return Err(CharlotteError::msg("host is not in the URL allow-list"));
        }

        Ok(())
    }

    /// Validate the URL and every address returned by DNS.
    pub async fn check_resolved(&self, url_str: &str) -> Result<(), CharlotteError> {
        self.check(url_str)?;
        if !self.block_private_ips {
            return Ok(());
        }
        let url = self.parse(url_str)?;
        let host = url
            .host_str()
            .ok_or_else(|| CharlotteError::msg("URL has no host"))?;
        if host.parse::<IpAddr>().is_ok() {
            return Ok(());
        }
        let port = url
            .port_or_known_default()
            .ok_or_else(|| CharlotteError::msg("URL has no network port"))?;
        let addresses = tokio::net::lookup_host((host, port))
            .await
            .map_err(|_| CharlotteError::msg("URL host could not be resolved"))?
            .map(|address| address.ip())
            .collect::<Vec<_>>();
        if addresses.is_empty() {
            return Err(CharlotteError::msg("URL host resolved to no addresses"));
        }
        for address in addresses {
            reject_non_public_ip(address)?;
        }
        Ok(())
    }

    fn parse(&self, url_str: &str) -> Result<Url, CharlotteError> {
        let u = Url::parse(url_str).map_err(|_| CharlotteError::msg("invalid URL"))?;
        let scheme = u.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(CharlotteError::msg("URL scheme is not allowed"));
        }
        if !u.username().is_empty() || u.password().is_some() {
            return Err(CharlotteError::msg("URL user information is not allowed"));
        }
        Ok(u)
    }
}

fn reject_non_public_ip(ip: IpAddr) -> Result<(), CharlotteError> {
    if is_non_public_ip(ip) {
        Err(CharlotteError::msg("blocked non-public network address"))
    } else {
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
            let octets = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || v4.is_multicast()
                || octets[0] == 0
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 198 && matches!(octets[1], 18 | 19))
                || octets[0] >= 240
        }
        IpAddr::V6(v6) => {
            let segments = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6.is_multicast()
                || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        }
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
    fn permissive_still_rejects_non_http_urls_and_credentials() {
        let p = UrlPolicy::permissive();
        assert!(p.check("file:///etc/passwd").is_err());
        assert!(p.check("https://user:secret@example.com/").is_err());
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

    #[test]
    fn non_public_ranges_include_cgnat_and_documentation_addresses() {
        for address in ["100.64.0.1", "198.18.0.1", "2001:db8::1"] {
            let ip = address.parse().expect("static IP");
            assert!(is_non_public_ip(ip), "{address} should be blocked");
        }
    }
}
