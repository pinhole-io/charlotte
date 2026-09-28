//! Heuristic substring blocklist for common analytics/ads hosts (not a full adblocker).

/// Returns true if the URL likely points at analytics, ads, or social tracking pixels.
pub(crate) fn url_looks_like_tracker_or_ad(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    TRACKER_SUBSTRINGS.iter().any(|s| lower.contains(s))
}

/// Substrings matched against the full request URL (scheme + host + path).
const TRACKER_SUBSTRINGS: &[&str] = &[
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "adservice.google",
    "pagead2.googlesyndication",
    "googlesyndication.com",
    "googleadservices.com",
    "facebook.net/tr",
    "facebook.com/tr",
    "connect.facebook.net",
    "scorecardresearch.com",
    "hotjar.com",
    "ads.linkedin.com",
    "px.ads.linkedin.com",
    "adsafeprotected.com",
    "adnxs.com",
    "amazon-adsystem.com",
    "criteo.com",
    "taboola.com",
    "outbrain.com",
    "moatads.com",
    "chartbeat.net",
    "segment.io",
    "segment.com",
    "cdn.segment.com",
    "mixpanel.com",
    "mxpnl.com",
    "clarity.ms",
    "browser.sentry-cdn.com",
    "plausible.io",
    "hm.baidu.com",
    "static.ads-twitter.com",
    "analytics.twitter.com",
    "widgets.outbrain.com",
    "cdn.optimizely.com",
    "bat.bing.com",
];
