//! Integration test: Obscura/CDP endpoint via Playwright client.

use charlotte::{BrowserConfig, BrowserHandle};

#[tokio::test]
#[ignore = "requires running CDP endpoint (set CHARLOTTE_CDP_ENDPOINT)"]
async fn navigate_and_read_example_com_over_cdp() {
    let endpoint = std::env::var("CHARLOTTE_CDP_ENDPOINT")
        .expect("set CHARLOTTE_CDP_ENDPOINT, e.g. http://127.0.0.1:9222");
    let browser = BrowserHandle::connect_over_cdp(BrowserConfig::default(), &endpoint)
        .await
        .expect("cdp connect");

    let nav = browser
        .navigate("https://example.com/")
        .await
        .expect("goto");
    assert!(
        nav.get("url")
            .and_then(|u| u.as_str())
            .unwrap_or("")
            .contains("example.com")
    );

    let md_val = browser
        .page_markdown(None, 600_000, 32_000)
        .await
        .expect("page_markdown");
    let md = md_val
        .get("markdown")
        .and_then(|m| m.as_str())
        .unwrap_or("");
    assert!(
        md.to_lowercase().contains("example") || md.to_lowercase().contains("domain"),
        "unexpected markdown: {md:?}"
    );

    browser.close().await.expect("close");
}
