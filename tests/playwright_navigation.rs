//! Integration test: real Chromium via Playwright. Ignored so `cargo test` does not require browsers.

use charlotte::BrowserHandle;

#[tokio::test]
#[ignore = "requires Playwright browsers (see charlotte README)"]
async fn navigate_and_read_example_com() {
    let browser = BrowserHandle::new(Default::default())
        .await
        .expect("browser launch");
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

    let content = browser.page_content(None).await.expect("content");
    let text = content.get("text").and_then(|t| t.as_str()).unwrap_or("");
    assert!(
        text.to_lowercase().contains("example") || text.to_lowercase().contains("domain"),
        "unexpected body text: {text:?}"
    );

    let empty_sel = browser
        .page_content(Some(""))
        .await
        .expect("empty selector recovers");
    let text2 = empty_sel.get("text").and_then(|t| t.as_str()).unwrap_or("");
    assert!(
        !text2.is_empty(),
        "empty selector should behave like full page: {text2:?}"
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
