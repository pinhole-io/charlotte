//! Integration test: real Chromium via Playwright. Ignored so `cargo test` does not require browsers.

use charlotte::{BrowserBackendKind, BrowserConfig, BrowserHandle, BrowserHost, BrowserHostConfig};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

#[tokio::test]
#[ignore = "requires Playwright browsers (see charlotte README)"]
async fn typed_credential_stays_in_one_hosted_context() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/login"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            r#"<html><body><input id="password" type="password"></body></html>"#,
            "text/html",
        ))
        .mount(&server)
        .await;
    let host = BrowserHost::start(BrowserHostConfig {
        backend: BrowserBackendKind::Playwright,
        browser: BrowserConfig::default(),
        cdp_endpoint: None,
    })
    .await
    .expect("browser host");
    let first = host.open_session().await.expect("first context");
    let second = host.open_session().await.expect("second context");
    let url = format!("{}/login", server.uri());
    first.navigate(&url).await.expect("first navigate");
    second.navigate(&url).await.expect("second navigate");

    first
        .fill("#password", "marker-password")
        .await
        .expect("type credential");
    let first_value = first
        .evaluate_js("() => document.querySelector('#password').value", None)
        .await
        .expect("first value");
    let second_value = second
        .evaluate_js("() => document.querySelector('#password').value", None)
        .await
        .expect("second value");
    assert_eq!(first_value["value"], "marker-password");
    assert_eq!(second_value["value"], "");

    first.close().await.expect("close first");
    second.close().await.expect("close second");
    host.close().await.expect("close host");
}
