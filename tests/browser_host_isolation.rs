use charlotte::{BrowserBackendKind, BrowserConfig, BrowserHost, BrowserHostConfig, UrlPolicy};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
#[ignore = "requires the Playwright Chromium installation"]
async fn hosted_sessions_isolate_cookies_pages_and_tabs() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html><body>ok</body></html>"))
        .mount(&server)
        .await;

    let mut browser = BrowserConfig::default();
    browser.url_policy = UrlPolicy::permissive();
    let host = BrowserHost::start(BrowserHostConfig {
        backend: BrowserBackendKind::Playwright,
        browser,
        cdp_endpoint: None,
    })
    .await
    .expect("browser host");
    let first = host.open_session().await.expect("first session");
    let second = host.open_session().await.expect("second session");

    first
        .navigate(&server.uri())
        .await
        .expect("first navigation");
    second
        .navigate(&server.uri())
        .await
        .expect("second navigation");
    first
        .evaluate_js(
            "() => { document.cookie = 'charlotte_session=first'; window.open(location.href); }",
            None,
        )
        .await
        .expect("set cookie and open tab");

    let first_cookie = first
        .evaluate_js("() => document.cookie", None)
        .await
        .expect("first cookie");
    let second_cookie = second
        .evaluate_js("() => document.cookie", None)
        .await
        .expect("second cookie");
    assert!(
        first_cookie
            .pointer("/value")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|cookie| cookie.contains("charlotte_session=first"))
    );
    assert_eq!(
        second_cookie
            .pointer("/value")
            .and_then(serde_json::Value::as_str),
        Some("")
    );

    let first_tabs = first.list_tabs().await.expect("first tabs");
    let second_tabs = second.list_tabs().await.expect("second tabs");
    assert!(
        first_tabs
            .pointer("/tabs")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tabs| {
                tabs.len()
                    > second_tabs
                        .pointer("/tabs")
                        .and_then(serde_json::Value::as_array)
                        .map_or(0, Vec::len)
            })
    );

    first.close().await.expect("close first session");
    second.close().await.expect("close second session");
    host.close().await.expect("close browser host");
}
