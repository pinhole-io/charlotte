//! Mocked HTTP for Charlotte `web_search` via SearXNG JSON API.

use std::time::Duration;

use charlotte::SearchBackendKind;
use charlotte::search::{SearchConfig, web_search};
use serde_json::json;
use superglue::http::{ClientConfig, HttpClient};
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn web_search_searxng_maps_json_results() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/search"))
        .and(query_param("q", "apple social"))
        .and(query_param("format", "json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "query": "apple social",
            "number_of_results": 1,
            "results": [{
                "title": "Apple",
                "url": "https://www.apple.com/",
                "content": "Official Apple site",
                "engine": "duckduckgo"
            }]
        })))
        .mount(&mock)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http client");
    let search_config = SearchConfig::new(SearchBackendKind::Searxng, mock.uri(), Duration::ZERO);

    let out = web_search(&http, None, &search_config, "apple social", 8)
        .await
        .expect("web_search");

    assert_eq!(out.get("backend").and_then(|v| v.as_str()), Some("searxng"));
    let results = out
        .get("results")
        .and_then(|v| v.as_array())
        .expect("results array");
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("title").and_then(|v| v.as_str()),
        Some("Apple")
    );
    assert_eq!(
        results[0].get("url").and_then(|v| v.as_str()),
        Some("https://www.apple.com/")
    );
    assert_eq!(
        results[0].get("snippet").and_then(|v| v.as_str()),
        Some("Official Apple site")
    );
}
