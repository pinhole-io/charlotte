//! Jev-guided crawl against Wiremock + a scripted page actor (no Chrome, no TypeSafe).

use std::sync::Arc;

use charlotte::{
    AgenticCrawlConfig, AgenticLlm, ObservePage, RawTarget, ScriptedPageActor, agentic_crawl,
};
use serde_json::json;
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::providers::{ProviderCredentials, ProviderId};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn jev_answer(id: &str, choice: &str) -> serde_json::Value {
    json!({
        "model": "jev-latest",
        "answers": {
            id: {
                "type": "choice",
                "choice": choice,
                "probabilities": {},
                "confidence": 0.9
            }
        },
        "usage": { "input_tokens": 8, "output_tokens": 2 }
    })
}

async fn mount_jev_once(mock: &MockServer, id: &str, choice: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(jev_answer(id, choice)))
        .up_to_n_times(1)
        .mount(mock)
        .await;
}

async fn mount_jev_sequence(mock: &MockServer, pairs: &[(&str, &str)]) {
    for (id, choice) in pairs {
        mount_jev_once(mock, id, choice).await;
    }
}

fn mock_llm(http: &HttpClient, mock_uri: String) -> AgenticLlm {
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock_uri);
    AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    }
}

async fn mount_jev_rest(mock: &MockServer, id: &str, choice: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(jev_answer(id, choice)))
        .mount(mock)
        .await;
}

fn page(url: &str, title: &str, label: &str, href: &str) -> ObservePage {
    ObservePage {
        title: title.into(),
        url: url.into(),
        markdown: format!("{title} body about rust docs"),
        targets: vec![RawTarget {
            id: "t1".into(),
            kind: "link".into(),
            label: label.into(),
            href: Some(href.into()),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn page_two_links(
    url: &str,
    title: &str,
    first: (&str, &str, &str),
    second: (&str, &str, &str),
) -> ObservePage {
    ObservePage {
        title: title.into(),
        url: url.into(),
        markdown: format!("{title} body"),
        targets: vec![
            RawTarget {
                id: first.0.into(),
                kind: "link".into(),
                label: first.1.into(),
                href: Some(first.2.into()),
                ..Default::default()
            },
            RawTarget {
                id: second.0.into(),
                kind: "link".into(),
                label: second.1.into(),
                href: Some(second.2.into()),
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

#[tokio::test]
async fn agentic_crawl_uses_system_one_then_stops() {
    let mock = MockServer::start().await;
    mount_jev_rest(&mock, "done", "yes").await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut actor = ScriptedPageActor::new(vec![page(
        "https://example.com/",
        "Example",
        "Rust book",
        "https://example.com/rust",
    )]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "read the rust book");
    cfg.use_env_jev = false;
    cfg.max_steps = 3;
    cfg.max_pages = 5;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
    assert_eq!(out["pages_kept"], json!(1));
    assert_eq!(out["via_jev"]["steps"], json!(1));
    assert!(actor.actions.iter().any(|a| a == "start"));
    assert!(actor.actions.iter().any(|a| a == "stop"));
}

#[tokio::test]
async fn agentic_crawl_without_jev_does_not_follow() {
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut actor = ScriptedPageActor::new(vec![page(
        "https://example.com/",
        "Home",
        "Docs",
        "https://example.com/docs",
    )]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "docs coverage");
    cfg.use_env_jev = false;
    cfg.max_steps = 2;
    cfg.max_pages = 2;

    let err = agentic_crawl(&mut actor, &cfg, &http, None)
        .await
        .expect_err("crawl must require Jev credentials");
    assert!(
        err.to_string().contains("TYPESAFE_API_KEY"),
        "expected credentials error, got {err}"
    );
    assert!(
        !actor.actions.iter().any(|a| a.starts_with("follow:")),
        "must not follow without Jev, actions={:?}",
        actor.actions
    );
}

#[tokio::test]
async fn agentic_crawl_jev_follow_then_stop() {
    let mock = MockServer::start().await;
    mount_jev_sequence(
        &mock,
        &[("done", "no"), ("act", "follow"), ("target", "t001")],
    )
    .await;
    mount_jev_rest(&mock, "done", "yes").await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut actor = ScriptedPageActor::new(vec![
        page(
            "https://example.com/",
            "Home",
            "Docs",
            "https://example.com/docs",
        ),
        page(
            "https://example.com/docs",
            "Docs",
            "Home",
            "https://example.com/",
        ),
    ]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "docs coverage");
    cfg.use_env_jev = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
    assert!(
        actor.actions.iter().any(|a| a == "follow:t1"),
        "expected Jev follow, got {:?}",
        actor.actions
    );
    assert_eq!(out["via_jev"]["steps"], json!(2));
}

#[tokio::test]
async fn agentic_crawl_jev_get_links_then_stop() {
    let mock = MockServer::start().await;
    mount_jev_sequence(&mock, &[("done", "no"), ("act", "get_links")]).await;
    mount_jev_rest(&mock, "done", "yes").await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut actor = ScriptedPageActor::new(vec![page(
        "https://example.com/",
        "Home",
        "Docs",
        "https://example.com/docs",
    )]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "docs coverage");
    cfg.use_env_jev = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert!(
        actor.actions.iter().any(|a| a == "get_links"),
        "expected get_links, got {:?}",
        actor.actions
    );
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
}

#[tokio::test]
async fn agentic_crawl_jev_scroll_then_stop() {
    let mock = MockServer::start().await;
    mount_jev_sequence(&mock, &[("done", "no"), ("act", "scroll_down")]).await;
    mount_jev_rest(&mock, "done", "yes").await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut long_page = page(
        "https://example.com/",
        "Home",
        "Docs",
        "https://example.com/docs",
    );
    long_page.scroll_y = 0;
    long_page.viewport_h = 800;
    long_page.scroll_height = 4000;
    long_page.at_top = true;
    long_page.at_bottom = false;
    let mut actor = ScriptedPageActor::new(vec![long_page]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "docs coverage");
    cfg.use_env_jev = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert!(
        actor.actions.iter().any(|a| a == "scroll:600"),
        "expected scroll, got {:?}",
        actor.actions
    );
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
}

#[tokio::test]
async fn agentic_crawl_stops_before_repeating_an_action_from_the_same_page_state() {
    let mock = MockServer::start().await;
    mount_jev_sequence(
        &mock,
        &[
            ("done", "no"),
            ("act", "scroll_down"),
            ("done", "no"),
            ("act", "scroll_down"),
            ("done", "no"),
            ("act", "scroll_down"),
            ("done", "no"),
            ("act", "scroll_down"),
        ],
    )
    .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = mock_llm(&http, mock.uri());
    let mut long_page = page(
        "https://example.com/",
        "Home",
        "Docs",
        "https://example.com/docs",
    );
    long_page.scroll_y = 0;
    long_page.viewport_h = 800;
    long_page.scroll_height = 4000;
    long_page.at_top = true;
    long_page.at_bottom = false;
    let mut actor = ScriptedPageActor::new(vec![long_page]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "find the docs");
    cfg.use_env_jev = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["stopped_reason"], json!("repeated_action"));
    assert_eq!(
        actor.actions.iter().filter(|a| *a == "scroll:600").count(),
        1,
        "the repeated action must not reach the actor: {:?}",
        actor.actions
    );
}

/// On the seed page the trail has no earlier entry, so `back` is not on the menu
/// and an answer of `back` cannot dispatch. The old code looped on it forever.
#[tokio::test]
async fn agentic_crawl_rejects_back_on_the_seed_page() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "jev-latest",
            "answers": {
                "act": {
                    "type": "choice",
                    "choice": "back",
                    "probabilities": {},
                    "confidence": 0.99
                }
            },
            "usage": { "input_tokens": 8, "output_tokens": 2 }
        })))
        .mount(&mock)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut actor = ScriptedPageActor::new(vec![page(
        "https://example.com/",
        "Home",
        "Guide",
        "https://example.com/guide",
    )]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "find the changelog");
    cfg.use_env_jev = false;
    cfg.max_steps = 6;
    cfg.max_pages = 6;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["stopped_reason"], json!("jev_unavailable"));
    assert!(
        !actor
            .actions
            .iter()
            .any(|a| a.starts_with("navigate:https") && a != "navigate:https://example.com/"),
        "back must not navigate off the trail, got {:?}",
        actor.actions
    );
}

/// A dead end is recoverable: Jev follows a link, then walks the trail back.
#[tokio::test]
async fn agentic_crawl_jev_back_returns_to_the_previous_page() {
    let mock = MockServer::start().await;
    mount_jev_sequence(
        &mock,
        &[
            ("done", "no"),
            ("act", "follow"),
            ("target", "t001"),
            ("done", "no"),
            ("act", "back"),
            ("done", "no"),
            ("act", "stop"),
        ],
    )
    .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut actor = ScriptedPageActor::new(vec![
        page(
            "https://example.com/",
            "Home",
            "Guide",
            "https://example.com/guide",
        ),
        page(
            "https://example.com/guide",
            "Guide",
            "Home",
            "https://example.com/",
        ),
    ]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "find the changelog");
    cfg.use_env_jev = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
    assert!(
        actor
            .actions
            .iter()
            .rev()
            .any(|a| a == "navigate:https://example.com/"),
        "expected back to navigate to the previous trail page, got {:?}",
        actor.actions
    );
    let acts: Vec<&str> = out["trace"]
        .as_array()
        .expect("trace")
        .iter()
        .filter_map(|row| row["act"].as_str())
        .collect();
    assert_eq!(acts, vec!["follow", "back", "stop"], "trace: {acts:?}");
}

/// After follow + back, the opened href is gone from this page's menu.
/// A second follow must take the remaining link.
#[tokio::test]
async fn agentic_crawl_second_follow_from_a_page_cannot_reuse_the_first_href() {
    let mock = MockServer::start().await;
    mount_jev_sequence(
        &mock,
        &[
            ("done", "no"),
            ("act", "follow"),
            ("target", "t001"),
            ("done", "no"),
            ("act", "back"),
            ("done", "no"),
            ("act", "follow"),
            ("target", "t001"),
            ("done", "no"),
            ("act", "stop"),
        ],
    )
    .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let llm = AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    };

    let mut actor = ScriptedPageActor::new(vec![
        page_two_links(
            "https://example.com/",
            "Home",
            ("t1", "tools", "https://example.com/tools"),
            ("t2", "computers", "https://example.com/computers"),
        ),
        page(
            "https://example.com/tools",
            "Tools",
            "Home",
            "https://example.com/",
        ),
        page(
            "https://example.com/computers",
            "Computers",
            "Home",
            "https://example.com/",
        ),
    ]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "find a laptop listing");
    cfg.use_env_jev = false;
    cfg.max_steps = 6;
    cfg.max_pages = 6;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    let follows: Vec<&String> = actor
        .actions
        .iter()
        .filter(|a| a.starts_with("follow:"))
        .collect();
    assert_eq!(
        follows,
        vec![&"follow:t1".to_string(), &"follow:t2".to_string()],
        "second follow must take the remaining link, got {:?}",
        actor.actions
    );
    let acts: Vec<&str> = out["trace"]
        .as_array()
        .expect("trace")
        .iter()
        .filter_map(|row| row["act"].as_str())
        .collect();
    assert_eq!(
        acts,
        vec!["follow", "back", "follow", "stop"],
        "trace: {acts:?}"
    );
}

#[tokio::test]
async fn agentic_crawl_jev_done_yes_keeps_listing_and_stops() {
    let mock = MockServer::start().await;
    mount_jev_sequence(
        &mock,
        &[
            ("done", "no"),
            ("act", "follow"),
            ("target", "t001"),
            ("done", "yes"),
        ],
    )
    .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = mock_llm(&http, mock.uri());
    let mut actor = ScriptedPageActor::new(vec![
        page(
            "https://www.craigslist.org/search/area/fresno?cat=sya",
            "fresno computers for sale",
            "Dell laptop",
            "https://www.craigslist.org/view/d/fresno/dell-laptop",
        ),
        ObservePage {
            title: "Dell E7440 laptop 4th Gen i5 - computers".into(),
            url: "https://www.craigslist.org/view/d/fresno/dell-laptop".into(),
            markdown: "Dell E7440 laptop listed today.".into(),
            targets: vec![RawTarget {
                id: "t9".into(),
                kind: "link".into(),
                label: "next ad".into(),
                href: Some("https://www.craigslist.org/view/d/fresno/other".into()),
                ..Default::default()
            }],
            ..Default::default()
        },
    ]);
    let mut cfg = AgenticCrawlConfig::new(
        "https://www.craigslist.org/search/area/fresno?cat=sya",
        "find a current laptop listing",
    );
    cfg.use_env_jev = false;
    cfg.max_steps = 6;
    cfg.max_pages = 6;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
    assert_eq!(out["pages_kept"], json!(2));
    let acts: Vec<&str> = out["trace"]
        .as_array()
        .expect("trace")
        .iter()
        .filter_map(|row| row["act"].as_str())
        .collect();
    assert_eq!(acts, vec!["follow", "stop"], "trace: {acts:?}");
    assert_eq!(
        actor
            .actions
            .iter()
            .filter(|a| a.starts_with("follow:"))
            .count(),
        1,
        "must not open another listing after done=yes, actions={:?}",
        actor.actions
    );
}

#[tokio::test]
async fn agentic_crawl_jev_keeps_going_until_enough_results() {
    let mock = MockServer::start().await;
    mount_jev_sequence(
        &mock,
        &[
            ("done", "no"),
            ("act", "follow"),
            ("target", "t001"),
            ("done", "no"),
            ("act", "back"),
            ("done", "no"),
            ("act", "follow"),
            ("target", "t001"),
            ("done", "yes"),
        ],
    )
    .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = mock_llm(&http, mock.uri());
    let mut actor = ScriptedPageActor::new(vec![
        page_two_links(
            "https://www.craigslist.org/search/area/fresno?cat=sya",
            "fresno computers for sale",
            (
                "t1",
                "Dell laptop",
                "https://www.craigslist.org/view/d/fresno/dell",
            ),
            (
                "t2",
                "HP laptop",
                "https://www.craigslist.org/view/d/fresno/hp",
            ),
        ),
        ObservePage {
            title: "Dell laptop".into(),
            url: "https://www.craigslist.org/view/d/fresno/dell".into(),
            markdown: "Dell listing".into(),
            targets: vec![],
            ..Default::default()
        },
        ObservePage {
            title: "HP laptop".into(),
            url: "https://www.craigslist.org/view/d/fresno/hp".into(),
            markdown: "HP listing".into(),
            targets: vec![],
            ..Default::default()
        },
    ]);
    let mut cfg = AgenticCrawlConfig::new(
        "https://www.craigslist.org/search/area/fresno?cat=sya",
        "find 2 current laptop listings",
    );
    cfg.use_env_jev = false;
    cfg.max_steps = 8;
    cfg.max_pages = 8;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("jev_stop"));
    let titles: Vec<&str> = out["pages"]
        .as_array()
        .expect("pages")
        .iter()
        .filter_map(|row| row["title"].as_str())
        .collect();
    assert!(
        titles.contains(&"Dell laptop") && titles.contains(&"HP laptop"),
        "both listings must be kept, pages={titles:?}"
    );
    let acts: Vec<&str> = out["trace"]
        .as_array()
        .expect("trace")
        .iter()
        .filter_map(|row| row["act"].as_str())
        .collect();
    assert_eq!(
        acts,
        vec!["follow", "back", "follow", "stop"],
        "trace: {acts:?}"
    );
}

#[tokio::test]
async fn agentic_crawl_stops_when_goal_page_reached() {
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut actor = ScriptedPageActor::new(vec![ObservePage {
        title: "Dog food - Wikipedia".into(),
        url: "https://en.wikipedia.org/wiki/Dog_food".into(),
        markdown: "Dog food is food for dogs.".into(),
        targets: vec![RawTarget {
            id: "t1".into(),
            kind: "link".into(),
            label: "Cat food".into(),
            href: Some("https://en.wikipedia.org/wiki/Cat_food".into()),
            ..Default::default()
        }],
        ..Default::default()
    }]);
    let mut cfg = AgenticCrawlConfig::new(
        "https://en.wikipedia.org/wiki/Dog_food",
        "reach Dog food from Sun",
    );
    cfg.use_env_jev = false;
    cfg.max_steps = 8;
    cfg.max_pages = 8;

    let out = agentic_crawl(&mut actor, &cfg, &http, None)
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("goal_reached"));
    assert_eq!(out["pages_kept"], json!(1));
    assert!(
        !actor.actions.iter().any(|a| a.starts_with("follow:")),
        "must not keep walking after the goal page, actions={:?}",
        actor.actions
    );
}

#[tokio::test]
#[ignore = "needs Obscura CDP, uv, and browser-use"]
async fn agentic_crawl_live_example_com() {
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "read the example page heading");
    cfg.use_env_jev = true;
    cfg.max_steps = 2;
    cfg.max_pages = 1;
    cfg.same_origin_only = true;
    let out = charlotte::agentic_crawl_with_sidecar(&cfg, &http, None)
        .await
        .expect("live sidecar crawl");
    assert_eq!(out["ok"], json!(true));
    assert!(out["pages_kept"].as_u64().unwrap_or(0) >= 1);
}
