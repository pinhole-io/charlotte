//! System One–guided crawl against Wiremock + a scripted page actor (no Chrome, no TypeSafe).

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

fn system_one_answer(id: &str, choice: &str) -> serde_json::Value {
    system_one_answers(&[(id, choice)])
}

fn system_one_answers(pairs: &[(&str, &str)]) -> serde_json::Value {
    let mut answers = serde_json::Map::new();
    for (id, choice) in pairs {
        answers.insert(
            (*id).to_string(),
            json!({
                "type": "choice",
                "choice": choice,
                "probabilities": {},
                "confidence": 0.9
            }),
        );
    }
    json!({
        "model": "jev-latest",
        "answers": answers,
        "usage": { "input_tokens": 8, "output_tokens": 2 }
    })
}

fn chat_completion(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    })
}

fn llm_on_mock(http: &HttpClient, mock_uri: &str) -> AgenticLlm {
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock_uri);
    creds.insert_key(ProviderId::OpenAi, "sk-test");
    creds.insert_base_url(ProviderId::OpenAi, mock_uri);
    AgenticLlm {
        http: http.clone(),
        hooks: Arc::new(HookRegistry::new()),
        guardrails: Arc::new(GuardrailRegistry::new()),
        options: ChatOptions {
            api_key: secrecy::SecretString::from("sk-test"),
            model: "mock".into(),
            base_url: mock_uri.to_string(),
            provider_credentials: Some(Arc::new(creds)),
            ..Default::default()
        },
    }
}

async fn mount_system_one_once(mock: &MockServer, id: &str, choice: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(system_one_answer(id, choice)))
        .up_to_n_times(1)
        .mount(mock)
        .await;
}

async fn mount_system_one_sequence(mock: &MockServer, pairs: &[(&str, &str)]) {
    for (id, choice) in pairs {
        mount_system_one_once(mock, id, choice).await;
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

async fn mount_system_one_rest(mock: &MockServer, id: &str, choice: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(system_one_answer(id, choice)))
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
    mount_system_one_rest(&mock, "done", "yes").await;

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
    cfg.use_env_system_one = false;
    cfg.max_steps = 3;
    cfg.max_pages = 5;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
    assert_eq!(out["pages_kept"], json!(1));
    assert_eq!(out["via_system_one"]["steps"], json!(1));
    assert!(actor.actions.iter().any(|a| a == "start"));
    assert!(actor.actions.iter().any(|a| a == "stop"));
}

#[tokio::test]
async fn agentic_crawl_without_system_one_does_not_follow() {
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut actor = ScriptedPageActor::new(vec![page(
        "https://example.com/",
        "Home",
        "Docs",
        "https://example.com/docs",
    )]);
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "docs coverage");
    cfg.use_env_system_one = false;
    cfg.max_steps = 2;
    cfg.max_pages = 2;

    let err = agentic_crawl(&mut actor, &cfg, &http, None)
        .await
        .expect_err("crawl must require System One credentials");
    assert!(
        err.to_string().contains("TYPESAFE_API_KEY"),
        "expected credentials error, got {err}"
    );
    assert!(
        !actor.actions.iter().any(|a| a.starts_with("follow:")),
        "must not follow without System One, actions={:?}",
        actor.actions
    );
}

#[tokio::test]
async fn agentic_crawl_system_one_follow_then_stop() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
        &mock,
        &[("done", "no"), ("act", "follow"), ("target", "t001")],
    )
    .await;
    mount_system_one_rest(&mock, "done", "yes").await;

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
    cfg.use_env_system_one = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
    assert!(
        actor.actions.iter().any(|a| a == "follow:t1"),
        "expected System One follow, got {:?}",
        actor.actions
    );
    assert_eq!(out["via_system_one"]["steps"], json!(2));
}

#[tokio::test]
async fn agentic_crawl_system_one_get_links_then_stop() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(&mock, &[("done", "no"), ("act", "get_links")]).await;
    mount_system_one_rest(&mock, "done", "yes").await;

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
    cfg.use_env_system_one = false;
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
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
}

#[tokio::test]
async fn agentic_crawl_system_one_scroll_then_stop() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(&mock, &[("done", "no"), ("act", "scroll_down")]).await;
    mount_system_one_rest(&mock, "done", "yes").await;

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
    cfg.use_env_system_one = false;
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
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
}

#[tokio::test]
async fn agentic_crawl_stops_before_repeating_an_action_from_the_same_page_state() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
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
    cfg.use_env_system_one = false;
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
    cfg.use_env_system_one = false;
    cfg.max_steps = 6;
    cfg.max_pages = 6;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["stopped_reason"], json!("system_one_unavailable"));
    assert!(
        !actor
            .actions
            .iter()
            .any(|a| a.starts_with("navigate:https") && a != "navigate:https://example.com/"),
        "back must not navigate off the trail, got {:?}",
        actor.actions
    );
}

/// A dead end is recoverable: System One follows a link, then walks the trail back.
#[tokio::test]
async fn agentic_crawl_system_one_back_returns_to_the_previous_page() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
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
    cfg.use_env_system_one = false;
    cfg.max_steps = 4;
    cfg.max_pages = 4;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
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
    mount_system_one_sequence(
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
    cfg.use_env_system_one = false;
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

fn credential_page(url: &str, reflected: Option<&str>, kind: &str) -> ObservePage {
    let reflected = reflected.unwrap_or("Sign in");
    ObservePage {
        title: format!("{reflected} title"),
        url: url.to_string(),
        markdown: format!("{reflected} body"),
        targets: vec![RawTarget {
            id: "t1".to_string(),
            kind: kind.to_string(),
            label: format!("{reflected} field"),
            section: format!("{reflected} section"),
            in_viewport: true,
            ..RawTarget::default()
        }],
        ..ObservePage::default()
    }
}

#[tokio::test]
async fn credential_crawl_stops_after_javascript_navigation_leaves_origin() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
        &mock,
        &[("done", "no"), ("act", "click"), ("target", "t001")],
    )
    .await;
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = mock_llm(&http, mock.uri());
    let mut actor = ScriptedPageActor::new(vec![
        credential_page("https://example.com/login", None, "button"),
        credential_page("https://wrong.example/collect", None, "input"),
    ]);
    let mut config = AgenticCrawlConfig::new("https://example.com/login", "sign in");
    config.use_env_system_one = false;
    config.max_steps = 3;
    config.inputs = vec![charlotte::CrawlInput::new("password", "marker-password")];
    config.credential_origins = Some(vec!["https://example.com".to_string()]);

    let output = agentic_crawl(&mut actor, &config, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(output["stopped_reason"], "credential_origin_left");
    assert!(!output.to_string().contains("wrong.example"));
    assert!(
        actor
            .actions
            .iter()
            .all(|action| !action.starts_with("type_secret:"))
    );
}

#[tokio::test]
async fn credential_crawl_redacts_reflected_values_before_system_one_and_result() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
        &mock,
        &[("done", "no"), ("act", "type"), ("target", "t001")],
    )
    .await;
    mount_system_one_rest(&mock, "done", "yes").await;
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = mock_llm(&http, mock.uri());
    let mut actor = ScriptedPageActor::new(vec![
        credential_page("https://example.com/login", None, "input"),
        credential_page(
            "https://example.com/account",
            Some("marker-password"),
            "input",
        ),
    ]);
    let mut config = AgenticCrawlConfig::new("https://example.com/login", "sign in");
    config.use_env_system_one = false;
    config.max_steps = 3;
    config.inputs = vec![charlotte::CrawlInput::new("password", "marker-password")];
    config.credential_origins = Some(vec!["https://example.com".to_string()]);

    let output = agentic_crawl(&mut actor, &config, &http, Some(&llm))
        .await
        .expect("crawl");
    assert!(!output.to_string().contains("marker-password"));
    assert!(
        actor
            .actions
            .iter()
            .any(|action| action == "type_secret:t1")
    );
    let requests = mock.received_requests().await.expect("requests");
    assert!(
        requests
            .iter()
            .all(|request| !String::from_utf8_lossy(&request.body).contains("marker-password"))
    );
}

#[tokio::test]
async fn agentic_crawl_system_one_done_yes_keeps_listing_and_stops() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
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
    cfg.use_env_system_one = false;
    cfg.max_steps = 6;
    cfg.max_pages = 6;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
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
async fn agentic_crawl_system_one_keeps_going_until_enough_results() {
    let mock = MockServer::start().await;
    mount_system_one_sequence(
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
    cfg.use_env_system_one = false;
    cfg.max_steps = 8;
    cfg.max_pages = 8;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
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
    cfg.use_env_system_one = false;
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
async fn agentic_crawl_extracts_when_system_one_asks() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(system_one_answers(&[("done", "yes"), ("extract", "yes")])),
        )
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completion("Pit bulls are a dog type, not one breed.")),
        )
        .mount(&mock)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = llm_on_mock(&http, &mock.uri());
    let mut actor = ScriptedPageActor::new(vec![page(
        "https://en.wikipedia.org/wiki/Pit_bull",
        "Pit bull",
        "History",
        "https://en.wikipedia.org/wiki/Bull_and_terrier",
    )]);
    let mut cfg = AgenticCrawlConfig::new(
        "https://en.wikipedia.org/wiki/Pit_bull",
        "why people fear pit bulls",
    );
    cfg.use_env_system_one = false;
    cfg.max_steps = 2;
    cfg.max_pages = 2;

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["stopped_reason"], json!("system_one_stop"));
    assert_eq!(out["via_system_one"]["extracts"], json!(1));
    assert_eq!(
        out["pages"][0]["extract"]["summary"],
        json!("Pit bulls are a dog type, not one breed.")
    );
    let excerpt = out["pages"][0]["excerpt"].as_str().expect("excerpt");
    assert!(
        excerpt.contains("Pit bull body"),
        "excerpt should be page prose, got {excerpt}"
    );
}

#[tokio::test]
async fn agentic_crawl_uses_extract_schema_without_an_extract_answer() {
    let mock = MockServer::start().await;
    mount_system_one_rest(&mock, "done", "yes").await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completion(r#"{"position":"opposes breed bans"}"#)),
        )
        .mount(&mock)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let llm = llm_on_mock(&http, &mock.uri());
    let mut actor = ScriptedPageActor::new(vec![page(
        "https://www.aspca.org/position-statement-pit-bulls",
        "ASPCA position",
        "policy",
        "https://www.aspca.org/next",
    )]);
    let mut cfg = AgenticCrawlConfig::new(
        "https://www.aspca.org/position-statement-pit-bulls",
        "ASPCA position on pit bulls",
    );
    cfg.use_env_system_one = false;
    cfg.max_steps = 2;
    cfg.max_pages = 2;
    cfg.extract_schema = Some(json!({
        "type": "object",
        "properties": { "position": { "type": "string" } }
    }));

    let out = agentic_crawl(&mut actor, &cfg, &http, Some(&llm))
        .await
        .expect("crawl");
    assert_eq!(out["via_system_one"]["extracts"], json!(1));
    assert_eq!(
        out["pages"][0]["extract"]["data"]["position"],
        json!("opposes breed bans")
    );
}

#[tokio::test]
#[ignore = "needs Obscura CDP, uv, and browser-use"]
async fn agentic_crawl_live_example_com() {
    let http = HttpClient::new(ClientConfig::default()).expect("http");
    let mut cfg = AgenticCrawlConfig::new("https://example.com/", "read the example page heading");
    cfg.use_env_system_one = true;
    cfg.max_steps = 2;
    cfg.max_pages = 1;
    cfg.same_origin_only = true;
    let out = charlotte::agentic_crawl_with_sidecar(&cfg, &http, None)
        .await
        .expect("live sidecar crawl");
    assert_eq!(out["ok"], json!(true));
    assert!(out["pages_kept"].as_u64().unwrap_or(0) >= 1);
}
