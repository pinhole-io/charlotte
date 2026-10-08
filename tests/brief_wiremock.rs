//! Mocked HTTP for [`charlotte::expand_user_request_to_research_brief`] (no real API key).

use charlotte::SearchConfig;
use charlotte::expand_user_request_to_research_brief;
use serde_json::json;
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn research_brief_hits_chat_completions() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "## Goal\nAnswer from tools only.\n\n## Success criteria\nCitations present.\n"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        })))
        .mount(&mock)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http client");
    let base = ChatOptions {
        base_url: mock.uri(),
        api_key: secrecy::SecretString::from("sk-test"),
        model: "mock".into(),
        max_tool_rounds: 4,
        temperature: Some(0.2),
        ..Default::default()
    };

    let brief = expand_user_request_to_research_brief(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        &base,
        &SearchConfig::default(),
        "Summarize the Rust book.",
    )
    .await
    .expect("brief expansion");

    assert!(
        brief.contains("Goal") || brief.contains("Success"),
        "unexpected brief: {brief:?}"
    );
}

#[tokio::test]
async fn brief_gate_uses_system_one_when_typesafe_present() {
    use charlotte::judge_brief_gate;
    use std::sync::Arc;
    use superglue::providers::{ProviderCredentials, ProviderId};

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "jev-latest",
            "answers": {
                "gate": {
                    "type": "choice",
                    "choice": "direct",
                    "probabilities": {},
                    "confidence": 0.9
                },
                "shape": {
                    "type": "choice",
                    "choice": "open-web",
                    "probabilities": {},
                    "confidence": 0.8
                }
            },
            "usage": { "input_tokens": 8, "output_tokens": 2 }
        })))
        .mount(&mock)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http client");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::TypeSafe, "ts-test");
    creds.insert_base_url(ProviderId::TypeSafe, mock.uri());
    let base = ChatOptions {
        api_key: secrecy::SecretString::from("sk-test"),
        model: "mock".into(),
        provider_credentials: Some(Arc::new(creds)),
        ..Default::default()
    };

    let decision = judge_brief_gate(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        &base,
        "what is the latest rust version",
    )
    .await
    .expect("gate");
    assert!(!decision.expand);
    assert!(decision.via_system_one);
    assert_eq!(decision.shape.map(|s| s.as_str()), Some("open-web"));
}
