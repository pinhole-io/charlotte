//! HTTP crawl service.
//!
//! `GET /health` and `POST /crawl` sit on [`crate::CrawlRuntime`]. The default
//! bind is `127.0.0.1:11236`. A non-loopback bind requires `CHARLOTTE_CRAWL_TOKEN`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::{Json as JsonBody, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, extract::rejection::JsonRejection};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::browser::BrowserConfig;
use crate::browser::url_policy::UrlPolicy;
use crate::config::apply_browser_config_from_env;
use crate::crawl::capacity::{
    CrawlCapacityConfig, CrawlExecutor, CrawlRuntime, effective_deadline, is_url_rejection,
};
use crate::error::CharlotteError;

const DEFAULT_BIND: &str = "127.0.0.1:11236";
const MAX_URL_BYTES: usize = 8 * 1024;

struct AppState {
    crawl: Arc<dyn CrawlExecutor>,
    token: Option<SecretString>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CrawlRequest {
    urls: Vec<String>,
    #[serde(default)]
    client_timeout_s: Option<f64>,
}

/// Build the crawl router. `token` is required on every route when it is set.
pub(crate) fn router(crawl: Arc<dyn CrawlExecutor>, token: Option<SecretString>) -> Router {
    let state = Arc::new(AppState { crawl, token });
    Router::new()
        .route("/health", get(health))
        .route("/crawl", post(crawl_route))
        .with_state(state)
}

/// Start the crawl service and run until shutdown.
pub async fn run() -> Result<(), CharlotteError> {
    let addr = bind_from_env()?;
    let token = token_from_env();
    ensure_public_bind_has_token(addr, &token)?;
    let capacity = CrawlCapacityConfig::server().apply_env()?;
    let mut browser = apply_browser_config_from_env(BrowserConfig::default());
    browser.headless = true;
    browser.url_policy.block_private_ips = true;
    let runtime = Arc::new(CrawlRuntime::start(capacity, browser).await?);
    let crawl: Arc<dyn CrawlExecutor> = runtime.clone();
    let app = router(crawl, token);
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|error| {
        CharlotteError::msg(format!("crawl service failed to bind {addr}: {error}"))
    })?;
    tracing::info!(%addr, "crawl service listening");
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await;
    let close_result = runtime.close().await;
    if let Err(error) = result {
        return Err(CharlotteError::msg(format!(
            "crawl service stopped: {error}"
        )));
    }
    close_result
}

pub(crate) fn bind_from_env() -> Result<SocketAddr, CharlotteError> {
    let raw = std::env::var("CHARLOTTE_CRAWL_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_string());
    raw.trim()
        .parse()
        .map_err(|_| CharlotteError::msg("CHARLOTTE_CRAWL_BIND is not a socket address"))
}

pub(crate) fn token_from_env() -> Option<SecretString> {
    let Ok(raw) = std::env::var("CHARLOTTE_CRAWL_TOKEN") else {
        return None;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(SecretString::from(trimmed.to_string()))
    }
}

pub(crate) fn ensure_public_bind_has_token(
    addr: SocketAddr,
    token: &Option<SecretString>,
) -> Result<(), CharlotteError> {
    if addr.ip().is_loopback() || token.is_some() {
        Ok(())
    } else {
        Err(CharlotteError::msg(
            "CHARLOTTE_CRAWL_TOKEN is required when the crawl service is not bound to a loopback address",
        ))
    }
}

async fn health(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(response) = reject_unauthorized(&state, &headers) {
        return response;
    }
    if state.crawl.ready_browser_count() == 0 {
        return json_status(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"detail": "browser is not connected"}),
        );
    }
    json_status(
        StatusCode::OK,
        json!({
            "status": "ok",
            "ready_browsers": state.crawl.ready_browser_count(),
            "browser_pool_size": state.crawl.browser_pool_size(),
        }),
    )
}

async fn crawl_route(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<JsonBody<CrawlRequest>, JsonRejection>,
) -> Response {
    if let Some(response) = reject_unauthorized(&state, &headers) {
        return response;
    }
    let request = match body {
        Ok(JsonBody(request)) => request,
        Err(_) => return bad_request("request body must be one JSON object"),
    };
    let Some(url) = single_http_url(&request.urls) else {
        return bad_request("expected one http or https URL");
    };
    if let Err(error) = public_url(&url) {
        return bad_request(error);
    }
    if let Some(secs) = request.client_timeout_s
        && (!secs.is_finite() || secs <= 0.0)
    {
        return bad_request("client_timeout_s must be greater than zero");
    }
    let deadline = effective_deadline(
        state.crawl.server_deadline(),
        request.client_timeout_s,
        state.crawl.cancel_margin(),
    );
    let started = Instant::now();
    tracing::info!(
        url = %url,
        deadline_s = deadline.as_secs_f64(),
        "crawl request started"
    );
    match state.crawl.scrape(&url, deadline).await {
        Ok(page) => {
            let elapsed_s = started.elapsed().as_secs_f64();
            tracing::info!(
                url = %page.url,
                elapsed_s,
                success = page.success,
                status_code = page.status_code,
                "crawl request completed"
            );
            json_status(
                StatusCode::OK,
                json!({
                    "results": [{
                        "url": page.url,
                        "markdown": page.markdown,
                        "success": page.success,
                        "status_code": page.status_code,
                        "error_message": page.error_message,
                    }],
                    "server_processing_time_s": elapsed_s,
                }),
            )
        }
        Err(error) => crawl_error_response(&url, started.elapsed(), error),
    }
}

fn crawl_error_response(url: &str, elapsed: Duration, error: CharlotteError) -> Response {
    tracing::warn!(
        url,
        elapsed_s = elapsed.as_secs_f64(),
        %error,
        "crawl request failed"
    );
    match &error {
        CharlotteError::Overloaded {
            reason,
            retry_after,
        } => overload_response(reason.as_str(), *retry_after),
        CharlotteError::DeadlineExceeded { deadline } => json_status(
            StatusCode::GATEWAY_TIMEOUT,
            json!({
                "detail": "crawl exceeded the server deadline",
                "deadline_s": deadline.as_secs_f64(),
            }),
        ),
        _ if is_url_rejection(&error) => bad_request("URL is not allowed"),
        _ => json_status(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": "crawl failed"}),
        ),
    }
}

fn overload_response(reason: &str, retry_after: Duration) -> Response {
    let mut response = json_status(
        StatusCode::TOO_MANY_REQUESTS,
        json!({
            "detail": "crawl overloaded",
            "reason": reason,
        }),
    );
    let secs = retry_after.as_secs().max(1).to_string();
    if let Ok(value) = HeaderValue::from_str(&secs) {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}

fn reject_unauthorized(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    let expected = state.token.as_ref()?;
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    if bearer_matches(expected, presented) {
        None
    } else {
        Some(json_status(
            StatusCode::UNAUTHORIZED,
            json!({"detail": "unauthorized"}),
        ))
    }
}

fn bearer_matches(expected: &SecretString, header: Option<&str>) -> bool {
    let Some(header) = header else {
        return false;
    };
    let Some(presented) = header.strip_prefix("Bearer ") else {
        return false;
    };
    let expected = expected.expose_secret().as_bytes();
    let presented = presented.as_bytes();
    if expected.len() != presented.len() {
        return false;
    }
    let mut diff = 0u8;
    for (left, right) in expected.iter().zip(presented.iter()) {
        diff |= left ^ right;
    }
    diff == 0
}

fn single_http_url(urls: &[String]) -> Option<String> {
    let [url] = urls else {
        return None;
    };
    let url = url.trim();
    if url.is_empty() || url.len() > MAX_URL_BYTES {
        return None;
    }
    Some(url.to_string())
}

fn public_url(url: &str) -> Result<(), &'static str> {
    let policy = UrlPolicy {
        block_private_ips: true,
        ..UrlPolicy::default()
    };
    match policy.check(url) {
        Ok(()) => Ok(()),
        Err(error) => match error.to_string().as_str() {
            "invalid URL" | "URL scheme is not allowed" => Err("expected one http or https URL"),
            _ => Err("URL is not allowed"),
        },
    }
}

fn bad_request(detail: &str) -> Response {
    json_status(StatusCode::BAD_REQUEST, json!({"detail": detail}))
}

fn json_status(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::crawl::capacity::CrawlPage;
    use crate::error::OverloadReason;

    #[derive(Clone, Copy)]
    enum Script {
        Page,
        Overloaded,
        Deadline,
    }

    struct FakeCrawl {
        ready: usize,
        script: Script,
    }

    #[async_trait]
    impl CrawlExecutor for FakeCrawl {
        fn ready_browser_count(&self) -> usize {
            self.ready
        }

        fn browser_pool_size(&self) -> usize {
            2
        }

        fn server_deadline(&self) -> Duration {
            Duration::from_secs(80)
        }

        fn cancel_margin(&self) -> Duration {
            Duration::from_secs(5)
        }

        async fn scrape(
            &self,
            url: &str,
            _deadline: Duration,
        ) -> Result<CrawlPage, CharlotteError> {
            match self.script {
                Script::Page => Ok(CrawlPage {
                    url: url.to_string(),
                    markdown: Some("# Example".to_string()),
                    success: true,
                    status_code: Some(200),
                    error_message: None,
                }),
                Script::Overloaded => Err(CharlotteError::Overloaded {
                    reason: OverloadReason::Admission,
                    retry_after: Duration::from_secs(15),
                }),
                Script::Deadline => Err(CharlotteError::DeadlineExceeded {
                    deadline: Duration::from_secs(80),
                }),
            }
        }
    }

    fn app(script: Script, ready: usize) -> Router {
        router(Arc::new(FakeCrawl { ready, script }), None)
    }

    async fn send(app: Router, request: Request<Body>) -> (StatusCode, Value, HeaderMap) {
        let response = app.oneshot(request).await.expect("router response");
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
        (status, json, headers)
    }

    fn post_json(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/crawl")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request")
    }

    #[test]
    fn public_bind_requires_a_token() {
        let public_addr: SocketAddr = "0.0.0.0:11236".parse().expect("address");
        assert!(ensure_public_bind_has_token(public_addr, &None).is_err());
        let loopback: SocketAddr = "127.0.0.1:11236".parse().expect("address");
        assert!(ensure_public_bind_has_token(loopback, &None).is_ok());
    }

    #[tokio::test]
    async fn health_reports_readiness() {
        let (down, _, _) = send(
            app(Script::Page, 0),
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(down, StatusCode::SERVICE_UNAVAILABLE);

        let (up, body, _) = send(
            app(Script::Page, 1),
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(up, StatusCode::OK);
        assert_eq!(body["ready_browsers"], 1);
    }

    #[tokio::test]
    async fn crawl_maps_success_overload_deadline_and_bad_requests() {
        let (ok, body, _) = send(
            app(Script::Page, 1),
            post_json(r#"{"urls":["https://example.com"]}"#),
        )
        .await;
        assert_eq!(ok, StatusCode::OK);
        assert_eq!(body["results"][0]["markdown"], "# Example");
        assert_eq!(body["results"][0]["success"], true);

        let (overloaded, body, headers) = send(
            app(Script::Overloaded, 1),
            post_json(r#"{"urls":["https://example.com"]}"#),
        )
        .await;
        assert_eq!(overloaded, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body["reason"], "admission");
        assert_eq!(headers[header::RETRY_AFTER], "15");

        let (deadline, _, _) = send(
            app(Script::Deadline, 1),
            post_json(r#"{"urls":["https://example.com"]}"#),
        )
        .await;
        assert_eq!(deadline, StatusCode::GATEWAY_TIMEOUT);

        let (two_urls, _, _) = send(
            app(Script::Page, 1),
            post_json(r#"{"urls":["https://example.com","https://example.com/b"]}"#),
        )
        .await;
        assert_eq!(two_urls, StatusCode::BAD_REQUEST);

        let (ftp, _, _) = send(
            app(Script::Page, 1),
            post_json(r#"{"urls":["ftp://example.com/file"]}"#),
        )
        .await;
        assert_eq!(ftp, StatusCode::BAD_REQUEST);

        let (private_ip, _, _) = send(
            app(Script::Page, 1),
            post_json(r#"{"urls":["http://127.0.0.1/"]}"#),
        )
        .await;
        assert_eq!(private_ip, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn token_is_required_when_configured() {
        let app = router(
            Arc::new(FakeCrawl {
                ready: 1,
                script: Script::Page,
            }),
            Some(SecretString::from("crawl-token".to_string())),
        );
        let (missing, _, _) = send(
            app.clone(),
            post_json(r#"{"urls":["https://example.com"]}"#),
        )
        .await;
        assert_eq!(missing, StatusCode::UNAUTHORIZED);

        let authed = Request::builder()
            .method("POST")
            .uri("/crawl")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::AUTHORIZATION, "Bearer crawl-token")
            .body(Body::from(
                r#"{"urls":["https://example.com"]}"#.to_string(),
            ))
            .expect("request");
        let (ok, _, _) = send(app, authed).await;
        assert_eq!(ok, StatusCode::OK);
    }
}
