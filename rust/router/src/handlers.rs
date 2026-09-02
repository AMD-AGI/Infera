///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! axum HTTP surface + shared app state.

use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Value};

use crate::anthropic::{self, SseTranslator};
use crate::breaker::CircuitBreaker;
use crate::cache_control::{attach_cache_hints, parse_cache_hints};
use crate::policy::Policy;
use crate::pool::SharedPool;
use crate::proxy;

const REQUEST_ID_HEADER: &str = "x-infera-request-id";
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub pool: SharedPool,
    pub policy: Arc<dyn Policy>,
    pub http: reqwest::Client,
    pub started: Instant,
    pub retries: usize,
    /// Per-worker failure memory. Shared across threads and across requests —
    /// that persistence across requests is the whole point (see breaker.rs).
    pub breaker: Arc<CircuitBreaker>,
    /// Present only with `--request-transport nats`. Which workers actually go
    /// over it is still per-worker: one that failed to start its NATS consumer
    /// registers itself as `http` and is dialled directly, exactly as on the
    /// Python side.
    pub nats: Option<Arc<crate::nats_request::NatsRequestClient>>,
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/health", get(health))
        .route("/v1/workers", get(workers))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        // axum's default 2 MiB cap on `Bytes` would 413 long-context prompts, but
        // fully disabling the limit allows unbounded buffering into memory (DoS).
        // Raise the limit enough for expected prompts; the engine still enforces `--context-length`.
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(state)
}

async fn chat(State(st): State<AppState>, body: Bytes) -> Response {
    proxy::dispatch(&st, body, "/v1/chat/completions").await
}

async fn completions(State(st): State<AppState>, body: Bytes) -> Response {
    proxy::dispatch(&st, body, "/v1/completions").await
}

/// OpenAI Responses API — the wire protocol the Codex CLI/SDK speaks by default.
///
/// Stateless calls only. SGLang keeps `store`/`previous_response_id` state in a
/// per-process dict (`serving_responses.py`'s `response_store`), so the
/// retrieve/cancel sub-routes and conversation continuation cannot be routed
/// across a fleet; clients should send `store: false`.
///
/// kv-aware DOES cover these: `BlockHasher::hash_for` normalises the `input`
/// field into the chat body `OpenAIServingResponses._make_request` builds and
/// renders that, so `/v1/responses` and `/v1/chat/completions` hash identically.
/// The exception is a `previous_response_id` whose history lives in the engine
/// -- unreproducible by construction -- which routes on load and logs why.
async fn responses(State(st): State<AppState>, body: Bytes) -> Response {
    proxy::dispatch(&st, body, "/v1/responses").await
}

/// Anthropic Messages API translated through the OpenAI Chat worker path.
async fn messages(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(version) = headers
        .get("anthropic-version")
        .and_then(|value| value.to_str().ok())
        .filter(|version| *version != "2023-06-01")
    {
        tracing::info!(
            anthropic_version = version,
            "accepting an Anthropic API version not covered by regression tests"
        );
    }
    if headers.contains_key("x-api-key") || headers.contains_key(header::AUTHORIZATION) {
        tracing::debug!("Anthropic auth header accepted without router-side validation");
    }

    let mut anthropic_body: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            tracing::warn!(%error, "Anthropic Messages request contains malformed JSON");
            return anthropic_error(StatusCode::BAD_REQUEST, "malformed JSON in request body");
        }
    };
    let model = match anthropic_body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
    {
        Some(model) => model.to_string(),
        None => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "missing or empty 'model' field (required by Anthropic Messages spec)",
            )
        }
    };

    crate::cache_control::strip_internal_hints(&mut anthropic_body);
    let hints = parse_cache_hints(&anthropic_body);
    let mut openai_body = match anthropic::translate_request(&anthropic_body) {
        Ok(value) => value,
        Err(message) => return anthropic_error(StatusCode::BAD_REQUEST, &message),
    };
    let stream = openai_body
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if stream {
        let obj = openai_body
            .as_object_mut()
            .expect("translated Anthropic request is an object");
        let options = obj
            .entry("stream_options")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("stream_options inserted as an object");
        options.entry("include_usage").or_insert(Value::Bool(true));
    }

    let worker_body = match serde_json::to_vec(&openai_body) {
        Ok(encoded) => Bytes::from(encoded),
        Err(error) => {
            tracing::error!(%error, "failed to encode translated Anthropic request");
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to encode translated request",
            );
        }
    };
    let mut routing_request = openai_body;
    attach_cache_hints(&mut routing_request, &hints);

    let request_id = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("{:032x}", rand::random::<u128>()));
    let upstream =
        proxy::dispatch_routed(&st, &routing_request, worker_body, "/v1/chat/completions").await;
    let status = upstream.status();
    if status.is_client_error() || status.is_server_error() {
        return with_request_id(upstream, &request_id);
    }

    if stream {
        return translate_anthropic_stream(upstream, &model, &request_id);
    }

    let (parts, upstream_body) = upstream.into_parts();
    let bytes = match to_bytes(upstream_body, MAX_RESPONSE_BYTES).await {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "failed to read OpenAI response for Anthropic translation");
            return anthropic_error(StatusCode::BAD_GATEWAY, "failed to read worker response");
        }
    };
    let openai_response: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => {
            let response = Response::from_parts(parts, Body::from(bytes));
            return with_request_id(response, &request_id);
        }
    };
    let translated =
        anthropic::translate_response(&openai_response, Some(&model), Some(&request_id));
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(REQUEST_ID_HEADER, request_id)
        .body(Body::from(translated.to_string()))
        .expect("Anthropic JSON response is valid")
}

/// Convert an OpenAI SSE body to Anthropic events without buffering the stream.
fn translate_anthropic_stream(upstream: Response, model: &str, request_id: &str) -> Response {
    let (_, body) = upstream.into_parts();
    let source = body.into_data_stream();
    let translator = SseTranslator::new(model, Some(request_id));
    let translated = futures::stream::unfold(
        (source, translator, false),
        |(mut source, mut translator, finished)| async move {
            if finished {
                return None;
            }
            loop {
                match source.next().await {
                    Some(Ok(chunk)) => {
                        let output = translator.push(&chunk);
                        if !output.is_empty() {
                            return Some((
                                Ok::<Bytes, axum::Error>(Bytes::from(output)),
                                (source, translator, false),
                            ));
                        }
                    }
                    Some(Err(error)) => {
                        let output = translator.error(&format!("worker stream failed: {error}"));
                        return Some((
                            Ok::<Bytes, axum::Error>(Bytes::from(output)),
                            (source, translator, true),
                        ));
                    }
                    None => {
                        let output = translator.finish();
                        if output.is_empty() {
                            return None;
                        }
                        return Some((Ok(Bytes::from(output)), (source, translator, true)));
                    }
                }
            }
        },
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(REQUEST_ID_HEADER, request_id)
        .body(Body::from_stream(translated))
        .expect("Anthropic SSE response is valid")
}

/// Build an Anthropic-compatible error response.
fn anthropic_error(status: StatusCode, message: &str) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": message,
                },
            })
            .to_string(),
        ))
        .expect("Anthropic error response is valid")
}

/// Attach the router request id to an existing response.
fn with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = request_id.parse() {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

async fn health(State(st): State<AppState>) -> impl IntoResponse {
    let snap = st.pool.load();
    Json(json!({ "status": "ok", "active_workers": snap.active_count() }))
}

async fn workers(State(st): State<AppState>) -> impl IntoResponse {
    let snap = st.pool.load();
    let list: Vec<&crate::pool::Worker> = snap.all.iter().map(|w| w.as_ref()).collect();
    Json(json!({ "workers": list }))
}

async fn models(State(st): State<AppState>) -> impl IntoResponse {
    let snap = st.pool.load();
    let mut seen = std::collections::BTreeSet::new();
    for w in &snap.all {
        if w.is_active() && !w.model_name.is_empty() {
            seen.insert(w.model_name.clone());
        }
    }
    let data: Vec<_> = seen
        .into_iter()
        .map(|id| json!({ "id": id, "object": "model", "owned_by": "infera" }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn metrics(State(st): State<AppState>) -> impl IntoResponse {
    let snap = st.pool.load();
    let mut out = format!(
        "# infera-router (rust)\n\
         infera_router_active_workers {}\n\
         infera_router_uptime_seconds {}\n",
        snap.active_count(),
        st.started.elapsed().as_secs()
    );
    // Non-zero state means the router is routing around a worker that
    // discovery still reports ACTIVE — the gap this metric exists to show.
    for (worker_id, state, trips) in st.breaker.snapshot() {
        let v = match state {
            crate::breaker::BreakerState::Closed => 0,
            crate::breaker::BreakerState::HalfOpen => 1,
            crate::breaker::BreakerState::Open => 2,
        };
        // Escaped: worker ids come from discovery records, and a stray quote,
        // backslash or newline in one would not corrupt a single line but end
        // the whole exposition, failing every scrape of this endpoint.
        let worker_id = escape_label_value(&worker_id);
        out.push_str(&format!(
            "infera_router_worker_breaker_state{{worker_id=\"{worker_id}\"}} {v}\n\
             infera_router_worker_breaker_trips_total{{worker_id=\"{worker_id}\"}} {trips}\n"
        ));
    }
    // A 0 here means kv-aware routing is silently off for that worker: every
    // block hash misses, the policy degrades to load balancing, and no other
    // signal shows it. Alert on 0; -1 just means the engine served no tokenize
    // endpoint to check against.
    out.push_str(
        "# HELP infera_router_render_parity 1 = this worker confirmed the router renders \
             the prompt it does, 0 = DIVERGED (kv-aware is off for it), -1 = not checkable\n\
         # TYPE infera_router_render_parity gauge\n",
    );
    for (worker_id, model, v) in st.policy.render_parity() {
        out.push_str(&format!(
            "infera_router_render_parity{{worker_id=\"{}\",model=\"{}\"}} {v}\n",
            escape_label_value(&worker_id),
            escape_label_value(&model),
        ));
    }
    // Survives the worker, unlike the gauge above: `retain` drops a departed
    // worker's series, so a fleet rolling THROUGH broken replicas leaves no
    // gauge behind to alert on. Mirrors the Python router's counter of the
    // same name -- an alert on `increase(...[1h]) > 0` used to be silently
    // dead on this backend.
    let diverged = st.policy.render_parity_diverged();
    if !diverged.is_empty() {
        out.push_str(
            "# HELP infera_router_render_parity_diverged_total workers ever found to render \
                 prompts differently from this router\n\
             # TYPE infera_router_render_parity_diverged_total counter\n",
        );
        for (model, n) in diverged {
            out.push_str(&format!(
                "infera_router_render_parity_diverged_total{{model=\"{}\"}} {n}\n",
                escape_label_value(&model),
            ));
        }
    }
    // The server-side template defaults each worker reported. Watch the number
    // of DISTINCT variant labels before trusting any of this: 1 means the fleet
    // is uniform and a single --kv-default-chat-template-kwargs would have been
    // enough; 2 or more means it would not, and the per-worker tier is load
    // bearing. A worker missing from here is one the router could not ask, and
    // is being hashed with the router's own default.
    let variants = st.policy.render_variants();
    if !variants.is_empty() {
        out.push_str(
            "# HELP infera_router_render_variant the --default-chat-template-kwargs this \
                 worker reported, which the router renders its requests with\n\
             # TYPE infera_router_render_variant gauge\n",
        );
        for (worker_id, label) in variants {
            out.push_str(&format!(
                "infera_router_render_variant{{worker_id=\"{}\",variant=\"{}\"}} 1\n",
                escape_label_value(&worker_id),
                escape_label_value(&label),
            ));
        }
    }
    out
}

/// Escape a Prometheus label value: backslash, double quote and newline, per
/// the text exposition format.
fn escape_label_value(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}
