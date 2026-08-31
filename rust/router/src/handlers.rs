///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! axum HTTP surface + shared app state.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use crate::breaker::CircuitBreaker;
use crate::policy::Policy;
use crate::pool::SharedPool;
use crate::proxy;

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
