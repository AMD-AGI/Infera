///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! axum HTTP surface + shared app state.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

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
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/health", get(health))
        .route("/v1/workers", get(workers))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn chat(State(st): State<AppState>, body: Bytes) -> Response {
    proxy::dispatch(&st, body, "/v1/chat/completions").await
}

async fn completions(State(st): State<AppState>, body: Bytes) -> Response {
    proxy::dispatch(&st, body, "/v1/completions").await
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
    format!(
        "# infera-router (rust)\n\
         infera_router_active_workers {}\n\
         infera_router_uptime_seconds {}\n",
        snap.active_count(),
        st.started.elapsed().as_secs()
    )
}
