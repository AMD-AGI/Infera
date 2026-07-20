///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Request dispatch + mixed (non-PD) forward with pre-first-byte failover.
//! The streaming path relays the worker's SSE bytes verbatim via
//! `Body::from_stream`, so per-token work runs on Tokio's threads, not ours.

use std::collections::HashSet;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::Response;
use futures::Stream;
use serde_json::Value;

use crate::dp;
use crate::handlers::AppState;
use crate::policy::{ActiveGuard, Role};
use crate::pool::{DisaggMode, RouteTarget, Snapshot};
use crate::util::json_error;

/// A byte stream that owns an `ActiveGuard`: when the streamed body ends (client
/// done, disconnect, or drop), the guard drops and fires `on_request_finished`,
/// so a cost-aware policy's in-flight load stays balanced for streamed requests.
pub(crate) struct GuardedStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    _guard: ActiveGuard,
}

impl GuardedStream {
    pub(crate) fn new(
        inner: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
        guard: ActiveGuard,
    ) -> Self {
        GuardedStream {
            inner: Box::pin(inner),
            _guard: guard,
        }
    }
}

impl Stream for GuardedStream {
    type Item = reqwest::Result<Bytes>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // GuardedStream is Unpin (Pin<Box<..>> + ActiveGuard are both Unpin).
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

/// Upstream client: unbounded connection pool, no read timeout (generations run
/// arbitrarily long), bounded connect so unreachable workers fail fast.
pub fn build_upstream_client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(60))
        .pool_max_idle_per_host(1024)
        .build()?)
}

pub async fn dispatch(state: &AppState, raw: Bytes, path: &'static str) -> Response {
    let v: serde_json::Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let model = v.get("model").and_then(|m| m.as_str()).unwrap_or("");
    let stream = v.get("stream").and_then(|b| b.as_bool()).unwrap_or(false);

    let guard = state.pool.load();
    let snap: &Snapshot = &guard;

    let has_p = !snap.list_active(model, DisaggMode::Prefill).is_empty();
    let has_d = !snap.list_active(model, DisaggMode::Decode).is_empty();
    if has_p && has_d {
        return crate::disagg::dispatch(state, snap, model, &v, raw, stream, path).await;
    }
    mixed_dispatch(state, snap, model, &v, raw, stream, path).await
}

async fn mixed_dispatch(
    state: &AppState,
    snap: &Snapshot,
    model: &str,
    request: &Value,
    raw: Bytes,
    stream: bool,
    path: &str,
) -> Response {
    let candidates = snap.list_active(model, DisaggMode::Mixed);
    if candidates.is_empty() {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("no active mixed worker for model={model:?}"),
        );
    }

    let mut tried: HashSet<String> = HashSet::new();
    let mut last_err: Option<Response> = None;
    for _ in 0..(1 + state.retries) {
        let avail: Vec<_> = candidates
            .iter()
            .filter(|w| !tried.contains(&w.worker_id))
            .cloned()
            .collect();
        if avail.is_empty() {
            break;
        }
        let pick = state.policy.pick(&avail, request, Role::Mixed);
        tried.insert(pick.target.worker.worker_id.clone());
        // Load guard: started here, dropped when this attempt fails (fail-over)
        // or — on success — when the response body is fully sent.
        let guard = ActiveGuard::start(
            state.policy.clone(),
            vec![(pick.target.route_key(), pick.blocks.clone())],
        );
        match attempt(state, &pick.target, &raw, stream, path, guard).await {
            Ok(resp) => return resp,
            Err(err_resp) => last_err = Some(err_resp),
        }
    }
    last_err.unwrap_or_else(|| json_error(StatusCode::SERVICE_UNAVAILABLE, "all workers failed"))
}

/// One attempt. `Err(resp)` means the failure happened before any client data
/// was sent (unreachable / >=400 before streaming), so the caller may fail over.
/// `guard` is held for the whole attempt: on a streamed success it's moved into
/// the response body, otherwise it drops here (balancing the load refcount).
async fn attempt(
    state: &AppState,
    target: &RouteTarget,
    raw: &Bytes,
    stream: bool,
    path: &str,
    guard: ActiveGuard,
) -> Result<Response, Response> {
    let worker = &target.worker;
    let url = format!("{}{}", worker.url, path);
    let mut req = state
        .http
        .post(&url)
        .header(header::CONTENT_TYPE, "application/json")
        .body(raw.clone());
    if let Some(r) = target.dp_rank {
        req = req.header(dp::DP_RANK_HEADER, r.to_string());
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return Err(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {} unreachable: {e}", worker.worker_id),
            ))
        }
    };

    let status = resp.status();
    if status.is_client_error() || status.is_server_error() {
        let body = resp.text().await.unwrap_or_default();
        return Err(json_error(
            status,
            &format!(
                "worker {} error {}: {}",
                worker.worker_id,
                status.as_u16(),
                &body[..body.len().min(500)]
            ),
        ));
    }

    if stream {
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(GuardedStream::new(
                resp.bytes_stream(),
                guard,
            )))
            .expect("stream response is valid"))
    } else {
        let _guard = guard; // held until the unary body is read below
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        match resp.bytes().await {
            Ok(bytes) => Ok(Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, ct)
                .body(Body::from(bytes))
                .expect("unary response is valid")),
            Err(e) => Err(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {} read failed: {e}", worker.worker_id),
            )),
        }
    }
}
