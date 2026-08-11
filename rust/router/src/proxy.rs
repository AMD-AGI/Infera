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
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::Response;
use futures::Stream;
use serde_json::Value;

use crate::breaker::is_worker_fault;
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

/// One attempt over NATS, with the same contract as the HTTP one: `Err` only
/// before any byte reached the client, so the caller may fail over.
///
/// Streaming commits on the first data frame -- after that a failure can only
/// be reported inside the stream, because the client already has a 200 and
/// part of a body. Unary accumulates server-side, so anything that goes wrong
/// is still failoverable.
async fn attempt_nats(
    nats: &Arc<crate::nats_request::NatsRequestClient>,
    target: &RouteTarget,
    raw: &Bytes,
    stream: bool,
    path: &str,
    guard: ActiveGuard,
) -> Result<Response, Response> {
    use crate::nats_request::Frame;

    let worker = &target.worker;
    let wid = worker.worker_id.clone();
    let body: serde_json::Value = match serde_json::from_slice(raw) {
        Ok(v) => v,
        Err(e) => {
            return Err(json_error(
                StatusCode::BAD_REQUEST,
                &format!("request body is not JSON: {e}"),
            ))
        }
    };
    // Same envelope the Python router publishes; the worker side is shared.
    let mut headers = serde_json::Map::new();
    if let Some(r) = target.dp_rank {
        headers.insert(
            dp::DP_RANK_HEADER.to_string(),
            serde_json::Value::String(r.to_string()),
        );
    }
    let payload = serde_json::json!({
        "path": path,
        "stream": stream,
        "headers": if headers.is_empty() { serde_json::Value::Null } else { serde_json::Value::Object(headers) },
        "body": body,
    });
    let encoded = match serde_json::to_vec(&payload) {
        Ok(v) => v,
        Err(e) => {
            return Err(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("encoding the nats request: {e}"),
            ))
        }
    };

    let mut reply = match nats.dispatch(&wid, &encoded).await {
        Ok(r) => r,
        Err(e) => {
            return Err(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {wid} unreachable over nats: {e}"),
            ))
        }
    };

    if !stream {
        let mut chunks: Vec<Bytes> = Vec::new();
        let mut status = StatusCode::OK;
        loop {
            match reply.next().await {
                Some(Frame::Data(b)) => chunks.push(b),
                Some(Frame::Done { status: s }) => {
                    status = StatusCode::from_u16(s).unwrap_or(StatusCode::OK);
                    break;
                }
                Some(Frame::Error { status: s, message }) => {
                    return Err(json_error(
                        s.and_then(|c| StatusCode::from_u16(c).ok())
                            .unwrap_or(StatusCode::BAD_GATEWAY),
                        &format!("worker {wid} nats failed: {}", trim(&message)),
                    ))
                }
                None => break,
            }
        }
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut buf = Vec::with_capacity(total);
        for c in &chunks {
            buf.extend_from_slice(c);
        }
        // A 5xx before any data is a worker fault and retryable; a 4xx belongs
        // to the request and every worker would answer the same, so it is
        // returned rather than retried. Same rule as the HTTP path.
        if is_worker_fault(status.as_u16()) {
            return Err(Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(buf))
                .expect("unary response is valid"));
        }
        let _guard = guard;
        return Ok(Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(buf))
            .expect("unary response is valid"));
    }

    // Peek one frame: data commits to this worker, anything else can still be
    // retried elsewhere.
    let first = match reply.next().await {
        Some(Frame::Data(b)) => b,
        Some(Frame::Error { status: s, message }) => {
            return Err(json_error(
                s.and_then(|c| StatusCode::from_u16(c).ok())
                    .unwrap_or(StatusCode::BAD_GATEWAY),
                &format!("worker {wid} nats stream failed: {}", trim(&message)),
            ))
        }
        Some(Frame::Done { status }) => {
            let code = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
            if is_worker_fault(code.as_u16()) {
                return Err(json_error(
                    code,
                    &format!("worker {wid} ended the stream with {code}"),
                ));
            }
            // Nothing to stream, but not a fault: answer with an empty body
            // rather than failing over a request the worker considers done.
            let _guard = guard;
            return Ok(Response::builder()
                .status(code)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::empty())
                .expect("stream response is valid"));
        }
        None => {
            return Err(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {wid} closed the nats reply with no frames"),
            ))
        }
    };

    // Committed. The guard moves into the body so the policy's in-flight count
    // is released when the client finishes, disconnects, or drops -- and the
    // ReplyStream moves in with it, so dropping the body cancels the worker.
    // `None` for the reply state ends the stream.
    let body = futures::stream::unfold(
        (Some(reply), Some(first), wid.clone()),
        |(reply, pending, wid)| async move {
            if let Some(b) = pending {
                return Some((Ok::<Bytes, std::io::Error>(b), (reply, None, wid)));
            }
            let mut r = reply?;
            match r.next().await {
                Some(Frame::Data(b)) => Some((Ok(b), (Some(r), None, wid))),
                Some(Frame::Error { message, .. }) => {
                    tracing::warn!(
                        "stream from worker {wid} failed mid-stream: {}",
                        trim(&message)
                    );
                    // The client already has a 200 and part of a body, so the
                    // failure can only be delivered inside the stream.
                    let chunk = Bytes::from(format!(
                        "data: {{\"error\":\"worker {wid} stream failed mid-stream\"}}\n\n"
                    ));
                    Some((Ok(chunk), (None, None, wid)))
                }
                Some(Frame::Done { .. }) | None => None,
            }
        },
    );
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(GuardedBody {
            inner: Box::pin(body),
            _guard: guard,
        }))
        .expect("stream response is valid"))
}

fn trim(s: &str) -> &str {
    &s[..s.len().min(500)]
}

/// `GuardedStream` for a non-reqwest stream: same job, different item type.
struct GuardedBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    _guard: ActiveGuard,
}

impl Stream for GuardedBody {
    type Item = Result<Bytes, std::io::Error>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
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
        // Drop workers the breaker has open. Falls back to the unfiltered list
        // when every candidate is open — a request served by a probably-bad
        // worker beats turning a partial outage into a 503.
        let avail = state.breaker.filter(&avail, |w| w.worker_id.as_str());
        let pick = state.policy.pick(&avail, request, Role::Mixed);
        tried.insert(pick.target.worker.worker_id.clone());
        // Load guard: started here, dropped when this attempt fails (fail-over)
        // or — on success — when the response body is fully sent.
        let guard = ActiveGuard::start(
            state.policy.clone(),
            vec![(pick.target.route_key(), pick.blocks.clone())],
        );
        let wid = pick.target.worker.worker_id.clone();
        match attempt(state, &pick.target, &raw, stream, path, guard).await {
            Ok(resp) => {
                state.breaker.record_success(&wid);
                return resp;
            }
            Err(err_resp) => {
                // `attempt` only returns Err before any byte reached the client,
                // so a mid-stream failure can never trip the breaker. 4xx is
                // failed over but not held against the worker — see
                // is_worker_fault().
                if is_worker_fault(err_resp.status().as_u16()) {
                    state.breaker.record_failure(&wid);
                } else {
                    // A 4xx is not held against the worker, but the probe slot
                    // it consumed has to come back or one bad client wedges a
                    // recovering worker out of rotation.
                    state.breaker.record_neutral(&wid);
                }
                last_err = Some(err_resp);
            }
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
    // Per worker, not per router: a worker whose NATS consumer failed to start
    // registers as `http` and is dialled directly even when the transport is
    // otherwise NATS.
    if worker.request_transport == "nats" {
        if let Some(nats) = state.nats.clone() {
            return attempt_nats(&nats, target, raw, stream, path, guard).await;
        }
        return Err(json_error(
            StatusCode::BAD_GATEWAY,
            &format!(
                "worker {} registered the nats transport but this router has none",
                worker.worker_id
            ),
        ));
    }
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
