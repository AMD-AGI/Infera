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

use tokio::sync::oneshot;

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
use crate::util::{json_error, truncate_chars};

type AttemptResult = Result<Response, Box<Response>>;

const SSE_DONE_MARKER: &[u8] = b"data: [DONE]";
const RESPONSES_DONE_MARKER: &[u8] = b"event: response.completed";

/// Rolling match for an SSE success marker, which may straddle two chunks.
struct DoneMarker {
    marker: &'static [u8],
    matched: usize,
    seen: bool,
}

impl DoneMarker {
    fn for_path(path: &str) -> Self {
        DoneMarker {
            marker: if path == "/v1/responses" {
                RESPONSES_DONE_MARKER
            } else {
                SSE_DONE_MARKER
            },
            matched: 0,
            seen: false,
        }
    }

    fn feed(&mut self, chunk: &[u8]) {
        if self.seen {
            return;
        }
        for &byte in chunk {
            // The marker has no proper border, so a mismatch can only restart
            // the match at the byte that failed it.
            self.matched = if byte == self.marker[self.matched] {
                self.matched + 1
            } else {
                usize::from(byte == self.marker[0])
            };
            if self.matched == self.marker.len() {
                self.seen = true;
                return;
            }
        }
    }
}

/// A byte stream that owns an `ActiveGuard`: when the streamed body ends (client
/// done, disconnect, or drop), the guard drops and fires `on_request_finished`,
/// so a cost-aware policy's in-flight load stays balanced for streamed requests.
///
/// With an `on_end` channel the stream also reports how it ended. Completion is
/// then the SSE terminator, not EOF: an upstream that closes mid-generation
/// ends the body without error, and reading that as success would leave the
/// other PD leg holding the bootstrap room with nobody aborting it.
pub(crate) struct GuardedStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
    _guard: ActiveGuard,
    completed: bool,
    failed: bool,
    // Only tracked for an `on_end` stream; `None` keeps EOF meaning completion.
    done: Option<DoneMarker>,
    on_end: Option<oneshot::Sender<StreamEnd>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamEnd {
    Complete,
    Incomplete,
}

impl GuardedStream {
    pub(crate) fn new(
        inner: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
        guard: ActiveGuard,
    ) -> Self {
        Self::new_with_incomplete_abort(inner, guard, "", None)
    }

    pub(crate) fn new_with_incomplete_abort(
        inner: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
        guard: ActiveGuard,
        path: &str,
        on_end: Option<oneshot::Sender<StreamEnd>>,
    ) -> Self {
        GuardedStream {
            inner: Box::pin(inner),
            _guard: guard,
            completed: false,
            failed: false,
            done: on_end.as_ref().map(|_| DoneMarker::for_path(path)),
            on_end,
        }
    }
}

impl Stream for GuardedStream {
    type Item = reqwest::Result<Bytes>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // GuardedStream is Unpin (Pin<Box<..>> + ActiveGuard are both Unpin).
        let this = self.get_mut();
        let out = this.inner.as_mut().poll_next(cx);
        match &out {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(done) = this.done.as_mut() {
                    done.feed(chunk);
                    // The client holds the whole answer as soon as the
                    // terminator reaches it, and may drop the body without
                    // ever polling for EOF.
                    this.completed = done.seen;
                }
            }
            // A break after the terminator still leaves the client short of
            // the bytes it was about to read.
            Poll::Ready(Some(Err(_))) => {
                this.failed = true;
                this.completed = false;
            }
            Poll::Ready(None) => {
                let terminated = this.done.as_ref().is_none_or(|d| d.seen);
                this.completed = !this.failed && terminated;
            }
            _ => {}
        }
        out
    }
}

impl Drop for GuardedStream {
    fn drop(&mut self) {
        if let Some(tx) = self.on_end.take() {
            let end = if self.completed {
                StreamEnd::Complete
            } else {
                StreamEnd::Incomplete
            };
            let _ = tx.send(end);
        }
    }
}

/// One attempt over NATS, with the same contract as the HTTP one: `Err` only
/// before any byte reached the client, so the caller may fail over.
///
/// Streaming commits on the first data frame -- after that a failure can only
/// be reported inside the stream, because the client already has a 200 and
/// part of a body. Unary accumulates server-side, so anything that goes wrong
/// is still failoverable.
// clippy's result_large_err wants the Err boxed. Both variants are the same
// `Response` here: this Result is a two-way tag -- "already sent, do not fail
// over" vs "nothing sent, you may" -- and not an error channel. Boxing only
// the Err would take the return from 136 to 128 bytes, since `Response` alone
// is 128 and the Ok side pins that floor, and would pay an allocation on every
// failover for those 8. What the lint guards against is a large Err riding up
// a `?` chain: neither of these is propagated with `?`, both are matched one
// frame up.
#[allow(clippy::result_large_err)]
async fn attempt_nats(
    nats: &Arc<crate::nats_request::NatsRequestClient>,
    target: &RouteTarget,
    raw: &Bytes,
    stream: bool,
    path: &str,
    guard: ActiveGuard,
) -> AttemptResult {
    use crate::nats_request::Frame;

    let worker = &target.worker;
    let wid = worker.worker_id.clone();

    // A worker at its backlog limit has not seen a byte of this request, so
    // this is a pre-first-byte failure: returning Err lets the caller try a
    // freer worker, and only a fleet-wide backlog reaches the client as 429.
    // Without this the throttle changed the publish path to JetStream without
    // ever refusing anything outside PD.
    if !nats.admit(&wid).await {
        // Built through serde rather than formatted: a worker id comes from a
        // Pod annotation and a quote in it would produce malformed JSON.
        let body =
            serde_json::json!({ "error": format!("worker {wid} request backlog over limit") })
                .to_string();
        return Err(Box::new(
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header(header::CONTENT_TYPE, "application/json")
                .header("Retry-After", "1")
                .body(Body::from(body))
                .expect("429 response is valid"),
        ));
    }

    let body: serde_json::Value = match serde_json::from_slice(raw) {
        Ok(v) => v,
        Err(e) => {
            return Err(Box::new(json_error(
                StatusCode::BAD_REQUEST,
                &format!("request body is not JSON: {e}"),
            )))
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
            return Err(Box::new(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("encoding the nats request: {e}"),
            )))
        }
    };

    let mut reply = match nats.dispatch(&wid, &encoded).await {
        Ok(r) => r,
        Err(e) => {
            return Err(Box::new(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {wid} unreachable over nats: {e}"),
            )))
        }
    };

    if !stream {
        let mut chunks: Vec<Bytes> = Vec::new();
        let mut status = StatusCode::OK;
        let mut done_seen = false;
        loop {
            match reply.next().await {
                Some(Frame::Data(b)) => chunks.push(b),
                Some(Frame::Done { status: s }) => {
                    status = StatusCode::from_u16(s).unwrap_or(StatusCode::OK);
                    done_seen = true;
                    break;
                }
                Some(Frame::Error { status: s, message }) => {
                    return Err(Box::new(json_error(
                        s.and_then(|c| StatusCode::from_u16(c).ok())
                            .unwrap_or(StatusCode::BAD_GATEWAY),
                        &format!("worker {wid} nats failed: {}", trim(&message)),
                    )))
                }
                None => break,
            }
        }
        // The subscription ended without the worker saying it was done. That
        // is not a 200 with a short body: it is a worker that stopped talking
        // mid-request, and reporting it as success would also record it as
        // healthy against the breaker.
        if !done_seen {
            return Err(Box::new(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {wid} closed the nats reply without finishing"),
            )));
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
            return Err(Box::new(
                Response::builder()
                    .status(status)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(buf))
                    .expect("unary response is valid"),
            ));
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
            return Err(Box::new(json_error(
                s.and_then(|c| StatusCode::from_u16(c).ok())
                    .unwrap_or(StatusCode::BAD_GATEWAY),
                &format!("worker {wid} nats stream failed: {}", trim(&message)),
            )))
        }
        Some(Frame::Done { status }) => {
            let code = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
            if is_worker_fault(code.as_u16()) {
                return Err(Box::new(json_error(
                    code,
                    &format!("worker {wid} ended the stream with {code}"),
                )));
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
            return Err(Box::new(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {wid} closed the nats reply with no frames"),
            )))
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
        .body(Body::from_stream(guarded(body, guard)))
        .expect("stream response is valid"))
}

fn trim(s: &str) -> &str {
    truncate_chars(s, 500)
}

/// `GuardedStream` for a non-reqwest stream: same job, different item type.
pub(crate) struct GuardedBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    _guard: ActiveGuard,
    completed: bool,
    failed: bool,
    // Only tracked for an `on_end` body; the frame protocol carries its own
    // terminator, so the marker only makes completion observable earlier.
    done: Option<DoneMarker>,
    on_end: Option<oneshot::Sender<StreamEnd>>,
}

/// Tie a byte stream to an `ActiveGuard`, so the policy's in-flight count is
/// released when the client finishes, disconnects, or drops.
pub(crate) fn guarded(
    inner: impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    guard: ActiveGuard,
) -> GuardedBody {
    guarded_with_incomplete_abort(inner, guard, "", None)
}

pub(crate) fn guarded_with_incomplete_abort(
    inner: impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    guard: ActiveGuard,
    path: &str,
    on_end: Option<oneshot::Sender<StreamEnd>>,
) -> GuardedBody {
    GuardedBody {
        inner: Box::pin(inner),
        _guard: guard,
        completed: false,
        failed: false,
        done: on_end.as_ref().map(|_| DoneMarker::for_path(path)),
        on_end,
    }
}

impl Stream for GuardedBody {
    type Item = Result<Bytes, std::io::Error>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let out = this.inner.as_mut().poll_next(cx);
        match &out {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(done) = this.done.as_mut() {
                    done.feed(chunk);
                    // Same race as the HTTP leg: the client can drop the body
                    // between the terminal bytes and the done frame.
                    this.completed = done.seen;
                }
            }
            Poll::Ready(Some(Err(_))) => {
                this.failed = true;
                this.completed = false;
            }
            Poll::Ready(None) => this.completed = !this.failed,
            _ => {}
        }
        out
    }
}

impl Drop for GuardedBody {
    fn drop(&mut self) {
        if let Some(tx) = self.on_end.take() {
            let end = if self.completed {
                StreamEnd::Complete
            } else {
                StreamEnd::Incomplete
            };
            let _ = tx.send(end);
        }
    }
}

/// Upstream client: unbounded connection pool, bounded connect so unreachable
/// workers fail fast, and a read timeout that only a stall can trip.
///
/// Deliberately not a total timeout: reqwest resets the read timeout on every
/// chunk, so a generation that keeps producing tokens runs as long as it needs
/// while an engine that goes quiet fails the stream instead of holding the
/// client open. The NATS transport has always had this; this is the HTTP half.
pub fn build_upstream_client(idle_timeout_s: f64) -> anyhow::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(60))
        .pool_max_idle_per_host(1024);
    if idle_timeout_s > 0.0 {
        builder = builder.read_timeout(Duration::from_secs_f64(idle_timeout_s));
    }
    Ok(builder.build()?)
}

pub async fn dispatch(state: &AppState, raw: Bytes, path: &'static str) -> Response {
    let mut v: serde_json::Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    // Drop a client-supplied stamp, then attach the one we parsed. Matches
    // Python `app.py` covering every OpenAI-shaped entry, including
    // `/v1/responses` whose converted chat body has no `prompt_cache_*`.
    crate::cache_control::strip_internal_hints(&mut v);
    let hints = crate::cache_control::parse_cache_hints(&v);
    let mut routing = v.clone();
    crate::cache_control::attach_cache_hints(&mut routing, &hints);
    let raw = match serde_json::to_vec(&v) {
        Ok(bytes) => Bytes::from(bytes),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    dispatch_routed(state, &routing, raw, path).await
}

/// Dispatch an encoded worker body using a separate routing representation.
///
/// Protocol adapters use this to attach router-only metadata without leaking
/// private fields to OpenAI-compatible workers.
pub(crate) async fn dispatch_routed(
    state: &AppState,
    routing_request: &Value,
    raw: Bytes,
    path: &'static str,
) -> Response {
    let model = routing_request
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("");
    let stream = routing_request
        .get("stream")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    let guard = state.pool.load();
    let snap: &Snapshot = &guard;

    let prefill = snap.list_active(model, DisaggMode::Prefill);
    let decode = snap.list_active(model, DisaggMode::Decode);
    let mixed = snap.list_active(model, DisaggMode::Mixed);
    let has_p = !prefill.is_empty();
    let has_d = !decode.is_empty();
    if has_p && has_d {
        return crate::disagg::dispatch(state, snap, model, routing_request, raw, stream, path)
            .await;
    }
    if has_p != has_d && mixed.is_empty() {
        let (present, missing, count) = if has_p {
            ("prefill", "decode", prefill.len())
        } else {
            ("decode", "prefill", decode.len())
        };
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!(
                "model={model:?} has {count} {present} worker(s) but no {missing} worker; \
                 PD dispatch requires both pools"
            ),
        );
    }
    if !has_p && !has_d && mixed.is_empty() {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("no active worker for model={model:?}"),
        );
    }
    mixed_dispatch(state, snap, model, routing_request, raw, stream, path).await
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
                last_err = Some(*err_resp);
            }
        }
    }
    last_err.unwrap_or_else(|| json_error(StatusCode::SERVICE_UNAVAILABLE, "all workers failed"))
}

/// One attempt. `Err(resp)` means the failure happened before any client data
/// was sent (unreachable / >=400 before streaming), so the caller may fail over.
/// `guard` is held for the whole attempt: on a streamed success it's moved into
/// the response body, otherwise it drops here (balancing the load refcount).
// Same two-way-tag Result as attempt_nats -- see the note there.
#[allow(clippy::result_large_err)]
async fn attempt(
    state: &AppState,
    target: &RouteTarget,
    raw: &Bytes,
    stream: bool,
    path: &str,
    guard: ActiveGuard,
) -> AttemptResult {
    let worker = &target.worker;
    // Per worker, not per router: a worker whose NATS consumer failed to start
    // registers as `http` and is dialled directly even when the transport is
    // otherwise NATS.
    if worker.request_transport == "nats" {
        if let Some(nats) = state.nats.clone() {
            return attempt_nats(&nats, target, raw, stream, path, guard).await;
        }
        return Err(Box::new(json_error(
            StatusCode::BAD_GATEWAY,
            &format!(
                "worker {} registered the nats transport but this router has none",
                worker.worker_id
            ),
        )));
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
            return Err(Box::new(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {} unreachable: {e}", worker.worker_id),
            )))
        }
    };

    let status = resp.status();
    if status.is_client_error() || status.is_server_error() {
        let body = resp.text().await.unwrap_or_default();
        return Err(Box::new(json_error(
            status,
            &format!(
                "worker {} error {}: {}",
                worker.worker_id,
                status.as_u16(),
                truncate_chars(&body, 500)
            ),
        )));
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
            Err(e) => Err(Box::new(json_error(
                StatusCode::BAD_GATEWAY,
                &format!("worker {} read failed: {e}", worker.worker_id),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::RoundRobin;
    use futures::StreamExt;

    fn guard() -> ActiveGuard {
        ActiveGuard::start(Arc::new(RoundRobin::new()), Vec::new())
    }

    fn sse(chunks: &[&'static str]) -> impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static {
        let items: Vec<reqwest::Result<Bytes>> = chunks
            .iter()
            .map(|c| Ok(Bytes::from_static(c.as_bytes())))
            .collect();
        futures::stream::iter(items)
    }

    /// Drain a guarded PD stream and report what it told the abort watcher.
    async fn end_of(chunks: &[&'static str]) -> StreamEnd {
        let (tx, rx) = oneshot::channel();
        let mut stream = GuardedStream::new_with_incomplete_abort(
            sse(chunks),
            guard(),
            "/v1/chat/completions",
            Some(tx),
        );
        while stream.next().await.is_some() {}
        drop(stream);
        rx.await.expect("the guarded stream reports its end")
    }

    async fn responses_end_of(chunks: &[&'static str]) -> StreamEnd {
        let (tx, rx) = oneshot::channel();
        let mut stream = GuardedStream::new_with_incomplete_abort(
            sse(chunks),
            guard(),
            "/v1/responses",
            Some(tx),
        );
        while stream.next().await.is_some() {}
        drop(stream);
        rx.await.expect("the guarded stream reports its end")
    }

    #[tokio::test]
    async fn a_terminated_sse_stream_completes() {
        assert_eq!(
            end_of(&["data: {\"x\":1}\n\n", "data: [DONE]\n\n"]).await,
            StreamEnd::Complete
        );
    }

    #[tokio::test]
    async fn an_upstream_eof_without_the_terminator_is_incomplete() {
        // The decode leg closed mid-generation. The byte stream ended without
        // an error, so only the missing terminator separates this from a
        // finished request -- and the prefill leg still has to be aborted.
        assert_eq!(
            end_of(&["data: {\"x\":1}\n\n"]).await,
            StreamEnd::Incomplete
        );
    }

    #[tokio::test]
    async fn a_terminator_split_across_chunks_still_completes() {
        assert_eq!(
            end_of(&["data: {\"x\":1}\n\ndata: [DO", "NE]\n\n"]).await,
            StreamEnd::Complete
        );
    }

    #[tokio::test]
    async fn a_responses_terminator_split_across_chunks_still_completes() {
        assert_eq!(
            responses_end_of(&[
                "event: response.cre",
                "ated\ndata: {}\n\nevent: response.com",
                "pleted\ndata: {}\n\n"
            ])
            .await,
            StreamEnd::Complete
        );
    }

    #[tokio::test]
    async fn a_truncated_terminator_is_incomplete() {
        assert_eq!(end_of(&["data: [DON"]).await, StreamEnd::Incomplete);
    }

    #[tokio::test]
    async fn a_terminator_lookalike_does_not_complete() {
        // A generated token may spell the sentinel; only the SSE field does.
        assert_eq!(
            end_of(&["data: {\"content\":\"[DONE]\"}\n\n"]).await,
            StreamEnd::Incomplete
        );
    }

    #[tokio::test]
    async fn a_restarted_match_still_finds_the_terminator() {
        // The first attempt fails nine bytes in and the real marker begins at
        // the byte that broke it, so a matcher that only ever moves forward
        // would miss it.
        assert_eq!(
            end_of(&["data: [DOdata: [DON", "E]\n\n"]).await,
            StreamEnd::Complete
        );
    }

    #[tokio::test]
    async fn a_stream_without_an_abort_watcher_completes_on_eof() {
        // Mixed (non-PD) traffic has no pair to abort, so EOF stays completion.
        let mut stream = GuardedStream::new(sse(&["data: {\"x\":1}\n\n"]), guard());
        while stream.next().await.is_some() {}
        assert!(stream.completed);
    }

    /// A transport error, built without opening a socket: reqwest rejects the
    /// scheme before it dials.
    async fn transport_error() -> reqwest::Error {
        reqwest::Client::new()
            .get("ftp://127.0.0.1/")
            .send()
            .await
            .expect_err("reqwest refuses a non-http scheme")
    }

    #[tokio::test]
    async fn a_delivered_terminator_completes_a_dropped_stream() {
        // The client holds the whole answer once `data: [DONE]` reaches it and
        // may drop the body there, never polling the stream to EOF. Reading
        // that as incomplete would abort a request that already succeeded.
        let (tx, rx) = oneshot::channel();
        let mut stream = GuardedStream::new_with_incomplete_abort(
            sse(&["data: {\"x\":1}\n\n", "data: [DONE]\n\n"]),
            guard(),
            "/v1/chat/completions",
            Some(tx),
        );
        stream
            .next()
            .await
            .expect("the first chunk")
            .expect("a chunk");
        stream
            .next()
            .await
            .expect("the terminal chunk")
            .expect("a chunk");
        drop(stream);
        assert_eq!(
            rx.await.expect("the guarded stream reports its end"),
            StreamEnd::Complete
        );
    }

    #[tokio::test]
    async fn an_error_after_the_terminator_is_incomplete() {
        // The terminator was seen, but the body then broke before the client
        // could read it out, so the pair still has to be aborted.
        let (tx, rx) = oneshot::channel();
        let items: Vec<reqwest::Result<Bytes>> = vec![
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(transport_error().await),
        ];
        let mut stream = GuardedStream::new_with_incomplete_abort(
            futures::stream::iter(items),
            guard(),
            "/v1/chat/completions",
            Some(tx),
        );
        stream
            .next()
            .await
            .expect("the terminal chunk")
            .expect("a chunk");
        stream
            .next()
            .await
            .expect("the failure")
            .expect_err("a transport error");
        drop(stream);
        assert_eq!(
            rx.await.expect("the guarded stream reports its end"),
            StreamEnd::Incomplete
        );
    }

    fn nats_sse(
        chunks: &[&'static str],
    ) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
        let items: Vec<Result<Bytes, std::io::Error>> = chunks
            .iter()
            .map(|c| Ok(Bytes::from_static(c.as_bytes())))
            .collect();
        futures::stream::iter(items)
    }

    #[tokio::test]
    async fn a_delivered_terminator_completes_a_dropped_nats_body() {
        // Same race on the NATS PD leg: the client can drop the body between
        // the terminal bytes and the `Frame::Done` that ends the stream.
        let (tx, rx) = oneshot::channel();
        let mut body = guarded_with_incomplete_abort(
            nats_sse(&["event: response.completed\ndata: {}\n\n"]),
            guard(),
            "/v1/responses",
            Some(tx),
        );
        body.next()
            .await
            .expect("the terminal chunk")
            .expect("a chunk");
        drop(body);
        assert_eq!(
            rx.await.expect("the guarded body reports its end"),
            StreamEnd::Complete
        );
    }

    #[tokio::test]
    async fn a_nats_error_after_the_terminator_is_incomplete() {
        // `Frame::Done` with a 5xx, or a missing done frame, surfaces as an
        // error item after the terminal bytes; it still means incomplete.
        let (tx, rx) = oneshot::channel();
        let items: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(std::io::Error::other("decode NATS stream ended with 503")),
        ];
        let mut body = guarded_with_incomplete_abort(
            futures::stream::iter(items),
            guard(),
            "/v1/chat/completions",
            Some(tx),
        );
        body.next()
            .await
            .expect("the terminal chunk")
            .expect("a chunk");
        body.next()
            .await
            .expect("the failure")
            .expect_err("a stream error");
        drop(body);
        assert_eq!(
            rx.await.expect("the guarded body reports its end"),
            StreamEnd::Incomplete
        );
    }

    #[tokio::test]
    async fn a_nats_body_without_a_terminator_still_completes_on_eof() {
        // NATS carries its own terminator: a clean `Frame::Done` ends the
        // stream, so EOF stays completion even with no marker in the bytes.
        let (tx, rx) = oneshot::channel();
        let mut body = guarded_with_incomplete_abort(
            nats_sse(&["data: {\"x\":1}\n\n"]),
            guard(),
            "/v1/chat/completions",
            Some(tx),
        );
        while body.next().await.is_some() {}
        drop(body);
        assert_eq!(
            rx.await.expect("the guarded body reports its end"),
            StreamEnd::Complete
        );
    }
}
