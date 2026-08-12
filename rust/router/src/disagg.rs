///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! PD dual-dispatch for SGLang bootstrap (concurrent topology).
//!
//! Both legs get the same bootstrap fields and are POSTed concurrently; the
//! decode leg streams back to the client while the prefill leg runs to
//! completion in a detached task. The prefill task is NEVER cancelled — if its
//! request is dropped the bootstrap_room handoff is lost and decode hangs on
//! KVPoll until a ~300s timeout. A detached `tokio::spawn` gives us exactly
//! that: it outlives the client connection.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::Response;
use serde_json::{Map, Value};

use crate::breaker::{is_worker_fault, CircuitBreaker};
use crate::dp;
use crate::handlers::AppState;
use crate::nats_request::{Frame, NatsRequestClient};
use crate::policy::{ActiveGuard, Role};
use crate::pool::{DisaggMode, RouteTarget, Snapshot};
use crate::protocol;
use crate::proxy::GuardedStream;
use crate::util::{json_error, truncate_chars};

const DECODE_OPEN_RETRIES: u32 = 3;

/// Entry point. Caller guarantees the model has both prefill and decode workers.
pub async fn dispatch(
    state: &AppState,
    snap: &Snapshot,
    model: &str,
    request: &Value,
    raw: Bytes,
    stream: bool,
    path: &str,
) -> Response {
    // role_hint lets a cost-aware policy weight P (cache-heavy: a hit skips a
    // whole prefill pass) differently from D (route by load).
    // Each pool is filtered against the breaker independently: a wedged prefill
    // and a wedged decode are different events against different pools, and one
    // open breaker must not remove the other role's healthy workers.
    let p_avail = state
        .breaker
        .filter(snap.list_active(model, DisaggMode::Prefill), |w| {
            w.worker_id.as_str()
        });
    let d_avail = state
        .breaker
        .filter(snap.list_active(model, DisaggMode::Decode), |w| {
            w.worker_id.as_str()
        });
    let p_pick = state.policy.pick(&p_avail, request, Role::Prefill);
    let d_pick = state.policy.pick(&d_avail, request, Role::Decode);
    let p = p_pick.target;
    let d = d_pick.target;
    // One guard for both legs; dropped when the decode body finishes streaming
    // (or on any early error path), balancing the in-flight load refcount.
    let guard = ActiveGuard::start(
        state.policy.clone(),
        vec![
            (p.route_key(), p_pick.blocks),
            (d.route_key(), d_pick.blocks),
        ],
    );

    let proto = match protocol::resolve_pd_protocol(&p.worker, &d.worker) {
        Ok(pr) => pr,
        Err(e) => return json_error(StatusCode::NOT_IMPLEMENTED, &e.to_string()),
    };

    let base: Map<String, Value> = match serde_json::from_slice::<Value>(&raw) {
        Ok(Value::Object(m)) => m,
        Ok(_) => return json_error(StatusCode::BAD_REQUEST, "body must be a JSON object"),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };

    let room = dp::align_room_to_prefill_rank(rand::random::<u64>() >> 1, &p);

    let mut p_body = base.clone();
    let mut d_body = base;
    let shaped = match proto {
        // SGLang: both legs carry the SAME top-level bootstrap fields.
        protocol::PdProtocol::SglangBootstrap => {
            protocol::annotate_sglang(&mut p_body, &p.worker, room)
                .and_then(|_| protocol::annotate_sglang(&mut d_body, &p.worker, room))
        }
        // vLLM Mooncake: ASYMMETRIC — prefill runs prefill+1tok & pushes KV; decode
        // pulls it via the prefill's bootstrap and generates the rest.
        protocol::PdProtocol::VllmMooncake => {
            protocol::annotate_vllm_prefill(&mut p_body, room);
            protocol::annotate_vllm_decode(&mut d_body, &p.worker, room)
        }
    };
    if let Err(e) = shaped {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    // Tell the decode worker which prefill DP rank holds its KV.
    if let Some(rank) = p.dp_rank {
        d_body.insert("disagg_prefill_dp_rank".into(), Value::from(rank));
    }

    // Both legs over NATS only when both workers registered for it. The KV
    // transfer is engine-to-engine either way (the bootstrap_room travels in
    // the bodies), so the delivery channel is all that changes.
    if let Some(nats) = state.nats.clone() {
        if p.worker.request_transport == "nats" && d.worker.request_transport == "nats" {
            // Either leg being at its backlog limit refuses the whole request:
            // dispatching half a PD pair would leave the other worker holding a
            // bootstrap_room nobody completes.
            if !(nats.admit(&p.worker.worker_id).await && nats.admit(&d.worker.worker_id).await) {
                drop(guard);
                return Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("Retry-After", "1")
                    .body(Body::from(
                        r#"{"error":"PD worker request backlog over limit"}"#,
                    ))
                    .expect("429 response is valid");
            }
            return dual_nats(state, &nats, &p, &d, path, p_body, d_body, stream, guard).await;
        }
    }

    let p_url = format!("{}{}", p.worker.url, path);
    let d_url = format!("{}{}", d.worker.url, path);

    if stream {
        stream_dual(state, &p, &d, p_url, d_url, p_body, d_body, guard).await
    } else {
        unary_dual(state, &p, &d, p_url, d_url, p_body, d_body, guard).await
    }
}

/// Streaming: fire prefill in the background, stream decode back.
#[allow(clippy::too_many_arguments)]
async fn stream_dual(
    state: &AppState,
    p: &RouteTarget,
    d: &RouteTarget,
    p_url: String,
    d_url: String,
    p_body: Map<String, Value>,
    d_body: Map<String, Value>,
    guard: ActiveGuard,
) -> Response {
    spawn_prefill_drain(
        state.http.clone(),
        state.breaker.clone(),
        p.worker.worker_id.clone(),
        p_url,
        p_body,
        p.dp_rank,
    );

    match open_decode(state, d, &d_url, &d_body).await {
        Ok(resp) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            // guard drops when the decode stream ends -> on_request_finished.
            .body(Body::from_stream(GuardedStream::new(
                resp.bytes_stream(),
                guard,
            )))
            .expect("stream response is valid"),
        Err(msg) => json_error(StatusCode::BAD_GATEWAY, &msg),
    }
}

/// Non-streaming: POST both legs concurrently, return the decode JSON.
#[allow(clippy::too_many_arguments)]
async fn unary_dual(
    state: &AppState,
    p: &RouteTarget,
    d: &RouteTarget,
    p_url: String,
    d_url: String,
    p_body: Map<String, Value>,
    d_body: Map<String, Value>,
    guard: ActiveGuard,
) -> Response {
    // Held until both legs finish (dropped at fn end) -> on_request_finished.
    let _guard = guard;
    let p_fut = post_leg(state, &p_url, p_body, p.dp_rank);
    let d_fut = post_leg(state, &d_url, d_body, d.dp_rank);
    let (p_res, d_res) = tokio::join!(p_fut, d_fut);

    // Prefill: drain + log; its output is discarded (KV goes engine→engine).
    match p_res {
        Ok(resp) => {
            let st = resp.status();
            let _ = resp.bytes().await;
            if st.is_client_error() || st.is_server_error() {
                tracing::warn!(
                    "prefill {} returned {} (decode may hang)",
                    p_url,
                    st.as_u16()
                );
            }
            if is_worker_fault(st.as_u16()) {
                state.breaker.record_failure(&p.worker.worker_id);
            } else if st.is_success() {
                state.breaker.record_success(&p.worker.worker_id);
            } else {
                state.breaker.record_neutral(&p.worker.worker_id);
            }
        }
        Err(e) => {
            tracing::warn!("prefill {} failed: {e}", p_url);
            state.breaker.record_failure(&p.worker.worker_id);
        }
    }

    match d_res {
        Ok(resp) => {
            let st = resp.status();
            if is_worker_fault(st.as_u16()) {
                state.breaker.record_failure(&d.worker.worker_id);
            } else if st.is_success() {
                state.breaker.record_success(&d.worker.worker_id);
            } else {
                state.breaker.record_neutral(&d.worker.worker_id);
            }
            let ct = content_type(&resp);
            match resp.bytes().await {
                Ok(bytes) => Response::builder()
                    .status(st)
                    .header(header::CONTENT_TYPE, ct)
                    .body(Body::from(bytes))
                    .expect("unary response is valid"),
                Err(e) => json_error(
                    StatusCode::BAD_GATEWAY,
                    &format!("decode {} read failed: {e}", d.worker.worker_id),
                ),
            }
        }
        Err(e) => {
            state.breaker.record_failure(&d.worker.worker_id);
            json_error(
                StatusCode::BAD_GATEWAY,
                &format!("decode {} unreachable: {e}", d.worker.worker_id),
            )
        }
    }
}

/// Both legs over NATS. Prefill is published and drained detached; decode's
/// reply is what reaches the client.
#[allow(clippy::too_many_arguments)]
async fn dual_nats(
    state: &AppState,
    nats: &Arc<NatsRequestClient>,
    p: &RouteTarget,
    d: &RouteTarget,
    path: &str,
    p_body: Map<String, Value>,
    d_body: Map<String, Value>,
    stream: bool,
    guard: ActiveGuard,
) -> Response {
    // Prefill is never streamed: its output is discarded, only its effect on
    // the KV plane matters.
    let p_payload = leg_payload(path, false, p.dp_rank, p_body);
    let d_payload = leg_payload(path, stream, d.dp_rank, d_body);

    spawn_prefill_drain_nats(
        nats.clone(),
        state.breaker.clone(),
        p.worker.worker_id.clone(),
        p_payload,
    );

    let wid = d.worker.worker_id.clone();
    let mut reply = match nats.dispatch(&wid, &d_payload).await {
        Ok(r) => r,
        Err(e) => {
            state.breaker.record_failure(&wid);
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("decode {wid} unreachable over nats: {e}"),
            );
        }
    };

    if !stream {
        let mut buf: Vec<u8> = Vec::new();
        let mut status = StatusCode::OK;
        let mut done_seen = false;
        loop {
            match reply.next().await {
                Some(Frame::Data(b)) => buf.extend_from_slice(&b),
                Some(Frame::Done { status: s }) => {
                    status = StatusCode::from_u16(s).unwrap_or(StatusCode::OK);
                    done_seen = true;
                    break;
                }
                Some(Frame::Error { status: s, message }) => {
                    // 504 on a timeout, 502 for a worker-side failure.
                    let code = s
                        .and_then(|c| StatusCode::from_u16(c).ok())
                        .unwrap_or(StatusCode::BAD_GATEWAY);
                    score_leg(&state.breaker, &wid, code.as_u16());
                    tracing::warn!(
                        "decode (nats) {wid} failed: {}",
                        truncate_chars(&message, 200)
                    );
                    return json_error(code, &format!("decode {wid} nats failed"));
                }
                None => break,
            }
        }
        if !done_seen {
            // The worker stopped talking without finishing. Scoring this as a
            // 200 would record the exact profile the breaker exists to catch --
            // a worker that accepts work and then goes quiet -- as health.
            state.breaker.record_failure(&wid);
            drop(guard);
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("decode {wid} closed the nats reply without finishing"),
            );
        }
        score_leg(&state.breaker, &wid, status.as_u16());
        drop(guard);
        return Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(buf))
            .expect("unary response is valid");
    }

    // Streaming commits at hand-off, as the HTTP path does. Success is not
    // recorded yet: an accepted request says nothing about whether this worker
    // produces anything, and the failure profile worth catching is exactly the
    // one that accepts and then goes quiet. It is recorded on the first byte.
    let breaker = state.breaker.clone();
    let body = futures::stream::unfold(
        (Some(reply), breaker, wid, false),
        |(reply, breaker, wid, served)| async move {
            let mut r = reply?;
            match r.next().await {
                Some(Frame::Data(b)) => {
                    if !served && !b.is_empty() {
                        // Bytes are flowing, so this worker is doing the work.
                        breaker.record_success(&wid);
                    }
                    let served = served || !b.is_empty();
                    Some((
                        Ok::<Bytes, std::io::Error>(b),
                        (Some(r), breaker, wid, served),
                    ))
                }
                Some(Frame::Error { message, .. }) => {
                    tracing::warn!(
                        "decode (nats) {wid} stream failed: {}",
                        truncate_chars(&message, 200)
                    );
                    if !served {
                        breaker.record_failure(&wid);
                    }
                    let chunk = Bytes::from(format!(
                        "data: {{\"error\":\"decode {wid} nats stream failed\"}}\n\n"
                    ));
                    Some((Ok(chunk), (None, breaker, wid, served)))
                }
                Some(Frame::Done { .. }) | None => None,
            }
        },
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(crate::proxy::guarded(body, guard)))
        .expect("stream response is valid")
}

/// The request envelope the worker side expects, shared with the mixed path.
fn leg_payload(
    path: &str,
    stream: bool,
    dp_rank: Option<i64>,
    body: Map<String, Value>,
) -> Vec<u8> {
    let headers = match dp_rank {
        Some(r) => {
            let mut m = Map::new();
            m.insert(dp::DP_RANK_HEADER.to_string(), Value::from(r.to_string()));
            Value::Object(m)
        }
        None => Value::Null,
    };
    serde_json::to_vec(&serde_json::json!({
        "path": path,
        "stream": stream,
        "headers": headers,
        "body": Value::Object(body),
    }))
    .expect("the envelope is serialisable")
}

/// Detached prefill over NATS: must run to completion, because the prefill
/// engine needs the whole request to register the bootstrap_room and push KV to
/// decode. Spawned rather than awaited, so a client disconnect cannot cancel it.
fn spawn_prefill_drain_nats(
    nats: Arc<NatsRequestClient>,
    breaker: Arc<CircuitBreaker>,
    worker_id: String,
    payload: Vec<u8>,
) {
    tokio::spawn(async move {
        let mut reply = match nats.dispatch(&worker_id, &payload).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    "prefill (nats) {worker_id} failed: {e} (decode may hang on KVPoll)"
                );
                breaker.record_failure(&worker_id);
                return;
            }
        };
        loop {
            match reply.next().await {
                Some(Frame::Data(_)) => {}
                Some(Frame::Done { status }) => {
                    // Detached, so this never reaches the client -- but a
                    // prefill that fails still leaves decode hanging on KVPoll,
                    // which is exactly the failure worth remembering.
                    if !StatusCode::from_u16(status).is_ok_and(|s| s.is_success()) {
                        tracing::warn!(
                            "prefill (nats) {worker_id} returned {status} (decode may hang on KVPoll)"
                        );
                    }
                    score_leg(&breaker, &worker_id, status);
                    return;
                }
                Some(Frame::Error { message, .. }) => {
                    tracing::warn!(
                        "prefill (nats) {worker_id} failed: {} (decode may hang on KVPoll)",
                        truncate_chars(&message, 200)
                    );
                    breaker.record_failure(&worker_id);
                    return;
                }
                None => return,
            }
        }
    });
}

/// One leg's outcome against the worker that produced it. The two legs are
/// different workers with independent health, so neither is scored for the
/// other's failure. A 4xx is the request's fault, not the worker's.
///
/// Used for the NATS legs too, where the status arrives in the `done` frame:
/// a `done` says the request finished, not that it succeeded, since the worker
/// proxies whatever its engine returned. Scoring on the frame kind alone would
/// read every failed prefill as health.
fn score_leg(breaker: &Arc<CircuitBreaker>, worker_id: &str, status: u16) {
    if is_worker_fault(status) {
        breaker.record_failure(worker_id);
    } else if (200..400).contains(&status) {
        breaker.record_success(worker_id);
    } else {
        breaker.record_neutral(worker_id);
    }
}

/// Detached prefill POST: runs to completion so the KV transfer isn't aborted.
/// Never awaited by the request path, so a client disconnect can't cancel it.
fn spawn_prefill_drain(
    http: reqwest::Client,
    breaker: Arc<CircuitBreaker>,
    worker_id: String,
    url: String,
    body: Map<String, Value>,
    dp_rank: Option<i64>,
) {
    tokio::spawn(async move {
        let mut req = http.post(&url).json(&Value::Object(body));
        if let Some(r) = dp_rank {
            req = req.header(dp::DP_RANK_HEADER, r.to_string());
        }
        match req.send().await {
            Ok(resp) => {
                let st = resp.status();
                let _ = resp.bytes().await; // drain to keep the connection open
                if st.is_client_error() || st.is_server_error() {
                    tracing::warn!(
                        "prefill {url} returned {} (decode may hang on KVPoll)",
                        st.as_u16()
                    );
                }
                // This leg is detached, so its outcome never reaches the client
                // — but a prefill that 5xx's still leaves decode hanging on
                // KVPoll, which is exactly the failure worth remembering.
                if is_worker_fault(st.as_u16()) {
                    breaker.record_failure(&worker_id);
                } else if st.is_success() {
                    breaker.record_success(&worker_id);
                } else {
                    breaker.record_neutral(&worker_id);
                }
            }
            Err(e) => {
                tracing::warn!("prefill {url} failed: {e} (decode may hang on KVPoll)");
                breaker.record_failure(&worker_id);
            }
        }
    });
}

/// POST the decode leg, retrying on pre-flight transport errors (the engine
/// hasn't seen the body yet, so re-sending the same bootstrap_room is safe).
async fn open_decode(
    state: &AppState,
    d: &RouteTarget,
    url: &str,
    body: &Map<String, Value>,
) -> Result<reqwest::Response, String> {
    let mut backoff = Duration::from_millis(50);
    for attempt in 0..=DECODE_OPEN_RETRIES {
        match post_leg(state, url, body.clone(), d.dp_rank).await {
            Ok(resp) => {
                let st = resp.status();
                if st.is_client_error() || st.is_server_error() {
                    if is_worker_fault(st.as_u16()) {
                        state.breaker.record_failure(&d.worker.worker_id);
                    } else {
                        state.breaker.record_neutral(&d.worker.worker_id);
                    }
                    let txt = resp.text().await.unwrap_or_default();
                    return Err(format!(
                        "decode {} error {}: {}",
                        d.worker.worker_id,
                        st.as_u16(),
                        truncate_chars(&txt, 300)
                    ));
                }
                state.breaker.record_success(&d.worker.worker_id);
                return Ok(resp);
            }
            Err(e) if attempt < DECODE_OPEN_RETRIES => {
                tracing::info!(
                    "decode open retry {}/{DECODE_OPEN_RETRIES} for {url}: {e}",
                    attempt + 1
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            }
            Err(e) => {
                // Exhausted the in-request retries: this worker is not merely
                // slow to accept a connection.
                state.breaker.record_failure(&d.worker.worker_id);
                return Err(format!("decode {} unreachable: {e}", d.worker.worker_id));
            }
        }
    }
    unreachable!("loop returns on the final attempt")
}

fn post_leg(
    state: &AppState,
    url: &str,
    body: Map<String, Value>,
    dp_rank: Option<i64>,
) -> impl std::future::Future<Output = reqwest::Result<reqwest::Response>> {
    // `.json()` sets content-type: application/json itself.
    let mut req = state.http.post(url).json(&Value::Object(body));
    if let Some(r) = dp_rank {
        req = req.header(dp::DP_RANK_HEADER, r.to_string());
    }
    req.send()
}

fn content_type(resp: &reqwest::Response) -> String {
    resp.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breaker::CircuitBreaker;

    fn breaker() -> Arc<CircuitBreaker> {
        Arc::new(CircuitBreaker::default())
    }

    #[test]
    fn a_leg_that_failed_is_not_scored_as_healthy() {
        // The NATS `done` frame carries the engine's status, so 500 has to be
        // read out of it. Taking the frame's arrival as success is the trap:
        // the prefill leg's reply is discarded, so nothing else would notice a
        // worker that fails every request while decode hangs on KVPoll.
        let b = breaker();
        for _ in 0..3 {
            score_leg(&b, "p1", 500);
        }
        assert_eq!(b.state_of("p1").as_str(), "open");
    }

    #[test]
    fn a_4xx_leg_does_not_accumulate_towards_tripping() {
        // The request is bad, and every worker would answer the same.
        let b = breaker();
        for _ in 0..5 {
            score_leg(&b, "p1", 400);
        }
        assert_eq!(b.state_of("p1").as_str(), "closed");
    }

    #[test]
    fn a_successful_leg_clears_the_failure_count() {
        let b = breaker();
        score_leg(&b, "p1", 500);
        score_leg(&b, "p1", 500);
        score_leg(&b, "p1", 200);
        score_leg(&b, "p1", 500);
        assert_eq!(
            b.state_of("p1").as_str(),
            "closed",
            "three failures have to be consecutive, or the count only ever climbs"
        );
    }

    #[test]
    fn the_legs_are_scored_independently() {
        // A decode that answers cannot vouch for a prefill that did not.
        let b = breaker();
        for _ in 0..3 {
            score_leg(&b, "p1", 500);
            score_leg(&b, "d1", 200);
        }
        assert_eq!(b.state_of("p1").as_str(), "open");
        assert_eq!(b.state_of("d1").as_str(), "closed");
    }
}
