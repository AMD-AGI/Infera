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

use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::Response;
use serde_json::{Map, Value};

use crate::dp;
use crate::handlers::AppState;
use crate::policy::{ActiveGuard, Role};
use crate::pool::{DisaggMode, RouteTarget, Snapshot};
use crate::protocol;
use crate::proxy::GuardedStream;
use crate::util::json_error;

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
    let p_pick = state.policy.pick(
        snap.list_active(model, DisaggMode::Prefill),
        request,
        Role::Prefill,
    );
    let d_pick = state.policy.pick(
        snap.list_active(model, DisaggMode::Decode),
        request,
        Role::Decode,
    );
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
    spawn_prefill_drain(state.http.clone(), p_url, p_body, p.dp_rank);

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
        }
        Err(e) => tracing::warn!("prefill {} failed: {e}", p_url),
    }

    match d_res {
        Ok(resp) => {
            let st = resp.status();
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
        Err(e) => json_error(
            StatusCode::BAD_GATEWAY,
            &format!("decode {} unreachable: {e}", d.worker.worker_id),
        ),
    }
}

/// Detached prefill POST: runs to completion so the KV transfer isn't aborted.
/// Never awaited by the request path, so a client disconnect can't cancel it.
fn spawn_prefill_drain(
    http: reqwest::Client,
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
            }
            Err(e) => tracing::warn!("prefill {url} failed: {e} (decode may hang on KVPoll)"),
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
                    let txt = resp.text().await.unwrap_or_default();
                    return Err(format!(
                        "decode {} error {}: {}",
                        d.worker.worker_id,
                        st.as_u16(),
                        &txt[..txt.len().min(300)]
                    ));
                }
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
            Err(e) => return Err(format!("decode {} unreachable: {e}", d.worker.worker_id)),
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
