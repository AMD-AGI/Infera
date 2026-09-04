///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Router functional tests: drive real HTTP requests through the assembled
//! axum app against mock upstream workers, so one case exercises the whole
//! path (handlers → proxy/disagg → policy → pool → protocol/dp) at once.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use infera_router::block_hasher::BlockHasher;
use infera_router::breaker::CircuitBreaker;
use infera_router::handlers::{app, AppState};
use infera_router::kv_event::KvEventClient;
use infera_router::policy::{KvEventAwarePolicy, RoundRobin};
use infera_router::pool::{Snapshot, Worker};
use infera_router::proxy;

// ---------------------------------------------------------------------------
// Mock upstream worker
// ---------------------------------------------------------------------------

struct Hit {
    body: Value,
    dp_rank: Option<String>,
    path: String,
}

struct MockState {
    status: u16,
    sse: bool,
    reply: Value,
    hits: Mutex<Vec<Hit>>,
}

impl MockState {
    fn hit_count(&self) -> usize {
        self.hits.lock().unwrap().len()
    }
}

async fn mock_handle(
    State(s): State<Arc<MockState>>,
    uri: Uri,
    headers: HeaderMap,
    raw: Bytes,
) -> Response {
    let body: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    let dp_rank = headers
        .get(infera_router::dp::DP_RANK_HEADER)
        .and_then(|h| h.to_str().ok())
        .map(str::to_string);
    let path = uri.path().to_string();
    s.hits.lock().unwrap().push(Hit {
        body,
        dp_rank,
        path,
    });

    if s.status != 200 {
        return (StatusCode::from_u16(s.status).unwrap(), "upstream error").into_response();
    }
    if s.sse {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        return Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from(sse))
            .unwrap();
    }
    (StatusCode::OK, Json(s.reply.clone())).into_response()
}

/// Spawn a mock worker on a random port. Returns (base_url, shared state).
async fn spawn_mock(status: u16, sse: bool, reply: Value) -> (String, Arc<MockState>) {
    let state = Arc::new(MockState {
        status,
        sse,
        reply,
        hits: Mutex::new(Vec::new()),
    });
    let router = Router::new()
        .route("/v1/chat/completions", post(mock_handle))
        .route("/v1/completions", post(mock_handle))
        .route("/v1/responses", post(mock_handle))
        // Stand in for a real engine, which caps a prompt by context length
        // rather than by request bytes.
        .layer(DefaultBodyLimit::disable())
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://127.0.0.1:{port}"), state)
}

// ---------------------------------------------------------------------------
// Router under test
// ---------------------------------------------------------------------------

fn worker(spec: Value) -> Arc<Worker> {
    Arc::new(serde_json::from_value(spec).expect("worker json"))
}

fn make_state(workers: Vec<Arc<Worker>>, retries: usize) -> AppState {
    AppState {
        pool: Arc::new(ArcSwap::from_pointee(Snapshot::build(workers))),
        policy: Arc::new(RoundRobin::new()),
        http: proxy::build_upstream_client().unwrap(),
        started: Instant::now(),
        retries,
        breaker: Arc::new(CircuitBreaker::default()),
        nats: None,
    }
}

async fn spawn_router(state: AppState) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });
    format!("http://127.0.0.1:{port}")
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

// ---------------------------------------------------------------------------
// Mixed (non-PD) dispatch
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_unary_ok() {
    let (url, mock) = spawn_mock(200, false, json!({"answer": 42})).await;
    let state = make_state(
        vec![worker(json!({
            "worker_id": "w1", "url": url, "model_name": "m", "disagg_mode": "mixed"
        }))],
        0,
    );
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m", "stream": false}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.json::<Value>().await.unwrap()["answer"], 42);
    assert_eq!(mock.hit_count(), 1);
}

/// The Codex CLI/SDK speaks the OpenAI Responses API, not chat completions, so
/// an unregistered `/v1/responses` is a 404 raised by the route table before
/// any policy runs — a whole client family locked out with no worker-side trace.
/// The dispatch chain is path-generic, so registering the route is all it takes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_responses_forwards_verbatim() {
    let (url, mock) = spawn_mock(200, false, json!({"object": "response"})).await;
    let state = make_state(
        vec![worker(json!({
            "worker_id": "w1", "url": url, "model_name": "m", "disagg_mode": "mixed"
        }))],
        0,
    );
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/responses"))
        .json(&json!({"model": "m", "input": "1+1=?", "store": false}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.json::<Value>().await.unwrap()["object"], "response");

    let hit = &mock.hits.lock().unwrap()[0];
    // Forwarded to the same path, not rewritten onto the chat endpoint.
    assert_eq!(hit.path, "/v1/responses");
    // A Responses body carries `input`, never `messages`; nothing may rewrite it.
    assert_eq!(hit.body["input"], "1+1=?");
    assert!(hit.body.get("messages").is_none());
}

/// A long-context prompt is a normal request, not an oversized one: axum's
/// default 2 MiB body cap used to 413 it before it reached a worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_forwards_body_over_axum_default_limit() {
    let (url, mock) = spawn_mock(200, false, json!({"answer": 42})).await;
    let state = make_state(
        vec![worker(json!({
            "worker_id": "w1", "url": url, "model_name": "m", "disagg_mode": "mixed"
        }))],
        0,
    );
    let router = spawn_router(state).await;
    let prompt = "x".repeat((2 << 20) + 4096);

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m", "prompt": prompt}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let hits = mock.hits.lock().unwrap();
    assert_eq!(hits[0].body["prompt"].as_str().unwrap().len(), prompt.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_round_robin_spreads_load() {
    let (url_a, a) = spawn_mock(200, false, json!({"w": "a"})).await;
    let (url_b, b) = spawn_mock(200, false, json!({"w": "b"})).await;
    let state = make_state(
        vec![
            worker(
                json!({"worker_id": "a", "url": url_a, "model_name": "m", "disagg_mode": "mixed"}),
            ),
            worker(
                json!({"worker_id": "b", "url": url_b, "model_name": "m", "disagg_mode": "mixed"}),
            ),
        ],
        0,
    );
    let router = spawn_router(state).await;

    for _ in 0..4 {
        let r = client()
            .post(format!("{router}/v1/chat/completions"))
            .json(&json!({"model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }
    // 4 requests, round-robin over 2 workers → 2 each.
    assert_eq!(a.hit_count(), 2);
    assert_eq!(b.hit_count(), 2);
}

// ---------------------------------------------------------------------------
// Multimodal image-affinity routing (engine-agnostic: same OpenAI vision body
// that both sglang and vLLM receive flows the full path handlers → parse →
// extract_image_keys → KvEventAwarePolicy affinity → upstream).
// ---------------------------------------------------------------------------

fn make_kv_state(workers: Vec<Arc<Worker>>, retries: usize) -> AppState {
    AppState {
        pool: Arc::new(ArcSwap::from_pointee(Snapshot::build(workers))),
        policy: Arc::new(KvEventAwarePolicy::new(
            Arc::new(KvEventClient::new()),
            BlockHasher::disabled(),
            20.0,
            None,
            None,
        )),
        http: proxy::build_upstream_client().unwrap(),
        started: Instant::now(),
        retries,
        breaker: Arc::new(CircuitBreaker::default()),
        nats: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mm_affinity_colocates_repeat_image() {
    let (url_a, a) = spawn_mock(200, false, json!({"w": "a"})).await;
    let (url_b, b) = spawn_mock(200, false, json!({"w": "b"})).await;
    let state = make_kv_state(
        vec![
            worker(json!({"worker_id": "a", "url": url_a, "model_name": "m",
                          "disagg_mode": "mixed", "kv_block_size": 16})),
            worker(json!({"worker_id": "b", "url": url_b, "model_name": "m",
                          "disagg_mode": "mixed", "kv_block_size": 16})),
        ],
        0,
    );
    let router = spawn_router(state).await;

    // Standard OpenAI vision request — the identical body an OpenAI client sends
    // to either sglang or vLLM. The router keys affinity off the image URL.
    let img = json!({"model": "m", "stream": false, "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://cdn.example/cat.png"}}
        ]
    }]});

    for _ in 0..5 {
        let r = client()
            .post(format!("{router}/v1/chat/completions"))
            .json(&img)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }

    // All five identical-image requests land on ONE worker (its warm vision
    // cache), instead of round-robining across both.
    let (ha, hb) = (a.hit_count(), b.hit_count());
    assert_eq!(ha + hb, 5, "all requests served");
    assert!(
        ha == 5 || hb == 5,
        "image affinity co-locates repeats: a={ha} b={hb}"
    );

    // The image survived the hop to the upstream (routing didn't strip content).
    let winner = if ha == 5 { &a } else { &b };
    let body = &winner.hits.lock().unwrap()[0].body;
    assert_eq!(
        body["messages"][0]["content"][1]["image_url"]["url"],
        "https://cdn.example/cat.png"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_failover_to_healthy_worker() {
    // First candidate 500s; with retries=1 the router must fail over to the
    // second and return its 200. RoundRobin picks index 0 first (fresh counter).
    let (url_bad, bad) = spawn_mock(500, false, json!(null)).await;
    let (url_ok, ok) = spawn_mock(200, false, json!({"ok": true})).await;
    let state = make_state(
        vec![
            worker(
                json!({"worker_id": "bad", "url": url_bad, "model_name": "m", "disagg_mode": "mixed"}),
            ),
            worker(
                json!({"worker_id": "ok", "url": url_ok, "model_name": "m", "disagg_mode": "mixed"}),
            ),
        ],
        1,
    );
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.json::<Value>().await.unwrap()["ok"], true);
    assert_eq!(bad.hit_count(), 1);
    assert_eq!(ok.hit_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn breaker_stops_reselecting_a_dead_worker() {
    // The regression behind issue #82, end to end through the real router
    // rather than against the breaker in isolation.
    //
    // Failover already made every one of these ten requests succeed, so a test
    // that only checked status codes passed before the fix and after it. What
    // was broken is the *cost*: `tried` is per-request, so RoundRobin kept
    // offering the dead worker its turn -- 5 of 10 requests paid a wasted
    // upstream round trip. The assertion that matters is bad.hit_count(), which
    // is 5 without the breaker and 3 with it.
    let (url_bad, bad) = spawn_mock(500, false, json!(null)).await;
    let (url_ok, ok) = spawn_mock(200, false, json!({"ok": true})).await;
    let state = make_state(
        vec![
            worker(
                json!({"worker_id": "bad", "url": url_bad, "model_name": "m", "disagg_mode": "mixed"}),
            ),
            worker(
                json!({"worker_id": "ok", "url": url_ok, "model_name": "m", "disagg_mode": "mixed"}),
            ),
        ],
        1,
    );
    let router = spawn_router(state).await;

    for _ in 0..10 {
        let resp = client()
            .post(format!("{router}/v1/chat/completions"))
            .json(&json!({"model": "m"}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            200,
            "failover must still serve every request"
        );
    }

    // Default threshold is 3. RoundRobin offers `bad` on every other request,
    // so it takes 3 of its turns to trip; after that it is out of rotation and
    // the 5s cooldown does not elapse within the test.
    assert_eq!(
        bad.hit_count(),
        3,
        "dead worker must stop being re-picked after the threshold (was 5 before the fix)"
    );
    assert_eq!(ok.hit_count(), 10, "healthy worker still serves everything");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn breaker_ignores_client_errors() {
    // A 400 comes from the request, not the worker: it would be returned by
    // every worker in the fleet, so counting it would circuit-break all of
    // them. Ten bad requests must leave the worker in rotation.
    let (url, w) = spawn_mock(400, false, json!(null)).await;
    let state = make_state(
        vec![worker(
            json!({"worker_id": "w1", "url": url, "model_name": "m", "disagg_mode": "mixed"}),
        )],
        0,
    );
    let router = spawn_router(state).await;

    for _ in 0..10 {
        let _ = client()
            .post(format!("{router}/v1/chat/completions"))
            .json(&json!({"model": "m"}))
            .send()
            .await
            .unwrap();
    }
    assert_eq!(
        w.hit_count(),
        10,
        "4xx must not take a healthy worker out of rotation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_streaming_relays_sse() {
    let (url, _mock) = spawn_mock(200, true, json!(null)).await;
    let state = make_state(
        vec![worker(json!({
            "worker_id": "w1", "url": url, "model_name": "m", "disagg_mode": "mixed"
        }))],
        0,
    );
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m", "stream": true}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "text/event-stream"
    );
    let text = resp.text().await.unwrap();
    assert!(text.contains("data: "), "expected SSE frames, got {text:?}");
    assert!(text.contains("[DONE]"), "expected [DONE], got {text:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_model_is_503() {
    let (url, _mock) = spawn_mock(200, false, json!(null)).await;
    let state = make_state(
        vec![worker(json!({
            "worker_id": "w1", "url": url, "model_name": "known", "disagg_mode": "mixed"
        }))],
        0,
    );
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "unknown"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 503);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_json_is_400() {
    let state = make_state(vec![], 0);
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .header(CONTENT_TYPE, "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
}

// ---------------------------------------------------------------------------
// PD (disaggregated) dispatch — exercises protocol + dp submodules
// ---------------------------------------------------------------------------

fn prefill(url: &str, dp_size: Option<i64>) -> Arc<Worker> {
    let mut spec = json!({
        "worker_id": "p", "url": url, "model_name": "m", "disagg_mode": "prefill",
        "disagg_meta": {"protocol": "sglang-bootstrap", "params": {"bootstrap_addr": "10.0.0.1:9000"}}
    });
    if let Some(sz) = dp_size {
        spec["dp_size"] = json!(sz);
    }
    worker(spec)
}

fn decode(url: &str) -> Arc<Worker> {
    worker(json!({
        "worker_id": "d", "url": url, "model_name": "m", "disagg_mode": "decode",
        "disagg_meta": {"protocol": "sglang-bootstrap"}
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd_unary_injects_matching_bootstrap_room() {
    let (p_url, p) = spawn_mock(200, false, json!({"who": "prefill"})).await;
    let (d_url, d) = spawn_mock(200, false, json!({"who": "decode"})).await;
    let state = make_state(vec![prefill(&p_url, None), decode(&d_url)], 0);
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m", "stream": false}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    // The client sees the DECODE leg's body, not prefill's.
    assert_eq!(resp.json::<Value>().await.unwrap()["who"], "decode");

    let p_hit = &p.hits.lock().unwrap()[0].body;
    let d_hit = &d.hits.lock().unwrap()[0].body;
    // Bootstrap fields come from the prefill worker's advertised addr.
    assert_eq!(d_hit["bootstrap_host"], "10.0.0.1");
    assert_eq!(d_hit["bootstrap_port"], 9000);
    // Both legs must carry the SAME room or the KV handoff can't rendezvous.
    assert!(p_hit["bootstrap_room"].is_number());
    assert_eq!(p_hit["bootstrap_room"], d_hit["bootstrap_room"]);
}

/// PD dual-dispatch is path-generic: a Responses request must reach both legs on
/// `/v1/responses` with the same bootstrap trio as a chat request gets. (The
/// engine side needs a patch to stop dropping those fields — see
/// `deploy/docker/patches/sglang_disagg/patch_responses_pd_bootstrap.py` — but
/// that is not the router's contract to keep.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd_responses_injects_bootstrap_on_both_legs() {
    let (p_url, p) = spawn_mock(200, false, json!({"who": "prefill"})).await;
    let (d_url, d) = spawn_mock(200, false, json!({"who": "decode"})).await;
    let state = make_state(vec![prefill(&p_url, None), decode(&d_url)], 0);
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/responses"))
        .json(&json!({"model": "m", "input": "1+1=?", "store": false}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.json::<Value>().await.unwrap()["who"], "decode");

    let p_hit = &p.hits.lock().unwrap()[0];
    let d_hit = &d.hits.lock().unwrap()[0];
    assert_eq!(p_hit.path, "/v1/responses");
    assert_eq!(d_hit.path, "/v1/responses");
    assert_eq!(d_hit.body["bootstrap_host"], "10.0.0.1");
    assert_eq!(d_hit.body["bootstrap_port"], 9000);
    assert!(p_hit.body["bootstrap_room"].is_number());
    assert_eq!(p_hit.body["bootstrap_room"], d_hit.body["bootstrap_room"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd_dp_multiplexed_prefill_pins_rank() {
    // A dp_size=2 prefill worker (dp_rank unset) fans out to per-rank targets;
    // RoundRobin picks rank 0 first, so the room aligns to rank 0, the prefill
    // leg carries the DP-rank header, and decode is told which rank holds its KV.
    let (p_url, p) = spawn_mock(200, false, json!({"who": "prefill"})).await;
    let (d_url, d) = spawn_mock(200, false, json!({"who": "decode"})).await;
    let state = make_state(vec![prefill(&p_url, Some(2)), decode(&d_url)], 0);
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m", "stream": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let p_hits = p.hits.lock().unwrap();
    let d_hits = d.hits.lock().unwrap();
    assert_eq!(p_hits[0].dp_rank.as_deref(), Some("0"));
    // room aligned so room % dp_size == rank(0).
    let room = p_hits[0].body["bootstrap_room"].as_u64().unwrap();
    assert_eq!(room % 2, 0);
    // Decode is told the prefill DP rank holding its KV.
    assert_eq!(d_hits[0].body["disagg_prefill_dp_rank"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd_streaming_relays_decode_and_fires_prefill() {
    let (p_url, p) = spawn_mock(200, false, json!(null)).await;
    let (d_url, _d) = spawn_mock(200, true, json!(null)).await;
    let state = make_state(vec![prefill(&p_url, None), decode(&d_url)], 0);
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m", "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("[DONE]"), "expected decode SSE, got {text:?}");

    // Prefill is fired on a detached task; give it a beat to land.
    let deadline = Instant::now() + Duration::from_secs(2);
    while p.hit_count() == 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(
        p.hit_count(),
        1,
        "prefill leg must run even while streaming"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd_protocol_mismatch_is_501() {
    let (p_url, _p) = spawn_mock(200, false, json!(null)).await;
    let (d_url, _d) = spawn_mock(200, false, json!(null)).await;
    // Prefill advertises no sglang-bootstrap protocol → unsupported connector.
    let p = worker(json!({
        "worker_id": "p", "url": p_url, "model_name": "m", "disagg_mode": "prefill",
        "disagg_meta": {"protocol": "mooncake"}
    }));
    let state = make_state(vec![p, decode(&d_url)], 0);
    let router = spawn_router(state).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&json!({"model": "m"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 501);
}

// ---------------------------------------------------------------------------
// Introspection endpoints
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn introspection_endpoints_report_fleet() {
    let (url, _mock) = spawn_mock(200, false, json!(null)).await;
    let state = make_state(
        vec![
            worker(
                json!({"worker_id": "w1", "url": url, "model_name": "m", "disagg_mode": "mixed"}),
            ),
            worker(
                json!({"worker_id": "w2", "url": "http://x", "model_name": "m", "disagg_mode": "mixed", "status": "draining"}),
            ),
        ],
        0,
    );
    let router = spawn_router(state).await;
    let c = client();

    let health: Value = c
        .get(format!("{router}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["active_workers"], 1); // w2 is draining

    let models: Value = c
        .get(format!("{router}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["m"]);

    let workers: Value = c
        .get(format!("{router}/v1/workers"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(workers["workers"].as_array().unwrap().len(), 2);

    let metrics = c
        .get(format!("{router}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        metrics.contains("infera_router_active_workers 1"),
        "got {metrics:?}"
    );
}

// ---------------------------------------------------------------------------
// kv-aware self-heal: the router asks a worker to flush a chain that never
// anchored. Everything upstream of the POST is covered by in-crate unit tests;
// this is the hop they cannot reach -- that the request actually leaves the
// router, over HTTP, at the endpoint the engine serves.
// ---------------------------------------------------------------------------

/// Spawn a worker that records which cache-flush endpoints it was asked for.
async fn spawn_flush_mock() -> (String, Arc<Mutex<Vec<String>>>) {
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    let router = Router::new()
        .route(
            "/flush_cache",
            post(move || {
                let s = s.clone();
                async move {
                    s.lock().unwrap().push("/flush_cache".to_string());
                    (StatusCode::OK, "Cache flushed.")
                }
            }),
        )
        .layer(DefaultBodyLimit::disable());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://127.0.0.1:{port}"), seen)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_flush_request_reaches_the_worker_over_http() {
    let (url, seen) = spawn_flush_mock().await;
    let flush = infera_router::kv_selfheal::spawn(proxy::build_upstream_client().unwrap());

    let w = worker(json!({
        "worker_id": "a", "url": url, "model_name": "m", "engine": "sglang",
    }));
    flush.request(&w);

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && seen.lock().unwrap().is_empty() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        ["/flush_cache"],
        "the engine's own cache-flush endpoint is what re-emits the rooted event"
    );

    // The detector re-arms on every unanchored batch -- it has to, or one
    // refusal would retire the repair for good -- so holding the POST down to
    // one is the actor's job: a worker with a flush in flight, and then inside
    // its cooldown, is asked no again. Repeating it would clear a cache that
    // is in the middle of being rebuilt.
    flush.request(&w);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_reachable_only_over_nats_is_not_flushed_over_http() {
    let (url, seen) = spawn_flush_mock().await;
    let flush = infera_router::kv_selfheal::spawn(proxy::build_upstream_client().unwrap());

    // Its `url` may not be routable from here at all, so an HTTP POST would
    // fail on every retry while looking like an unreachable worker.
    flush.request(&worker(json!({
        "worker_id": "a", "url": url, "model_name": "m", "engine": "sglang",
        "request_transport": "nats",
    })));

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(seen.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// POST /v1/responses/input_tokens (LiteLLM CountTokens)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn responses_input_tokens_counts_pre_tokenized_ids() {
    // Integer `prompt` hashes without a tokenizer; the handler still requires
    // Responses `input` so this is the wiring check, not a template render.
    let router = spawn_router(make_kv_state(vec![], 0)).await;
    let r = client()
        .post(format!("{router}/v1/responses/input_tokens"))
        .json(&json!({"model": "m", "input": "hi", "prompt": [1, 2, 3, 4]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["input_tokens"], 4);
    assert_eq!(body["object"], "response.input_tokens");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn responses_input_tokens_refuses_previous_response_id() {
    let router = spawn_router(make_kv_state(vec![], 0)).await;
    let r = client()
        .post(format!("{router}/v1/responses/input_tokens"))
        .json(&json!({
            "model": "m",
            "input": "hi",
            "previous_response_id": "resp_1",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn responses_input_tokens_requires_input() {
    let router = spawn_router(make_kv_state(vec![], 0)).await;
    let r = client()
        .post(format!("{router}/v1/responses/input_tokens"))
        .json(&json!({"model": "m"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn responses_input_tokens_unavailable_without_tokenizer() {
    let router = spawn_router(make_state(vec![], 0)).await;
    let r = client()
        .post(format!("{router}/v1/responses/input_tokens"))
        .json(&json!({"model": "m", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
}
