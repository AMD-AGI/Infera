//! The startup check that kv-aware routing is actually going to work.
//!
//! Every test here is really the same assertion: a router whose render
//! disagrees with the engine's must SAY SO, because nothing else will. That
//! failure produces no error, no dropped request and no unhealthy worker --
//! only a hit rate of zero, which looks exactly like a cold cache.
//!
//! These drive a real HTTP server rather than a mocked client, because half of
//! what the probe has to get right is HTTP-shaped: the 404 fallback to the
//! unprefixed alias, a refusal that must read as "unknown" rather than
//! "diverged", and a worker that never answers at all.

use std::sync::Arc;

use axum::{routing::post, Json, Router};
use infera_router::block_hasher::BlockHasher;
use infera_router::pool::Worker;
use infera_router::render_probe::{probe_worker, Parity, ParityRegistry};
use serde_json::{json, Value};

/// What the fake engine answers with.
#[derive(Clone)]
enum Engine {
    /// Echo back exactly what the router would produce, so the bodies agree.
    Agree,
    /// Agree, then corrupt one token, so exactly one body diverges.
    Corrupt(usize),
    /// Refuse with this status on both paths.
    Refuse(u16),
    /// 404 on `/v1/tokenize`, agree on `/tokenize`.
    OnlyAlias,
}

#[derive(Clone)]
struct Fake {
    engine: Engine,
    hasher: Arc<BlockHasher>,
    paths: Arc<std::sync::Mutex<Vec<String>>>,
}

async fn tokenize(prefixed: bool, f: Fake, body: Value) -> axum::response::Response {
    use axum::response::IntoResponse;
    f.paths.lock().unwrap().push(
        if prefixed {
            "/v1/tokenize"
        } else {
            "/tokenize"
        }
        .to_string(),
    );
    let ids = f.hasher.token_ids_for(&body).unwrap_or_default();
    match f.engine {
        Engine::Refuse(code) => axum::http::StatusCode::from_u16(code)
            .unwrap()
            .into_response(),
        Engine::OnlyAlias if prefixed => axum::http::StatusCode::NOT_FOUND.into_response(),
        Engine::Corrupt(n) => {
            let mut ids = ids;
            if body.get("tools").is_some() && !ids.is_empty() {
                let at = n.min(ids.len() - 1);
                ids[at] = 999_999;
            }
            Json(json!({"tokens": ids, "count": ids.len()})).into_response()
        }
        _ => Json(json!({"tokens": ids, "count": ids.len()})).into_response(),
    }
}

/// A worker backed by an engine that tokenizes with the very hasher under test.
/// Agreement is therefore the *default*, and each test perturbs one thing --
/// which is the right polarity: we are testing that the probe notices trouble,
/// not that two copies of the same code agree.
async fn fake_worker(engine: Engine, hasher: Arc<BlockHasher>) -> (Arc<Worker>, Fake) {
    let f = Fake {
        engine,
        hasher,
        paths: Arc::new(std::sync::Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route(
            "/v1/tokenize",
            post({
                let f = f.clone();
                move |Json(b): Json<Value>| tokenize(true, f.clone(), b)
            }),
        )
        .route(
            "/tokenize",
            post({
                let f = f.clone();
                move |Json(b): Json<Value>| tokenize(false, f.clone(), b)
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let w: Worker = serde_json::from_value(json!({
        "worker_id": "w1",
        "url": format!("http://127.0.0.1:{port}"),
        "model_name": "glm53",
        "engine": "sglang",
    }))
    .unwrap();
    (Arc::new(w), f)
}

fn hasher() -> Arc<BlockHasher> {
    // No tokenizer configured: `token_ids_for` declines every body, which is
    // the "router already knows it cannot reproduce this" state. Tests that
    // need real ids set INFERA_TEST_RENDER_PARITY (see block_hasher.rs).
    Arc::new(BlockHasher::disabled())
}

fn model_hasher() -> Option<Arc<BlockHasher>> {
    let spec = std::env::var("INFERA_TEST_RENDER_PARITY").ok()?;
    let path = spec.split(',').next()?.split_once('=')?.1.to_string();
    Some(Arc::new(BlockHasher::load(&path)))
}

#[tokio::test]
async fn a_router_that_cannot_render_does_not_bother_the_engine() {
    // `token_ids_for` returning None is a KNOWN gap, honestly handled: the
    // request routes on load. The probe exists to find the cases where we are
    // confidently wrong, so this must read as "unknown", not "diverged" -- and
    // must not ask the engine a question whose answer it could not use.
    let h = hasher();
    let (w, f) = fake_worker(Engine::Agree, Arc::clone(&h)).await;
    let (verdict, _) = probe_worker(&h, &reqwest::Client::new(), &w).await;
    assert_eq!(verdict, Parity::Unknown);
    assert!(f.paths.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_unreachable_worker_is_unknown_not_diverged() {
    // This runs from the discovery hook. A probe that treated "engine down" as
    // "engine disagrees" would page the moment a worker restarted.
    let h = hasher();
    let w: Worker = serde_json::from_value(json!({
        "worker_id": "w1",
        // Reserved-for-documentation address: guaranteed not to answer.
        "url": "http://192.0.2.1:9",
        "model_name": "glm53",
        "engine": "sglang",
    }))
    .unwrap();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(300))
        .build()
        .unwrap();
    let (verdict, _) = probe_worker(&h, &client, &Arc::new(w)).await;
    assert_eq!(verdict, Parity::Unknown);
}

#[tokio::test]
async fn agreement_is_confirmed() {
    let Some(h) = model_hasher() else { return };
    let (w, _) = fake_worker(Engine::Agree, Arc::clone(&h)).await;
    let (verdict, detail) = probe_worker(&h, &reqwest::Client::new(), &w).await;
    assert_eq!(verdict, Parity::Confirmed, "{detail}");
}

#[tokio::test]
async fn one_bad_body_among_good_ones_still_fails() {
    // Divergence is usually conditional -- plain chat renders fine, a body with
    // tools does not. Passing on the majority would miss exactly the agentic
    // traffic kv-aware is bought for. The message must also name the token
    // index: an operator's first question is whether we diverge in the preamble
    // (template/kwargs, every prompt affected) or deep in the conversation.
    let Some(h) = model_hasher() else { return };
    let (w, _) = fake_worker(Engine::Corrupt(3), Arc::clone(&h)).await;
    let (verdict, detail) = probe_worker(&h, &reqwest::Client::new(), &w).await;
    assert_eq!(verdict, Parity::Diverged, "{detail}");
    assert!(detail.contains("diverges at token 3"), "{detail}");
    assert!(detail.contains("tools"), "{detail}");
}

#[tokio::test]
async fn falls_back_to_the_unprefixed_alias() {
    let Some(h) = model_hasher() else { return };
    let (w, f) = fake_worker(Engine::OnlyAlias, Arc::clone(&h)).await;
    let (verdict, detail) = probe_worker(&h, &reqwest::Client::new(), &w).await;
    assert_eq!(verdict, Parity::Confirmed, "{detail}");
    assert_eq!(f.paths.lock().unwrap()[0], "/v1/tokenize");
    assert_eq!(f.paths.lock().unwrap()[1], "/tokenize");
}

#[tokio::test]
async fn a_refusing_endpoint_is_unknown() {
    let Some(h) = model_hasher() else { return };
    let (w, _) = fake_worker(Engine::Refuse(500), Arc::clone(&h)).await;
    let (verdict, _) = probe_worker(&h, &reqwest::Client::new(), &w).await;
    assert_eq!(verdict, Parity::Unknown);
}

#[test]
fn a_departed_worker_stops_being_reported() {
    // The gauge is keyed by worker_id. One that outlives its worker keeps
    // exporting a verdict -- possibly a 0 -- about something not running, which
    // is how a fixed fleet keeps paging.
    let reg = ParityRegistry::default();
    for id in ["w1", "w2"] {
        assert!(reg.claim(id));
        reg.record(id, "glm53", Parity::Confirmed);
    }
    reg.retain(|id| id == "w2");
    assert_eq!(
        reg.snapshot(),
        vec![("w2".to_string(), "glm53".to_string(), 1)]
    );
}

#[test]
fn a_worker_is_only_probed_once() {
    // A verdict takes four round trips to earn, which is longer than the gap
    // between discovery snapshots. Without a claim, every snapshot spawns
    // another probe against a worker still answering the last one.
    let reg = ParityRegistry::default();
    assert!(reg.claim("w1"));
    assert!(!reg.claim("w1"), "a probe is already in flight");
    reg.record("w1", "glm53", Parity::Confirmed);
    assert!(!reg.claim("w1"), "this one already has a verdict");
}

#[test]
fn a_verdict_for_a_departed_worker_is_dropped() {
    // `retain` runs while a probe is in flight -- it holds a 10s timeout per
    // body against a worker that may be shutting down, which is exactly when
    // it is slowest to answer. Recording behind `retain`'s back resurrects the
    // gauge for a worker that is gone.
    let reg = ParityRegistry::default();
    assert!(reg.claim("w1"));
    reg.retain(|_| false);
    reg.record("w1", "glm53", Parity::Diverged);
    assert!(reg.snapshot().is_empty());
}

#[test]
fn the_gauge_encoding_keeps_diverged_distinct_from_unchecked() {
    // Both are "not confirmed", and collapsing them is the whole trap: -1 is
    // normal on an engine without the endpoint, 0 means kv-aware is off.
    let reg = ParityRegistry::default();
    for (id, verdict) in [
        ("a", Parity::Confirmed),
        ("b", Parity::Diverged),
        ("c", Parity::Unknown),
    ] {
        assert!(reg.claim(id));
        reg.record(id, "m", verdict);
    }
    let v: Vec<i8> = reg.snapshot().into_iter().map(|(_, _, g)| g).collect();
    assert_eq!(v, vec![1, 0, -1]);
}
