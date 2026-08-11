///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Real-broker test for the NATS kv-event feed: publish an actual msgpack
//! `KVEventBatch` onto the subject a worker's relay uses, and assert the router
//! mirrors it into the cache view.
//!
//! This closes the gap the in-crate tests cannot: the subject encoding, the
//! JetStream stream shape, and the replay policy only ever meet the Python
//! relay on a live broker. Getting any of them wrong looks identical to a fleet
//! with no cache activity -- the router runs, routes, and simply never scores a
//! hit.
//!
//! Skipped when no broker is reachable, so it does not fail a plain `cargo
//! test`. Point it at one with `INFERA_TEST_NATS=nats://127.0.0.1:4222`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use infera_router::kv_event::KvEventClient;
use infera_router::kv_event_nats::{self, KV_EVENTS_SUBJECT_PREFIX, KV_VIEW_BUCKET};
use infera_router::pool::Worker;
use rmpv::Value as Mv;

fn broker_url() -> Option<String> {
    std::env::var("INFERA_TEST_NATS")
        .ok()
        .filter(|s| !s.is_empty())
}

fn worker(id: &str, block_size: i64) -> Worker {
    serde_json::from_value(serde_json::json!({
        "worker_id": id, "url": "http://x", "kv_block_size": block_size,
    }))
    .unwrap()
}

/// A BlockStored as msgspec encodes it: a tagged array.
fn block_stored(hashes: &[u64], parent: Option<u64>, token_ids: &[u32], bs: i64) -> Mv {
    Mv::Array(vec![
        Mv::from("BlockStored"),
        Mv::Array(hashes.iter().map(|h| Mv::from(*h)).collect()),
        parent.map(Mv::from).unwrap_or(Mv::Nil),
        Mv::Array(token_ids.iter().map(|t| Mv::from(*t)).collect()),
        Mv::from(bs),
        Mv::Nil,
    ])
}

/// `KVEventBatch = [ts, events, attn_dp_rank?]` -- msgspec `array_like`, the
/// same shape the ZMQ test builds.
fn batch(events: Vec<Mv>) -> Vec<u8> {
    let v = Mv::Array(vec![Mv::from(0.0_f64), Mv::Array(events)]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &v).unwrap();
    buf
}

#[tokio::test]
async fn a_published_batch_reaches_the_cache_view() {
    let url = match broker_url() {
        Some(u) => u,
        None => {
            eprintln!("skipping: set INFERA_TEST_NATS to a reachable broker");
            return;
        }
    };
    let wid = format!("10.0.0.1:{}", 9000 + (std::process::id() % 500));
    let token = URL_SAFE_NO_PAD.encode(wid.as_bytes());

    // Published *before* the router subscribes, which is the case that decides
    // between DeliverPolicy All and New: a BlockStored names its parent, and an
    // event whose parent was never seen is dropped, so replaying only new
    // messages would build an empty view for a worker that had already cached.
    let nc = async_nats::connect(&url).await.expect("connect");
    let js = async_nats::jetstream::new(nc.clone());
    let subject = format!("{KV_EVENTS_SUBJECT_PREFIX}.{token}.0");
    js.publish(
        subject.clone(),
        batch(vec![block_stored(
            &[111],
            None,
            &(0..16).collect::<Vec<_>>(),
            16,
        )])
        .into(),
    )
    .await
    .expect("publish root")
    .await
    .expect("ack root");
    js.publish(
        subject,
        batch(vec![block_stored(
            &[222],
            Some(111),
            &(16..32).collect::<Vec<_>>(),
            16,
        )])
        .into(),
    )
    .await
    .expect("publish child")
    .await
    .expect("ack child");

    let client = Arc::new(KvEventClient::nats_fed());
    client.on_worker_added(&worker(&wid, 16));
    let feed = client.clone();
    let feed_url = url.clone();
    tokio::spawn(async move {
        if let Err(e) = kv_event_nats::run(feed, Some(&feed_url)).await {
            eprintln!("kv feed stopped: {e:#}");
        }
    });

    let deadline = Instant::now() + Duration::from_secs(10);
    while client.total_blocks(&wid) < 2 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        client.total_blocks(&wid),
        2,
        "both blocks of the chain must be replayed, including the one published \
         before this router existed"
    );
}

#[tokio::test]
async fn the_bucket_bootstraps_a_cold_start_without_clobbering_a_live_view() {
    let url = match broker_url() {
        Some(u) => u,
        None => {
            eprintln!("skipping: set INFERA_TEST_NATS to a reachable broker");
            return;
        }
    };
    let wid = format!("10.0.1.1:{}", 9000 + (std::process::id() % 500));
    let token = URL_SAFE_NO_PAD.encode(wid.as_bytes());

    let nc = async_nats::connect(&url).await.expect("connect");
    let js = async_nats::jetstream::new(nc);
    let store = match js.get_key_value(KV_VIEW_BUCKET).await {
        Ok(s) => s,
        Err(_) => js
            .create_key_value(async_nats::jetstream::kv::Config {
                bucket: KV_VIEW_BUCKET.to_string(),
                ..Default::default()
            })
            .await
            .expect("bucket"),
    };
    let view = |hashes: Vec<u64>| {
        let mut buf = Vec::new();
        rmpv::encode::write_value(
            &mut buf,
            &Mv::Array(hashes.into_iter().map(Mv::from).collect()),
        )
        .unwrap();
        buf
    };
    store
        .put(format!("{token}.0"), view(vec![501, 502, 503]).into())
        .await
        .expect("seed");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("infera_router=debug")
        .try_init();
    let client = Arc::new(KvEventClient::nats_fed());
    client.on_worker_added(&worker(&wid, 16));
    let feed = client.clone();
    let feed_url = url.clone();
    tokio::spawn(async move {
        if let Err(e) = kv_event_nats::run(feed, Some(&feed_url)).await {
            eprintln!("kv feed stopped: {e:#}");
        }
    });

    let deadline = Instant::now() + Duration::from_secs(10);
    while client.total_blocks(&wid) < 3 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        client.total_blocks(&wid),
        3,
        "a cold start must be seeded from the bucket rather than waiting for \
         the worker to cache something new"
    );

    // A relay that desynced republishes an empty view. Applying it would wipe
    // the view and collapse cache hits to zero.
    store
        .put(format!("{token}.0"), view(vec![]).into())
        .await
        .expect("empty");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        client.total_blocks(&wid),
        3,
        "an empty snapshot must never clear a view that already has one"
    );
}
