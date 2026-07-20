///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Real-socket integration test for the kv-event subscriber: publish an actual
//! msgspec-format `KVEventBatch` over a live ZMQ PUB and assert the client
//! mirrors it into the per-rank view. This closes the one gap the in-crate unit
//! tests can't cover — the real ZMQ recv + msgpack decode path — which is
//! exactly the SGLang/vLLM wire-format compatibility risk.

use std::time::{Duration, Instant};

use infera_router::hasher::hash_request;
use infera_router::kv_event::KvEventClient;
use infera_router::pool::Worker;
use rmpv::Value as Mv;

fn worker(id: &str, endpoint: &str, block_size: i64) -> Worker {
    serde_json::from_value(serde_json::json!({
        "worker_id": id, "url": "http://x",
        "kv_events_endpoint": endpoint, "kv_block_size": block_size,
    }))
    .unwrap()
}

/// Encode a BlockStored event as msgspec does: a tagged array
/// `["BlockStored", block_hashes, parent_block_hash, token_ids, block_size, lora_id]`.
fn block_stored(block_hashes: &[u64], parent: Option<u64>, token_ids: &[u32], bs: i64) -> Mv {
    Mv::Array(vec![
        Mv::String("BlockStored".into()),
        Mv::Array(block_hashes.iter().map(|&h| Mv::from(h)).collect()),
        parent.map(Mv::from).unwrap_or(Mv::Nil),
        Mv::Array(token_ids.iter().map(|&t| Mv::from(t)).collect()),
        Mv::from(bs),
        Mv::Nil, // lora_id
    ])
}

/// Encode a KVEventBatch: array_like `[ts, events, attn_dp_rank?]`.
fn batch(events: Vec<Mv>) -> Vec<u8> {
    let v = Mv::Array(vec![Mv::from(0.0_f64), Mv::Array(events)]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &v).unwrap();
    buf
}

#[test]
fn subscriber_decodes_real_zmq_msgpack_into_view() {
    let ctx = zmq::Context::new();
    let pub_sock = ctx.socket(zmq::PUB).unwrap();
    // Bind to an ephemeral port and read back the concrete endpoint.
    pub_sock.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = pub_sock.get_last_endpoint().unwrap().unwrap();

    let client = KvEventClient::new();
    client.on_worker_added(&worker("w", &endpoint, 4));

    // PUB/SUB is a slow joiner: publish repeatedly until the SUB has connected
    // and the client's view reflects the two stored blocks (8 tokens / bs 4).
    let payload = batch(vec![block_stored(
        &[111, 222],
        None,
        &[1, 2, 3, 4, 5, 6, 7, 8],
        4,
    )]);
    let query = hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
    assert_eq!(query.len(), 2);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut hits = 0;
    while Instant::now() < deadline {
        pub_sock
            .send_multipart([b"kv-events".as_ref(), payload.as_ref()], 0)
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
        hits = client.prefix_hits("w", None, &query);
        if hits == 2 {
            break;
        }
    }
    assert_eq!(
        hits, 2,
        "client should mirror the 2 stored blocks decoded from the real ZMQ stream"
    );

    // A divergent second block breaks the prefix at 1 (same chain semantics).
    let q2 = hash_request(&[1, 2, 3, 4, 9, 9, 9, 9], 4);
    assert_eq!(client.prefix_hits("w", None, &q2), 1);

    client.shutdown();
}
