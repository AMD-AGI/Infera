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

/// The same event as SGLang emits it under EAGLE/MTP: `token_ids` carries the
/// overlapping bigrams `(t[i], t[i+1])` that key the radix tree, not bare ints.
fn block_stored_bigram(block_hashes: &[u64], parent: Option<u64>, flat: &[u32], bs: i64) -> Mv {
    let pairs: Vec<Mv> = flat
        .windows(2)
        .map(|w| Mv::Array(vec![Mv::from(w[0]), Mv::from(w[1])]))
        .collect();
    Mv::Array(vec![
        Mv::String("BlockStored".into()),
        Mv::Array(block_hashes.iter().map(|&h| Mv::from(h)).collect()),
        parent.map(Mv::from).unwrap_or(Mv::Nil),
        Mv::Array(pairs),
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

/// The MTP shape, over the same real socket.
///
/// This is the wire-level twin of the in-crate `decodes_sglang_bigram_batch_under_mtp`
/// unit test, and the one that would have caught the original bug in a live
/// deployment: a `--router-backend rust` router subscribed to an MTP-enabled
/// SGLang leg received pairs, decoded every element to `None`, and kept an empty
/// view forever. Nothing errored — kv-aware simply degraded to load balancing.
///
/// The assertion is that the bigram view hashes to *the same blocks* the query
/// side computes over the flat tokens, not merely that it is non-empty.
#[test]
fn subscriber_decodes_bigram_tokens_under_mtp() {
    let ctx = zmq::Context::new();
    let pub_sock = ctx.socket(zmq::PUB).unwrap();
    pub_sock.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = pub_sock.get_last_endpoint().unwrap().unwrap();

    let client = KvEventClient::new();
    client.on_worker_added(&worker("w", &endpoint, 4));

    // Flat tokens 1..=9 -> 8 bigrams -> two blocks of 4 whose first elements are
    // [1,2,3,4] and [5,6,7,8]: exactly the slice the query side chunks.
    let payload = batch(vec![block_stored_bigram(
        &[333, 444],
        None,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
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
        "bigram token_ids must mirror to the same blocks as the flat slice; \
         0 here is the original bug (every pair dropped, view stays empty)"
    );

    client.shutdown();
}

/// The failure the chain-health accounting exists to report, over a real socket.
///
/// An event whose `parent_block_hash` the router never saw cannot be placed on
/// the router's own chain, so it is dropped -- and the drop also withholds that
/// event's hash from the map, which orphans everything downstream of it in turn.
/// A worker that served traffic before the router subscribed emits exactly this
/// shape: the single `parent = None` root event is already gone from a PUB that
/// retains nothing, and the view then stays empty for the life of the process
/// while `/health` stays green and kv-aware quietly routes on load alone.
///
/// The second half is the repair. `AllBlocksCleared` is the only event that
/// rebuilds an anchor from nothing -- it puts both sides at the one state they
/// can agree on -- which is why flushing a worker's cache fixes a chain that no
/// amount of further traffic would.
#[test]
fn an_orphaned_event_is_dropped_and_a_clear_re_anchors_the_chain() {
    let ctx = zmq::Context::new();
    let pub_sock = ctx.socket(zmq::PUB).unwrap();
    pub_sock.bind("tcp://127.0.0.1:*").unwrap();
    let endpoint = pub_sock.get_last_endpoint().unwrap().unwrap();

    let client = KvEventClient::new();
    client.on_worker_added(&worker("w", &endpoint, 4));

    let publish = |payload: &[u8]| {
        pub_sock
            .send_multipart([b"kv-events".as_ref(), payload], 0)
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
    };

    // A rooted event first, so that the later assertion of absence means the
    // event was dropped rather than that the slow joiner had not connected yet.
    let rooted = batch(vec![block_stored(
        &[111, 222],
        None,
        &[1, 2, 3, 4, 5, 6, 7, 8],
        4,
    )]);
    let q_rooted = hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && client.prefix_hits("w", None, &q_rooted) < 2 {
        publish(&rooted);
    }
    assert_eq!(
        client.prefix_hits("w", None, &q_rooted),
        2,
        "the subscription must be live before absence proves anything"
    );

    // Parent 999 was never stored, so this span has nowhere to attach.
    let q_orphan = hash_request(&[5, 5, 5, 5, 6, 6, 6, 6], 4);
    let orphan = batch(vec![block_stored(
        &[333, 444],
        Some(999),
        &[5, 5, 5, 5, 6, 6, 6, 6],
        4,
    )]);
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        publish(&orphan);
    }
    assert_eq!(
        client.prefix_hits("w", None, &q_orphan),
        0,
        "an event whose parent was never seen must not enter the view"
    );
    assert_eq!(
        client.total_blocks("w"),
        2,
        "and dropping it must leave the chain that did resolve alone"
    );

    // Clear, then the same span rooted: the chain re-anchors on the seed and the
    // blocks that were unreachable a moment ago land.
    let repair = batch(vec![
        Mv::Array(vec![Mv::String("AllBlocksCleared".into())]),
        block_stored(&[333, 444], None, &[5, 5, 5, 5, 6, 6, 6, 6], 4),
    ]);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && client.prefix_hits("w", None, &q_orphan) < 2 {
        publish(&repair);
    }
    assert_eq!(
        client.prefix_hits("w", None, &q_orphan),
        2,
        "a cleared chain must re-anchor on the next rooted event"
    );

    client.shutdown();
}
