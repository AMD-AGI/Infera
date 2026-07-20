///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Router-side mirror of each worker's KV cache state — the Rust twin of
//! `infera.router.kv_event.client` + `events`.
//!
//! A rank-multiplexed SGLang worker (`--dp-size N`, DP attention) publishes each
//! DP rank's kv-events on its own port (`base + rank`). We open one ZMQ SUB
//! subscriber thread per rank and keep a per-rank chained-hash view + a
//! worker-hash -> router-hash translation map, so the policy can score cache
//! locality against the *specific* rank a request would land on.
//!
//! Wire format matches SGLang/vLLM's msgspec `KVEventBatch` (array_like; each
//! event is a tagged array `[ClassName, ...fields]`).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::hasher::{hash_chunk, ROUTER_SEED};
use crate::pool::Worker;

const TOPIC: &[u8] = b"kv-events";
const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 5_000;
const RECV_TIMEOUT_MS: i32 = 500; // so a removed worker's threads notice shutdown

/// rank -> the set of chained router hashes cached on that DP rank.
type RankViews = HashMap<i64, HashSet<u64>>;
/// rank -> (worker's own block hash -> our chained router hash).
#[allow(clippy::type_complexity)]
type RankMaps = HashMap<i64, HashMap<u64, u64>>;

/// One worker's per-rank cache mirror.
struct WorkerViews {
    block_size: usize,
    views: RankViews,
    maps: RankMaps,
}

impl WorkerViews {
    fn new(block_size: usize) -> Self {
        WorkerViews {
            block_size: block_size.max(1),
            views: HashMap::new(),
            maps: HashMap::new(),
        }
    }
}

/// A decoded KV cache event (only the fields we act on).
enum Event {
    Stored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
    },
    Removed {
        block_hashes: Vec<u64>,
    },
    Cleared,
}

/// worker_id -> (per-worker stop flag, one subscriber thread per DP rank).
#[allow(clippy::type_complexity)]
type SubThreads = HashMap<String, (Arc<AtomicBool>, Vec<JoinHandle<()>>)>;

pub struct KvEventClient {
    ctx: zmq::Context,
    state: Arc<Mutex<HashMap<String, WorkerViews>>>,
    threads: Mutex<SubThreads>,
}

impl Default for KvEventClient {
    fn default() -> Self {
        Self::new()
    }
}

impl KvEventClient {
    pub fn new() -> Self {
        KvEventClient {
            ctx: zmq::Context::new(),
            state: Arc::new(Mutex::new(HashMap::new())),
            threads: Mutex::new(HashMap::new()),
        }
    }

    /// Longest matching prefix of `query` present in the worker/rank's view.
    /// Mirrors `_cache_hits`: break on the first block that isn't cached.
    pub fn prefix_hits(&self, worker_id: &str, dp_rank: Option<i64>, query: &[u64]) -> usize {
        let state = self.state.lock().expect("kv view mutex poisoned");
        let wv = match state.get(worker_id) {
            Some(w) => w,
            None => return 0,
        };
        let view = match wv.views.get(&dp_rank.unwrap_or(0)) {
            Some(v) => v,
            None => return 0,
        };
        let mut n = 0;
        for h in query {
            if !view.contains(h) {
                break;
            }
            n += 1;
        }
        n
    }

    /// Total cached blocks across all ranks of a worker (telemetry/tests).
    pub fn total_blocks(&self, worker_id: &str) -> usize {
        let state = self.state.lock().expect("kv view mutex poisoned");
        state
            .get(worker_id)
            .map(|wv| wv.views.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }

    pub fn on_worker_added(&self, w: &Worker) {
        let endpoint = match &w.kv_events_endpoint {
            Some(ep) if !ep.is_empty() => ep.clone(),
            _ => return,
        };
        {
            let mut t = self.threads.lock().expect("kv threads mutex poisoned");
            if t.contains_key(&w.worker_id) {
                return;
            }
            let block_size = w.kv_block_size.unwrap_or(1).max(1) as usize;
            self.state
                .lock()
                .expect("kv view mutex poisoned")
                .insert(w.worker_id.clone(), WorkerViews::new(block_size));

            let multiplexed = w.dp_size.unwrap_or(1) > 1 && w.dp_rank.is_none();
            let n_ranks = if multiplexed {
                w.dp_size.unwrap_or(1)
            } else {
                1
            };
            let stop = Arc::new(AtomicBool::new(false));
            let mut handles = Vec::new();
            for r in 0..n_ranks {
                let ep = offset_endpoint(&endpoint, r);
                let ctx = self.ctx.clone();
                let state = self.state.clone();
                let worker_id = w.worker_id.clone();
                let stop_c = stop.clone();
                handles.push(std::thread::spawn(move || {
                    run_subscriber(ctx, state, worker_id, r, ep, stop_c);
                }));
            }
            t.insert(w.worker_id.clone(), (stop, handles));
        }
        tracing::info!(
            worker = %w.worker_id,
            ranks = w.dp_size.unwrap_or(1),
            endpoint = %endpoint,
            "kv events: subscribing"
        );
    }

    pub fn on_worker_removed(&self, worker_id: &str) {
        let entry = self
            .threads
            .lock()
            .expect("kv threads mutex poisoned")
            .remove(worker_id);
        if let Some((stop, handles)) = entry {
            stop.store(true, Ordering::Relaxed);
            for h in handles {
                let _ = h.join();
            }
        }
        self.state
            .lock()
            .expect("kv view mutex poisoned")
            .remove(worker_id);
        tracing::info!(worker = %worker_id, "kv events: unsubscribed");
    }

    /// Reconcile subscriptions against the current active fleet: subscribe any
    /// new worker, unsubscribe any that disappeared. Idempotent (safe to call on
    /// every discovery snapshot).
    pub fn sync(&self, workers: &[Arc<Worker>]) {
        use std::collections::HashSet;
        let current: HashSet<&str> = workers.iter().map(|w| w.worker_id.as_str()).collect();
        let known: Vec<String> = self
            .threads
            .lock()
            .expect("kv threads mutex poisoned")
            .keys()
            .cloned()
            .collect();
        for id in known {
            if !current.contains(id.as_str()) {
                self.on_worker_removed(&id);
            }
        }
        for w in workers {
            self.on_worker_added(w);
        }
    }

    pub fn shutdown(&self) {
        let ids: Vec<String> = self
            .threads
            .lock()
            .expect("kv threads mutex poisoned")
            .keys()
            .cloned()
            .collect();
        for id in ids {
            self.on_worker_removed(&id);
        }
    }
}

/// SGLang's `offset_endpoint_port`: rank r publishes on `base_port + r`.
fn offset_endpoint(endpoint: &str, rank: i64) -> String {
    if rank == 0 {
        return endpoint.to_string();
    }
    match endpoint.rsplit_once(':') {
        Some((head, port)) => match port.parse::<i64>() {
            Ok(p) => format!("{head}:{}", p + rank),
            Err(_) => endpoint.to_string(),
        },
        None => endpoint.to_string(),
    }
}

/// Outer loop: (re)establish the SUB socket on any failure, honouring `stop`.
fn run_subscriber(
    ctx: zmq::Context,
    state: Arc<Mutex<HashMap<String, WorkerViews>>>,
    worker_id: String,
    rank: i64,
    endpoint: String,
    stop: Arc<AtomicBool>,
) {
    let mut backoff = INITIAL_BACKOFF_MS;
    while !stop.load(Ordering::Relaxed) {
        match subscribe_once(&ctx, &state, &worker_id, rank, &endpoint, &stop) {
            Ok(()) => return, // stop requested
            Err(e) => {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                tracing::warn!(
                    worker = %worker_id, rank, err = %e, backoff_ms = backoff,
                    "kv subscriber errored; retrying"
                );
                std::thread::sleep(Duration::from_millis(backoff));
                backoff = (backoff * 2).min(MAX_BACKOFF_MS);
            }
        }
    }
}

fn subscribe_once(
    ctx: &zmq::Context,
    state: &Arc<Mutex<HashMap<String, WorkerViews>>>,
    worker_id: &str,
    rank: i64,
    endpoint: &str,
    stop: &Arc<AtomicBool>,
) -> Result<(), zmq::Error> {
    let sock = ctx.socket(zmq::SUB)?;
    sock.set_rcvtimeo(RECV_TIMEOUT_MS)?;
    sock.connect(endpoint)?;
    sock.set_subscribe(TOPIC)?;
    while !stop.load(Ordering::Relaxed) {
        match sock.recv_multipart(0) {
            Ok(frames) => {
                if let Some(payload) = frames.last() {
                    match decode_batch(payload) {
                        Ok(events) => apply_events(state, worker_id, rank, &events),
                        Err(e) => tracing::warn!(worker = %worker_id, err = %e, "kv decode failed"),
                    }
                }
            }
            Err(zmq::Error::EAGAIN) => continue, // rcvtimeo fired; re-check stop
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn apply_events(
    state: &Arc<Mutex<HashMap<String, WorkerViews>>>,
    worker_id: &str,
    rank: i64,
    events: &[Event],
) {
    let mut guard = state.lock().expect("kv view mutex poisoned");
    let wv = match guard.get_mut(worker_id) {
        Some(w) => w,
        None => return, // worker removed mid-flight
    };
    let bs = wv.block_size;
    let view = wv.views.entry(rank).or_default();
    // Split borrow: take the map for this rank too.
    let map = wv.maps.entry(rank).or_default();
    for ev in events {
        match ev {
            Event::Stored {
                block_hashes,
                parent_block_hash,
                token_ids,
            } => {
                let mut parent = match parent_block_hash {
                    None => ROUTER_SEED,
                    Some(ph) => match map.get(ph) {
                        Some(rh) => *rh,
                        None => continue, // chain broken: missing parent, drop
                    },
                };
                let n = token_ids.len() / bs;
                for i in 0..n {
                    let chunk = &token_ids[i * bs..(i + 1) * bs];
                    parent = hash_chunk(parent, chunk);
                    view.insert(parent);
                    if let Some(wh) = block_hashes.get(i) {
                        map.insert(*wh, parent);
                    }
                }
            }
            Event::Removed { block_hashes } => {
                for wh in block_hashes {
                    if let Some(rh) = map.remove(wh) {
                        view.remove(&rh);
                    }
                }
            }
            Event::Cleared => {
                view.clear();
                map.clear();
            }
        }
    }
}

// ---- msgpack decode (msgspec array_like tagged structs) --------------------

fn decode_batch(bytes: &[u8]) -> Result<Vec<Event>, String> {
    let val = rmpv::decode::read_value(&mut &bytes[..]).map_err(|e| e.to_string())?;
    let arr = val.as_array().ok_or("batch is not an array")?;
    // KVEventBatch = [ts, events, attn_dp_rank?]; events is arr[1].
    let events = arr
        .get(1)
        .and_then(|v| v.as_array())
        .ok_or("batch[1] (events) is not an array")?;
    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        if let Some(parsed) = parse_event(ev) {
            out.push(parsed);
        }
    }
    Ok(out)
}

fn parse_event(ev: &rmpv::Value) -> Option<Event> {
    // vLLM's KVCacheEvent base is `msgspec.Struct(tag=True)` WITHOUT `array_like`,
    // so its events are tagged MAPS ({"type": tag, "block_hashes": [...], ...}),
    // whereas SGLang/atom use tagged ARRAYS ([tag, ...fields]). Handle both.
    if ev.as_map().is_some() {
        return parse_event_map(ev);
    }
    let a = ev.as_array()?;
    let tag = a.first()?.as_str()?;
    match tag {
        // [tag, block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium?]
        "BlockStored" => Some(Event::Stored {
            block_hashes: a.get(1).map(as_u64_vec).unwrap_or_default(),
            parent_block_hash: a.get(2).and_then(as_u64_any),
            token_ids: a.get(3).map(as_u32_vec).unwrap_or_default(),
        }),
        // [tag, block_hashes, medium?]
        "BlockRemoved" => Some(Event::Removed {
            block_hashes: a.get(1).map(as_u64_vec).unwrap_or_default(),
        }),
        "AllBlocksCleared" => Some(Event::Cleared),
        _ => None,
    }
}

/// vLLM tagged-MAP event: `{"type": <tag>, <field>: <value>, ...}` (msgspec
/// `tag=True` map; the tag key is "type"). Fields are matched by NAME, so vLLM's
/// extra fields (lora_name, extra_keys, group_idx, kv_cache_spec_*) are ignored.
fn parse_event_map(ev: &rmpv::Value) -> Option<Event> {
    let map = ev.as_map()?;
    let get = |k: &str| {
        map.iter()
            .find(|(key, _)| key.as_str() == Some(k))
            .map(|(_, v)| v)
    };
    match get("type")?.as_str()? {
        "BlockStored" => Some(Event::Stored {
            block_hashes: get("block_hashes").map(as_u64_vec).unwrap_or_default(),
            parent_block_hash: get("parent_block_hash").and_then(as_u64_any),
            token_ids: get("token_ids").map(as_u32_vec).unwrap_or_default(),
        }),
        "BlockRemoved" => Some(Event::Removed {
            block_hashes: get("block_hashes").map(as_u64_vec).unwrap_or_default(),
        }),
        "AllBlocksCleared" => Some(Event::Cleared),
        _ => None,
    }
}

/// Read any msgpack integer as u64 (msgspec may encode as int or uint).
fn as_u64_any(v: &rmpv::Value) -> Option<u64> {
    match v {
        rmpv::Value::Integer(i) => i.as_u64().or_else(|| i.as_i64().map(|x| x as u64)),
        _ => None,
    }
}

fn as_u64_vec(v: &rmpv::Value) -> Vec<u64> {
    v.as_array()
        .map(|a| a.iter().filter_map(as_u64_any).collect())
        .unwrap_or_default()
}

fn as_u32_vec(v: &rmpv::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| as_u64_any(x).map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, ep: Option<&str>, bs: i64, dp_size: Option<i64>) -> Worker {
        serde_json::from_value(serde_json::json!({
            "worker_id": id, "url": "http://x",
            "kv_events_endpoint": ep, "kv_block_size": bs, "dp_size": dp_size,
        }))
        .unwrap()
    }

    // ---- per-engine wire-format fidelity ----------------------------------
    // vLLM, SGLang and Atom all publish the msgspec `KVEventBatch` (array_like;
    // each event is a tagged array), but with engine-specific SHAPES. These
    // tests build each engine's real shape and drive it through the SAME decoder
    // (`decode_batch`) + view maintenance the live subscriber uses.

    use rmpv::Value as Mv;

    fn enc(v: Mv) -> Vec<u8> {
        let mut b = Vec::new();
        rmpv::encode::write_value(&mut b, &v).unwrap();
        b
    }
    fn ints(v: &[u64]) -> Mv {
        Mv::Array(v.iter().map(|&x| Mv::from(x)).collect())
    }
    fn toks(v: &[u32]) -> Mv {
        Mv::Array(v.iter().map(|&x| Mv::from(x)).collect())
    }
    fn seq(a: u32, b: u32) -> Vec<u32> {
        (a..=b).collect()
    }
    /// Feed raw msgpack bytes through the REAL decode path into a worker/rank view.
    fn feed_wire(c: &KvEventClient, worker: &str, rank: i64, bytes: &[u8]) {
        let events = decode_batch(bytes).expect("decode_batch");
        apply_events(&c.state, worker, rank, &events);
    }

    #[test]
    fn decodes_sglang_dp_packed_batch() {
        // SGLang under --dp-size packs SEVERAL blocks per BlockStored and tags
        // the batch with attn_dp_rank (a 3-element KVEventBatch). Our decoder
        // reads events from batch[1] and ignores attn_dp_rank; the subscriber
        // supplies the rank (base_port+rank), so we apply at rank 1.
        let c = KvEventClient::new();
        c.on_worker_added(&worker("sgl", Some("tcp://127.0.0.1:6001"), 16, Some(2)));
        let stored = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            ints(&[900, 901]), // 2 packed blocks
            Mv::Nil,           // parent = root
            toks(&seq(1, 32)), // 32 tokens = 2 blocks of 16
            Mv::from(16i64),   // block_size
            Mv::Nil,           // lora_id
        ]);
        // batch = [ts, [events], attn_dp_rank]
        let batch = enc(Mv::Array(vec![
            Mv::from(1.0),
            Mv::Array(vec![stored]),
            Mv::from(1i64),
        ]));
        feed_wire(&c, "sgl", 1, &batch);
        let q = crate::hasher::hash_request(&seq(1, 32), 16);
        assert_eq!(q.len(), 2);
        assert_eq!(c.prefix_hits("sgl", Some(1), &q), 2);
        assert_eq!(
            c.prefix_hits("sgl", Some(0), &q),
            0,
            "other DP rank untouched"
        );
        c.shutdown();
    }

    #[test]
    fn decodes_vllm_batch_with_medium_field() {
        // vLLM sets the trailing `medium` field (offload tier), so its
        // BlockStored is a 7-element array; our decoder must ignore it.
        // BlockRemoved likewise carries `medium` (3-element).
        let c = KvEventClient::new();
        c.on_worker_added(&worker("vllm", Some("tcp://127.0.0.1:6002"), 16, None));
        let stored = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            ints(&[500, 501]),
            Mv::Nil,
            toks(&seq(1, 32)),
            Mv::from(16i64),
            Mv::Nil,
            Mv::String("GPU".into()), // medium (7th element)
        ]);
        feed_wire(
            &c,
            "vllm",
            0,
            &enc(Mv::Array(vec![Mv::from(2.0), Mv::Array(vec![stored])])),
        );
        let q = crate::hasher::hash_request(&seq(1, 32), 16);
        assert_eq!(c.prefix_hits("vllm", None, &q), 2);
        // BlockRemoved carrying medium evicts the FIRST block → prefix breaks at 0.
        let removed = Mv::Array(vec![
            Mv::String("BlockRemoved".into()),
            ints(&[500]),
            Mv::String("GPU".into()),
        ]);
        feed_wire(
            &c,
            "vllm",
            0,
            &enc(Mv::Array(vec![Mv::from(3.0), Mv::Array(vec![removed])])),
        );
        assert_eq!(c.total_blocks("vllm"), 1);
        assert_eq!(c.prefix_hits("vllm", None, &q), 0, "first block evicted");
        c.shutdown();
    }

    #[test]
    fn decodes_vllm_map_event() {
        // Current vLLM (`vllm/distributed/kv_events.py`) makes its KVCacheEvent
        // base `msgspec.Struct(tag=True)` WITHOUT `array_like`, so events are
        // tagged MAPS ({"type": tag, field: value, ...}) with many extra fields —
        // NOT the tagged array the router originally assumed. Decode by field name.
        let c = KvEventClient::new();
        c.on_worker_added(&worker("vllm", Some("tcp://127.0.0.1:6003"), 16, None));
        let kv = |k: &str, v: Mv| (Mv::String(k.into()), v);
        let stored = Mv::Map(vec![
            kv("type", Mv::String("BlockStored".into())),
            kv("block_hashes", ints(&[700, 701])),
            kv("parent_block_hash", Mv::Nil),
            kv("token_ids", toks(&seq(1, 32))),
            kv("block_size", Mv::from(16i64)),
            kv("lora_id", Mv::Nil),
            kv("medium", Mv::String("GPU".into())),
            // vLLM's extra fields the decoder must ignore:
            kv("lora_name", Mv::Nil),
            kv("extra_keys", Mv::Nil),
            kv("group_idx", Mv::from(0i64)),
            kv("kv_cache_spec_kind", Mv::Nil),
        ]);
        // batch stays an array_like KVEventBatch [ts, events, dp_rank]; only the
        // events inside are maps.
        feed_wire(
            &c,
            "vllm",
            0,
            &enc(Mv::Array(vec![Mv::from(2.0), Mv::Array(vec![stored])])),
        );
        let q = crate::hasher::hash_request(&seq(1, 32), 16);
        assert_eq!(c.prefix_hits("vllm", None, &q), 2, "map event decoded");
        // A tagged-MAP BlockRemoved evicts the first block too.
        let removed = Mv::Map(vec![
            (Mv::String("type".into()), Mv::String("BlockRemoved".into())),
            (Mv::String("block_hashes".into()), ints(&[700])),
            (Mv::String("medium".into()), Mv::String("GPU".into())),
        ]);
        feed_wire(
            &c,
            "vllm",
            0,
            &enc(Mv::Array(vec![Mv::from(3.0), Mv::Array(vec![removed])])),
        );
        assert_eq!(c.prefix_hits("vllm", None, &q), 0, "first block evicted");
        c.shutdown();
    }

    #[test]
    fn decodes_atom_single_block_per_event() {
        // Atom re-publishes ONE block per BlockStored (block_hashes=[bh]),
        // chaining `parent_block_hash` across events; medium omitted (6-element).
        let c = KvEventClient::new();
        c.on_worker_added(&worker("atom", Some("tcp://127.0.0.1:6003"), 16, None));
        let b1 = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            ints(&[10]),       // single block
            Mv::Nil,           // parent = root
            toks(&seq(1, 16)), // one block of 16
            Mv::from(16i64),
            Mv::Nil,
        ]);
        let b2 = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            ints(&[11]),
            Mv::from(10u64), // parent = block 10 (chained off the previous event)
            toks(&seq(17, 32)),
            Mv::from(16i64),
            Mv::Nil,
        ]);
        feed_wire(
            &c,
            "atom",
            0,
            &enc(Mv::Array(vec![Mv::from(4.0), Mv::Array(vec![b1, b2])])),
        );
        let q = crate::hasher::hash_request(&seq(1, 32), 16);
        assert_eq!(q.len(), 2);
        assert_eq!(
            c.prefix_hits("atom", None, &q),
            2,
            "two single-block events chain into a 2-block prefix"
        );
        c.shutdown();
    }

    #[test]
    fn atom_chain_breaks_on_missing_parent() {
        // If Atom's parent block is missing (cold-start / out-of-order), the
        // decoder drops that event rather than mis-chaining.
        let c = KvEventClient::new();
        c.on_worker_added(&worker("atom", Some("tcp://127.0.0.1:6004"), 16, None));
        let orphan = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            ints(&[11]),
            Mv::from(999u64), // parent never stored → chain broken
            toks(&seq(17, 32)),
            Mv::from(16i64),
            Mv::Nil,
        ]);
        feed_wire(
            &c,
            "atom",
            0,
            &enc(Mv::Array(vec![Mv::from(5.0), Mv::Array(vec![orphan])])),
        );
        assert_eq!(
            c.total_blocks("atom"),
            0,
            "orphan event dropped, not mis-chained"
        );
        c.shutdown();
    }

    // Drive the view-maintenance chain directly (no sockets) — this is the core
    // correctness property: a BlockStored feeds token_ids through the SAME chain
    // as the query side, so prefix_hits matches hash_request.
    #[test]
    fn stored_then_prefix_hits_match_query_chain() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5557"), 4, None));

        // Worker stores 2 blocks (8 tokens, block_size 4) from a cold root.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![111, 222],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }],
        );

        // Query with the same tokens → the router-side hash_request must be a
        // full 2-block prefix hit against the mirrored view.
        let q = crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(q.len(), 2);
        assert_eq!(c.prefix_hits("w", None, &q), 2);
        // A divergent second block breaks the prefix at 1.
        let q2 = crate::hasher::hash_request(&[1, 2, 3, 4, 9, 9, 9, 9], 4);
        assert_eq!(c.prefix_hits("w", None, &q2), 1);
        c.shutdown();
    }

    #[test]
    fn removed_and_cleared_evict() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5558"), 4, None));
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![111, 222],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }],
        );
        assert_eq!(c.total_blocks("w"), 2);
        // Remove the first worker block → its router hash leaves the view.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Removed {
                block_hashes: vec![111],
            }],
        );
        assert_eq!(c.total_blocks("w"), 1);
        apply_events(&c.state, "w", 0, &[Event::Cleared]);
        assert_eq!(c.total_blocks("w"), 0);
        c.shutdown();
    }

    #[test]
    fn per_rank_views_are_isolated() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5559"), 4, Some(2)));
        // rank 0 caches [1..8], rank 1 caches nothing.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![1],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
            }],
        );
        let q = crate::hasher::hash_request(&[1, 2, 3, 4], 4);
        assert_eq!(c.prefix_hits("w", Some(0), &q), 1);
        assert_eq!(c.prefix_hits("w", Some(1), &q), 0); // isolated per DP rank
        c.shutdown();
    }

    #[test]
    fn offset_endpoint_bumps_port_per_rank() {
        assert_eq!(offset_endpoint("tcp://h:5557", 0), "tcp://h:5557");
        assert_eq!(offset_endpoint("tcp://h:5557", 3), "tcp://h:5560");
    }
}
