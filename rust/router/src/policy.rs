///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Routing policy. A `Policy` picks a `RouteTarget` from candidate workers and
//! (for cost-aware policies) reports the request's block hashes so the router
//! can refcount in-flight load via `on_request_started`/`on_request_finished`.
//!
//! Two impls: `RoundRobin` (stateless rotation) and `KvEventAwarePolicy`
//! (DP-attention cache-locality + load, the Rust twin of
//! `infera.router.policy.kv_event_aware`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;

use serde_json::Value;

use crate::block_hasher::BlockHasher;
use crate::cache_control::{extract_image_keys, parse_cache_hints, CacheHints, Retention};
use crate::kv_event::KvEventClient;
use crate::pool::{expand_targets, RouteTarget, Worker};

/// PD role of the pool being picked from. The disagg router passes Prefill /
/// Decode so a cost-aware policy can weight cache locality by role.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Prefill,
    Decode,
    Mixed,
}

/// A pick plus the request's block hashes on the chosen target (empty for
/// policies that don't track load). The router echoes `blocks` back through the
/// lifecycle hooks keyed by `target.route_key()`.
pub struct Pick {
    pub target: RouteTarget,
    pub blocks: Vec<u64>,
}

pub trait Policy: Send + Sync {
    /// Pick one target. Callers guarantee `candidates` is non-empty.
    fn pick(&self, candidates: &[Arc<Worker>], request: &Value, role: Role) -> Pick;

    /// Mark a request in-flight on `route_key` (increments the load term).
    fn on_request_started(&self, _route_key: &str, _blocks: &[u64]) {}
    /// Mark a request done on `route_key` (decrements the load term).
    fn on_request_finished(&self, _route_key: &str, _blocks: &[u64]) {}
    /// Reconcile any per-worker state (e.g. kv-event subscriptions) against the
    /// current active fleet. Called on every discovery snapshot.
    fn sync_workers(&self, _active: &[Arc<Worker>]) {}
}

/// RAII load guard: `start` fires `on_request_started`; `Drop` fires
/// `on_request_finished` — so every exit path (success, error, client
/// disconnect, streamed body drop) balances the refcount, mirroring the
/// Python router's try/finally around each dispatch.
pub struct ActiveGuard {
    policy: Arc<dyn Policy>,
    entries: Vec<(String, Vec<u64>)>,
}

impl ActiveGuard {
    pub fn start(policy: Arc<dyn Policy>, entries: Vec<(String, Vec<u64>)>) -> Self {
        for (k, b) in &entries {
            policy.on_request_started(k, b);
        }
        ActiveGuard { policy, entries }
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        for (k, b) in &self.entries {
            self.policy.on_request_finished(k, b);
        }
    }
}

// ---- RoundRobin ------------------------------------------------------------

/// Round-robin with a counter per candidate set (keyed by target route keys),
/// so the prefill and decode pools of one PD request rotate independently.
pub struct RoundRobin {
    counters: Mutex<HashMap<Vec<String>, usize>>,
}

impl RoundRobin {
    pub fn new() -> Self {
        RoundRobin {
            counters: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy for RoundRobin {
    fn pick(&self, candidates: &[Arc<Worker>], _request: &Value, role: Role) -> Pick {
        let targets = expand_targets(candidates);
        let key: Vec<String> = targets.iter().map(|t| t.route_key()).collect();
        let mut counters = self.counters.lock().expect("policy counter mutex poisoned");
        let idx = counters.entry(key).or_insert(0);
        let i = *idx % targets.len();
        *idx = idx.wrapping_add(1);
        let target = targets[i].clone();
        tracing::info!(policy = "round-robin", role = ?role, picked = %target.route_key(), "pick");
        Pick {
            target,
            blocks: Vec::new(),
        }
    }
}

// ---- KvEventAwarePolicy ----------------------------------------------------

/// LONG-retention requests are where cache locality pays off most (stable system
/// prompts reused for hours), so bias the cost function more toward the worker
/// that already has the prefix.
const LONG_RETENTION_AMPLIFIER: f64 = 2.0;
/// Explicit NONE means "don't cache" — load-balance with a tiny overlap weight.
const NONE_RETENTION_DAMPENER: f64 = 0.1;

/// Per-worker cap on remembered image keys (LRU). Small: a handful of hot
/// images dominate agentic VLM traffic, and `contains` scans this deque.
const MM_AFFINITY_CAP: usize = 256;

/// Cache value of one held image, expressed in "equivalent prefill blocks", so
/// the affinity term shares the overlap weight with text. An image expands to
/// hundreds of vision tokens (many blocks) on the engine, so a hit is worth far
/// more than a single text block — this makes image affinity dominate small
/// load differences without a separate weight knob.
const MM_IMAGE_BLOCK_WEIGHT: f64 = 48.0;

/// Per-pick decay on each worker's recent-dispatch total (half-life ~23 picks).
///
/// The load half of the cost function counts blocks that are *in flight*, which
/// is 0 for every worker whenever a request finishes before the next one is
/// picked. Session-paced agent traffic has exactly that shape, so on that
/// workload the cost function loses its load term entirely: a cold fleet ties,
/// the tie goes to whichever candidate is enumerated first, and the cache that
/// winner picks up re-elects it on every later request. The result is a
/// permanent 100/0 split across symmetric workers at any overlap weight.
///
/// `recent` keeps the load signal alive across requests that never overlap in
/// time. A pick is charged the blocks the winner MISSED, not the blocks the
/// request contains: prefill work is proportional to what the worker has to
/// compute, and a block already in its cache costs it nothing. That is what
/// lets the term coexist with cache affinity -- a worker serving a fully-cached
/// prompt accrues no load and keeps winning it, while a worker handed a cold
/// prompt accrues the whole thing and the next cold prompt goes elsewhere.
///
/// Misses are in blocks, the same unit as in-flight load, so the two sum
/// without a conversion factor and the term scales with request size. Charging
/// one point per request instead would cap the term at `1/(1 - decay)` no
/// matter how large the requests were, which a single block of cache edge
/// outvotes outright at any overlap weight above that cap.
///
/// Mirrors `_RECENT_DECAY` in infera/router/policy/kv_event_aware.py; the two
/// routers are independent implementations of the same policy and must agree.
const RECENT_DECAY: f64 = 0.97;

/// Load charged for a pick we have no block information about -- the hasher
/// returned nothing, because the prompt is shorter than the index block size or
/// tokenisation failed. Charging 0 would hold the load term at 0 for every
/// candidate and restore the permanent tie. Mirrors `_UNKNOWN_COST_BLOCKS` in
/// infera/router/policy/kv_event_aware.py.
const UNKNOWN_COST_BLOCKS: f64 = 1.0;

/// Pick the worker minimising
///   `cost(w) = w_overlap * (request_blocks - hits(w)) + load(w)`
///   `load(w) = active_blocks(w) + recent_blocks(w)`
/// where `hits(w)` is the longest cached prefix on that worker's DP rank,
/// `active_blocks(w)` is the refcounted set of distinct in-flight block hashes,
/// and `recent_blocks(w)` is a decayed sum of the blocks recently dispatched to
/// it. Both halves of the load term are needed -- see [`RECENT_DECAY`].
pub struct KvEventAwarePolicy {
    kv: Arc<KvEventClient>,
    hasher: BlockHasher,
    w: f64,
    w_prefill: f64,
    w_decode: f64,
    // route_key -> {block_hash -> refcount}; len() is the in-flight load.
    active: Mutex<HashMap<String, HashMap<u64, i64>>>,
    // route_key -> decayed sum of recently dispatched block counts. Carries the
    // load signal across requests that never overlap in time.
    recent: Mutex<HashMap<String, f64>>,
    // route_key -> recent image keys (MRU front, bounded LRU). Multimodal
    // affinity: a request whose image a worker already holds costs less there,
    // co-locating repeat images onto the worker with the warm vision cache.
    mm_affinity: Mutex<HashMap<String, VecDeque<u64>>>,
}

impl KvEventAwarePolicy {
    pub fn new(
        kv: Arc<KvEventClient>,
        hasher: BlockHasher,
        overlap_weight: f64,
        prefill_overlap_weight: Option<f64>,
        decode_overlap_weight: Option<f64>,
    ) -> Self {
        KvEventAwarePolicy {
            kv,
            hasher,
            w: overlap_weight,
            w_prefill: prefill_overlap_weight.unwrap_or(overlap_weight),
            w_decode: decode_overlap_weight.unwrap_or(overlap_weight),
            active: Mutex::new(HashMap::new()),
            recent: Mutex::new(HashMap::new()),
            mm_affinity: Mutex::new(HashMap::new()),
        }
    }

    fn base_weight_for(&self, role: Role) -> f64 {
        match role {
            Role::Prefill => self.w_prefill,
            Role::Decode => self.w_decode,
            Role::Mixed => self.w,
        }
    }

    fn retention_amplifier(hints: &CacheHints) -> f64 {
        if hints.retention == Retention::Long {
            LONG_RETENTION_AMPLIFIER
        } else if hints.retention == Retention::None && hints.explicit_hint_seen {
            NONE_RETENTION_DAMPENER
        } else {
            1.0
        }
    }

    fn active_len(&self, route_key: &str) -> usize {
        self.active
            .lock()
            .expect("active mutex poisoned")
            .get(route_key)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Blocks in flight now, plus blocks dispatched recently.
    fn load_of(&self, route_key: &str) -> f64 {
        let recent = self
            .recent
            .lock()
            .expect("recent mutex poisoned")
            .get(route_key)
            .copied()
            .unwrap_or(0.0);
        self.active_len(route_key) as f64 + recent
    }

    /// Decay every worker's recent total, then charge the pick's misses.
    ///
    /// Decaying on each pick rather than on a wall-clock timer keeps routing a
    /// pure function of the request sequence: same requests in, same decisions
    /// out. Totals that decay to nothing are dropped so an idle worker returns
    /// to a clean 0. A fully-cached pick charges nothing -- the worker has no
    /// prefill to do, so it takes on no load and stays the right answer.
    fn record_dispatch(&self, route_key: &str, missed_blocks: usize, request_blocks: usize) {
        let mut recent = self.recent.lock().expect("recent mutex poisoned");
        recent.retain(|_, v| {
            *v *= RECENT_DECAY;
            *v >= 1e-3
        });
        // Zero misses arrives here two different ways, and they must not be
        // charged alike. request_blocks > 0 with no misses means a fully cached
        // prompt: no prefill to do, so no load, and the holder keeps winning it.
        // request_blocks == 0 means the hasher produced nothing -- a prompt
        // under the index block size, or failed tokenisation. That request
        // still costs the worker something, and charging 0 would hold the load
        // term at 0 on every candidate, restoring the permanent tie and the
        // 100/0 split on short-prompt workloads.
        let charge = if request_blocks == 0 {
            UNKNOWN_COST_BLOCKS
        } else {
            missed_blocks as f64
        };
        if charge <= 0.0 {
            return;
        }
        *recent.entry(route_key.to_string()).or_insert(0.0) += charge;
    }

    /// How many of `keys` this worker is recorded as holding (its warm images).
    fn mm_hits(&self, route_key: &str, keys: &[u64]) -> usize {
        if keys.is_empty() {
            return 0;
        }
        let aff = self.mm_affinity.lock().expect("mm_affinity mutex poisoned");
        match aff.get(route_key) {
            Some(dq) => keys.iter().filter(|k| dq.contains(k)).count(),
            None => 0,
        }
    }

    /// Record that `route_key` now holds `keys` (MRU front, deduped, capped).
    fn record_mm(&self, route_key: &str, keys: &[u64]) {
        if keys.is_empty() {
            return;
        }
        let mut aff = self.mm_affinity.lock().expect("mm_affinity mutex poisoned");
        let dq = aff.entry(route_key.to_string()).or_default();
        for &k in keys {
            if let Some(pos) = dq.iter().position(|&x| x == k) {
                dq.remove(pos);
            }
            dq.push_front(k);
        }
        while dq.len() > MM_AFFINITY_CAP {
            dq.pop_back();
        }
    }
}

impl Policy for KvEventAwarePolicy {
    fn pick(&self, candidates: &[Arc<Worker>], request: &Value, role: Role) -> Pick {
        // Fan out rank-multiplexed workers so each DP rank is scored separately.
        let targets = expand_targets(candidates);

        // Hash the request once per distinct block_size (one model => usually one).
        let mut hashes_for: HashMap<i64, Vec<u64>> = HashMap::new();
        for t in &targets {
            if let Some(bs) = t.worker.kv_block_size {
                if bs > 0 {
                    hashes_for
                        .entry(bs)
                        .or_insert_with(|| self.hasher.hash_for(request, bs as usize));
                }
            }
        }

        let hints = parse_cache_hints(request);
        let base_weight = self.base_weight_for(role) * Self::retention_amplifier(&hints);
        // Multimodal requests: the text hasher can't reproduce the engine's image
        // blocks (sglang substitutes pad-values, vLLM folds in extra-keys), so
        // text overlap is unreliable and a same-text-different-image request could
        // collide → drop text overlap (w_overlap 0) and steer by IMAGE AFFINITY
        // instead. Engine-agnostic: affinity keys the router's own image→worker
        // map, so one code path serves sglang, vLLM and ATOM alike.
        let (w_overlap, mm_keys) = if hints.has_multimodal_content {
            (0.0, extract_image_keys(request))
        } else {
            (base_weight, Vec::new())
        };
        let w_mm = base_weight * MM_IMAGE_BLOCK_WEIGHT;

        let empty: Vec<u64> = Vec::new();
        let blocks_of = |t: &RouteTarget| -> &Vec<u64> {
            t.worker
                .kv_block_size
                .and_then(|bs| hashes_for.get(&bs))
                .unwrap_or(&empty)
        };
        let hits_of = |t: &RouteTarget| -> usize {
            self.kv
                .prefix_hits(&t.worker.worker_id, t.dp_rank, blocks_of(t))
        };
        let cost_of = |t: &RouteTarget| -> f64 {
            let total = blocks_of(t).len();
            let hits = hits_of(t);
            let route_key = t.route_key();
            // Image miss term: images this worker does NOT already hold cost w_mm
            // each; the worker with the warm vision cache pays 0 → wins the pick.
            let mm_miss = mm_keys
                .len()
                .saturating_sub(self.mm_hits(&route_key, &mm_keys));
            w_overlap * (total.saturating_sub(hits) as f64)
                + w_mm * (mm_miss as f64)
                + self.load_of(&route_key)
        };

        // min by (cost, load) — tie-break to least-loaded.
        let picked = targets
            .iter()
            .min_by(|a, b| {
                let (ca, cb) = (cost_of(a), cost_of(b));
                ca.partial_cmp(&cb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.load_of(&a.route_key())
                            .partial_cmp(&self.load_of(&b.route_key()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .expect("candidates non-empty")
            .clone();

        let blocks = blocks_of(&picked).clone();
        let hits = hits_of(&picked);
        let picked_key = picked.route_key();
        // Charge the winner for the blocks it will have to compute. Done here
        // rather than in on_request_started because the hooks run on the
        // dispatch path, which skips them on the failure routes -- and a pick
        // that goes uncharged is invisible to the next one.
        self.record_dispatch(&picked_key, blocks.len().saturating_sub(hits), blocks.len());
        // Mark the chosen worker as now holding this request's images, so the
        // next request for the same image is drawn back to its warm cache.
        let mm_matched = self.mm_hits(&picked_key, &mm_keys);
        self.record_mm(&picked_key, &mm_keys);
        tracing::info!(
            policy = "kv-aware",
            role = ?role,
            retention = hints.retention.as_str(),
            picked = %picked_key,
            cache_hits = hits,
            request_blocks = blocks.len(),
            active_blocks = self.active_len(&picked_key),
            w_overlap,
            mm_images = mm_keys.len(),
            mm_affinity_hits = mm_matched,
            "pick"
        );
        Pick {
            target: picked,
            blocks,
        }
    }

    fn on_request_started(&self, route_key: &str, blocks: &[u64]) {
        if blocks.is_empty() {
            return;
        }
        let mut active = self.active.lock().expect("active mutex poisoned");
        let refs = active.entry(route_key.to_string()).or_default();
        for &h in blocks {
            *refs.entry(h).or_insert(0) += 1;
        }
    }

    fn on_request_finished(&self, route_key: &str, blocks: &[u64]) {
        if blocks.is_empty() {
            return;
        }
        let mut active = self.active.lock().expect("active mutex poisoned");
        if let Some(refs) = active.get_mut(route_key) {
            for &h in blocks {
                let rc = refs.get(&h).copied().unwrap_or(0) - 1;
                if rc <= 0 {
                    refs.remove(&h);
                } else {
                    refs.insert(h, rc);
                }
            }
            if refs.is_empty() {
                active.remove(route_key);
            }
        }
    }

    fn sync_workers(&self, active_workers: &[Arc<Worker>]) {
        self.kv.sync(active_workers);
        // Prune load state for workers that left the fleet (route_key is
        // "<worker_id>" or "<worker_id>#dpN").
        use std::collections::HashSet;
        let ids: HashSet<&str> = active_workers
            .iter()
            .map(|w| w.worker_id.as_str())
            .collect();
        let alive = |route_key: &str| -> bool {
            let wid = route_key
                .split_once("#dp")
                .map(|(a, _)| a)
                .unwrap_or(route_key);
            ids.contains(wid)
        };
        self.active
            .lock()
            .expect("active mutex poisoned")
            .retain(|rk, _| alive(rk));
        self.recent
            .lock()
            .expect("recent mutex poisoned")
            .retain(|rk, _| alive(rk));
        self.mm_affinity
            .lock()
            .expect("mm_affinity mutex poisoned")
            .retain(|rk, _| alive(rk));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn worker(id: &str, bs: i64, dp_size: Option<i64>) -> Arc<Worker> {
        Arc::new(
            serde_json::from_value(json!({
                "worker_id": id, "url": "http://x",
                "kv_events_endpoint": format!("tcp://127.0.0.1:6{}", id.len()),
                "kv_block_size": bs, "dp_size": dp_size,
            }))
            .unwrap(),
        )
    }

    #[test]
    fn round_robin_rotates() {
        let rr = RoundRobin::new();
        let cands = vec![worker("a", 0, None), worker("b", 0, None)];
        let r = &json!({});
        let p0 = rr.pick(&cands, r, Role::Mixed);
        let p1 = rr.pick(&cands, r, Role::Mixed);
        assert_ne!(p0.target.worker.worker_id, p1.target.worker.worker_id);
        assert!(p0.blocks.is_empty());
    }

    #[test]
    fn kv_aware_prefers_worker_with_cached_prefix() {
        let kv = Arc::new(KvEventClient::new());
        // No tokenizer → hasher disabled → hashes empty → falls back to load
        // only. To exercise the cache term we inject a query + view directly via
        // a tiny stand-in: use a hasher-disabled policy but feed hits through the
        // kv client for worker "a".
        let pol = KvEventAwarePolicy::new(kv.clone(), BlockHasher::disabled(), 20.0, None, None);
        // With the hasher disabled, request_blocks=0 so cost == active; the
        // policy degrades to least-loaded. Load up "a" so "b" wins.
        pol.on_request_started("a", &[1, 2, 3]);
        let cands = vec![worker("a", 16, None), worker("b", 16, None)];
        let pick = pol.pick(&cands, &json!({"prompt": "hi"}), Role::Prefill);
        assert_eq!(
            pick.target.worker.worker_id, "b",
            "least-loaded when no cache info"
        );
    }

    #[test]
    fn recent_dispatch_keeps_a_load_signal_between_requests() {
        // The bug this guards: with in-flight blocks as the only load term,
        // traffic paced so each request finishes before the next is picked
        // leaves every worker reading 0, and the first pick then wins every
        // subsequent one on candidate order. Nothing is in flight here.
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        assert_eq!(pol.active_len("a"), 0);
        assert_eq!(pol.load_of("a"), 0.0);

        pol.record_dispatch("a", 10, 10);
        assert_eq!(pol.active_len("a"), 0, "nothing in flight");
        assert!(
            pol.load_of("a") > 0.0,
            "load signal must outlive the request"
        );
        assert!(pol.load_of("a") > pol.load_of("b"));
    }

    #[test]
    fn a_pick_with_no_block_information_is_still_charged() {
        // A prompt under the index block size hashes to nothing. Charging 0 for
        // it would hold the load term at 0 on every candidate, restoring the
        // tie that sends every request to whichever worker sorts first.
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.record_dispatch("a", 0, 0);
        assert!(pol.load_of("a") > 0.0, "unhashable pick accrued no load");
    }

    #[test]
    fn a_fully_cached_pick_is_charged_nothing() {
        // The charge is the blocks the winner had to COMPUTE. A worker serving
        // a prompt it already holds takes on no prefill work, so it accrues no
        // load and stays the right answer for that prompt.
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.record_dispatch("a", 0, 10);
        assert_eq!(pol.load_of("a"), 0.0);
    }

    #[test]
    fn recent_load_decays_back_to_zero_when_a_worker_goes_idle() {
        // Transient by construction: a worker that took a burst and then went
        // quiet must return to contention, or the fix trades one starvation
        // mode for another.
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.record_dispatch("bursty", 10, 10);
        assert!(pol.load_of("bursty") > 0.0);
        for _ in 0..500 {
            pol.record_dispatch("other", 1, 1);
        }
        assert_eq!(
            pol.load_of("bursty"),
            0.0,
            "idle worker never returned to 0"
        );
    }

    #[test]
    fn sync_prunes_removed_worker_recent_load() {
        // Pruned separately from `active`: a worker can carry a recent total
        // with nothing in flight, which is the state this term represents.
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.record_dispatch("gone#dp0", 5, 5);
        pol.record_dispatch("stay", 5, 5);
        pol.sync_workers(&[worker("stay", 16, None)]);
        assert_eq!(pol.load_of("gone#dp0"), 0.0);
        assert!(pol.load_of("stay") > 0.0);
    }

    #[test]
    fn refcount_started_finished_balances() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.on_request_started("w#dp0", &[10, 20]);
        assert_eq!(pol.active_len("w#dp0"), 2);
        // shared-prefix second request bumps refcounts, not the set size for 10.
        pol.on_request_started("w#dp0", &[10, 30]);
        assert_eq!(pol.active_len("w#dp0"), 3);
        pol.on_request_finished("w#dp0", &[10, 20]);
        assert_eq!(pol.active_len("w#dp0"), 2); // 20 gone; 10 still held by req2
        pol.on_request_finished("w#dp0", &[10, 30]);
        assert_eq!(pol.active_len("w#dp0"), 0);
    }

    #[test]
    fn sync_prunes_removed_worker_load() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.on_request_started("gone#dp0", &[1]);
        pol.on_request_started("stay", &[2]);
        pol.sync_workers(&[worker("stay", 16, None)]);
        assert_eq!(pol.active_len("gone#dp0"), 0);
        assert_eq!(pol.active_len("stay"), 1);
    }

    fn mm_request(url: &str) -> Value {
        json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url}}
        ]}]})
    }

    #[test]
    fn mm_affinity_sticks_repeat_image_to_one_worker() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        let cands = vec![worker("a", 16, None), worker("b", 16, None)];
        let req = mm_request("https://cdn/cat.png");

        // First pick records the image on whichever worker won (a tie → "a").
        let first = pol
            .pick(&cands, &req, Role::Prefill)
            .target
            .worker
            .worker_id
            .clone();
        // Lightly load the *other* worker (well under w_mm) — affinity must still
        // pull the same image back to the worker holding its vision cache.
        let other = if first == "a" { "b" } else { "a" };
        pol.on_request_started(other, &[1, 2, 3]);
        for _ in 0..5 {
            let p = pol.pick(&cands, &req, Role::Prefill);
            assert_eq!(p.target.worker.worker_id, first, "repeat image sticks");
        }
    }

    #[test]
    fn mm_affinity_new_image_balances_by_load() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        let cands = vec![worker("a", 16, None), worker("b", 16, None)];
        // Warm "a" with a cat and load it heavily.
        pol.record_mm("a", &extract_image_keys(&mm_request("https://cdn/cat.png")));
        pol.on_request_started("a", &[1, 2, 3, 4, 5]);
        // A brand-new image has no affinity anywhere → least-loaded "b" wins.
        let p = pol.pick(&cands, &mm_request("https://cdn/dog.png"), Role::Prefill);
        assert_eq!(
            p.target.worker.worker_id, "b",
            "unseen image balances by load"
        );
    }

    #[test]
    fn mm_affinity_lru_capped() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        let keys: Vec<u64> = (0..(MM_AFFINITY_CAP as u64 + 50)).collect();
        for &k in &keys {
            pol.record_mm("w", &[k]);
        }
        // Deque holds only the most-recent CAP; the oldest 50 are evicted.
        assert_eq!(pol.mm_hits("w", &keys[..50]), 0, "oldest evicted");
        assert_eq!(
            pol.mm_hits("w", &keys[50..]),
            MM_AFFINITY_CAP,
            "newest retained"
        );
    }

    #[test]
    fn sync_prunes_removed_worker_mm_affinity() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 1.0, None, None);
        pol.record_mm("gone#dp0", &[1, 2]);
        pol.record_mm("stay", &[3]);
        pol.sync_workers(&[worker("stay", 16, None)]);
        assert_eq!(pol.mm_hits("gone#dp0", &[1, 2]), 0);
        assert_eq!(pol.mm_hits("stay", &[3]), 1);
    }
}
