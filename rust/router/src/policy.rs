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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use serde_json::Value;

use crate::block_hasher::BlockHasher;
use crate::cache_control::{extract_image_keys, hints_for_hashed_body, CacheHints, Retention};
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

    /// `(worker_id, model, gauge)` render-parity verdicts for `/metrics`, where
    /// the gauge is 1 confirmed / 0 diverged / -1 unknown. Empty for policies
    /// that do not render prompts and therefore cannot disagree with an engine
    /// about them.
    fn render_parity(&self) -> Vec<(String, String, i8)> {
        Vec::new()
    }

    /// `(worker_id, variant label)` for the server-side template defaults each
    /// worker reported. Exported so the fleet's variant *distribution* is
    /// visible before anyone trusts it: a fleet reporting one variant never
    /// needed the per-worker tier, and a fleet reporting two is one that a
    /// single router-wide flag could not have been right for.
    fn render_variants(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// `(model, count)` of workers ever judged diverged. Unlike the gauge this
    /// outlives the worker, so a fleet that rolls through broken replicas
    /// leaves a trace.
    fn render_parity_diverged(&self) -> Vec<(String, u64)> {
        Vec::new()
    }
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

/// Consecutive picks that ask for blocks and find none before the policy says
/// so. Any single miss is ordinary -- a genuinely new prefix has to land
/// somewhere -- and only the run length separates that from a kv-event feed
/// that stopped working. Sized to fire inside one benchmark rather than one
/// shift, while staying quiet through a burst of unique prompts.
const ZERO_HIT_ALARM: u64 = 64;

/// Repeat interval after the first alarm, so a feed that stays broken keeps
/// saying so without one line per request.
const ZERO_HIT_ALARM_REPEAT: u64 = 1024;

/// Pick the worker minimising
///   `cost(w) = -w_overlap * hits(w) + load(w)`
///   `load(w) = active_blocks(w) + recent_blocks(w)`
/// where `hits(w)` is the longest cached prefix on that worker's DP rank,
/// `active_blocks(w)` is the refcounted set of distinct in-flight block hashes,
/// and `recent_blocks(w)` is a decayed sum of the blocks recently dispatched to
/// it. Both halves of the load term are needed -- see [`RECENT_DECAY`].
pub struct KvEventAwarePolicy {
    kv: Arc<KvEventClient>,
    hasher: Arc<BlockHasher>,
    /// Per-worker verdict from the startup render-parity probe -- see
    /// `crate::render_probe`. Exported on /metrics; the router never routes on
    /// it, because a worker whose render we cannot match is still a worker.
    parity: Arc<crate::render_probe::ParityRegistry>,
    /// Which server-side template defaults each worker renders with. Requests
    /// are hashed once per *variant*, not once per worker: a fleet launched
    /// from one workload has one variant and renders exactly as often as it did
    /// before this existed.
    variants: Arc<crate::render_variant::VariantRegistry>,
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
    /// Consecutive picks that requested blocks and found none on the winner.
    /// Per request, a dead event feed and a workload of unique prompts look
    /// identical; the streak is the only thing that tells them apart.
    zero_hit_streak: AtomicU64,
    /// Where to send "this worker's chain needs a cache flush" requests. `None`
    /// leaves the router purely observational, which is what the tests want.
    flush_tx: Option<crate::kv_selfheal::FlushRequests>,
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
            hasher: Arc::new(hasher),
            parity: Arc::new(Default::default()),
            variants: Arc::new(Default::default()),
            w: overlap_weight,
            w_prefill: prefill_overlap_weight.unwrap_or(overlap_weight),
            w_decode: decode_overlap_weight.unwrap_or(overlap_weight),
            active: Mutex::new(HashMap::new()),
            recent: Mutex::new(HashMap::new()),
            mm_affinity: Mutex::new(HashMap::new()),
            zero_hit_streak: AtomicU64::new(0),
            flush_tx: None,
        }
    }

    /// Let the policy repair a worker whose kv-event chain never anchored, by
    /// asking it to flush its prefix cache. Builder rather than a constructor
    /// argument: every existing caller wants the observational default.
    pub fn with_self_heal(mut self, tx: crate::kv_selfheal::FlushRequests) -> Self {
        self.flush_tx = Some(tx);
        self
    }

    /// The server-side template defaults to render with. Builder for the same
    /// reason as `with_self_heal`: every existing caller wants the empty
    /// variant, which is what the router has always assumed.
    pub fn with_variants(mut self, variants: crate::render_variant::VariantRegistry) -> Self {
        self.variants = Arc::new(variants);
        self
    }

    /// Watch for kv-aware having gone blind, and name which way it went.
    ///
    /// Two failures produce the same zero-hit picks and need opposite fixes, so
    /// the view size decides between them: an empty view means no events are
    /// being applied at all (a dead feed -- the router never sees the cache),
    /// while a populated one means events arrive but the router's own hashes
    /// never match them (its tokenisation or block size disagrees with the
    /// engine's). Neither shows up in `/health`, request latency, or error
    /// rate; without this the only symptom is a routing decision nobody reads.
    ///
    /// `overlap_steered` says whether this pick was actually made on text
    /// prefix overlap. When it was not, a zero-hit result is the expected
    /// outcome rather than a symptom, and must not reach the counter.
    fn note_hit_outcome(
        &self,
        picked: &RouteTarget,
        blocks: usize,
        hits: usize,
        overlap_steered: bool,
    ) {
        // Ahead of every return in this function, not merely ahead of the
        // alarm. The ask is armed by the event feed, so no property of this
        // pick bears on whether it should be delivered -- while the returns
        // below select for precisely the picks a broken worker does not
        // produce. `hits > 0` is the sharp one: a worker whose stale seeded
        // view still matches a hot shared prefix reports hits on every
        // request, so its repair would sit armed and never be handed to
        // anyone. `blocks == 0` costs the same way on a fleet serving only
        // vision traffic. And `zero_hit_streak` cannot stand in for either: it
        // is one counter for the whole fleet, reset by a hit on any worker, so
        // a fleet where one chain is dead and another is healthy never reaches
        // the alarm -- exactly the case the self-heal exists for.
        //
        // The lock is not new cost on the pick path: `pick` already takes this
        // mutex once per candidate to score it, and twice more for the winner.
        if let Some(tx) = &self.flush_tx {
            if self.kv.take_flush_request(&picked.worker.worker_id) {
                tx.request(&picked.worker);
            }
        }

        if blocks == 0 {
            return; // nothing could have hit; says nothing either way
        }
        if hits > 0 {
            let prev = self.zero_hit_streak.swap(0, Ordering::Relaxed);
            if prev >= ZERO_HIT_ALARM {
                tracing::info!(after_misses = prev, "kv-aware: cache locality recovered");
            }
            return;
        }

        // A multimodal pick is not steered by text overlap: `pick` zeroes
        // `w_overlap` for it, precisely because the text hasher cannot
        // reproduce the engine's image blocks (sglang substitutes pad-values,
        // vLLM folds in extra-keys). Its miss is therefore the designed
        // outcome, not evidence -- counting it made a healthy vision fleet
        // trip "kv event feed is not being applied" at 64 requests and then
        // repeat it forever, pointing at a feed that was fine.
        if !overlap_steered {
            return;
        }

        let streak = self.zero_hit_streak.fetch_add(1, Ordering::Relaxed) + 1;
        if streak != ZERO_HIT_ALARM && !streak.is_multiple_of(ZERO_HIT_ALARM_REPEAT) {
            return;
        }
        let indexed = self.kv.total_blocks(&picked.worker.worker_id);
        if indexed == 0 {
            tracing::warn!(
                streak,
                worker = %picked.route_key(),
                "kv-aware: consecutive picks found no cached prefix and the router \
                 holds no blocks for this worker -- its kv event feed is not being \
                 applied, so routing has degenerated to load-only"
            );
        } else {
            tracing::warn!(
                streak,
                worker = %picked.route_key(),
                indexed_blocks = indexed,
                "kv-aware: consecutive picks found no cached prefix even though the \
                 router holds blocks for this worker -- request hashing disagrees \
                 with the engine's (tokeniser or block size)"
            );
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

        // Hash the request once per distinct (block_size, render variant).
        //
        // Both halves are usually 1. A model has one page size, and a fleet
        // launched from one workload has one set of server-side template
        // defaults -- so this is one render, as it always was. The key exists
        // for the fleet that is NOT uniform, where a single hash cannot be
        // right for every candidate: the worker holding
        // `--default-chat-template-kwargs` renders a different preamble, so its
        // blocks are different blocks, and asking its KV view about ours is
        // asking the wrong question.
        //
        // Deliberately not keyed on `engine`, unlike the Python router's: there
        // the engine selects which tokenizer loader runs, here there is one.
        // Adding it would key a dimension this hasher does not vary on and
        // render the same prompt twice.
        // Normalised BEFORE the variant is applied, and once for the whole
        // fleet: the engine turns a `/v1/responses` body into a chat body
        // (`_make_request`) and only then merges its server-side template
        // defaults (`_process_messages`). The other order writes
        // `chat_template_kwargs` onto a body `to_chat_body` rebuilds from
        // scratch, dropping the variant for `/v1/responses` alone.
        let base = crate::responses_input::normalised(request);
        let mut hashes_for: HashMap<(i64, u64), Vec<u64>> = HashMap::new();
        let mut key_of: Vec<Option<(i64, u64)>> = Vec::with_capacity(targets.len());
        for t in &targets {
            let key = match t.worker.kv_block_size {
                Some(bs) if bs > 0 => {
                    let variant = self.variants.for_worker(&t.worker.worker_id);
                    let key = (bs, variant.id());
                    hashes_for.entry(key).or_insert_with(|| {
                        self.hasher.hash_for(&variant.apply(&base), bs as usize)
                    });
                    Some(key)
                }
                _ => None,
            };
            key_of.push(key);
        }

        // Retention may come from an edge-attached hint (`/v1/messages`
        // `cache_control` does not survive translation). Multimodal is read
        // from `base`: a Responses image lives on `input` and is invisible on
        // the raw-body hint `handlers` stamped.
        let hints = hints_for_hashed_body(request, &base);
        let base_weight = self.base_weight_for(role) * Self::retention_amplifier(&hints);
        // Multimodal requests: the text hasher can't reproduce the engine's image
        // blocks (sglang substitutes pad-values, vLLM folds in extra-keys), so
        // text overlap is unreliable and a same-text-different-image request could
        // collide → drop text overlap (w_overlap 0) and steer by IMAGE AFFINITY
        // instead. Engine-agnostic: affinity keys the router's own image→worker
        // map, so one code path serves sglang, vLLM and ATOM alike.
        let (w_overlap, mm_keys) = if hints.has_multimodal_content {
            (0.0, extract_image_keys(&base))
        } else {
            (base_weight, Vec::new())
        };
        let w_mm = base_weight * MM_IMAGE_BLOCK_WEIGHT;

        let empty: Vec<u64> = Vec::new();
        // By target position, not by worker: two targets can now want different
        // blocks for the same request, so there is nothing on the worker alone
        // to look them up by.
        let blocks_at = |i: usize| -> &Vec<u64> {
            key_of[i]
                .as_ref()
                .and_then(|k| hashes_for.get(k))
                .unwrap_or(&empty)
        };
        let hits_at = |i: usize| -> usize {
            let t = &targets[i];
            self.kv
                .prefix_hits(&t.worker.worker_id, t.dp_rank, blocks_at(i))
        };
        let cost_at = |i: usize| -> f64 {
            let t = &targets[i];
            let hits = hits_at(i);
            let route_key = t.route_key();
            // Image miss term: images this worker does NOT already hold cost w_mm
            // each; the worker with the warm vision cache pays 0 → wins the pick.
            let mm_miss = mm_keys
                .len()
                .saturating_sub(self.mm_hits(&route_key, &mm_keys));
            // Credit hits rather than charging misses. The two are the same
            // ranking whenever every candidate hashes to the same number of
            // blocks -- `w_overlap * request_blocks` is then a constant added to
            // every cost, and constants cancel in an argmin. They stop being the
            // same once `blocks_of` can differ per target, which it does as soon
            // as two workers render the prompt differently (a per-worker
            // `--default-chat-template-kwargs`, say). Charging misses would then
            // penalise the worker whose preamble is merely longer, by an amount
            // that has nothing to do with what either one has cached.
            -w_overlap * (hits as f64) + w_mm * (mm_miss as f64) + self.load_of(&route_key)
        };

        // min by (cost, load) — tie-break to least-loaded.
        let picked_i = (0..targets.len())
            .min_by(|&a, &b| {
                let (ca, cb) = (cost_at(a), cost_at(b));
                ca.partial_cmp(&cb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.load_of(&targets[a].route_key())
                            .partial_cmp(&self.load_of(&targets[b].route_key()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .expect("candidates non-empty");
        let picked = targets[picked_i].clone();

        let blocks = blocks_at(picked_i).clone();
        let hits = hits_at(picked_i);
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
        self.note_hit_outcome(&picked, blocks.len(), hits, w_overlap > 0.0);
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

    fn render_parity(&self) -> Vec<(String, String, i8)> {
        self.parity.snapshot()
    }

    fn render_variants(&self) -> Vec<(String, String)> {
        self.variants.snapshot()
    }

    fn render_parity_diverged(&self) -> Vec<(String, u64)> {
        self.parity.diverged_totals()
    }

    fn sync_workers(&self, active_workers: &[Arc<Worker>]) {
        self.kv.sync(active_workers);
        // Confirm, once per worker, that what we render is what it renders. A
        // divergence here is the one kv-aware failure that produces no error
        // anywhere, so it has to be actively looked for.
        crate::render_probe::spawn_probes(
            Arc::clone(&self.hasher),
            Arc::clone(&self.parity),
            Arc::clone(&self.variants),
            active_workers,
        );
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
        self.parity.retain(|wid| ids.contains(wid));
        self.variants.retain(|wid| ids.contains(wid));
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

    /// The router asks the worker to repair itself, rather than only logging.
    ///
    /// This is the glue the ZMQ transport depends on: there, the router
    /// subscribes only after the worker registers, so the rooted event is
    /// always already gone and no worker-side flush can help. Detection happens
    /// under the view mutex on a runtime-less thread and so cannot do the POST
    /// itself; this is the handoff that carries it to somewhere that can.
    #[test]
    fn a_worker_whose_chain_never_anchored_is_asked_to_flush() {
        use rmpv::Value as Mv;

        let kv = Arc::new(KvEventClient::nats_fed());
        let cands = vec![worker("a", 16, None)];
        kv.on_worker_added(&cands[0]);

        let (tx, mut rx) = crate::kv_selfheal::channel();
        let pol = KvEventAwarePolicy::new(kv.clone(), BlockHasher::disabled(), 20.0, None, None)
            .with_self_heal(tx);

        // An event naming a parent this router never saw -- what a worker that
        // warmed up before anyone subscribed emits for the rest of its life.
        let orphan = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            Mv::Array(vec![Mv::from(11u64)]),
            Mv::from(999u64),
            Mv::Array((1..=16u32).map(Mv::from).collect()),
            Mv::from(16i64),
            Mv::Nil,
        ]);
        let mut payload = Vec::new();
        rmpv::encode::write_value(
            &mut payload,
            &Mv::Array(vec![Mv::from(1.0), Mv::Array(vec![orphan])]),
        )
        .unwrap();
        kv.apply_encoded_batch("a", 0, &payload);

        // A pick that wanted blocks and found none: the moment the router has
        // both the evidence and the worker's address in hand.
        let targets = expand_targets(&cands);
        pol.note_hit_outcome(&targets[0], 1, 0, true);

        let asked = rx.try_recv().expect("the router must ask for the flush");
        assert_eq!(asked.worker_id, "a");
        assert!(
            rx.try_recv().is_err(),
            "and ask once -- repeating it would clear a live cache on every miss"
        );
    }

    /// The repair reaches a broken worker that is still reporting cache hits.
    ///
    /// Delivery sat below the `blocks == 0` and `hits > 0` returns, so an armed
    /// repair could only ever be handed over on a zero-hit pick. A worker whose
    /// chain died holding a hot shared prefix keeps matching it out of the
    /// stale view and reports hits on every request, so its ask stayed armed
    /// forever -- and the ask is armed by the event feed, which knows the chain
    /// is dead, not by a pick, which cannot tell.
    #[test]
    fn a_repair_is_delivered_even_on_a_pick_that_found_hits() {
        use rmpv::Value as Mv;

        let kv = Arc::new(KvEventClient::nats_fed());
        let cands = vec![worker("a", 16, None)];
        kv.on_worker_added(&cands[0]);

        let (tx, mut rx) = crate::kv_selfheal::channel();
        let pol = KvEventAwarePolicy::new(kv.clone(), BlockHasher::disabled(), 20.0, None, None)
            .with_self_heal(tx);

        let orphan = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            Mv::Array(vec![Mv::from(11u64)]),
            Mv::from(999u64),
            Mv::Array((1..=16u32).map(Mv::from).collect()),
            Mv::from(16i64),
            Mv::Nil,
        ]);
        let mut payload = Vec::new();
        rmpv::encode::write_value(
            &mut payload,
            &Mv::Array(vec![Mv::from(1.0), Mv::Array(vec![orphan])]),
        )
        .unwrap();
        kv.apply_encoded_batch("a", 0, &payload);

        // The pick hits: the stale view still covers this prefix. Says nothing
        // about the chain, and must not withhold the repair.
        let targets = expand_targets(&cands);
        pol.note_hit_outcome(&targets[0], 4, 4, true);
        assert_eq!(
            rx.try_recv()
                .expect("a hit must not swallow the repair")
                .worker_id,
            "a"
        );
    }

    /// The same, for a pick that asked for no blocks at all.
    #[test]
    fn a_repair_is_delivered_on_a_pick_that_wanted_no_blocks() {
        use rmpv::Value as Mv;

        let kv = Arc::new(KvEventClient::nats_fed());
        let cands = vec![worker("a", 16, None)];
        kv.on_worker_added(&cands[0]);

        let (tx, mut rx) = crate::kv_selfheal::channel();
        let pol = KvEventAwarePolicy::new(kv.clone(), BlockHasher::disabled(), 20.0, None, None)
            .with_self_heal(tx);

        let orphan = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            Mv::Array(vec![Mv::from(11u64)]),
            Mv::from(999u64),
            Mv::Array((1..=16u32).map(Mv::from).collect()),
            Mv::from(16i64),
            Mv::Nil,
        ]);
        let mut payload = Vec::new();
        rmpv::encode::write_value(
            &mut payload,
            &Mv::Array(vec![Mv::from(1.0), Mv::Array(vec![orphan])]),
        )
        .unwrap();
        kv.apply_encoded_batch("a", 0, &payload);

        let targets = expand_targets(&cands);
        pol.note_hit_outcome(&targets[0], 0, 0, false);
        assert_eq!(
            rx.try_recv()
                .expect("a short prompt must not swallow the repair")
                .worker_id,
            "a"
        );
    }

    #[test]
    fn a_healthy_chain_is_never_asked_to_flush() {
        let kv = Arc::new(KvEventClient::nats_fed());
        let cands = vec![worker("a", 16, None)];
        kv.on_worker_added(&cands[0]);

        let (tx, mut rx) = crate::kv_selfheal::channel();
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 20.0, None, None)
            .with_self_heal(tx);

        // Zero hits on their own are ordinary -- a genuinely new prefix has to
        // land somewhere -- and flushing over them would be pure damage.
        let targets = expand_targets(&cands);
        for _ in 0..ZERO_HIT_ALARM + 1 {
            pol.note_hit_outcome(&targets[0], 1, 0, true);
        }
        assert!(rx.try_recv().is_err());
    }

    /// A vision workload must not read as a broken kv-event feed.
    ///
    /// `pick` zeroes `w_overlap` for a multimodal request deliberately -- the
    /// text hasher cannot reproduce the engine's image blocks, so text overlap
    /// is not what steers the pick and its hit count is not a measurement.
    /// Every one of those designed misses still reached the counter, so a fleet
    /// serving nothing but images warned that its "kv event feed is not being
    /// applied" at 64 requests and repeated it every 1024 after -- against a
    /// feed that was fine, and pointing at a tokeniser/block-size mismatch that
    /// did not exist.
    #[test]
    fn multimodal_misses_do_not_trip_the_dead_feed_alarm() {
        let kv = Arc::new(KvEventClient::new());
        let pol = KvEventAwarePolicy::new(kv, BlockHasher::disabled(), 20.0, None, None);
        let cands = vec![worker("a", 16, None)];
        // Pre-tokenized `prompt` so the disabled hasher still yields blocks --
        // this pick genuinely asks for two blocks and finds neither.
        let ids: Vec<u64> = (1..=32).collect();
        let mm = json!({"prompt": ids, "images": ["https://cdn/cat.png"]});
        let text = json!({"prompt": ids});
        assert!(!pol.pick(&cands, &mm, Role::Prefill).blocks.is_empty());

        for _ in 0..ZERO_HIT_ALARM + 1 {
            pol.pick(&cands, &mm, Role::Prefill);
        }
        assert_eq!(
            pol.zero_hit_streak.load(Ordering::Relaxed),
            0,
            "a miss the router designed for is not evidence of anything"
        );

        // The same miss on a text request is evidence, and still counts.
        pol.pick(&cands, &text, Role::Prefill);
        assert_eq!(pol.zero_hit_streak.load(Ordering::Relaxed), 1);
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
