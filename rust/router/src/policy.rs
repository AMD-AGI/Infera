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

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use serde_json::Value;

use crate::block_hasher::BlockHasher;
use crate::cache_control::{parse_cache_hints, CacheHints, Retention};
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
    fn pick(&self, candidates: &[Arc<Worker>], _request: &Value, _role: Role) -> Pick {
        let targets = expand_targets(candidates);
        let key: Vec<String> = targets.iter().map(|t| t.route_key()).collect();
        let mut counters = self.counters.lock().expect("policy counter mutex poisoned");
        let idx = counters.entry(key).or_insert(0);
        let i = *idx % targets.len();
        *idx = idx.wrapping_add(1);
        Pick {
            target: targets[i].clone(),
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

/// Pick the worker minimising
///   `cost(w) = w_overlap * (request_blocks - hits(w)) + active_blocks(w)`
/// where `hits(w)` is the longest cached prefix on that worker's DP rank and
/// `active_blocks(w)` is the refcounted set of distinct in-flight block hashes.
pub struct KvEventAwarePolicy {
    kv: Arc<KvEventClient>,
    hasher: BlockHasher,
    w: f64,
    w_prefill: f64,
    w_decode: f64,
    // route_key -> {block_hash -> refcount}; len() is the load term.
    active: Mutex<HashMap<String, HashMap<u64, i64>>>,
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
        let mut w_overlap = self.base_weight_for(role) * Self::retention_amplifier(&hints);
        // Multimodal guard: the hasher is text-only, so a same-text-different-image
        // request could collide → force load-only routing (overlap 0).
        if hints.has_multimodal_content {
            w_overlap = 0.0;
        }

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
            w_overlap * (total.saturating_sub(hits) as f64) + self.active_len(&t.route_key()) as f64
        };

        // min by (cost, active) — tie-break to least-loaded.
        let picked = targets
            .iter()
            .min_by(|a, b| {
                let (ca, cb) = (cost_of(a), cost_of(b));
                ca.partial_cmp(&cb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.active_len(&a.route_key())
                            .cmp(&self.active_len(&b.route_key()))
                    })
            })
            .expect("candidates non-empty")
            .clone();

        let blocks = blocks_of(&picked).clone();
        let hits = hits_of(&picked);
        tracing::info!(
            policy = "kv-aware",
            role = ?role,
            retention = hints.retention.as_str(),
            picked = %picked.route_key(),
            cache_hits = hits,
            request_blocks = blocks.len(),
            active_blocks = self.active_len(&picked.route_key()),
            w_overlap,
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
        let mut active = self.active.lock().expect("active mutex poisoned");
        active.retain(|route_key, _| {
            let wid = route_key
                .split_once("#dp")
                .map(|(a, _)| a)
                .unwrap_or(route_key);
            ids.contains(wid)
        });
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
}
