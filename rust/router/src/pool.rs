///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Worker model + immutable pool snapshot.
//!
//! `Snapshot` is built once per fleet change by the discovery task and swapped
//! atomically (`ArcSwap`), so the request hot path reads it lock-free.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};

pub type SharedPool = Arc<ArcSwap<Snapshot>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisaggMode {
    Mixed,
    Prefill,
    Decode,
}

fn default_mode() -> DisaggMode {
    DisaggMode::Mixed
}
fn default_status() -> String {
    "active".to_string()
}
fn default_transport() -> String {
    "http".to_string()
}

/// Mirror of `infera.common.worker_pool.WorkerInfo` as registered in etcd.
/// Extra fields in the payload (e.g. `kv`) are ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub worker_id: String,
    pub url: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub engine: String,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default = "default_mode")]
    pub disagg_mode: DisaggMode,
    #[serde(default)]
    pub disagg_meta: serde_json::Value,
    #[serde(default)]
    pub kv_events_endpoint: Option<String>,
    #[serde(default)]
    pub kv_block_size: Option<i64>,
    #[serde(default)]
    pub dp_rank: Option<i64>,
    #[serde(default)]
    pub dp_size: Option<i64>,
    #[serde(default = "default_transport")]
    pub request_transport: String,
}

impl Worker {
    pub fn is_active(&self) -> bool {
        self.status == "active"
    }
}

/// Immutable view of the fleet. Pre-indexes active workers by (model, mode) so
/// the hot path is a single hashmap lookup.
pub struct Snapshot {
    pub all: Vec<Arc<Worker>>,
    active_by: HashMap<(String, DisaggMode), Vec<Arc<Worker>>>,
}

impl Snapshot {
    pub fn empty() -> Self {
        Snapshot {
            all: Vec::new(),
            active_by: HashMap::new(),
        }
    }

    pub fn build(workers: Vec<Arc<Worker>>) -> Self {
        let mut active_by: HashMap<(String, DisaggMode), Vec<Arc<Worker>>> = HashMap::new();
        for w in &workers {
            if !w.is_active() {
                continue;
            }
            active_by
                .entry((w.model_name.clone(), w.disagg_mode))
                .or_default()
                .push(w.clone());
        }
        Snapshot {
            all: workers,
            active_by,
        }
    }

    pub fn list_active(&self, model: &str, mode: DisaggMode) -> &[Arc<Worker>] {
        self.active_by
            .get(&(model.to_string(), mode))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn active_count(&self) -> usize {
        self.all.iter().filter(|w| w.is_active()).count()
    }
}

/// A dispatch target: a worker plus, for rank-multiplexed SGLang workers
/// (`dp_size>1`, `dp_rank` unset), the specific DP rank to pin.
#[derive(Clone)]
pub struct RouteTarget {
    pub worker: Arc<Worker>,
    pub dp_rank: Option<i64>,
}

impl RouteTarget {
    /// Stable per-target key (== worker_id for single-rank workers).
    pub fn route_key(&self) -> String {
        match self.dp_rank {
            None => self.worker.worker_id.clone(),
            Some(r) => format!("{}#dp{r}", self.worker.worker_id),
        }
    }
}

fn is_rank_multiplexed(w: &Worker) -> bool {
    w.dp_size.unwrap_or(1) > 1 && w.dp_rank.is_none()
}

/// One target per worker, except rank-multiplexed workers fan out per DP rank.
pub fn expand_targets(workers: &[Arc<Worker>]) -> Vec<RouteTarget> {
    let mut out = Vec::new();
    for w in workers {
        if is_rank_multiplexed(w) {
            for r in 0..w.dp_size.unwrap_or(1) {
                out.push(RouteTarget {
                    worker: w.clone(),
                    dp_rank: Some(r),
                });
            }
        } else {
            out.push(RouteTarget {
                worker: w.clone(),
                dp_rank: None,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn worker(spec: serde_json::Value) -> Arc<Worker> {
        Arc::new(serde_json::from_value(spec).unwrap())
    }

    #[test]
    fn defaults_and_active_filter() {
        // minimal payload → serde defaults kick in (status=active, mode=mixed)
        let w: Worker = serde_json::from_value(json!({
            "worker_id": "w", "url": "http://x", "extra_ignored": {"kv": 1}
        }))
        .unwrap();
        assert!(w.is_active());
        assert_eq!(w.disagg_mode, DisaggMode::Mixed);
        assert_eq!(w.request_transport, "http");
    }

    #[test]
    fn snapshot_indexes_only_active_by_model_and_mode() {
        let snap = Snapshot::build(vec![
            worker(
                json!({"worker_id": "a", "url": "u", "model_name": "m", "disagg_mode": "mixed"}),
            ),
            worker(
                json!({"worker_id": "b", "url": "u", "model_name": "m", "disagg_mode": "mixed", "status": "draining"}),
            ),
            worker(
                json!({"worker_id": "c", "url": "u", "model_name": "other", "disagg_mode": "mixed"}),
            ),
        ]);
        assert_eq!(snap.active_count(), 2);
        assert_eq!(snap.list_active("m", DisaggMode::Mixed).len(), 1);
        assert_eq!(snap.list_active("m", DisaggMode::Prefill).len(), 0);
        assert_eq!(snap.list_active("missing", DisaggMode::Mixed).len(), 0);
    }

    #[test]
    fn expand_fans_out_only_rank_multiplexed_workers() {
        // dp_size>1 with dp_rank unset → one target per rank
        let muxed = worker(json!({"worker_id": "p", "url": "u", "dp_size": 3}));
        let t = expand_targets(&[muxed]);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].route_key(), "p#dp0");
        assert_eq!(t[2].route_key(), "p#dp2");

        // dp_rank already pinned → single target, key == worker_id
        let pinned = worker(json!({"worker_id": "q", "url": "u", "dp_size": 3, "dp_rank": 1}));
        let t = expand_targets(&[pinned]);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].route_key(), "q");
    }
}
