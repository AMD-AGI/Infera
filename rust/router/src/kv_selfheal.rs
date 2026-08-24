///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Ask a worker to flush its prefix cache, to re-anchor a dead kv-event chain.
//!
//! The router's cache index is chained: each `BlockStored` names the
//! `parent_block_hash` it hangs off, and only the radix root reports `None`, so
//! the index can only be built forward from that one rooted event. An engine
//! binds its event publisher at launch and warms up immediately, so on the ZMQ
//! transport — where the router subscribes only *after* the worker registers —
//! the anchor has always already been broadcast to nobody. Every later event
//! then names a parent the router never saw and is dropped, along with its own
//! hash, orphaning its children in turn. Nothing errors: kv-aware just routes on
//! load alone behind a green `/health`.
//!
//! `AllBlocksCleared` is the only event that rebuilds an anchor from nothing,
//! and only the worker can emit one — so the router has to ask. This module is
//! the asking: a channel any thread can post to and one task that does the HTTP.
//!
//! The split is not stylistic. Detection happens in `kv_event::apply_events`,
//! which holds the global view mutex that every routing decision also takes,
//! and on the ZMQ path runs on a bare `std::thread` with no runtime attached —
//! so it can neither block nor `tokio::spawn`. An unbounded sender is callable
//! from both worlds without either.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::pool::Worker;

/// Bound on a single flush POST. The upstream client deliberately has no total
/// timeout — generations run arbitrarily long — but a flush that has not
/// answered in this long is a wedged worker, and waiting on it forever would
/// silently retire the self-heal for every other worker behind it.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(15);

/// Minimum spacing between flushes of the same worker.
///
/// The trigger is already one-shot per broken-chain episode, so this only
/// catches the case that would be genuinely destructive: a chain that breaks,
/// gets flushed, and breaks again immediately for some reason a flush cannot
/// fix. Without the floor that loop would clear a live worker's GPU prefix cache
/// on repeat, turning a routing degradation into a throughput collapse.
const FLUSH_COOLDOWN: Duration = Duration::from_secs(300);

/// Per-engine cache-flush endpoint; mirrors `infera/engine/flush.py`.
fn flush_path(engine: &str) -> &'static str {
    match engine {
        "vllm" => "/reset_prefix_cache",
        _ => "/flush_cache",
    }
}

/// Handle for requesting a flush. Cloneable, and `request` never blocks.
#[derive(Clone)]
pub struct FlushRequests {
    tx: mpsc::UnboundedSender<Arc<Worker>>,
}

impl FlushRequests {
    /// Queue a flush for `w`. Safe to call from a non-async thread, from under
    /// a lock, or on the request hot path: it is a lock-free enqueue.
    pub fn request(&self, w: &Arc<Worker>) {
        // A closed channel means the drain task is gone, which is a router
        // teardown, not something to report per worker.
        let _ = self.tx.send(w.clone());
    }
}

/// A handle whose requests land in a receiver the caller holds, instead of in
/// the actor. Lets a test observe what the router decided to flush without
/// standing up an HTTP server for it.
#[cfg(test)]
pub(crate) fn channel() -> (FlushRequests, mpsc::UnboundedReceiver<Arc<Worker>>) {
    let (tx, rx) = mpsc::unbounded_channel();
    (FlushRequests { tx }, rx)
}

/// Start the flush actor and return its handle.
///
/// `http` should be the shared upstream client — it is the one configured for
/// reaching workers, and building a second would mean a second connection pool.
pub fn spawn(http: reqwest::Client) -> FlushRequests {
    let (tx, mut rx) = mpsc::unbounded_channel::<Arc<Worker>>();
    tokio::spawn(async move {
        let mut last: HashMap<String, Instant> = HashMap::new();
        while let Some(w) = rx.recv().await {
            // A worker reachable only over the NATS request envelope may have
            // no route from here at all, so an HTTP flush would fail on every
            // retry while looking like an unreachable worker.
            if w.request_transport != "http" {
                tracing::info!(
                    worker = %w.worker_id, transport = %w.request_transport,
                    "kv events: chain has no anchor, but this worker is not reachable over \
                     HTTP -- cannot flush its cache, so kv-aware stays load-only for it"
                );
                continue;
            }
            let now = Instant::now();
            if let Some(prev) = last.get(&w.worker_id) {
                if now.duration_since(*prev) < FLUSH_COOLDOWN {
                    continue;
                }
            }
            last.insert(w.worker_id.clone(), now);

            // Spawned rather than awaited: a slow worker must not delay the
            // flush of the next one, and both are typically broken at once.
            let http = http.clone();
            tokio::spawn(async move {
                flush_one(&http, &w).await;
            });
        }
    });
    FlushRequests { tx }
}

async fn flush_one(http: &reqwest::Client, w: &Arc<Worker>) {
    let url = format!("{}{}", w.url, flush_path(&w.engine));
    tracing::warn!(
        worker = %w.worker_id, %url,
        "kv events: chain never anchored -- flushing this worker's prefix cache so it \
         re-emits a rooted event. Cache locality is lost for the requests that follow, \
         and rebuilt from the next ones."
    );
    match http.post(&url).timeout(FLUSH_TIMEOUT).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!(worker = %w.worker_id, "kv events: cache flushed; chain will re-anchor");
        }
        Ok(resp) => {
            // SGLang answers 400 while its scheduler is busy. Not retried here:
            // the next orphaned batch re-arms the request, and a worker serving
            // enough traffic to stay busy is producing those continuously.
            tracing::warn!(
                worker = %w.worker_id, status = %resp.status(),
                "kv events: cache flush refused; chain stays unanchored for now"
            );
        }
        Err(e) => {
            tracing::warn!(worker = %w.worker_id, err = %e, "kv events: cache flush failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(engine: &str, transport: &str) -> Arc<Worker> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "worker_id": "w1", "url": "http://x:1", "engine": engine,
                "request_transport": transport,
            }))
            .unwrap(),
        )
    }

    #[test]
    fn flush_path_follows_the_engine() {
        assert_eq!(flush_path("sglang"), "/flush_cache");
        assert_eq!(flush_path("vllm"), "/reset_prefix_cache");
        // An unknown engine gets the SGLang endpoint rather than nothing: it is
        // the default deployment, and a wrong 404 is cheaper than never healing.
        assert_eq!(flush_path("something-else"), "/flush_cache");
    }

    #[tokio::test]
    async fn request_never_blocks_and_survives_a_dead_actor() {
        let (tx, rx) = mpsc::unbounded_channel::<Arc<Worker>>();
        let req = FlushRequests { tx };
        drop(rx); // actor gone
                  // The detector calls this from under the view mutex; it must not panic
                  // or block when nobody is listening.
        req.request(&worker("sglang", "http"));
    }
}
