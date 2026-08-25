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

/// Minimum spacing between flushes of the same worker after one lands.
///
/// The detector is not one-shot -- it re-asks on every batch that still reads
/// as unanchored -- so this floor is the only thing standing between a chain
/// that breaks again immediately for some reason a flush cannot fix and a loop
/// that clears a live worker's GPU prefix cache on repeat, turning a routing
/// degradation into a throughput collapse.
const FLUSH_COOLDOWN: Duration = Duration::from_secs(300);

/// Spacing after a flush that did *not* land -- refused, errored, or timed out.
///
/// Nothing was discarded, so the full cooldown would be pure delay: it would
/// retire the repair for five minutes over an SGLang scheduler that was busy
/// for one second. Short enough to retry usefully, long enough that a worker
/// answering 404 or refusing forever costs one request every half minute.
const FLUSH_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// Per-engine cache-flush endpoint; mirrors `_FLUSH_PATHS` in
/// `infera/engine/flush.py`. `None` means this engine has no flush to ask for.
fn flush_path(engine: &str) -> Option<&'static str> {
    match engine {
        "vllm" => Some("/reset_prefix_cache"),
        // ATOM re-roots its chain on every cold sequence and emits no clear
        // event, so there is nothing here to repair and no endpoint to repair
        // it with; see the `_FLUSH_PATHS` comment on the Python side.
        "atom" => None,
        _ => Some("/flush_cache"),
    }
}

/// What one worker's flush attempt settled on. Only a flush that actually
/// happened earns the long cooldown.
enum Outcome {
    /// The engine cleared its cache; the chain re-anchors on the next request.
    Flushed,
    /// Refused, unreachable, or never answered. Nothing was discarded.
    Unresolved,
}

/// Per-worker gate on dispatching a flush.
enum Gate {
    /// A POST is out. Its answer decides the cooldown, so nothing else is sent
    /// for this worker until then -- otherwise a 15s flush against a wedged
    /// worker would be re-sent on every orphaned batch in the meantime.
    InFlight,
    /// Nothing is sent for this worker before this instant.
    Until(Instant),
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
    spawn_with(http, FLUSH_COOLDOWN, FLUSH_RETRY_BACKOFF)
}

/// `spawn` with the two spacings injected, so a test can exercise the gate
/// without waiting minutes of wall clock for it.
fn spawn_with(http: reqwest::Client, cooldown: Duration, backoff: Duration) -> FlushRequests {
    let (tx, mut rx) = mpsc::unbounded_channel::<Arc<Worker>>();
    tokio::spawn(async move {
        // Outcomes come back here rather than being recorded at dispatch: what
        // the cooldown has to space out is flushes that *discarded a cache*,
        // and stamping the clock before the POST spends five minutes of it on
        // a request the worker refused or never answered.
        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<(String, Outcome)>();
        let mut gate: HashMap<String, Gate> = HashMap::new();
        loop {
            tokio::select! {
                Some((worker_id, outcome)) = done_rx.recv() => {
                    let wait = match outcome {
                        Outcome::Flushed => cooldown,
                        Outcome::Unresolved => backoff,
                    };
                    gate.insert(worker_id, Gate::Until(Instant::now() + wait));
                }
                req = rx.recv() => {
                    let Some(w) = req else { break }; // router teardown
                    // A worker reachable only over the NATS request envelope may
                    // have no route from here at all, so an HTTP flush would fail
                    // on every retry while looking like an unreachable worker.
                    if w.request_transport != "http" {
                        tracing::info!(
                            worker = %w.worker_id, transport = %w.request_transport,
                            "kv events: chain has no anchor, but this worker is not reachable \
                             over HTTP -- cannot flush its cache, so kv-aware stays load-only \
                             for it"
                        );
                        continue;
                    }
                    let Some(path) = flush_path(&w.engine) else {
                        tracing::info!(
                            worker = %w.worker_id, engine = %w.engine,
                            "kv events: chain reads as unanchored, but this engine has no \
                             cache-flush endpoint -- leaving its cache alone"
                        );
                        continue;
                    };
                    // Expired gates are dropped here so a long-lived router does
                    // not keep an entry for every worker it has ever flushed.
                    let now = Instant::now();
                    gate.retain(|_, g| match g {
                        Gate::InFlight => true,
                        Gate::Until(t) => *t > now,
                    });
                    if gate.contains_key(&w.worker_id) {
                        continue; // already asking, or still cooling down
                    }
                    gate.insert(w.worker_id.clone(), Gate::InFlight);

                    // Spawned rather than awaited: a slow worker must not delay
                    // the flush of the next one, and both are typically broken
                    // at once.
                    let http = http.clone();
                    let done = done_tx.clone();
                    tokio::spawn(async move {
                        let outcome = flush_one(&http, &w, path).await;
                        // A closed receiver means the actor is gone with the
                        // router; there is no cooldown left to record.
                        let _ = done.send((w.worker_id.clone(), outcome));
                    });
                }
            }
        }
    });
    FlushRequests { tx }
}

async fn flush_one(http: &reqwest::Client, w: &Arc<Worker>, path: &str) -> Outcome {
    let url = format!("{}{}", w.url, path);
    tracing::warn!(
        worker = %w.worker_id, %url,
        "kv events: chain never anchored -- flushing this worker's prefix cache so it \
         re-emits a rooted event. Cache locality is lost for the requests that follow, \
         and rebuilt from the next ones."
    );
    match http.post(&url).timeout(FLUSH_TIMEOUT).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!(worker = %w.worker_id, "kv events: cache flushed; chain will re-anchor");
            Outcome::Flushed
        }
        Ok(resp) => {
            // SGLang answers 400 while its scheduler is busy. Not retried here:
            // the next orphaned batch re-arms the request, and a worker serving
            // enough traffic to stay busy is producing those continuously. It
            // gets the short backoff, not the cooldown -- no cache was cleared.
            tracing::warn!(
                worker = %w.worker_id, status = %resp.status(),
                "kv events: cache flush refused; chain stays unanchored for now"
            );
            Outcome::Unresolved
        }
        Err(e) => {
            tracing::warn!(worker = %w.worker_id, err = %e, "kv events: cache flush failed");
            Outcome::Unresolved
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
        assert_eq!(flush_path("sglang"), Some("/flush_cache"));
        assert_eq!(flush_path("vllm"), Some("/reset_prefix_cache"));
        // An unknown engine gets the SGLang endpoint rather than nothing: it is
        // the default deployment, and a wrong 404 is cheaper than never healing.
        assert_eq!(flush_path("something-else"), Some("/flush_cache"));
        // ATOM is the one engine we know has nothing to flush: it re-roots per
        // sequence and emits no clear event. Guessing an endpoint for it only
        // produced a 404 that read here as a worker refusing the flush.
        assert_eq!(flush_path("atom"), None);
    }

    /// An HTTP server that answers `status` to everything and counts requests.
    async fn mock_engine(status: &'static str, hits: Arc<AtomicUsize>) -> String {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = l.accept().await {
                let hits = hits.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = sock.read(&mut buf).await;
                    hits.fetch_add(1, Ordering::SeqCst);
                    let _ = sock
                        .write_all(
                            format!(
                                "HTTP/1.1 {status}\r\ncontent-length: 0\r\n\
                                 connection: close\r\n\r\n"
                            )
                            .as_bytes(),
                        )
                        .await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn worker_at(url: &str) -> Arc<Worker> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "worker_id": "w1", "url": url, "engine": "sglang",
                "request_transport": "http",
            }))
            .unwrap(),
        )
    }

    async fn wait_for(hits: &AtomicUsize, n: usize) {
        for _ in 0..200 {
            if hits.load(Ordering::SeqCst) >= n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "expected {n} flush POSTs, saw {}",
            hits.load(Ordering::SeqCst)
        );
    }

    /// A flush the worker refused discarded nothing, so it must not spend the
    /// cooldown. Stamping the clock at dispatch did exactly that: an SGLang
    /// scheduler that was busy for one second retired the repair for five
    /// minutes, and the log line claiming "the next orphaned batch re-arms the
    /// request" was not true of any batch inside that window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refused_flush_does_not_spend_the_cooldown() {
        let hits = Arc::new(AtomicUsize::new(0));
        let url = mock_engine("400 Bad Request", hits.clone()).await;
        let flush = spawn_with(
            reqwest::Client::new(),
            Duration::from_secs(300),
            Duration::from_millis(50),
        );
        let w = worker_at(&url);

        flush.request(&w);
        wait_for(&hits, 1).await;

        // Inside the short backoff, nothing new goes out.
        flush.request(&w);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(hits.load(Ordering::SeqCst), 1, "asked again too soon");

        // Past it, the ask is honoured again -- the worker was busy, not fixed.
        tokio::time::sleep(Duration::from_millis(80)).await;
        flush.request(&w);
        wait_for(&hits, 2).await;
    }

    /// The converse, and the reason the cooldown exists: a flush that *did*
    /// clear a cache holds the floor, so a chain that breaks again immediately
    /// cannot turn a routing degradation into a throughput collapse.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_flush_that_landed_holds_the_cooldown() {
        let hits = Arc::new(AtomicUsize::new(0));
        let url = mock_engine("200 OK", hits.clone()).await;
        let flush = spawn_with(
            reqwest::Client::new(),
            Duration::from_secs(300),
            Duration::from_millis(1),
        );
        let w = worker_at(&url);

        flush.request(&w);
        wait_for(&hits, 1).await;
        for _ in 0..3 {
            flush.request(&w);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    /// An engine with no flush endpoint is left alone rather than sent a POST
    /// that 404s -- which would come back as `Unresolved` and re-ask forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_engine_with_no_flush_endpoint_is_never_posted_to() {
        let hits = Arc::new(AtomicUsize::new(0));
        let url = mock_engine("200 OK", hits.clone()).await;
        let flush = spawn(reqwest::Client::new());
        flush.request(&Arc::new(
            serde_json::from_value(serde_json::json!({
                "worker_id": "w1", "url": url, "engine": "atom",
                "request_transport": "http",
            }))
            .unwrap(),
        ));
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(hits.load(Ordering::SeqCst), 0);
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
