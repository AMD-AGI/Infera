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
use std::time::{Duration, Instant};

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

/// Health of one rank's event chain.
///
/// Per rank, not per worker, because the chains are: `--dp-size N` gives each
/// attention rank its own radix tree, its own publisher and its own subscriber
/// thread, and they anchor independently. Sharing one set of counters across
/// them fails in both directions — a rank that is applying events masks a dead
/// sibling out of the accounting entirely, and a rank's clear zeroes the
/// counters worker-wide, so the next benign orphan anywhere reads as a chain
/// that never anchored and arms a destructive flush.
#[derive(Default)]
struct ChainHealth {
    /// `Stored` events dropped because `parent_block_hash` resolved to nothing.
    ///
    /// A dropped event never records its own hash in `maps`, so its children
    /// are orphaned in turn: one lost anchor silently kills the whole chain and
    /// the view stays empty for good. From outside that is indistinguishable
    /// from a cold cache -- `/health` is green, requests succeed, and kv-aware
    /// has quietly degenerated into load-only routing. Counting the drops is
    /// what makes the difference observable.
    orphaned: u64,
    /// `Stored` events that did resolve. `applied` stuck at 0 while `orphaned`
    /// climbs is the dead-chain signature; both climbing together is the benign
    /// case of an event racing its parent's eviction.
    applied: u64,
    /// Next `orphaned` value worth a log line. Geometric, so a broken chain
    /// reports itself on the first event and then stops flooding.
    next_warn: u64,
    /// Set when this rank's chain is provably unanchored and clearing the
    /// engine's cache is the only repair. Raised here and consumed elsewhere on
    /// purpose: the events are applied under the global view mutex, on a plain
    /// subscriber thread with no runtime, so this side of the handoff can do no
    /// I/O at all. See `take_flush_request`.
    needs_flush: bool,
    /// When the KV bucket last replaced this rank's view (`seed_rank_view`).
    ///
    /// A rank the bucket is actively mirroring has a working index without a
    /// chain, so it must not be flushed: see `SEED_COVERAGE_WINDOW`.
    seeded_at: Option<Instant>,
}

/// How long a KV-bucket seed counts as covering a rank whose chain is dead.
///
/// The worker-side relay coalesces bucket writes to one per two seconds and
/// only writes a rank its own (anchored) subscriber saw activity on — the same
/// activity that produces the orphaned events here. So while events flow,
/// seeds flow with them, and a lapse this long means the bucket half of the
/// relay is broken rather than merely quiet.
const SEED_COVERAGE_WINDOW: Duration = Duration::from_secs(60);

impl ChainHealth {
    fn new() -> Self {
        ChainHealth {
            next_warn: 1,
            ..Default::default()
        }
    }
}

/// One worker's per-rank cache mirror.
struct WorkerViews {
    block_size: usize,
    views: RankViews,
    maps: RankMaps,
    health: HashMap<i64, ChainHealth>,
}

impl WorkerViews {
    fn new(block_size: usize) -> Self {
        WorkerViews {
            block_size: block_size.max(1),
            views: HashMap::new(),
            maps: HashMap::new(),
            health: HashMap::new(),
        }
    }
}

/// A decoded KV cache event (only the fields we act on).
enum Event {
    Stored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
        /// vLLM's `kv_cache_spec_kind`. `None` on SGLang and on vLLM builds
        /// predating the field; see `is_indexable_spec_kind`.
        spec_kind: Option<String>,
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
    /// True when events arrive over one NATS subscription instead of a ZMQ
    /// socket per worker. Registration still happens per worker -- the views
    /// need a block size to be written into -- but no socket is opened.
    nats_fed: bool,
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
            nats_fed: false,
        }
    }

    /// A client fed by `kv_event_nats`, which writes into the same views.
    pub fn nats_fed() -> Self {
        KvEventClient {
            nats_fed: true,
            ..Self::new()
        }
    }

    /// Decode one worker's event batch and apply it, sharing the decoder and
    /// the view logic with the ZMQ path so the two cannot drift.
    pub(crate) fn apply_encoded_batch(&self, worker_id: &str, rank: i64, payload: &[u8]) {
        match decode_batch(payload) {
            Ok(events) => apply_events(&self.state, worker_id, rank, &events),
            Err(e) => tracing::warn!(worker = %worker_id, err = %e, "kv decode failed (nats)"),
        }
    }

    /// Seed a rank's view from the KV bucket, for a cold start.
    ///
    /// Two guards, both load-bearing. An empty snapshot is never applied: a
    /// desynced relay can publish one, and it would wipe a view built from the
    /// ordered stream and collapse cache hits to zero. And a rank whose own
    /// chain is live keeps the view it built, because the ordered stream is the
    /// authoritative source and the bucket is only a shortcut.
    ///
    /// "Live" is `applied > 0`, not "non-empty" — that distinction is the whole
    /// point. A router starting cold against a rolled JetStream seeds a view it
    /// then cannot maintain: the snapshot is a set of chained router hashes with
    /// no worker-block-hash pairing, so `maps` stays empty, every later event
    /// orphans, and nothing is ever added or evicted. Freezing that first
    /// snapshot in place would leave the router routing on a view that only
    /// drifts further from the worker. Re-seeding keeps it a couple of seconds
    /// behind the relay's own anchored view instead — good enough to route on,
    /// and it is what lets `apply_events` hold off the destructive flush.
    pub(crate) fn seed_rank_view(&self, worker_id: &str, rank: i64, snapshot: Vec<u64>) {
        if snapshot.is_empty() {
            return;
        }
        let mut state = self.state.lock().expect("kv view mutex poisoned");
        let wv = match state.get_mut(worker_id) {
            Some(w) => w,
            None => return, // not tracking this worker (yet)
        };
        let anchored = wv.health.get(&rank).is_some_and(|h| h.applied > 0);
        {
            let view = wv.views.entry(rank).or_default();
            if anchored && !view.is_empty() {
                return;
            }
            *view = snapshot.into_iter().collect();
        }
        let health = wv.health.entry(rank).or_insert_with(ChainHealth::new);
        health.seeded_at = Some(Instant::now());
        // Withdraw any ask raised before this seed landed. `apply_events` only
        // reconsiders when the next batch arrives, so without this a coverage
        // lapse that armed the flush -- an idle spell longer than the window,
        // then one orphaned batch a second ahead of the relay's next write --
        // would stay armed through the refresh that disproves it, and fire
        // whenever the next zero-hit pick happened to come along.
        health.needs_flush = false;
    }

    pub(crate) fn drop_rank_view(&self, worker_id: &str, rank: i64) {
        let mut state = self.state.lock().expect("kv view mutex poisoned");
        if let Some(wv) = state.get_mut(worker_id) {
            wv.views.remove(&rank);
            // The key is gone from the bucket, so the bucket is no longer
            // covering this rank and must not go on suppressing its flush.
            if let Some(h) = wv.health.get_mut(&rank) {
                h.seeded_at = None;
            }
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

    /// Take the pending "this worker's chain needs a cache flush" request, if any.
    ///
    /// Check-and-clear, so a request is delivered to exactly one caller and two
    /// callers cannot both act on the same ask.
    ///
    /// It is *not* one-shot per episode. `apply_events` re-raises the flag on
    /// every batch that still reads as unanchored, which is deliberate: a flush
    /// can be refused (SGLang answers 400 while its scheduler is busy) or never
    /// arrive, and a one-shot ask would then retire the repair permanently for
    /// a worker that is still broken. What keeps that from flushing a worker
    /// repeatedly is the consumer, not this flag — `kv_selfheal` holds one
    /// in-flight POST plus a cooldown per worker. An `AllBlocksCleared` does
    /// stop the asks at the source, by resetting the counters they derive from.
    ///
    /// Any rank asking is enough, and every asking rank is cleared: the engine
    /// exposes one cache-flush endpoint for the whole process, so a single POST
    /// re-anchors all of them and leaving the others armed would only spend the
    /// cooldown flushing a cache that was already cleared.
    pub fn take_flush_request(&self, worker_id: &str) -> bool {
        let mut state = self.state.lock().expect("kv view mutex poisoned");
        let Some(wv) = state.get_mut(worker_id) else {
            return false;
        };
        let mut asked = false;
        for h in wv.health.values_mut() {
            asked |= std::mem::take(&mut h.needs_flush);
        }
        asked
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
        if self.nats_fed {
            // No per-worker socket: ingestion is the one global subscription.
            // A missing block size is not a 1 -- that produces a view which
            // never matches and hides the fault -- so the worker is left
            // untracked and kv-aware routing simply skips it.
            let block_size = match w.kv_block_size {
                Some(bs) if bs > 0 => bs as usize,
                _ => {
                    tracing::error!(
                        worker = %w.worker_id,
                        "kv events (nats): NOT tracking -- it registered no kv_block_size, \
                         so kv-aware routing is off for this worker"
                    );
                    return;
                }
            };
            let mut state = self.state.lock().expect("kv view mutex poisoned");
            state
                .entry(w.worker_id.clone())
                .or_insert_with(|| WorkerViews::new(block_size));
            tracing::info!(worker = %w.worker_id, block_size, "kv events (nats): tracking");
            return;
        }
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

/// Per-socket sequence tracking for the ZMQ transport.
///
/// SGLang's `ZmqEventPublisher` numbers every published batch with a counter
/// that starts at 0 when the publisher process starts, and sends it as an
/// 8-byte big-endian frame between the topic and the payload. Reading it turns
/// two otherwise invisible failures into statements of fact rather than the
/// guesses the downstream orphan counter has to make:
///
/// * a *first* sequence above 0 means this subscription began after the
///   publisher did. Everything before it is unrecoverable -- a PUB socket
///   retains nothing -- and what is in there includes the `parent_block_hash =
///   None` roots that every later event chains to. The view for this rank
///   cannot be built from the live stream at all, no matter how long it runs.
/// * a sequence *below* the last one means the publisher process restarted and
///   its counter began again, so the blocks still indexed for this rank belong
///   to an engine that no longer exists. ZMQ reconnects underneath us without
///   surfacing an error, so this is the only place that transition is visible.
///
/// A gap is detection only -- the one event that can re-anchor a chain from
/// nothing is `AllBlocksCleared`, which only a cache flush produces. A restart
/// is not: it says the indexed blocks describe a process that no longer exists,
/// and the caller drops them (`reset_rank`).
struct SeqTracker {
    /// Sequence of the last batch seen, `None` until the first one arrives.
    last: Option<u64>,
    /// Batches known to have been dropped between two observed sequences.
    lost: u64,
    /// Next `lost` value worth a log line, geometric like the orphan counter so
    /// a subscriber that has fallen permanently behind reports once, not once
    /// per batch.
    next_warn: u64,
}

impl SeqTracker {
    fn new() -> Self {
        Self {
            last: None,
            lost: 0,
            next_warn: 1,
        }
    }

    fn observe(&mut self, worker_id: &str, rank: i64, seq: u64) -> SeqVerdict {
        let prev = match self.last.replace(seq) {
            Some(p) => p,
            None => {
                if seq == 0 {
                    tracing::info!(
                        worker = %worker_id, rank,
                        "kv events: subscribed from the publisher's first batch"
                    );
                } else {
                    tracing::warn!(
                        worker = %worker_id, rank, first_seq = seq,
                        "kv events: subscribed mid-stream -- the batches before this one \
                         were published before the subscription existed and a PUB socket \
                         retains nothing; they hold the rooted events every later event \
                         chains to, so this rank's view cannot be built from the live \
                         stream and kv-aware will route on load alone until the worker's \
                         cache is flushed"
                    );
                }
                return SeqVerdict::Continue;
            }
        };
        if seq == prev.wrapping_add(1) {
            return SeqVerdict::Continue;
        }
        if seq <= prev {
            tracing::warn!(
                worker = %worker_id, rank, prev_seq = prev, seq,
                "kv events: sequence restarted -- the publisher process was replaced, so \
                 the blocks still indexed for this rank belonged to an engine that no longer \
                 exists and have been dropped; kv-aware reports no hits for this rank until \
                 the new engine's own events refill it"
            );
            self.lost = 0;
            self.next_warn = 1;
            return SeqVerdict::PublisherRestarted;
        }
        let missed = seq - prev - 1;
        self.lost += missed;
        if self.lost >= self.next_warn {
            tracing::warn!(
                worker = %worker_id, rank, missed, lost_total = self.lost, seq,
                "kv events: batches dropped in transit (subscriber past the publisher's \
                 high-water mark, or a silent reconnect); every block descending from a \
                 store inside the gap is orphaned from here on"
            );
            self.next_warn = self.lost.saturating_mul(10);
        }
        SeqVerdict::Continue
    }
}

/// What `observe` concluded about where a batch sits in the publisher's stream.
#[derive(Debug, PartialEq, Eq)]
enum SeqVerdict {
    /// Contiguous, or a gap: the view is still the old engine's and still true
    /// as far as it goes.
    Continue,
    /// The counter went backwards, so the publisher process was replaced.
    PublisherRestarted,
}

/// Drop everything indexed for one rank, the way its own `AllBlocksCleared`
/// would.
///
/// Leaving a dead engine's blocks in place is worse than having no view at all.
/// They are reported as prefix hits, so the affected prompts are steered at the
/// one worker whose cache is certainly cold -- and those same phantom hits keep
/// `applied > 0` and reset the zero-hit streak, so neither the chain-health
/// alarm nor the self-heal ever notices. Forgetting them costs the router
/// nothing it can trust: the new process starts with an empty cache, and its
/// first events are rooted, so the chain re-anchors on its own.
fn reset_rank(state: &Arc<Mutex<HashMap<String, WorkerViews>>>, worker_id: &str, rank: i64) {
    let mut guard = state.lock().expect("kv view mutex poisoned");
    let Some(wv) = guard.get_mut(worker_id) else {
        return;
    };
    wv.views.entry(rank).or_default().clear();
    wv.maps.entry(rank).or_default().clear();
    // Counters too: they describe the old process's chain, and carrying its
    // orphan total into a rank whose view is now legitimately empty would arm a
    // flush against an engine that has nothing left to flush.
    wv.health.insert(rank, ChainHealth::new());
}

/// Split a published message into `(sequence, payload)`.
///
/// SGLang sends `(topic, seq, payload)`; vLLM and the crate's own fixtures send
/// `(topic, payload)`. The payload is the last frame either way -- taking it
/// from the end is what has always made both shapes work -- and the sequence is
/// read only when the middle frame is present and is the eight bytes the
/// publisher writes. An unrecognised shape degrades to no sequence tracking
/// rather than to a dropped batch.
fn split_frames(frames: &[Vec<u8>]) -> Option<(Option<u64>, &[u8])> {
    let payload = frames.last()?;
    let seq = match frames.len() {
        3 => <[u8; 8]>::try_from(frames[1].as_slice())
            .ok()
            .map(u64::from_be_bytes),
        _ => None,
    };
    Some((seq, payload.as_slice()))
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
    // Outlives the socket: a reconnect is itself a gap, and reporting it needs
    // the sequence observed before the socket was torn down.
    let mut seq = SeqTracker::new();
    while !stop.load(Ordering::Relaxed) {
        match subscribe_once(&ctx, &state, &worker_id, rank, &endpoint, &stop, &mut seq) {
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
    seq: &mut SeqTracker,
) -> Result<(), zmq::Error> {
    let sock = ctx.socket(zmq::SUB)?;
    sock.set_rcvtimeo(RECV_TIMEOUT_MS)?;
    sock.connect(endpoint)?;
    sock.set_subscribe(TOPIC)?;
    while !stop.load(Ordering::Relaxed) {
        match sock.recv_multipart(0) {
            Ok(frames) => {
                if let Some((n, payload)) = split_frames(&frames) {
                    if let Some(n) = n {
                        if seq.observe(worker_id, rank, n) == SeqVerdict::PublisherRestarted {
                            reset_rank(state, worker_id, rank);
                        }
                    }
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
    let (mut orphaned, mut applied, mut cleared) = (0u64, 0u64, false);
    for ev in events {
        match ev {
            Event::Stored {
                block_hashes,
                parent_block_hash,
                token_ids,
                spec_kind,
            } => {
                if !is_indexable_spec_kind(spec_kind.as_deref()) {
                    continue;
                }
                // Index every block the token span covers, but only trust a
                // block hash when the two lengths agree.
                //
                // Measured on a hybrid model (Qwen3.5-0.8B, the same shapes
                // Kimi-K3 emits): dropping a length-disagreeing event whole
                // costs more than it saves -- 18/32 requests hit a full prefix
                // versus 22/32 when the leading blocks are indexed. The view
                // entries ARE correct: chained from a resolved parent over a
                // contiguous span. The hash-to-block PAIRING is what is not.
                //
                // So fill the view and skip the map. `map` is what a later
                // event resolves its parent against, and on a sparse event the
                // surviving hash need not describe the leading chunk -- vLLM
                // gives no offset. Writing it binds an engine hash to the wrong
                // block and poisons the chain from there; withholding it makes
                // a later event miss its parent and be dropped, which
                // under-reports hits instead of mis-reporting them.
                let n = token_ids.len() / bs;
                let aligned = n == block_hashes.len();
                let mut parent = match parent_block_hash {
                    None => ROUTER_SEED,
                    Some(ph) => match map.get(ph) {
                        Some(rh) => *rh,
                        None => {
                            orphaned += 1;
                            continue; // chain broken: missing parent, drop
                        }
                    },
                };
                applied += 1;
                for i in 0..n {
                    let chunk = &token_ids[i * bs..(i + 1) * bs];
                    parent = hash_chunk(parent, chunk);
                    view.insert(parent);
                    if aligned {
                        map.insert(block_hashes[i], parent);
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
                cleared = true;
                // Drop what this batch counted *before* the clear along with
                // the chain it described. The reset below only zeroes the
                // running totals, so leaving these to be added back after it
                // would leave `applied == 0 && orphaned > 0` -- arming a
                // destructive flush against a chain that re-anchored this
                // instant, and silently, since `next_warn` is reset too.
                orphaned = 0;
                applied = 0;
            }
        }
    }

    // The `views`/`maps` borrows end above, so the counters are reachable again.
    let indexed = wv.views.get(&rank).map_or(0, |v| v.len());
    let health = wv.health.entry(rank).or_insert_with(ChainHealth::new);
    if cleared {
        // This rank's chain just re-anchored on the root; whatever was dropped
        // before that describes a state this one no longer shares. Only this
        // rank's: the other ranks' chains are untouched by it, and zeroing
        // theirs would forget a sibling that is genuinely dead.
        *health = ChainHealth::new();
    }
    health.orphaned += orphaned;
    health.applied += applied;
    // `applied == 0` alongside a non-zero `orphaned` is the one unambiguous
    // reading: not a single store event has ever been placed for this rank, so
    // the router is not missing part of a chain, it never had the anchor.
    // Only `AllBlocksCleared` rebuilds one, and only the worker can emit that,
    // so the repair has to be asked for. The narrow condition matters because
    // the ask is destructive -- it discards real GPU prefix cache -- and the
    // benign case (an event racing its parent's eviction) climbs both counters
    // together and is excluded by construction.
    //
    // Unless the KV bucket is covering this rank. A router that started cold
    // against a rolled JetStream reads exactly like a dead chain -- it has no
    // `maps` to resolve any parent against -- but `seed_rank_view` is meanwhile
    // replacing its view from the relay's own anchored mirror every couple of
    // seconds. That is a working index, and trading the worker's real GPU
    // prefix cache for a marginally fresher one is not a trade worth making.
    // The window is what keeps this honest: a relay whose bucket writer died
    // (it logs, and keeps forwarding) stops refreshing, coverage lapses, and
    // the flush arms on the next orphaned batch.
    let bucket_covered = health
        .seeded_at
        .is_some_and(|t| t.elapsed() < SEED_COVERAGE_WINDOW);
    // Armed on *this batch's* orphans, not the running total, and withdrawn
    // again as soon as either half of the reading stops holding. Reading the
    // total re-armed on batches that orphaned nothing, and nothing ever lowered
    // the flag, so a request raised during a momentary coverage lapse outlived
    // every piece of evidence for it and fired at the next zero-hit pick --
    // which, on an idle rank, is minutes or hours later against a worker whose
    // cache is fine. A quiet batch is left alone: only evidence to the contrary
    // withdraws the ask, not the mere absence of new evidence for it.
    if orphaned > 0 && health.applied == 0 && !bucket_covered {
        health.needs_flush = true;
    } else if bucket_covered || health.applied > 0 {
        health.needs_flush = false;
    }
    if orphaned > 0 && health.orphaned >= health.next_warn {
        tracing::warn!(
            worker = %worker_id,
            rank,
            orphaned = health.orphaned,
            applied = health.applied,
            indexed_blocks = indexed,
            "kv events: dropped store events whose parent was never seen; while \
             `applied` stays 0 the chain has lost its anchor and every later \
             event is dropped with it, leaving kv-aware to route on load alone"
        );
        health.next_warn = health.orphaned.saturating_mul(10);
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
            // SGLang's array form has no group fields; None => fail open.
            spec_kind: None,
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

/// Which KV-cache groups carry a usable hash-per-block.
///
/// vLLM emits one `BlockStored` per group. On a hybrid model -- Kimi-K3 is 3
/// KDA/Mamba groups plus 1 MLA group -- the Mamba groups run prefix caching in
/// "align" mode, where all but one block per step is a null block skipped when
/// the hash list is built, while `token_ids` still spans the whole range.
/// Measured: Mamba groups report 3840 tokens against ONE hash at
/// block_size=768; the MLA group reports 3840 against five.
///
/// Those events cannot be indexed -- nothing says which chunk the surviving
/// hash covers. They also actively corrupt the view: vLLM's block hash does not
/// mix in the group id, so at equal block sizes a Mamba hash collides with an
/// attention hash and overwrites its entry in the engine-hash -> router-hash
/// map, breaking the parent chain for every block after it.
///
/// `mla_attention` is mandatory in this set. Kimi-K3's attention layers are
/// MLA, so a filter written as `== "full_attention"` would drop 100% of its
/// usable events -- the same empty view, reached from the other side.
///
/// `None` means SGLang or a vLLM build predating the field: fail OPEN, since
/// those streams were indexed before this filter existed. Upstream: vllm#44451.
fn is_indexable_spec_kind(kind: Option<&str>) -> bool {
    match kind {
        None => true,
        Some(k) => matches!(
            k,
            "full_attention" | "mla_attention" | "sink_full_attention"
        ),
    }
}

/// vLLM tagged-MAP event: `{"type": <tag>, <field>: <value>, ...}` (msgspec
/// `tag=True` map; the tag key is "type"). Fields are matched by NAME.
///
/// `kv_cache_spec_kind` IS read: vLLM emits one event per KV-cache group and
/// only the attention groups pair one hash with one block. Ignoring it, as this
/// did, meant a hybrid model's Mamba groups were indexed too -- see
/// `is_indexable_spec_kind`.
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
            spec_kind: get("kv_cache_spec_kind")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
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

/// One flat token id from a `BlockStored`'s `token_ids`, in either view the
/// engine may report. Under EAGLE/MTP, SGLang keys its radix tree on bigrams and
/// sends each block's tokens as the overlapping pairs `(t[i], t[i+1])`; the first
/// element of each pair rebuilds `t[start:end]`, which is the flat slice
/// `hash_request` chunks on the query side (radix nodes split on page boundaries,
/// so the two chunkings stay aligned). Read as ints, a pair is not an integer, so
/// every element would be dropped and the view would silently stay empty.
fn as_u32_any(v: &rmpv::Value) -> Option<u32> {
    match v {
        rmpv::Value::Array(pair) => pair.first().and_then(as_u64_any).map(|n| n as u32),
        _ => as_u64_any(v).map(|n| n as u32),
    }
}

fn as_u32_vec(v: &rmpv::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| a.iter().filter_map(as_u32_any).collect())
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
    fn decodes_sglang_bigram_batch_under_mtp() {
        // With MTP/EAGLE the radix key is a bigram view, so token_ids arrives as
        // overlapping (t[i], t[i+1]) PAIRS. The view must still hash to what the
        // query side computes over the flat tokens, or kv-aware never hits.
        let c = KvEventClient::new();
        c.on_worker_added(&worker("sgl", Some("tcp://127.0.0.1:6005"), 16, Some(2)));
        // 16 bigrams over flat tokens 1..=17 -> one block whose first elements
        // are exactly 1..=16.
        let pairs: Vec<Mv> = (1u32..=16)
            .map(|t| Mv::Array(vec![Mv::from(t), Mv::from(t + 1)]))
            .collect();
        let stored = Mv::Array(vec![
            Mv::String("BlockStored".into()),
            ints(&[910]),
            Mv::Nil,
            Mv::Array(pairs),
            Mv::from(16i64),
            Mv::Nil,
        ]);
        let batch = enc(Mv::Array(vec![
            Mv::from(1.0),
            Mv::Array(vec![stored]),
            Mv::from(0i64),
        ]));
        feed_wire(&c, "sgl", 0, &batch);
        let q = crate::hasher::hash_request(&seq(1, 16), 16);
        assert_eq!(q.len(), 1);
        assert_eq!(
            c.prefix_hits("sgl", Some(0), &q),
            1,
            "bigram pairs must hash like the flat token slice"
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

    /// vLLM emits one BlockStored per KV-cache GROUP. Only the attention groups
    /// pair one hash with one block; a hybrid model's Mamba groups report the
    /// whole token span against a single hash (measured on Kimi-K3: 3840 tokens,
    /// 1 hash, block_size 768, against the MLA group's 3840/5).
    #[test]
    fn non_attention_groups_are_not_indexed() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5601"), 4, None));

        for kind in ["mamba", "sliding_window", "encoder_only_attention"] {
            apply_events(
                &c.state,
                "w",
                0,
                &[Event::Stored {
                    block_hashes: vec![111, 222],
                    parent_block_hash: None,
                    spec_kind: Some(kind.to_string()),
                    token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
                }],
            );
        }
        let q = crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(c.prefix_hits("w", None, &q), 0);
        c.shutdown();
    }

    /// `mla_attention` is mandatory in the allowed set: Kimi-K3's attention
    /// layers are MLA, so a filter written as `== "full_attention"` would drop
    /// 100% of that model's usable events.
    #[test]
    fn attention_groups_are_indexed() {
        for kind in ["full_attention", "mla_attention", "sink_full_attention"] {
            let c = KvEventClient::new();
            c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5602"), 4, None));
            apply_events(
                &c.state,
                "w",
                0,
                &[Event::Stored {
                    block_hashes: vec![111, 222],
                    parent_block_hash: None,
                    spec_kind: Some(kind.to_string()),
                    token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
                }],
            );
            let q = crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
            assert_eq!(c.prefix_hits("w", None, &q), 2, "kind={kind}");
            c.shutdown();
        }
    }

    /// A sparse event -- more token blocks than hashes -- still contributes its
    /// blocks to the view; only the hashes are withheld, because nothing says
    /// which chunk the surviving one describes. Measured: dropping the event
    /// whole cut full-prefix hits from 22/32 to 18/32 on a hybrid model.
    #[test]
    fn sparse_event_indexes_blocks_but_not_hashes() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5603"), 4, None));
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![111], // one hash for two blocks of tokens
                parent_block_hash: None,
                spec_kind: Some("mla_attention".to_string()),
                token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }],
        );
        let q = crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(c.prefix_hits("w", None, &q), 2, "blocks are visible");

        // The hash was not mapped, so a child naming it as parent is dropped
        // rather than chained off a block it may not describe.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![222],
                parent_block_hash: Some(111),
                spec_kind: Some("mla_attention".to_string()),
                token_ids: vec![9, 10, 11, 12],
            }],
        );
        let q2 = crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4);
        assert_eq!(
            c.prefix_hits("w", None, &q2),
            2,
            "child dropped, not mis-chained"
        );
        c.shutdown();
    }

    /// The property the filter exists for. Both groups get the SAME block hash
    /// (vLLM mixes no group id in), so a Mamba event arriving AFTER the
    /// attention event overwrites `map[hash] -> router_hash` with a hash over
    /// its own tokens; the next chunk then chains off the wrong node and the
    /// view holds a block no query can reproduce.
    #[test]
    fn a_later_mamba_event_cannot_poison_the_attention_chain() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5604"), 4, None));

        let stored = |hashes: Vec<u64>, parent, kind: &str, toks: Vec<u32>| Event::Stored {
            block_hashes: hashes,
            parent_block_hash: parent,
            spec_kind: Some(kind.to_string()),
            token_ids: toks,
        };
        apply_events(
            &c.state,
            "w",
            0,
            &[stored(
                vec![111, 222],
                None,
                "mla_attention",
                vec![1, 2, 3, 4, 5, 6, 7, 8],
            )],
        );
        // same hashes, different tokens, arriving second
        apply_events(
            &c.state,
            "w",
            0,
            &[stored(
                vec![111, 222],
                None,
                "mamba",
                vec![90, 91, 92, 93, 94, 95, 96, 97],
            )],
        );
        // follow-on chunk, parented on the attention group's second block
        apply_events(
            &c.state,
            "w",
            0,
            &[stored(
                vec![333],
                Some(222),
                "mla_attention",
                vec![9, 10, 11, 12],
            )],
        );

        let q = crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4);
        assert_eq!(q.len(), 3);
        assert_eq!(
            c.prefix_hits("w", None, &q),
            3,
            "a later non-attention event overwrote the attention group's hash map"
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
                spec_kind: None,
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
                spec_kind: None,
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

    /// The never-anchored chain asks to be repaired, and asks exactly once.
    ///
    /// `applied == 0 && orphaned > 0` is the whole gate, and it has to stay
    /// narrow: the request discards a live worker's GPU prefix cache, so a
    /// chain that is merely lossy must not trip it.
    #[test]
    fn a_chain_that_never_anchored_asks_for_a_flush() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5570"), 4, None));
        assert!(
            !c.take_flush_request("w"),
            "a worker that has seen no events has nothing to repair"
        );

        // An event naming a parent nobody ever stored: the anchor is gone.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert!(c.take_flush_request("w"), "an unanchored chain must ask");
        assert!(
            !c.take_flush_request("w"),
            "and the request is consumed, so two readers cannot both flush"
        );

        // Consumed is not the same as answered. A flush can be refused or never
        // arrive, so a chain that still reads as dead asks again on the next
        // batch; `kv_selfheal` is what spaces the POSTs out.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![11],
                parent_block_hash: Some(998),
                token_ids: vec![5, 6, 7, 8],
                spec_kind: None,
            }],
        );
        assert!(
            c.take_flush_request("w"),
            "a still-dead chain must re-ask, or one refused flush retires the repair"
        );
    }

    /// The KV bucket is a working index without a chain, so it holds the flush.
    ///
    /// A router that starts cold against a rolled JetStream has no `maps` and
    /// orphans every event -- indistinguishable from a dead chain by the
    /// counters alone. But the worker-side relay keeps mirroring its own
    /// anchored view into the bucket, and `seed_rank_view` keeps replacing this
    /// one from it. Flushing there would discard the worker's real GPU prefix
    /// cache to fix an index that is already being kept current.
    #[test]
    fn a_rank_the_bucket_is_mirroring_does_not_ask_for_a_flush() {
        let c = KvEventClient::nats_fed();
        c.on_worker_added(&worker("w", None, 4, None));
        c.seed_rank_view("w", 0, vec![7, 8, 9]);

        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert!(
            !c.take_flush_request("w"),
            "a bucket-backed rank routes fine; the flush would be pure loss"
        );
        assert_eq!(c.total_blocks("w"), 3, "and the seeded view is still there");

        // The other rank of the same worker has no bucket key, so nothing is
        // covering it and it must still ask.
        apply_events(
            &c.state,
            "w",
            1,
            &[Event::Stored {
                block_hashes: vec![20],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert!(
            c.take_flush_request("w"),
            "coverage is per rank: an unmirrored rank is still a dead chain"
        );
    }

    /// The bucket refreshing a rank withdraws a request raised before it.
    ///
    /// The failure this closes: on NATS transport, a router that restarted
    /// while the workers kept serving goes quiet for longer than
    /// `SEED_COVERAGE_WINDOW`, so coverage lapses. Traffic resumes, the first
    /// batch orphans and arms the flush a second or so ahead of the relay's
    /// next (<=2s-coalesced) bucket write -- and the refresh that disproves it
    /// arrives too late to matter, because nothing reconsidered the flag. The
    /// request then sat armed with no expiry until some later zero-hit pick
    /// spent it on a worker whose cache was fine.
    #[test]
    fn a_returning_bucket_withdraws_the_request_the_lapse_armed() {
        let c = KvEventClient::nats_fed();
        c.on_worker_added(&worker("w", None, 4, None));
        // Coverage has lapsed (no seed yet) and a batch orphans: armed.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        // The relay's next bucket write lands.
        c.seed_rank_view("w", 0, vec![7, 8, 9]);
        assert!(
            !c.take_flush_request("w"),
            "the seed is the evidence the rank is covered; the ask predates it"
        );
    }

    /// Arming reads *this batch's* orphans, not the running total.
    ///
    /// Reading the total re-armed on every later batch, including ones that
    /// orphaned nothing -- so a single lapse's worth of orphans kept the flush
    /// armed indefinitely, and coverage returning could never settle it.
    #[test]
    fn a_batch_that_orphaned_nothing_does_not_re_arm_the_request() {
        let c = KvEventClient::nats_fed();
        c.on_worker_added(&worker("w", None, 4, None));
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert!(c.take_flush_request("w"), "the orphaned batch asks");

        // Consumed. A batch carrying nothing that orphans -- here an eviction
        // for a block the router never indexed -- is not fresh evidence, so it
        // must not raise the ask again behind the consumer's back.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Removed {
                block_hashes: vec![777],
            }],
        );
        assert!(!c.take_flush_request("w"));

        // A chain that is still dead does re-ask on its next orphan, though --
        // the flush it asked for may have been refused.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![11],
                parent_block_hash: Some(999),
                token_ids: vec![5, 6, 7, 8],
                spec_kind: None,
            }],
        );
        assert!(c.take_flush_request("w"));
    }

    /// A bucket key that goes away takes its cover with it.
    #[test]
    fn dropping_a_rank_view_stops_holding_back_its_flush() {
        let c = KvEventClient::nats_fed();
        c.on_worker_added(&worker("w", None, 4, None));
        c.seed_rank_view("w", 0, vec![7, 8, 9]);
        c.drop_rank_view("w", 0);

        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert!(
            c.take_flush_request("w"),
            "the relay stopped publishing this rank, so nothing is keeping it current"
        );
    }

    /// The bucket seeds a cold rank repeatedly, but never corrects a live one.
    #[test]
    fn the_bucket_refreshes_a_dead_rank_and_defers_to_a_live_one() {
        let c = KvEventClient::nats_fed();
        c.on_worker_added(&worker("w", None, 4, None));

        // Nothing has ever resolved on this rank, so the bucket is all it has:
        // freezing the first snapshot would leave it drifting from the worker.
        c.seed_rank_view("w", 0, vec![1, 2, 3]);
        c.seed_rank_view("w", 0, vec![4, 5]);
        assert_eq!(c.total_blocks("w"), 2, "a dead rank tracks the bucket");

        // Rank 1 builds its own view from the ordered stream. That one is
        // authoritative and the bucket must not walk over it.
        apply_events(
            &c.state,
            "w",
            1,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert_eq!(c.total_blocks("w"), 3);
        c.seed_rank_view("w", 1, vec![90, 91, 92, 93]);
        assert_eq!(
            c.total_blocks("w"),
            3,
            "a live chain keeps the view it built"
        );
    }

    #[test]
    fn a_merely_lossy_chain_does_not_ask_for_a_flush() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5571"), 4, None));
        // A rooted event lands first, so the chain provably has its anchor.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        // A later event racing its parent's eviction is ordinary, and flushing
        // over it would throw away the working cache this worker does hold.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![11],
                parent_block_hash: Some(999),
                token_ids: vec![5, 6, 7, 8],
                spec_kind: None,
            }],
        );
        assert!(
            !c.take_flush_request("w"),
            "orphans alongside a live anchor are benign; only applied == 0 is the fault"
        );
    }

    #[test]
    fn a_clear_withdraws_a_pending_flush_request() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5572"), 4, None));
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: Some(999),
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        // The flush landed by another route (a worker-side flush on startup, or
        // an operator). The repair already happened, so the ask is stale --
        // acting on it would clear a cache that has just been rebuilt.
        apply_events(&c.state, "w", 0, &[Event::Cleared]);
        assert!(!c.take_flush_request("w"));
    }

    /// A dead rank still asks, even while a sibling is healthy.
    ///
    /// `--dp-size N` gives each attention rank its own radix tree, publisher and
    /// subscriber thread, so the anchors are lost or kept independently. With
    /// the accounting shared across ranks, one rank applying events keeps
    /// `applied > 0` forever and the dead rank's orphans never read as a fault
    /// -- the worker serves half its requests off an index that will never fill
    /// and nothing ever says so.
    /// A publisher restart drops what the dead engine had indexed.
    ///
    /// Leaving it in place was worse than an empty view: the stale blocks are
    /// reported as prefix hits, so the matching prompts are steered at the one
    /// worker whose cache is certainly cold, and those phantom hits keep
    /// `applied > 0` and reset the zero-hit streak -- so neither the chain
    /// alarm nor the self-heal ever fired. The condition was detected and
    /// logged; nothing acted on it.
    #[test]
    fn a_publisher_restart_drops_the_dead_engines_blocks() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5590"), 4, None));
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        assert_eq!(c.total_blocks("w"), 1);

        let mut seq = SeqTracker::new();
        assert_eq!(seq.observe("w", 0, 7), SeqVerdict::Continue);
        assert_eq!(seq.observe("w", 0, 8), SeqVerdict::Continue);
        // A gap is not a restart: the blocks indexed so far are still this
        // engine's, and dropping them would throw away a working view.
        assert_eq!(seq.observe("w", 0, 40), SeqVerdict::Continue);
        reset_rank_if(&c, seq.observe("w", 0, 0));
        assert_eq!(
            c.total_blocks("w"),
            0,
            "the counter went backwards: those blocks are a process that is gone"
        );

        // And the counters went with them, so the empty view does not read as a
        // dead chain and arm a flush against an engine with nothing to flush.
        assert!(!c.take_flush_request("w"));

        // The new process's first events are rooted, so the chain re-anchors.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: None,
                token_ids: vec![9, 9, 9, 9],
                spec_kind: None,
            }],
        );
        assert_eq!(c.total_blocks("w"), 1);
        assert!(!c.take_flush_request("w"));
    }

    /// What `subscribe_once` does with a verdict, minus the socket.
    fn reset_rank_if(c: &KvEventClient, v: SeqVerdict) {
        if v == SeqVerdict::PublisherRestarted {
            reset_rank(&c.state, "w", 0);
        }
    }

    #[test]
    fn a_dead_rank_asks_even_though_another_rank_is_healthy() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5574"), 4, None));
        // Rank 0 has its anchor.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        // Rank 1 joined after its own anchor was published to nobody.
        for i in 0..5u64 {
            apply_events(
                &c.state,
                "w",
                1,
                &[Event::Stored {
                    block_hashes: vec![100 + i],
                    parent_block_hash: Some(999),
                    token_ids: vec![5, 6, 7, 8],
                    spec_kind: None,
                }],
            );
        }
        assert!(
            c.take_flush_request("w"),
            "a rank that never applied an event is a dead chain regardless of its siblings"
        );
        assert!(!c.take_flush_request("w"), "and the ask is consumed");
    }

    /// The other direction: one rank's clear must not license a flush that
    /// another rank's benign orphan asks for.
    ///
    /// The engine's flush endpoint is process-wide, so acting on this would
    /// discard every rank's prefix cache to repair a chain that is not broken.
    #[test]
    fn a_clear_on_one_rank_does_not_reset_another_ranks_accounting() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5575"), 4, None));
        // Rank 1 is anchored and healthy.
        apply_events(
            &c.state,
            "w",
            1,
            &[Event::Stored {
                block_hashes: vec![10],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                spec_kind: None,
            }],
        );
        // Rank 0 clears -- its own accounting resets, and only its own.
        apply_events(&c.state, "w", 0, &[Event::Cleared]);
        // An ordinary eviction race on rank 1, which still has its anchor.
        apply_events(
            &c.state,
            "w",
            1,
            &[Event::Stored {
                block_hashes: vec![11],
                parent_block_hash: Some(999),
                token_ids: vec![5, 6, 7, 8],
                spec_kind: None,
            }],
        );
        assert!(
            !c.take_flush_request("w"),
            "rank 1 kept its anchor; a sibling's clear must not make its orphan look fatal"
        );
    }

    /// The same withdrawal, with the clear inside the batch rather than after
    /// it — which is the shape the wire actually carries.
    ///
    /// SGLang batches everything one scheduler step produced, so an orphan and
    /// the clear that answers it arrive together far more often than they
    /// arrive apart. Counting the orphan against the chain that replaced it
    /// would flush a worker whose chain is healthy as of the same batch, and
    /// the clear resets the warn threshold, so it would do it without a log.
    #[test]
    fn a_clear_later_in_the_same_batch_withdraws_the_request_too() {
        let c = KvEventClient::new();
        c.on_worker_added(&worker("w", Some("tcp://127.0.0.1:5573"), 4, None));
        apply_events(
            &c.state,
            "w",
            0,
            &[
                Event::Stored {
                    block_hashes: vec![10],
                    parent_block_hash: Some(999),
                    token_ids: vec![1, 2, 3, 4],
                    spec_kind: None,
                },
                Event::Cleared,
            ],
        );
        assert!(
            !c.take_flush_request("w"),
            "the orphan describes the chain the clear just discarded"
        );

        // And the accounting is genuinely reset, not merely masked: a rooted
        // event after the clear still reads as an anchored chain.
        apply_events(
            &c.state,
            "w",
            0,
            &[Event::Stored {
                block_hashes: vec![11],
                parent_block_hash: None,
                token_ids: vec![5, 6, 7, 8],
                spec_kind: None,
            }],
        );
        assert!(!c.take_flush_request("w"));
        assert_eq!(c.total_blocks("w"), 1);
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
                spec_kind: None,
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

    // ---- sequence frame ---------------------------------------------------

    fn frames(parts: &[&[u8]]) -> Vec<Vec<u8>> {
        parts.iter().map(|p| p.to_vec()).collect()
    }

    /// The payload must come off the end for both wire shapes. Reading the
    /// sequence by position is what would break vLLM and the Atom fixtures, so
    /// the two-frame form has to keep decoding with tracking simply switched off.
    #[test]
    fn split_frames_reads_sglang_sequence_and_leaves_two_frame_publishers_alone() {
        let sgl = frames(&[TOPIC, &7u64.to_be_bytes(), b"payload"]);
        assert_eq!(split_frames(&sgl), Some((Some(7), b"payload".as_ref())));

        let vllm = frames(&[TOPIC, b"payload"]);
        assert_eq!(split_frames(&vllm), Some((None, b"payload".as_ref())));

        // A middle frame that is not the publisher's 8 bytes is not a sequence,
        // and must not cost us the batch.
        let odd = frames(&[TOPIC, b"xx", b"payload"]);
        assert_eq!(split_frames(&odd), Some((None, b"payload".as_ref())));

        assert_eq!(split_frames(&[]), None);
    }

    /// A publisher counter starts at 0, so a higher first sequence is proof the
    /// subscription missed the rooted events and not merely a suspicion drawn
    /// from a run of orphans. Contiguity, gaps and a restart are the other three
    /// states; none of them may panic or wrap.
    #[test]
    fn seq_tracker_classifies_first_batch_gaps_and_restart() {
        let mut t = SeqTracker::new();
        t.observe("w", 0, 0); // clean start
        t.observe("w", 0, 1); // contiguous
        assert_eq!(t.lost, 0);

        t.observe("w", 0, 5); // 2,3,4 dropped
        assert_eq!(t.lost, 3);
        t.observe("w", 0, 9); // 6,7,8 dropped
        assert_eq!(t.lost, 6);

        t.observe("w", 0, 0); // publisher replaced: counter began again
        assert_eq!(t.last, Some(0));
        assert_eq!(t.lost, 0, "a new publisher's history is not the old one's");

        // Joining mid-stream is the cold-start case and is not a gap: there is
        // no previous sequence for those batches to be missing from.
        let mut late = SeqTracker::new();
        late.observe("w", 0, 4_000);
        assert_eq!((late.last, late.lost), (Some(4_000), 0));
    }
}
