///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Per-worker circuit breaker. Mirrors `infera/router/breaker.py` — same three
//! states, same thresholds, same all-open fallback — so the two data planes
//! behave identically under a wedged worker.
//!
//! Failover on its own is not enough. It retries a failed dispatch elsewhere,
//! but the memory of that failure lives in a per-request `tried` set that is
//! dropped when the request returns, so the next request scores the same broken
//! worker as if nothing had happened. A worker that answers `/health`, stays
//! ACTIVE in etcd, and fails before the first byte therefore taxes *every*
//! request, indefinitely.
//!
//! Unlike the Python side this is shared across tokio worker threads, so the
//! map lives behind a `Mutex`. The critical sections are a hash lookup and a
//! few integer writes; contention is not a concern at any plausible request
//! rate, and a lock-free design here would buy nothing for the complexity.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Statuses that are evidence about the *worker*, not the request.
///
/// Failover retries on any pre-first-byte error including 4xx, which is
/// correct — re-asking costs nothing. Feeding 4xx to the breaker is not: a
/// malformed request returns 400 from every worker it touches, so one bad
/// client would trip the whole healthy fleet. 429 is excluded for a different
/// reason: it means "full right now", which load accounting already routes
/// around, and a doubling cooldown is far too heavy for transient backpressure.
pub fn is_worker_fault(status: u16) -> bool {
    status >= 500 || status == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl BreakerState {
    pub fn as_str(self) -> &'static str {
        match self {
            BreakerState::Closed => "closed",
            BreakerState::Open => "open",
            BreakerState::HalfOpen => "half_open",
        }
    }
}

#[derive(Debug)]
struct Entry {
    consecutive_failures: u32,
    state: BreakerState,
    /// Instant after which an open breaker becomes half-open.
    opens_until: Instant,
    /// Cooldown applied on the *next* trip; doubles each time a probe fails.
    next_cooldown: Duration,
    /// Set while a half-open probe is in flight, so only one is admitted.
    probe_in_flight: bool,
    trips: u64,
}

pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    max_cooldown: Duration,
    entries: Mutex<HashMap<String, Entry>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown: Duration, max_cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            max_cooldown,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Whether this worker may be dispatched to right now.
    ///
    /// Transitions Open -> HalfOpen as a side effect once the cooldown has
    /// elapsed, because the alternative is a background timer whose only job is
    /// to flip a flag this function already has to read. Call it once per
    /// candidate per request: in HalfOpen it *consumes* the single probe slot.
    pub fn allows(&self, worker_id: &str) -> bool {
        self.allows_at(worker_id, Instant::now())
    }

    /// A threshold of 0 turns the breaker off entirely, so an operator can fall
    /// back to plain failover without a code change.
    fn enabled(&self) -> bool {
        self.failure_threshold > 0
    }

    fn allows_at(&self, worker_id: &str, now: Instant) -> bool {
        if !self.enabled() {
            return true;
        }
        let mut map = self.entries.lock().expect("breaker mutex poisoned");
        let Some(e) = map.get_mut(worker_id) else {
            return true;
        };
        match e.state {
            BreakerState::Closed => return true,
            BreakerState::Open => {
                if now < e.opens_until {
                    return false;
                }
                e.state = BreakerState::HalfOpen;
                e.probe_in_flight = false;
                tracing::info!(worker = worker_id, "breaker half-open, admitting one probe");
            }
            BreakerState::HalfOpen => {}
        }
        if e.probe_in_flight {
            return false;
        }
        e.probe_in_flight = true;
        true
    }

    /// Drop workers whose breaker is open.
    ///
    /// If *every* candidate is open the full list is returned instead of an
    /// empty one: a request served by a probably-bad worker beats a guaranteed
    /// 503, and refusing to route would turn a partial outage into a total one.
    pub fn filter<W: Clone>(&self, workers: &[W], id_of: impl Fn(&W) -> &str) -> Vec<W> {
        let allowed: Vec<W> = workers
            .iter()
            .filter(|w| self.allows(id_of(w)))
            .cloned()
            .collect();
        if !allowed.is_empty() {
            return allowed;
        }
        if !workers.is_empty() {
            tracing::warn!(
                candidates = workers.len(),
                "breaker: all candidates open; routing anyway rather than failing"
            );
        }
        workers.to_vec()
    }

    pub fn record_success(&self, worker_id: &str) {
        let mut map = self.entries.lock().expect("breaker mutex poisoned");
        if let Some(e) = map.get_mut(worker_id) {
            if e.state != BreakerState::Closed {
                tracing::info!(worker = worker_id, "breaker: worker recovered, closing");
            }
            e.consecutive_failures = 0;
            e.state = BreakerState::Closed;
            e.probe_in_flight = false;
            e.next_cooldown = self.cooldown;
        }
    }

    /// Record a pre-first-byte dispatch failure. Callers must gate this on
    /// [`is_worker_fault`] when the failure carries an HTTP status.
    pub fn record_failure(&self, worker_id: &str) {
        self.record_failure_at(worker_id, Instant::now());
    }

    fn record_failure_at(&self, worker_id: &str, now: Instant) {
        if !self.enabled() {
            return;
        }
        let mut map = self.entries.lock().expect("breaker mutex poisoned");
        let e = map.entry(worker_id.to_string()).or_insert_with(|| Entry {
            consecutive_failures: 0,
            state: BreakerState::Closed,
            opens_until: now,
            next_cooldown: self.cooldown,
            probe_in_flight: false,
            trips: 0,
        });
        e.consecutive_failures += 1;
        let was_probe = e.state == BreakerState::HalfOpen;
        e.probe_in_flight = false;

        if was_probe {
            // A failed probe reopens immediately and backs off further, without
            // waiting out the threshold again — we already know it is bad. The
            // common cause (a worker wedged on a bad KV handoff) does not clear
            // on the first retry, and a fixed cooldown would probe it at a
            // constant rate forever.
            e.next_cooldown = (e.next_cooldown * 2).min(self.max_cooldown);
        } else if e.consecutive_failures < self.failure_threshold {
            return;
        }
        e.state = BreakerState::Open;
        e.opens_until = now + e.next_cooldown;
        e.trips += 1;
        tracing::warn!(
            worker = worker_id,
            cooldown_s = e.next_cooldown.as_secs_f64(),
            failures = e.consecutive_failures,
            "breaker: worker open"
        );
    }

    pub fn state_of(&self, worker_id: &str) -> BreakerState {
        self.entries
            .lock()
            .expect("breaker mutex poisoned")
            .get(worker_id)
            .map(|e| e.state)
            .unwrap_or(BreakerState::Closed)
    }

    /// `(worker_id, state, trips)` for metrics export.
    pub fn snapshot(&self) -> Vec<(String, BreakerState, u64)> {
        let map = self.entries.lock().expect("breaker mutex poisoned");
        let mut out: Vec<_> = map
            .iter()
            .map(|(k, e)| (k.clone(), e.state, e.trips))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(3, Duration::from_secs(5), Duration::from_secs(60))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Time is driven forward explicitly rather than slept, so the cooldown
    /// behaviour is tested at full speed and deterministically.
    fn cb() -> CircuitBreaker {
        CircuitBreaker::new(3, Duration::from_secs(5), Duration::from_secs(20))
    }

    #[derive(Clone)]
    struct W(&'static str);

    #[test]
    fn threshold_zero_disables_it() {
        // Without the guard, `failures >= 0` would trip on the first failure —
        // the exact opposite of what --breaker-failure-threshold=0 promises.
        let b = CircuitBreaker::new(0, Duration::from_secs(5), Duration::from_secs(20));
        for _ in 0..20 {
            b.record_failure("w1");
        }
        assert!(b.allows("w1"));
        assert_eq!(b.state_of("w1"), BreakerState::Closed);
        let ws = vec![W("a"), W("b")];
        assert_eq!(b.filter(&ws, |w| w.0).len(), 2);
    }

    #[test]
    fn unknown_worker_is_allowed() {
        let b = cb();
        assert!(b.allows("w1"));
        assert_eq!(b.state_of("w1"), BreakerState::Closed);
    }

    #[test]
    fn failures_below_threshold_do_not_open() {
        let b = cb();
        b.record_failure("w1");
        b.record_failure("w1");
        assert!(b.allows("w1"));
        assert_eq!(b.state_of("w1"), BreakerState::Closed);
    }

    #[test]
    fn opens_at_threshold_and_excludes() {
        let b = cb();
        for _ in 0..3 {
            b.record_failure("w1");
        }
        assert_eq!(b.state_of("w1"), BreakerState::Open);
        assert!(!b.allows("w1"));
    }

    #[test]
    fn success_resets_the_count() {
        let b = cb();
        b.record_failure("w1");
        b.record_failure("w1");
        b.record_success("w1");
        b.record_failure("w1");
        b.record_failure("w1");
        assert_eq!(b.state_of("w1"), BreakerState::Closed);
    }

    #[test]
    fn half_open_admits_exactly_one_probe() {
        let b = cb();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record_failure_at("w1", t0);
        }
        let t1 = t0 + Duration::from_millis(5_100);
        assert!(b.allows_at("w1", t1), "cooldown elapsed -> one probe");
        assert_eq!(b.state_of("w1"), BreakerState::HalfOpen);
        assert!(
            !b.allows_at("w1", t1),
            "a second concurrent request must not also probe"
        );
    }

    #[test]
    fn successful_probe_closes() {
        let b = cb();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record_failure_at("w1", t0);
        }
        let t1 = t0 + Duration::from_millis(5_100);
        b.allows_at("w1", t1);
        b.record_success("w1");
        assert_eq!(b.state_of("w1"), BreakerState::Closed);
        assert!(b.allows_at("w1", t1));
    }

    #[test]
    fn failed_probe_reopens_with_doubled_cooldown() {
        let b = cb();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record_failure_at("w1", t0);
        }
        let t1 = t0 + Duration::from_millis(5_100);
        b.allows_at("w1", t1);
        b.record_failure_at("w1", t1); // probe fails -> reopen, 5s -> 10s
        assert_eq!(b.state_of("w1"), BreakerState::Open);

        let t2 = t1 + Duration::from_millis(5_100); // old cooldown would be up
        assert!(!b.allows_at("w1", t2), "backoff must have doubled");
        let t3 = t1 + Duration::from_millis(10_100);
        assert!(b.allows_at("w1", t3));
    }

    #[test]
    fn cooldown_is_capped() {
        let b = cb();
        let mut t = Instant::now();
        for _ in 0..3 {
            b.record_failure_at("w1", t);
        }
        for _ in 0..6 {
            t += Duration::from_secs(1000);
            b.allows_at("w1", t);
            b.record_failure_at("w1", t);
        }
        assert!(
            b.allows_at("w1", t + Duration::from_millis(20_100)),
            "cooldown must not grow without bound"
        );
    }

    #[test]
    fn filter_drops_open_workers() {
        let b = cb();
        for _ in 0..3 {
            b.record_failure("bad");
        }
        let ws = vec![W("good"), W("bad")];
        let got = b.filter(&ws, |w| w.0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "good");
    }

    #[test]
    fn filter_returns_all_when_every_worker_is_open() {
        let b = cb();
        for id in ["a", "b"] {
            for _ in 0..3 {
                b.record_failure(id);
            }
        }
        let ws = vec![W("a"), W("b")];
        assert_eq!(b.filter(&ws, |w| w.0).len(), 2);
    }

    #[test]
    fn filter_of_empty_is_empty() {
        let b = cb();
        let ws: Vec<W> = vec![];
        assert!(b.filter(&ws, |w| w.0).is_empty());
    }

    #[test]
    fn workers_are_independent() {
        let b = cb();
        for _ in 0..3 {
            b.record_failure("bad");
        }
        assert!(b.allows("good"));
    }

    #[test]
    fn success_on_unknown_worker_is_harmless() {
        let b = cb();
        b.record_success("never-seen");
        assert!(b.allows("never-seen"));
    }

    #[test]
    fn snapshot_reports_trips() {
        let b = cb();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record_failure_at("w1", t0);
        }
        let t1 = t0 + Duration::from_millis(5_100);
        b.allows_at("w1", t1);
        b.record_failure_at("w1", t1);
        let snap = b.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].1, BreakerState::Open);
        assert_eq!(snap[0].2, 2, "initial trip plus the failed probe");
    }

    #[test]
    fn client_errors_are_not_worker_faults() {
        for s in [400u16, 404, 422, 429] {
            assert!(!is_worker_fault(s), "{s} must not trip the breaker");
        }
        for s in [0u16, 500, 502, 503, 504] {
            assert!(is_worker_fault(s), "{s} must trip the breaker");
        }
    }

    #[test]
    fn a_bad_client_cannot_trip_the_fleet() {
        let b = cb();
        let ws = vec![W("a"), W("b"), W("c")];
        for _ in 0..10 {
            for w in b.filter(&ws, |w| w.0) {
                if is_worker_fault(400) {
                    b.record_failure(w.0);
                }
            }
        }
        for w in &ws {
            assert_eq!(b.state_of(w.0), BreakerState::Closed);
        }
    }

    #[test]
    fn the_regression_this_exists_for() {
        // A worker that fails every dispatch must stop being selected. Before
        // this type existed `tried` was per-request, so `bad` was offered on
        // all ten requests.
        let b = cb();
        let ws = vec![W("good"), W("bad")];
        let mut offered_bad = 0;
        for _ in 0..10 {
            let cands = b.filter(&ws, |w| w.0);
            if cands.iter().any(|w| w.0 == "bad") {
                offered_bad += 1;
                b.record_failure("bad");
            }
            b.record_success("good");
        }
        assert_eq!(offered_bad, 3, "bad worker must stop being offered");
    }

    #[test]
    fn concurrent_probes_admit_only_one() {
        // The Python breaker is single-loop; this one is shared across tokio
        // threads, so the half-open slot has to be safe under real contention.
        use std::sync::Arc;
        let b = Arc::new(cb());
        for _ in 0..3 {
            b.record_failure("w1");
        }
        // Force half-open by driving the clock through the private hook.
        let t = Instant::now() + Duration::from_secs(6);
        assert!(b.allows_at("w1", t));
        b.record_failure_at("w1", t); // back to open, then reopen at 10s
        let t2 = t + Duration::from_secs(11);

        let admitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hs: Vec<_> = (0..16)
            .map(|_| {
                let b = b.clone();
                let admitted = admitted.clone();
                std::thread::spawn(move || {
                    if b.allows_at("w1", t2) {
                        admitted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(
            admitted.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly one of 16 racing threads may probe"
        );
    }
}
