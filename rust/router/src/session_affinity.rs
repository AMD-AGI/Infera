//! Single-router session affinity. Pins constrain selection, never bypass accounting.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use serde_json::Value;

use crate::policy::{Pick, Policy, Role};
use crate::pool::{expand_targets, RouteTarget, Worker};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Prefill,
    Both,
}

type Key = (String, String, Role);
struct Entry {
    target: Option<RouteTarget>,
    generation: u64,
    active: usize,
    idle_since: Instant,
}
struct Entries {
    values: HashMap<Key, Arc<Mutex<Entry>>>,
    swept: Instant,
}

#[derive(Default)]
struct Counters {
    active: [AtomicUsize; 3],
    hits: [AtomicUsize; 3],
    selected: [AtomicUsize; 3],
}
fn role_index(role: Role) -> usize {
    match role {
        Role::Prefill => 0,
        Role::Decode => 1,
        Role::Mixed => 2,
    }
}

pub struct Sessions {
    mode: Mode,
    counters: Arc<Counters>,
    ttl: Duration,
    capacity: usize,
    entries: Mutex<Entries>,
}

impl Default for Sessions {
    fn default() -> Self {
        Self::new(Mode::Off, Duration::from_secs(3600), 65536)
    }
}

impl Sessions {
    pub fn new(mode: Mode, ttl: Duration, capacity: usize) -> Self {
        Self {
            mode,
            counters: Arc::new(Counters::default()),
            ttl,
            capacity,
            entries: Mutex::new(Entries {
                values: HashMap::new(),
                swept: Instant::now(),
            }),
        }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let mode = match std::env::var("INFERA_SESSION_AFFINITY")
            .as_deref()
            .unwrap_or("off")
        {
            "off" => Mode::Off,
            "prefill" => Mode::Prefill,
            "both" => Mode::Both,
            other => anyhow::bail!("invalid INFERA_SESSION_AFFINITY: {other}"),
        };
        let ttl: u64 = std::env::var("INFERA_SESSION_AFFINITY_TTL_SECS")
            .unwrap_or_else(|_| "3600".into())
            .parse()?;
        anyhow::ensure!(
            (1..=86400).contains(&ttl),
            "session TTL must be 1..86400 seconds"
        );
        tracing::info!(?mode, ttl, "session affinity configuration");
        Ok(Self::new(mode, Duration::from_secs(ttl), 65536))
    }

    pub fn metrics(&self) -> String {
        let mut out = String::new();
        for (i, role) in ["prefill", "decode", "mixed"].iter().enumerate() {
            for (name, counter) in [
                ("active", &self.counters.active[i]),
                ("hits_total", &self.counters.hits[i]),
                ("selected_total", &self.counters.selected[i]),
            ] {
                out.push_str(&format!(
                    "infera_router_session_{name}{{role=\"{role}\"}} {}\n",
                    counter.load(Ordering::Relaxed)
                ));
            }
        }
        out
    }

    fn lease(&self, entry: Arc<Mutex<Entry>>, generation: u64, role: Role) -> Lease {
        let role = role_index(role);
        self.counters.active[role].fetch_add(1, Ordering::Relaxed);
        Lease {
            entry,
            generation,
            role,
            counters: self.counters.clone(),
        }
    }

    pub fn header<'a>(&self, headers: &'a HeaderMap) -> Result<Option<&'a str>, &'static str> {
        if self.mode == Mode::Off {
            return Ok(None);
        }
        let mut values = headers.get_all("x-dynamo-session-id").iter();
        let Some(value) = values.next() else {
            return Ok(None);
        };
        if values.next().is_some() {
            return Err("duplicate X-Dynamo-Session-ID");
        }
        let id = value.to_str().map_err(|_| "invalid X-Dynamo-Session-ID")?;
        if id.is_empty() || id.len() > 1024 {
            return Err("X-Dynamo-Session-ID must be 1..1024 bytes");
        }
        Ok(Some(id))
    }

    pub fn pick(
        &self,
        policy: &dyn Policy,
        candidates: &[Arc<Worker>],
        request: &Value,
        role: Role,
        model: &str,
        session: Option<&str>,
    ) -> (Pick, Option<Lease>) {
        let enabled =
            self.mode == Mode::Both || (self.mode == Mode::Prefill && role == Role::Prefill);
        let Some(session) = session.filter(|_| enabled) else {
            return (policy.pick(candidates, request, role), None);
        };
        let now = Instant::now();
        let entry = {
            let mut entries = self.entries.lock().expect("session map poisoned");
            if now.duration_since(entries.swept) >= Duration::from_secs(30)
                || entries.values.len() >= self.capacity
            {
                entries.values.retain(|_, entry| {
                    Arc::strong_count(entry) > 1
                        || entry.try_lock().map_or(true, |e| {
                            e.active > 0 || now.duration_since(e.idle_since) < self.ttl
                        })
                });
                entries.swept = now;
            }
            let key = (model.to_owned(), session.to_owned(), role);
            if let Some(entry) = entries.values.get(&key) {
                Some(entry.clone())
            } else if entries.values.len() < self.capacity {
                let entry = Arc::new(Mutex::new(Entry {
                    target: None,
                    generation: 0,
                    active: 0,
                    idle_since: now,
                }));
                entries.values.insert(key, entry.clone());
                Some(entry)
            } else {
                None
            }
        };
        let Some(entry) = entry else {
            tracing::warn!(
                ?role,
                "session affinity capacity reached; using normal selection"
            );
            return (policy.pick(candidates, request, role), None);
        };
        // Only initialization holds this per-session lock while choosing. Other sessions proceed.
        let mut e = entry.lock().expect("session entry poisoned");
        let expired = e.active == 0 && now.duration_since(e.idle_since) >= self.ttl;
        let target = e.target.as_ref().filter(|_| !expired).and_then(|old| {
            expand_targets(candidates)
                .into_iter()
                .find(|t| t.route_key() == old.route_key() && *t.worker == *old.worker)
        });
        let (pick, reason) = if let Some(target) = target {
            e.active += 1;
            let generation = e.generation;
            drop(e);
            self.counters.hits[role_index(role)].fetch_add(1, Ordering::Relaxed);
            let lease = self.lease(entry, generation, role);
            let pick = policy.pick_target(&target, request, role);
            tracing::info!(?role, target=%pick.target.route_key(), reason="hit", "session affinity pick");
            return (pick, Some(lease));
        } else {
            let reason = if expired {
                "expired"
            } else if e.target.is_some() {
                "invalid"
            } else {
                "new"
            };
            (policy.pick(candidates, request, role), reason)
        };
        e.target = Some(pick.target.clone());
        e.generation = e.generation.wrapping_add(1);
        e.active += 1;
        let generation = e.generation;
        drop(e);
        tracing::info!(?role, target=%pick.target.route_key(), reason, "session affinity pick");
        self.counters.selected[role_index(role)].fetch_add(1, Ordering::Relaxed);
        (pick, Some(self.lease(entry, generation, role)))
    }
}

pub struct Lease {
    counters: Arc<Counters>,
    role: usize,
    entry: Arc<Mutex<Entry>>,
    generation: u64,
}
#[derive(Clone)]
pub struct Binding {
    entry: std::sync::Weak<Mutex<Entry>>,
    generation: u64,
}
impl Binding {
    pub fn invalidate(&self) {
        if let Some(entry) = self.entry.upgrade() {
            let mut e = entry.lock().expect("session entry poisoned");
            if e.generation == self.generation {
                e.target = None;
            }
        }
    }
}
impl Lease {
    pub fn binding(&self) -> Binding {
        Binding {
            entry: Arc::downgrade(&self.entry),
            generation: self.generation,
        }
    }
    pub fn invalidate(&self) {
        self.binding().invalidate();
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut e = self.entry.lock().expect("session entry poisoned");
        e.active -= 1;
        self.counters.active[self.role].fetch_sub(1, Ordering::Relaxed);
        if e.active == 0 {
            e.idle_since = Instant::now();
        }
    }
}

/// NATS errors may become successful SSE writes; inspect frames before encoding.
pub(crate) fn observe_reply(frame: Option<&crate::nats_request::Frame>, bindings: &[Binding]) {
    use crate::nats_request::Frame;
    if matches!(frame, Some(Frame::Data(_)))
        || matches!(frame, Some(Frame::Done { status }) if (200..300).contains(status))
    {
        return;
    }
    for binding in bindings {
        binding.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{ActiveGuard, RoundRobin};
    use serde_json::json;

    fn workers() -> Vec<Arc<Worker>> {
        vec![Arc::new(
            serde_json::from_value(json!({
                "worker_id":"w", "url":"http://localhost", "model_name":"m", "dp_size":8
            }))
            .unwrap(),
        )]
    }
    fn pick(s: &Sessions, p: &dyn Policy, role: Role, id: Option<&str>) -> (Pick, Option<Lease>) {
        s.pick(p, &workers(), &json!({}), role, "m", id)
    }

    #[test]
    fn modes_and_models_do_not_share_bindings() {
        let s = Sessions::new(Mode::Prefill, Duration::from_secs(3600), 10);
        let p = RoundRobin::new();
        let (a, lease) = pick(&s, &p, Role::Prefill, Some("x"));
        drop(lease);
        let (b, _) = pick(&s, &p, Role::Prefill, Some("x"));
        assert_eq!(a.target.dp_rank, b.target.dp_rank);
        let (c, _) = s.pick(
            &p,
            &workers(),
            &json!({}),
            Role::Prefill,
            "other-model",
            Some("x"),
        );
        assert_ne!(a.target.dp_rank, c.target.dp_rank);
        let (d, lease) = pick(&s, &p, Role::Decode, Some("x"));
        assert!(lease.is_none());
        let (e, _) = pick(&s, &p, Role::Decode, Some("x"));
        assert_ne!(d.target.dp_rank, e.target.dp_rank);
        let off = Sessions::default();
        assert!(pick(&off, &p, Role::Prefill, Some("x")).1.is_none());
        assert!(pick(&s, &p, Role::Prefill, None).1.is_none());
    }

    #[test]
    fn ttl_waits_for_active_leases_and_renews_on_release() {
        let s = Sessions::new(Mode::Both, Duration::from_secs(10), 10);
        let p = RoundRobin::new();
        let (a, lease) = pick(&s, &p, Role::Prefill, Some("x"));
        let lease = lease.unwrap();
        lease.entry.lock().unwrap().idle_since = Instant::now() - Duration::from_secs(20);
        let (b, second) = pick(&s, &p, Role::Prefill, Some("x"));
        assert_eq!(a.target.dp_rank, b.target.dp_rank);
        drop(second);
        drop(lease);
        let entry = s
            .entries
            .lock()
            .unwrap()
            .values
            .values()
            .next()
            .unwrap()
            .clone();
        assert!(entry.lock().unwrap().idle_since.elapsed() < Duration::from_secs(1));
        entry.lock().unwrap().idle_since = Instant::now() - Duration::from_secs(20);
        let (c, _) = pick(&s, &p, Role::Prefill, Some("x"));
        assert_ne!(a.target.dp_rank, c.target.dp_rank);
    }

    #[test]
    fn old_failure_cannot_invalidate_new_binding() {
        let s = Sessions::new(Mode::Both, Duration::from_secs(10), 10);
        let p = RoundRobin::new();
        let (a, old) = pick(&s, &p, Role::Prefill, Some("x"));
        old.as_ref().unwrap().invalidate();
        let (b, current) = pick(&s, &p, Role::Prefill, Some("x"));
        assert_ne!(a.target.dp_rank, b.target.dp_rank);
        old.as_ref().unwrap().invalidate();
        let (c, _) = pick(&s, &p, Role::Prefill, Some("x"));
        assert_eq!(b.target.dp_rank, c.target.dp_rank);
        drop(old);
        drop(current);
    }

    #[test]
    fn removed_rank_or_replaced_worker_is_reselected() {
        let s = Sessions::new(Mode::Both, Duration::from_secs(10), 10);
        let p = RoundRobin::new();
        let (_, old) = pick(&s, &p, Role::Decode, Some("x"));
        let mut replacement = (*workers()[0]).clone();
        replacement.url = "http://new-worker".into();
        replacement.dp_size = Some(1);
        let (new, _) = s.pick(
            &p,
            &[Arc::new(replacement)],
            &json!({}),
            Role::Decode,
            "m",
            Some("x"),
        );
        assert_eq!(new.target.dp_rank, None);
        assert_eq!(new.target.worker.url, "http://new-worker");
        old.unwrap().invalidate();
        assert!(s
            .entries
            .lock()
            .unwrap()
            .values
            .values()
            .next()
            .unwrap()
            .lock()
            .unwrap()
            .target
            .is_some());
    }

    #[test]
    fn simultaneous_first_requests_use_one_rank() {
        let s = Arc::new(Sessions::new(Mode::Both, Duration::from_secs(10), 10));
        let p = Arc::new(RoundRobin::new());
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let (s, p, b) = (s.clone(), p.clone(), barrier.clone());
                std::thread::spawn(move || {
                    b.wait();
                    let (pick, _lease) = pick(&s, p.as_ref(), Role::Prefill, Some("x"));
                    pick.target.dp_rank
                })
            })
            .collect();
        let ranks: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert!(ranks.iter().all(|r| *r == ranks[0]));
        assert_eq!(
            s.entries
                .lock()
                .unwrap()
                .values
                .values()
                .next()
                .unwrap()
                .lock()
                .unwrap()
                .active,
            0
        );
    }

    #[test]
    fn prefill_lease_detaches_even_without_legacy_early_release() {
        let s = Sessions::new(Mode::Both, Duration::from_secs(10), 10);
        let p = Arc::new(RoundRobin::new());
        let (_, pl) = pick(&s, p.as_ref(), Role::Prefill, Some("x"));
        let (_, dl) = pick(&s, p.as_ref(), Role::Decode, Some("x"));
        let pe = pl.as_ref().unwrap().entry.clone();
        let de = dl.as_ref().unwrap().entry.clone();
        let mut guard = ActiveGuard::start(p, vec![]).with_sessions(pl, dl);
        let pg = guard.take_prefill_session().unwrap();
        drop(guard);
        assert_eq!(de.lock().unwrap().active, 0);
        assert_eq!(pe.lock().unwrap().active, 1);
        drop(pg);
        assert_eq!(pe.lock().unwrap().active, 0);
    }

    #[test]
    fn capacity_does_not_evict_inflight_session() {
        let s = Sessions::new(Mode::Both, Duration::from_secs(10), 1);
        let p = RoundRobin::new();
        let (_, lease) = pick(&s, &p, Role::Prefill, Some("x"));
        assert!(pick(&s, &p, Role::Prefill, Some("y")).1.is_none());
        drop(lease);
        s.entries
            .lock()
            .unwrap()
            .values
            .values()
            .next()
            .unwrap()
            .lock()
            .unwrap()
            .idle_since = Instant::now() - Duration::from_secs(20);
        assert!(pick(&s, &p, Role::Prefill, Some("y")).1.is_some());
    }

    #[test]
    fn nats_terminal_failures_invalidate_even_when_encoded_as_sse_data() {
        use crate::nats_request::Frame;
        for terminal in [
            None,
            Some(Frame::Done { status: 500 }),
            Some(Frame::Error {
                status: Some(502),
                message: "failed".into(),
            }),
        ] {
            let sessions = Sessions::new(Mode::Both, Duration::from_secs(10), 8);
            let policy = RoundRobin::new();
            let (_, lease) = pick(&sessions, &policy, Role::Decode, Some("s"));
            let lease = lease.unwrap();
            let bindings = [lease.binding()];
            observe_reply(
                Some(&Frame::Data(axum::body::Bytes::from_static(b"data"))),
                &bindings,
            );
            observe_reply(Some(&Frame::Done { status: 200 }), &bindings);
            assert!(lease.entry.lock().unwrap().target.is_some());
            observe_reply(terminal.as_ref(), &bindings);
            assert!(lease.entry.lock().unwrap().target.is_none());
        }
    }

    #[test]
    fn header_is_explicit_and_disabled_mode_ignores_it() {
        let mut h = HeaderMap::new();
        h.insert("x-correlation-id", "not-a-session".parse().unwrap());
        let s = Sessions::new(Mode::Both, Duration::from_secs(10), 1);
        assert_eq!(s.header(&h).unwrap(), None);
        h.insert("X-Dynamo-Session-ID", "session".parse().unwrap());
        assert_eq!(s.header(&h).unwrap(), Some("session"));
        h.append("X-Dynamo-Session-ID", "duplicate".parse().unwrap());
        assert!(s.header(&h).is_err());
        assert_eq!(Sessions::default().header(&h).unwrap(), None);
    }
}
