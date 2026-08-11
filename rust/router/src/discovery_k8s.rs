///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Kubernetes-native worker discovery.
//!
//! The Rust half of `infera.common.discovery_k8s`. Workers publish their
//! registration into their own Pod annotation (`infera.amd.com/worker-info`)
//! and this lists + watches those Pods through the in-cluster API server, so
//! Pod lifetime replaces the etcd lease.
//!
//! Same shape as the etcd backend: a working set is maintained here and
//! republished as an immutable `Snapshot` on every change, so the request hot
//! path stays lock-free.
//!
//! Three states, not two, and the middle one is the point:
//!
//! * **Registered** — annotated, Running, not condemned. A routing candidate.
//! * **Draining** — the Pod has a `deletionTimestamp` but is still Running. It
//!   stops being a candidate immediately while its record survives, so
//!   `/v1/workers` shows a rolling update rather than a fleet that appears to
//!   be losing workers. Kubernetes stamps this *before* the process is
//!   signalled, which is the whole reason a graceful drain is possible here and
//!   not on etcd.
//! * **Gone** — deleted, no longer Running, or the annotation was cleared. The
//!   worker clears it itself on SIGTERM, before draining, so this is also how a
//!   kill with no Pod deletion (liveness probe, node shutdown) is observed.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use serde_json::Value;

use crate::breaker::CircuitBreaker;
use crate::k8s;
use crate::policy::Policy;
use crate::pool::{SharedPool, Snapshot, Worker};

/// Pod annotation carrying the JSON worker registration record.
pub const WORKER_INFO_ANNOTATION: &str = "infera.amd.com/worker-info";

/// What the pool calls a worker that is condemned but still finishing work.
/// `Worker::is_active` is false for it, which is what removes it from routing.
const STATUS_DRAINING: &str = "draining";

/// One tracked Pod. Keyed by Pod name because a DELETE event carries the Pod,
/// not the worker id.
struct Tracked {
    worker: Arc<Worker>,
    /// Whether the fleet has already been told this worker is leaving. Set when
    /// it starts draining, so the later removal does not announce it twice.
    announced: bool,
}

pub async fn run(
    selector: String,
    namespace: String,
    pool: SharedPool,
    policy: Arc<dyn Policy>,
    breaker: Arc<CircuitBreaker>,
) {
    let mut tracked: HashMap<String, Tracked> = HashMap::new();
    let mut backoff = 1u64;
    // None means "re-list first": either the first run, or the watch's
    // resourceVersion went stale.
    let mut resource_version: Option<String> = None;

    loop {
        let step = async {
            if resource_version.is_none() {
                resource_version = relist(
                    &selector,
                    &namespace,
                    &mut tracked,
                    &pool,
                    &policy,
                    &breaker,
                )
                .await?;
            }
            watch_once(
                &selector,
                &namespace,
                resource_version.clone(),
                &mut tracked,
                &pool,
                &policy,
                &breaker,
            )
            .await
        };
        match step.await {
            Ok(rv) => {
                // `None` here is a stale-resourceVersion signal, not an error:
                // the next iteration re-lists instead of retrying a version the
                // API server has already compacted past.
                resource_version = rv;
                backoff = 1;
            }
            Err(e) => {
                tracing::warn!("k8s discovery error: {e}; retry in {backoff}s");
                resource_version = None;
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(30);
            }
        }
    }
}

/// List every matching Pod, rebuild the working set, and return the
/// resourceVersion to watch from.
///
/// A list is a complete snapshot, so anything still tracked that it does not
/// mention has been deleted with its event lost. That is routine rather than an
/// edge case -- the watch's resourceVersion expires every few minutes and any
/// Pod deleted inside a reconnect window produces no event anyone sees. It
/// matters more now that a draining worker keeps its record: removal used to be
/// immediate, which bounded how long a stale entry could survive.
async fn relist(
    selector: &str,
    namespace: &str,
    tracked: &mut HashMap<String, Tracked>,
    pool: &SharedPool,
    policy: &Arc<dyn Policy>,
    breaker: &Arc<CircuitBreaker>,
) -> Result<Option<String>> {
    let client = k8s::make_client(Some(Duration::from_secs(10)))?;
    let url = format!(
        "{}/api/v1/namespaces/{}/pods",
        k8s::api_server_base(),
        namespace
    );
    let resp = client
        .get(&url)
        .query(&[("labelSelector", selector)])
        .send()
        .await
        .context("listing pods")?
        .error_for_status()
        .context("listing pods")?;
    let body: Value = resp.json().await.context("decoding the pod list")?;

    let mut seen: HashSet<String> = HashSet::new();
    if let Some(items) = body.get("items").and_then(|i| i.as_array()) {
        for pod in items {
            if let Some(name) = pod_name(pod) {
                seen.insert(name);
            }
            handle_pod(pod, false, tracked);
        }
    }
    tracked.retain(|pod, t| {
        let keep = seen.contains(pod);
        if !keep {
            tracing::info!(
                "k8s: pod {pod} absent from list; dropping worker {}",
                t.worker.worker_id
            );
        }
        keep
    });

    publish(pool, policy, breaker, tracked);
    let rv = body
        .get("metadata")
        .and_then(|m| m.get("resourceVersion"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    tracing::info!(
        "k8s (re)list: {} worker(s) for selector {selector:?} in ns {namespace} (rv={rv:?})",
        tracked.len()
    );
    Ok(rv)
}

/// Stream watch events until the connection ends. `Ok(None)` means the
/// resourceVersion went stale and the caller must re-list.
async fn watch_once(
    selector: &str,
    namespace: &str,
    resource_version: Option<String>,
    tracked: &mut HashMap<String, Tracked>,
    pool: &SharedPool,
    policy: &Arc<dyn Policy>,
    breaker: &Arc<CircuitBreaker>,
) -> Result<Option<String>> {
    let client = k8s::make_client(None)?;
    let url = format!(
        "{}/api/v1/namespaces/{}/pods",
        k8s::api_server_base(),
        namespace
    );
    let mut req = client
        .get(&url)
        .query(&[("labelSelector", selector), ("watch", "true")]);
    if let Some(rv) = resource_version.as_deref() {
        req = req.query(&[("resourceVersion", rv)]);
    }
    let resp = req
        .send()
        .await
        .context("opening the pod watch")?
        .error_for_status()
        .context("opening the pod watch")?;

    let mut rv = resource_version;
    let mut stream = resp.bytes_stream();
    // Events are newline-delimited JSON, and a chunk boundary can land anywhere
    // -- including mid-object -- so lines are reassembled rather than parsed per
    // chunk.
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.context("reading the pod watch")?);
        while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let line = &line[..line.len() - 1];
            if line.is_empty() {
                continue;
            }
            let event: Value = match serde_json::from_slice(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            match event.get("type").and_then(|t| t.as_str()) {
                // A 410 Gone arrives as an ERROR event rather than an HTTP
                // status. Returning None re-lists instead of reconnecting with
                // the same expired version forever.
                Some("ERROR") => {
                    tracing::info!(
                        "k8s watch stale (re-listing): {}",
                        event
                            .get("object")
                            .and_then(|o| o.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("ERROR")
                    );
                    return Ok(None);
                }
                Some(kind @ ("ADDED" | "MODIFIED" | "DELETED")) => {
                    let pod = match event.get("object") {
                        Some(p) => p,
                        None => continue,
                    };
                    if let Some(v) = pod
                        .get("metadata")
                        .and_then(|m| m.get("resourceVersion"))
                        .and_then(|v| v.as_str())
                    {
                        rv = Some(v.to_string());
                    }
                    if handle_pod(pod, kind == "DELETED", tracked) {
                        publish(pool, policy, breaker, tracked);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(rv)
}

fn pod_name(pod: &Value) -> Option<String> {
    pod.get("metadata")
        .and_then(|m| m.get("name"))
        .and_then(|n| n.as_str())
        .filter(|n| !n.is_empty())
        .map(str::to_string)
}

fn pod_running(pod: &Value) -> bool {
    pod.get("status")
        .and_then(|s| s.get("phase"))
        .and_then(|p| p.as_str())
        == Some("Running")
}

/// True once the API server has stamped the Pod for deletion.
///
/// A terminating Pod keeps `phase: Running` until its containers exit, so
/// liveness alone cannot see it. The gap is not academic: the operator injects
/// a `preStop sleep` before SIGTERM, and for that whole delay the Pod is
/// condemned, still Running, and -- without this -- still a routing candidate.
fn pod_terminating(pod: &Value) -> bool {
    pod.get("metadata")
        .and_then(|m| m.get("deletionTimestamp"))
        .is_some_and(|v| !v.is_null())
}

/// Apply one Pod observation. Returns true when the fleet changed.
fn handle_pod(pod: &Value, deleted: bool, tracked: &mut HashMap<String, Tracked>) -> bool {
    let name = match pod_name(pod) {
        Some(n) => n,
        None => return false,
    };
    let raw = pod
        .get("metadata")
        .and_then(|m| m.get("annotations"))
        .and_then(|a| a.get(WORKER_INFO_ANNOTATION))
        .and_then(|v| v.as_str());

    // Gone: an explicit delete, the annotation cleared (the worker deregistered,
    // which is how its drain begins), or no longer Running.
    if deleted || raw.is_none() || !pod_running(pod) {
        return match tracked.remove(&name) {
            Some(t) => {
                tracing::info!(
                    "k8s: worker {} removed (pod deleted / not ready)",
                    t.worker.worker_id
                );
                true
            }
            None => false,
        };
    }

    // Condemned but still serving. Checked separately from Running because a
    // terminating Pod stays Running until its containers exit; it leaves
    // routing now and its record survives until it actually goes.
    if pod_terminating(pod) {
        return match tracked.get_mut(&name) {
            Some(t) if !t.announced => {
                let mut w = (*t.worker).clone();
                w.status = STATUS_DRAINING.to_string();
                tracing::info!(
                    "k8s: worker {} draining (pod terminating, record kept)",
                    w.worker_id
                );
                t.worker = Arc::new(w);
                t.announced = true;
                true
            }
            // Already draining, or never registered.
            _ => false,
        };
    }

    let worker: Worker = match serde_json::from_str(raw.unwrap_or_default()) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("k8s: bad worker-info annotation on pod {name}: {e}");
            return false;
        }
    };
    let existed = tracked.contains_key(&name);
    let unchanged = tracked
        .get(&name)
        .is_some_and(|t| !t.announced && same_worker(&t.worker, &worker));
    if unchanged {
        // Pods generate MODIFIED events for things the router does not care
        // about -- status conditions, resourceVersion bumps on every heartbeat
        // annotation refresh. Republishing on those would rebuild the snapshot
        // and re-run policy reconciliation several times a second per worker.
        return false;
    }
    tracing::info!(
        "k8s: worker {} {} (pod={name}, {}, model={})",
        worker.worker_id,
        if existed { "updated" } else { "registered" },
        worker.url,
        worker.model_name
    );
    tracked.insert(
        name,
        Tracked {
            worker: Arc::new(worker),
            announced: false,
        },
    );
    true
}

/// Whether a re-registration carries anything the router acts on.
fn same_worker(a: &Worker, b: &Worker) -> bool {
    a.worker_id == b.worker_id
        && a.url == b.url
        && a.model_name == b.model_name
        && a.status == b.status
        && a.disagg_mode == b.disagg_mode
        && a.kv_events_endpoint == b.kv_events_endpoint
        && a.kv_block_size == b.kv_block_size
        && a.dp_rank == b.dp_rank
        && a.dp_size == b.dp_size
        && a.request_transport == b.request_transport
}

fn publish(
    pool: &SharedPool,
    policy: &Arc<dyn Policy>,
    breaker: &Arc<CircuitBreaker>,
    tracked: &HashMap<String, Tracked>,
) {
    let all: Vec<Arc<Worker>> = tracked.values().map(|t| t.worker.clone()).collect();
    // Cost-aware policies reconcile against the workers still being routed to,
    // so a draining worker has to be absent here: its KV subscription and block
    // accounting must stop now, when it stops being a target, not when its Pod
    // object finally disappears.
    let active: Vec<Arc<Worker>> = all.iter().filter(|w| w.is_active()).cloned().collect();
    policy.sync_workers(&active);
    // The breaker keeps entries for draining workers -- they are still serving
    // in-flight requests, and failures there are still theirs. Entries go when
    // the record does.
    breaker.retain_workers(&all.iter().map(|w| w.worker_id.clone()).collect());
    pool.store(Arc::new(Snapshot::build(all)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pod(name: &str, annotated: bool, phase: &str, terminating: bool) -> Value {
        let mut meta = json!({ "name": name });
        if annotated {
            meta["annotations"] = json!({
                WORKER_INFO_ANNOTATION: json!({
                    "worker_id": "10.0.0.1:8080",
                    "url": "http://10.0.0.1:8080",
                    "model_name": "m",
                    "engine": "sglang",
                }).to_string()
            });
        }
        if terminating {
            meta["deletionTimestamp"] = json!("2026-08-04T03:00:00Z");
        }
        json!({ "metadata": meta, "status": { "phase": phase } })
    }

    fn running(name: &str) -> Value {
        pod(name, true, "Running", false)
    }

    fn ids(tracked: &HashMap<String, Tracked>) -> Vec<String> {
        let all: Vec<Arc<Worker>> = tracked.values().map(|t| t.worker.clone()).collect();
        Snapshot::build(all)
            .list_active("m", crate::pool::DisaggMode::Mixed)
            .iter()
            .map(|w| w.worker_id.clone())
            .collect()
    }

    #[test]
    fn a_running_annotated_pod_registers() {
        let mut t = HashMap::new();
        assert!(handle_pod(&running("w-0"), false, &mut t));
        assert_eq!(ids(&t), vec!["10.0.0.1:8080"]);
    }

    #[test]
    fn a_terminating_pod_leaves_routing_while_still_running() {
        // The case this exists for. `phase` is still Running -- only the
        // deletion timestamp separates a healthy worker from one inside its
        // preStop delay, and every request routed there in that window is work
        // that gets cut.
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        assert!(handle_pod(
            &pod("w-0", true, "Running", true),
            false,
            &mut t
        ));

        assert!(ids(&t).is_empty(), "must not be a routing candidate");
        assert_eq!(t.len(), 1, "but the record has to survive the drain");
    }

    #[test]
    fn a_draining_worker_is_announced_once_not_once_per_event() {
        // Repeated MODIFIED events for a terminating Pod are routine.
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        assert!(handle_pod(
            &pod("w-0", true, "Running", true),
            false,
            &mut t
        ));
        assert!(
            !handle_pod(&pod("w-0", true, "Running", true), false, &mut t),
            "the second one changes nothing, so it must not republish"
        );
    }

    #[test]
    fn clearing_the_annotation_removes_the_worker() {
        // The kill path with no deletionTimestamp: a liveness probe restart or
        // a manual kill leaves the Pod object untouched, so the annotation the
        // worker clears on SIGTERM is the only signal there is.
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        assert!(handle_pod(
            &pod("w-0", false, "Running", false),
            false,
            &mut t
        ));
        assert!(t.is_empty());
    }

    #[test]
    fn a_deleted_pod_is_removed() {
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        assert!(handle_pod(&running("w-0"), true, &mut t));
        assert!(t.is_empty());
    }

    #[test]
    fn a_pod_that_stopped_running_is_removed() {
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        assert!(handle_pod(
            &pod("w-0", true, "Failed", false),
            false,
            &mut t
        ));
        assert!(t.is_empty());
    }

    #[test]
    fn an_unchanged_pod_does_not_republish() {
        // Pods emit MODIFIED for things the router does not act on, including
        // the resourceVersion bump from every heartbeat annotation refresh.
        // Republishing on those rebuilds the snapshot and re-runs policy
        // reconciliation several times a second per worker.
        let mut t = HashMap::new();
        assert!(handle_pod(&running("w-0"), false, &mut t));
        assert!(!handle_pod(&running("w-0"), false, &mut t));
    }

    #[test]
    fn a_worker_that_changed_address_republishes() {
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        let mut moved = running("w-0");
        moved["metadata"]["annotations"][WORKER_INFO_ANNOTATION] = json!(json!({
            "worker_id": "10.0.0.9:8080",
            "url": "http://10.0.0.9:8080",
            "model_name": "m",
            "engine": "sglang",
        })
        .to_string());
        assert!(handle_pod(&moved, false, &mut t));
        assert_eq!(ids(&t), vec!["10.0.0.9:8080"]);
    }

    #[test]
    fn a_bad_annotation_is_ignored_rather_than_dropping_the_worker() {
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        let mut bad = running("w-0");
        bad["metadata"]["annotations"][WORKER_INFO_ANNOTATION] = json!("{not json");
        assert!(!handle_pod(&bad, false, &mut t));
        assert_eq!(ids(&t), vec!["10.0.0.1:8080"], "the good record stands");
    }

    #[test]
    fn a_draining_worker_is_absent_from_the_policys_view() {
        // The reconcile the policy gets must not list a worker that is no
        // longer a target: its KV subscription and block accounting have to
        // stop when it stops receiving, not when its Pod object disappears.
        let mut t = HashMap::new();
        handle_pod(&running("w-0"), false, &mut t);
        handle_pod(&pod("w-0", true, "Running", true), false, &mut t);

        let all: Vec<Arc<Worker>> = t.values().map(|x| x.worker.clone()).collect();
        let active: Vec<_> = all.iter().filter(|w| w.is_active()).collect();
        assert_eq!(all.len(), 1, "still reported by /v1/workers");
        assert!(active.is_empty(), "but not reconciled as a live target");
    }
}
