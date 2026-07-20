///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! etcd worker discovery via the v3 HTTP/JSON gateway (same wire as the Python
//! Registry): snapshot the `<prefix>` range, then watch PUT/DELETE and swap a
//! fresh immutable `Snapshot` on each change.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;

use crate::policy::Policy;
use crate::pool::{SharedPool, Snapshot, Worker};

pub async fn run(base: String, prefix: String, pool: SharedPool, policy: Arc<dyn Policy>) {
    let prefix = if prefix.ends_with('/') {
        prefix
    } else {
        format!("{prefix}/")
    };
    let mut backoff = 1u64;
    loop {
        match discover_once(&base, &prefix, &pool, &policy).await {
            Ok(()) => backoff = 1,
            Err(e) => {
                tracing::warn!("etcd discovery error: {e}; retry in {backoff}s");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(30);
            }
        }
    }
}

fn b64(s: &[u8]) -> String {
    STANDARD.encode(s)
}

/// etcd convention: range_end = prefix with the last byte incremented selects
/// the whole prefix.
fn range_end(prefix: &str) -> Vec<u8> {
    let mut pb = prefix.as_bytes().to_vec();
    if let Some(last) = pb.last_mut() {
        *last += 1;
    }
    pb
}

#[derive(Serialize)]
struct RangeReq {
    key: String,
    range_end: String,
}

async fn discover_once(
    base: &str,
    prefix: &str,
    pool: &SharedPool,
    policy: &Arc<dyn Policy>,
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().build()?;
    let re = range_end(prefix);

    // 1. snapshot the current fleet
    let req = RangeReq {
        key: b64(prefix.as_bytes()),
        range_end: b64(&re),
    };
    let resp: Value = client
        .post(format!("{base}/v3/kv/range"))
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut workers: HashMap<String, Arc<Worker>> = HashMap::new();
    if let Some(kvs) = resp.get("kvs").and_then(|k| k.as_array()) {
        for kv in kvs {
            if let Some((id, w)) = parse_kv(prefix, kv) {
                workers.insert(id, Arc::new(w));
            }
        }
    }
    tracing::info!(
        "etcd snapshot: {} worker(s) under {}",
        workers.len(),
        prefix
    );
    publish(pool, policy, &workers);

    // 2. watch for changes (long-lived NDJSON stream)
    let create = serde_json::json!({
        "create_request": { "key": b64(prefix.as_bytes()), "range_end": b64(&re) }
    });
    let resp = client
        .post(format!("{base}/v3/watch"))
        .header("content-type", "application/json")
        .body(format!("{create}\n"))
        .send()
        .await?
        .error_for_status()?;

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = &line[..line.len() - 1];
            if line.is_empty() {
                continue;
            }
            if let Ok(msg) = serde_json::from_slice::<Value>(line) {
                if apply_watch(prefix, &msg, &mut workers) {
                    publish(pool, policy, &workers);
                }
            }
        }
    }
    Ok(())
}

fn publish(pool: &SharedPool, policy: &Arc<dyn Policy>, workers: &HashMap<String, Arc<Worker>>) {
    let all: Vec<Arc<Worker>> = workers.values().cloned().collect();
    // Let cost-aware policies reconcile per-worker state (kv-event subscriptions,
    // load bookkeeping) against the new fleet before we swap the snapshot in.
    policy.sync_workers(&all);
    pool.store(Arc::new(Snapshot::build(all)));
}

/// Decode one etcd kv entry into (worker_id, Worker). Logs + skips malformed.
fn parse_kv(prefix: &str, kv: &Value) -> Option<(String, Worker)> {
    let key_b64 = kv.get("key")?.as_str()?;
    let val_b64 = kv.get("value")?.as_str()?;
    let key = String::from_utf8(STANDARD.decode(key_b64).ok()?).ok()?;
    let id = key.strip_prefix(prefix)?.to_string();
    let val = STANDARD.decode(val_b64).ok()?;
    match serde_json::from_slice::<Worker>(&val) {
        Ok(w) => Some((id, w)),
        Err(e) => {
            tracing::warn!("etcd: bad worker value at {key}: {e}");
            None
        }
    }
}

/// Apply a watch message to the working set. Returns true if it changed.
fn apply_watch(prefix: &str, msg: &Value, workers: &mut HashMap<String, Arc<Worker>>) -> bool {
    // The JSON gateway wraps each gRPC message under a top-level "result".
    let result = msg.get("result").unwrap_or(msg);
    let events = match result.get("events").and_then(|e| e.as_array()) {
        Some(e) => e,
        None => return false,
    };
    let mut changed = false;
    for ev in events {
        let ev_type = ev.get("type").and_then(|t| t.as_str()).unwrap_or("PUT");
        let kv = match ev.get("kv") {
            Some(k) => k,
            None => continue,
        };
        if ev_type == "DELETE" {
            // DELETE events only carry the key.
            let key = match kv
                .get("key")
                .and_then(|k| k.as_str())
                .and_then(|s| STANDARD.decode(s).ok())
                .and_then(|b| String::from_utf8(b).ok())
            {
                Some(k) => k,
                None => continue,
            };
            if let Some(id) = key.strip_prefix(prefix) {
                if workers.remove(id).is_some() {
                    changed = true;
                    tracing::info!("etcd: worker {id} removed");
                }
            }
        } else if let Some((id, w)) = parse_kv(prefix, kv) {
            tracing::info!("etcd: worker {id} registered/updated ({})", w.url);
            workers.insert(id, Arc::new(w));
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn kv(key: &str, value: &str) -> Value {
        json!({ "key": b64(key.as_bytes()), "value": b64(value.as_bytes()) })
    }

    #[test]
    fn range_end_increments_last_byte() {
        // "/infera/workers/" → last byte '/' (0x2f) becomes '0' (0x30)
        let re = range_end("/infera/workers/");
        let mut expected = b"/infera/workers/".to_vec();
        *expected.last_mut().unwrap() += 1;
        assert_eq!(re, expected);
    }

    #[test]
    fn parse_kv_decodes_worker_and_strips_prefix() {
        let prefix = "/infera/workers/";
        let val = r#"{"worker_id":"w1","url":"http://host:8000","model_name":"m"}"#;
        let (id, w) = parse_kv(prefix, &kv("/infera/workers/w1", val)).unwrap();
        assert_eq!(id, "w1");
        assert_eq!(w.url, "http://host:8000");

        // malformed value → skipped (None), not a panic
        assert!(parse_kv(prefix, &kv("/infera/workers/bad", "not json")).is_none());
    }

    #[test]
    fn apply_watch_handles_put_then_delete() {
        let prefix = "/infera/workers/";
        let mut workers: HashMap<String, Arc<Worker>> = HashMap::new();

        let put = json!({"result": {"events": [
            {"type": "PUT", "kv": kv("/infera/workers/w1", r#"{"worker_id":"w1","url":"u"}"#)}
        ]}});
        assert!(apply_watch(prefix, &put, &mut workers));
        assert!(workers.contains_key("w1"));

        let del = json!({"result": {"events": [
            {"type": "DELETE", "kv": {"key": b64(b"/infera/workers/w1")}}
        ]}});
        assert!(apply_watch(prefix, &del, &mut workers));
        assert!(workers.is_empty());

        // an events-less message is a no-op
        assert!(!apply_watch(prefix, &json!({"result": {}}), &mut workers));
    }
}
