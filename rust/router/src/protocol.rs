///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! PD coordination protocol. Covers SGLang bootstrap (concurrent topology):
//! both legs get the same top-level `bootstrap_host/port/room` and are POSTed
//! concurrently. Other connectors (vLLM mooncake/mori, atom) are served by the
//! Python backend.

use serde_json::{Map, Value};

use crate::pool::Worker;

const SGLANG_BOOTSTRAP: &str = "sglang-bootstrap";
const VLLM_MOONCAKE: &str = "vllm-mooncake";

/// The PD coordination protocol both legs speak. Selects how `disagg::dispatch`
/// shapes the prefill vs decode bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdProtocol {
    /// SGLang bootstrap: both legs get the same top-level bootstrap fields.
    SglangBootstrap,
    /// vLLM Mooncake: asymmetric `kv_transfer_params` — prefill pushes KV, decode
    /// pulls it via the prefill's bootstrap server.
    VllmMooncake,
}

/// Resolve the PD protocol from the workers' advertised `disagg_meta.protocol`.
/// Both legs must speak the same supported protocol; errors on an unsupported
/// connector (atom/mori — use the Python backend) or a P/D mismatch.
pub fn resolve_pd_protocol(p: &Worker, d: &Worker) -> anyhow::Result<PdProtocol> {
    let pp = p.disagg_meta.get("protocol").and_then(|v| v.as_str());
    let dp = d.disagg_meta.get("protocol").and_then(|v| v.as_str());
    match (pp, dp) {
        (Some(SGLANG_BOOTSTRAP), Some(SGLANG_BOOTSTRAP)) => Ok(PdProtocol::SglangBootstrap),
        (Some(VLLM_MOONCAKE), Some(VLLM_MOONCAKE)) => Ok(PdProtocol::VllmMooncake),
        _ => anyhow::bail!(
            "unsupported or mismatched PD protocol (prefill={pp:?}, decode={dp:?}); \
             rust backend supports sglang-bootstrap and vllm-mooncake — use \
             --router-backend python for others (atom/mori)"
        ),
    }
}

/// Both workers must advertise the sglang-bootstrap protocol. Kept for callers
/// that only handle the SGLang path; new code should use `resolve_pd_protocol`.
pub fn require_sglang_bootstrap(p: &Worker, d: &Worker) -> anyhow::Result<()> {
    match resolve_pd_protocol(p, d)? {
        PdProtocol::SglangBootstrap => Ok(()),
        PdProtocol::VllmMooncake => anyhow::bail!("expected sglang-bootstrap, got vllm-mooncake"),
    }
}

/// Inject SGLang's bootstrap fields (from the prefill worker's advertised
/// address) into a leg body. Decode and prefill legs get identical fields.
pub fn annotate_sglang(
    body: &mut Map<String, Value>,
    prefill: &Worker,
    room: u64,
) -> anyhow::Result<()> {
    let addr = prefill
        .disagg_meta
        .get("params")
        .and_then(|v| v.get("bootstrap_addr"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "prefill {} missing disagg_meta.params.bootstrap_addr",
                prefill.worker_id
            )
        })?;
    let (host, port) = addr
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("bad bootstrap_addr {addr:?} (want host:port)"))?;
    let port: i64 = port
        .parse()
        .map_err(|_| anyhow::anyhow!("bad bootstrap port in {addr:?}"))?;
    body.insert("bootstrap_host".into(), Value::from(host));
    body.insert("bootstrap_port".into(), Value::from(port));
    body.insert("bootstrap_room".into(), Value::from(room));
    Ok(())
}

// --- vLLM Mooncake ---------------------------------------------------------
// Mirrors `infera/router/disagg_protocols/vllm_mooncake.py`: a concurrent push
// where the prefill worker runs prefill (+1 token) and pushes KV over Mooncake,
// and the decode worker discovers the prefill via its bootstrap server and
// generates the rest. The two legs get ASYMMETRIC bodies (unlike SGLang).

/// Deterministic transfer_id from the room so the concurrently-dispatched
/// prefill/decode legs agree without threading state (mirrors the reference
/// proxy's `xfer-<id>`).
fn transfer_id_for_room(room: u64) -> String {
    format!("xfer-{room:032x}")
}

fn required_param<'a>(w: &'a Worker, key: &str) -> anyhow::Result<&'a str> {
    w.disagg_meta
        .get("params")
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("worker {} missing disagg_meta.params.{key}", w.worker_id))
}

/// Prefill leg: prefill only (max_tokens=1, non-streaming) plus the
/// `kv_transfer_params` that tell vLLM to push its KV to a remote decode.
pub fn annotate_vllm_prefill(body: &mut Map<String, Value>, room: u64) {
    body.insert("max_tokens".into(), Value::from(1));
    body.insert("stream".into(), Value::from(false));
    body.remove("stream_options");
    if body.contains_key("max_completion_tokens") {
        body.insert("max_completion_tokens".into(), Value::from(1));
    }
    let mut kv = Map::new();
    kv.insert("do_remote_decode".into(), Value::Bool(true));
    kv.insert("do_remote_prefill".into(), Value::Bool(false));
    kv.insert(
        "transfer_id".into(),
        Value::from(transfer_id_for_room(room)),
    );
    body.insert("kv_transfer_params".into(), Value::Object(kv));
}

/// Decode leg: generate the remaining tokens, pulling the prefilled KV from the
/// prefill's Mooncake bootstrap. For a DP>1 prefill, address the steered rank's
/// engine (`engine_id_dp{room % dp_size}`, recovered the same way vLLM does).
pub fn annotate_vllm_decode(
    body: &mut Map<String, Value>,
    prefill: &Worker,
    room: u64,
) -> anyhow::Result<()> {
    // Prefill already produced 1 token; decode does the rest (floor 1).
    for k in ["max_tokens", "max_completion_tokens"] {
        if let Some(n) = body.get(k).and_then(Value::as_i64) {
            body.insert(k.into(), Value::from((n - 1).max(1)));
        }
    }
    let mut engine_id = required_param(prefill, "engine_id")?.to_string();
    let dp_size = prefill.dp_size.unwrap_or(1);
    if dp_size > 1 {
        engine_id = format!("{engine_id}_dp{}", room % dp_size as u64);
    }
    let bootstrap_addr = required_param(prefill, "bootstrap_addr")?.to_string();
    let mut kv = Map::new();
    kv.insert("do_remote_decode".into(), Value::Bool(false));
    kv.insert("do_remote_prefill".into(), Value::Bool(true));
    kv.insert("remote_bootstrap_addr".into(), Value::from(bootstrap_addr));
    kv.insert("remote_engine_id".into(), Value::from(engine_id));
    kv.insert(
        "transfer_id".into(),
        Value::from(transfer_id_for_room(room)),
    );
    body.insert("kv_transfer_params".into(), Value::Object(kv));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn worker(meta: Value) -> Worker {
        serde_json::from_value(json!({
            "worker_id": "w", "url": "http://x", "disagg_meta": meta
        }))
        .unwrap()
    }

    fn sglang() -> Value {
        json!({"protocol": "sglang-bootstrap", "params": {"bootstrap_addr": "10.0.0.1:9000"}})
    }

    fn vllm(dp_size: i64) -> Worker {
        serde_json::from_value(json!({
            "worker_id": "p", "url": "http://x", "dp_size": dp_size,
            "disagg_meta": {
                "protocol": "vllm-mooncake",
                "params": {"engine_id": "eng-42", "bootstrap_addr": "http://10.0.0.1:8998"}
            }
        }))
        .unwrap()
    }

    #[test]
    fn resolve_protocol_dispatches_by_tag() {
        let sp = worker(sglang());
        let sd = worker(json!({"protocol": "sglang-bootstrap"}));
        assert_eq!(
            resolve_pd_protocol(&sp, &sd).unwrap(),
            PdProtocol::SglangBootstrap
        );

        let vp = vllm(1);
        let vd = worker(json!({"protocol": "vllm-mooncake"}));
        assert_eq!(
            resolve_pd_protocol(&vp, &vd).unwrap(),
            PdProtocol::VllmMooncake
        );

        // mismatched P/D protocols and unsupported connectors error out.
        assert!(resolve_pd_protocol(&sp, &vd).is_err());
        assert!(resolve_pd_protocol(&worker(json!({"protocol": "mori"})), &vd).is_err());
    }

    #[test]
    fn vllm_prefill_leg_is_prefill_only_and_pushes_kv() {
        let mut body = Map::new();
        body.insert("max_tokens".into(), Value::from(128));
        body.insert("max_completion_tokens".into(), Value::from(128));
        body.insert("stream_options".into(), json!({"include_usage": true}));
        annotate_vllm_prefill(&mut body, 42);
        assert_eq!(body["max_tokens"], 1);
        assert_eq!(body["max_completion_tokens"], 1);
        assert_eq!(body["stream"], false);
        assert!(!body.contains_key("stream_options"));
        let kv = &body["kv_transfer_params"];
        assert_eq!(kv["do_remote_decode"], true);
        assert_eq!(kv["do_remote_prefill"], false);
        assert_eq!(kv["transfer_id"], format!("xfer-{:032x}", 42));
    }

    #[test]
    fn vllm_decode_leg_pulls_kv_and_decrements_tokens() {
        let p = vllm(1);
        let mut body = Map::new();
        body.insert("max_tokens".into(), Value::from(128));
        annotate_vllm_decode(&mut body, &p, 42).unwrap();
        assert_eq!(body["max_tokens"], 127); // prefill did 1
        let kv = &body["kv_transfer_params"];
        assert_eq!(kv["do_remote_prefill"], true);
        assert_eq!(kv["do_remote_decode"], false);
        assert_eq!(kv["remote_bootstrap_addr"], "http://10.0.0.1:8998");
        assert_eq!(kv["remote_engine_id"], "eng-42"); // dp_size=1 -> base engine_id
        assert_eq!(kv["transfer_id"], format!("xfer-{:032x}", 42));
    }

    #[test]
    fn vllm_decode_addresses_steered_dp_rank() {
        let p = vllm(4);
        let mut body = Map::new();
        // room % dp_size selects the prefill rank (3) -> engine_id_dp3
        annotate_vllm_decode(&mut body, &p, 7).unwrap();
        assert_eq!(body["kv_transfer_params"]["remote_engine_id"], "eng-42_dp3");
        // floor: with no/max_tokens=1 decode stays >= 1
        let mut b2 = Map::new();
        b2.insert("max_tokens".into(), Value::from(1));
        annotate_vllm_decode(&mut b2, &p, 7).unwrap();
        assert_eq!(b2["max_tokens"], 1);
    }

    #[test]
    fn vllm_decode_errors_on_missing_params() {
        let bad: Worker =
            serde_json::from_value(json!({"worker_id": "p", "url": "http://x", "dp_size": 1,
                "disagg_meta": {"protocol": "vllm-mooncake", "params": {}}}))
            .unwrap();
        let mut body = Map::new();
        assert!(annotate_vllm_decode(&mut body, &bad, 1).is_err());
    }

    #[test]
    fn require_bootstrap_matches_both_legs() {
        let p = worker(sglang());
        let d = worker(json!({"protocol": "sglang-bootstrap"}));
        assert!(require_sglang_bootstrap(&p, &d).is_ok());

        let other = worker(json!({"protocol": "mooncake"}));
        assert!(require_sglang_bootstrap(&other, &d).is_err());
        assert!(require_sglang_bootstrap(&p, &worker(json!({}))).is_err());
    }

    #[test]
    fn annotate_injects_bootstrap_fields() {
        let p = worker(sglang());
        let mut body = Map::new();
        annotate_sglang(&mut body, &p, 42).unwrap();
        assert_eq!(body["bootstrap_host"], "10.0.0.1");
        assert_eq!(body["bootstrap_port"], 9000);
        assert_eq!(body["bootstrap_room"], 42);
    }

    #[test]
    fn annotate_rejects_bad_addr() {
        // missing addr entirely
        let mut b = Map::new();
        assert!(
            annotate_sglang(&mut b, &worker(json!({"protocol": "sglang-bootstrap"})), 1).is_err()
        );
        // no host:port separator
        let mut b = Map::new();
        let w =
            worker(json!({"protocol": "sglang-bootstrap", "params": {"bootstrap_addr": "noport"}}));
        assert!(annotate_sglang(&mut b, &w, 1).is_err());
        // non-numeric port
        let mut b = Map::new();
        let w =
            worker(json!({"protocol": "sglang-bootstrap", "params": {"bootstrap_addr": "h:abc"}}));
        assert!(annotate_sglang(&mut b, &w, 1).is_err());
    }
}
