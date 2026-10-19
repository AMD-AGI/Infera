///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Data-parallel rank steering (mirrors `infera.router.dp_routing`). SGLang and
//! vLLM read the rank from a header, ATOM from the request body.

use axum::body::Bytes;
use serde_json::{Map, Value};

use crate::pool::{RouteTarget, Worker};

/// Header that pins a request to a DP rank (SGLang + vLLM both honour it).
pub const DP_RANK_HEADER: &str = "X-Data-Parallel-Rank";

/// ATOM's request field for the same thing; it does not read the header. An
/// explicit rank wins over ATOM's own load balancing and session affinity.
pub const DP_RANK_BODY_FIELD: &str = "data_parallel_rank";

fn rank_in_body(w: &Worker) -> bool {
    w.engine == "atom"
}

/// The rank to send as `DP_RANK_HEADER`: none when the target has no rank or
/// its engine reads the rank from the body.
pub fn header_rank(target: &RouteTarget) -> Option<i64> {
    target.dp_rank.filter(|_| !rank_in_body(&target.worker))
}

/// Pin an ATOM target's rank in the body. No-op for other engines and for
/// targets without a rank.
pub fn insert_body_rank(body: &mut Map<String, Value>, target: &RouteTarget) {
    if let Some(rank) = target.dp_rank.filter(|_| rank_in_body(&target.worker)) {
        body.insert(DP_RANK_BODY_FIELD.into(), Value::from(rank));
    }
}

/// [`insert_body_rank`] on an encoded body. Returns it unchanged when there is
/// nothing to pin or it is not a JSON object (the engine rejects that anyway).
pub fn with_body_rank(raw: &Bytes, target: &RouteTarget) -> Bytes {
    if target.dp_rank.is_none() || !rank_in_body(&target.worker) {
        return raw.clone();
    }
    match serde_json::from_slice::<Value>(raw) {
        Ok(Value::Object(mut body)) => {
            insert_body_rank(&mut body, target);
            Bytes::from(serde_json::to_vec(&body).expect("a parsed body re-encodes"))
        }
        _ => raw.clone(),
    }
}

/// Rewrite the low residue so `room % dp_size == dp_rank`, keeping the high
/// bits random. SGLang's `follow_bootstrap_room` balancer derives the prefill
/// rank from `bootstrap_room % dp_size` and rejects a leg that lands elsewhere.
pub fn align_room_to_prefill_rank(room: u64, prefill: &RouteTarget) -> u64 {
    match (prefill.dp_rank, prefill.worker.dp_size) {
        (Some(rank), Some(size)) if size > 0 => {
            let size = size as u64;
            room - (room % size) + rank as u64
        }
        _ => room,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::Worker;
    use std::sync::Arc;

    fn target(dp_rank: Option<i64>, dp_size: Option<i64>) -> RouteTarget {
        engine_target("sglang", dp_rank, dp_size)
    }

    fn engine_target(engine: &str, dp_rank: Option<i64>, dp_size: Option<i64>) -> RouteTarget {
        let w: Worker = serde_json::from_value(serde_json::json!({
            "worker_id": "w", "url": "http://x", "engine": engine, "dp_size": dp_size
        }))
        .unwrap();
        RouteTarget {
            worker: Arc::new(w),
            dp_rank,
        }
    }

    #[test]
    fn atom_rank_goes_in_the_body_not_a_header() {
        let t = engine_target("atom", Some(2), Some(4));
        assert_eq!(header_rank(&t), None);
        let raw = Bytes::from_static(br#"{"model":"m","stream":true}"#);
        let body: Value = serde_json::from_slice(&with_body_rank(&raw, &t)).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"model": "m", "stream": true, "data_parallel_rank": 2})
        );
    }

    #[test]
    fn header_engines_keep_the_body() {
        for engine in ["sglang", "vllm"] {
            let t = engine_target(engine, Some(2), Some(4));
            assert_eq!(header_rank(&t), Some(2));
            let raw = Bytes::from_static(br#"{"model":"m"}"#);
            assert_eq!(with_body_rank(&raw, &t), raw);
        }
    }

    #[test]
    fn rankless_atom_target_is_untouched() {
        let t = engine_target("atom", None, None);
        assert_eq!(header_rank(&t), None);
        let mut body = Map::new();
        insert_body_rank(&mut body, &t);
        assert!(body.is_empty());
        let raw = Bytes::from_static(b"not json");
        assert_eq!(with_body_rank(&raw, &t), raw);
    }

    #[test]
    fn aligns_residue_to_rank() {
        let r = align_room_to_prefill_rank(1003, &target(Some(2), Some(4)));
        assert_eq!(r % 4, 2);
        // high bits preserved (only the low residue is rewritten)
        assert_eq!(r / 4, 1003 / 4);
    }

    #[test]
    fn no_rank_or_zero_size_is_identity() {
        assert_eq!(align_room_to_prefill_rank(777, &target(None, Some(4))), 777);
        // size==0 must not divide by zero — falls through to identity
        assert_eq!(
            align_room_to_prefill_rank(777, &target(Some(1), Some(0))),
            777
        );
        assert_eq!(align_room_to_prefill_rank(777, &target(Some(1), None)), 777);
    }
}
