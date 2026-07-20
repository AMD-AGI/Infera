///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! SGLang data-parallel rank steering (mirrors `infera.router.dp_routing`).

use crate::pool::RouteTarget;

/// Header that pins a request to a DP rank (SGLang + vLLM both honour it).
pub const DP_RANK_HEADER: &str = "X-Data-Parallel-Rank";

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
        let w: Worker = serde_json::from_value(serde_json::json!({
            "worker_id": "w", "url": "http://x", "dp_size": dp_size
        }))
        .unwrap();
        RouteTarget {
            worker: Arc::new(w),
            dp_rank,
        }
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
