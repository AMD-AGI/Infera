#!/usr/bin/env python3
"""Build the KV-aware DP load-balancing overlay of vllm/v1/engine/core_client.py (v0.34.0) from the image's own file.

vLLM's DP balancer (DPLBAsyncMPClient.get_core_engine_for_request) scores a rank by waiting + running and adds KV only as
a penalty on waiting requests. On the GLM-5.2 PD decode the step time follows the slowest rank, and the a3 step fit
(26.7 + 0.74 max_req + 3.9 max_ctx(M)) puts a full rank's KV (about 2.96M tokens) at about 15.6 requests. With
VPD_DP_LB_KV_WEIGHT=W the score also gets W * kv_cache_usage, and between the coordinator's 100 ms snapshots the chosen
rank's usage is raised by the request's prompt tokens / VPD_DP_LB_KV_CAP_TOKENS (like the local waiting increment).
W unset or 0 keeps v0.34.0's behaviour exactly.
  make_dplb_kv_overlay.py ORIG OUT
"""
import sys

HEAD_OLD = "import weakref\n"
HEAD_NEW = HEAD_OLD + (
    "import os as _vpd_os\n"
    "\n"
    "_VPD_LB_KV_W = float(_vpd_os.environ.get(\"VPD_DP_LB_KV_WEIGHT\") or 0)\n"
    "_VPD_LB_KV_CAP = float(_vpd_os.environ.get(\"VPD_DP_LB_KV_CAP_TOKENS\") or 2955712)\n"
    "if _VPD_LB_KV_W:\n"
    "    print(f\"[vpd] DP LB KV weight {_VPD_LB_KV_W} cap {_VPD_LB_KV_CAP:.0f} tokens\", file=sys.stderr, flush=True)\n"
)

SCORE_OLD = "                score: float = max(self.client_count * inflight, waiting + running)\n"
SCORE_NEW = SCORE_OLD + (
    "                if _VPD_LB_KV_W:\n"
    "                    score += _VPD_LB_KV_W * kv_cache_usage\n"
)

PICK_OLD = "            current_counts[eng_index][0] += self.client_count\n"
PICK_NEW = PICK_OLD + (
    "            if _VPD_LB_KV_W:\n"
    "                ids = request.prompt_token_ids\n"
    "                current_counts[eng_index][2] += (len(ids) if ids is not None else 0) / _VPD_LB_KV_CAP\n"
)


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    for old, new in ((HEAD_OLD, HEAD_NEW), (SCORE_OLD, SCORE_NEW), (PICK_OLD, PICK_NEW)):
        n = s.count(old)
        if n != 1:
            sys.exit(f"expected exactly one {old.strip()!r} in {src}, found {n}")
        s = s.replace(old, new)
    open(out, "w").write(s)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
