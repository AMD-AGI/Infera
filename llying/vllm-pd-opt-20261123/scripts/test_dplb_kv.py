#!/usr/bin/env python3
"""CPU check of the KV-aware DP balancing overlay: load the patched core_client.py as a module (in the vLLM image) and
drive DPLBAsyncMPClient.get_core_engine_for_request with a stub client.
  VPD_DP_LB_KV_WEIGHT=W test_dplb_kv.py PATCHED_CORE_CLIENT_PY
Checks: with W the low-KV rank wins over an equal-count high-KV rank; the chosen rank's usage rises by prompt / cap;
with W=0 the choice equals v0.34.0's (count-only) choice.
"""
import importlib.util
import os
import sys
from collections import defaultdict
from types import SimpleNamespace


def load(path, w):
    os.environ["VPD_DP_LB_KV_WEIGHT"] = str(w)
    spec = importlib.util.spec_from_file_location(f"vpd_core_client_{w}".replace(".", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pick(m, counts, prompt_len=100_000):
    stub = SimpleNamespace(
        streaming_reqs=set(), reqs_in_flight={}, lb_engines=[list(c) for c in counts], eng_start_index=0,
        client_count=1, engine_inflight=defaultdict(int), core_engines=[b"\x00\x00", b"\x01\x00", b"\x02\x00", b"\x03\x00"],
    )
    req = SimpleNamespace(resumable=False, request_id="r", data_parallel_rank=None, pooling_params=None,
                          prompt_token_ids=[0] * prompt_len)
    eng = m.DPLBAsyncMPClient.get_core_engine_for_request(stub, req)
    return stub.core_engines.index(eng), stub.lb_engines


def main():
    path = sys.argv[1]
    w = float(os.environ.get("VPD_DP_LB_KV_WEIGHT") or 14)
    m = load(path, w)
    assert m._VPD_LB_KV_W == w, m._VPD_LB_KV_W
    # rank 0 has the fewest requests but a full KV; rank 2 one request more and a low KV
    counts = [[0, 14, 0.97], [0, 16, 0.80], [0, 15, 0.60], [0, 16, 0.75]]
    i, after = pick(m, counts)
    assert i == 2, (i, after)
    assert abs(after[2][2] - (0.60 + 100_000 / m._VPD_LB_KV_CAP)) < 1e-9, after
    print(f"W={w}: picked rank {i} (counts {counts}); usage after {after[2][2]:.4f}")
    m0 = load(path, 0)
    assert m0._VPD_LB_KV_W == 0
    i0, after0 = pick(m0, counts)
    assert i0 == 0 and after0[0][2] == 0.97, (i0, after0)
    print(f"W=0: picked rank {i0} (count-only, as v0.34.0)")
    print("dplb kv overlay test passed")


if __name__ == "__main__":
    main()
