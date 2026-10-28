"""Print one window of the PP4 prefill ring as text: for each rank, its GPU step spans and the scheduler thread's main
phases (recv, send commit, get_new_batch_prefill, run_batch, process_batch_result, syncs > 5 ms) in time order, with
times in ms from the window start, so the critical path through a few beats can be read by eye.
  python3 prefill_ring_timeline.py TRACE_PP0 .. TRACE_PP3 [--from-step 40] [--steps 6] [--min-ms 3]"""

import argparse
import bisect
import collections
import gzip
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs=4)
ap.add_argument("--from-step", type=int, default=40)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--min-ms", type=float, default=3.0)
a = ap.parse_args()

PH = [
    ("recv", r"parallel_state\.py\(\d+\): recv_(tensor_dict|object)$"),
    ("commit", r"scheduler_pp_mixin\.py\(\d+\): _pp_commit_comm_work$"),
    ("newbatch", r"scheduler\.py\(\d+\): get_new_batch_prefill$"),
    ("evict", r"unified_radix_cache\.py\(\d+\): evict_for_alloc$"),
    ("loadback", r"unified_radix_cache\.py\(\d+\): init_load_back$"),
    ("run_batch", r"scheduler\.py\(\d+\): run_batch$"),
    ("result", r"scheduler\.py\(\d+\): process_batch_result$"),
    ("chunk", r"prefill\.py\(\d+\): process_prefill_chunk$"),
    ("bootstrap", r"prefill\.py\(\d+\): process_bootstrapped_queue$"),
    ("inflight", r"prefill\.py\(\d+\): process_disagg_prefill_inflight_queue$"),
]


def load(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")), key=lambda e: e["ts"])
    tid = ann[0]["tid"]
    ext = [e for e in ann if not e["name"].startswith("step[DRAFT")]
    ext_ts = [e["ts"] for e in ext]
    ann_ts = [e["ts"] for e in ann]
    ann_step = [bisect.bisect_right(ext_ts, e["ts"]) - 1 for e in ann]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cnt = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
    main = cnt.most_common(1)[0][0]
    ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
    launch = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
              and "correlation" in e.get("args", {}) and e.get("tid") == tid}
    g0, g1 = [None] * len(ext), [None] * len(ext)
    for i, k in enumerate(ops):
        lt = launch.get(k["args"].get("correlation"))
        if lt is None:
            continue
        ai = bisect.bisect_right(ann_ts, lt) - 1
        if ai < 0 or lt > ann[ai]["ts"] + ann[ai]["dur"] or ann_step[ai] < 0:
            continue
        si = ann_step[ai]
        en = min(k["ts"] + k["dur"], ops[i + 1]["ts"]) if i + 1 < len(ops) else k["ts"] + k["dur"]
        g0[si] = k["ts"] if g0[si] is None else min(g0[si], k["ts"])
        g1[si] = en if g1[si] is None else max(g1[si], en)
    phases = []
    for e in ev:
        if e.get("ph") != "X" or e.get("tid") != tid:
            continue
        if e.get("cat") == "python_function":
            for n, r in PH:
                if re.search(r, e["name"]):
                    phases.append((e["ts"], e["dur"], n))
        elif e.get("cat") == "cuda_runtime" and re.search(r"Memcpy|Synchronize", e["name"]) and e["dur"] > 5000:
            phases.append((e["ts"], e["dur"], "sync"))
    phases.sort()
    return dict(g0=g0, g1=g1, phases=phases)


R = [load(p) for p in a.traces]
i0, i1 = a.from_step, a.from_step + a.steps
t0 = min(R[r]["g0"][i0] for r in range(4) if R[r]["g0"][i0] is not None) - 400e3
t1 = max(R[r]["g1"][i1] for r in range(4) if R[r]["g1"][i1] is not None) + 100e3
ms = lambda t: (t - t0) / 1e3
for r in range(4):
    print(f"=== PP{r}")
    rows = []
    for i in range(i0 - 2, i1 + 1):
        if R[r]["g0"][i] is not None and t0 <= R[r]["g0"][i] <= t1:
            rows.append((R[r]["g0"][i], f"GPU step {i:3d}  {ms(R[r]['g0'][i]):8.1f} -> {ms(R[r]['g1'][i]):8.1f}  ({(R[r]['g1'][i] - R[r]['g0'][i]) / 1e3:.0f} ms)"))
    for ts, dur, n in R[r]["phases"]:
        if t0 <= ts <= t1 and dur >= a.min_ms * 1e3:
            rows.append((ts, f"      {n:10s} {ms(ts):8.1f} -> {ms(ts + dur):8.1f}  ({dur / 1e3:.0f} ms)"))
    for _, s in sorted(rows):
        print(s)
