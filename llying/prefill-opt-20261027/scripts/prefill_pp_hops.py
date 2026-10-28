"""Where the PP4 prefill ring loses time between stages, from the torch-profiler traces of all PP ranks (same node clock;
step i of every rank is the same microbatch). For each hop r -> r+1 and step i:
  hop      = GPU start of step i on r+1 - GPU end of step i on r       (time the microbatch spends between the two GPUs)
  late     = start of r+1's recv for step i - GPU end on r (> 0: r+1 was still busy elsewhere when the data could go)
  wait     = r+1 blocked inside that recv
  prep     = GPU start on r+1 - end of that recv                        (batch build + launch before the first kernel)
and the ring closure: PP0 GPU start of step i+L - PP3 GPU end of step i, for the pipeline depth L given (pp_size + async
depth). Also each rank's GPU step time, so beat = (sum of 4 GPU steps + 4 hops incl. closure) / L can be checked.
  python3 prefill_pp_hops.py TRACE_PP0 .. TRACE_PP3 [--depth 5] [--max-gap-ms 5000]"""

import argparse
import bisect
import gzip
import json
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs=4)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--max-gap-ms", type=float, default=5000)
a = ap.parse_args()

RECV = re.compile(r"parallel_state\.py\(\d+\): recv_(tensor_dict|object)$")


def load(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")), key=lambda e: e["ts"])
    tid = ann[0]["tid"]
    ext = [e for e in ann if not e["name"].startswith("step[DRAFT")]
    ext_ts = [e["ts"] for e in ext]
    ann_ts = [e["ts"] for e in ann]
    ann_step = [bisect.bisect_right(ext_ts, e["ts"]) - 1 for e in ann]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    streams = {}
    for k in gpu:
        if "nccl" not in k["name"].lower():
            streams[k["args"].get("stream")] = streams.get(k["args"].get("stream"), 0) + 1
    main = max(streams, key=streams.get)
    ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
    launch = {e["args"]["correlation"]: e["ts"] for e in ev
              if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}) and e.get("tid") == tid}
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
    recv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                  if e.get("ph") == "X" and e.get("tid") == tid and e.get("cat") == "python_function" and RECV.search(e["name"]))
    del ev, gpu, ops, launch
    return dict(n=len(ext), ann=ext_ts, g0=g0, g1=g1, recv=recv)


R = [load(p) for p in a.traces]
n = min(r["n"] for r in R)


def recv_for(r, i):
    """recv calls of rank r between the step i-1 and step i annotations: (first start, last end, blocked total)."""
    lo = R[r]["ann"][i - 1] if i > 0 else float("-inf")
    hi = R[r]["ann"][i]
    xs = [(s, e) for s, e in R[r]["recv"] if lo <= s and e <= hi]
    if not xs:
        return None
    return xs[0][0], xs[-1][1], sum(e - s for s, e in xs)


ok = [i for i in range(1, n) if all(R[r]["g0"][i] is not None and R[r]["g1"][i] is not None for r in range(4))
      and all(R[r]["g0"][i] - R[r]["g1"][i - 1] < a.max_gap_ms * 1e3 for r in range(4) if R[r]["g1"][i - 1] is not None)]
print(f"{len(ok)} steps with GPU spans on all 4 ranks (gaps < {a.max_gap_ms:.0f} ms)")
ms = lambda x: x / 1e3
for r in range(4):
    d = [ms(R[r]["g1"][i] - R[r]["g0"][i]) for i in ok]
    print(f"PP{r} GPU step: median {st.median(d):6.1f} mean {st.mean(d):6.1f} ms")


def summ(name, xs):
    xs = sorted(xs)
    if not xs:
        return f"{name} n=0"
    return f"{name} p50 {st.median(xs):6.1f} mean {st.mean(xs):6.1f} p90 {xs[int(.9 * len(xs))]:6.1f}"


for r in range(3):
    hop, late, wait, prep = [], [], [], []
    for i in ok:
        rv = recv_for(r + 1, i)
        h = ms(R[r + 1]["g0"][i] - R[r]["g1"][i])
        hop.append(h)
        if rv is None:
            continue
        s, e, blk = rv
        late.append(ms(s - R[r]["g1"][i]))
        wait.append(ms(blk))
        prep.append(ms(R[r + 1]["g0"][i] - e))
    print(f"hop PP{r}->PP{r + 1}: {summ('hop', hop)} | {summ('late', late)} | {summ('wait', wait)} | {summ('prep', prep)}")
cl = [ms(R[0]["g0"][i + a.depth] - R[3]["g1"][i]) for i in ok if i + a.depth < n and R[0]["g0"][i + a.depth] is not None]
print(f"closure PP3 step i end -> PP0 step i+{a.depth} start: {summ('', cl)}")
beat = [ms(R[0]["g0"][i] - R[0]["g0"][i - 1]) for i in ok if R[0]["g0"][i - 1] is not None]
print(f"PP0 beat (GPU start to start): {summ('', beat)}")
