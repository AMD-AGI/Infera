"""Shrink one prefill PP-rank torch-profiler trace to what the ring / eviction analyses need, with absolute UTC times
(seconds; wall = baseTimeNanoseconds / 1e9 + ts / 1e6):
  steps    step[...] annotations of the scheduler thread, with the main-stream GPU span / busy time of the ops they launched
  sched    scheduler-thread python frames and HIP runtime calls >= --min-ms
  others   python frames >= --min-ms of every other thread (KV transfer, HiCache, ...), for CPU / GIL contention checks
  copies   GPU memcpy ops on any stream (ts, dur, stream, name, bytes)
  python3 trace_compact.py TRACE.json.gz OUT.json.gz [--min-ms 1]"""

import argparse
import bisect
import collections
import gzip
import json

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("out")
ap.add_argument("--min-ms", type=float, default=1.0)
a = ap.parse_args()

raw = json.load(gzip.open(a.trace, "rt"))
base = raw["baseTimeNanoseconds"] / 1e9
ev = raw["traceEvents"]
del raw
W = lambda ts: base + ts / 1e6
min_us = a.min_ms * 1e3

tnames = {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tnames[f"{e['pid']}/{e['tid']}"] = e.get("args", {}).get("name")

ann = sorted((e for e in ev if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("step[")),
             key=lambda e: e["ts"])
tid = ann[0]["tid"]
pid = ann[0]["pid"]
ext = [e for e in ann if not e["name"].startswith("step[DRAFT")]
ext_ts = [e["ts"] for e in ext]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cnt = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
main = cnt.most_common(1)[0][0]
ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
launch = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
          and "correlation" in e.get("args", {}) and e.get("tid") == tid}
n = len(ext)
g0, g1, busy, nops = [None] * n, [None] * n, [0.0] * n, [0] * n
for i, k in enumerate(ops):
    lt = launch.get(k["args"].get("correlation"))
    if lt is None:
        continue
    si = bisect.bisect_right(ext_ts, lt) - 1
    if si < 0 or lt > ext[si]["ts"] + ext[si]["dur"]:
        continue
    en = min(k["ts"] + k["dur"], ops[i + 1]["ts"]) if i + 1 < len(ops) else k["ts"] + k["dur"]
    g0[si] = k["ts"] if g0[si] is None else min(g0[si], k["ts"])
    g1[si] = en if g1[si] is None else max(g1[si], en)
    busy[si] += max(0.0, en - k["ts"])
    nops[si] += 1
steps = [dict(name=e["name"], t=W(e["ts"]), dur=e["dur"] / 1e6, g0=W(g0[i]) if g0[i] is not None else None,
              g1=W(g1[i]) if g1[i] is not None else None, busy=busy[i] / 1e6, nops=nops[i]) for i, e in enumerate(ext)]

sched, others = [], []
for e in ev:
    if e.get("ph") != "X" or e.get("dur", 0) < min_us:
        continue
    c = e.get("cat")
    if c not in ("python_function", "cuda_runtime", "user_annotation"):
        continue
    row = [round(W(e["ts"]), 6), round(e["dur"] / 1e6, 6), e["name"][:160]]
    if e.get("tid") == tid and e.get("pid") == pid:
        sched.append([c[0]] + row)
    elif c == "python_function":
        others.append([f"{e.get('pid')}/{e.get('tid')}"] + row)
copies = [[round(W(e["ts"]), 6), round(e["dur"] / 1e6, 6), e["args"].get("stream"), e["name"][:60], e["args"].get("bytes")]
          for e in gpu if e.get("cat") == "gpu_memcpy"]
nccl = [[round(W(e["ts"]), 6), round(e["dur"] / 1e6, 6), e["args"].get("stream"), e["name"][:60]]
        for e in gpu if "nccl" in e["name"].lower()]
out = dict(trace=a.trace, base=base, sched_tid=f"{pid}/{tid}", main_stream=main, threads=tnames, steps=steps,
           sched=sorted(sched, key=lambda r: r[1]), others=sorted(others, key=lambda r: r[1]), copies=sorted(copies),
           nccl=sorted(nccl))
json.dump(out, gzip.open(a.out, "wt"))
print(f"{a.trace}: {len(steps)} steps, sched {len(sched)}, others {len(others)}, copies {len(copies)}, nccl {len(nccl)}, "
      f"threads {len(tnames)}; first step {steps[0]['t']:.3f} last {steps[-1]['t']:.3f}")
