"""Which PP rank's CPU holds up the prefill pipeline, from torch-profiler traces of all PP ranks taken together
(same node clock; step i of every rank is the same batch).
For rank r and step i (main-stream ops capped at the next op's start, as in prefill_gpu_timeline.py):
  ready_r(i) = max(own GPU free: end_r(i-1), data in: end_{r-1}(i) for r > 0, end of PP0's output recv for r = 0)
  delay_r(i) = start_r(i) - ready_r(i)   (> 0: the rank's own scheduler thread launched late)
and splits delay_r(i) by what the scheduler thread was doing in [ready, start]: blocking syncs, PP recv / send waits,
or CPU work (by sglang phase). Also prints each rank's GPU idle before every step split the same way.
  python3 prefill_pp_critical.py TRACE_PP0 TRACE_PP1 ... [--steps N]"""

import argparse
import bisect
import collections
import gzip
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+")
ap.add_argument("--show", type=int, default=0, help="print per-step rows for the first N steps")
ap.add_argument("--why-ms", type=float, default=100, help="in per-step rows, split delays above this")
a = ap.parse_args()

WAITS = [
    ("sync", lambda e: e.get("cat") == "cuda_runtime" and re.search(r"Synchronize", e["name"]) and e["dur"] > 200),
    ("pp recv", lambda e: e.get("cat") == "python_function" and re.search(r"parallel_state\.py\(\d+\): recv_(tensor_dict|object)$", e["name"])),
    ("pp send", lambda e: e.get("cat") == "python_function" and re.search(r"scheduler_pp_mixin\.py\(\d+\): _pp_commit_comm_work$", e["name"])),
    ("hicache pp sync", lambda e: e.get("cat") == "python_function" and re.search(r"unified_radix_cache\.py\(\d+\): (_pp_sync|_drain_async_work)$", e["name"])),
]
WORK = [
    ("evict/load-back", r"unified_radix_cache\.py\(\d+\): (evict_for_alloc|init_load_back)$"),
    ("get_new_batch_prefill", r"scheduler\.py\(\d+\): get_new_batch_prefill$"),
    ("launch (run_batch)", r"scheduler\.py\(\d+\): run_batch$"),
    ("process_batch_result", r"scheduler\.py\(\d+\): process_batch_result$"),
    ("process_prefill_chunk", r"prefill\.py\(\d+\): process_prefill_chunk$"),
    ("hicache events", r"scheduler\.py\(\d+\): _process_hicache_events$"),
    ("bootstrap / inflight", r"prefill\.py\(\d+\): (process_bootstrapped_queue|process_disagg_prefill_inflight_queue)$"),
]


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(iv, t0, t1):
    """iv: sorted disjoint [s, e]; total overlap with [t0, t1] and the clipped pieces."""
    j = bisect.bisect_left([x[1] for x in iv], t0)
    tot, parts = 0.0, []
    for s, e in iv[j:]:
        if s >= t1:
            break
        x0, x1 = max(s, t0), min(e, t1)
        if x1 > x0:
            tot += x1 - x0
            parts.append((x0, x1))
    return tot, parts


def load(path):
    ev = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))["traceEvents"]
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")), key=lambda e: e["ts"])
    tid = ann[0]["tid"]
    ext = [e for e in ann if not e["name"].startswith("step[DRAFT")]
    ext_ts = [e["ts"] for e in ext]
    ann_ts = [e["ts"] for e in ann]
    ann_step = [bisect.bisect_right(ext_ts, e["ts"]) - 1 for e in ann]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    st = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
    main = st.most_common(1)[0][0]
    ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
    ends = [min(k["ts"] + k["dur"], ops[i + 1]["ts"]) if i + 1 < len(ops) else k["ts"] + k["dur"] for i, k in enumerate(ops)]
    launch = {e["args"]["correlation"]: e["ts"] for e in ev
              if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}) and e.get("tid") == tid}
    g0 = [None] * len(ext)
    g1 = [None] * len(ext)
    for k, en in zip(ops, ends):
        lt = launch.get(k["args"].get("correlation"))
        if lt is None:
            continue
        ai = bisect.bisect_right(ann_ts, lt) - 1
        if ai < 0 or lt > ann[ai]["ts"] + ann[ai]["dur"]:
            continue
        si = ann_step[ai]  # a DRAFT annotation belongs to the EXTEND step before it
        if si < 0:
            continue
        g0[si] = k["ts"] if g0[si] is None else min(g0[si], k["ts"])
        g1[si] = en if g1[si] is None else max(g1[si], en)
    thr = [e for e in ev if e.get("ph") == "X" and e.get("tid") == tid and e.get("cat") in ("python_function", "cuda_runtime")]
    waits = {n: union([(e["ts"], e["ts"] + e["dur"]) for e in thr if f(e)]) for n, f in WAITS}
    work = {n: union([(e["ts"], e["ts"] + e["dur"]) for e in thr if e.get("cat") == "python_function" and re.search(r, e["name"])])
            for n, r in WORK}
    # PP0: end of the output recv (from the last rank) inside each step's pre-launch window
    recv = waits["pp recv"]
    del ev, thr, gpu, ops, ends, launch
    return dict(name=path, n=len(ext), g0=g0, g1=g1, launch=ext_ts, waits=waits, work=work, recv=recv)


R = [load(p) for p in a.traces]
for r in R:
    print(f"rank file {r['name'].split('/')[-1]}: {r['n']} steps")
n = min(r["n"] for r in R)


def classify(r, t0, t1, acc):
    """Split [t0, t1] of rank r's scheduler thread: waits first (sync inside pp recv counts as recv), then work phases."""
    left = [(t0, t1)]
    for name in ("pp recv", "pp send", "hicache pp sync", "sync"):
        nl = []
        for s, e in left:
            tot, parts = overlap(r["waits"][name], s, e)
            acc[name] += tot
            cur = s
            for x0, x1 in parts:
                if x0 > cur:
                    nl.append((cur, x0))
                cur = x1
            if cur < e:
                nl.append((cur, e))
        left = nl
    for s, e in left:
        rest = e - s
        for name, _ in WORK:  # innermost-first order is not guaranteed; the first matching phase wins
            tot, _ = overlap(r["work"][name], s, e)
            if tot > 0:
                acc["cpu: " + name] += tot
                rest -= tot
                break
        acc["cpu: other"] += max(rest, 0)


tot_delay = [collections.Counter() for _ in R]
tot_idle = [collections.Counter() for _ in R]
sum_delay = [0.0] * len(R)
sum_idle = [0.0] * len(R)
busy = [0.0] * len(R)
cnt = 0
for i in range(1, n):
    if any(r["g0"][i] is None or r["g1"][i - 1] is None for r in R):
        continue
    if any(r["g0"][i] - r["g1"][i - 1] > 5e6 for r in R):  # profiler already stopped on some rank
        continue
    cnt += 1
    row = []
    for ri, r in enumerate(R):
        start, own_free = r["g0"][i], r["g1"][i - 1]
        busy[ri] += r["g1"][i] - r["g0"][i]
        # data in = end of the last PP recv before the launch if the thread waited in it, else its start
        tot, parts = overlap(r["recv"], own_free - 2e6, start)
        if parts:
            s, e = next(iv for iv in reversed(r["recv"]) if iv[0] <= parts[-1][0])
            data = e if e - s > 3e3 else s
        else:
            data = own_free
        ready = max(own_free, data)
        idle = max(start - own_free, 0)
        delay = max(start - ready, 0)
        sum_idle[ri] += idle
        sum_delay[ri] += delay
        if idle > 0:
            classify(r, own_free, start, tot_idle[ri])
        if delay > 0:
            classify(r, ready, start, tot_delay[ri])
        cell = f"PP{ri} idle {idle / 1e3:5.0f} delay {delay / 1e3:5.0f}"
        if delay > a.why_ms * 1e3:
            c = collections.Counter()
            classify(r, ready, start, c)
            cell += " (" + ", ".join(f"{k} {v / 1e3:.0f}" for k, v in c.most_common(2)) + ")"
        row.append(cell)
    if i <= a.show:
        print(f"  step {i:2d}: " + " | ".join(row))

print(f"\n{cnt} steps aligned across {len(R)} ranks")
for ri, r in enumerate(R):
    print(f"PP{ri}: GPU busy {busy[ri] / cnt / 1e3:.0f} ms/step, idle before step {sum_idle[ri] / cnt / 1e3:.0f} ms/step, "
          f"own launch delay {sum_delay[ri] / cnt / 1e3:.0f} ms/step")
    for title, acc in (("   idle split", tot_idle[ri]), ("   own-delay split", tot_delay[ri])):
        print(title + ": " + ", ".join(f"{k} {v / cnt / 1e3:.0f}" for k, v in acc.most_common() if v / cnt > 1e3))
