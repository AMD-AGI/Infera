"""What a prefill PP rank's scheduler thread does inside run_batch before the forward's first GPU kernel is launched,
and before / after the run_batch call in the same event-loop iteration (output recv, proxy-send commit). One trace.
For every EXTEND step: run_batch entry -> first kernel launch (cuda_runtime launch on the scheduler thread) and the
python functions (direct children and grandchildren of run_batch) that cover that interval, summed over steps.
  python3 prefill_launch_prep.py TRACE [--top 25]"""

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()

ev = json.load(gzip.open(a.trace, "rt"))["traceEvents"]
ann = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")
              and not e["name"].startswith("step[DRAFT")), key=lambda e: e["ts"])
tid = ann[0]["tid"]
py = sorted((e for e in ev if e.get("ph") == "X" and e.get("tid") == tid and e.get("cat") == "python_function"),
            key=lambda e: e["ts"])
rt = sorted((e for e in ev if e.get("ph") == "X" and e.get("tid") == tid and e.get("cat") == "cuda_runtime"
             and re.search(r"LaunchKernel|ModuleLaunch|hipExtModuleLaunchKernel|hipLaunch|GraphLaunch", e["name"])),
            key=lambda e: e["ts"])
rt_ts = [e["ts"] for e in rt]
runs = [e for e in py if re.search(r"scheduler\.py\(\d+\): run_batch$", e["name"])]
print(f"{len(runs)} run_batch calls, {len(rt)} kernel launches on the scheduler thread")
pre, total = [], []
acc = collections.Counter()
cnt = collections.Counter()
for r in runs:
    t0, t1 = r["ts"], r["ts"] + r["dur"]
    j = bisect.bisect_left(rt_ts, t0)
    if j >= len(rt) or rt[j]["ts"] > t1:
        continue
    tk = rt[j]["ts"]
    pre.append((tk - t0) / 1e3)
    total.append(r["dur"] / 1e3)
    lo = bisect.bisect_left([e["ts"] for e in py], t0)
    for e in py[lo:]:
        if e["ts"] >= tk:
            break
        if e["ts"] + e["dur"] > t1 or e is r:
            continue
        x0, x1 = e["ts"], min(e["ts"] + e["dur"], tk)
        if x1 - x0 > 500:
            acc[e["name"]] += (x1 - x0) / 1e3
            cnt[e["name"]] += 1
n = len(pre)
print(f"run_batch -> first kernel launch: median {st.median(pre):.1f} mean {st.mean(pre):.1f} ms; run_batch total median "
      f"{st.median(total):.1f} mean {st.mean(total):.1f} ms ({n} steps)")
print("python frames inside that interval (> 0.5 ms each; nested frames overlap, ms per step):")
for name, v in acc.most_common(a.top):
    print(f"  {v / n:7.2f} ms/step  x{cnt[name] / n:5.1f}/step  {name[:150]}")
