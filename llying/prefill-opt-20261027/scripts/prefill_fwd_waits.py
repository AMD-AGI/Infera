"""Which blocking runtime call makes a prefill PP rank's scheduler wait for its own in-flight forward, per step.
For step i: the forward's GPU end g1(i); the blocking calls on the scheduler thread (hipMemcpy*, *Synchronize, > 2 ms)
that end within --slack-ms after g1(i) waited for that forward. Prints, per call site (python stack), how many steps it
was the one, the mean wait, and the gap from g1(i) to the next forward's GPU start split into
"CPU after the wait" (block end -> run_batch of i+1 -> its first kernel) and the rest.
  python3 prefill_fwd_waits.py TRACE [--slack-ms 3] [--depth 6]"""

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--slack-ms", type=float, default=3.0)
ap.add_argument("--depth", type=int, default=6)
a = ap.parse_args()

ev = json.load(gzip.open(a.trace, "rt"))["traceEvents"]
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
rt_all = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and e.get("tid") == tid]
launch = {e["args"]["correlation"]: e["ts"] for e in rt_all if "correlation" in e.get("args", {})}
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
blocks = sorted((e for e in rt_all if re.search(r"Memcpy|Synchronize", e["name"]) and e["dur"] > 2000), key=lambda e: e["ts"] + e["dur"])
bend = [e["ts"] + e["dur"] for e in blocks]
py = sorted((e for e in ev if e.get("ph") == "X" and e.get("tid") == tid and e.get("cat") == "python_function"), key=lambda e: e["ts"])
py_ts = [e["ts"] for e in py]
runs = [e for e in py if re.search(r"scheduler\.py\(\d+\): run_batch$", e["name"])]
run_ts = [e["ts"] for e in runs]


def stack(t):
    """innermost python frames covering time t (sglang files only), outermost first."""
    j = bisect.bisect_right(py_ts, t)
    fr = [e for e in py[max(0, j - 4000):j] if e["ts"] <= t <= e["ts"] + e["dur"] and "sglang" in e["name"]]
    fr.sort(key=lambda e: e["ts"])
    names = [re.sub(r".*/sglang/srt/", "", e["name"]) for e in fr]
    return " > ".join(names[-a.depth:])


site = collections.defaultdict(list)
after, rest, nowait = [], [], 0
for i in range(len(ext) - 1):
    if g1[i] is None or g0[i + 1] is None or g0[i + 1] - g1[i] > 5e6:
        continue
    j = bisect.bisect_left(bend, g1[i])
    hit = None
    while j < len(blocks) and bend[j] <= g1[i] + a.slack_ms * 1e3:
        if blocks[j]["ts"] < g1[i]:
            hit = blocks[j]
            break
        j += 1
    gap = (g0[i + 1] - g1[i]) / 1e3
    if hit is None:
        nowait += 1
        continue
    s = stack(hit["ts"] + 1) + f" [{hit['name']}]"
    site[s].append((hit["dur"] / 1e3, gap))
    k = bisect.bisect_left(run_ts, hit["ts"] + hit["dur"])
    if k < len(runs):
        after.append((runs[k]["ts"] - (hit["ts"] + hit["dur"])) / 1e3)
        rest.append(gap - after[-1])
tot = sum(len(v) for v in site.values())
print(f"{a.trace.split('/')[-1]}: {tot} steps where a blocking call returned within {a.slack_ms} ms of the forward's end, "
      f"{nowait} steps without such a wait")
for s, v in sorted(site.items(), key=lambda x: -len(x[1])):
    print(f"  {len(v):4d} steps  wait mean {st.mean(x for x, _ in v):6.1f} ms  next-forward gap median {st.median(g for _, g in v):6.1f} ms  {s[:260]}")
if after:
    print(f"  wait end -> next run_batch entry: median {st.median(after):.1f} mean {st.mean(after):.1f} ms; "
          f"run_batch entry -> next forward on GPU: median {st.median(rest):.1f} mean {st.mean(rest):.1f} ms")
