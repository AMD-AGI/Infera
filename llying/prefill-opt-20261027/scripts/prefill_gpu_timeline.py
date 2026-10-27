"""Real GPU busy / idle timeline of a prefill PP rank's main stream from a torch-profiler trace.
On ROCm some ops get an end timestamp far too late (a 64 KB HtoD or a 5 us kernel "lasting" 200-700 ms while the
following ops of the same stream already run), so summing durations overstates the GPU time. Here each main-stream
op's end is capped at the next op's start on that stream, which gives
  - per step (step[...] annotation, by launch): GPU span from its first to its last op, busy time inside that span,
  - the main-stream idle gap before each step's first op, and what the launching thread did during the gap
    (the longest sglang frames of the scheduler thread overlapping the gap),
  - totals: window, busy, idle, and the idle split into "between steps" and "inside steps".
  python3 prefill_gpu_timeline.py TRACE.json.gz [--gap-frames 3]"""

import argparse
import bisect
import collections
import gzip
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--gap-frames", type=int, default=3)
a = ap.parse_args()

ev = json.load(gzip.open(a.trace, "rt") if a.trace.endswith(".gz") else open(a.trace))["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
stream_time = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
main = stream_time.most_common(1)[0][0]
ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
launch = {e["args"]["correlation"]: e for e in ev
          if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")), key=lambda e: e["ts"])
step_ts = [s["ts"] for s in steps]
sched_tid = steps[0]["tid"] if steps else None
frames = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("python_function", "user_annotation")
                 and e.get("tid") == sched_tid and e["dur"] > 5e3), key=lambda e: e["ts"])

ends = [min(k["ts"] + k["dur"], ops[i + 1]["ts"]) if i + 1 < len(ops) else k["ts"] + k["dur"] for i, k in enumerate(ops)]
inflated = sum(k["dur"] for k in ops) - sum(e - k["ts"] for k, e in zip(ops, ends))
per_step = collections.defaultdict(list)
for i, k in enumerate(ops):
    l = launch.get(k["args"].get("correlation"))
    if l is None:
        continue
    si = bisect.bisect_right(step_ts, l["ts"]) - 1
    if si >= 0 and l["ts"] <= steps[si]["ts"] + steps[si]["dur"]:
        per_step[si].append(i)

w0, w1 = ops[0]["ts"], ends[-1]
busy = sum(e - k["ts"] for k, e in zip(ops, ends))
print(f"== {a.trace}\nmain stream {main}: window {(w1 - w0) / 1e3:.0f} ms, busy {busy / 1e3:.0f} ms "
      f"({100 * busy / (w1 - w0):.0f}%), idle {(w1 - w0 - busy) / 1e3:.0f} ms; summed durations exceed capped ones by {inflated / 1e3:.0f} ms")


def what_cpu_did(t0, t1, n):
    """Longest scheduler-thread frames inside [t0, t1] (clipped), innermost names."""
    j = bisect.bisect_left([f["ts"] for f in frames], t0 - 5e6)
    c = collections.Counter()
    for f in frames[j:]:
        if f["ts"] > t1:
            break
        x = min(t1, f["ts"] + f["dur"]) - max(t0, f["ts"])
        if x > 0.3 * (t1 - t0) and f["dur"] < 30e6:
            name = re.sub(r".*/(sglang/srt/|sglang/)", "", f["name"])
            c[name] = max(c[name], x)
    deep = sorted(c.items(), key=lambda kv: kv[1])[:n]
    return "; ".join(f"{k[:70]} {v / 1e3:.0f}" for k, v in deep)


prev_end = None
tot_gap = tot_in = 0.0
rows = []
for si in sorted(per_step):
    idx = per_step[si]
    s0, s1 = ops[idx[0]]["ts"], max(ends[i] for i in idx)
    b = sum(ends[i] - ops[i]["ts"] for i in idx)
    gap = (s0 - prev_end) if prev_end is not None else 0.0
    tot_gap += max(gap, 0)
    tot_in += (s1 - s0) - b
    cpu = what_cpu_did(prev_end, s0, a.gap_frames) if prev_end is not None and gap > 20e3 else ""
    rows.append((si, gap, s1 - s0, b))
    print(f"  step {si:2d} {steps[si]['name'][5:38]:33s} gap before {gap / 1e3:7.1f} ms | span {(s1 - s0) / 1e3:6.1f} busy {b / 1e3:6.1f} ms"
          + (f" | CPU in gap: {cpu}" if cpu else ""))
    prev_end = s1
if rows:
    spans = sorted(r[2] for r in rows)
    gaps = sorted(r[1] for r in rows[1:])
    print(f"steps {len(rows)}: span median {spans[len(spans) // 2] / 1e3:.1f} ms, mean {sum(spans) / len(spans) / 1e3:.1f}; "
          f"gap median {gaps[len(gaps) // 2] / 1e3:.1f} ms, mean {sum(gaps) / max(len(gaps), 1) / 1e3:.1f}; "
          f"idle between steps {tot_gap / 1e3:.0f} ms, idle inside steps {tot_in / 1e3:.0f} ms")
