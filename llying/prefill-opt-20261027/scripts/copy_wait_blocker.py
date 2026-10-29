"""What a scheduler-thread blocking copy waited behind on the GPU, from one torch-profiler trace.
For every hipMemcpy* call on the scheduler thread longer than --min-ms: its GPU memcpy op (same correlation id), the queueing
delay (GPU start - call start), and the GPU op on ANY stream whose end is closest before that GPU start (within --slack-ms).
Same stream = ordinary stream order; another stream = a cross-stream dependency the code did not ask for (e.g. hardware-
queue sharing when there are more streams than GPU_MAX_HW_QUEUES), or the copy engine being busy.
  python3 copy_wait_blocker.py TRACE [--min-ms 20] [--slack-ms 0.5]"""

import argparse
import bisect
import collections
import gzip
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--min-ms", type=float, default=20)
ap.add_argument("--slack-ms", type=float, default=0.5)
a = ap.parse_args()

ev = json.load(gzip.open(a.trace, "rt"))["traceEvents"]
ann = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")]
tid = ann[0]["tid"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cnt = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
main = cnt.most_common(1)[0][0]
by_corr = {}
for k in gpu:
    c = k["args"].get("correlation")
    if c is not None:
        by_corr.setdefault(c, k)
ends = sorted(((k["ts"] + k["dur"], k) for k in gpu), key=lambda x: x[0])
end_ts = [t for t, _ in ends]


def kind(k):
    n = k["name"].lower()
    if "nccl" in n:
        return "nccl"
    if k.get("cat") == "gpu_memcpy":
        return "memcpy:" + (k["args"].get("kind") or k["name"])[:20]
    return "forward-kernel" if k["args"].get("stream") == main else "kernel"


calls = [e for e in ev if e.get("ph") == "X" and e.get("tid") == tid and e.get("cat") == "cuda_runtime"
         and re.search(r"Memcpy", e["name"]) and e["dur"] > a.min_ms * 1e3]
agg = collections.defaultdict(lambda: [0, 0.0])
tot = 0.0
for c in calls:
    op = by_corr.get(c["args"].get("correlation"))
    if op is None:
        agg[("no gpu op", "", "")][0] += 1
        continue
    wait = op["ts"] - c["ts"]
    j = bisect.bisect_right(end_ts, op["ts"]) - 1
    blk = None
    while j >= 0 and end_ts[j] >= op["ts"] - a.slack_ms * 1e3:
        if ends[j][1] is not op:
            blk = ends[j][1]
            break
        j -= 1
    if blk is None:
        key = ("nothing ended just before", "", "")
    else:
        same = blk["args"].get("stream") == op["args"].get("stream")
        key = ("same stream" if same else "OTHER stream", kind(blk), f"stream {blk['args'].get('stream')}")
    agg[key][0] += 1
    agg[key][1] += wait / 1e3
    tot += wait / 1e3
print(f"{a.trace.split('/')[-1]}: main stream {main}; {len(calls)} blocking memcpy calls > {a.min_ms:.0f} ms, "
      f"GPU-side queueing {tot:.0f} ms in total")
for key, (n, w) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(f"  {n:4d} calls  queueing {w:8.0f} ms  {' | '.join(key)}")
