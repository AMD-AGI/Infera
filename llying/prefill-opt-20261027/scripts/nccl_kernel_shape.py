"""NCCL / RCCL kernels in a torch-profiler trace: name, grid / block, stream, count, total and median duration, and how much
of their time overlaps the main (forward) stream's kernels -- a spinning p2p kernel holds CUs while the forward runs.
  python3 nccl_kernel_shape.py TRACE"""

import bisect
import collections
import gzip
import json
import statistics as st
import sys

ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cnt = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
main = cnt.most_common(1)[0][0]
ops = sorted(((k["ts"], k["ts"] + k["dur"]) for k in gpu if k["args"].get("stream") == main))
merged = []
for s, e in ops:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
ends = [m[1] for m in merged]


def overlap(s, e):
    j = bisect.bisect_left(ends, s)
    t = 0.0
    for a, b in merged[j:]:
        if a >= e:
            break
        t += max(0.0, min(b, e) - max(a, s))
    return t


groups = collections.defaultdict(list)
for k in gpu:
    if "nccl" in k["name"].lower():
        a = k["args"]
        key = (k["name"][:60], str(a.get("grid")), str(a.get("block")), a.get("stream"))
        groups[key].append((k["dur"], overlap(k["ts"], k["ts"] + k["dur"])))
print(f"main stream {main}; forward busy {sum(e - s for s, e in merged) / 1e3:.0f} ms")
for (name, grid, block, stream), v in sorted(groups.items(), key=lambda x: -sum(d for d, _ in x[1])):
    tot = sum(d for d, _ in v) / 1e3
    ov = sum(o for _, o in v) / 1e3
    print(f"{len(v):5d}x  total {tot:8.1f} ms  median {st.median(d for d, _ in v) / 1e3:7.2f} ms  overlapping the forward {ov:8.1f} ms  "
          f"grid {grid} block {block} stream {stream}  {name}")
