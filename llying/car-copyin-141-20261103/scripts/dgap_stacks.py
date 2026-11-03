"""Where the decode GPU idles: for every GPU-idle gap > 50 us, the blocking HIP runtime calls (memcpy / synchronize) and
the long Python frames running on the CPU during the gap, with their enclosing Python stack (same thread).
python3 dgap_stacks.py TRACE.json.gz"""
import bisect
import collections
import gzip
import json
import re
import sys

d = json.load(gzip.open(sys.argv[1]))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
nsteps = sum(1 for e in ev if e.get("name") == "hipGraphLaunch") // 3
k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
gaps = []
end = k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    if e["ts"] - end > 50:
        gaps.append((end, e["ts"]))
    end = max(end, e["ts"] + e["dur"])
total_gap = sum(b - a for a, b in gaps)

py = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "python_function":
        py[e["tid"]].append(e)
for t in py:
    py[t].sort(key=lambda e: e["ts"])
starts = {t: [e["ts"] for e in v] for t, v in py.items()}
short = lambda n: re.sub(r"^.*/sglang/srt/", "", re.sub(r"^.*site-packages/", "", n))[:95]


def stack(tid, ts, depth=7):
    v, s = py.get(tid, []), starts.get(tid, [])
    i = bisect.bisect_right(s, ts)
    enc = [e for e in v[max(0, i - 4000):i] if e["ts"] + e["dur"] >= ts]
    enc.sort(key=lambda e: e["dur"])
    return [short(e["name"]) for e in enc if "nvtx_utils" not in e["name"] and "contextlib" not in e["name"]][:depth]


def overlap(e, a, b):
    return max(0, min(e["ts"] + e["dur"], b) - max(e["ts"], a))


BLOCK = re.compile(r"hipMemcpyWithStream|hipStreamSynchronize|hipEventSynchronize|hipDeviceSynchronize|hipMemcpy$")
rt = sorted((e for e in ev if e.get("cat") == "cuda_runtime" and BLOCK.search(e["name"])), key=lambda e: e["ts"])
rts = [e["ts"] for e in rt]
by = collections.defaultdict(lambda: [0.0, 0, None])
covered = 0.0
for a, b in gaps:
    i = bisect.bisect_left(rts, a - 200000)
    hit = False
    for e in rt[i:]:
        if e["ts"] > b:
            break
        o = overlap(e, a, b)
        if o <= 0:
            continue
        st = stack(e["tid"], e["ts"])
        key = (e["name"], tuple(st[:3]))
        by[key][0] += o
        by[key][1] += 1
        by[key][2] = st
        hit = True
    if hit:
        covered += b - a
print(f"steps {nsteps}, GPU idle {total_gap / 1e3 / nsteps:.2f} ms/step; gaps with a blocking runtime call: {covered / 1e3 / nsteps:.2f} ms/step")
for (name, _), (o, n, st) in sorted(by.items(), key=lambda x: -x[1][0])[:8]:
    print(f"\n{o / 1e3 / nsteps:.2f} ms/step  n={n}  {name}")
    for f in st:
        print("     ", f)

# gaps without a blocking call: the longest Python frame on any thread during the gap
print("\n--- longest gaps without a blocking runtime call")
nb = []
for a, b in gaps:
    i = bisect.bisect_left(rts, a - 200000)
    if any(overlap(e, a, b) > 0 for e in rt[i:i + 50]):
        continue
    nb.append((b - a, a, b))
for g, a, b in sorted(nb, reverse=True)[:4]:
    print(f"\n gap {g / 1e3:.1f} ms")
    for t, v in py.items():
        s = starts[t]
        i = bisect.bisect_right(s, b)
        inner = [e for e in v[max(0, i - 4000):i] if overlap(e, a, b) > 0.5 * g and e["dur"] < 20 * g]
        inner.sort(key=lambda e: e["dur"])
        if inner:
            print(f"   tid {t}: " + " <- ".join(short(e["name"]) for e in inner[:5]))
