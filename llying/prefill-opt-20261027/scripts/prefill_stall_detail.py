"""What a prefill PP rank's main stream waits for in a stall: for every main-stream GPU op longer than --thr ms
(normally microsecond kernels that absorb a GPU-side event wait), prints
  - the op, its launch site (innermost sglang / aiter frames) and the gap to the previous main-stream op,
  - the last hipStreamWaitEvent / synchronize issued by the launching thread before the launch, with its frames
    (which event the stream was told to wait for), and how long before the launch it was issued,
  - the side-stream GPU work overlapping the op, per stream: total, count, span, and the launch site of the largest,
  - for those side ops, when they were launched on the CPU relative to the stalled op's start (late launch vs slow copy).
  python3 prefill_stall_detail.py TRACE.json.gz [--thr 40] [--max 20]"""

import argparse
import bisect
import collections
import gzip
import json
import re


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)["traceEvents"]


def short(path):
    for key in ("sglang/srt/", "aiter/", "sglang/"):
        i = path.find(key)
        if i >= 0:
            return path[i:]
    return path


def frames_at(py_by_tid, tid, ts, n=4):
    """Innermost n sglang / aiter python frames enclosing ts on thread tid."""
    evs = py_by_tid.get(tid, [])
    i = bisect.bisect_right(py_ts[tid], ts) if evs else 0
    enc = [e for e in evs[max(0, i - 4000):i] if e["ts"] + e["dur"] >= ts]
    enc.sort(key=lambda e: e["ts"])
    out = []
    for e in enc:
        m = re.match(r"(.*?)\((\d+)\): (.*)", e["name"])
        if m and ("sglang/" in m.group(1) or "aiter/" in m.group(1)):
            out.append(f"{short(m.group(1))}:{m.group(2)} {m.group(3)}")
    return " > ".join(out[-n:])


ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--thr", type=float, default=40.0)
ap.add_argument("--max", type=int, default=20)
a = ap.parse_args()

ev = load(a.trace)
gpu = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
             key=lambda e: e["ts"])
rt = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
launch = {e["args"]["correlation"]: e for e in rt if "correlation" in e.get("args", {})}
waits_by_tid = collections.defaultdict(list)
for e in rt:
    if re.search(r"StreamWaitEvent|EventSynchronize|StreamSynchronize|DeviceSynchronize", e["name"]):
        waits_by_tid[e["tid"]].append(e)
for v in waits_by_tid.values():
    v.sort(key=lambda e: e["ts"])
wait_ts = {t: [w["ts"] for w in v] for t, v in waits_by_tid.items()}
py_by_tid = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "python_function":
        py_by_tid[e["tid"]].append(e)
for v in py_by_tid.values():
    v.sort(key=lambda e: e["ts"])
py_ts = {t: [e["ts"] for e in v] for t, v in py_by_tid.items()}

stream_time = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
main = stream_time.most_common(1)[0][0]
mains = [k for k in gpu if k["args"].get("stream") == main]
others = [k for k in gpu if k["args"].get("stream") != main]
o_ts = [k["ts"] for k in others]
t_origin = gpu[0]["ts"]
print(f"== {a.trace}\nmain stream {main}; main ops {len(mains)}; side ops {len(others)}")

long_ops = [(i, k) for i, k in enumerate(mains) if k["dur"] > a.thr * 1e3]
print(f"main ops > {a.thr} ms: {len(long_ops)}, total {sum(k['dur'] for _, k in long_ops) / 1e3:.0f} ms")
for i, k in long_ops[:a.max]:
    t0, t1 = k["ts"], k["ts"] + k["dur"]
    prev = mains[i - 1] if i else None
    gap = (t0 - (prev["ts"] + prev["dur"])) / 1e3 if prev else 0.0
    l = launch.get(k["args"].get("correlation"))
    print(f"\n[{(t0 - t_origin) / 1e6:8.3f} s] {k['dur'] / 1e3:7.1f} ms  {k['name'][:70]}  (gap after previous main op {gap:.2f} ms)")
    if l is not None:
        print(f"    launched {(l['ts'] - t0) / 1e3:+.1f} ms vs GPU start @ {frames_at(py_by_tid, l['tid'], l['ts'])}")
        ws = waits_by_tid.get(l["tid"], [])
        j = bisect.bisect_right(wait_ts[l["tid"]], l["ts"]) - 1 if ws else -1
        for w in ws[max(0, j - 1):j + 1][::-1]:
            print(f"    last {w['name']} {(l['ts'] - w['ts']) / 1e3:.2f} ms before launch @ {frames_at(py_by_tid, w['tid'], w['ts'], 5)}")
    per = collections.defaultdict(lambda: [0.0, 0, None, None, None])
    j = max(0, bisect.bisect_left(o_ts, t0 - 30e6))
    while j < len(others) and others[j]["ts"] < t1:
        o = others[j]
        x = min(t1, o["ts"] + o["dur"]) - max(t0, o["ts"])
        if x > 0:
            key = f"s{o['args'].get('stream')} {o['name'][:48]}"
            p = per[key]
            p[0] += x
            p[1] += 1
            p[2] = o["ts"] if p[2] is None else min(p[2], o["ts"])
            p[3] = o["ts"] + o["dur"] if p[3] is None else max(p[3], o["ts"] + o["dur"])
            if p[4] is None or o["dur"] > p[4]["dur"]:
                p[4] = o
        j += 1
    for key, (tot, n, s, e, big) in sorted(per.items(), key=lambda kv: -kv[1][0])[:5]:
        bl = launch.get(big["args"].get("correlation"))
        site = frames_at(py_by_tid, bl["tid"], bl["ts"], 3) if bl else ""
        lt = f"launched {(bl['ts'] - t0) / 1e3:+.1f} ms" if bl else ""
        print(f"    overlap {tot / 1e3:6.1f} ms x{n:<4d} {key}  span {(s - t0) / 1e3:+.0f}..{(e - t0) / 1e3:+.0f} ms; "
              f"largest {big['dur'] / 1e3:.1f} ms {lt} @ {site}")
