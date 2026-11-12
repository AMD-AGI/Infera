#!/usr/bin/env python3
"""Per-PP-rank analysis of a prefill torch-profiler trace (e1ab gsptr runs): GPU busy / idle over the profiled window, the
scheduler's time inside eviction (evict_pool_shortfalls), and how much of that eviction time the GPU spent idle -- the part
of eviction that can delay the pipeline. Usage: trace_evict.py TRACE.json.gz..."""
import gzip
import json
import sys


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):
    """Total overlap between two sorted disjoint interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


for path in sys.argv[1:]:
    ev = json.load(gzip.open(path))["traceEvents"]
    kern, comm, evict, wb, batches = [], [], [], [], []
    for e in ev:
        if e.get("ph") != "X":
            continue
        s, d = e.get("ts", 0), e.get("dur", 0)
        cat, name = e.get("cat", ""), e.get("name", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            # RCCL kernels (PP send/recv) spin while waiting for the peer: count them apart from compute.
            (comm if ("nccl" in name.lower() or "rccl" in name.lower()) else kern).append((s, s + d))
        elif cat == "python_function":
            if "evict_pool_shortfalls" in name:
                evict.append((s, s + d))
            elif "writing_check" in name:
                wb.append((s, s + d))
            elif "_pp_launch_batch" in name:
                batches.append((s, s + d))
    if not kern:
        print(path, "no GPU events")
        continue
    # Window: from the first to the last batch launch of the profiled steps (stray long events otherwise stretch it).
    t0, t1 = min(s for s, e in batches), max(e for s, e in batches)
    clip = lambda iv: union([(max(s, t0), min(e, t1)) for s, e in iv if e > t0 and s < t1])
    busy, cbusy = clip(kern), clip(comm)
    wall = t1 - t0
    gbusy = sum(e - s for s, e in busy)
    commonly = sum(e - s for s, e in cbusy) - overlap(cbusy, busy)
    ev_u = union([(max(s, t0), min(e, t1)) for s, e in evict if e > t0 and s < t1])
    ev_t = sum(e - s for s, e in ev_u)
    ev_gpu = overlap(ev_u, busy)
    wb_t = sum(e - s for s, e in union(wb))
    rank = path.split("PP-")[-1].split(".")[0]
    print(
        f"PP{rank}: window {wall / 1e6:.2f} s ({wall / max(len(batches) - 1, 1) / 1e3:.0f} ms/launch), compute busy {gbusy / 1e6:.2f} s ({gbusy / wall:.0%}), "
        f"RCCL-only {commonly / 1e6:.2f} s, neither {(wall - gbusy - commonly) / 1e6:.2f} s; "
        f"launches {len(batches)}; eviction calls {len(evict)}, in eviction {ev_t / 1e6:.2f} s "
        f"(GPU busy during it {ev_gpu / 1e6:.2f} s = {ev_gpu / max(ev_t, 1):.0%}, GPU idle during it {(ev_t - ev_gpu) / 1e6:.2f} s); "
        f"writing_check {wb_t / 1e6:.2f} s"
    )
