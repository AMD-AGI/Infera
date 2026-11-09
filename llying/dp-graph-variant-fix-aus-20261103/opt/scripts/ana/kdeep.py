#!/usr/bin/env python3
"""Independent per-layer / per-kernel breakdown of SGLang decode torch-profiler traces (one file per DP rank).

Unlike dtrace_deep.py, graph type is taken from the CPU annotation that encloses each hipGraphLaunch
(`step[VERIFY bs=N]`, `draft`, `draft_extend`) instead of launch order, and the verify graph is cut into layers at
the per-layer all-gather, so every kernel gets a position in its layer. Layers with the same kernel-name sequence
form one signature; each signature is printed with mean duration per position (over steps, layers and ranks).

Usage: kdeep.py [--skip-ms 60] [--json OUT] TRACE_RANK0.json.gz ...
"""
import argparse
import collections
import gzip
import json
import multiprocessing as mp
import re
import statistics as st

LAYER_ANCHOR = re.compile(r"allgather")


def short(name, n=110):
    name = re.sub(r"^void ", "", name)
    return name if len(name) <= n else name[:n] + "…"


def load(args):
    path, skip_us = args
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    ann = sorted(
        (e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"
         and (e["name"].startswith("step[") or e["name"] in ("draft", "draft_extend"))),
        key=lambda e: e["ts"],
    )
    gl = sorted((e for e in ev if e.get("ph") == "X" and e.get("name") == "hipGraphLaunch"), key=lambda e: e["ts"])
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    t0 = ks[0]["ts"] + skip_us

    def enclosing(e):
        best = None
        for a in ann:
            if a["ts"] <= e["ts"] and e["ts"] + e["dur"] <= a["ts"] + a["dur"] + 1:
                if best is None or a["dur"] < best["dur"]:
                    best = a
            elif a["ts"] > e["ts"]:
                break
        return best

    launches = []
    for g in gl:
        a = enclosing(g)
        if a is None:
            kind, bs = "unknown", None
        elif a["name"].startswith("step["):
            m = re.match(r"step\[(\w+) bs=(\d+)\]", a["name"])
            kind, bs = (m.group(1).lower(), int(m.group(2))) if m else (a["name"], None)
        else:
            kind, bs = a["name"], None
        launches.append({"corr": g["args"]["correlation"], "kind": kind, "bs": bs, "kernels": []})
    by_corr = {l["corr"]: l for l in launches}
    eager = []
    for k in ks:
        l = by_corr.get(k["args"].get("correlation"))
        (l["kernels"] if l is not None else eager).append(k)
    # Busy/idle over the window after skip: union of kernel intervals.
    win = [k for k in ks if k["ts"] >= t0]
    busy, cur_s, cur_e = 0.0, None, None
    for k in win:
        s, e = k["ts"], k["ts"] + k["dur"]
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    wall = (win[-1]["ts"] + win[-1]["dur"] - win[0]["ts"]) if win else 0.0
    out = {"path": path, "launches": [], "eager_us": sum(k["dur"] for k in eager if k["ts"] >= t0),
           "busy_us": busy, "wall_us": wall}
    for l in launches:
        if not l["kernels"] or l["kernels"][0]["ts"] < t0:
            continue
        out["launches"].append({
            "kind": l["kind"], "bs": l["bs"],
            "start": l["kernels"][0]["ts"],
            "k": [(k["name"], k["dur"]) for k in l["kernels"]],
        })
    return out


def split_layers(kernels):
    """Cut a verify graph's kernels into [prologue, layer0, layer1, ...] at each all-gather (one per layer)."""
    idx = [i for i, (n, _) in enumerate(kernels) if LAYER_ANCHOR.search(n)]
    if not idx:
        return [kernels], []
    # A layer = from the kernel after the previous layer's reduce-scatter ... simpler: anchor-to-anchor windows,
    # shifted so each window starts right after the previous anchor's reduce-scatter partner.
    rs = [i for i, (n, _) in enumerate(kernels) if "reduce_scatter" in n]
    cuts = [0] + [r + 1 for r in rs]
    segs = [kernels[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    tail = kernels[cuts[-1]:]
    return segs, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--skip-ms", type=float, default=60.0)
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    with mp.Pool(min(len(a.traces), 16)) as p:
        ranks = p.map(load, [(t, a.skip_ms * 1e3) for t in a.traces])
    R = len(ranks)

    print(f"# {R} ranks")
    for r, rk in enumerate(ranks):
        c = collections.Counter(l["kind"] for l in rk["launches"])
        nk = collections.defaultdict(set)
        for l in rk["launches"]:
            nk[l["kind"]].add(len(l["k"]))
        print(f"rank{r}: launches {dict(c)}  kernels/launch {({k: sorted(v) for k, v in nk.items()})}  "
              f"eager {rk['eager_us'] / 1e3:.1f} ms  busy {rk['busy_us'] / 1e3:.1f} / wall {rk['wall_us'] / 1e3:.1f} ms")

    # Per-step batch sizes across ranks (verify launches in order).
    vs = [[l for l in rk["launches"] if l["kind"] == "verify"] for rk in ranks]
    n = min(len(v) for v in vs)
    bs = [[v[i]["bs"] for v in vs] for i in range(n)]
    sums = [sum(b) for b in bs]
    maxs = [max(b) for b in bs]
    print(f"\n# verify steps per rank: {n}")
    print(f"per-step bs across ranks (first 6): {bs[:6]}")
    print(f"sum bs  median {st.median(sums)}  range {min(sums)}..{max(sums)}")
    print(f"max bs  median {st.median(maxs)}  range {min(maxs)}..{max(maxs)}; max/mean median "
          f"{st.median([m / (s / R) for m, s in zip(maxs, sums)]):.2f}")
    # Step period from verify graph starts (rank 0).
    starts = [l["start"] for l in vs[0][:n]]
    per = [(b - a_) / 1e3 for a_, b in zip(starts, starts[1:])]
    print(f"step period rank0 (verify start to next): median {st.median(per):.2f} ms  "
          f"p10 {sorted(per)[len(per) // 10]:.2f}  p90 {sorted(per)[len(per) * 9 // 10]:.2f}")
    vtime = [sum(d for _, d in v[i]["k"]) / 1e3 for v in vs for i in range(n)]
    print(f"verify graph GPU time per step: mean {st.mean(vtime):.2f} ms")
    for kind in ("draft", "draft_extend"):
        t = [sum(d for _, d in l["k"]) / 1e3 for rk in ranks for l in rk["launches"] if l["kind"] == kind]
        cnt = [sum(1 for l in rk["launches"] if l["kind"] == kind) for rk in ranks]
        if t:
            print(f"{kind} graph GPU time: mean {st.mean(t):.2f} ms per launch, {st.mean(cnt) / n:.2f} launches/step")

    # Per-kernel totals in verify (ms/step, mean over ranks).
    tot = collections.Counter()
    calls = collections.Counter()
    for v in vs:
        for i in range(n):
            for nm, d in v[i]["k"]:
                tot[nm] += d
                calls[nm] += 1
    print(f"\n# verify graph: top kernels (ms/step mean over ranks, calls/step, us/call)")
    grand = sum(tot.values()) / R / n / 1e3
    for nm, us in tot.most_common(a.top):
        c = calls[nm] / R / n
        print(f"  {us / R / n / 1e3:6.3f}  {c:6.1f}  {us / calls[nm]:7.2f}  {short(nm)}")
    print(f"  total {grand:.2f} ms/step, {len(tot)} distinct kernels")

    # Layer signatures.
    sig_stats = collections.defaultdict(lambda: {"count": 0, "dur": None, "names": None, "layer_ids": set()})
    prologue = collections.Counter()
    for v in vs:
        for i in range(n):
            segs, tail = split_layers(v[i]["k"])
            for li, seg in enumerate(segs):
                names = tuple(re.sub(r"\s+", " ", nm) for nm, _ in seg)
                s = sig_stats[names]
                if s["dur"] is None:
                    s["dur"] = [0.0] * len(seg)
                    s["names"] = names
                s["count"] += 1
                s["layer_ids"].add(li)
                for j, (_, d) in enumerate(seg):
                    s["dur"][j] += d
            for nm, d in tail:
                prologue[nm] += d
    print(f"\n# layer signatures (cut after each reduce-scatter); tail after the last layer per step:")
    for nm, d in prologue.most_common(8):
        print(f"  tail {d / R / n / 1e3:6.3f} ms/step  {short(nm, 90)}")
    for names, s in sorted(sig_stats.items(), key=lambda x: -x[1]["count"]):
        layers = sorted(s["layer_ids"])
        per_step = s["count"] / R / n
        total = sum(s["dur"]) / s["count"]
        print(f"\n## signature: {len(names)} kernels, {per_step:.1f} layers/step (layer idx {layers[0]}..{layers[-1]}, "
              f"{len(layers)} distinct), {total:.1f} us/layer, {total * per_step / 1e3:.2f} ms/step")
        for j, nm in enumerate(names):
            print(f"   {j:3d} {s['dur'][j] / s['count']:8.2f} us  {short(nm, 120)}")
    if a.json:
        json.dump({"bs": bs, "kernel_ms_per_step": {k: v / R / n / 1e3 for k, v in tot.items()},
                   "calls_per_step": {k: v / R / n for k, v in calls.items()}}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
