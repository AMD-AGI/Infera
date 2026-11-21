#!/usr/bin/env python3
"""GPU idle and fusion-candidate analysis of one rank's calibrated-profiler window.

Usage: idle_fusion.py TRACE_DIR PID [--seq-step N] [--seq-len 160]
Prints (JSON + text):
  * window: GPU busy = union of kernel intervals; idle split into inside-forward gaps (between kernels launched by the
    same forward) and between-forward gaps (last kernel of a forward -> first kernel of the next);
  * per stage (VERIFY / DRAFT / ...): per-step median of GPU span, busy (union), kernel sum, inside gaps, CPU forward
    wall (t0..t1) and the gap to the next forward;
  * kernel-size histogram for VERIFY (calls/step and ms/step by duration bucket);
  * the kernel sequence of one VERIFY step (name, us, gap before) to read the per-layer pattern.
"""
import argparse
import collections
import json
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_trace  # noqa: E402


def union_ms(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace_dir", type=Path)
    p.add_argument("pid")
    p.add_argument("--seq-step", type=int, default=5, help="index of the VERIFY step whose sequence is printed")
    p.add_argument("--seq-len", type=int, default=160)
    a = p.parse_args()
    sd = json.loads((a.trace_dir / f"steps-{a.pid}.json").read_text())
    steps = sorted(sd["steps"], key=lambda s: s["t0"])
    st = [s["t0"] for s in steps]
    tr = calib_trace.load(a.trace_dir / f"trace-{a.pid}.json")
    per = collections.defaultdict(list)
    for d in tr["dispatch"]:
        api = tr["api"].get(d["corr"])
        if not api:
            continue
        i = bisect_right(st, api["start"]) - 1
        if i >= 0 and steps[i]["t0"] <= api["start"] <= steps[i]["t1"]:
            per[i].append(d)
    idx = sorted(per)
    allk = [d for i in idx for d in per[i]]
    g0, g1 = min(d["start"] for d in allk), max(d["end"] for d in allk)
    busy = union_ms([(d["start"], d["end"]) for d in allk])
    rows = []
    for n, i in enumerate(idx):
        ks = sorted(per[i], key=lambda d: d["start"])
        s0, s1 = ks[0]["start"], max(d["end"] for d in ks)
        b = union_ms([(d["start"], d["end"]) for d in ks])
        nxt = None
        if n + 1 < len(idx):
            nxt = (min(d["start"] for d in per[idx[n + 1]]) - s1) / 1e6
        rows.append({"i": i, "stage": steps[i]["stage"], "bs": steps[i].get("bs"), "span": (s1 - s0) / 1e6, "busy": b,
                     "ksum": sum(d["end"] - d["start"] for d in ks) / 1e6, "inside_gap": (s1 - s0) / 1e6 - b,
                     "cpu_fwd": (steps[i]["t1"] - steps[i]["t0"]) / 1e6, "gap_to_next": nxt, "n": len(ks)})
    between = sum(r["gap_to_next"] for r in rows if r["gap_to_next"] is not None and r["gap_to_next"] > 0)
    inside = sum(r["inside_gap"] for r in rows)
    out = {"pid": a.pid, "gpu_window_ms": (g1 - g0) / 1e6, "busy_ms": busy,
           "busy_frac": busy / ((g1 - g0) / 1e6), "inside_forward_gap_ms": inside,
           "between_forward_gap_ms": between, "stages": {}}
    for stg in sorted({r["stage"] for r in rows}):
        rs = [r for r in rows if r["stage"] == stg]
        med = lambda k: statistics.median([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else None  # noqa: E731
        out["stages"][stg] = {"steps": len(rs), "kernels_per_step": med("n"), "span_ms": med("span"),
                              "busy_ms": med("busy"), "kernel_sum_ms": med("ksum"), "inside_gap_ms": med("inside_gap"),
                              "cpu_forward_ms": med("cpu_fwd"), "gap_to_next_ms": med("gap_to_next")}
    # duration histogram for VERIFY
    edges = [5, 10, 20, 50, 100, 200, 1e9]
    hist = collections.Counter()
    hist_ms = collections.Counter()
    vsteps = [r["i"] for r in rows if r["stage"] == "VERIFY"]
    for i in vsteps:
        for d in per[i]:
            us = (d["end"] - d["start"]) / 1e3
            b = next(e for e in edges if us < e)
            hist[b] += 1
            hist_ms[b] += us / 1e3
    nv = max(1, len(vsteps))
    out["verify_duration_hist"] = {f"<{int(e) if e < 1e9 else 'inf'}us": {"calls_per_step": hist[e] / nv,
                                                                         "ms_per_step": hist_ms[e] / nv} for e in edges}
    print(json.dumps(out, indent=1))
    if vsteps:
        i = vsteps[min(a.seq_step, len(vsteps) - 1)]
        ks = sorted(per[i], key=lambda d: d["start"])
        print(f"\n# VERIFY step {i} (bs {steps[i].get('bs')}): first {a.seq_len} of {len(ks)} kernels  [gap_us dur_us name]")
        prev = None
        for d in ks[:a.seq_len]:
            gap = (d["start"] - prev) / 1e3 if prev is not None else 0.0
            print(f"{gap:8.1f} {(d['end'] - d['start']) / 1e3:8.1f}  {d['name'][:110]}")
            prev = d["end"]


if __name__ == "__main__":
    main()
