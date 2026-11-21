#!/usr/bin/env python3
"""Per-kernel VERIFY / DRAFT / EXTEND table from a calibrated-profiler window of the Muse server.

Usage: calib_table.py TRACE_DIR --out OUT_PREFIX [--bs-min N --bs-max N] [--full-attn-min-us 150]
TRACE_DIR holds trace-<pid>.json (json-tool) and steps-<pid>.json (hook/sitecustomize.py) per process.
Only processes with a steps file are analysed (= scheduler ranks that ran the window).

Phase rule (same as research/a18-trace/analyze.py for torch traces): dispatch -> correlation id -> launching
HIP API record (CPU start, CLOCK_BOOTTIME) -> enclosing ModelRunner.forward [t0, t1] from steps-<pid>.json.
Stage labels follow sglang build_step_span_name: VERIFY (TARGET_VERIFY, target model), DRAFT (any forward of
the draft worker), else forward_mode name (EXTEND, DECODE, IDLE ...). Dispatches launched inside the window
but outside every forward -> OTHER; no HIP API record -> UNATTRIBUTED (should be ~0; gate in RUNPLAN).
Per-step values divide by the number of steps of that stage (after the optional bs filter).
Buckets reproduce kf2/tasks/phase3/ANALYSIS.md section 7.1 (torch-profiler, TP4 C40 bs 26, v3-006 p1).
"""
import argparse
import json
import re
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_trace  # noqa: E402

# Reference: torch-profiler VERIFY ms/step at bs 26 (ANALYSIS.md 7.1, TP0) and DRAFT 2.41 (7.4).
REFERENCE_VERIFY = {"moe_mxfp4": 12.66, "moe_bf16": 2.53, "attn_full": 9.72, "attn_swa": 1.36,
                    "dense_gemm": 8.51, "allreduce": 6.08, "norm_rope_elem": 1.8, "_total": 42.62}
REFERENCE_DRAFT_TOTAL = 2.41

BUCKETS = [
    ("allreduce", r"cross_device_reduce|allreduce|all_reduce|nccl|rccl|reduce_scatter|allgather|all_gather"),
    ("moe_mxfp4", r"_expert_a4|_route_quant_a4|_prepare_a4|_scatter_a4|mfma_moe|_mxfp4_quant_rows"),
    ("moe_bf16", r"moe\w*flydsl|flydsl\w*moe"),
    ("attn_full", r"_seg_reduce|_plan_kernel|_vv2_"),
    ("attn_swa", r"_reduce_norm_gate|_verify_(partials|reduce)"),
    ("attn_verify_fly", r"muse_verify_fly"),        # split full/swa by per-grid median duration below
    ("attn_other", r"unified_attention|_fwd_kernel|attention|attn"),
    ("moe_router", r"_router_|routing|moe_sorting|topk_softmax"),
    ("dense_gemm", r"Cijk_|_joint_fp8|_joint_epilogue|_gemm_a16_w16|hgemm|bf16gemm|gemm|GEMM"),
    ("norm_rope_elem", r"norm|rope|rotary|rms|elementwise|at::native|copy|Copy|fill|Fill|Cat"),
]
BUCKET_RE = [(b, re.compile(p)) for b, p in BUCKETS]


def bucket_of(name):
    for b, r in BUCKET_RE:
        if r.search(name):
            return b
    return "misc"


def analyse_rank(trace_path, steps_path, a):
    steps_doc = json.loads(steps_path.read_text())
    w0, w1 = steps_doc["window_start_ns"], steps_doc["window_end_ns"]
    steps = sorted(steps_doc["steps"], key=lambda s: s["t0"])
    starts = [s["t0"] for s in steps]
    tr = calib_trace.load(trace_path)

    def stage_of(t):
        i = bisect_right(starts, t) - 1
        if i >= 0 and steps[i]["t0"] <= t <= steps[i]["t1"]:
            return i
        return None

    keep_step = [True] * len(steps)
    for i, s in enumerate(steps):
        if s["stage"] == "VERIFY" and ((a.bs_min and s["bs"] < a.bs_min) or (a.bs_max and s["bs"] > a.bs_max)):
            keep_step[i] = False
    per_step_kernels = defaultdict(list)
    other, unattr, n_in = [], [], 0
    causality = 0
    for d in tr["dispatch"]:
        api = tr["api"].get(d["corr"])
        if api is None:
            if w0 <= d["start"] <= w1:
                unattr.append(d)
            continue
        t = api["start"]
        if not (w0 <= t <= w1):
            continue
        n_in += 1
        if d["start"] < t:
            causality += 1
        i = stage_of(t)
        if i is None:
            other.append(d)
        else:
            per_step_kernels[i].append(d)

    # split muse_verify_fly by grid: groups whose median duration >= threshold are full attention
    fly_groups = defaultdict(list)
    for i, ks in per_step_kernels.items():
        for d in ks:
            if "muse_verify_fly" in d["name"]:
                fly_groups[d["grid"]].append(d["end"] - d["start"])
    fly_full = {g for g, v in fly_groups.items() if statistics.median(v) >= a.full_attn_min_us * 1e3}

    def bucket(d):
        b = bucket_of(d["name"])
        if b == "attn_verify_fly":
            b = "attn_full" if d["grid"] in fly_full else "attn_swa"
        return b

    stages = defaultdict(lambda: {"steps": 0, "bs": [], "gpu_span_ms": [], "kernel_ms": defaultdict(float),
                                  "kernel_calls": defaultdict(int), "kernel_durs": defaultdict(list),
                                  "bucket_ms": defaultdict(float)})
    for i, s in enumerate(steps):
        if not keep_step[i]:
            continue
        S = stages[s["stage"]]
        S["steps"] += 1
        S["bs"].append(s["bs"])
        ks = per_step_kernels.get(i, [])
        if ks:
            S["gpu_span_ms"].append((max(d["end"] for d in ks) - min(d["start"] for d in ks)) / 1e6)
        for d in ks:
            dur = (d["end"] - d["start"]) / 1e6
            S["kernel_ms"][d["name"]] += dur
            S["kernel_calls"][d["name"]] += 1
            S["kernel_durs"][d["name"]].append(dur)
            S["bucket_ms"][bucket(d)] += dur
    out = {"pid": steps_doc["pid"], "tp_rank": steps_doc.get("tp_rank"), "window_ms": (w1 - w0) / 1e6,
           "close_reason": steps_doc.get("close_reason"), "dispatches_in_window": n_in,
           "unattributed_no_api": len(unattr),
           "unattributed_ms": sum((d["end"] - d["start"]) / 1e6 for d in unattr),
           "missing_symbol_in_window": sum(1 for i in per_step_kernels for d in per_step_kernels[i]
                                           if d["name"] == "<missing symbol>"),
           "other_outside_forward": len(other),
           "other_ms": sum((d["end"] - d["start"]) / 1e6 for d in other),
           "causality_violations": causality,
           "fly_grid_groups": {str(g): {"n": len(v), "median_us": statistics.median(v) / 1e3,
                                        "class": "full" if g in fly_full else "swa"} for g, v in fly_groups.items()},
           "stages": {}}
    for name, S in stages.items():
        n = max(1, S["steps"])
        out["stages"][name] = {
            "steps": S["steps"], "bs_median": statistics.median(S["bs"]) if S["bs"] else None,
            "bs_min": min(S["bs"]) if S["bs"] else None, "bs_max": max(S["bs"]) if S["bs"] else None,
            "gpu_span_ms_per_step": statistics.median(S["gpu_span_ms"]) if S["gpu_span_ms"] else None,
            "kernel_ms_per_step": sum(S["kernel_ms"].values()) / n,
            "buckets_ms_per_step": {b: v / n for b, v in sorted(S["bucket_ms"].items(), key=lambda x: -x[1])},
            "kernels": sorted(([k, S["kernel_calls"][k] / n, v / n, statistics.median(S["kernel_durs"][k]) * 1e3,
                                bucket_of(k)] for k, v in S["kernel_ms"].items()), key=lambda x: -x[2])[:60]}
    return out


def md(res, a):
    L = [f"# Calibrated-profiler per-step table ({a.trace_dir})", "",
         "Columns: ms/step = summed kernel time per step of that stage (overlap double-counts; not wall).", ""]
    for r in res:
        L.append(f"## pid {r['pid']} tp_rank {r['tp_rank']}: window {r['window_ms']:.0f} ms, "
                 f"{r['dispatches_in_window']} dispatches, unattributed {r['unattributed_no_api']} "
                 f"({r['unattributed_ms']:.1f} ms), "
                 f"other {r['other_outside_forward']} ({r['other_ms']:.1f} ms), causality violations "
                 f"{r['causality_violations']}")
        for st, S in sorted(r["stages"].items()):
            L.append(f"### {st}: {S['steps']} steps, bs median {S['bs_median']} ({S['bs_min']}-{S['bs_max']}), "
                     f"GPU span/step {S['gpu_span_ms_per_step']}, kernel ms/step {S['kernel_ms_per_step']:.2f}")
            L.append("| bucket | ms/step | torch ref (VERIFY bs 26) |")
            L.append("|---|---|---|")
            for b, v in S["buckets_ms_per_step"].items():
                ref = REFERENCE_VERIFY.get(b, "") if st == "VERIFY" else ""
                L.append(f"| {b} | {v:.2f} | {ref} |")
            if st == "VERIFY":
                L.append(f"| _total | {S['kernel_ms_per_step']:.2f} | {REFERENCE_VERIFY['_total']} |")
            if st == "DRAFT":
                L.append(f"| _total | {S['kernel_ms_per_step']:.2f} | {REFERENCE_DRAFT_TOTAL} |")
            L.append("")
            L.append("| kernel | calls/step | ms/step | median us | bucket |")
            L.append("|---|---|---|---|---|")
            for k, c, v, m, b in S["kernels"][:25]:
                L.append(f"| `{k[:90]}` | {c:.1f} | {v:.3f} | {m:.1f} | {b} |")
            L.append("")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bs-min", type=int, default=0)
    p.add_argument("--bs-max", type=int, default=0)
    p.add_argument("--full-attn-min-us", type=float, default=150.0)
    a = p.parse_args()
    res = []
    for sp in sorted(a.trace_dir.glob("steps-*.json")):
        pid = sp.stem.split("-", 1)[1]
        tp = a.trace_dir / f"trace-{pid}.json"
        sd = json.loads(sp.read_text())
        if sd.get("window_start_ns") is None or sd.get("window_end_ns") is None:
            print(f"skip {sp.name}: no closed window (non-forward process)", file=sys.stderr)
            continue
        if not tp.exists():
            print(f"missing {tp} (not finalized?)", file=sys.stderr)
            continue
        res.append(analyse_rank(tp, sp, a))
    res.sort(key=lambda r: (r["tp_rank"] is None, r["tp_rank"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(res, indent=1))
    Path(str(a.out) + ".md").write_text(md(res, a))
    print(str(a.out) + ".md")


if __name__ == "__main__":
    main()
