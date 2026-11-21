#!/usr/bin/env python3
"""Per-role hot-op table from calibrated-profiler windows of vLLM engines (kf port of Muse calib_table.py).

Usage: vllm_table.py RUN_CALIB_DIR --out OUT_PREFIX
RUN_CALIB_DIR holds one sub-directory per engine (prefill0, decode0, ...) with trace-<pid>.json (json-tool) and
steps-<pid>.json (hook/sitecustomize.py, vLLM port) per DP rank. Role = engine name without trailing digits.

Attribution: dispatch -> correlation id -> launching HIP API record (CPU start, CLOCK_BOOTTIME) -> enclosing step
record [t0, t1] (stages DECODE / MIXED / DUMMY from execute_model, SAMPLE from sample_tokens incl. the DSpark draft
forward, DUMMY_BATCH). Launched inside the window but outside every step -> OTHER; no HIP API record ->
UNATTRIBUTED. "share" = kernel time / all attributed kernel time of the role's window (summed kernel durations;
overlapping streams double-count, so shares are of GPU kernel time, not of wall time).
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

# First match wins. Refine against the real kernel names (results/kernel_names.md) before quoting bucket numbers.
BUCKETS = [
    ("mhc", r"mhc|sinkhorn|hyper_?conn|mega_mhc"),
    ("comm", r"cross_device_reduce|allreduce|all_reduce|nccl|rccl|reduce_scatter|allgather|all_gather|quick_?reduce|custom_ar"),
    ("moe", r"moe|expert|topk_softmax|grouped_topk|_mxfp4_moe|fmoe|ck_moe|asm_moe|moe_sorting|biased_grouped"),
    ("indexer", r"indexer|top_?k512|topk_pack|sparse_logits|fp8_mqa_logits|mqa_logits"),
    ("attention", r"mla|flash|attn|attention|paged|unified_attention|swa|compress|kv_cache|reshape_and_cache"),
    ("gemm", r"gemm|Cijk|matmul|hipblaslt|bf16gemm|a8w8|a4w4|a16w16|f4gemm|wvSplitK|skinny|mm_kernel"),
    ("quant", r"quant|mxfp|fp8|fp4|scale"),
    ("sampling", r"sample|rejection|argmax|softmax|gumbel|exponential|multinomial|draft|dspark"),
    ("norm_rope_elem", r"norm|rope|rotary|rms|elementwise|at::native|copy|Copy|fill|Fill|Cat|silu|act|add|mul|index"),
]
BUCKET_RE = [(b, re.compile(p, re.I)) for b, p in BUCKETS]


def bucket_of(name):
    for b, r in BUCKET_RE:
        if r.search(name):
            return b
    return "misc"


def med(xs):
    return statistics.median(xs) if xs else None


def analyse_rank(trace_path, steps_path):
    sd = json.loads(steps_path.read_text())
    w0, w1 = sd["window_start_ns"], sd["window_end_ns"]
    steps = sorted(sd["steps"], key=lambda s: s["t0"])
    starts = [s["t0"] for s in steps]
    tr = calib_trace.load(trace_path)

    def step_of(t):
        i = bisect_right(starts, t) - 1
        return i if i >= 0 and steps[i]["t0"] <= t <= steps[i]["t1"] else None

    per_step = defaultdict(list)
    other, unattr, n_in, causality = [], [], 0, 0
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
        i = step_of(t)
        if i is None:
            other.append(d)
        else:
            per_step[i].append(d)
    # SAMPLE records belong to the preceding execute_model step: report them as their own stage, but tag with the
    # preceding step's stage so DECODE-step draft cost and MIXED-step draft cost can be separated.
    prev = None
    for s in steps:
        if s["stage"] in ("DECODE", "MIXED"):
            prev = s["stage"]
        elif s["stage"] == "SAMPLE":
            s["after"] = prev
    stages = defaultdict(lambda: {"records": 0, "rec": [], "span": [], "kms": defaultdict(float),
                                  "calls": defaultdict(int), "durs": defaultdict(list), "grids": defaultdict(set),
                                  "graph": 0, "n": 0})
    for i, s in enumerate(steps):
        key = s["stage"] if s["stage"] != "SAMPLE" else f"SAMPLE_after_{s.get('after')}"
        S = stages[key]
        S["records"] += 1
        S["rec"].append(s)
        ks = per_step.get(i, [])
        if ks:
            S["span"].append((max(d["end"] for d in ks) - min(d["start"] for d in ks)) / 1e6)
        for d in ks:
            dur = (d["end"] - d["start"]) / 1e6
            S["kms"][d["name"]] += dur
            S["calls"][d["name"]] += 1
            S["durs"][d["name"]].append(dur)
            S["grids"][d["name"]].add(d["grid"])
            S["n"] += 1
            S["graph"] += 1 if d["graph_exec"] else 0
    out = {"pid": sd["pid"], "dp_rank": sd.get("dp_rank"), "window_ms": (w1 - w0) / 1e6,
           "close_reason": sd.get("close_reason"), "dispatches_in_window": n_in,
           "unattributed_no_api": len(unattr), "unattributed_ms": sum((d["end"] - d["start"]) / 1e6 for d in unattr),
           "missing_symbol": sum(1 for i in per_step for d in per_step[i] if d["name"] == "<missing symbol>"),
           "other_outside_step": len(other), "other_ms": sum((d["end"] - d["start"]) / 1e6 for d in other),
           "causality_violations": causality, "stages": {}}
    for name, S in stages.items():
        n = max(1, S["records"])
        rec = S["rec"]
        tok = {k: [r[k] for r in rec if k in r] for k in ("reqs", "tokens", "prefill_tokens", "prefill_reqs", "spec_tokens", "max_q")}
        out["stages"][name] = {
            "records": S["records"],
            "tokens": {k: {"median": med(v), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in tok.items()},
            "gpu_span_ms_median": med(S["span"]),
            "kernel_ms_total": sum(S["kms"].values()),
            "kernel_ms_per_record": sum(S["kms"].values()) / n,
            "dispatches": S["n"], "graph_replayed": S["graph"],
            "kernels": sorted(([k, S["calls"][k] / n, v / n, v, med(S["durs"][k]) * 1e3, bucket_of(k),
                                sorted(str(g) for g in S["grids"][k])[:3]] for k, v in S["kms"].items()),
                              key=lambda x: -x[3]),
        }
    return out


def role_rollup(ranks):
    """Shares over all ranks and all stages of one role (summed kernel ms)."""
    tot = defaultdict(float)
    by_stage = defaultdict(float)
    for r in ranks:
        for st, S in r["stages"].items():
            for k, _c, _v, total, _m, _b, _g in S["kernels"]:
                tot[k] += total
                by_stage[st] += total
    grand = sum(tot.values()) or 1.0
    buckets = defaultdict(float)
    for k, v in tot.items():
        buckets[bucket_of(k)] += v
    kern = sorted(tot.items(), key=lambda x: -x[1])
    mhc_rank = next((i + 1 for i, (k, _) in enumerate(kern) if bucket_of(k) == "mhc"), None)
    bucket_rank = sorted(buckets.items(), key=lambda x: -x[1])
    return {"kernel_ms_total": grand, "stage_ms": dict(by_stage),
            "buckets": [(b, v, v / grand) for b, v in bucket_rank],
            "mhc_bucket_rank": next((i + 1 for i, (b, _) in enumerate(bucket_rank) if b == "mhc"), None),
            "mhc_first_kernel_rank": mhc_rank,
            "top_kernels": [(k, v, v / grand, bucket_of(k)) for k, v in kern[:40]]}


def md(res, roll):
    L = ["# Calibrated-profiler hot-op tables (vLLM engines)", "",
         "ms = summed kernel durations (streams overlap -> shares are of GPU kernel time, not wall time).", ""]
    for role, R in roll.items():
        L += [f"## role {role}: {R['kernel_ms_total']:.1f} kernel-ms in the windows; mHC bucket rank "
              f"{R['mhc_bucket_rank']}, first mHC kernel rank {R['mhc_first_kernel_rank']}", "",
              "stage kernel-ms: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(R["stage_ms"].items(), key=lambda x: -x[1])), "",
              "| bucket | kernel ms | share |", "|---|---|---|"]
        L += [f"| {b} | {v:.2f} | {s * 100:.1f}% |" for b, v, s in R["buckets"]]
        L += ["", "| # | kernel | ms | share | bucket |", "|---|---|---|---|---|"]
        L += [f"| {i + 1} | `{k[:100]}` | {v:.2f} | {s * 100:.2f}% | {b} |" for i, (k, v, s, b) in enumerate(R["top_kernels"][:30])]
        L.append("")
    for eng, ranks in res.items():
        for r in ranks:
            L.append(f"## {eng} pid {r['pid']} dp_rank {r['dp_rank']}: window {r['window_ms']:.0f} ms ({r['close_reason']}), "
                     f"{r['dispatches_in_window']} dispatches, unattributed {r['unattributed_no_api']} ({r['unattributed_ms']:.1f} ms), "
                     f"outside steps {r['other_outside_step']} ({r['other_ms']:.1f} ms), missing symbols {r['missing_symbol']}, "
                     f"causality violations {r['causality_violations']}")
            for st, S in sorted(r["stages"].items()):
                t = S["tokens"]
                tk = " ".join(f"{k} {v['median']}[{v['min']}-{v['max']}]" for k, v in t.items() if v["median"] is not None)
                L.append(f"### {st}: {S['records']} records; {tk}; GPU span median {S['gpu_span_ms_median']}; "
                         f"kernel ms/record {S['kernel_ms_per_record']:.2f}; graph-replayed {S['graph_replayed']}/{S['dispatches']}")
                L += ["| kernel | calls/rec | ms/rec | median us | bucket | grids |", "|---|---|---|---|---|---|"]
                L += [f"| `{k[:90]}` | {c:.1f} | {v:.3f} | {m:.1f} | {b} | {' '.join(g)} |" for k, c, v, _t, m, b, g in S["kernels"][:20]]
                L.append("")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("calib_dir", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    res = {}
    for eng in sorted(x for x in a.calib_dir.iterdir() if x.is_dir()):
        ranks = []
        for sp in sorted(eng.glob("steps-*.json")):
            pid = sp.stem.split("-", 1)[1]
            tp = eng / f"trace-{pid}.json"
            sd = json.loads(sp.read_text())
            if sd.get("window_start_ns") is None or sd.get("window_end_ns") is None:
                continue
            if not tp.exists():
                print(f"missing {tp} (not finalized?)", file=sys.stderr)
                continue
            ranks.append(analyse_rank(tp, sp))
        if ranks:
            res[eng.name] = ranks
    roles = defaultdict(list)
    for eng, ranks in res.items():
        roles[re.sub(r"\d+$", "", eng)].extend(ranks)
    roll = {role: role_rollup(rs) for role, rs in roles.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps({"engines": res, "roles": roll}, indent=1))
    Path(str(a.out) + ".md").write_text(md(res, roll))
    print(str(a.out) + ".md")


if __name__ == "__main__":
    main()
