#!/usr/bin/env python3
"""Attribute every GPU kernel in a graph-OFF torch-profiler trace to its phase and launching op.

For each kernel: phase = innermost annotation among step[...]/draft_extend/draft (host side, at
launch time), op = innermost host cpu_op enclosing the launch, with its recorded Input Dims and
dtypes. Aggregated per (phase, op, input dims, kernel); time is the kernel's device duration.

Usage: kernel_shapes.yihou.py TRACE.json.gz [--iterations N] [--top K] [--out FILE.md]
Only a graph-OFF trace carries host ops with shapes; graph-ON replays hide them.
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

SKIP_OPS = {"aten::empty", "aten::empty_strided", "aten::to", "aten::_to_copy", "aten::view",
            "aten::as_strided", "aten::slice", "aten::select", "aten::reshape", "aten::t",
            "aten::transpose", "aten::expand", "aten::unsqueeze", "aten::squeeze", "aten::lift_fresh",
            "aten::detach", "aten::alias", "aten::_reshape_alias", "aten::resolve_conj", "aten::resolve_neg"}


def short(name, n=90):
    return name if len(name) <= n else name[: n - 3] + "..."


def dims(op):
    a = op.get("args", {})
    shapes = a.get("Input Dims") or []
    types = a.get("Input type") or []
    parts = []
    for s, t in zip(shapes, types + [""] * len(shapes)):
        if s:
            parts.append(f"{t}{s}" if t and t not in ("ScalarList", "Scalar") else str(s))
    return "; ".join(parts)


def attribute_context(events):
    """correlation id -> (phase, innermost host op, outermost aiter/sgl_kernel op) at launch."""
    host = defaultdict(list)
    launches = {}
    for e in events:
        cat = e.get("cat")
        if cat in ("cpu_op", "user_annotation"):
            host[e["tid"]].append(e)
        elif cat == "cuda_runtime" and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e
    # Per launch, the stack of host intervals open at its timestamp (sweep per thread).
    by_tid = defaultdict(list)
    for corr, launch in launches.items():
        by_tid[launch["tid"]].append(launch)
    context = {}
    for tid, ls in by_tid.items():
        ops = sorted(host[tid], key=lambda e: (e["ts"], -e.get("dur", 0)))
        ls.sort(key=lambda e: e["ts"])
        stack, i = [], 0
        for launch in ls:
            t = launch["ts"]
            while i < len(ops) and ops[i]["ts"] <= t:
                stack.append(ops[i])
                i += 1
            stack = [o for o in stack if o["ts"] + o.get("dur", 0) >= t]
            # sglang labels the draft-extend forward's own span step[DRAFT ...]; the enclosing
            # draft_extend / draft annotations disambiguate it from the draft decode steps.
            names = [o["name"] for o in stack if o["cat"] == "user_annotation"]
            if "draft_extend" in names:
                phase = "DRAFT_EXTEND_V2 (draft_extend)"
            elif any(n.startswith("step[VERIFY") for n in names):
                phase = "TARGET_VERIFY (step[VERIFY])"
            elif any(n.startswith("step[DRAFT") for n in names):
                phase = "draft DECODE x4 (draft > step[DRAFT])"
            elif "draft" in names:
                phase = "draft glue (draft, outside steps)"
            else:
                phase = "<none>"
            cpu = [o for o in stack if o["cat"] == "cpu_op" and o["name"] not in SKIP_OPS]
            op = cpu[-1] if cpu else None
            outer = next((o for o in cpu if o["name"].startswith(("aiter::", "sgl_kernel::", "vllm::", "sglang::"))), None)
            context[launch["args"]["correlation"]] = (phase, op, outer)
    return context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--iterations", type=int, default=1, help="profiled iterations, to report per-iteration time")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    events = json.load(gzip.open(a.trace))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    context = attribute_context(events)
    agg = defaultdict(lambda: {"n": 0, "us": 0.0, "grid": None})
    phase_total = defaultdict(float)
    for k in kernels:
        phase, op, outer = context.get(k["args"].get("correlation"), ("<unlinked>", None, None))
        opname = op["name"] if op else "-"
        if outer is not None and outer is not op:
            opname = f"{outer['name']} > {opname}"
        key = (phase, short(k["name"]), opname, dims(op) if op else "")
        row = agg[key]
        row["n"] += 1
        row["us"] += k.get("dur", 0)
        row["grid"] = k["args"].get("grid")
        phase_total[phase] += k.get("dur", 0)
    it = max(1, a.iterations)
    lines = [f"# Kernel → op → input shapes — {a.trace}", "",
             f"Per-iteration device time over {it} profiled iteration(s). Phase from host annotations.", "",
             "| phase | kernel us / iter | share of total |", "|---|---|---|"]
    total = sum(phase_total.values())
    for p, us in sorted(phase_total.items(), key=lambda x: -x[1]):
        lines.append(f"| {p} | {us / it:.1f} | {100 * us / total:.1f}% |")
    for p in sorted(phase_total, key=lambda x: -phase_total[x]):
        rows = sorted(((k, v) for k, v in agg.items() if k[0] == p), key=lambda kv: -kv[1]["us"])
        lines += ["", f"## {p}", "", "| kernel | launching op | input dims | grid | calls/iter | us/iter | % phase |",
                  "|---|---|---|---|---|---|---|"]
        for (ph, kname, opname, d), v in rows[: a.top]:
            lines.append(f"| `{kname}` | `{short(opname, 70)}` | {d} | {v['grid']} | {v['n'] / it:g} | "
                         f"{v['us'] / it:.1f} | {100 * v['us'] / phase_total[p]:.1f}% |")
    text = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
