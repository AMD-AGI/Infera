#!/usr/bin/env python3
"""Rank kernels by graph-ON device time and attach shapes from a graph-OFF run of the same point.

graph-ON gives the deployment-mode time ranking but no host ops (replay runs no Python);
graph-OFF gives the launching op + Input Dims (aten/aiter/sgl_kernel ops) and, via the shape
recorder, Triton launch args. Join key: (phase, kernel name). Same shapes in both modes: the
graph is captured at exactly the batch the eager run uses (bs=[local_bs]).

Usage: join_hot_shapes.yihou.py ON_TRACE OFF_TRACE OFF_SHAPES.jsonl --on-iters 5 --off-iters 2
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location("ks", Path(__file__).with_name("kernel_shapes.yihou.py"))
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)


def attribute(trace):
    """(phase, kernel) -> {us, n, ops: {(op, dims): n}} using kernel_shapes' attribution logic."""
    sys.argv = ["x", trace, "--out", "/dev/null"]
    rows = defaultdict(lambda: {"us": 0.0, "n": 0, "ops": defaultdict(int), "grid": set()})
    import gzip
    events = json.load(gzip.open(trace))["traceEvents"]
    context = ks.attribute_context(events)
    for e in events:
        if e.get("cat") != "kernel":
            continue
        phase, op, outer = context.get(e["args"].get("correlation"), ("<unlinked>", None, None))
        r = rows[(phase, e["name"])]
        r["us"] += e.get("dur", 0)
        r["n"] += 1
        if e["args"].get("grid"):
            r["grid"].add(tuple(e["args"]["grid"]))
        if op is not None:
            name = op["name"] if outer is None or outer is op else f"{outer['name']} > {op['name']}"
            r["ops"][(name, ks.dims(op))] += 1
    return rows


def triton_shapes(path):
    out = defaultdict(set)
    for line in open(path):
        r = json.loads(line)
        if r["event"] == "triton":
            sig = "; ".join(f"{k}={v['dtype']}{v['shape']}" for k, v in r["tensors"].items() if v)
            out[(r.get("mode"), r["kernel"])].add(sig)
    return out


MODE_OF_PHASE = {"TARGET_VERIFY (step[VERIFY])": "TARGET_VERIFY",
                 "DRAFT_EXTEND_V2 (draft_extend)": "DRAFT_EXTEND_V2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("on_trace")
    ap.add_argument("off_trace")
    ap.add_argument("off_shapes")
    ap.add_argument("--on-iters", type=int, required=True)
    ap.add_argument("--off-iters", type=int, required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out")
    a = ap.parse_args()
    on, off = attribute(a.on_trace), attribute(a.off_trace)
    tri = triton_shapes(a.off_shapes)
    lines = []
    for phase, mode in MODE_OF_PHASE.items():
        rows = sorted(((k, v) for k, v in on.items() if k[0] == phase), key=lambda kv: -kv[1]["us"])
        total = sum(v["us"] for _, v in rows)
        lines += ["", f"## {mode} — graph-ON {total / a.on_iters:.0f} us/iter (rank 0)", "",
                  "| # | kernel | us/iter | % | calls/iter | grid (graph-OFF) | launching op | shapes |",
                  "|---|---|---|---|---|---|---|---|"]
        for i, ((_, kname), v) in enumerate(rows[: a.top], 1):
            o = off.get((phase, kname))
            ops = sorted(o["ops"].items(), key=lambda kv: -kv[1]) if o else []
            grid = ", ".join(str(list(g)) for g in sorted(o["grid"])) if o else ""
            base = kname.split("(")[0]
            tshapes = next((s for (m, k), s in tri.items() if m == mode and base.startswith(k)), None)
            if ops:
                opname = ops[0][0][0]
                shapes = " / ".join(sorted({d for (n, d), _ in ops if d}))
            elif tshapes:
                opname, shapes = "triton", " / ".join(sorted(tshapes))
            else:
                opname, shapes = "-", ""
            lines.append(f"| {i} | `{ks.short(kname, 70)}` | {v['us'] / a.on_iters:.1f} | "
                         f"{100 * v['us'] / total:.1f}% | {v['n'] / a.on_iters:g} | {grid} | `{ks.short(opname, 60)}` | {shapes} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
