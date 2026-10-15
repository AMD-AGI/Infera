#!/usr/bin/env python3
"""Turn a shapes_rank_N_yihou.jsonl into per-forward-mode operator shape tables (markdown).

Usage: summarize_shapes.yihou.py SHAPES.jsonl [--out FILE.md]
Rows are unique (mode, module-pattern, in, out) with the layer indices that produced them.
"""
import argparse
import json
import re
from collections import OrderedDict, defaultdict

LAYER = re.compile(r"\.layers\.(\d+)\.")
SKIP_CLASSES = {"DeepseekV2Model", "DeepseekModelNextN", "DeepseekV2DecoderLayer"}


def sig(x):
    if x is None:
        return ""
    if isinstance(x, dict) and "shape" in x:
        return f"{x['dtype']}{x['shape']}"
    if isinstance(x, dict):
        # layer_id is a per-layer scalar; keeping it would split identical-shape rows by layer.
        return ", ".join(f"{k}={sig(v)}" for k, v in x.items() if k != "layer_id" and sig(v))
    if isinstance(x, list):
        return ", ".join(s for s in (sig(i) for i in x) if s)
    return str(x)


def layer_ranges(layers):
    layers = sorted(set(layers))
    if not layers:
        return "-"
    out, start, prev = [], layers[0], layers[0]
    for n in layers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes")
    ap.add_argument("--out")
    a = ap.parse_args()
    records = [json.loads(line) for line in open(a.shapes)]
    meta = records[0] if records and records[0]["event"] == "meta" else {}
    forwards = [r for r in records if r["event"] == "forward_begin"]
    stacks, calls = defaultdict(list), []
    params = {}
    for r in records:
        if r["event"] == "module":
            if r.get("params"):
                params[LAYER.sub(".layers.*.", r["module"])] = r["params"]
            stacks[(r["module"], r.get("mode"))].append(len(calls))
            calls.append({**r, "output": None})
        elif r["event"] == "module_out":
            stack = stacks[(r["module"], r.get("mode"))]
            if stack:
                calls[stack.pop()]["output"] = r["output"]
        elif r["event"] == "attn_backend" and r["method"] != "forward":
            calls.append({"module": f"<attn_backend {r['backend']}.{r['method']}>", "class": r["backend"],
                          "mode": r.get("mode"), "args": r.get("args"), "kwargs": r.get("kwargs"),
                          "output": None, "layer": None})
    rows = OrderedDict()
    for c in calls:
        if c.get("class") in SKIP_CLASSES:
            continue
        m = LAYER.search(c["module"])
        pattern = LAYER.sub(".layers.*.", c["module"])
        key = (c.get("mode"), pattern, c.get("class"), sig(c.get("args")), sig(c.get("kwargs")), sig(c.get("output")))
        row = rows.setdefault(key, {"layers": [], "count": 0, "class": c.get("class")})
        row["count"] += 1
        if m:
            row["layers"].append(int(m.group(1)))
    lines = [f"# Operator shapes — {a.shapes}", "",
             f"meta: `{json.dumps(meta)}`", "", "## Forwards in the captured iteration", "",
             "| runner | mode | batch | tokens | extend_seq_lens |", "|---|---|---|---|---|"]
    for f in forwards:
        lines.append(f"| {f['runner']} | {f['mode']} | {f['batch_size']} | {f['num_tokens']} | {f.get('extend_seq_lens')} |")
    for mode in OrderedDict.fromkeys(k[0] for k in rows):
        lines += ["", f"## {mode}", "", "| module | class | inputs | kwargs | output | calls | layers |",
                  "|---|---|---|---|---|---|---|"]
        for (md, pattern, _cls, ins, kws, outs), row in rows.items():
            if md != mode:
                continue
            lines.append(f"| `{pattern}` | {row['class']} | {ins} | {kws} | {outs} | {row['count']} | "
                         f"{layer_ranges(row['layers'])} |")
    if params:
        lines += ["", "## Parameters (direct, per module pattern)", "", "| module | params |", "|---|---|"]
        for pattern, p in params.items():
            text = sig(p)
            if text:
                lines.append(f"| `{pattern}` | {text} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
