#!/usr/bin/env python3
"""Per-step GPU time of named kernel groups in a profile_decode trace window (kf overlay attribution).

Usage: kernel_groups.py TRACE.json.gz [STEPS=5]. Groups match kernel names (case-insensitive substrings); a kernel
in no group counts under 'other'. Prints JSON: ms/step and calls/step per group and per distinct kernel name.
"""
import collections
import gzip
import json
import sys

GROUPS = {
    "pa_mqa_logits": ("pa_mqa_logits",),
    "radix_topk": ("radix_topk",),
    "sparse_mla": ("sparse_mla",),
    "moe_stage1": ("moe1", "moe_stage1", "stage1"),
    "moe_stage2": ("moe2", "gemm2_a4w4", "moe_stage2", "stage2"),
    "moe_sorting": ("moe_sorting",),
    "mega_mhc": ("mega_mhc",),
    "nccl": ("nccl",),
    "engram": ("engram",),
}


def group_of(name):
    low = name.lower()
    for group, keys in GROUPS.items():
        if any(key in low for key in keys):
            return group
    return "other"


def main():
    path, steps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    events = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    groups = collections.defaultdict(lambda: [0.0, 0])
    names = collections.defaultdict(lambda: [0.0, 0, ""])
    for e in events:
        g = group_of(e["name"])
        groups[g][0] += e["dur"]; groups[g][1] += 1
        n = names[e["name"]]; n[0] += e["dur"]; n[1] += 1; n[2] = g
    out = {"trace": path, "steps": steps,
           "groups": {g: {"ms_per_step": v[0] / 1000 / steps, "calls_per_step": v[1] / steps} for g, v in sorted(groups.items())},
           "kernels": [{"name": k, "group": v[2], "ms_per_step": v[0] / 1000 / steps, "calls_per_step": v[1] / steps}
                       for k, v in sorted(names.items(), key=lambda kv: -kv[1][0]) if v[2] != "other"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
