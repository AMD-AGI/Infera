#!/usr/bin/env python3
"""Where does the engine's tuned FlyDSL MXFP8 GEMM (flydsl_bpreshuffle_mxfp8_*) come from, and which stream does it
launch on? (EXPERIMENT 4.11: the shared expert's GEMMs stay on the main stream although vLLM enqueues them on its
auxiliary stream.) Searches the installed aiter / vllm packages and prints the matching lines with context.
"""
import os
import re
import sys

import aiter
import vllm

ROOTS = [os.path.dirname(aiter.__file__), os.path.dirname(vllm.__file__)]
PATTERNS = [r"bpreshuffle_mxfp8", r"flydsl_bpreshuffle", r"def gemm_a8w8_blockscale_bpreshuffle\b",
            r"def gemm_a8w8_mxfp8_bpreshuffle\b"]


def show(path, i, lines, ctx=6):
    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
    print(f"--- {path}:{i + 1}")
    for j in range(lo, hi):
        print(f"{j + 1:6d} {lines[j].rstrip()[:200]}")


hits = 0
for root in ROOTS:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            try:
                lines = open(p, errors="replace").readlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if any(re.search(pt, line) for pt in PATTERNS):
                    show(p, i, lines)
                    hits += 1
                    if hits > 40:
                        sys.exit(0)
# The stream argument of FlyDSL launches.
fly = os.path.join(os.path.dirname(aiter.__file__), "ops", "flydsl")
for dirpath, _, files in os.walk(fly):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(dirpath, f)
            for i, line in enumerate(open(p, errors="replace").readlines()):
                if re.search(r"stream", line) and re.search(r"current_stream|cuda_stream|stream=|_stream\b", line):
                    print(f"STREAM {p}:{i + 1}: {line.strip()[:180]}")
