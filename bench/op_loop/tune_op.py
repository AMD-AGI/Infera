#!/usr/bin/env python3
"""Tune ring of the MoE-experts optimize loop (issue #40).

Coordinate-descent autotune of the ``infera_decode`` kernel's block/warp config
at a given shape: sweeps the gate/up kernel, then the down/combine kernel,
keeping only numerically-correct configs (rel < 1e-2 vs builtin), reports the
best, and ``--inject`` bakes it into the plugin (rewrites ``_TUNE_DEFAULTS`` in
infera/engine/vllm/ops/moe.py) — the "植入" step, so re-profiling picks it up.

  python tune_op.py --tokens 1                 # tune, print winner
  python tune_op.py --tokens 1 --inject        # tune + write into the plugin
"""

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from moe_experts_loop import build_inputs, timed  # noqa: E402

KEYS = (
    "INFERA_MOE_GU_BLOCK_I",
    "INFERA_MOE_GU_BLOCK_H",
    "INFERA_MOE_GU_WARPS",
    "INFERA_MOE_DN_BLOCK_H",
    "INFERA_MOE_DN_BLOCK_I",
    "INFERA_MOE_DN_WARPS",
)
GU_GRID = [(bi, bh, w) for bi in (16, 32, 64) for bh in (256, 512, 1024) for w in (4, 8)]
DN_GRID = [(bh, bi, w) for bh in (8, 16, 32) for bi in (256, 512, 1024) for w in (4, 8)]


def measure(fn, x, w1, w2, tw, ti, E, ref):
    try:
        out = fn(x, w1, w2, tw, ti, global_num_experts=E)
        rel = ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        if rel > 1e-2:
            return None, rel
        return timed(lambda: fn(x, w1, w2, tw, ti, global_num_experts=E)), rel
    except Exception:  # noqa: BLE001
        return None, float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--inject", action="store_true")
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    E, H, Dm, K, T = args.experts, args.hidden, args.inter, args.topk, args.tokens
    x, w1, w2, tw, ti = build_inputs(E, H, Dm, K, T, dt, "cuda")
    from vllm.model_executor.layers.fused_moe import fused_experts as builtin

    from infera.engine.vllm.ops.moe import _EXPERTS_VARIANTS, _TUNE_DEFAULTS

    fn = _EXPERTS_VARIANTS["infera_decode"]
    ref = builtin(x, w1, w2, tw, ti, global_num_experts=E).float()
    base_t = timed(lambda: builtin(x, w1, w2, tw, ti, global_num_experts=E))

    cfg = list(_TUNE_DEFAULTS)  # start from current baked defaults

    def setenv(c):
        for k, v in zip(KEYS, c):
            os.environ[k] = str(v)

    def best_over(grid, slot):  # slot: 0 for GU (keys 0:3), 3 for DN (keys 3:6)
        best, best_ms = None, float("inf")
        for combo in grid:
            trial = cfg.copy()
            trial[slot : slot + 3] = list(combo)
            setenv(trial)
            ms, rel = measure(fn, x, w1, w2, tw, ti, E, ref)
            if ms is not None and ms < best_ms:
                best, best_ms = combo, ms
        return best, best_ms

    print(f"tuning infera_decode @ T={T} E={E} H={H} I={Dm} top_k={K}; baseline {base_t:.4f} ms")
    gu, _ = best_over(GU_GRID, 0)
    cfg[0:3] = list(gu)
    dn, ms = best_over(DN_GRID, 3)
    cfg[3:6] = list(dn)
    setenv(cfg)
    ms, rel = measure(fn, x, w1, w2, tw, ti, E, ref)
    print(f"  default {tuple(_TUNE_DEFAULTS)} ")
    print(f"  best    {tuple(cfg)}  -> {ms:.4f} ms  ({base_t / ms:.2f}x vs builtin, rel {rel:.1e})")

    if args.inject:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "infera/engine/vllm/ops/moe.py")
        path = os.path.abspath(path)
        src = open(path).read()
        new = re.sub(r"_TUNE_DEFAULTS = \([^)]*\)", f"_TUNE_DEFAULTS = {tuple(cfg)}", src, count=1)
        open(path, "w").write(new)
        print(f"  injected _TUNE_DEFAULTS = {tuple(cfg)} into {path}")


if __name__ == "__main__":
    main()
