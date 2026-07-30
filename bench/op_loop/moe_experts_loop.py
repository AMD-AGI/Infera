#!/usr/bin/env python3
"""MoE-experts op optimization loop (issue #40).

The fast inner loop for iterating custom MoE kernels behind the Infera vLLM
op-injection plugin: measure the **built-in** experts kernel (baseline), the
**plugin op** (``infera.engine.vllm.ops.moe.infera_fused_experts`` — swap the
variant with ``INFERA_MOE_EXPERTS``), and a **pure-torch reference** (correctness
oracle), on identical inputs at a real model's MoE dimensions (Kimi-2.6 by
default). Reports median latency + max abs/rel error, so a new kernel is a
one-line swap → re-run → compare.

  python moe_experts_loop.py                       # Kimi-2.6 dims, builtin
  INFERA_MOE_EXPERTS=aiter python moe_experts_loop.py   # optimized variant
  python moe_experts_loop.py --experts 64 --tokens 256  # quick
"""

import argparse
import os

import torch


def build_inputs(E, H, Dm, top_k, T, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(T, H, dtype=dtype, device=device, generator=g) * 0.1
    # w1 = [gate; up] stacked → [E, 2I, H]; w2 = down → [E, H, Dm]
    w1 = (
        torch.randn(E, 2 * Dm, H, dtype=dtype, device=device, generator=g) * (H**-0.5)
    ).contiguous()
    w2 = (torch.randn(E, H, Dm, dtype=dtype, device=device, generator=g) * (Dm**-0.5)).contiguous()
    logits = torch.randn(T, E, dtype=torch.float32, device=device, generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    return x, w1, w2, weights.contiguous(), ids.to(torch.int32).contiguous()


def reference(x, w1, w2, topk_weights, topk_ids):
    """Pure-torch SwiGLU MoE (router weight applied on output) — the oracle."""
    T, H = x.shape
    E, twoI, _ = w1.shape
    Dm = twoI // 2
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(E):
        sel = topk_ids == e
        if not sel.any():
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        gu = xf[tok] @ w1[e].float().t()
        g, u = gu[:, :Dm], gu[:, Dm:]
        o = (torch.nn.functional.silu(g) * u) @ w2[e].float().t()
        out.index_add_(0, tok, o * topk_weights[tok, slot].unsqueeze(1))
    return out


def timed(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def err(a, b):
    a, b = a.float(), b.float()
    denom = b.abs().max().clamp_min(1e-6)
    return (a - b).abs().max().item(), ((a - b).abs().max() / denom).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=384)  # Kimi-2.6
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--skip-ref", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    dt = getattr(torch, args.dtype)
    E, H, Dm, top_k, T = args.experts, args.hidden, args.inter, args.topk, args.tokens
    print(
        f"MoE experts op @ Kimi-2.6 dims: E={E} H={H} Dm={Dm} top_k={top_k} tokens={T} dtype={args.dtype}\n"
        f"variant (INFERA_MOE_EXPERTS)={os.environ.get('INFERA_MOE_EXPERTS', 'builtin')}\n"
    )
    x, w1, w2, tw, ti = build_inputs(E, H, Dm, top_k, T, dt, dev)

    from vllm.model_executor.layers.fused_moe import fused_experts as builtin

    from infera.engine.vllm.ops.moe import infera_fused_experts as plugin_op

    def call(fn):
        return fn(x, w1, w2, tw, ti, global_num_experts=E)

    base_out = call(builtin)
    plug_out = call(plugin_op)

    rows = []
    if not args.skip_ref:
        ref_out = reference(x, w1, w2, tw, ti)
        rows.append(
            (
                "reference (torch oracle)",
                timed(lambda: reference(x, w1, w2, tw, ti), iters=5, warmup=1),
                err(ref_out, base_out),
            )
        )
    base_t = timed(lambda: call(builtin))
    plug_t = timed(lambda: call(plugin_op))
    rows.append(("baseline (built-in kernel)", base_t, (0.0, 0.0)))
    rows.append(("plugin op (infera_fused_experts)", plug_t, err(plug_out, base_out)))

    print(f"{'impl':34s} {'median ms':>10s} {'max|Δ| vs base':>16s} {'rel':>10s} {'speedup':>9s}")
    for name, ms, (ad, rd) in rows:
        sp = f"{base_t / ms:.2f}x" if ms > 0 else "-"
        print(f"{name:34s} {ms:10.3f} {ad:16.2e} {rd:10.2e} {sp:>9s}")


if __name__ == "__main__":
    main()
