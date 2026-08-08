"""MoE-experts op spec (issue #40) — the worked example for the scaffold.

Baseline = vLLM's built-in ``fused_experts`` (aiter on ROCm); candidate = the
plugin's ``infera_fused_experts`` (whichever variant ``INFERA_MOE_EXPERTS``
selects); reference = a pure-torch SwiGLU oracle. Default dims are Kimi-2.6's MoE
block. This is the template: a new op is one file like this + ``register_op``.
"""

import os

import framework as fw
import torch


def make_inputs(dims, dev):
    E, H, Dm, K, T = (dims["experts"], dims["hidden"], dims["inter"], dims["topk"], dims["tokens"])
    dt = getattr(torch, dims["dtype"])
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(T, H, dtype=dt, device=dev, generator=g) * 0.1
    w1 = (torch.randn(E, 2 * Dm, H, dtype=dt, device=dev, generator=g) * (H**-0.5)).contiguous()
    w2 = (torch.randn(E, H, Dm, dtype=dt, device=dev, generator=g) * (Dm**-0.5)).contiguous()
    logits = torch.randn(T, E, dtype=torch.float32, device=dev, generator=g)
    tw, ti = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
    return x, w1, w2, tw.contiguous(), ti.to(torch.int32).contiguous()


def baseline(x, w1, w2, tw, ti):
    from vllm.model_executor.layers.fused_moe import fused_experts

    return fused_experts(x, w1, w2, tw, ti, global_num_experts=w1.shape[0])


def candidate(x, w1, w2, tw, ti):
    from infera.engine.vllm.ops.moe import infera_fused_experts

    return infera_fused_experts(x, w1, w2, tw, ti, global_num_experts=w1.shape[0])


def reference(x, w1, w2, tw, ti):
    """Pure-torch SwiGLU MoE (router weight on output) — correctness oracle."""
    T, H = x.shape
    E, twoDm, _ = w1.shape
    Dm = twoDm // 2
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(E):
        sel = ti == e
        if not sel.any():
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        gu = xf[tok] @ w1[e].float().t()
        g, u = gu[:, :Dm], gu[:, Dm:]
        o = (torch.nn.functional.silu(g) * u) @ w2[e].float().t()
        out.index_add_(0, tok, o * tw[tok, slot].unsqueeze(1))
    return out


def traffic_bytes(dims):
    db = torch.finfo(getattr(torch, dims["dtype"])).bits // 8
    return dims["tokens"] * dims["topk"] * 3 * dims["inter"] * dims["hidden"] * db


_TUNE_ENV = (
    "INFERA_MOE_GU_BLOCK_I",
    "INFERA_MOE_GU_BLOCK_H",
    "INFERA_MOE_GU_WARPS",
    "INFERA_MOE_DN_BLOCK_H",
    "INFERA_MOE_DN_BLOCK_I",
    "INFERA_MOE_DN_WARPS",
)


def tune_grid(dims):
    # A small neighbourhood of the impactful knobs (warps + one block dim each).
    return [
        (32, gu_bh, gu_w, 8, dn_bi, dn_w)
        for gu_bh in (256, 512)
        for gu_w in (4, 8)
        for dn_bi in (256, 512)
        for dn_w in (4, 8)
    ]


def inject(cfg):
    import re

    p = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../infera/engine/vllm/ops/moe.py")
    )
    src = open(p).read()
    open(p, "w").write(
        re.sub(r"_TUNE_DEFAULTS = \([^)]*\)", f"_TUNE_DEFAULTS = {tuple(cfg)}", src, count=1)
    )


fw.register_op(
    fw.OpSpec(
        name="moe_experts",
        default_dims={
            "experts": 384,
            "hidden": 7168,
            "inter": 2048,
            "topk": 8,
            "tokens": 1,
            "dtype": "bfloat16",
        },
        make_inputs=make_inputs,
        baseline=baseline,
        candidate=candidate,
        reference=reference,
        traffic_bytes=traffic_bytes,
        tune_env=_TUNE_ENV,
        tune_grid=tune_grid,
        inject=inject,
    )
)
