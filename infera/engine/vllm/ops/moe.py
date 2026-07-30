###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""MoE experts injection seam (issue #40).

vLLM runs every MoE block through a modular kernel (``FusedMoEKernel`` in vLLM
0.23; its ``apply`` / ``apply_monolithic`` route → experts-GEMM → combine). We
patch those methods from ``register_ops`` (a general plugin, so ``vllm`` is
initialized) with a pass-through wrapper — the single point where an
Infera/HyperLoom experts kernel replaces the default while keeping vLLM's
routing / quant / EP-DP dispatch. This is the less-invasive alternative to
vLLM-ATOM's whole-model ``register_model`` wrapper.

Today it is a delegating pass-through (bitwise identical). ``INFERA_MOE_EXPERTS``
is reserved for selecting a custom experts implementation. The MoE modular-kernel
API moves between vLLM versions, so the patch is defensive: if the class/methods
aren't found it logs and no-ops rather than raising.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False

# MoE compute chokepoints to wrap, newest vLLM first. Extend as the API moves.
_KERNEL_CANDIDATES = (("vllm.model_executor.layers.fused_moe.modular_kernel", "FusedMoEKernel"),)
_METHODS = ("apply", "apply_monolithic")


def install_moe_ops() -> None:
    """Wrap the MoE modular-kernel compute methods with the Infera seam (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    import importlib

    kernel = None
    for mod_name, cls_name in _KERNEL_CANDIDATES:
        try:
            kernel = getattr(importlib.import_module(mod_name), cls_name)
            break
        except (ImportError, AttributeError):
            continue
    if kernel is None:
        logger.warning(
            "infera-vllm-ops: MoE seam not wired — no known modular-kernel class on this vLLM"
        )
        return

    experts = os.environ.get("INFERA_MOE_EXPERTS") or None
    wrapped = []
    for name in _METHODS:
        original = getattr(kernel, name, None)
        if original is None or getattr(original, "_infera_wrapped", False):
            continue
        setattr(kernel, name, _make_seam(original))
        wrapped.append(name)

    logger.info(
        "infera-vllm-ops: MoE experts seam on %s.{%s} (experts=%s)",
        kernel.__name__,
        ",".join(wrapped) or "<none>",
        experts or "pass-through",
    )


def _make_seam(original):
    """Delegating pass-through — the point a custom experts kernel is dropped in."""

    def _seam(self, *args, **kwargs):
        # TODO(#40): dispatch to the Infera experts-GEMM kernel when
        # INFERA_MOE_EXPERTS is set. Until then, delegate unchanged.
        return original(self, *args, **kwargs)

    _seam._infera_wrapped = True
    return _seam


# Custom experts-kernel variants register here (name -> callable with the same
# signature as vLLM's ``fused_experts``). The built-in kernel is the baseline
# (and on ROCm already dispatches to aiter); a variant is added with
# @register_experts_variant("name") and selected via ``INFERA_MOE_EXPERTS=name``.
_EXPERTS_VARIANTS: dict[str, object] = {}


def register_experts_variant(name: str):
    """Decorator: register a custom MoE experts kernel under ``name``."""

    def deco(fn):
        _EXPERTS_VARIANTS[name.lower()] = fn
        return fn

    return deco


def infera_fused_experts(*args, **kwargs):
    """The plugin's swappable MoE experts op (issue #40) — the unit the
    optimize-loop measures. ``INFERA_MOE_EXPERTS`` selects a registered variant;
    unset / ``builtin`` (or an unknown/failed variant) uses vLLM's built-in
    kernel, so the plugin op is a bitwise pass-through until a real kernel lands.
    """
    from vllm.model_executor.layers.fused_moe import fused_experts as _builtin

    variant = (os.environ.get("INFERA_MOE_EXPERTS") or "builtin").lower()
    fn = _EXPERTS_VARIANTS.get(variant)
    if fn is None:
        if variant not in ("builtin", "off", "0", ""):
            logger.warning("infera-vllm-ops: no MoE experts variant %r — using builtin", variant)
        return _builtin(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "infera-vllm-ops: experts variant %r failed (%s) — using builtin", variant, exc
        )
        return _builtin(*args, **kwargs)


# ---------------------------------------------------------------------------
# "infera_decode": decode-regime (small-M) MoE experts kernel (issue #40).
#
# At M = a few tokens x top_k, the experts op is a handful of skinny GEMVs and
# is HBM-bandwidth / launch-overhead bound; the general grouped-GEMM path
# (sort/align/scatter + tl.dot tiles sized for large M) leaves a lot on the
# table. This variant runs exactly two Triton launches and reads each selected
# expert's weights exactly once — the traffic lower bound:
#
#   kernel 1: per (token, slot) pair, expert e = topk_ids[t, s]:
#             h[t,s,:] = SiLU(x[t] @ w1[e, :I].T) * (x[t] @ w1[e, I:].T)
#             (gate/up read in one pass, fp32 accumulate, fp32 intermediate)
#   kernel 2: per token, out[t] = sum_s w2[e_ts] @ h[t,s] * topk_weight[t,s]
#             (down-proj + weighted combine fused, no atomics)
#
# Router weight on the *output* (apply_router_weight_on_input=False), matching
# vLLM's fused_experts / the harness oracle. bf16/fp16 weights, no quant.
# Above INFERA_MOE_DECODE_MAX_TOKENS tokens (default 16) it delegates to the
# built-in kernel: re-reading weights per pair cannot win once M is large.
# ---------------------------------------------------------------------------

try:  # Triton is present on the ROCm image; degrade gracefully elsewhere.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _infera_moe_gateup_silu(
        x_ptr,  # [T, H] activations
        w1_ptr,  # [E, 2I, H] gate;up stacked
        ids_ptr,  # [T*K] int expert ids
        h_ptr,  # [T*K, I] fp32 intermediate out
        H: tl.constexpr,
        I: tl.constexpr,  # noqa: E741
        stride_xt,
        stride_w1e,
        stride_w1r,
        TOPK: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pair = tl.program_id(0)  # token*TOPK + slot
        pid_i = tl.program_id(1)
        tok = pair // TOPK
        e = tl.load(ids_ptr + pair).to(tl.int64)
        ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)  # h-neuron rows
        wg_ptrs = w1_ptr + e * stride_w1e + ri[:, None] * stride_w1r
        wu_ptrs = wg_ptrs + I * stride_w1r
        acc_g = tl.zeros((BLOCK_I,), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            rh = h0 + tl.arange(0, BLOCK_H)
            xv = tl.load(x_ptr + tok * stride_xt + rh).to(tl.float32)
            wg = tl.load(wg_ptrs + rh[None, :]).to(tl.float32)
            wu = tl.load(wu_ptrs + rh[None, :]).to(tl.float32)
            acc_g += tl.sum(wg * xv[None, :], axis=1)
            acc_u += tl.sum(wu * xv[None, :], axis=1)
        h = acc_g * tl.sigmoid(acc_g) * acc_u  # SiLU(gate) * up
        tl.store(h_ptr + pair * I + ri, h)

    @triton.jit
    def _infera_moe_down_combine(
        h_ptr,  # [T*K, I] fp32 intermediate
        w2_ptr,  # [E, H, I] down
        ids_ptr,  # [T*K]
        tw_ptr,  # [T*K] fp32 router weights
        out_ptr,  # [T, H]
        H: tl.constexpr,
        I: tl.constexpr,  # noqa: E741
        stride_w2e,
        stride_w2r,
        stride_ot,
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        tok = tl.program_id(0)
        pid_h = tl.program_id(1)
        rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # output rows
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for slot in range(TOPK):
            pair = tok * TOPK + slot
            e = tl.load(ids_ptr + pair).to(tl.int64)
            rw = tl.load(tw_ptr + pair).to(tl.float32)
            w2_ptrs = w2_ptr + e * stride_w2e + rh[:, None] * stride_w2r
            acc_e = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for i0 in range(0, I, BLOCK_I):
                ri = i0 + tl.arange(0, BLOCK_I)
                hv = tl.load(h_ptr + pair * I + ri)
                wv = tl.load(w2_ptrs + ri[None, :]).to(tl.float32)
                acc_e += tl.sum(wv * hv[None, :], axis=1)
            acc += acc_e * rw
        tl.store(out_ptr + tok * stride_ot + rh, acc.to(out_ptr.dtype.element_ty))


_TUNE_KEYS = (
    "INFERA_MOE_GU_BLOCK_I",
    "INFERA_MOE_GU_BLOCK_H",
    "INFERA_MOE_GU_WARPS",
    "INFERA_MOE_DN_BLOCK_H",
    "INFERA_MOE_DN_BLOCK_I",
    "INFERA_MOE_DN_WARPS",
)
# Tuned on gfx942/gfx950 at Kimi-2.6 dims, T=1..8 (~5.2-5.3 TB/s effective on the
# selected-expert weight traffic). The tune ring rewrites this tuple in place
# (tune_op.py --inject), or supply INFERA_MOE_TUNE_FILE (JSON) / per-key env.
_TUNE_DEFAULTS = (32, 512, 8, 8, 512, 8)


def _load_tune_file() -> dict:
    path = os.environ.get("INFERA_MOE_TUNE_FILE")
    if not path:
        return {}
    try:
        import json

        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _decode_tunables():
    """Block sizes / warps. Precedence: per-key env > tune file > baked defaults,
    so profile→tune→inject→profile can drive the config without code edits."""
    fromfile = _load_tune_file()
    return tuple(
        int(os.environ.get(k, fromfile.get(k, d))) for k, d in zip(_TUNE_KEYS, _TUNE_DEFAULTS)
    )


@register_experts_variant("infera_decode")
def _infera_decode_experts(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation=None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map=None,
    quant_config=None,
    **kwargs,
):
    """Small-M fused SwiGLU experts kernel (see module comment above)."""
    import torch

    if not _HAS_TRITON:
        raise RuntimeError("triton not available")
    # bf16/fp16 GEMV only — delegate on any quantized/packed weights (e.g. Kimi
    # MXFP4 w1/w2 are fp4-packed, not float), which the builtin path handles.
    if w1.dtype not in (torch.bfloat16, torch.float16) or w2.dtype != w1.dtype:
        raise RuntimeError(f"unsupported weight dtype {w1.dtype}/{w2.dtype}")
    if activation is not None and "silu" not in str(activation).lower():
        raise RuntimeError(f"activation {activation!r} unsupported")
    if apply_router_weight_on_input:
        raise RuntimeError("apply_router_weight_on_input unsupported")
    if expert_map is not None:
        raise RuntimeError("expert_map unsupported")

    T, H = hidden_states.shape
    E, twoI, _ = w1.shape
    I = twoI // 2  # noqa: E741
    K = topk_ids.shape[1]

    # This kernel reads the expert weights once per (token, slot) pair, so it
    # wins while pairs are few relative to E (expert collisions across tokens
    # are rare and the grouped-GEMM's dedup buys nothing). Measured on gfx942:
    # win at T<=8 for E=384 (1.09-1.28x), win at T<=4 for E=64, lose beyond.
    max_tokens = int(os.environ.get("INFERA_MOE_DECODE_MAX_TOKENS", "16"))
    if T > max_tokens or 2 * T * K > E:  # large-M / collision-heavy: delegate.
        from vllm.model_executor.layers.fused_moe import fused_experts as _builtin

        return _builtin(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=quant_config,
            **kwargs,
        )

    gu_bi, gu_bh, gu_w, dn_bh, dn_bi, dn_w = _decode_tunables()
    if I % gu_bi or H % gu_bh or H % dn_bh or I % dn_bi:
        raise RuntimeError(f"dims H={H} I={I} not divisible by block sizes")

    x = hidden_states.contiguous()
    ids = topk_ids.reshape(-1).contiguous()
    tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
    hbuf = torch.empty((T * K, I), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)

    _infera_moe_gateup_silu[(T * K, I // gu_bi)](
        x,
        w1,
        ids,
        hbuf,
        H,
        I,
        x.stride(0),
        w1.stride(0),
        w1.stride(1),
        TOPK=K,
        BLOCK_I=gu_bi,
        BLOCK_H=gu_bh,
        num_warps=gu_w,
    )
    _infera_moe_down_combine[(T, H // dn_bh)](
        hbuf,
        w2,
        ids,
        tw,
        out,
        H,
        I,
        w2.stride(0),
        w2.stride(1),
        x.stride(0),
        TOPK=K,
        BLOCK_H=dn_bh,
        BLOCK_I=dn_bi,
        num_warps=dn_w,
    )
    return out
