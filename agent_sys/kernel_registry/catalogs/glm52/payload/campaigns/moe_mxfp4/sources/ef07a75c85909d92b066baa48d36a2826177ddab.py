"""GLM-5.2-MXFP4 decode-leg routed MoE (a4w4, MXFP4 weights + dynamic MXFP4 activations) on gfx950.

This file reproduces what sglang v0.5.21 (11bfd868) + aiter (e7d2453) run for this layer in the GLM-5.2
TP4 / attention-DP4 / EP1 deployment (SGLANG_USE_AITER=1, moe runner backend aiter, Quark MXFP4 scheme):

  weight load (once, NOT timed)  -- quark_w4a4_mxfp4_moe.py:946-992  process_weights_after_loading
      e8m0_shuffle(w13_scale), e8m0_shuffle(w2_scale)                -> prepare_weights() below
      shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16)); is_shuffled = True
  every forward (timed)          -- quark_w4a4_mxfp4_moe.py:1017-1068 apply_weights
      -> moe_runner/aiter.py:259-395 AiterRunnerCore.run
      -> aiter.fused_moe.fused_moe(..., quant_type=per_1x32, activation=Silu, gate_mode="separated",
                                   moe_sorting_dispatch_policy=2)          -> glm52_moe_mxfp4() below
  (aiter.fused_moe is wrapped by sglang's apply_aiter_small_moe_sort_patch, moe_sorting_small.py:427;
   the wrapper is inert for these shapes because topk_ids.numel() > 256.)

What aiter launches for these shapes (first-hand, torch profiler + get_2stage_cfgs trace, see program.md):
  opus_moe_sorting (P0_v2 + P23, 2 kernels)   -> sorted ids / weights / expert ids, zeroes the bf16 output
  dynamic_per_group_scaled_quant<bf16, fp4>   -> MXFP4 quant of hidden_states (group 32, e8m0)
  mxfp4_moe_sort_kernel                       -> scatter activation scales into sorted order
  mfma_moe1_silu_mul_afp4_wfp4_fp4_t32x64x256 -> gemm1 a4w4 + SiLU(gate)*up + fused MXFP4 quant of the result
  gemm2_a4w4_port_..._bm32_bn256_bk128_nt_atomic -> gemm2 a4w4, x topk weight, atomic bf16 accumulate

Public API (keep it stable: sglang calls it as a drop-in for the fused_moe call):
  prepare_weights(w13, w13_scale, w2, w2_scale)  -> layout used by glm52_moe_mxfp4 (weight-load time)
  glm52_moe_mxfp4(hidden_states, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids) -> bf16 [M, 6144]
"""
from __future__ import annotations

import os

# ---- FlyDSL gemm2 (down-proj) code-gen knobs -------------------------------------------------------------
# The a4w4 stage-2 port reads these at compile time (aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:182-201)
# and bakes them into the kernel name, so they select a different compiled instance rather than changing
# anything at run time.  Set before aiter compiles the gemm2 instance; they do not affect numerics.
#   MXFP4_G2_SPART = GroupNum*100 + M01 for ck_tile's GemmSpatiallyLocalTilePartitioner (default 402).
#     gemm2 is the per-byte-slowest of the two GEMMs (5.9 vs 6.8 TB/s) and its 19k workgroups tile a
#     403x48 (m_block, n_block) grid; 4x1 keeps a group's m-blocks -- i.e. one expert's w2 slab -- on one
#     XCD's L2 for longer than the stock 4x2.  Measured interleaved in-process A/B (paired per-rep ratios,
#     4-weight-set rotation): +0.4 % alone, ~+1 % together with MXFP4_G2_BHOIST=0.
#   MXFP4_G2_BHOIST = 0 disables hoisting the B (w2) pointer math out of the K-loop.
os.environ.setdefault("MXFP4_G2_SPART", "401")
os.environ.setdefault("MXFP4_G2_BHOIST", "0")

import torch

from aiter import ActivationType, QuantType
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle
from sglang.kernels.ops.moe.moe_sorting_small import apply_aiter_small_moe_sort_patch

# sglang AiterRunnerCore.__init__ applies this patch before the first fused_moe call.
apply_aiter_small_moe_sort_patch()
import aiter.fused_moe as _aiter_fused_moe  # noqa: E402  (module attribute is now the patched wrapper)

# sglang envs.SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY default (environ.py:908).
MOE_SORTING_DISPATCH_POLICY = 2

# ---------------------------------------------------------------------------------------------------------
# Kernel-config selection for the decode leg.
#
# aiter keys its tuned 2-stage MoE table on get_padded_M(M) = nextPow2(M) (fused_moe.py:1857); the value is
# used for nothing but that lookup (the only two call sites, fused_moe.py:1294 / 4044, pass it straight into
# the lru_cached get_2stage_cfgs; buffers are sized from the real M and from metadata.block_m).  For our
# M = 24*bs in [192, 336] that lands on the token=256 / token=512 rows, which select the block_m=32
# flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4 + t32x{128,256}x{256,128} stage-2 pair.
#
# The token=128 row of the same (gfx950, 6144, 512, 257, topk9, bf16, fp4x2/fp4x2, per_1x32) key selects a
# block_m=16 pair (flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt +
# flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16).  Halving block_m halves the number of
# padded sorted rows that both GEMMs gather (257 experts x 9*M/257 tokens => the per-expert remainder is
# the dominant term at these M), which is the one piece of non-weight traffic this bandwidth-bound op can
# still shed.  Measured in-process, interleaved A/B (3 reps, 4-weight-set rotation): +3.5 % (bs10) and
# +3.9 % (bs12) over the stock key, with block_m=32 keys (256/512/1024) all within noise of each other.
#
# Implemented as a narrowly scoped wrapper around aiter's get_padded_M: it only rewrites the key while this
# file's op is on the stack and only for the decode row counts we measured; every other M, and every other
# aiter caller in the process, gets aiter's own value.
_DECODE_KEY_M = 128               # tuned-table row to use for the decode leg
_DECODE_M_LO, _DECODE_M_HI = 97, 512   # row counts this override is validated for (M = 24*bs, bs 4..21)


def _install_decode_cfg_key():
    f = getattr(_aiter_fused_moe, "get_padded_M", None)
    if f is None or getattr(f, "_glm52_wrapped", False):
        return
    def get_padded_M(M, _orig=f):
        if _ACTIVE and _DECODE_M_LO <= M <= _DECODE_M_HI:
            return _DECODE_KEY_M
        return _orig(M)
    get_padded_M._glm52_wrapped = True
    get_padded_M._glm52_orig = f
    _aiter_fused_moe.get_padded_M = get_padded_M


# ---------------------------------------------------------------------------------------------------------
# Prologue sort: the fused (adaptive) sort instead of the two-kernel Opus sort.
#
# _moe_sorting_impl (fused_moe.py:430-440) routes the a4w4 aux sort to _adaptive_moe_sort -- one kernel that
# emits sorted ids/weights/expert ids, the m_indices + reverse_sorted extras and the atomic zero-init -- or,
# when _aux_uses_opus() says so, to the Opus multi-phase sorter (two kernels, P0_v2 + P23).  _aux_uses_opus
# picks Opus as soon as routed_rows >= num_experts, because at block_size 16 the fused path has no
# three-stage instance and runs its whole sort under blockIdx.x == 0, which costs O(routed_rows).  That
# crossover was measured on a much larger MoE (NE=896, topk=16); for this layer routed_rows is only
# 2160-3024, and the single-CTA scan is cheaper than 257 CTAs each scanning the full id list:
#   rocprofv3, bs12, 6 calls: Opus P0_v2 5.77 us + P23 5.06 us = 10.83 us  vs  fused 8.89 us (one launch).
# Both GEMMs are byte-identical across the two paths (gemm1 139.7/138.8 us, gemm2 73.8/73.9 us).
#
# Scoped to this file's op and to block_size 16, so every other aiter caller keeps aiter's own choice.
def _install_fused_prologue_sort():
    f = getattr(_aiter_fused_moe, "_aux_uses_opus", None)
    if f is None or getattr(f, "_glm52_wrapped", False):
        return

    def _aux_uses_opus(output_aux, block_size, routed_rows=None, num_experts=None, _orig=f):
        if _ACTIVE and int(block_size) == 16:
            return False
        return _orig(output_aux, block_size, routed_rows, num_experts)

    _aux_uses_opus._glm52_wrapped = True
    _aux_uses_opus._glm52_orig = f
    _aiter_fused_moe._aux_uses_opus = _aux_uses_opus


_ACTIVE = False
_install_decode_cfg_key()
_install_fused_prologue_sort()


def prepare_weights(w13: torch.Tensor, w13_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor):
    """Checkpoint layout -> kernel layout (QuarkW4A4MXFp4MoE.process_weights_after_loading, gfx950 branch).

    Inputs (logical / checkpoint layout, uint8):
      w13 [E, 2I, H/2] packed E2M1 (low nibble = even element), rows [0, I) gate, [I, 2I) up
      w13_scale [E, 2I, H/32] UE8M0;  w2 [E, H, I/2];  w2_scale [E, H, I/32]
    Runs once at weight-load time; not part of the timed op.
    """
    s0, s1, _ = w13_scale.shape
    w13_scale = e8m0_shuffle(w13_scale.reshape(s0 * s1, -1)).view(s0, s1, -1)
    s0, s1, _ = w2_scale.shape
    w2_scale = e8m0_shuffle(w2_scale.reshape(s0 * s1, -1)).view(s0, s1, -1)
    w13 = shuffle_weight(w13.contiguous(), (16, 16))
    w2 = shuffle_weight(w2.contiguous(), (16, 16))
    # apply_weights: uint8 storage viewed as float4_e2m1fn_x2, shuffled flag carried over
    w13 = w13.view(torch.float4_e2m1fn_x2)
    w2 = w2.view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    return w13, w2, w13_scale, w2_scale


def glm52_moe_mxfp4(
    hidden_states: torch.Tensor,   # bf16 [M, 6144]   (M = 24 * bs_graph after the DP all-gather)
    w13: torch.Tensor,             # prepare_weights() output
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,    # fp32 [M, 9]
    topk_ids: torch.Tensor,        # int32 [M, 9], column 8 = fused shared expert 256
) -> torch.Tensor:
    """out[t] = sum_k topk_weights[t,k] * W2_e(silu(W1_e x_t) * W3_e x_t), e = topk_ids[t,k]; a4w4 MXFP4.

    routed_scaling_factor is not applied here (aiter fused_moe has no such argument; see program.md).
    """
    global _ACTIVE
    _ACTIVE = True
    try:
        return _fused_moe_call(hidden_states, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids)
    finally:
        _ACTIVE = False


def _fused_moe_call(hidden_states, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids):
    return _aiter_fused_moe.fused_moe(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        expert_mask=None,                       # EP1: dispatcher.expert_mask_gpu stays None
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        bias1=None,
        bias2=None,
        hidden_pad=0,
        intermediate_pad=0,                     # I=512 is already 128-aligned (get_moe_weight_sizes)
        gate_mode=GateMode.SEPARATED.value,     # Quark checkpoints: gate rows then up rows
        moe_sorting_dispatch_policy=MOE_SORTING_DISPATCH_POLICY,
    )
