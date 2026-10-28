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
