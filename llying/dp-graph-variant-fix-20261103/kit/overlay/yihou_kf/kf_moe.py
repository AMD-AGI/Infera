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
# No `from __future__ import annotations`: FlyDSL reads the kernel parameter annotations (fx.Int64, ...) as
# live types when it traces the local gemm1 launcher below.
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


# ---------------------------------------------------------------------------------------------------------
# Prologue sort, one launch: expert-partitioned multi-CTA sort + bf16 zero fill (HIP, built with hiprtc).
#
# aiter's fused sort (sort_quant_kernel_impl<257,9,16,6144,128,1024,true,false>) runs the whole sort in
# CTA 0 of a 128-CTA launch: count from global, scan, a placement loop that re-reads ids/weights from global
# for every pass, and one thread per expert writing its padding / block ids.  All ~16k scattered dword
# stores leave through that one CU.  Measured (rocprofv3, bs10/bs12, this shape): 8.2-8.9 us cold-L2,
# 7.0-7.5 us warm; the zero fill alone is ~1 us of that and runs on the other 127 CTAs.
#
# Replacement, same buffers and the same output contract (moe_sort_quant.cuh:133-213):
#   * ceil(NE/EPC) sort CTAs of 1024 threads.  Every sort CTA loads all M*topk ids + weights once (one
#     global round trip, <= 4 pairs per thread, kept in registers) and builds the full per-expert count in
#     LDS with atomics; for pairs of its own EPC experts the atomic's return value is the pair's rank in
#     the expert's segment, so counting and ranking are one pass.  Padded counts -> DPP wave scan -> starts.
#     The CTA then writes only its own experts' rows (sorted ids with the topk slot in bits 24..31,
#     m_indices, weights, reverse_sorted), their padding rows (id = m_index = M, weight 0) and block expert
#     ids.  Each expert segment is owned by exactly one CTA, so per-CTA atomic ranks never collide, and the
#     scattered stores are spread over ~33 CUs instead of one.
#   * with the pre-quantized gemm1 (GLM52_MOE_G1=3, below): ceil(M*K/32 / 1024) CTAs quantize hidden_states
#     to token-major MXFP4 (a_q [M, K/2] + e8m0 a_s [M, K/32]), one 32-element group per thread, with the
#     exact rule of aiter's gemm1 inline quant (bitwise-identical gemm1 output); they finish inside the sort
#     CTAs' critical path (prologue 5.2-5.5 us traced either way).
#   * the remaining CTAs zero the [M, model_dim] bf16 accumulation buffer (16 B per store); with the
#     pre-quant CTAs present they do it themselves after quantizing (_FILL_IN_QUANT).
# Only the order of rows inside an expert's segment can differ from aiter's (aiter's is also atomic
# order); both GEMMs address rows by sorted position and gemm2 accumulates with atomics either way.
# Measured standalone (rocprofv3): 5.0-5.3 us cold-L2 / 5.5-5.7 us warm vs 8.2-8.9 / 7.0-7.5 us.
# Validated against aiter's sort on the workset routings and on uniform / skewed / sparse / duplicate /
# exact-16 routings for M in 1..455: identical num_valid_ids, block expert ids, padding, per-segment
# (id, weight) multisets, m_indices, reverse_sorted round trip, and an all-zero output buffer.
#
# Scoped like the wrappers above (this op only, block_size 16, atomic + aux).  Falls back to aiter's sort
# for shapes outside the kernel's limits (M*topk > 4096, > 1024 experts), when hiprtc is unavailable, or
# when a graph capture would be the first use (module load is not capturable).
_HIP_PROLOGUE = os.environ.get("GLM52_MOE_HIP_PROLOGUE", "1") == "1"
_HIP_SORT_EPC = 8             # experts per sort CTA
# threads per CTA (sort, quant and fill).  512 vs 1024: prologue -0.08..-0.28 us (rocprofv3 graph trace, one run
# each, run-to-run noise ~0.15 us); the sort needs NT >= num_experts (one expert per thread in the scan).
_HIP_SORT_NT = int(os.environ.get("GLM52_MOE_SORT_NT", "512"))
_HIP_SORT_MAXP = 4096 // _HIP_SORT_NT                                # pairs per sort thread -> M*topk <= 4096
_HIP_SORT_MAXE = 1024
# With the pre-quant CTAs present they also zero moe_buf after quantizing (they finish ~1.4 us before the sort
# CTAs), instead of ceil(n_vec / 1024) extra fill CTAs: 78 instead of ~258 CTAs at bs10, prologue -0.1 us
# (rocprofv3 graph trace, inside noise for one of four cases).  GLM52_MOE_FILL_IN_QUANT=0 restores the fill CTAs.
_FILL_IN_QUANT = os.environ.get("GLM52_MOE_FILL_IN_QUANT", "1") == "1"

# _ONE_ATOMIC: count every pair with the returning LDS atomic (its value is used only for the CTA's own experts)
# instead of a returning + a non-returning instruction per wave; prologue -0.08..-0.16 us (rocprofv3 graph trace).
_HIP_SORT_SRC = ("#define _ONE_ATOMIC " + os.environ.get("GLM52_MOE_ONE_ATOMIC", "1") + "\n"
                 # moe_buf zero fill with non-temporal stores: no dirty L2 lines left for gemm1's start to write
                 # back.  rocprofv3 graph trace: prologue -0.32 us, gemm1 -1.08 us; untraced driver A/B (3 reps):
                 # mean case speedup 1.0967 -> 1.0980.  GLM52_MOE_NT_ZERO=0 restores plain stores.
                 + ("#define ZSTORE(p, v) __builtin_nontemporal_store((v), (p))\n"
                    if os.environ.get("GLM52_MOE_NT_ZERO", "1") == "1" else "#define ZSTORE(p, v) (*(p) = (v))\n")
                 + f"#define GLM52_NT {_HIP_SORT_NT}\n#define GLM52_MAXP {_HIP_SORT_MAXP}\n") + r"""
typedef int glm52_v4i __attribute__((ext_vector_type(4)));
typedef __bf16 glm52_bf16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int glm52_dpp_inclusive_scan_wave(int x) {
    int t;
    t = __builtin_amdgcn_mov_dpp(x, 0x111, 0xF, 0xF, true); x += t;     // row_shr:1
    t = __builtin_amdgcn_mov_dpp(x, 0x112, 0xF, 0xF, true); x += t;     // row_shr:2
    t = __builtin_amdgcn_mov_dpp(x, 0x114, 0xF, 0xF, true); x += t;     // row_shr:4
    t = __builtin_amdgcn_mov_dpp(x, 0x118, 0xF, 0xF, true); x += t;     // row_shr:8
    t = __builtin_amdgcn_update_dpp(0, x, 0x142, 0xA, 0xF, true); x += t;  // row_bcast:15
    t = __builtin_amdgcn_update_dpp(0, x, 0x143, 0xC, 0xF, true); x += t;  // row_bcast:31
    return x;
}

extern "C" __global__ void __launch_bounds__(GLM52_NT)
glm52_moe_sort_fill(const int* __restrict__ topk_ids,       // [M*topk]
                    const float* __restrict__ topk_w,       // [M*topk]
                    int* __restrict__ sorted_ids,
                    float* __restrict__ sorted_w,
                    int* __restrict__ sorted_eids,
                    int* __restrict__ num_valid,            // [2]
                    int* __restrict__ reverse_sorted,       // [M*topk]
                    int* __restrict__ m_indices,
                    glm52_v4i* __restrict__ zero_out,       // bf16 [M, model_dim] as 16 B vectors
                    int M, int n_pairs, int topk, int ne, int epc, int n_sort, long long n_zero_vec,
                    const glm52_v4i* __restrict__ hidden,   // bf16 [M, K] as 16 B vectors (quant CTAs)
                    glm52_v4i* __restrict__ a_q,            // fp4x2 [M, K/2] as 16 B vectors
                    unsigned char* __restrict__ a_s,        // e8m0 [M, K/32]
                    int n_quant, int k_groups, int fill_in_quant)
{
    constexpr int NT = GLM52_NT, MAXP = GLM52_MAXP, MAXE = 1024, BM = 16, NW = NT / 64;
    const int tid = threadIdx.x;
    if ((int)blockIdx.x < n_sort) {
        __shared__ int cnt[MAXE];
        __shared__ int start_s[MAXE];
        __shared__ int wave_tot[NW];
        const int e0 = blockIdx.x * epc;
        const int e1 = min(e0 + epc, ne);
        int ids[MAXP];
        float ws[MAXP];
#pragma unroll
        for (int k = 0; k < MAXP; ++k) {
            const int i = tid + k * NT;
            const bool v = i < n_pairs;
            ids[k] = v ? topk_ids[i] : -1;
            ws[k] = v ? topk_w[i] : 0.f;
        }
        for (int e = tid; e < ne; e += NT) cnt[e] = 0;
        __syncthreads();
        int rank[MAXP];
#pragma unroll
        for (int k = 0; k < MAXP; ++k) {
            const int e = ids[k];
            if (_ONE_ATOMIC) {
                rank[k] = (e >= 0 && e < ne) ? atomicAdd(&cnt[e], 1) : 0;
            } else {
                rank[k] = 0;
                if (e >= e0 && e < e1) rank[k] = atomicAdd(&cnt[e], 1);
                else if (e >= 0 && e < ne) atomicAdd(&cnt[e], 1);
            }
        }
        __syncthreads();
        const int lane = tid & 63, wave = tid >> 6;
        const int c = tid < ne ? cnt[tid] : 0;
        const int p = (c + BM - 1) / BM * BM;
        const int v = glm52_dpp_inclusive_scan_wave(p);
        if (lane == 63) wave_tot[wave] = v;
        __syncthreads();
        int pre = 0;
#pragma unroll
        for (int w = 0; w < NW; ++w) pre += (w < wave) ? wave_tot[w] : 0;
        const int incl = v + pre;
        if (tid >= e0 && tid < e1) start_s[tid] = incl - p;
        __syncthreads();
        if (blockIdx.x == 0 && tid == ne - 1) { num_valid[0] = incl; num_valid[1] = M; }
#pragma unroll
        for (int k = 0; k < MAXP; ++k) {
            const int e = ids[k];
            if (e >= e0 && e < e1) {
                const int i = tid + k * NT;
                const int tok = i / topk;
                const int slot = i - tok * topk;
                const int sp = start_s[e] + rank[k];
                sorted_ids[sp] = (tok & 0x00FFFFFF) | ((slot & 0xFF) << 24);
                m_indices[sp] = tok & 0x00FFFFFF;
                sorted_w[sp] = ws[k];
                reverse_sorted[i] = sp;
            }
        }
        const int pv = M & 0x00FFFFFF;          // padding row: gemm1 drops the load, gemm2 the store
        for (int idx = tid; idx < epc * BM; idx += NT) {
            const int e = e0 + idx / BM, j = idx % BM;
            if (e < e1) {
                const int ce = cnt[e];
                const int pe = (ce + BM - 1) / BM * BM;
                if (ce + j < pe) {
                    const int pos = start_s[e] + ce + j;
                    sorted_ids[pos] = pv;
                    m_indices[pos] = pv;
                    sorted_w[pos] = 0.f;
                }
            }
        }
        for (int idx = tid; idx < epc * 32; idx += NT) {
            const int e = e0 + idx / 32;
            if (e < e1) {
                const int nb = (cnt[e] + BM - 1) / BM;
                const int b0 = start_s[e] / BM;
                for (int j = idx % 32; j < nb; j += 32) sorted_eids[b0 + j] = e;
            }
        }
    } else if ((int)blockIdx.x < n_sort + n_quant) {
        // MXFP4 quant of hidden_states, one 32-element group per thread, bit-identical to the inline quant
        // of aiter's gemm1 (mxfp4_gemm1.py _iq_block_amax / _inline_quant_core_batch / _iq_pack_store and
        // mxfp4_gemm_common.py _inline_e8m0 / _e8m0_roundup): amax over the bf16 magnitude bits, e8m0 =
        // round-up exponent of amax / 6 capped at 254, elements via v_cvt_scalef32_pk_fp4_bf16.
        const long long g = (long long)(blockIdx.x - n_sort) * NT + tid;
        if (g < (long long)M * k_groups) {
            glm52_v4i v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = hidden[g * 4 + i];
            unsigned m = 0u;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned d = (unsigned)v[i][j] & 0x7FFF7FFFu;
                    m = max(m, max(d & 0xFFFFu, d >> 16));
                }
            const float amax = __uint_as_float(m << 16);
            const unsigned wi = __float_as_uint(__fmul_rn(amax, __uint_as_float(0x3E2AAAABu)));   // f32(1/6)
            const unsigned bexp = ((wi + 0x7FFFFFu) >> 23) & 0xFFu;
            const unsigned e8 = bexp < 254u ? bexp : 254u;
            const float sc = __uint_as_float(e8 << 23);
            glm52_v4i o;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // dwords as separate scalars: converting straight from vector elements made the compiler
                // feed element 0 to all four converts (checked in the disassembly)
                const glm52_bf16x2 p0 = __builtin_bit_cast(glm52_bf16x2, (unsigned)v[i].x);
                const glm52_bf16x2 p1 = __builtin_bit_cast(glm52_bf16x2, (unsigned)v[i].y);
                const glm52_bf16x2 p2 = __builtin_bit_cast(glm52_bf16x2, (unsigned)v[i].z);
                const glm52_bf16x2 p3 = __builtin_bit_cast(glm52_bf16x2, (unsigned)v[i].w);
                unsigned pk = 0u;
                pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, p0, sc, 0);
                pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, p1, sc, 1);
                pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, p2, sc, 2);
                pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(pk, p3, sc, 3);
                o[i] = (int)pk;
            }
            a_q[g] = o;
            a_s[g] = (unsigned char)e8;
        }
        if (fill_in_quant) {
            const glm52_v4i z = {0, 0, 0, 0};
            const long long stride = (long long)n_quant * NT;
            for (long long i = (long long)(blockIdx.x - n_sort) * NT + tid; i < n_zero_vec; i += stride)
                ZSTORE(zero_out + i, z);
        }
    } else {
        const glm52_v4i z = {0, 0, 0, 0};
        const long long stride = (long long)(gridDim.x - n_sort - n_quant) * NT;
        for (long long i = (long long)(blockIdx.x - n_sort - n_quant) * NT + tid; i < n_zero_vec; i += stride)
            ZSTORE(zero_out + i, z);
    }
}
"""

# Output (moe_buf) allocation granularity.
#
# gemm2 accumulates the 9 weighted expert outputs of every token into moe_buf with bf16 atomics.  In the
# driver's graph (4 calls per replay, the previous call's output still alive) a plain torch.empty puts
# consecutive calls' 2.4-4.1 MB outputs at alternating offsets of the caching allocator's 20 MiB segments,
# and the calls whose buffer starts at a non-4-MiB offset ran gemm2 ~3.5 us slower (MI355X n02-25, rocprofv3
# graph trace: 75.0-76.0 vs 71.6-72.2 us at bs8..12).  Requesting the buffer in whole 4 MiB blocks keeps every
# call's moe_buf on a 4 MiB boundary of its segment and all four calls in the fast mode (71.4-73.2 us);
# a 2 MiB offset inside an 8 MiB block made every call slow (73.4-79.6 us).  The mechanism (which physical
# property of the placement the atomics depend on) is not established.  Interleaved driver A/B (4 reps):
# mean case speedup 1.0905 -> 1.0965.  The returned tensor is a contiguous [M, model_dim] view at offset 0
# of its block; at most 4 MiB - 1 B of padding per call.  GLM52_MOE_MOEBUF_ROUND_MIB=0 restores
# the plain allocation.
_MOEBUF_ROUND = int(os.environ.get("GLM52_MOE_MOEBUF_ROUND_MIB", "4")) * 1024 * 1024

_hip_sort_mods: dict = {}     # device index -> (launch fn, hipFunction_t) or None (unavailable)


def _hip_lib(stem):
    """dlopen the HIP library instance torch already uses (same runtime => torch's streams are valid)."""
    import ctypes
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split()[-1] if line.strip() else ""
                if os.path.basename(path).startswith(stem + ".so"):
                    return ctypes.CDLL(path)
    except OSError:
        pass
    return ctypes.CDLL(stem + ".so")


def _hip_sort_build(device_index):
    import ctypes
    rtc = _hip_lib("libhiprtc")
    hip = _hip_lib("libamdhip64")
    prog = ctypes.c_void_p()
    if rtc.hiprtcCreateProgram(ctypes.byref(prog), _HIP_SORT_SRC.encode(), b"glm52_moe_sort_fill.hip",
                               0, None, None) != 0:
        raise RuntimeError("hiprtcCreateProgram failed")
    try:
        arch = torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]
        opts = [f"--offload-arch={arch}".encode(), b"-O3"]
        if rtc.hiprtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)) != 0:
            n = ctypes.c_size_t()
            rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(max(1, n.value))
            rtc.hiprtcGetProgramLog(prog, log)
            raise RuntimeError("hiprtc compile failed: " + log.value.decode(errors="replace"))
        n = ctypes.c_size_t()
        rtc.hiprtcGetCodeSize(prog, ctypes.byref(n))
        code = ctypes.create_string_buffer(n.value)
        rtc.hiprtcGetCode(prog, code)
    finally:
        rtc.hiprtcDestroyProgram(ctypes.byref(prog))
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    with torch.cuda.device(device_index):
        if hip.hipModuleLoadData(ctypes.byref(mod), code) != 0:
            raise RuntimeError("hipModuleLoadData failed")
        if hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"glm52_moe_sort_fill") != 0:
            raise RuntimeError("hipModuleGetFunction failed")
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    launch.restype = ctypes.c_int
    return launch, fn, code          # keep the code buffer alive with the module


def _hip_sort_fn(device_index):
    st = _hip_sort_mods.get(device_index, False)
    if st is False:
        if torch.cuda.is_current_stream_capturing():
            return None              # first use inside a capture: module load is not capturable
        try:
            st = _hip_sort_build(device_index)
        except Exception as e:       # noqa: BLE001 - any failure -> keep aiter's sort
            import warnings
            warnings.warn(f"glm52 HIP prologue sort unavailable, using aiter's sort: {e}")
            st = None
        _hip_sort_mods[device_index] = st
    return st


def _hip_moe_sort(st, topk_ids, topk_weights, num_experts, topk, model_dim, hidden=None):
    import ctypes
    launch, fn, _ = st
    device = topk_ids.device
    M = topk_ids.shape[0]
    BM = 16
    # buffers exactly as aiter's _adaptive_moe_sort (fused_moe.py:160-176) allocates them
    active = min(num_experts, M * topk)
    max_sorted = (((M * topk + active * (BM - 1)) + BM - 1) // BM) * BM
    sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.empty(max_sorted // BM, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device=device)
    reverse_sorted = torch.empty(M * topk, dtype=torch.int32, device=device)
    m_indices = torch.empty(max_sorted, dtype=torch.int32, device=device)
    # gemm2's bf16 atomics accumulate into moe_buf; request it in whole _MOEBUF_ROUND blocks (see there)
    nel = M * model_dim
    if _MOEBUF_ROUND > 0:
        rnd = _MOEBUF_ROUND // 2
        moe_buf = torch.empty(-(-nel // rnd) * rnd, dtype=torch.bfloat16, device=device)[:nel].view(M, model_dim)
    else:
        moe_buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)
    n_vec = (M * model_dim * 2) // 16
    n_sort = -(-num_experts // _HIP_SORT_EPC)
    n_fill = min(max(1, -(-n_vec // _HIP_SORT_NT)), 1024)
    n_quant, k_groups, a_q, a_s = 0, 0, None, None
    if hidden is not None:
        k_groups = hidden.shape[1] // 32
        a_q = torch.empty((M, hidden.shape[1] // 2), dtype=torch.uint8, device=device)
        a_s = torch.empty((M, k_groups), dtype=torch.uint8, device=device)
        n_quant = -(-(M * k_groups) // _HIP_SORT_NT)
    args = [ctypes.c_void_p(t.data_ptr()) for t in (topk_ids, topk_weights, sorted_token_ids, sorted_weights,
                                                       sorted_expert_ids, num_valid_ids, reverse_sorted,
                                                       m_indices, moe_buf)]
    args += [ctypes.c_int(M), ctypes.c_int(M * topk), ctypes.c_int(topk), ctypes.c_int(num_experts),
             ctypes.c_int(_HIP_SORT_EPC), ctypes.c_int(n_sort), ctypes.c_longlong(n_vec)]
    args += [ctypes.c_void_p(t.data_ptr() if t is not None else 0) for t in (hidden, a_q, a_s)]
    fiq = 1 if (n_quant > 0 and _FILL_IN_QUANT) else 0
    if fiq:
        n_fill = 0
    args += [ctypes.c_int(n_quant), ctypes.c_int(k_groups), ctypes.c_int(fiq)]
    params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = torch.cuda.current_stream(device).cuda_stream
    r = launch(fn, n_sort + n_quant + n_fill, 1, 1, _HIP_SORT_NT, 1, 1, 0, ctypes.c_void_p(stream), params, None)
    if r != 0:
        raise RuntimeError(f"glm52 prologue sort: hipModuleLaunchKernel returned {r}")
    if hidden is not None:
        global _PREQ
        _PREQ = (hidden, M, a_q, a_s)
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf, m_indices, reverse_sorted


def _install_hip_prologue():
    f = getattr(_aiter_fused_moe, "_adaptive_moe_sort", None)
    if f is None or getattr(f, "_glm52_wrapped", False):
        return

    def _adaptive_moe_sort(topk_ids, topk_weights, num_experts, topk, block_size, model_dim, *,
                           atomic=False, emit_aux=False, skip_quant=False, moebuf_dtype=torch.bfloat16, _orig=f):
        if (_ACTIVE and _HIP_PROLOGUE and int(block_size) == 16 and atomic and emit_aux and not skip_quant
                and moebuf_dtype == torch.bfloat16 and topk_ids.is_cuda
                and topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
                and topk_ids.is_contiguous() and topk_weights.is_contiguous()
                and topk_ids.dim() == 2 and topk_ids.shape[1] == topk and topk_weights.shape == topk_ids.shape
                and 0 < topk_ids.shape[0] * topk <= _HIP_SORT_NT * _HIP_SORT_MAXP
                and 0 < num_experts <= min(_HIP_SORT_MAXE, _HIP_SORT_NT) and (model_dim * 2) % 16 == 0):
            dev = topk_ids.device.index if topk_ids.device.index is not None else torch.cuda.current_device()
            st = _hip_sort_fn(dev)
            if st is not None:
                h = _CUR_HIDDEN
                use_h = (h is not None and _G1_VARIANT == 3 and h.dtype == torch.bfloat16 and h.is_cuda
                         and h.dim() == 2 and h.is_contiguous() and h.shape[0] == topk_ids.shape[0]
                         and h.shape[1] == model_dim and model_dim % 256 == 0)
                return _hip_moe_sort(st, topk_ids, topk_weights, int(num_experts), int(topk), int(model_dim),
                                     hidden=h if use_h else None)
        return _orig(topk_ids, topk_weights, num_experts, topk, block_size, model_dim, atomic=atomic,
                     emit_aux=emit_aux, skip_quant=skip_quant, moebuf_dtype=moebuf_dtype)

    _adaptive_moe_sort._glm52_wrapped = True
    _adaptive_moe_sort._glm52_orig = f
    _aiter_fused_moe._adaptive_moe_sort = _adaptive_moe_sort


# ---------------------------------------------------------------------------------------------------------
# gemm1: aiter's FlyDSL stage-1 kernel with the fused shared expert's tiles spread over the XCDs.
#
# Local copy of aiter's launcher (compile_gemm1_a4w4_port, mxfp4_gemm1.py:1497-1783) for exactly the decode
# instance this file selects (BM16, inline activation quant + hidden prefetch, BN128, xcd_swizzle 2, NE257).
# Variants 1/2 run aiter's kernel body (_gemm1_body) unchanged; variant 3 runs _g1_body_preq below.  The
# gemm1 output of every variant is bitwise identical to aiter's (checked at bs 8/10/12/14 on the workset
# routings; end-to-end on synthetic routings incl. no shared column, skewed, M = 1..600).
#
# aiter's mapping hands every XCD a contiguous run of total_m_blocks tiles.  The fused shared expert (id 256,
# every token, sorted last) owns the final S = ceil(M/16) m-blocks, so all of its 8*S tiles land on XCD 7.
# Here XCD xc (= workgroup id % 8, verified with HW_REG_XCC_ID) first runs the S shared m-blocks of n-block
# xc -- one 1/8 slice of the shared expert's w13 per XCD, read by S concurrent workgroups that get it from
# that XCD's L2 -- then its share of the routed tiles in aiter's xcd_swizzle grouping.  The shared tiles load
# B with the temporal policy, all other tiles keep aiter's non-temporal B loads.  S is derived on the device
# from sorted_expert_ids (trailing m-blocks of the last expert, at most ceil(M/16)), so the launch arguments
# are aiter's; any S keeps the mapping a bijection onto [0, 8 * total_m_blocks), so correctness does not
# depend on the routing.
#
# Pre-quantized activations (variant 3, default): the inline quant made every one of the 8 n-block tiles of
# an m-block read its 16 bf16 hidden rows (512 B per row and K-tile) and the spread placement makes every XCD
# read all M rows for the shared tiles.  The HIP prologue now quantizes hidden_states once (token-major
# MXFP4 + e8m0, same rule, bitwise-identical result) and gemm1 runs _g1_body_preq -- aiter's body with the
# A path reading those 136 B per row and K-tile straight into the same LDS layout.
#
# Measured on MI355X (gfx950), this shape (n03-33):
#   gemm1 alone, eager, 4 rotating weight sets, bs10 / bs12: aiter 133.0 / 133.9 us -> spread 129.7 / 130.7
#     -> spread + pre-quant 126.9 / 127.3 us
#   rocprof-compute PMC, spread: fabric reads 906.7 -> 895.5 MB (bs10), 917.8 -> 905.4 MB (bs12)
#   scored benchmark, traced (graph replay): gemm1 137.0 / 137.9 us -> 132.2 / 131.2 us; prologue unchanged
#   driver --bench-mode, 7 interleaved untraced reps vs aiter's launcher: case_ms -2.5 .. -5.9 us per case
# Tried and not kept (same session): several shared m-blocks per workgroup, either as barrier-locked
# 512..1024-thread workgroups (+6 / +22 us) or as one B stream feeding 2-3 m-blocks in registers (bitwise
# exact, but fabric bytes unchanged: the shared B was already served by L2); other cache policies for the
# routed B (any policy without the NT bit +12 %, NT variants equal); NT B scales (+1..2 us); a 2-D (m x n)
# split of the shared tiles over the XCDs (within 0.5 us); gemm2 with the partitioner in n-major order
# (+7..10 us).
#
# GLM52_MOE_G1: 0 = aiter's own launcher, 1 = local copy with aiter's mapping (control), 2 = spread,
# 3 = spread + pre-quantized activations (default; falls back to 2 when this call's prologue did not
# produce them, e.g. aiter's sort fallback for M*topk > 4096).
_G1_VARIANT = int(os.environ.get("GLM52_MOE_G1", "3"))

try:
    import functools as _functools

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, rocdl
    from flydsl.expr.typing import T, as_ir_value
    from aiter.ops.flydsl import moe_kernels as _aiter_moe_kernels
    from aiter.ops.flydsl import mxfp4_gemm1_kernels as _aiter_g1k
    from aiter.ops.flydsl.kernels import mxfp4_gemm1 as _aiter_g1
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import _gemm1_body
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _global_i32_at, _udiv, _umod
    _G1_IMPORT_OK = True
except Exception:  # noqa: BLE001 - any import drift -> aiter's own gemm1
    _G1_IMPORT_OK = False


if _G1_IMPORT_OK:
    from flydsl.expr import range_constexpr
    from aiter.ops.flydsl.kernels import mxfp4_gemm_common as _g1c

    @flyc.jit
    def _g1_body_preq(
        lds_raw_ptr, arg_aq, arg_as, arg_bq, arg_bscale, arg_eids, arg_mind, arg_aqout, arg_ascaleout,
        tile, lane, wave, i32_ntok, i32_total_m_blocks, *, b_nt, K, N_OUT, NE, k_stages,
    ):
        """aiter's _gemm1_body (mxfp4_gemm1.py:58-1457) specialised to this decode instance (BM16, BN128, BK256,
        fp4 in/out, SiLU, native BM16 scale layout, 4 waves, k_wave 1, epi_splits 1), with the A operand read
        already quantized (token-major MXFP4 a_q [M, K/2] + e8m0 a_s [M, K/32], written by the prologue with
        the inline quant's exact rule) instead of bf16 hidden rows quantized per tile.  Every lane stores the
        same 4 B of fp4 and the same packed scale dword to the same LDS location as aiter's inline-quant
        path (_iq_pack_store / inline_quant_pack_write), so the MFMA inputs and the output are bit-identical,
        while each tile reads 136 B of A per row and K-tile instead of 512 B."""
        BM, BN, BK = 16, 128, 256
        KH_TILE = BK // 2
        K_HALF = _g1c.k_half_for(K)
        K_TILES = K // BK
        K_GROUPS = K // 32
        kBS_stride_n0_dw = _g1c.kbs_stride_n0_dw_for(K)
        kBS_per_expert_dw = _g1c.kbs_per_expert_dw_for(N_OUT, K)
        BQ_BYTES = _g1c.bq_bytes_for(NE, N_OUT, K)
        BSCALE_BYTES = _g1c.bscale_bytes_for(NE, N_OUT, K)
        NUM_N_BLOCKS = N_OUT // BN
        inter = N_OUT // 2
        OUT_AS_PER_CHUNK_DW = _g1c.kas_per_chunk_dw_for(inter)
        OUT_ROW_BYTES = _g1c.k_half_for(inter)
        kAStages = k_stages + 1
        kUnroll = K_TILES - k_stages
        BN_INT = BN // 2
        N_REPS = BN // (4 * 16)          # 2
        ACC_LDS_STRIDE = BN + 4
        S_AQ_BYTES = kAStages * BM * KH_TILE

        wave_n = wave % fx.Int32(4)
        mem_1x1 = fx.make_layout(1, 1)
        mem_2x1 = fx.make_layout(2, 1)
        mem_4x1 = fx.make_layout(4, 1)

        n_block_idx = tile % fx.Int32(NUM_N_BLOCKS)
        m_block_idx = tile // fx.Int32(NUM_N_BLOCKS)
        e = rocdl.readfirstlane(T.i32, as_ir_value(_global_i32_at(arg_eids, m_block_idx)))
        m_row = m_block_idx * fx.Int32(BM)

        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)

        bq_tiles = _g1c._global_i32_buffer_tiles(arg_bq, BQ_BYTES, 4)
        bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if b_nt else 0), fx.Int32)
        bscale_tiles = _g1c._global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
        bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        aq_tiles = _g1c._global_i32_buffer_tiles(arg_aq, fx.Int64(i32_ntok * fx.Int32(K_HALF)), 1)
        aq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        as_tiles = _g1c._global_i32_buffer_tiles(arg_as, fx.Int64(i32_ntok * fx.Int32(K_GROUPS)), 2)
        as_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

        rcls = wave_n * fx.Int32(4) + lane_div_16
        row_tok = _global_i32_at(arg_mind, m_row + rcls)

        N0_HALF = N_OUT // 32
        b_load_s_base = []
        for j in range_constexpr(N_REPS):
            tile_il = n_block_idx * fx.Int32(BN // 16) + wave_n * fx.Int32(N_REPS) + fx.Int32(j)
            gsel = tile_il & fx.Int32(1)
            n0 = tile_il >> fx.Int32(1)
            col = (gsel * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
            v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
            b_load_s_base.append(rocdl.readfirstlane(T.i32, v))
        np_gate = n_block_idx * fx.Int32(2) + wave_n // fx.Int32(2)
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
        b_scale_s_base, b_scale_s_base_hi = [], []
        for mw in range_constexpr(2):
            base = (e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)) * fx.Int32(4)
            base = rocdl.readfirstlane(T.i32, base)
            b_scale_s_base.append(base)
            b_scale_s_base_hi.append(base + fx.Int32(16 * _g1c.kBS_stride_k0_dw * 4))

        scale_atoms = _g1c._scale_mma_atoms("fp4")
        accm = [fx.make_rmem_tensor(mem_4x1, fx.Float32) for _ in range(N_REPS)]
        for J in range_constexpr(N_REPS):
            accm[J].store(fx.Vector.filled(4, 0.0, fx.Float32))
        b = [[[None, None] for _ in range(N_REPS)] for _ in range(k_stages)]
        b_scale_v = [[None, None] for _ in range(k_stages)]

        # LDS: [s_aq (kAStages x BM x KH_TILE) | s_asc (K_TILES x 256 B)], reused as the f32 epilogue accumulator
        flat = fx.make_view(fx.recast_iter(fx.Int32, lds_raw_ptr), fx.make_layout(S_AQ_BYTES // 4, 1))
        s_aq_x4 = fx.logical_divide(flat, mem_4x1)
        s_aq_x1 = fx.logical_divide(flat, mem_1x1)
        aflat = fx.make_view(fx.recast_iter(fx.Int32, fx.add_offset(lds_raw_ptr, S_AQ_BYTES)),
                             fx.make_layout(K_TILES * 64, 1))
        asc_x1 = fx.logical_divide(aflat, mem_1x1)
        accflat = fx.make_view(fx.recast_iter(fx.Float32, lds_raw_ptr), fx.make_layout(BM * ACC_LDS_STRIDE, 1))
        acc_tiles = fx.logical_divide(accflat, mem_1x1)
        i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def issue_a_ds_read(slot):
            mask = _g1c._lds_swizzle_mask(lane_mod_16)
            a = [None, None]
            for k in range_constexpr(2):
                lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
                off = fx.Int32(slot * (BM * KH_TILE)) + lane_mod_16 * fx.Int32(KH_TILE) + lds_col
                r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
                fx.copy(i32x4_copy_atom, fx.slice(s_aq_x4, (None, off // fx.Int32(16))), r)
                a[k] = r
            return a

        def issue_a_scale_ds_read(kt):
            lds_dw = fx.Int32(kt * 64) + lane_div_16 * fx.Int32(16) + lane_mod_16
            return as_ir_value(_g1c._global_i32_load(asc_x1, lds_dw))

        lib = lane & fx.Int32(3)
        lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
        r_in_chunk = wave_n * fx.Int32(4) + lane_div_16
        aq_voff = row_tok * fx.Int32(K_HALF) + lane_shr2_and3 * fx.Int32(16) + lib * fx.Int32(4)
        as_voff = row_tok * fx.Int32(K_GROUPS)

        def a_load(kt):
            # this lane's 8 fp4 elements (4 B) of both 128-element halves of K-tile kt, plus the row's 8 group
            # scales of kt (two dwords: halves 0 and 1, group lane_shr2_and3 in byte lane_shr2_and3)
            out = []
            for half in range_constexpr(2):
                r = fx.make_rmem_tensor(mem_1x1, fx.Int32)
                s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * KH_TILE + half * 64))
                fx.copy(aq_copy_atom, fx.slice(aq_tiles, (None, aq_voff // fx.Int32(4))), r,
                        soffset=s_soff // fx.Int32(4))
                out.append(fx.Int32(r.load()[0]))
            rs = fx.make_rmem_tensor(mem_2x1, fx.Int32)
            s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * 8))
            fx.copy(as_copy_atom, fx.slice(as_tiles, (None, as_voff // fx.Int32(8))), rs,
                    soffset=s_soff // fx.Int32(4))
            sv = rs.load()
            out.append(fx.Int32(sv[0]))
            out.append(fx.Int32(sv[1]))
            return out

        def a_store(slot, kt, av):
            r = r_in_chunk
            mask_r = _g1c._lds_swizzle_mask(r)
            row_off = fx.Int32(slot * (BM * KH_TILE)) + r * fx.Int32(KH_TILE)
            for half in range_constexpr(2):
                kb_in_kt = fx.Int32(half * 4) + lane_shr2_and3
                off = row_off + ((kb_in_kt * fx.Int32(16)) ^ mask_r) + lib * fx.Int32(4)
                _g1c._scalar_store(s_aq_x1, off // fx.Int32(4), av[half], fx.Int32)
            sh = lane_shr2_and3 * fx.Int32(8)
            e0 = av[2].shrui(sh) & fx.Int32(0xFF)
            e1 = av[3].shrui(sh) & fx.Int32(0xFF)
            lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
            soff = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
            _g1c._scalar_store(asc_x1, soff // fx.Int32(4), e0 | (e1 << fx.Int32(16)), fx.Int32)

        def issue_b_load_j(b_slot, K_C, j):
            v = (lane_div_16 * fx.Int32(256)) + (lane_mod_16 * fx.Int32(16)) + fx.Int32(K_C * 2048)
            for half in range_constexpr(2):
                tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
                r = fx.make_rmem_tensor(mem_4x1, fx.Int32)
                fx.copy(bq_copy_atom, fx.slice(bq_tiles, (None, tile_idx)), r,
                        soffset=b_load_s_base[j] // fx.Int32(4))
                b_slot[j][half] = r

        def issue_b_scale_load(bs_slot, K_C):
            v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
            K_C_HI = K_C // 16
            imm = (K_C - K_C_HI * 16) * (_g1c.kBS_stride_k0_dw * 4)
            for mw in range_constexpr(2):
                s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
                idx = (v + fx.Int32(imm)) // fx.Int32(4)
                r = fx.make_rmem_tensor(mem_1x1, fx.Int32)
                fx.copy(bscale_copy_atom, fx.slice(bscale_tiles, (None, idx)), r, soffset=s_off // fx.Int32(4))
                bs_slot[mw] = r.load()[0]

        def _mma(ci, opsel_a, opsel_b, a_frag, b_frag, sa, sb):
            fx.gemm(scale_atoms[(opsel_a, opsel_b)], ci, a_frag, b_frag, ci, scale_a=sa, scale_b=sb)

        def mfma_j(b_slot, a_cur, asc_cur, bs_slot, J):
            bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
            sb = bs_slot[J % 2]
            if (wave_n & fx.Int32(1)) == fx.Int32(0):
                _mma(accm[J], 0, 0, a_cur[0], bJ0, asc_cur, sb)
                _mma(accm[J], 2, 2, a_cur[1], bJ1, asc_cur, sb)
            else:
                _mma(accm[J], 0, 1, a_cur[0], bJ0, asc_cur, sb)
                _mma(accm[J], 2, 3, a_cur[1], bJ1, asc_cur, sb)

        # ---- K loop (aiter run_k_slice structure: k_stages prologue, kUnroll steady steps, k_stages drain)
        apf = a_load(0)
        for K_C in range_constexpr(k_stages):
            cur = apf
            if const_expr(K_C + 1 < K_TILES):
                apf = a_load(K_C + 1)
            a_store(K_C, K_C, cur)
            issue_b_load_j(b[K_C], K_C, 0)
            issue_b_load_j(b[K_C], K_C, 1)
            issue_b_scale_load(b_scale_v[K_C], K_C)

        for OFFSET in range_constexpr(kUnroll):
            K_C = k_stages + OFFSET
            read_slot = OFFSET % kAStages
            write_slot = K_C % kAStages
            slot_b = OFFSET % k_stages
            cur = apf
            if const_expr(OFFSET + 1 < kUnroll):
                apf = a_load(K_C + 1)
            gpu.barrier()
            a_cur = issue_a_ds_read(read_slot)
            asc_cur = issue_a_scale_ds_read(K_C - k_stages)
            for J in range_constexpr(N_REPS):
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
                mfma_j(b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J)
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
                issue_b_load_j(b[slot_b], K_C, J)
                rocdl.sched_barrier(0)
            issue_b_scale_load(b_scale_v[slot_b], K_C)
            a_store(write_slot, K_C, cur)

        for S in range_constexpr(k_stages):
            kt = K_TILES - k_stages + S
            gpu.barrier()
            a_cur = issue_a_ds_read(kt % kAStages)
            asc_cur = issue_a_scale_ds_read(kt)
            for J in range_constexpr(N_REPS):
                mfma_j(b[kt % k_stages], a_cur, asc_cur, b_scale_v[kt % k_stages], J)

        gpu.barrier()

        # ---- epilogue (aiter store_acc_split / run_epilogue / store_scales, BM16 BN128 native scale)
        acc_layout = fx.make_layout((BM, BN), (ACC_LDS_STRIDE, 1))
        acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        def acc_store(idx, value):
            r = fx.make_rmem_tensor(mem_1x1, fx.Float32)
            r.store(fx.Vector.from_elements([fx.Float32(value)], fx.Float32))
            fx.copy(acc_copy_atom, r, fx.slice(acc_tiles, (None, idx)))

        def acc_load(idx):
            r = fx.make_rmem_tensor(mem_1x1, fx.Float32)
            fx.copy(acc_copy_atom, fx.slice(acc_tiles, (None, idx)), r)
            return r.load()[0]

        tx_i32 = wave_n * fx.Int32(64) + lane
        m_lane = tx_i32 // fx.Int32(16)
        n_lane = tx_i32 % fx.Int32(16)
        wave_grp = n_lane // fx.Int32(4)
        kk = n_lane % fx.Int32(4)

        row_base = lane_div_16 * fx.Int32(4)
        for J in range_constexpr(N_REPS):
            is_up = (J % 2) == 1
            col_local = wave_n * fx.Int32(BN // 8) + fx.Int32((J // 2) * 16) + lane_mod_16
            lds_col = fx.Int32(BN_INT) + col_local if is_up else col_local
            vec = fx.Vector(fx.memref_load_vec(accm[J]))
            for vv in range_constexpr(4):
                acc_store(_g1c._layout_idx(acc_layout, row_base + fx.Int32(vv), lds_col), vec[vv])
        gpu.barrier()

        aqout_layout = fx.make_layout((BM, OUT_ROW_BYTES), (OUT_ROW_BYTES, 1))
        aqout_tiles = _g1c._global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
        ascaleout_layout = fx.make_layout((i32_total_m_blocks, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1))
        ascaleout_i16_tiles = _g1c._global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
        if wave_grp < fx.Int32(2):
            gate_vs = [None] * 8
            up_vs = [None] * 8
            for ee in range_constexpr(8):
                col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
                gate_col = wave_grp * fx.Int32(32) + col_in_grp
                up_col = fx.Int32(BN_INT) + gate_col
                gate_vs[ee] = acc_load(_g1c._layout_idx(acc_layout, m_lane, gate_col))
                up_vs[ee] = acc_load(_g1c._layout_idx(acc_layout, m_lane, up_col))
            result = _g1c._activation_mul_batch(gate_vs, up_vs, act="silu", situ_beta=1.0,
                                                situ_linear_beta=1.0, swiglu_limit=7.0)
            local_max = _g1c._fabs_f32(result[0])
            for ee in range_constexpr(1, 8):
                local_max = local_max.maximumf(_g1c._fabs_f32(result[ee]))
            lm_i = _g1c._inline_dpp_quad_amax(local_max.bitcast(fx.Int32))
            local_max = lm_i.bitcast(fx.Float32)
            e8m0, qscale = _g1c._e8m0_from_amax(local_max, max_norm=6.0)
            qscale_raw = as_ir_value(qscale)
            packed_i32 = as_ir_value(fx.Int32(0))
            for w in range_constexpr(4):
                packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed_i32, as_ir_value(result[2 * w]),
                                                          as_ir_value(result[2 * w + 1]), qscale_raw, w)
            byte_pos = n_block_idx * fx.Int32(BN_INT // 2) + wave_grp * fx.Int32(16) + kk * fx.Int32(4)
            store_off = _g1c._layout_idx(aqout_layout, m_row + m_lane, byte_pos)
            _g1c._scalar_store(aqout_tiles, store_off // fx.Int32(4), fx.Int32(packed_i32), fx.Int32)
            if kk == fx.Int32(0):
                ku = n_block_idx >> fx.Int32(2)
                scale_wave_grp = (n_block_idx * fx.Int32(2) + wave_grp) & fx.Int32(3)
                ikxdl = (n_block_idx >> fx.Int32(1)) & fx.Int32(1)
                dword_off = _g1c._layout_idx(ascaleout_layout, m_block_idx, ku, scale_wave_grp, m_lane)
                dup = e8m0 | (e8m0 << fx.Int32(8))
                _g1c._scalar_store(ascaleout_i16_tiles, dword_off * fx.Int32(2) + ikxdl, dup, fx.Int16)

    @_functools.cache
    def _glm52_g1_launcher(variant, D_HIDDEN, D_INTER, NE, BN, xcd_swizzle):
        BM, BK = 16, 256
        KH_TILE = BK // 2
        N_OUT = 2 * D_INTER
        K_TILES_TOTAL = D_HIDDEN // BK
        NUM_N_BLOCKS = N_OUT // BN
        NXCD = 8
        epi_splits = _aiter_g1.default_epi_splits(BM, BN, 1, 4)
        k_stages = _aiter_g1.default_k_stages(BM, BN, KH_TILE, K_TILES_TOTAL, N_OUT, 1, epi_splits)
        _, _, _, lds_bytes = _aiter_g1._bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL, 1, epi_splits, k_stages)

        spread = variant in (2, 3)
        preq = variant == 3
        if spread:
            assert NUM_N_BLOCKS == NXCD
        block_threads = 256
        name = (f"gemm1_a4w4_port_fp4_h{D_HIDDEN}_i{D_INTER}_ne{NE}_bm16_iq_hpf_sep_native_scale_bn{BN}"
                f"_xcd{xcd_swizzle}_glm52v{variant}")

        @fx.struct
        class SharedStorage:
            raw: fx.Array[fx.Uint8, lds_bytes, 16]

        @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
        def gemm1_kernel(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_mind: fx.Int64,
            i32_ntok: fx.Int32,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_hidden: fx.Int64,
            arg_bias: fx.Int64,
        ):
            lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            tx_i32 = fx.Int32(tx)
            bx_i32 = fx.Int32(bx)
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
            total_m_blocks = cumsum0 // fx.Int32(BM)
            bound = total_m_blocks * fx.Int32(NUM_N_BLOCKS)
            xq = _udiv(bound, NXCD)
            xr = _umod(bound, NXCD)

            def _xcd(pid):
                # aiter's mapping (mxfp4_gemm1.py:1682-1694)
                xc = _umod(pid, NXCD)
                wgid = xc * xq + (xc < xr).select(xc, xr) + _udiv(pid, NXCD)
                ng = fx.Int32(xcd_swizzle * NUM_N_BLOCKS)
                group_id = wgid // ng
                first_pid_m = group_id * fx.Int32(xcd_swizzle)
                remaining_m = total_m_blocks - first_pid_m
                sw = fx.Int32(xcd_swizzle)
                group_size_m = (remaining_m < sw).select(remaining_m, sw)
                wig = wgid % ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(NUM_N_BLOCKS) + n_block

            def _shared_m_blocks():
                # trailing m-blocks of the last expert in sorted order, at most ceil(M / BM) of them
                s0 = (i32_ntok + fx.Int32(BM - 1)) // fx.Int32(BM)
                s0 = (s0 < total_m_blocks).select(s0, total_m_blocks)
                e_first = _global_i32_at(arg_eids, total_m_blocks - s0)
                e_last = _global_i32_at(arg_eids, total_m_blocks - fx.Int32(1))
                s = (e_first == e_last).select(s0, fx.Int32(0))
                return fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(s)))

            def _spread(pid, n_sh):
                # XCD xc = pid % 8 first runs the n_sh shared m-blocks of n-block xc, then an equal contiguous
                # share (n_rt tiles) of the routed tiles grouped as in aiter's xcd_swizzle mapping.
                xc = _umod(pid, NXCD)
                li = _udiv(pid, NXCD)
                n_rt = total_m_blocks - n_sh
                is_sh = li < n_sh
                r = is_sh.select(fx.Int32(0), li - n_sh)
                wgid = xc * n_rt + r
                ng = fx.Int32(xcd_swizzle * NUM_N_BLOCKS)
                group_id = wgid // ng
                first_pid_m = group_id * fx.Int32(xcd_swizzle)
                remaining_m = n_rt - first_pid_m
                sw = fx.Int32(xcd_swizzle)
                gsm = (remaining_m < sw).select(remaining_m, sw)
                gsm = (gsm < fx.Int32(1)).select(fx.Int32(1), gsm)
                wig = wgid % ng
                m_rt = first_pid_m + (wig % gsm)
                n_rt_blk = wig // gsm
                m_block = is_sh.select(n_rt + li, m_rt)
                n_block = is_sh.select(xc, n_rt_blk)
                return m_block * fx.Int32(NUM_N_BLOCKS) + n_block

            def _body(tile, nt):
                _gemm1_body(
                    lds_raw_ptr, arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_mind,
                    arg_aqout, arg_ascaleout, arg_hidden, arg_bias, tile, lane, wave, nt,
                    i32_ntok, total_m_blocks,
                    BM=BM, BN=BN, BK=BK, inline_quant=True, prefetch_hidden=True, a_dtype="fp4",
                    out_dtype="fp4", act="silu", situ_beta=1.0, situ_linear_beta=1.0,
                    swiglu_limit=7.0, enable_bias=False, K=D_HIDDEN, N_OUT=N_OUT, NE=NE,
                    interleave=False, native_scale_layout=True, num_waves=4, k_wave=1,
                    epi_splits=epi_splits, k_stages=k_stages,
                )

            def _pbody(tile, b_nt):
                # arg_aq / arg_ascale carry the prologue's token-major MXFP4 activations and e8m0 scales
                _g1_body_preq(
                    lds_raw_ptr, arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_mind, arg_aqout,
                    arg_ascaleout, tile, lane, wave, i32_ntok, total_m_blocks,
                    b_nt=b_nt, K=D_HIDDEN, N_OUT=N_OUT, NE=NE, k_stages=k_stages,
                )

            if bx_i32 < bound:
                if const_expr(spread):
                    n_sh = _shared_m_blocks()
                    tile = _spread(bx_i32, n_sh)
                    # workgroup-uniform (SGPR) condition: both bodies contain barriers
                    m_blk = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(tile // fx.Int32(NUM_N_BLOCKS))))
                    sh_lo = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(total_m_blocks - n_sh)))
                    if m_blk >= sh_lo:
                        if const_expr(preq):
                            _pbody(tile, False)  # shared expert: temporal B loads
                        else:
                            _body(tile, False)
                    else:
                        if const_expr(preq):
                            _pbody(tile, True)
                        else:
                            _body(tile, True)
                else:
                    _body(_xcd(bx_i32), True)


        @flyc.jit
        def launch_gemm1(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_mind: fx.Int64,
            i32_ntok: fx.Int32,
            i32_grid: fx.Int32,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_hidden: fx.Int64,
            arg_bias: fx.Int64,
            stream: fx.Stream,
        ):
            grid_x = fx.Int64(i32_grid)
            gemm1_kernel(
                arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_mind, i32_ntok,
                arg_aqout, arg_ascaleout, arg_hidden, arg_bias,
            ).launch(grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream)

        return launch_gemm1


_g1_ready: set = set()      # launcher keys already launched once outside a graph capture


def _g1_engages(kw):
    """True for exactly the stage-1 instance this file's decode config selects (BM16 inline quant)."""
    try:
        return (_G1_IMPORT_OK and _G1_VARIANT in (1, 2, 3)
                and kw.get("BM") == 16 and kw.get("use_nt") is True and kw.get("inline_quant") is True
                and kw.get("prefetch_hidden") is True and kw.get("BN") == 128 and kw.get("BK", 256) == 256
                and not kw.get("interleave", False) and kw.get("xcd_swizzle", 0) == 2
                and kw.get("native_scale_layout") is True and kw.get("a_dtype", "fp4") == "fp4"
                and kw.get("out_dtype", "fp4") == "fp4" and kw.get("act", "silu") == "silu"
                and kw.get("bias") is None and kw.get("num_waves", 4) == 4 and kw.get("k_wave", 1) == 1
                and kw.get("stream") is None
                and kw["D_HIDDEN"] % 256 == 0 and 2 <= kw["D_HIDDEN"] // 256 <= 32
                and 2 * kw["D_INTER"] == 128 * 8)
    except Exception:  # noqa: BLE001
        return False


def _install_g1_hook():
    if not _G1_IMPORT_OK:
        return
    f = getattr(_aiter_g1k, "flydsl_mxfp4_gemm1", None)
    if f is None or getattr(f, "_glm52_wrapped", False):
        return

    def flydsl_mxfp4_gemm1(_orig=f, **kw):
        if not (_ACTIVE and _g1_engages(kw)):
            return _orig(**kw)
        variant = _G1_VARIANT
        a_ptr, as_ptr = kw["a_quant"].data_ptr(), kw["a_scale_sorted_shuffled"].data_ptr()
        if variant == 3:
            pq = _PREQ
            if (pq is not None and pq[0].data_ptr() == kw["hidden_states"].data_ptr()
                    and pq[1] == int(kw["n_tokens"]) and pq[0].shape[1] == int(kw["D_HIDDEN"])):
                a_ptr, as_ptr = pq[2].data_ptr(), pq[3].data_ptr()
            else:
                variant = 2          # no pre-quantized activations from this call's prologue
        key = (variant, int(kw["D_HIDDEN"]), int(kw["D_INTER"]), int(kw["NE"]), 128, 2)
        if key not in _g1_ready and torch.cuda.is_current_stream_capturing():
            return _orig(**kw)       # first launch inside a capture: keep aiter's (already built) kernel
        launch = _glm52_g1_launcher(*key)
        grid = _aiter_g1.gemm1_grid(kw["n_tokens"], 16, NE=kw["NE"], TOPK=kw["topk"], INTER=kw["D_INTER"],
                                    BN=128)
        _aiter_moe_kernels._run_compiled(
            launch,
            (
                a_ptr,
                as_ptr,
                kw["w1_u8"].data_ptr(),
                kw["w1_scale_u8"].data_ptr(),
                kw["sorted_expert_ids"].data_ptr(),
                kw["cumsum_tensor"].data_ptr(),
                kw["m_indices"].data_ptr(),
                kw["n_tokens"],
                grid,
                kw["inter_sorted_quant"].data_ptr(),
                kw["inter_sorted_shuffled_scale"].data_ptr(),
                kw["hidden_states"].data_ptr(),
                0,
                torch.cuda.current_stream(),
            ),
        )
        _g1_ready.add(key)

    flydsl_mxfp4_gemm1._glm52_wrapped = True
    flydsl_mxfp4_gemm1._glm52_orig = f
    _aiter_g1k.flydsl_mxfp4_gemm1 = flydsl_mxfp4_gemm1


# ---------------------------------------------------------------------------------------------------------
# gemm2 instance for the decode row: BK 256 instead of the tuned row's BK 128.
#
# The block_m=16 decode row (see _DECODE_KEY_M) pairs gemm1 with flydsl_moe2_layout_..._t16x128x128_atomic_nt_sbm16.
# K of gemm2 is only I = 512, so a BK-128 tile runs four short K steps per 16x128 output tile; the t16x128x256
# instance of the same aiter kernel (same sorting layout, same atomic bf16 epilogue, same SPART/BHOIST knobs) runs
# two.  Measured on MI355X (n02-25), driver --bench-mode, 5 interleaved reps on top of the gemm1 change below:
# mean case speedup 1.0918 -> 1.0981 (every case faster, -0.4 .. -1.8 us); n03-33 (iter 12): 1.0909 -> 1.0978.
# Tried with it, not better: t16x256x128, t16x256x256 (equal), t16x512x256 (-1.5 %), persist (-0.3 %), SPART
# 201/402/801 (equal or worse), the non-atomic "reduce" epilogue (gemm2 80.3 us + 5.6 us reduction vs 77.0 us
# atomic, eager), BN 64 (no such instance).
# Scoped to this file's op and to exactly the decode row's stage-2 name; GLM52_MOE_G2NAME overrides the
# replacement (empty string = keep aiter's instance).
_G2_DECODE_FROM = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16"

_G2_NAME = os.environ.get("GLM52_MOE_G2NAME", "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16")


def _install_g2_name_override():
    import dataclasses
    import functools
    f = getattr(_aiter_fused_moe, "get_2stage_cfgs", None)
    if f is None or getattr(f, "_glm52_wrapped", False):
        return
    cache = {}

    def get_2stage_cfgs(*args, _orig=f, **kwargs):
        md = _orig(*args, **kwargs)
        if not (_ACTIVE and _G2_NAME) or md is None or not isinstance(getattr(md, "stage2", None),
                                                                        functools.partial):
            return md
        if md.stage2.keywords.get("kernelName2") != _G2_DECODE_FROM:
            return md
        key = (id(md), _G2_NAME)
        if key not in cache:
            kw = dict(md.stage2.keywords)
            kw["kernelName2"] = _G2_NAME
            cache[key] = (md, dataclasses.replace(md, stage2=functools.partial(md.stage2.func, *md.stage2.args, **kw)))
        return cache[key][1]

    get_2stage_cfgs._glm52_wrapped = True
    get_2stage_cfgs._glm52_orig = f
    _aiter_fused_moe.get_2stage_cfgs = get_2stage_cfgs


_ACTIVE = False
_CUR_HIDDEN = None      # hidden_states of the op call in flight (read by the HIP prologue for the pre-quant)
_PREQ = None            # (hidden, M, a_q, a_s) written by this call's prologue, consumed by the gemm1 hook
_install_decode_cfg_key()
_install_fused_prologue_sort()
_install_hip_prologue()
_install_g1_hook()
_install_g2_name_override()


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
    global _ACTIVE, _CUR_HIDDEN, _PREQ
    _ACTIVE = True
    _CUR_HIDDEN = hidden_states
    _PREQ = None            # only this call's prologue may provide the pre-quantized activations
    try:
        return _fused_moe_call(hidden_states, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids)
    finally:
        _ACTIVE = False
        _CUR_HIDDEN = None


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
