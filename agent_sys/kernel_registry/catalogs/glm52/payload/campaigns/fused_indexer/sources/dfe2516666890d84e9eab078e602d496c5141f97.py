"""GLM-5.2 gfx950 FUSED decode DSA indexer (rows <= 48), as sglang 0.5.21 / 0.5.22 runs it on MI355X.

ONE public entry point: ``fused_decode_indexer`` = Gfx950FusedIndexer.run (sglang 93034eba91
srt/layers/attention/dsa/gfx950_fused_indexer.py:210; identical kernels in v0.5.21). Indexer.forward_cuda serves
every DECODE / TARGET_VERIFY call with 1 <= rows <= 48 (MAX_ROWS = bs 8 x 6 draft tokens) through these four kernels
instead of the standard indexer, i.e. it is THE indexer for per-rank decode batches of 1..8 requests:
  node 1 dual_gemv_bf16      q_proj = q_lora @ wq_b.T [rows, 4096]   and   kw = x @ [wk ; weights_proj].T [rows, 160]
  node 2 qk_rope_hadamard_quant_and_cache: k LayerNorm | rope | Hadamard(q, k) | fp8 act-quant(q) | index-K quant +
         write of this step's K into the paged cache | head gate
  node 3 logits_hist         paged fp8 MQA logits over the row's context + per-row 12-bit histogram
  node 4 topk_transform      top-2048 + page_size 64 -> slot transform (histogram-guided), int32 slots, -1 padded
The HIP sources below are verbatim copies of sglang kernels/jit/csrc/dsa_gfx950/*.cuh (identical in v0.5.21 and
93034eba91); they are JIT-built with sglang's load_jit from a content-addressed temp dir, with the production flags.
Edit them (or replace any node) freely; keep the entry point's contract.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from functools import cache

import torch

HEAD_DIM = 128
ROPE_DIM = 64
N_HEADS = 32
INDEX_TOPK = 2048
PAGE_SIZE = 64
CACHE_TOK_STRIDE = 132
Q_LORA_RANK = 2048
HIDDEN_SIZE = 6144
MAX_ROWS = 48
LOGITS_HIST_BITS = 12
LOGITS_BLOCKS_PER_ROW = 32


def logits_bpr(rows: int, table_pages: int) -> int:
    """Blocks per row of the logits kernel: 32 (measured best for rows 12..48 at these contexts), raised so that a
    wave never walks more than 64 pages (its page ids live one per lane), and kept a multiple of 8 so that block x
    of every row lands on XCD x % 8 (sibling rows then read each page through the same L2)."""
    need = -(-table_pages // (8 * 64))
    return max(LOGITS_BLOCKS_PER_ROW, -(-need // 8) * 8)
# n4 (topk_transform) blocks per row.  The row-barrier + cursor-atomic overhead
# of each block outweighs the extra fill past ~48 here (measured: 48 beats the
# previous 64 on every case, most on T36), while G only repartitions the exact
# selection so the top-k set stays identical.
TOPK_G = 48

SRC_DUAL_GEMV_BF16 = r"""/// Dual bf16 GEMV for the gfx950 DSA indexer: Q [M,4096] and KW [M,160] in one launch.

#pragma once

#ifndef USE_ROCM
#error "dual_gemv_bf16.cuh targets CDNA3/4; it uses MFMA and wave64 v_dot2c"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

// Local bf16_t is the builtin __bf16 that ext_vector_type needs, not sglang::bf16_t.
namespace dsa_gfx950::dual_gemv {

using namespace ::sglang::host;

typedef unsigned short ushort_t;
typedef __bf16 bf16_t;
typedef bf16_t bf16x2_t __attribute__((ext_vector_type(2)));
typedef bf16_t bf16x8_t __attribute__((ext_vector_type(8)));
typedef float float4_t __attribute__((ext_vector_type(4)));

constexpr int kThreads = 256;
constexpr int kKq = 2048;
constexpr int kKk = 6144;
constexpr int kMaxRows = 48;  // fused_decode.MAX_ROWS
constexpr int kQCols = 16;
constexpr int kQSteps = kKq / 128;
constexpr int kQStageRows = 16;
constexpr int kQStageLd = kThreads + 1;  // 16-byte pad per row against LDS bank conflicts
constexpr int kKwVecs = kKk / (8 * kThreads);

__device__ __forceinline__ float dot2_bf16(unsigned int x, unsigned int w, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, x), __builtin_bit_cast(bf16x2_t, w), acc, false);
}

// round-to-nearest-even, matching torch's fp32 -> bf16 cast
__device__ __forceinline__ ushort_t f32_to_bf16_rne(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN -> quiet NaN
    return static_cast<ushort_t>((u >> 16) | 0x0040u);
  }
  unsigned int lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<ushort_t>(u >> 16);
}

// K offset of MFMA step s, in hipBLASLt MT16x16x512's reduction order so Q matches it bit for bit.
__device__ __forceinline__ int q_step_k(int s) {
  return (s >> 2) * 512 + (s & 3) * 32;
}

// 16 Q columns for all M rows; the weights load once and the first 16 x rows go through LDS.
__device__ __forceinline__ void gemv_q(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    ushort_t* __restrict__ O,
    long ldo,
    int M,
    int col0,
    float* __restrict__ smem) {
  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int frag = lane & 15;
  const int kofs = wave * 128 + (lane >> 4) * 8;
  const bf16_t* wp = w + (long)(col0 + frag) * kKq + kofs;
  const bf16_t* xp = x + (long)frag * kKq + kofs;
  bf16x8_t a[kQSteps];
  bf16x8_t b[kQSteps];
  {
    bf16x8_t xv[kQStageRows];
#pragma unroll
    for (int r = 0; r < kQStageRows; ++r)
      if (r < M) xv[r] = *reinterpret_cast<const bf16x8_t*>(x + (long)r * kKq + threadIdx.x * 8);
#pragma unroll
    for (int s = 0; s < kQSteps; ++s)
      b[s] = *reinterpret_cast<const bf16x8_t*>(wp + q_step_k(s));
    bf16x8_t* xs = reinterpret_cast<bf16x8_t*>(smem);
#pragma unroll
    for (int r = 0; r < kQStageRows; ++r)
      if (r < M) xs[r * kQStageLd + threadIdx.x] = xv[r];
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kQSteps; ++s)
      a[s] = frag < M ? xs[frag * kQStageLd + (kofs + q_step_k(s)) / 8] : bf16x8_t{};
    __syncthreads();
  }
  const int groups = (M + 15) >> 4;
  for (int g = 0; g < groups; ++g) {
    const bool more = g + 1 < groups;
    const bool next_valid = 16 * (g + 1) + frag < M;
    const bf16_t* xn = xp + (long)16 * (g + 1) * kKq;
    float4_t acc = {};
#pragma unroll
    for (int s = 0; s < kQSteps; ++s) {
      acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a[s], b[s], acc, 0, 0, 0);
      if (more) {
        a[s] = next_valid ? *reinterpret_cast<const bf16x8_t*>(xn + q_step_k(s)) : bf16x8_t{};
      }
    }
    // [group][wave][column][row], so the wave-order sum below reads stride 256.
    *reinterpret_cast<float4_t*>(smem + ((g * 4 + wave) * 16 + frag) * 16 + (lane >> 4) * 4) = acc;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < M * kQCols; i += kThreads) {
    const int r = i >> 4;
    const int c = i & 15;
    const float* p = smem + ((r >> 4) * 64 + c) * 16 + (r & 15);
    float sum = p[0] + p[256];
    sum += p[512];
    sum += p[768];
    O[(long)r * ldo + col0 + c] = f32_to_bf16_rne(sum);
  }
}

// NCOL KW columns for R rows from row0.
template <int R, int NCOL>
__device__ __forceinline__ void gemv_kw(
    const uint4* __restrict__ Xv,
    long ldxv,
    const uint4* __restrict__ Wv,
    long ldwv,
    ushort_t* __restrict__ O,
    long ldo,
    int M,
    int row0,
    int col0,
    int N,
    float* __restrict__ smem) {
  const int tid = threadIdx.x;
  unsigned int xr[R][kKwVecs][4];
#pragma unroll
  for (int m = 0; m < R; ++m) {
#pragma unroll
    for (int p = 0; p < kKwVecs; ++p) {
      uint4 t = {0u, 0u, 0u, 0u};
      if (row0 + m < M) t = Xv[(long)(row0 + m) * ldxv + tid + p * kThreads];
      xr[m][p][0] = t.x;
      xr[m][p][1] = t.y;
      xr[m][p][2] = t.z;
      xr[m][p][3] = t.w;
    }
  }
  uint4 wr[NCOL][kKwVecs];
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
    const int col = col0 + c;
    // clamp instead of branch: out-of-range columns are computed and dropped
    const uint4* wp = Wv + (long)(col < N ? col : 0) * ldwv + tid;
#pragma unroll
    for (int p = 0; p < kKwVecs; ++p) {
      wr[c][p] = wp[p * kThreads];
    }
  }
  float acc[NCOL][R] = {};
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
#pragma unroll
    for (int p = 0; p < kKwVecs; ++p) {
      unsigned int w4[4] = {wr[c][p].x, wr[c][p].y, wr[c][p].z, wr[c][p].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int m = 0; m < R; ++m)
          acc[c][m] = dot2_bf16(xr[m][p][e], w4[e], acc[c][m]);
      }
    }
  }
  const int wave = tid >> 6;
  const int wlane = tid & 63;
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) {
#pragma unroll
    for (int c = 0; c < NCOL; ++c)
#pragma unroll
      for (int m = 0; m < R; ++m)
        acc[c][m] += __shfl_down(acc[c][m], off, 64);
  }
  if (wlane == 0) {
#pragma unroll
    for (int c = 0; c < NCOL; ++c)
#pragma unroll
      for (int m = 0; m < R; ++m)
        smem[(wave * NCOL + c) * R + m] = acc[c][m];
  }
  __syncthreads();
  if (tid < NCOL * R) {
    const int c = tid / R;
    const int m = tid - c * R;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w)
      s += smem[(w * NCOL + c) * R + m];
    const int col = col0 + c;
    if (col < N && row0 + m < M) O[(long)(row0 + m) * ldo + col] = f32_to_bf16_rne(s);
  }
}

// The KW blocks lead the grid so their loads queue ahead of the Q weight stream.
template <int R, int NCOL>
__global__ __launch_bounds__(kThreads) void dual_gemv_kernel(
    const bf16_t* __restrict__ Xq,
    const bf16_t* __restrict__ Wq,
    ushort_t* __restrict__ Oq,
    long ldoq,
    int nblk_q,
    const uint4* __restrict__ Xk,
    long ldxk,
    const uint4* __restrict__ Wk,
    long ldwk,
    ushort_t* __restrict__ Ok,
    long ldok,
    int Nk,
    int nblk_kc,
    int M) {
  constexpr int kSmemQ = kQStageRows * kQStageLd * 4;
  static_assert(kSmemQ >= (kMaxRows / 16) * 4 * 16 * 16 && kSmemQ >= 4 * NCOL * R);
  __shared__ __attribute__((aligned(16))) float smem[kSmemQ];
  const int nblk_k = gridDim.x - nblk_q;
  int bid = blockIdx.x;
  bid = bid < nblk_k ? bid + nblk_q : bid - nblk_k;
  if (bid < nblk_q) {
    gemv_q(Xq, Wq, Oq, ldoq, M, bid * kQCols, smem);
  } else {
    const int b = bid - nblk_q;
    const int rc = b / nblk_kc;
    const int cb = b - rc * nblk_kc;
    gemv_kw<R, NCOL>(Xk, ldxk, Wk, ldwk, Ok, ldok, M, rc * R, cb * NCOL, Nk, smem);
  }
}

struct Args {
  const bf16_t* Xq;
  const bf16_t* Wq;
  ushort_t* Oq;
  long ldoq;
  int Nq;
  const uint4* Xk;
  long ldxk;
  const uint4* Wk;
  long ldwk;
  ushort_t* Ok;
  long ldok;
  int Nk;
  int M;
  hipStream_t stream;
};

template <int R, int NCOL>
void launch(const Args& a) {
  const int nblk_q = a.Nq / kQCols;
  const int nblk_kc = (a.Nk + NCOL - 1) / NCOL;
  const int nblk_kr = (a.M + R - 1) / R;
  dual_gemv_kernel<R, NCOL><<<dim3(nblk_q + nblk_kc * nblk_kr), dim3(kThreads), 0, a.stream>>>(
      a.Xq, a.Wq, a.Oq, a.ldoq, nblk_q, a.Xk, a.ldxk, a.Wk, a.ldwk, a.Ok, a.ldok, a.Nk, nblk_kc, a.M);
}

// KW block shape (rows x columns) per M range.
inline void dispatch(const Args& a) {
  if (a.M == 1) {
    launch<1, 1>(a);
  } else if (a.M == 2) {
    launch<2, 1>(a);
  } else if (a.M <= 4) {
    launch<4, 1>(a);
  } else if (a.M <= 6) {
    launch<2, 2>(a);
  } else if (a.M <= 8) {
    launch<4, 2>(a);
  } else if (a.M <= 24) {
    launch<4, 4>(a);
  } else {
    launch<8, 4>(a);
  }
}
}  // namespace dsa_gfx950::dual_gemv

struct DualGemvBf16Kernel {
  /// Oq = Xq @ Wq^T and Ok = Xk @ Wk^T, all bf16; outputs are overwritten.
  static void
  run(const tvm::ffi::TensorView Xq,
      const tvm::ffi::TensorView Wq,
      const tvm::ffi::TensorView Oq,
      const tvm::ffi::TensorView Xk,
      const tvm::ffi::TensorView Wk,
      const tvm::ffi::TensorView Ok) {
    using namespace host;
    namespace impl = dsa_gfx950::dual_gemv;

    auto M_ = SymbolicSize{"num_rows"};
    auto Kq_ = SymbolicSize{"k_q"};
    auto Nq_ = SymbolicSize{"n_q"};
    auto Kk_ = SymbolicSize{"k_k"};
    auto Nk_ = SymbolicSize{"n_k"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // Q reads x at row * K, so a padded Xq or Wq would silently read wrong rows.
    TensorMatcher({M_, Kq_}).with_dtype<bf16_t>().with_device(device_).with_strides({Kq_, 1}).verify(Xq);
    TensorMatcher({Nq_, Kq_}).with_dtype<bf16_t>().with_device(device_).with_strides({Kq_, 1}).verify(Wq);
    TensorMatcher({M_, Nq_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Oq);
    TensorMatcher({M_, Kk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Xk);
    TensorMatcher({Nk_, Kk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Wk);
    TensorMatcher({M_, Nk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Ok);

    const auto M = static_cast<int>(M_.unwrap());
    RuntimeCheck(M >= 1 && M <= impl::kMaxRows, "dual_gemv supports M in [1,", impl::kMaxRows, "], got ", M);
    const auto Kq = static_cast<int>(Kq_.unwrap());
    const auto Kk = static_cast<int>(Kk_.unwrap());
    RuntimeCheck(
        Kk == impl::kKk && Kq == impl::kKq, "dual_gemv is built for Kk=6144, Kq=2048; got Kk=", Kk, " Kq=", Kq);

    impl::Args a;
    a.Xq = reinterpret_cast<const impl::bf16_t*>(Xq.data_ptr());
    a.Wq = reinterpret_cast<const impl::bf16_t*>(Wq.data_ptr());
    a.Oq = reinterpret_cast<impl::ushort_t*>(Oq.data_ptr());
    a.Nq = static_cast<int>(Nq_.unwrap());
    a.ldoq = Oq.stride(0);
    a.Xk = reinterpret_cast<const uint4*>(Xk.data_ptr());
    a.Wk = reinterpret_cast<const uint4*>(Wk.data_ptr());
    a.Ok = reinterpret_cast<impl::ushort_t*>(Ok.data_ptr());
    a.Nk = static_cast<int>(Nk_.unwrap());
    a.ldxk = Xk.stride(0) / 8;
    a.ldwk = Wk.stride(0) / 8;
    a.ldok = Ok.stride(0);
    a.M = M;
    a.stream = LaunchKernel::resolve_device(device_.unwrap());

    RuntimeCheck(a.Nq % impl::kQCols == 0, "dual_gemv needs n_q % 16 == 0, got ", a.Nq);
    RuntimeCheck(Xk.stride(0) % 8 == 0 && Wk.stride(0) % 8 == 0, "the K operands must start on an 8-element boundary");

    impl::dispatch(a);
    RuntimeDeviceCheck();
  }
};

}  // namespace sglang
"""

SRC_QK_ROPE_HADAMARD_QUANT = r"""/// A copy of aiter's indexer_qk_rope_quant_and_cache_kernel with an inline
/// 128-point Hadamard, so the front end keeps rotate_activation and the fusion
/// at once.  aiter is untouched; this includes its headers (JIT dependency
/// "aiter"), so the maths match.

#pragma once

#ifndef USE_ROCM
#error "qk_rope_hadamard_quant.cuh targets gfx950; it folds a head onto one wave64"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "hip_reduce.h"
#include "opus/opus.hpp"
#include <algorithm>
#include <cstdint>

namespace sglang {

namespace dsa_gfx950::qk_rope {

using cache_t = opus::fp8_t;
using scalar_t = opus::bf16_t;

constexpr int HEAD_DIM = 128;
constexpr int ROPE_DIM = 64;
constexpr int LOG2_HEAD_DIM = 7;

// Folded onto one 64-lane wavefront: lane t owns dim t and t+64, so both halves
// of every reduction are an independent DPP tree.  Zero LDS, zero syncthreads,
// and bit-exact against the 128-thread form operand for operand.

constexpr int HALF_DIM = HEAD_DIM / 2;  // 64 == one wavefront

template <typename F>
__device__ __forceinline__ float wred64(float v, F op) {
  return wave_reduce<float, F, 64, true>(v, op);
}

// Interleaved rope on the low 64 dims, held one-per-lane in `v`.  The pair
// comes from a cross-lane shuffle instead of LDS; the arithmetic is the same.
__device__ __forceinline__ float
rope_lo(float v, int t, const scalar_t* __restrict__ cos_ptr, const scalar_t* __restrict__ sin_ptr) {
  const float pair_val = __shfl_xor(v, 1, 64);
  const int cos_idx = t / 2;
  const float cos_v = static_cast<float>(cos_ptr[cos_idx]);
  const float sin_v = static_cast<float>(sin_ptr[cos_idx]);
  v = (t % 2 == 0) ? (v * cos_v - pair_val * sin_v) : (v * cos_v + pair_val * sin_v);
  return static_cast<float>(static_cast<scalar_t>(v));
}

// 128-point Hadamard on a pair (lo = dim t, hi = dim t+64) held in one lane.
__device__ __forceinline__ void hadamard128_wave(float& lo, float& hi, int t, float scale) {
#pragma unroll
  for (int s = 0; s < LOG2_HEAD_DIM - 1; ++s) {  // strides 1, 2, 4, 8, 16, 32
    const int d = 1 << s;
    const float plo = __shfl_xor(lo, d, 64);
    const float phi = __shfl_xor(hi, d, 64);
    const bool a = ((t & d) == 0);  // (dim & d) is the same for dim and dim+64
    lo = a ? (lo + plo) : (plo - lo);
    hi = a ? (hi + phi) : (phi - hi);
  }
  // stride 64: partner of dim t is dim t+64 -- both already in this lane.
  const float nlo = lo + hi;
  const float nhi = lo - hi;  // (dim & 64) != 0 -> partner - v
  lo = nlo * scale;
  hi = nhi * scale;
}

// The k-side epilogue runs in its own block rather than at the end of the
// head_idx==0 blocks, where it would serialise behind them.
__global__ __launch_bounds__(HALF_DIM) void qk_rope_hadamard_quant_kernel(
    const scalar_t* __restrict__ q,
    cache_t* __restrict__ q_out,
    const scalar_t* __restrict__ weights,
    float* __restrict__ weights_out,
    const scalar_t* __restrict__ k,
    cache_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    const int n_heads,
    const int cache_block_size,
    const int cache_stride,
    const int q_stride_t,
    const int q_stride_h,
    const int q_out_stride_t,
    const int q_out_stride_h,
    const int weights_stride_t,
    const int weights_out_stride_t,
    const int k_stride_t,
    const int cos_stride0,
    const int sin_stride0,
    const float epsilon,
    const float weights_scale,
    const float hadamard_scale,
    const int max_position) {
  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;  // n_heads + 1 of them: the last is the k side
  const int t = threadIdx.x;        // owns dim t and dim t + 64
  const bool do_q = head_idx < n_heads;

  const int64_t slot_idx = slot_mapping[token_idx];
  // Clamped unconditionally: pos indexes cos_cache/sin_cache on both branches,
  // and clamping only the slot_idx < 0 one left the common path free to read at
  // an arbitrary offset.
  int64_t pos = positions[token_idx];
  pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
  const scalar_t* cos_ptr = cos_cache + pos * cos_stride0;
  const scalar_t* sin_ptr = sin_cache + pos * sin_stride0;

  auto max_func = [](float a, float b) { return fmaxf(a, b); };
  auto sum_func = [](float a, float b) { return a + b; };
  const float q_fp8_max = static_cast<float>(opus::finfo<cache_t>::max());

  // ------------------------------- q side ---------------------------------
  if (do_q) {
    const scalar_t* q_row = q + token_idx * q_stride_t + head_idx * q_stride_h;
    float lo = static_cast<float>(q_row[t]);
    float hi = static_cast<float>(q_row[t + HALF_DIM]);

    lo = rope_lo(lo, t, cos_ptr, sin_ptr);  // dims >=64 untouched

    hadamard128_wave(lo, hi, t, hadamard_scale);
    lo = static_cast<float>(static_cast<scalar_t>(lo));
    hi = static_cast<float>(static_cast<scalar_t>(hi));

    const float a0 = wred64(fabsf(lo), max_func);
    const float a1 = wred64(fabsf(hi), max_func);
    const float q_amax = max_func(a1, a0);  // block_reduce's order

    const float q_inv_fp8_max = 1.0f / q_fp8_max;
    // ue8m0: the scale is rounded up to a power of two.
    const float q_scale = exp2f(ceilf(log2f(fmaxf(q_amax, 1e-10f) * q_inv_fp8_max)));
    const float q_inv_scale = 1.0f / q_scale;
    cache_t* qo = q_out + token_idx * q_out_stride_t + head_idx * q_out_stride_h;
    qo[t] = opus::cast<cache_t>(lo * q_inv_scale);
    qo[t + HALF_DIM] = opus::cast<cache_t>(hi * q_inv_scale);
    if (t == 0) {
      const float w = static_cast<float>(weights[token_idx * weights_stride_t + head_idx]);
      const float head_scale = rsqrtf(static_cast<float>(n_heads));
      const scalar_t w_head = static_cast<scalar_t>(w * head_scale);
      const float softmax_scale = weights_scale / head_scale;
      weights_out[token_idx * weights_out_stride_t + head_idx] = static_cast<float>(w_head) * q_scale * softmax_scale;
    }
  }

  if (do_q || slot_idx < 0) return;

  // ------------------------------- k side ---------------------------------
  const scalar_t* k_row = k + token_idx * k_stride_t;
  const float xlo = static_cast<float>(k_row[t]);
  const float xhi = static_cast<float>(k_row[t + HALF_DIM]);

  const float s0 = wred64(xlo, sum_func);
  const float s1 = wred64(xhi, sum_func);
  const float mean = sum_func(s1, s0) / static_cast<float>(HEAD_DIM);

  const float clo = xlo - mean;
  const float chi = xhi - mean;
  const float q0 = wred64(clo * clo, sum_func);
  const float q1 = wred64(chi * chi, sum_func);
  const float ss = sum_func(q1, q0);
  const float inv_std = rsqrtf(ss / static_cast<float>(HEAD_DIM) + epsilon);

  float klo = clo * inv_std * norm_weight[t] + norm_bias[t];
  float khi = chi * inv_std * norm_weight[t + HALF_DIM] + norm_bias[t + HALF_DIM];
  klo = static_cast<float>(static_cast<scalar_t>(klo));
  khi = static_cast<float>(static_cast<scalar_t>(khi));

  klo = rope_lo(klo, t, cos_ptr, sin_ptr);

  hadamard128_wave(klo, khi, t, hadamard_scale);
  klo = static_cast<float>(static_cast<scalar_t>(klo));
  khi = static_cast<float>(static_cast<scalar_t>(khi));

  const float m0 = wred64(fabsf(klo), max_func);
  const float m1 = wred64(fabsf(khi), max_func);
  const float k_amax = max_func(m1, m0);
  const float k_scale = exp2f(ceilf(log2f(fmaxf(k_amax, 1e-4f) / q_fp8_max)));  // ue8m0

  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;
  const int64_t page_base = block_idx * cache_block_size * cache_stride;
  if (t == 0) {
    // One fp32 scale per token, after the page's K bytes: quant block == HEAD_DIM.
    const int64_t dst_scale_idx = page_base + cache_block_size * HEAD_DIM + block_offset * 4;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = k_scale;
  }
  const float k_inv_scale = 1.0f / k_scale;

  // Preshuffled page layout: 16x16 (token x dim) tiles.
  constexpr int TILE = 16;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int dim = t + h * HALF_DIM;
    const int token_tile_id = block_offset / TILE;
    const int token_in_tile = block_offset % TILE;
    const int col_tile_id = dim / TILE;
    const int col_in_tile = dim % TILE;
    const int64_t dst_offset = page_base + token_tile_id * (TILE * HEAD_DIM) + col_tile_id * (TILE * TILE) +
                               token_in_tile * TILE + col_in_tile;
    kv_cache[dst_offset] = opus::cast<cache_t>((h == 0 ? klo : khi) * k_inv_scale);
  }
}

}  // namespace dsa_gfx950::qk_rope

struct QkRopeHadamardQuantKernel {
  /// AITER indexer_qk_rope_quant_and_cache plus the inline 128-point Hadamard,
  /// specialised to ue8m0 scales, the preshuffled cache, interleaved rope and a
  /// 128-wide quant block; the Python gate admits nothing else.
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView q_out,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView weights_out,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView kv_cache,
      const tvm::ffi::TensorView slot_mapping,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView norm_bias,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView cos_cache,
      const tvm::ffi::TensorView sin_cache,
      double epsilon,
      double weights_scale) {
    using namespace host;
    namespace impl = dsa_gfx950::qk_rope;

    /// ``q_out`` and ``kv_cache`` are reinterpret_cast to a 1-byte cache_t and
    /// written as raw fp8, so what matters is the width, not the flavour: a
    /// wider dtype fills half the row and leaves the rest untouched.
    const auto byte_width = [](DLDataType dtype) -> int64_t { return (int64_t{dtype.bits} * dtype.lanes + 7) / 8; };

    const int num_tokens = static_cast<int>(std::min(k.size(0), slot_mapping.size(0)));
    // num_tokens becomes gridDim.x and indexes all six per-token tensors, but
    // only two of them took part in the min above.
    RuntimeCheck(
        q.size(0) >= num_tokens && q_out.size(0) >= num_tokens && weights.size(0) >= num_tokens &&
            weights_out.size(0) >= num_tokens && positions.size(0) >= num_tokens,
        "every per-token tensor must have at least num_tokens rows");
    const int head_dim = static_cast<int>(k.size(1));
    const int n_heads = static_cast<int>(q.size(1));
    RuntimeCheck(cos_cache.dim() == 2 && sin_cache.dim() == 2, "cos/sin must be 2-D");
    const int rope_dim = static_cast<int>(cos_cache.size(1)) * 2;
    const int cache_block_size = static_cast<int>(kv_cache.size(1));
    const int cache_stride = static_cast<int>(kv_cache.size(2));
    const int max_position = static_cast<int>(cos_cache.size(0));

    RuntimeCheck(head_dim == impl::HEAD_DIM, "head_dim must be 128");
    RuntimeCheck(rope_dim == impl::ROPE_DIM, "rope_dim must be 64");
    RuntimeCheck(is_type<bf16_t>(q.dtype()) && is_type<bf16_t>(k.dtype()), "q/k must be bf16");
    RuntimeCheck(is_type<bf16_t>(weights.dtype()), "weights must be bf16");
    RuntimeCheck(
        byte_width(q_out.dtype()) == 1 && byte_width(kv_cache.dtype()) == 1,
        "q_out and kv_cache must be 1-byte fp8, got ",
        q_out.dtype(),
        "/",
        kv_cache.dtype());
    RuntimeCheck(is_type<fp32_t>(weights_out.dtype()), "weights_out must be fp32");
    RuntimeCheck(
        is_type<fp32_t>(norm_weight.dtype()) && is_type<fp32_t>(norm_bias.dtype()), "norm params must be fp32");
    RuntimeCheck(cos_cache.stride(1) == 1 && sin_cache.stride(1) == 1, "cos/sin last dim must be contiguous");
    RuntimeCheck(is_type<int64_t>(slot_mapping.dtype()), "slot_mapping must be int64");
    RuntimeCheck(is_type<int64_t>(positions.dtype()), "positions must be int64");

    const auto* qp = reinterpret_cast<const impl::scalar_t*>(q.data_ptr());
    auto* qop = reinterpret_cast<impl::cache_t*>(q_out.data_ptr());
    const auto* wp = reinterpret_cast<const impl::scalar_t*>(weights.data_ptr());
    auto* wop = static_cast<fp32_t*>(weights_out.data_ptr());
    const auto* kp = reinterpret_cast<const impl::scalar_t*>(k.data_ptr());
    auto* cp = reinterpret_cast<impl::cache_t*>(kv_cache.data_ptr());
    const auto* sp = static_cast<const int64_t*>(slot_mapping.data_ptr());
    const auto* nwp = static_cast<const fp32_t*>(norm_weight.data_ptr());
    const auto* nbp = static_cast<const fp32_t*>(norm_bias.data_ptr());
    const auto* pp = static_cast<const int64_t*>(positions.data_ptr());
    const auto* cosp = reinterpret_cast<const impl::scalar_t*>(cos_cache.data_ptr());
    const auto* sinp = reinterpret_cast<const impl::scalar_t*>(sin_cache.data_ptr());

    // rotate_activation(x) == hadamard_transform(x, scale=x.size(-1) ** -0.5)
    const float had_scale = 1.0f / sqrtf(static_cast<float>(impl::HEAD_DIM));

    // The wave path indexes the head dimension directly, so the layout
    // invariants it relies on are asserted rather than carried as arguments.
    RuntimeCheck(q.stride(2) == 1 && q_out.stride(2) == 1 && k.stride(1) == 1, "wave kernel needs contiguous head dim");
    RuntimeCheck(
        weights.stride(1) == 1 && weights_out.stride(1) == 1, "wave kernel needs contiguous head axis on weights");

    LaunchKernel(dim3(num_tokens, n_heads + 1), dim3(impl::HALF_DIM), q.device())(
        impl::qk_rope_hadamard_quant_kernel,
        qp,
        qop,
        wp,
        wop,
        kp,
        cp,
        sp,
        nwp,
        nbp,
        pp,
        cosp,
        sinp,
        n_heads,
        cache_block_size,
        cache_stride,
        static_cast<int>(q.stride(0)),
        static_cast<int>(q.stride(1)),
        static_cast<int>(q_out.stride(0)),
        static_cast<int>(q_out.stride(1)),
        static_cast<int>(weights.stride(0)),
        static_cast<int>(weights_out.stride(0)),
        static_cast<int>(k.stride(0)),
        static_cast<int>(cos_cache.stride(0)),
        static_cast<int>(sin_cache.stride(0)),
        static_cast<float>(epsilon),
        static_cast<float>(weights_scale),
        had_scale,
        max_position);
  }
};

}  // namespace sglang
"""

SRC_PAGED_MQA_LOGITS = r"""/// GLM-5.2 DSA indexer, the logits kernel: paged MQA FP8 logits plus the fine
/// histogram and coarse summary topk_transform.cuh selects from.  One kernel
/// ships, logits_hist_m<8, 12>; WARPS and HB are template arguments.

#pragma once

#ifndef USE_ROCM
#error "paged_mqa_logits.cuh targets gfx950; it is raw CDNA4 MFMA and wave64"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace dsa_gfx950::logits {

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int i32x8;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int i32x4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f32x4;

#define PAGE_TOK 64
#define HD 128
#define HEADS 32
// topk_transform.cuh's TOPK: both kernels must agree on which rows get binned.
#define LG_TOPK 2048
#define TOK_STRIDE 132
#define PAGE_BYTES (PAGE_TOK * TOK_STRIDE)  // 8448
#define K_BYTES (PAGE_TOK * HD)             // 8192

union V8 {
  i32x8 v;
  i32x4 h[2];
};

__device__ __forceinline__ f32x4 mfma128(i32x8 a, i32x8 b, f32x4 c) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}

// wave64 lane exchanges on the VALU instead of through LDS:
//   xor32 <- v_permlane64_b32 (lane l reads lane l^32)
//   xor16 <- v_permlanex16_b32 with identity selectors (lane l reads lane l^16)
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i32x2;
__device__ __forceinline__ float xor32(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane32_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);
}
__device__ __forceinline__ float xor16(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane16_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);
}

// The top-k stage's histogram, built here while this kernel waits on KV loads.
// It shares three invariants with topk_transform.cuh: the key (order_bin_fast),
// the npages clamp, and do_hist's TOPK.
#define LG_CBITS 6
#define LG_CBINS (1 << LG_CBITS)

// Fused paged-MQA FP8 logits + in-loop fine histogram + coarse summary.  The
// histogram is what lets the top-k stage skip a separate scoring pass.

// order_key16(x) >> LOWB in 4 VALU ops after the cvt, with no branch and no
// v_cndmask.  Bit-identical to topk_transform.cuh's order_key16(x) >> LOWB for
// every finite x and for +/-0.
template <int LOWB>
__device__ __forceinline__ uint32_t order_bin_fast(float x) {
  const uint32_t h = (uint32_t)__half_as_ushort(__float2half_rn(x));
  const uint32_t m = (uint32_t)((int32_t)__float_as_int(x) >> 31);
  return (h >> LOWB) ^ (0x8000u >> LOWB) ^ (m & (0x7fffu >> LOWB));
}

template <int WARPS, int HB>
__global__ __launch_bounds__(WARPS * 64) void logits_hist_m(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ wgt,
    const int* __restrict__ seqlens,
    const int* __restrict__ ptable,
    float* __restrict__ out,
    unsigned int* __restrict__ ghist,
    int max_pages,
    int out_stride,
    int rank_rows) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  constexpr int PERT = NBIN / NTH;  // GLOBAL bins per thread (coarse map)
  constexpr int CSH = HB - LG_CBITS;
  constexpr int GRP = NTH / LG_CBINS;
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(NBIN % NTH == 0 && NTH >= LG_CBINS, "");
  static_assert(PERT <= LG_CBINS && (LG_CBINS % PERT) == 0, "a thread's contiguous run must sit inside one coarse bin");
  __shared__ unsigned int s_hist[NBIN];

  const int lane = threadIdx.x & 63;
  // Wave-uniform by construction; readfirstlane lets the compiler keep the page walk in SGPRs.
  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  int row = blockIdx.y;
  int seqlen;
  if (rank_rows) {
    // Longest rows first: blockIdx.y is a RANK, mapped to the row with that rank in descending length (ties by
    // index).  A bijection for any seqlens contents, so every row still gets exactly the same blocks and work; it
    // only changes which rows the hardware dispatches first when the grid needs more than one round of slots.
    // The host only sets rank_rows when gridDim.y <= 64.
    const int R = (int)gridDim.y;
    const int sl = lane < R ? seqlens[lane] : 0;
    int rank = 0;
    for (int j = 0; j < R; ++j) {
      const int sj = __builtin_amdgcn_readlane(sl, j);
      rank += (sj > sl || (sj == sl && j < lane)) ? 1 : 0;
    }
    const unsigned long long m = __ballot(lane < R && rank == (int)blockIdx.y);
    row = __builtin_amdgcn_readfirstlane(m ? (int)__ffsll(m) - 1 : (int)blockIdx.y);
    seqlen = __builtin_amdgcn_readlane(sl, row);
  } else {
    seqlen = seqlens[row];
  }
  // seqlen is device data, so no host check can bound it.  Without the clamp
  // an oversized row walks into the NEXT row's page table -- valid memory,
  // plausible page ids, confidently wrong logits.
  const int npages_raw = (seqlen + PAGE_TOK - 1) / PAGE_TOK;
  // Bound the walk by BOTH the page table and `out`.  The host only requires
  // that `out` cover whole pages -- a captured table is as wide as the graph
  // while the row is only as long as seqlens says -- so a table wider than
  // `out` is legal, and without the second term an oversized seqlen writes
  // past the end of the row.
  const int pages_out = out_stride / PAGE_TOK;
  const int npages_cap = max_pages < pages_out ? max_pages : pages_out;
  const int npages = npages_raw < npages_cap ? npages_raw : npages_cap;
  const int g = lane >> 4, c = lane & 15;
  // Nothing to rank when the row is shorter than topk: everything is selected,
  // so the histogram, its two barriers and the epilogue all compile out.
  //
  // This predicate MUST use the same clamped length the paired top-k kernel
  // applies (it clamps row_ends by the page-table width before its own
  // row_len <= TOPK early return).  A row binned here but skipped there is
  // never zeroed, and the histogram stays dirty for the life of the process.
  const int seqlen_eff = seqlen < npages_cap * PAGE_TOK ? seqlen : npages_cap * PAGE_TOK;
  const bool do_hist = (seqlen_eff > LG_TOPK);

  if (do_hist) {
    for (int i = threadIdx.x; i < NBIN; i += NTH)
      s_hist[i] = 0u;
  }

  const uint8_t* qp = q + (size_t)row * (size_t)(HEADS * HD) + (size_t)c * HD + g * 32;
  V8 qa0, qa1;
  qa0.h[0] = *(const i32x4*)(qp);
  qa0.h[1] = *(const i32x4*)(qp + 16);
  qa1.h[0] = *(const i32x4*)(qp + 16 * HD);
  qa1.h[1] = *(const i32x4*)(qp + 16 * HD + 16);
  const float* wp = wgt + (size_t)row * HEADS + g * 4;
  f32x4 w0 = *(const f32x4*)(wp);
  f32x4 w1 = *(const f32x4*)(wp + 16);

  const int nwaves = gridDim.x * WARPS;
  const int wid = blockIdx.x * WARPS + warp;
  const int koff = g * 512 + c * 16;
  // Pages of this wave: p = wid + it * nwaves, it in [0, cnt).  Uniform.
  const int cnt = wid < npages ? (npages - 1 - wid) / nwaves + 1 : 0;

  unsigned int* __restrict__ gh = ghist + (size_t)row * GHS;

  const int* __restrict__ pt = ptable + (size_t)row * max_pages;
  float* __restrict__ orow = out + (size_t)row * out_stride;

  // Page ids of the first 64 iterations, one per lane, read back with readlane (uniform lane index).
  const int pid = (lane < cnt) ? pt[wid + lane * nwaves] : 0;

  // K of one page through a per-page buffer resource: the page base lives in SGPRs and every lane offset is
  // koff + an immediate, so a page costs 9 loads and no per-lane address arithmetic.
  auto load_page = [&](V8 (&b)[4], float& ks, int phys) {
    const __amdgpu_buffer_rsrc_t rs =
        __builtin_amdgcn_make_buffer_rsrc((void*)(kv + (size_t)phys * PAGE_BYTES), (short)0, PAGE_BYTES, 0x00020000);
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      b[tt].h[0] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048, (tt >> 1) * 4096, 0));
      b[tt].h[1] =
          __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, koff + (tt & 1) * 2048 + 256, (tt >> 1) * 4096, 0));
    }
    ks = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rs, lane * 4, K_BYTES, 0));
  };

  // One page: MFMAs, head-weighted reduction, logits store and histogram.  Operands, scale and summation order
  // are exactly the original kernel's, so the logits are bit-identical.
  auto compute = [&](const V8 (&b)[4], const float ks, const int it) {
    float s[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const f32x4 z = {0.f, 0.f, 0.f, 0.f};
      f32x4 a0 = mfma128(qa0.v, b[tt].v, z);
      f32x4 a1 = mfma128(qa1.v, b[tt].v, z);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a0[i], 0.f), w0[i], acc);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a1[i], 0.f), w1[i], acc);
      s[tt] = acc;
    }
    // Cross-lane head reduction, one swap per level: permlane32_swap(x, y) yields [x_lo | y_lo] and
    // [x_hi | y_hi], whose sum is exactly the original "own + xor32 partner" of the selected operand (the upper
    // half adds the same two values in the other order; fp add is commutative, so the result is bit-identical).
    // permlane16_swap does the same on 16-lane rows.
    // NOTE: this LLVM folds fadd(swap[0], swap[1]) into swap[0] + swap[0] (miscompile seen on gfx950 / ROCm 7.2),
    // so the second result is laundered through an empty asm before the add.
    const i32x2 p02 =
        __builtin_amdgcn_permlane32_swap(__builtin_bit_cast(int, s[0]), __builtin_bit_cast(int, s[2]), false, false);
    const i32x2 p13 =
        __builtin_amdgcn_permlane32_swap(__builtin_bit_cast(int, s[1]), __builtin_bit_cast(int, s[3]), false, false);
    int h02 = p02[1], h13 = p13[1];
    asm volatile("" : "+v"(h02), "+v"(h13));
    const float A0 = __builtin_bit_cast(float, p02[0]) + __builtin_bit_cast(float, h02);
    const float A1 = __builtin_bit_cast(float, p13[0]) + __builtin_bit_cast(float, h13);
    const i32x2 pa =
        __builtin_amdgcn_permlane16_swap(__builtin_bit_cast(int, A0), __builtin_bit_cast(int, A1), false, false);
    int ha = pa[1];
    asm volatile("" : "+v"(ha));
    const float T = __builtin_bit_cast(float, pa[0]) + __builtin_bit_cast(float, ha);
    const int p = wid + it * nwaves;
    const bool live = p * PAGE_TOK + lane < seqlen;
    const float v = T * ks;
    (orow + (size_t)p * PAGE_TOK)[lane] = live ? v : -INFINITY;
    // Unconditional (adds 0 when not counted) so the page body stays one basic block.
    const uint32_t bb = order_bin_fast<LOWB>(v);
    atomicAdd(&s_hist[bb], (do_hist && live) ? 1u : 0u);
  };

  // Two K buffers, one page ahead: page it+1's loads are in flight under page it's MFMAs and reduction.
  V8 bA[4], bB[4];
  float ksA = 0.f, ksB = 0.f;
  // readfirstlane outside the branch: every path into the loop has then waited on pid, so the loop's readlanes
  // need no vmcnt(0) of their own.
  const int phys0 = __builtin_amdgcn_readfirstlane(pid);
  if (cnt > 0) load_page(bA, ksA, phys0);

  if (do_hist) __syncthreads();

  // The scheduler would otherwise sink the next page's loads below the current page's MFMAs (register pressure),
  // which serialises load -> use again.  An empty volatile asm that "rewrites" the current buffer is ordered after
  // the loads just issued, and the MFMAs depend on its outputs.
  auto fence = [&](V8 (&b)[4], float& ks) {
#pragma unroll
    for (int tt = 0; tt < 4; ++tt)
      asm volatile("" : "+v"(b[tt].v));
    asm volatile("" : "+v"(ks));
  };

  // The host sizes the grid so that a wave has at most 64 pages; the generic tail below keeps longer walks exact.
  const int cnt_p = cnt < 64 ? cnt : 64;
  int it = 0;
  for (; it + 2 < cnt_p; it += 2) {
    load_page(bB, ksB, __builtin_amdgcn_readlane(pid, it + 1));
    fence(bA, ksA);
    compute(bA, ksA, it);
    load_page(bA, ksA, __builtin_amdgcn_readlane(pid, it + 2));
    fence(bB, ksB);
    compute(bB, ksB, it + 1);
  }
  if (it + 1 < cnt_p) {
    load_page(bB, ksB, __builtin_amdgcn_readlane(pid, it + 1));
    fence(bA, ksA);
    compute(bA, ksA, it);
    compute(bB, ksB, it + 1);
    it += 2;
  } else if (it < cnt_p) {
    compute(bA, ksA, it);
    it += 1;
  }
  for (; it < cnt; ++it) {
    load_page(bA, ksA, __builtin_amdgcn_readfirstlane(pt[wid + it * nwaves]));
    compute(bA, ksA, it);
  }

  if (do_hist) {
    __syncthreads();

    // ---- coarse summary, mapped over the GLOBAL bin index ----
    // Thread tx owns global bins [tx*PERT, tx*PERT+PERT), which lies inside
    // coarse bin (tx*PERT)>>CSH.  Needs no barrier of its own.
    const int g0 = (int)threadIdx.x * PERT;
    unsigned int acc = 0u;
#pragma unroll
    for (int j = 0; j < PERT; ++j)
      acc += s_hist[g0 + j];
#pragma unroll
    for (int o = GRP >> 1; o >= 1; o >>= 1)
      acc += __shfl_down(acc, o, 64);
    if ((threadIdx.x & (GRP - 1)) == 0 && acc) atomicAdd(&gh[NBIN + (g0 >> CSH)], acc);

    for (int i = threadIdx.x; i < NBIN / 2; i += NTH) {
      const unsigned long long vv = (unsigned long long)s_hist[2 * i] | ((unsigned long long)s_hist[2 * i + 1] << 32);
      if (vv) atomicAdd((unsigned long long*)(gh + 2 * i), vv);
    }
  }
}

// Resident blocks of `fn` on the current device (per-device constant, cached); 0 if unknown.
inline long long resident_slots(const void* fn, int threads) {
  int dev = 0;
  static constexpr int kMaxDev = 16;
  static long long s_slots[kMaxDev] = {0};
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) {
    return 0;
  }
  if (s_slots[dev] == 0) {
    hipDeviceProp_t prop;
    int per_cu = 0;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, fn, threads, 0) != hipSuccess) {
      return 0;
    }
    s_slots[dev] = (long long)per_cu * (long long)prop.multiProcessorCount;
  }
  return s_slots[dev];
}

/// The fp8 dtype the MFMA reads as raw OCP e4m3 bytes.  Checked by hand rather
/// than through ``with_dtype<fp8_e4m3_t>``: on ROCm ``sglang::fp8_e4m3_t`` is
/// ``uint8_t``, which ``DLDataTypeTrait`` maps to kDLUInt -- that would reject a
/// genuine fp8 tensor and accept a uint8 one, exactly backwards.
inline auto is_fp8_e4m3fn(DLDataType dtype) -> bool {
  return dtype.lanes == 1 && dtype.bits == 8 && dtype.code == kDLFloat8_e4m3fn;
}

}  // namespace dsa_gfx950::logits

struct PagedMqaLogitsKernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView seqlens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView ghist,
      int64_t blocks_per_row) {
    using namespace host;
    namespace impl = dsa_gfx950::logits;

    // The launch below is logits_hist_m<8, HOST_HB>, so the histogram row
    // stride the kernel writes is fixed here too.  topk_transform.cuh checks
    // the same width from its side; without this one a mismatch overruns into
    // the next row's counters.
    constexpr int HOST_HB = 12;
    constexpr int64_t HOST_GHS = (int64_t{1} << HOST_HB) + LG_CBINS;

    auto rows_ = SymbolicSize{"rows"};
    auto pages_ = SymbolicSize{"table_pages"};
    auto cols_ = SymbolicSize{"logits_cols"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    // Rebinding one symbolic dtype across q and kv is the "kv must have the
    // same fp8 dtype as q" check; the options stay open because e4m3fn has no
    // DLDataTypeTrait on ROCm.
    auto fp8_ = SymbolicDType{};

    // No with_strides means TensorMatcher demands full contiguity, which is
    // what each of these wants.
    TensorMatcher({rows_, int64_t{HEADS}, int64_t{HD}}).with_dtype(fp8_).with_device(device_).verify(q);
    // PAGE_TOK and TOK_STRIDE are compiled in; a cache laid out differently
    // would be read as garbage rather than refused, hence the row width here.
    TensorMatcher({-1, int64_t{PAGE_TOK} * TOK_STRIDE}).with_dtype(fp8_).with_device(device_).verify(kv);
    RuntimeCheck(
        impl::is_fp8_e4m3fn(fp8_.unwrap()), "q and the kv cache must be fp8 e4m3fn; they are read as raw bytes");

    TensorMatcher({rows_, int64_t{HEADS}}).with_dtype<fp32_t>().with_device(device_).verify(weights);
    TensorMatcher({rows_, pages_}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({rows_, HOST_GHS})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({HOST_GHS, 1})
        .verify(ghist);
    // `out` is a column slice of the workspace, so only the row is packed.
    TensorMatcher({rows_, cols_}).with_dtype<fp32_t>().with_device(device_).with_strides({-1, 1}).verify(out);

    // One entry per row, but the caller may hand over a longer buffer.
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(seqlens);
    RuntimeCheck(
        seqlens.size(0) >= rows_.unwrap(),
        "seqlens needs one entry per row: got ",
        seqlens.size(0),
        " for ",
        rows_.unwrap(),
        " rows");

    // A captured page table is as wide as the graph while the row is as long
    // as seqlens says, so it may exceed `out`; what must hold is that `out`
    // covers whole pages.
    RuntimeCheck(cols_.unwrap() % PAGE_TOK == 0, "out row must be a whole number of pages, got ", cols_.unwrap());

    // Rank rows by length (longest dispatched first) only when the grid needs more than one round of resident
    // slots: with a single round the order is moot and the ranking would only add a round trip.  rows and
    // blocks_per_row are host constants and the slot count is a per-device constant, so this is graph-safe.
    const int rows_i = static_cast<int>(rows_.unwrap());
    const long long slots = impl::resident_slots(reinterpret_cast<const void*>(impl::logits_hist_m<8, HOST_HB>), 8 * 64);
    const int rank_rows = (rows_i <= 64 && blocks_per_row * rows_i > slots) ? 1 : 0;

    LaunchKernel(dim3(static_cast<int>(blocks_per_row), rows_i), dim3(8 * 64), device_.unwrap())(
        impl::logits_hist_m<8, HOST_HB>,
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        static_cast<fp32_t*>(weights.data_ptr()),
        static_cast<int*>(seqlens.data_ptr()),
        static_cast<int*>(page_table.data_ptr()),
        static_cast<fp32_t*>(out.data_ptr()),
        reinterpret_cast<unsigned int*>(ghist.data_ptr()),
        static_cast<int>(pages_.unwrap()),
        static_cast<int>(out.stride(0)),
        rank_rows);
  }
};

}  // namespace sglang

// PAGE_TOK / HD / TOK_STRIDE are generic enough to collide with another header
// compiled into the same translation unit; they exist only for the kernel above.
#undef PAGE_TOK
#undef HD
#undef HEADS
#undef LG_TOPK
#undef TOK_STRIDE
#undef PAGE_BYTES
#undef K_BYTES
#undef LG_CBITS
#undef LG_CBINS
"""

SRC_TOPK_TRANSFORM = r"""/// Top-k(2048) + page transform for the gfx950 DSA indexer, adapted from aiter's
/// coop_topk.cuh (MIT).  aiter runs one block per row; this splits a row across
/// G blocks that agree without communicating, via paged_mqa_logits.cuh's
/// histogram.  Also exports hist_stride: the ghist row stride is a compile-time
/// property of this kernel and the workspace must be sized on it.

#pragma once

#ifndef USE_ROCM
#error "topk_transform.cuh targets gfx950; it is wave64 DPP and HIP occupancy"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace dsa_gfx950::topk {

constexpr uint32_t TOPK = 2048u;
// Coarse bin width, in bits of the fp16 ordered key: trades candidate-set size
// against global-histogram traffic.
#ifndef DSA_TOPK_HIST_BITS
#define DSA_TOPK_HIST_BITS 12
#endif
constexpr uint32_t HIST_BITS = (uint32_t)DSA_TOPK_HIST_BITS;
constexpr uint32_t HIST_BINS = 1u << HIST_BITS;
constexpr uint32_t LOW_BITS = 16u - HIST_BITS;
// Hierarchical (two-level) threshold.
constexpr uint32_t CBITS = 6u;
constexpr uint32_t CBINS = 1u << CBITS;                // 64, one wave wide
constexpr uint32_t FINE_PER_CRS = HIST_BINS >> CBITS;  // 64 at HIST_BITS=12
static_assert(HIST_BITS >= CBITS, "coarse must be no wider than the fine histogram");
static_assert(FINE_PER_CRS <= 64u, "the fine group must fit in one wave");
// Row stride of the ghist workspace: the fine histogram followed by the coarse
// summary.  Both halves are zeroed by whoever owns the reset.
constexpr uint32_t GH_STRIDE = HIST_BINS + CBINS;
constexpr uint32_t GH_CRS_OFF = HIST_BINS;
#ifndef DSA_TOPK_BS
#define DSA_TOPK_BS 256
#endif
constexpr uint32_t BS = (uint32_t)DSA_TOPK_BS;  // threads per block
constexpr uint32_t RADIX = 256u;                // refinement radix
static_assert(BS % 64u == 0u, "BS must be a whole number of wave64 waves");
static_assert(BS >= 64u && BS <= 1024u, "HIP workgroup bound");
#ifndef DSA_TOPK_STAGE
#define DSA_TOPK_STAGE 512
#endif
constexpr uint32_t STAGE = (uint32_t)DSA_TOPK_STAGE;
// Candidates held in LDS by the refinement.  Above this it streams from global,
// which only happens on inputs where one coarse bin holds > REF_CAP elements.
constexpr uint32_t REF_CAP = 2048u;
// Candidate count up to which the refinement uses the O(n^2) rank method
// instead of the radix rounds.  See refine_row.
#ifndef DSA_TOPK_RANK_CAP
#define DSA_TOPK_RANK_CAP 256
#endif
constexpr uint32_t RANK_CAP = (uint32_t)DSA_TOPK_RANK_CAP;
// The parallel rank puts one thread on one j.
static_assert(RANK_CAP <= BS, "the parallel rank needs a thread per candidate");
// Per-row counters are padded to their own cache line, and the two counters a
// block updates live in one 8-byte word, so a block reserves output space with
// ONE returning global atomic instead of two and rows do not false-share.
constexpr uint32_t RC_STRIDE = 32u;
constexpr uint32_t RC_WIN = 0u;   // winners emitted so far
constexpr uint32_t RC_CAND = 1u;  // candidates appended so far
constexpr uint32_t RC_ARR = 4u;   // arrival counter (own dword)
// Departure half of the two-phase row barrier.  The arrival counter alone is
// not a barrier more than one block may pass: whoever reset it would race the
// blocks still spinning on it.
constexpr uint32_t RC_DEP = 5u;

// PTMODE 2 stages a window of the page table in LDS.  A block's slice is
// contiguous, so every position it can emit lies in one span of the table.
constexpr uint32_t PT_WIN = 256u;

__device__ __forceinline__ uint32_t order_key32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Ordered 16-bit key of x rounded to fp16.  __float2half_rn is monotonic, so
// key16(a) <= key16(b) whenever a <= b, which is what makes the three-way
// split in K2 sound.
__device__ __forceinline__ uint32_t order_key16(float x) {
  __half h = __float2half_rn(x);
  unsigned short bits = __half_as_ushort(h);
  unsigned short key = (bits & 0x8000) ? (unsigned short)(~bits) : (unsigned short)(bits | 0x8000);
  return (uint32_t)key;
}

// Physical KV slot of a row-relative position: pt64[row][p >> 6] * 64 + (p &
// 63) for page_size 64, which is the definition of page_table_1.  PB/PM carry
// the shift and mask so page_size 1 collapses to the identity.
__device__ __forceinline__ int32_t
slot_of(const int32_t* __restrict__ pt, uint32_t pos, uint32_t page_bits, uint32_t page_mask) {
  return (pt[pos >> page_bits] << page_bits) | (int32_t)(pos & page_mask);
}

struct Slice {
  uint32_t start, len;
};

// This block's slice, cut on float4 boundaries so every load stays 16B aligned.
__device__ __forceinline__ Slice slice_of(uint32_t row_len, uint32_t g, uint32_t G) {
  const uint32_t units = (row_len + 3u) / 4u;
  const uint32_t base = units / G;
  const uint32_t extra = units % G;
  const uint32_t my_u = base + (g < extra ? 1u : 0u);
  const uint32_t off_u = g * base + (g < extra ? g : extra);
  Slice s{};
  s.start = off_u * 4u;
  s.len = s.start >= row_len ? 0u : min(my_u * 4u, row_len - s.start);
  return s;
}

template <typename Op>
__device__ __forceinline__ void scan_slice(const float* __restrict__ in, Slice sl, Op op) {
  const uint32_t tx = threadIdx.x;
  const uint32_t vec_len = sl.len & ~3u;
  const float4* in4 = reinterpret_cast<const float4*>(in + sl.start);
  const uint32_t n4 = vec_len >> 2;

  uint32_t i = tx;
  for (; i + 3u * BS < n4; i += 4u * BS) {
    const float4 v0 = in4[i];
    const float4 v1 = in4[i + BS];
    const float4 v2 = in4[i + 2u * BS];
    const float4 v3 = in4[i + 3u * BS];
    const float4 vv[4] = {v0, v1, v2, v3};
#pragma unroll
    for (uint32_t u = 0; u < 4u; ++u) {
      const float4 v = vv[u];
      const float vals[4] = {v.x, v.y, v.z, v.w};
      const uint32_t bpos = sl.start + ((i + u * BS) << 2);
#pragma unroll
      for (uint32_t j = 0; j < 4u; ++j) {
        op(vals[j], bpos + j);
      }
    }
  }
  for (; i < n4; i += BS) {
    const float4 v = in4[i];
    const float vals[4] = {v.x, v.y, v.z, v.w};
    const uint32_t bpos = sl.start + (i << 2);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      op(vals[j], bpos + j);
    }
  }
  for (uint32_t t = vec_len + tx; t < sl.len; t += BS) {
    op(in[sl.start + t], sl.start + t);
  }
}

constexpr uint32_t WAVE = 64u;  // gfx950
constexpr uint32_t NWAVE = BS / WAVE;

__device__ __forceinline__ void hist_add_agg(uint32_t* hist, uint32_t bin) {
  const uint64_t active = __ballot(1);
  const int leader = __ffsll((unsigned long long)active) - 1;
  const uint32_t lead_bin = __shfl(bin, leader, WAVE);
  if (__all(bin == lead_bin)) {
    if ((int)(threadIdx.x % WAVE) == leader) {
      atomicAdd(&hist[lead_bin], (uint32_t)__popcll(active));
    }
  } else {
    atomicAdd(&hist[bin], 1u);
  }
}

template <uint32_t NB>
__device__ __forceinline__ void find_thr(
    const uint32_t* __restrict__ hist,
    uint32_t* __restrict__ wtot,
    uint32_t want,
    uint32_t* out_thr,
    uint32_t* out_above) {
  static_assert(NB % BS == 0, "each thread owns a whole run of bins");
  constexpr uint32_t PER = NB / BS;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % WAVE;
  const uint32_t wv = tx / WAVE;

  uint32_t local[PER];
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t j = 0; j < PER; ++j) {
    local[j] = hist[tx * PER + j];
    mine += local[j];
  }

  uint32_t incl = mine;
#pragma unroll
  for (uint32_t o = 1; o < WAVE; o <<= 1) {
    const uint32_t nv = __shfl_up(incl, o, WAVE);
    if (lane >= o) {
      incl += nv;
    }
  }
  if (lane == WAVE - 1u) {
    wtot[wv] = incl;
  }
  __syncthreads();

  uint32_t base = 0, total = 0;
#pragma unroll
  for (uint32_t w = 0; w < NWAVE; ++w) {
    const uint32_t t = wtot[w];
    if (w < wv) {
      base += t;
    }
    total += t;
  }
  uint32_t acc = total - (base + incl);  // strictly above this thread's group

#pragma unroll
  for (int j = (int)PER - 1; j >= 0; --j) {
    const uint32_t c = local[j];
    if (acc < want && acc + c >= want) {
      *out_thr = tx * PER + (uint32_t)j;
      *out_above = acc;
    }
    acc += c;
  }
  __syncthreads();  // wtot[] is scratch and out_thr/out_above must be visible
}

__device__ __forceinline__ uint32_t wave_suffix_sum(uint32_t v) {
  const uint32_t lane = threadIdx.x & (WAVE - 1u);
#pragma unroll
  for (uint32_t o = 1; o < WAVE; o <<= 1) {
    const uint32_t t = __shfl_down(v, o, WAVE);
    v += (lane + o < WAVE) ? t : 0u;
  }
  return v;
}

// The same threshold as find_thr<HIST_BINS>, from two levels, bit-identical.
// A histogram written without its coarse summary fails silently, so lane 0's
// suffix sum must equal row_len; on mismatch return HIER_BAD.
constexpr uint32_t HIER_BAD = 0xFFFFFFFFu;

__device__ __forceinline__ uint32_t
find_thr_hier_impl(const uint32_t* __restrict__ gh, uint32_t want, uint32_t crs, uint32_t total_must_be) {
  const uint32_t lane = threadIdx.x & (WAVE - 1u);

  const uint32_t s = wave_suffix_sum(crs);
  if (__shfl(s, 0, WAVE) != total_must_be) {
    return HIER_BAD;
  }
  uint32_t sn = __shfl_down(s, 1, WAVE);
  if (lane == WAVE - 1u) {
    sn = 0u;
  }
  const uint64_t b1 = __ballot(s >= want && sn < want);
  // Cannot be empty (total == row_len > want and lane 63 sees sn == 0), but a
  // zero ballot would index gh at -1, so it degrades to bin 0 -- which is also
  // what find_thr leaves in *out_thr when it finds nothing.
  const uint32_t j = b1 ? (uint32_t)(__ffsll((unsigned long long)b1) - 1) : 0u;
  const uint32_t abv = b1 ? __shfl(sn, (int)j, WAVE) : 0u;

  // level 2 -- the FINE_PER_CRS fine bins under coarse bin j.  Lanes past the
  // group contribute 0, so they can never satisfy the predicate (want2 >= 1).
  const uint32_t f = (lane < FINE_PER_CRS) ? gh[j * FINE_PER_CRS + lane] : 0u;
  const uint32_t sf = wave_suffix_sum(f);
  uint32_t sfn = __shfl_down(sf, 1, WAVE);
  if (lane == WAVE - 1u) {
    sfn = 0u;
  }
  const uint32_t want2 = want - abv;
  const uint64_t b2 = __ballot(sf >= want2 && sfn < want2);
  const uint32_t l = b2 ? (uint32_t)(__ffsll((unsigned long long)b2) - 1) : 0u;
  return j * FINE_PER_CRS + l;
}

// Exact refinement of the threshold bin + padding.

__device__ __forceinline__ float ld_val(const float* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ int32_t ld_idx(const int32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <bool IN_LDS>
__device__ __forceinline__ void refine_core(
    uint32_t n,
    uint32_t above,
    uint32_t remain,
    const uint32_t* __restrict__ s_key,
    const int32_t* __restrict__ s_slot,
    const float* __restrict__ cv,
    const int32_t* __restrict__ ci,
    int32_t* __restrict__ o,
    uint32_t* __restrict__ s_hist,
    uint32_t* __restrict__ s_grp,
    uint32_t* s_thr,
    uint32_t* s_above,
    uint32_t* s_emit) {
  const uint32_t tx = threadIdx.x;

  if (remain == 0 || n <= remain) {
    // Defensive: the threshold guarantees n >= remain, so this is the
    // degenerate path only.  Emit everything we have, then pad.
    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t p = above + i;
      if (p < TOPK) {
        o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i]);
      }
    }
    __syncthreads();
    for (uint32_t i = above + n + tx; i < TOPK; i += BS) {
      o[i] = -1;
    }
    return;
  }

  uint32_t prefix = 0;
  for (int r = 0; r < 4; ++r) {
    const uint32_t sh = 24u - (uint32_t)r * 8u;

    for (uint32_t i = tx; i < RADIX; i += BS) {
      s_hist[i] = 0u;
    }
    if (tx == 0) {
      *s_thr = 0;
      *s_above = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t key = IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i]));
      const bool in_play = (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
      if (in_play) {
        hist_add_agg(s_hist, (key >> sh) & 0xFFu);
      }
    }
    __syncthreads();

    find_thr<RADIX>(s_hist, s_grp, remain, s_thr, s_above);

    const uint32_t thr = *s_thr;
    const uint32_t abv = *s_above;
    // Both reads must complete before any thread can reach the next round's
    // reset of *s_thr / *s_above at the top of this loop.  The `continue` below
    // has no barrier after it, so without this a thread that took it would zero
    // *s_thr while a lagging wave was still reading it, and that wave would
    // carry `prefix |= 0` into every later round.
    __syncthreads();

    // Nothing above the threshold bin settles no winner, so the emit scan
    // would find nothing.  Skipping it matters on clustered rows, where the
    // early rounds are exactly this case.
    if (abv == 0 && r != 3) {
      prefix |= (thr << sh);
      continue;
    }

    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t key = IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i]));
      const bool in_play = (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
      if (!in_play) {
        continue;
      }
      const uint32_t bin = (key >> sh) & 0xFFu;
      // Last round: survivors in the threshold bin are numerically equal,
      // so any of them is a correct answer.  The TopK bound on the cursor
      // enforces the quota.
      if (bin > thr || (bin == thr && r == 3)) {
        const uint32_t p = atomicAdd(s_emit, 1u);
        if (p < TOPK) {
          o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i]);
        }
      }
    }
    __syncthreads();

    prefix |= (thr << sh);
    remain -= abv;
    if (remain == 0) {
      break;
    }
  }

  __syncthreads();
  // The emit loop above already wrote o[above .. *s_emit) directly, so all
  // that is left is the -1 padding.
  const uint32_t filled = *s_emit < TOPK ? *s_emit : TOPK;
  for (uint32_t i = filled + tx; i < TOPK; i += BS) {
    o[i] = -1;
  }
}

// Exact rank of the boundary candidates, bit-identical to the serial rank.
// Only the i axis is partitioned -- rank_i sums over ALL j in EVERY block --
// so a candidate on a partition boundary is still ranked against the full set.
template <bool PRANK = false>
__device__ __forceinline__ void refine_row(
    uint32_t row,
    uint32_t above,
    uint32_t n_raw,
    uint32_t cap,
    const int32_t* __restrict__ cand_idx,
    const float* __restrict__ cand_val,
    int32_t* __restrict__ o,
    uint32_t* __restrict__ s_hist,
    uint32_t* __restrict__ s_grp,
    uint32_t* __restrict__ s_key,
    int32_t* __restrict__ s_slot,
    uint32_t* s_thr,
    uint32_t* s_above,
    uint32_t* s_emit,
    int32_t pre_slot,
    float pre_val,
    uint32_t pb,
    uint32_t PB) {
  const uint32_t tx = threadIdx.x;
  const int32_t* __restrict__ ci = cand_idx + (size_t)row * cap;
  const float* __restrict__ cv = cand_val + (size_t)row * cap;

  const uint32_t n = n_raw > cap ? cap : n_raw;
  const uint32_t remain = above < TOPK ? TOPK - above : 0u;

  if (tx == 0) {
    *s_emit = above;
  }

  const bool in_lds = (n <= REF_CAP);
  if (in_lds) {
    // The caller has already issued cand_idx[tx] / cand_val[tx] without
    // waiting for `n`, collapsing row_ends -> cursor -> candidates into one
    // round trip.
    if (tx < n) {
      s_key[tx] = order_key32(pre_val);
      s_slot[tx] = pre_slot;
    }
    for (uint32_t i = tx + BS; i < n; i += BS) {
      s_key[i] = order_key32(ld_val(&cv[i]));
      s_slot[i] = ld_idx(&ci[i]);  // k_scatter already translated it
    }
  }
  __syncthreads();

  if (in_lds && n <= RANK_CAP && remain != 0 && n > remain) {
    if constexpr (PRANK) {
      const uint32_t lane = tx & (WAVE - 1u);
      const uint32_t wv = tx / WAVE;
      const uint32_t kj = (tx < n) ? s_key[tx] : 0u;
      for (uint32_t i = pb; i < n; i += PB) {
        const uint32_t ki = s_key[i];  // block-uniform read
        const bool p = (tx < n) && ((kj > ki) || (kj == ki && tx < i));
        const uint64_t m = __ballot(p);
        if (lane == 0) {
          s_grp[wv] = (uint32_t)__popcll((unsigned long long)m);
        }
        __syncthreads();
        uint32_t rank = 0;
#pragma unroll
        for (uint32_t q = 0; q < NWAVE; ++q) {
          rank += s_grp[q];
        }
        if (tx == 0 && rank < remain) {
          o[above + rank] = s_slot[i];
        }
        __syncthreads();  // s_grp is reused by the next i
      }
      return;
    }
    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t ki = s_key[i];
      uint32_t rank = 0;
      uint32_t j = 0;
      // Unrolled by 8: the naive loop waits on each LDS read, so it runs at
      // LDS latency rather than throughput.
      for (; j + 8u <= n; j += 8u) {
        const uint32_t k0 = s_key[j + 0], k1 = s_key[j + 1];
        const uint32_t k2 = s_key[j + 2], k3 = s_key[j + 3];
        const uint32_t k4 = s_key[j + 4], k5 = s_key[j + 5];
        const uint32_t k6 = s_key[j + 6], k7 = s_key[j + 7];
        rank += (k0 > ki) || (k0 == ki && (j + 0u) < i);
        rank += (k1 > ki) || (k1 == ki && (j + 1u) < i);
        rank += (k2 > ki) || (k2 == ki && (j + 2u) < i);
        rank += (k3 > ki) || (k3 == ki && (j + 3u) < i);
        rank += (k4 > ki) || (k4 == ki && (j + 4u) < i);
        rank += (k5 > ki) || (k5 == ki && (j + 5u) < i);
        rank += (k6 > ki) || (k6 == ki && (j + 6u) < i);
        rank += (k7 > ki) || (k7 == ki && (j + 7u) < i);
      }
      for (; j < n; ++j) {
        const uint32_t kj = s_key[j];
        rank += (kj > ki) || (kj == ki && j < i);
      }
      if (rank < remain) {
        o[above + rank] = s_slot[i];
      }
    }
  } else if (PRANK && pb != 0u) {
    // The radix fallback emits through a shared cursor and pads o[..TOPK):
    // one block's work by construction, so the rest of the row returns here.
  } else if (in_lds) {
    refine_core<true>(n, above, remain, s_key, s_slot, cv, ci, o, s_hist, s_grp, s_thr, s_above, s_emit);
  } else {
    refine_core<false>(n, above, remain, s_key, s_slot, cv, ci, o, s_hist, s_grp, s_thr, s_above, s_emit);
  }
}

// K2: threshold + scatter (page gather folded into the emit)

// Stage this block's span of the row's page table into LDS (PTMODE 2 only).
// A block's slice is contiguous, so the window is [pt_base, pt_base + npt).
__device__ __forceinline__ void stage_pt_window(
    int32_t* __restrict__ s_pt, const int32_t* __restrict__ pt_row, Slice sl, uint32_t pt_base, uint32_t page_bits) {
  const uint32_t want = sl.len ? (((sl.start + sl.len - 1u) >> page_bits) - pt_base + 1u) : 0u;
  const uint32_t npt = want < PT_WIN ? want : PT_WIN;
  for (uint32_t i = threadIdx.x; i < npt; i += BS) {
    s_pt[i] = pt_row[pt_base + i];
  }
}

template <int PTMODE, bool PRANK = false, int PBLK = 0>
__global__ __launch_bounds__(BS) void k_scatter(
    const float* __restrict__ logits,
    const int32_t* __restrict__ row_ends,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ out,
    uint32_t* __restrict__ ghist,
    int32_t* __restrict__ cursor,
    int32_t* __restrict__ cand_idx,
    float* __restrict__ cand_val,
    int64_t lg_stride,
    int64_t pt_stride,
    uint32_t page_bits,
    uint32_t page_mask,
    uint32_t cap,
    uint32_t G) {
  // Three working sets with disjoint lifetimes share one 16 KB block: the flat
  // histogram, then the staging buffers, then the refinement's keys and slots.
  // s_radix stays separate -- it is live at the same time as the keys.
  __shared__ union {
    uint32_t hist[HIST_BINS];
    struct {
      int32_t wbuf[STAGE];
      int32_t cidx[STAGE];
      float cval[STAGE];
    } stage;
    struct {
      uint32_t key[REF_CAP];
      int32_t slot[REF_CAP];
    } ref;
  } s_pool;
  static_assert(sizeof(s_pool) == HIST_BINS * 4u, "the flat histogram is the widest phase");
  __shared__ uint32_t s_radix[RADIX];
  __shared__ uint32_t s_grp[NWAVE];  // cross-wave scan fixup (see find_thr)
  __shared__ uint32_t s_thr, s_above;
  // Per-block staging.  Without it every emitted element would need its own
  // returning atomic on the row's single cursor.
  __shared__ uint32_t s_wcnt, s_ccnt;
  __shared__ int32_t s_wbase, s_cbase;
  // PTMODE is a template parameter and not a runtime flag for the same reason
  // IN_LDS is in refine_core: a runtime select would emit both forms.
  __shared__ int32_t s_pt[PTMODE == 2 ? PT_WIN : 1];
  __shared__ uint32_t s_emit;
  __shared__ uint32_t s_last;
  uint32_t* const s_hist = s_pool.hist;
  int32_t* const s_wbuf = s_pool.stage.wbuf;
  int32_t* const s_cidx = s_pool.stage.cidx;
  float* const s_cval = s_pool.stage.cval;
  uint32_t* const s_key = s_pool.ref.key;
  int32_t* const s_slot = s_pool.ref.slot;

  const uint32_t row = blockIdx.y;
  const uint32_t g = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  const int32_t rl_s = row_ends[row];
  // Issued before row_ends comes back: the address depends only on blockIdx.
  // Later would serialise row_ends -> coarse bins -> fine group.
  const uint32_t pre_crs = ((const uint32_t*)ghist)[(size_t)row * GH_STRIDE + GH_CRS_OFF + (threadIdx.x & (WAVE - 1u))];
  // row_ends is device data, so no host check can bound it.  Unclamped, an
  // oversized row reads past its page table, and under PTMODE 2 past s_pt --
  // which stays inside LDS and so returns garbage rather than faulting.
  const uint32_t pt_cap_pt = (uint32_t)((uint64_t)pt_stride << page_bits);
  // ...and past the logits row, which the scan below walks over [0, row_len).
  // The paired logits kernel bounds its write by the same two quantities, and
  // the two must agree: a row binned there but skipped here is never zeroed.
  const uint32_t pt_cap_lg = (uint32_t)lg_stride;
  const uint32_t pt_cap = pt_cap_pt < pt_cap_lg ? pt_cap_pt : pt_cap_lg;
  const uint32_t rl_u = rl_s > 0 ? (uint32_t)rl_s : 0u;
  const uint32_t row_len = rl_u < pt_cap ? rl_u : pt_cap;
  if (row_len <= TOPK) {
    // Rows short enough to need no selection: this kernel is their only
    // writer, so it emits the whole row here.
    const int32_t* __restrict__ pt0 = page_table + (int64_t)row * pt_stride;
    int32_t* __restrict__ o0 = out + (size_t)row * TOPK;
    for (uint32_t i = g * BS + tx; i < TOPK; i += G * BS) {
      o0[i] = i < row_len ? slot_of(pt0, i, page_bits, page_mask) : -1;
    }
    return;
  }

  int32_t* __restrict__ rc = cursor + (size_t)row * RC_STRIDE;

  // Issued first: independent of everything else, so the page-table staging
  // rides along under the barrier below.
  const Slice sl = slice_of(row_len, g, G);
  // First page-table entry this block can possibly touch.  Every position it
  // emits is inside its own slice, so the window is [pt_base, pt_base+npt).
  const uint32_t pt_base = (PTMODE == 2) ? (sl.start >> page_bits) : 0u;
  if constexpr (PTMODE == 2) {
    stage_pt_window(s_pt, page_table + (int64_t)row * pt_stride, sl, pt_base, page_bits);
  }

  const uint32_t* __restrict__ gh = (const uint32_t*)ghist + (size_t)row * GH_STRIDE;
  // Issued before the barrier so its two dependent loads overlap the
  // page-table window staging above.  Needs neither LDS nor a barrier.
  const uint32_t thr_h = find_thr_hier_impl(gh, TOPK, pre_crs, row_len);
  // The flat histogram is the fallback only.  Block-uniform.
  const bool need_flat = (thr_h == HIER_BAD);

  if (need_flat) {
    for (uint32_t i = tx; i < HIST_BINS; i += BS) {
      s_hist[i] = gh[i];
    }
  }
  if (tx == 0) {
    s_thr = 0;
    s_above = 0;
    s_wcnt = 0;
    s_ccnt = 0;
  }
  __syncthreads();

  if (need_flat) {
    find_thr<HIST_BINS>(s_hist, s_grp, TOPK, &s_thr, &s_above);
    __syncthreads();
  }

  const uint32_t thr = need_flat ? s_thr : thr_h;

  const float* __restrict__ in = logits + (int64_t)row * lg_stride;
  const int32_t* __restrict__ pt = page_table + (int64_t)row * pt_stride;
  int32_t* __restrict__ o = out + (size_t)row * TOPK;
  int32_t* __restrict__ ci = cand_idx + (size_t)row * cap;
  float* __restrict__ cv = cand_val + (size_t)row * cap;

  // The page-table lookup.  `if constexpr` so exactly one form is
  // emitted (see the s_pt declaration).
  auto SLOT = [&](uint32_t p) -> int32_t {
    if constexpr (PTMODE == 2) {
      return (s_pt[(p >> page_bits) - pt_base] << page_bits) | (int32_t)(p & page_mask);
    } else {
      return slot_of(pt, p, page_bits, page_mask);
    }
  };

  scan_slice(in, sl, [&](float v, uint32_t gi) {
    const uint32_t kb = order_key16(v) >> LOW_BITS;
    if (kb > thr) {
      // Winner outright: monotonicity of order_key16 guarantees it beats
      // every element of the threshold bin.
      const uint32_t p = atomicAdd(&s_wcnt, 1u);
      if (p < STAGE) {
        s_wbuf[p] = (int32_t)gi;
      } else {
        const unsigned long long old = atomicAdd((unsigned long long*)rc, 1ull);
        const uint32_t q = (uint32_t)old;
        // The threshold guarantees q < TopK; the bound is belt-and-braces
        // so that no reachable state can scribble past the output row.
        if (q < TOPK) {
          o[q] = SLOT(gi);
        }
      }
    } else if (kb == thr) {
      const uint32_t p = atomicAdd(&s_ccnt, 1u);
      if (p < STAGE) {
        s_cidx[p] = (int32_t)gi;
        s_cval[p] = v;
      } else {
        const unsigned long long old = atomicAdd((unsigned long long*)rc, 1ull << 32);
        const uint32_t q = (uint32_t)(old >> 32);
        if (q < cap) {
          __hip_atomic_store(&ci[q], SLOT(gi), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          __hip_atomic_store(&cv[q], v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  });

  __syncthreads();

  const uint32_t wn = s_wcnt < STAGE ? s_wcnt : STAGE;
  const uint32_t cn = s_ccnt < STAGE ? s_ccnt : STAGE;

  // Issue the page-table gathers first: they do not depend on the base this
  // block is about to reserve, so they overlap the atomic's round trip.
  constexpr uint32_t PERT = (STAGE + BS - 1u) / BS;
  int32_t wslot[PERT], cslot[PERT];
  float cvalr[PERT];
#pragma unroll
  for (uint32_t u = 0; u < PERT; ++u) {
    const uint32_t i = tx + u * BS;
    if (i < wn) {
      wslot[u] = SLOT((uint32_t)s_wbuf[i]);
    }
    if (i < cn) {
      cslot[u] = SLOT((uint32_t)s_cidx[i]);
      cvalr[u] = s_cval[i];
    }
  }

  // One thread reserves this block's span in both output streams with a single
  // packed atomic, so the two counters cannot be observed out of step.
  if (tx == 0) {
    const unsigned long long pack = ((unsigned long long)cn << 32) | (unsigned long long)wn;
    const unsigned long long old = atomicAdd((unsigned long long*)rc, pack);
    s_wbase = (int32_t)(uint32_t)old;
    s_cbase = (int32_t)(uint32_t)(old >> 32);
  }
  __syncthreads();

#pragma unroll
  for (uint32_t u = 0; u < PERT; ++u) {
    const uint32_t i = tx + u * BS;
    if (i < wn) {
      const uint32_t p = (uint32_t)s_wbase + i;
      if (p < TOPK) {
        o[p] = wslot[u];
      }
    }
    if (i < cn) {
      const uint32_t q = (uint32_t)s_cbase + i;
      if (q < cap) {
        // The only bytes another block reads inside this kernel, which is why
        // just these stores are agent-scope.
        __hip_atomic_store(&ci[q], cslot[u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&cv[q], cvalr[u], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }

  // Release, the cheap way: the candidate stores above are already
  // write-through to a coherent point, so waiting for them to retire is
  // enough -- no __threadfence().
  __builtin_amdgcn_s_waitcnt(/*vmcnt(0)*/ 0x0f70);
  __syncthreads();
  if (tx == 0) {
    const uint32_t old = __hip_atomic_fetch_add((uint32_t*)&rc[RC_ARR], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    // PRANK needs this block's arrival rank, the non-PRANK form only the
    // is-last flag, and no form needs both, so they share s_last.
    s_last = PRANK ? old : ((old + 1u == G) ? 1u : 0u);
  }
  __syncthreads();
  const uint32_t PB_N = (PBLK == 0 || (uint32_t)PBLK > G) ? G : (uint32_t)PBLK;
  const uint32_t pfirst = G - PB_N;
  const uint32_t pidx = s_last - pfirst;  // valid only if s_last>=pfirst
  // Under PRANK the arrival counter stops being an election (one block
  // continues, G-1 return) and becomes the arrival half of a row barrier.
  if constexpr (PRANK) {
    if (s_last < pfirst) {
      return;
    }
    if (tx == 0) {
      uint32_t v = __hip_atomic_load((uint32_t*)&rc[RC_ARR], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (v < G) {
        __builtin_amdgcn_s_sleep(2);
        v = __hip_atomic_load((uint32_t*)&rc[RC_ARR], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
  } else if (!s_last) {
    return;
  }

  // ACQUIRE side of the handshake: the cursor reads below are agent-scope
  // atomics, so no fence of our own.

  static_assert(GH_STRIDE % 4u == 0u, "vectorised reset needs 4 | stride");
  // Under PRANK the reset is spread over the row's blocks instead; the
  // barrier above guarantees every block has finished reading it.
  uint4* __restrict__ ghw4 = (uint4*)(ghist + (size_t)row * GH_STRIDE);
  const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
  for (uint32_t i = PRANK ? pidx * BS + tx : tx; i < GH_STRIDE / 4u; i += PRANK ? PB_N * BS : BS) {
    ghw4[i] = z4;
  }

  const uint32_t above = (uint32_t)__hip_atomic_load(&rc[RC_WIN], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const uint32_t nraw = (uint32_t)__hip_atomic_load(&rc[RC_CAND], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // The candidate prefetch is issued alongside the two cursor reads rather
  // than behind them.  Unconditionally in bounds (cap >= TOPK >= BS);
  // entries at or past n are discarded.
  const int32_t pre_slot = ld_idx(&cand_idx[(size_t)row * cap + tx]);
  const float pre_val = ld_val(&cand_val[(size_t)row * cap + tx]);
  __syncthreads();  // every thread has read them before they are cleared
  if (!PRANK && tx == 0) {
    rc[RC_WIN] = 0;
    rc[RC_CAND] = 0;
    rc[RC_ARR] = 0;
  }
  refine_row<PRANK>(
      row,
      above,
      nraw,
      cap,
      cand_idx,
      cand_val,
      o,
      s_radix,
      s_grp,
      s_key,
      s_slot,
      &s_thr,
      &s_above,
      &s_emit,
      pre_slot,
      pre_val,
      pidx,
      PB_N);
  // Departure half of the row barrier: the block that finishes first cannot
  // reset the counters while the other G-1 are still reading them, so the
  // last one out does it.
  if constexpr (PRANK) {
    __syncthreads();
    if (tx == 0) {
      const uint32_t d = __hip_atomic_fetch_add((uint32_t*)&rc[RC_DEP], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (d + 1u == PB_N) {
        rc[RC_WIN] = 0;
        rc[RC_CAND] = 0;
        rc[RC_ARR] = 0;
        rc[RC_DEP] = 0;
      }
    }
  }
}

// PRANK liveness guard.
inline bool prank_resident_ok(const void* fn, int blocks_per_row, int rows) {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return false;
  }
  // Both queries are per-device constants but hipGetDeviceProperties is a
  // heavyweight host call, and this runs once per launch -- once per DSA layer
  // on any path that is not inside a captured graph.  Cache per device.
  static constexpr int kMaxDev = 16;
  static int s_cu[kMaxDev] = {0};
  if (dev < 0 || dev >= kMaxDev) {
    return false;
  }
  if (s_cu[dev] == 0) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
      return false;
    }
    s_cu[dev] = prop.multiProcessorCount;
  }
  int per_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, fn, (int)BS, 0) != hipSuccess) {
    return false;
  }
  if (per_cu <= 0) {
    return false;
  }
  // Conservative on purpose. Only the last PBLK blocks of a row spin (the
  // earlier G - PBLK increment the arrival counter and return, freeing their
  // slots), so the barrier needs PBLK * rows co-resident, not G * rows. The
  // 4x margin at PBLK=16, G=64 is what absorbs the occupancy figure being a
  // whole-device theoretical maximum that ignores concurrently resident work.
  const long long slots = (long long)per_cu * (long long)s_cu[dev];
  return (long long)blocks_per_row * (long long)rows <= slots;
}
}  // namespace dsa_gfx950::topk

struct TopKTransformKernel {
  /// PTMODE 2 stages a per-block page-table window in LDS.  Page-table strides
  /// too wide for that fixed window use the numerically identical PTMODE 0
  /// global-gather specialization instead.
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView row_ends,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView ghist,
      const tvm::ffi::TensorView cursor,
      const tvm::ffi::TensorView cand_idx,
      const tvm::ffi::TensorView cand_val,
      int64_t g_per_row,
      int64_t page_size) {
    using namespace host;
    namespace impl = dsa_gfx950::topk;

    const int64_t R = logits.size(0);
    const int64_t L = logits.stride(0);
    const int64_t PTS = page_table.stride(0);
    RuntimeCheck(cand_idx.dim() == 2, "cand_idx must be [>=rows, cap]");
    const int64_t cap = cand_idx.size(1);
    const uint32_t G = static_cast<uint32_t>(g_per_row);
    // page_size must be 1 or a power of two; both give the identical mapping.
    uint32_t PB = 0;
    for (int64_t ps = page_size; ps > 1; ps >>= 1) {
      ++PB;
    }
    const uint32_t PM = (page_size > 1) ? static_cast<uint32_t>(page_size - 1) : 0u;
    RuntimeCheck(
        (page_size & (page_size - 1)) == 0 && page_size >= 1, "page_size must be a power of two, got ", page_size);

    // The histogram width is a compile-time property of the kernel.  Getting
    // it wrong on the host silently walks off the end of ghist and corrupts
    // the buffers next to it, which is exactly what happened once already.
    RuntimeCheck(
        ghist.numel() == R * static_cast<int64_t>(impl::GH_STRIDE),
        "ghist must be [rows, ",
        impl::GH_STRIDE,
        "], got ",
        ghist.numel());
    // TOPK is compiled in and every row write is `out + row * TOPK`, so a
    // narrower or shorter `out` overruns each row in turn.  The rest of these
    // bound the tensors the kernel indexes by blockIdx.y or by `cap`; each one
    // holds today only because fused_decode.py happens to allocate it that way.
    RuntimeCheck(
        out.dim() == 2 && out.size(0) >= R && out.size(1) == static_cast<int64_t>(impl::TOPK),
        "out must be [>=rows, ",
        impl::TOPK,
        "], got a ",
        out.dim(),
        "-D tensor");
    RuntimeCheck(
        is_type<int32_t>(row_ends.dtype()) && row_ends.is_contiguous() && row_ends.numel() >= R,
        "row_ends must be a contiguous int32 tensor with one entry per row");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= R, "page_table must be [>=rows, width]");
    RuntimeCheck(
        cand_val.dim() == 2 && cand_val.size(0) >= R && cand_val.size(1) >= cap,
        "cand_val must be at least as wide as cand_idx (",
        cap,
        ")");
    RuntimeCheck(cand_idx.size(0) >= R, "cand_idx must have one row per row");
    // scan_slice reads the row base as float4, so the row stride has to keep
    // that base 16B-aligned; the slice starts are multiples of 4 by construction.
    RuntimeCheck(
        logits.stride(1) == 1 && L % 4 == 0,
        "logits rows must be unit-stride with a stride(0) divisible by 4, got ",
        L);
    // The cursor carries the same zero-in/zero-out invariant as ghist, and the
    // consequence of getting it wrong is worse: under PRANK a stale arrival
    // counter hangs the row barrier rather than returning a wrong answer.
    RuntimeCheck(
        cursor.numel() == R * static_cast<int64_t>(impl::RC_STRIDE),
        "cursor must be [rows, ",
        impl::RC_STRIDE,
        "], got ",
        cursor.numel(),
        " elements for ",
        R,
        " rows");
    RuntimeCheck(cap >= static_cast<int64_t>(impl::TOPK), "candidate capacity must be >= TopK");
    // Every pointer below is a static_cast, which validates nothing -- unlike
    // the data_ptr<T>() these replaced. page_table is the one a caller supplies.
    RuntimeCheck(
        is_type<fp32_t>(logits.dtype()) && is_type<int32_t>(out.dtype()) && is_type<int32_t>(page_table.dtype()) &&
            is_type<int32_t>(ghist.dtype()) && is_type<int32_t>(cursor.dtype()) && is_type<int32_t>(cand_idx.dtype()) &&
            is_type<fp32_t>(cand_val.dtype()),
        "logits/cand_val must be fp32 and out/page_table/ghist/cursor/cand_idx int32");

    // PTMODE 2 is faster when its per-block page-table slice fits in LDS.  PTS
    // is a static graph property while row_ends is replay-time data, so use PTS
    // as the conservative upper bound.  PTMODE 0 has no page-table-width limit.
    const bool use_pt_window =
        (PTS + static_cast<int64_t>(G) - 1) / static_cast<int64_t>(G) + 2 <= static_cast<int64_t>(impl::PT_WIN);

    const auto stream = LaunchKernel::resolve_device(logits.device());
    dim3 grid(static_cast<unsigned>(G), static_cast<unsigned>(R), 1);
    dim3 blk(impl::BS, 1, 1);
#define SGL_DSA_SCATTER_ARGS                                                                                 \
  grid, blk, 0, stream, static_cast<fp32_t*>(logits.data_ptr()), static_cast<int32_t*>(row_ends.data_ptr()), \
      static_cast<int32_t*>(page_table.data_ptr()), static_cast<int32_t*>(out.data_ptr()),                   \
      static_cast<uint32_t*>(ghist.data_ptr()), static_cast<int32_t*>(cursor.data_ptr()),                    \
      static_cast<int32_t*>(cand_idx.data_ptr()), static_cast<fp32_t*>(cand_val.data_ptr()), L, PTS, PB, PM, \
      static_cast<uint32_t>(cap), G

    // The persistent-rank tail finishes the exact rank behind a row barrier, so
    // its blocks must be co-resident.  When they would not be, the ordinary
    // kernel-boundary form is launched instead and the selector cannot hang.
    if (use_pt_window &&
        impl::prank_resident_ok(
            reinterpret_cast<const void*>(impl::k_scatter<2, true, 16>), static_cast<int>(G), static_cast<int>(R))) {
      hipLaunchKernelGGL((impl::k_scatter<2, true, 16>), SGL_DSA_SCATTER_ARGS);
    } else if (use_pt_window) {
      hipLaunchKernelGGL((impl::k_scatter<2>), SGL_DSA_SCATTER_ARGS);
    } else if (impl::prank_resident_ok(
                   reinterpret_cast<const void*>(impl::k_scatter<0, true, 16>),
                   static_cast<int>(G),
                   static_cast<int>(R))) {
      hipLaunchKernelGGL((impl::k_scatter<0, true, 16>), SGL_DSA_SCATTER_ARGS);
    } else {
      hipLaunchKernelGGL((impl::k_scatter<0>), SGL_DSA_SCATTER_ARGS);
    }
    RuntimeDeviceCheck();
#undef SGL_DSA_SCATTER_ARGS
  }
};

struct TopKHistStride {
  /// Row stride of the ghist workspace = fine bins + the coarse summary.
  /// The workspace MUST be sized on this, not on the fine bin count: getting it
  /// wrong walks off the end of ghist into whatever is allocated next.
  static auto run() -> int64_t {
    return static_cast<int64_t>(dsa_gfx950::topk::GH_STRIDE);
  }
};

}  // namespace sglang
"""


def _src_dir() -> str:
    blob = "\0".join([SRC_DUAL_GEMV_BF16, SRC_QK_ROPE_HADAMARD_QUANT, SRC_PAGED_MQA_LOGITS, SRC_TOPK_TRANSFORM])
    h = hashlib.sha1(blob.encode()).hexdigest()[:16]
    d = os.path.join(os.environ.get("TMPDIR", tempfile.gettempdir()), f"kf_fused_indexer_{h}")
    os.makedirs(d, exist_ok=True)
    for name, s in (("dual_gemv_bf16", SRC_DUAL_GEMV_BF16), ("qk_rope_hadamard_quant", SRC_QK_ROPE_HADAMARD_QUANT),
                    ("paged_mqa_logits", SRC_PAGED_MQA_LOGITS), ("topk_transform", SRC_TOPK_TRANSFORM)):
        p = os.path.join(d, f"{name}.cuh")
        if not os.path.exists(p):
            tmp = f"{p}.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(s)
            os.replace(tmp, p)
    return d, h


@cache
def modules():
    """(gemv, qk, logits, topk), built like sglang kernels/ops/attention/dsa/hip_gfx950/loader.py."""
    from sglang.kernels.jit.utils import load_jit

    d, h = _src_dir()
    gemv = load_jit("kf_fi_dual_gemv", h, cuda_files=[f"{d}/dual_gemv_bf16.cuh"],
                    cuda_wrappers=[("dual_gemv_bf16", "DualGemvBf16Kernel::run")], extra_cuda_cflags=["-O3"])
    qk = load_jit("kf_fi_qk", h, cuda_files=[f"{d}/qk_rope_hadamard_quant.cuh"],
                  cuda_wrappers=[("indexer_qk_rope_hadamard_quant_and_cache", "QkRopeHadamardQuantKernel::run")],
                  extra_cuda_cflags=["-O3", "-Wno-unused-result"], extra_dependencies=["aiter"])
    logits = load_jit("kf_fi_logits", h, cuda_files=[f"{d}/paged_mqa_logits.cuh"],
                      cuda_wrappers=[("logits_hist", "PagedMqaLogitsKernel::run")],
                      extra_cuda_cflags=["-O3", "-fno-honor-nans", "-mllvm", "-amdgpu-mfma-vgpr-form", "-mllvm",
                                         "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false",
                                         "-Wno-unused-result"])
    topk = load_jit("kf_fi_topk", h, f"hist_bits{LOGITS_HIST_BITS}", cuda_files=[f"{d}/topk_transform.cuh"],
                    cuda_wrappers=[("topk_transform", "TopKTransformKernel::run"), ("hist_stride", "TopKHistStride::run")],
                    extra_cuda_cflags=["-O3", f"-DDSA_TOPK_HIST_BITS={LOGITS_HIST_BITS}"])
    return gemv, qk, logits, topk


class Workspace:
    """Persistent scratch (gfx950_fused_indexer._Workspace): allocated once, outside graph capture."""

    def __init__(self, *, device, max_cols: int, fp8_dtype=torch.float8_e4m3fn):
        rows = MAX_ROWS
        cap = max(INDEX_TOPK, max_cols)
        gh_stride = modules()[3].hist_stride()
        self.max_cols = max_cols
        self.logits = torch.empty((rows, max_cols), dtype=torch.float32, device=device)
        self.q_fp8 = torch.empty((rows, N_HEADS, HEAD_DIM), dtype=fp8_dtype, device=device)
        self.head_gate = torch.empty((rows, N_HEADS), dtype=torch.float32, device=device)
        self.q_proj = torch.empty((rows, N_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=device)
        self.kw = torch.empty((rows, HEAD_DIM + N_HEADS), dtype=torch.bfloat16, device=device)
        self.ghist = torch.zeros((rows, gh_stride), dtype=torch.int32, device=device)
        self.cursor = torch.zeros((rows, 32), dtype=torch.int32, device=device)
        self.cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=device)
        self.cand_val = torch.empty((rows, cap), dtype=torch.float32, device=device)


def fused_decode_indexer(
    *,
    ws: Workspace,
    q_lora: torch.Tensor,          # [rows, 2048] bf16
    x_bf16: torch.Tensor,          # [rows, 6144] bf16 hidden states
    w_q_b: torch.Tensor,           # [4096, 2048] bf16 (indexer wq_b)
    w_kw: torch.Tensor,            # [160, 6144] bf16 ([wk ; weights_proj])
    norm_w: torch.Tensor,          # [128] fp32 k LayerNorm weight
    norm_b: torch.Tensor,          # [128] fp32 k LayerNorm bias
    cos2d: torch.Tensor,           # [max_pos, 32] bf16
    sin2d: torch.Tensor,           # [max_pos, 32] bf16
    positions: torch.Tensor,       # [rows] int64
    out_cache_loc: torch.Tensor,   # [rows] int64 slots this step's K is written to
    kv_cache: torch.Tensor,        # [num_pages, 64, 132] fp8 view of the index-K cache (written)
    kv_cache_read: torch.Tensor,   # same cache, read view [num_pages, 64 * 132] fp8
    seqlens: torch.Tensor,         # [rows] int32 (logits length = top-k row end)
    page_table_64: torch.Tensor,   # [rows, W] int32 page ids (row-expanded)
    eps: float,
    weights_scale: float,
) -> torch.Tensor:
    """[rows, 2048] int32 physical slots of the top-2048 index scores (-1 padded). Also writes this step's K."""
    gemv, qk, logits_mod, topk_mod = modules()
    rows = q_lora.shape[0]
    cols = page_table_64.shape[1] * PAGE_SIZE
    q_fp8 = ws.q_fp8[:rows]
    head_gate = ws.head_gate[:rows]
    logits = ws.logits[:rows, :cols]
    q_proj = ws.q_proj[:rows]
    kw = ws.kw[:rows]
    gemv.dual_gemv_bf16(q_lora, w_q_b, q_proj, x_bf16, w_kw, kw)
    k, weights = kw[:, :HEAD_DIM], kw[:, HEAD_DIM:]
    qk.indexer_qk_rope_hadamard_quant_and_cache(
        q_proj.view(rows, N_HEADS, HEAD_DIM), q_fp8, weights, head_gate, k, kv_cache, out_cache_loc,
        norm_w, norm_b, positions, cos2d, sin2d, float(eps), float(weights_scale))
    out = torch.empty((rows, INDEX_TOPK), dtype=torch.int32, device=logits.device)
    logits_mod.logits_hist(q_fp8, kv_cache_read, head_gate, seqlens, page_table_64, logits, ws.ghist[:rows],
                           logits_bpr(rows, page_table_64.shape[1]))
    topk_mod.topk_transform(logits, seqlens, page_table_64, out, ws.ghist[:rows], ws.cursor[:rows],
                            ws.cand_idx[:rows], ws.cand_val[:rows], TOPK_G, PAGE_SIZE)
    return out
