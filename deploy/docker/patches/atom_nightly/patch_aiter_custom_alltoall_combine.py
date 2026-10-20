#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The DCP attention merge as one kernel: all-to-all, LSE combine, FP8 quant.

With patch_aiter_custom_alltoall.py, ATOM's DCP decode merge per layer is
pack -> custom all-to-all -> unpack + LSE combine + per-token FP8 quant (three
launches, ~6 + 31 + 8.5 us at graph size 40, 81 per GLM-5.2 step). The
receiving rank needs only its own heads of every rank's partials, so the
combine can read them straight from the peers' registered buffers, like the
all-to-all does, and write the quantized o_proj input directly: the received
copy and the unpack launch go away.

- ``alltoall_combine_quant``: rank r reads chunk r of every rank's input
  (data [N][rows][h_local][head_dim], then fp32 LSE [N][rows][h_local]),
  merges each (row, head) over the N partials with the LSE weights, and stores
  per-token FP8 rows plus one fp32 scale per row. The arithmetic is that of
  ATOM's _dcp_a2a_unpack_combine_quant_kernel (non-finite LSE -> -inf, NaN
  partials of empty ranks dropped, scale = row max * (1 / FP8 max)).
- ``dispatchAllGather`` runs it for gather_dim = -2000 - h_local, so the
  registered-buffer and CUDA-graph paths of all_gather_reg / unreg apply as for
  the all-to-all; no new pybind symbol.
- ``CustomAllreduce.supports_a2a_combine_quant`` / ``custom_a2a_combine_quant``:
  the Python entry; the caller packs the split layout only when supported.

Applies on top of patch_aiter_custom_alltoall.py and rebuilds
module_custom_all_reduce as that one does. Written against aiter 9205e1059 in
rocm/atom-dev:nightly_202610131505. Strict: see anchor_patch.py.
"""

import importlib.util
import os
import shutil
import sys

from anchor_patch import apply

TAG = "aiter-patch custom-alltoall-combine"

KERNEL = r"""
// infera: DCP merge in one kernel. Input of every rank: data
// [ngpus][rows][h_local][head_dim] (T), then fp32 LSE [ngpus][rows][h_local].
// Rank r merges chunk r of all ranks: out = fp8 [rows][h_local * head_dim],
// then fp32 scale [rows]. Needs head_dim % pack == 0 and h_local * head_dim
// <= 512 * pack (one pack per thread and row).
constexpr int kA2ACombineDim = -2000; // gather_dim = kA2ACombineDim - h_local

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) alltoall_combine_quant(RankData* _dp,
                                                                 RankSignals sg,
                                                                 Signal* self_sg,
                                                                 uint8_t* __restrict__ out,
                                                                 int rank,
                                                                 int rows,
                                                                 int h_local,
                                                                 int head_dim)
{
    constexpr int pack = 16 / sizeof(T);
    constexpr int warps = 512 / 64;
    using P = typename opus::vector_t<T, pack>;
    __shared__ float warp_max[warps];
    const int row_len        = h_local * head_dim;
    const int64_t data_elems = (int64_t)ngpus * rows * row_len;
    const float fp8_max      = opus::cast<opus::fp32_t>(opus::numeric_limits<opus::fp8_t>::max());
    const int e0             = threadIdx.x * pack;
    const bool active        = e0 < row_len;
    const int h              = e0 / head_dim;
    const T* src[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        src[i] = (const T*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);

    for(int b = blockIdx.x; b < rows; b += gridDim.x)
    {
        const int64_t chunk_row = (int64_t)rank * rows + b;
        float acc[pack];
#pragma unroll
        for(int j = 0; j < pack; ++j)
        {
            acc[j] = 0.f;
        }
        float amax = 0.f;
        if(active)
        {
            P v[ngpus];
            float lse[ngpus];
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                v[s]   = *reinterpret_cast<const P*>(src[s] + chunk_row * row_len + e0);
                lse[s] = reinterpret_cast<const float*>(src[s] + data_elems)[chunk_row * h_local + h];
            }
            // A rank without KV for the row reports lse -inf (or NaN) and o NaN.
            float lse_max = -INFINITY;
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                if(isnan(lse[s]) || lse[s] == INFINITY)
                    lse[s] = -INFINITY;
                lse_max = fmaxf(lse_max, lse[s]);
            }
            if(lse_max == -INFINITY)
                lse_max = 0.f;
            float sum = 0.f;
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                sum += expf(lse[s] - lse_max);
            }
            const float global_lse = logf(sum) + lse_max;
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                const float factor = expf(lse[s] - global_lse);
                if(isnan(factor) || factor == 0.f)
                    continue; // drops the NaN partials of empty ranks
#pragma unroll
                for(int j = 0; j < pack; ++j)
                {
                    acc[j] += upcast_s(v[s][j]) * factor;
                }
            }
#pragma unroll
            for(int j = 0; j < pack; ++j)
            {
                amax = fmaxf(amax, fabsf(acc[j]));
            }
        }
#pragma unroll
        for(int lane_mask = 32; lane_mask > 0; lane_mask >>= 1)
        {
            amax = fmaxf(amax, __shfl_xor(amax, lane_mask, 64));
        }
        if(threadIdx.x % 64 == 0)
            warp_max[threadIdx.x / 64] = amax;
        __syncthreads();
        float row_max = warp_max[0];
#pragma unroll
        for(int w = 1; w < warps; ++w)
        {
            row_max = fmaxf(row_max, warp_max[w]);
        }
        __syncthreads();
        // As the Triton combine: multiply by 1 / FP8 max, zero scale -> zero row.
        const float scale = row_max * (1.0f / fp8_max);
        const float inv   = scale > 0.f ? 1.0f / scale : 0.f;
        if(active)
        {
            opus::vector_t<opus::fp8_t, pack> q;
#pragma unroll
            for(int i = 0; i < pack / 4; ++i)
            {
                opus::fp32x4_t tmp;
#pragma unroll
                for(int j = 0; j < 4; ++j)
                {
                    tmp[j] = fminf(fmaxf(acc[i * 4 + j] * inv, -fp8_max), fp8_max);
                }
                *(reinterpret_cast<opus::fp8x4_t*>(&q) + i) = opus::cast<opus::fp8_t>(tmp);
            }
            *reinterpret_cast<opus::vector_t<opus::fp8_t, pack>*>(out + (int64_t)b * row_len + e0) = q;
        }
        if(threadIdx.x == 0)
            reinterpret_cast<float*>(out + (int64_t)rows * row_len)[b] = scale;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

"""

DISPATCH = """    // infera: DCP merge (all-to-all + LSE combine + FP8 quant), see alltoall_combine_quant.
    if(gather_dim <= kA2ACombineDim)
    {
        int h_local  = kA2ACombineDim - gather_dim;
        int head_dim = last_dim_size - 2; // input viewed as [ngpus * rows * h_local, head_dim + 2]
        int rows     = size / (world_size_ * h_local * last_dim_size);
        dim3 grid(std::min(rows, kMaxBlocks));
        auto* out = reinterpret_cast<uint8_t*>(output);
        switch(world_size_)
        {
        case 8:
            alltoall_combine_quant<T, 8>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, out, rank_, rows, h_local, head_dim);
            break;
        case 4:
            alltoall_combine_quant<T, 4>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, out, rank_, rows, h_local, head_dim);
            break;
        case 2:
            alltoall_combine_quant<T, 2>
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, out, rank_, rows, h_local, head_dim);
            break;
        default: printf("alltoall_combine_quant world_size error\\n");
        }
        return;
    }
"""

PY_METHOD = '''    # infera: dispatchAllGather runs the DCP merge kernel for this "dim" - h_local.
    _A2A_COMBINE_DIM = -2000

    def supports_a2a_combine_quant(
        self, send: torch.Tensor, h_local: int, head_dim: int, quant_dtype
    ) -> bool:
        """Whether custom_a2a_combine_quant runs for this split-layout input
        (see the method); otherwise the caller keeps its RCCL / all-to-all path."""
        return (
            self._ops_all_gather is None
            and send.dtype in (torch.float16, torch.bfloat16)
            and quant_dtype == fp8
            and head_dim % 8 == 0
            and 0 < h_local * head_dim <= 512 * 8
            and send.dim() == 2
            and send.shape[1] == head_dim + 2
            and send.shape[0] % (self.world_size * h_local) == 0
            and self.should_custom_ag(send)
        )

    def custom_a2a_combine_quant(self, send: torch.Tensor, h_local: int, head_dim: int):
        """DCP attention merge of ``send`` = [world_size * rows * h_local,
        head_dim + 2], holding data [world_size][rows][h_local][head_dim] and
        then the fp32 LSE [world_size][rows][h_local] (chunk i is for rank i).
        Returns (fp8 [rows, h_local, head_dim], fp32 scale [rows, 1]): the
        LSE-weighted merge of this rank's chunks, quantized per row. Buffer and
        capture rules as custom_all_gather; check supports_a2a_combine_quant first."""
        rows = send.shape[0] // (self.world_size * h_local)
        row_bytes = h_local * head_dim
        out = torch.empty(rows * (row_bytes + 4), dtype=torch.uint8, device=send.device)
        dim = self._A2A_COMBINE_DIM - h_local
        capturing = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        if capturing and self.enable_register_for_capturing:
            ops.all_gather_reg(self._ptr, send, out, dim)
        else:
            ops.all_gather_unreg(
                self._ptr,
                send,
                self._pool["input"].data_ptr,
                out,
                self._pool["input"].max_size,
                dim,
            )
        q = out[: rows * row_bytes].view(fp8).view(rows, h_local, head_dim)
        scale = out[rows * row_bytes :].view(torch.float32).view(rows, 1)
        return q, scale

    def fused_ar_rms(
'''

PATCH_CSRC = {
    "../csrc/include/custom_all_reduce.cuh": (
        "// infera: DCP merge in one kernel.",
        [
            (
                "// ========== reduce_scatter kernel start ==========\n",
                KERNEL + "// ========== reduce_scatter kernel start ==========\n",
            ),
            (
                "    // infera: all-to-all, see alltoall_vec.\n",
                DISPATCH + "    // infera: all-to-all, see alltoall_vec.\n",
            ),
        ],
    ),
    "dist/device_communicators/custom_all_reduce.py": (
        "# infera: dispatchAllGather runs the DCP merge kernel",
        [("    def fused_ar_rms(\n", PY_METHOD)],
    ),
}

MODULE = "module_custom_all_reduce"


def rebuild() -> int:
    """Rebuild the module from the patched sources, as the runtime JIT would."""
    root = importlib.util.find_spec("aiter").submodule_search_locations[0]
    os.environ.setdefault("GPU_ARCHS", os.environ.get("PYTORCH_ROCM_ARCH", "gfx950"))
    sys.path.insert(0, root)
    from jit import core  # noqa: E402  (import after GPU_ARCHS is set)

    so = f"{core.get_user_jit_dir()}/{MODULE}.so"
    cuh = os.path.join(root, "../csrc/include/custom_all_reduce.cuh")
    if os.path.exists(so):
        if os.path.getmtime(so) > os.path.getmtime(cuh):
            print(f"[{TAG}] {so} is newer than the patched header; not rebuilt")
            return 0
        os.remove(so)
    # The JIT compiles from a copy of the sources in its build dir; an image
    # that already built the module would recompile the old copy.
    shutil.rmtree(f"{core.get_user_jit_dir()}/build/{MODULE}", ignore_errors=True)
    args = core.get_args_of_build(MODULE)
    core.build_module(
        MODULE,
        args["srcs"],
        args["flags_extra_cc"],
        args["flags_extra_hip"],
        args["blob_gen_cmd"],
        args["extra_include"],
        args["extra_ldflags"],
        args["verbose"],
        args["is_python_module"],
        args["is_standalone"],
        args["torch_exclude"],
        args.get("third_party", []),
        args.get("hipify", False),
        flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
    )
    if not os.path.isfile(so):
        print(f"[{TAG}] FAILED: AITER build left no {so}")
        return 1
    print(f"[{TAG}] rebuilt {so}")
    return 0


if __name__ == "__main__":
    rc = apply(TAG, PATCH_CSRC, package="aiter")
    if rc == 0 and "--no-build" not in sys.argv:
        rc = rebuild()
    sys.exit(rc)
