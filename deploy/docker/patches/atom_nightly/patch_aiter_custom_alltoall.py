#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Two kernels on aiter's custom (IPC) all-reduce buffers for the DCP merge.

ATOM's DCP decode merges the partial attention of every rank once per layer
(81 per GLM-5.2 step): pack -> all_to_all_single -> unpack + LSE combine +
per-token FP8 quant. On RCCL the all-to-all alone (5.3 MB per rank at graph
size 40) takes ~40 us in the decode graph, ~3.3 ms per step, and no RCCL
setting helps (NCCL_PROTO / channel counts, measured). aiter's custom
all-gather moves the same bytes per rank (its 1.3 MB input x 4 ranks) in about
half the time, so this adds, next to it:

- ``alltoall_vec``: allgather_vec reading only the chunk addressed to this
  rank: rank r receives chunk r of every rank's input, in rank order (the
  all_to_all_single layout). Same start/end sync, same <= kMaxBlocks grid.
  Python entry ``CustomAllreduce.custom_all_to_all(inp)``, None where the path
  does not apply (gfx1250 bindings, a size the vector kernel cannot split);
  patch_atom_dcp_custom_a2a.py uses it.
- ``alltoall_combine_quant``: the all-to-all, the LSE combine and the FP8 quant
  in one kernel. With alltoall_vec the merge is still pack -> all-to-all ->
  unpack + combine + quant (~6 + 31 + 8.5 us at graph size 40), but the
  receiving rank needs only its own heads of every rank's partials, so this
  reads them straight from the peers' registered buffers: rank r reads chunk r
  of every rank's input (data [N][rows][h_local][head_dim], then fp32 LSE
  [N][rows][h_local]), merges each (row, head) over the N partials with the LSE
  weights, and stores per-token FP8 rows plus one fp32 scale per row; the
  received copy and the unpack launch go away. The arithmetic is that of ATOM's
  _dcp_a2a_unpack_combine_quant_kernel (non-finite LSE -> -inf, NaN partials of
  empty ranks dropped, scale = row max * (1 / FP8 max)). Python entry
  ``CustomAllreduce.supports_a2a_combine_quant`` / ``custom_a2a_combine_quant``;
  patch_atom_dcp_custom_a2a_combine.py packs the split layout only when supported.

``dispatchAllGather`` runs them for gather_dim == -1000 and -2000 - h_local, so
the existing all_gather_reg/unreg bindings, the copy-in pool and the CUDA-graph
buffer registration all apply unchanged; no new pybind symbol. Both are
instantiated for 2, 4 and 8 GPUs, and the Python entries refuse other sizes.

The patched module (module_custom_all_reduce) is rebuilt here, the way the
runtime JIT would; needs no GPU, the target archs come from GPU_ARCHS
(default PYTORCH_ROCM_ARCH). Written against aiter 9205e1059 in
rocm/atom-dev:nightly_202610131505. Strict: see anchor_patch.py.
"""

import importlib.util
import os
import shutil
import sys

from anchor_patch import apply

TAG = "aiter-patch custom-alltoall"

KERNELS = r"""
// infera: all-to-all. size = 16-byte packs per destination chunk; rank r
// receives chunk r of every rank's registered input, in rank order.
constexpr int kAllToAllDim = -1000;

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) alltoall_vec(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size)
{
    constexpr int tnum_gpu  = THREAD_NUM / ngpus;
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    int warp_id             = threadIdx.x / tnum_gpu;
    int lane_id             = threadIdx.x % tnum_gpu;
    int tid                 = blockIdx.x * tnum_gpu + lane_id;
    int stride              = gridDim.x * tnum_gpu;
    const P* ptrs[ngpus];

#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const P*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);

    for(int idx = tid; idx < size; idx += stride)
    {
        *(reinterpret_cast<P*>(&result[0]) + warp_id * size + idx) = ptrs[warp_id][rank * size + idx];
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

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
    // infera: all-to-all, see alltoall_vec.
    if(gather_dim == kAllToAllDim)
    {
        int chunk          = size / d / world_size_;
        int tnum_per_block = 512 / world_size_;
        int block_num      = (chunk + tnum_per_block - 1) / tnum_per_block;
        dim3 grid(std::min(block_num, kMaxBlocks));
        switch(world_size_)
        {
        case 8:
            alltoall_vec<T, 8><<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, chunk);
            break;
        case 4:
            alltoall_vec<T, 4><<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, chunk);
            break;
        case 2:
            alltoall_vec<T, 2><<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, chunk);
            break;
        default: printf("alltoall world_size error\\n");
        }
        return;
    }
"""

PY_METHODS = '''    # infera: dispatchAllGather runs the all-to-all kernel for this "dim".
    _ALL_TO_ALL_DIM = -1000
    # Both infera kernels are instantiated for these world sizes only.
    _A2A_WORLD_SIZES = (2, 4, 8)

    def custom_all_to_all(self, inp: torch.Tensor) -> torch.Tensor | None:
        """all_to_all_single over dim 0 of ``inp``: its world_size equal chunks
        go to ranks 0..N-1 and the result holds every rank's chunk for this
        rank, in rank order. Buffer and capture rules as custom_all_gather;
        None when this path does not apply (the caller falls back to RCCL)."""
        nbytes = inp.numel() * inp.element_size()
        if (
            self._ops_all_gather is not None
            or self.world_size not in self._A2A_WORLD_SIZES
            or self._INT_TO_FP_VIEW.get(inp.dtype, inp.dtype)
            not in (torch.float32, torch.float16, torch.bfloat16)
            or nbytes % (16 * self.world_size) != 0
            or not self.should_custom_ag(inp)
        ):
            return None
        view_dtype = self._INT_TO_FP_VIEW.get(inp.dtype) or inp.dtype
        x = inp.view(view_dtype)
        out = torch.empty_like(x)
        capturing = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        if capturing and self.enable_register_for_capturing:
            ops.all_gather_reg(self._ptr, x, out, self._ALL_TO_ALL_DIM)
        else:
            ops.all_gather_unreg(
                self._ptr,
                x,
                self._pool["input"].data_ptr,
                out,
                self._pool["input"].max_size,
                self._ALL_TO_ALL_DIM,
            )
        return out.view(inp.dtype)

    # infera: dispatchAllGather runs the DCP merge kernel for this "dim" - h_local.
    _A2A_COMBINE_DIM = -2000

    def supports_a2a_combine_quant(
        self, send: torch.Tensor, h_local: int, head_dim: int, quant_dtype
    ) -> bool:
        """Whether custom_a2a_combine_quant runs for this split-layout input
        (see the method); otherwise the caller keeps its RCCL / all-to-all path."""
        return (
            self._ops_all_gather is None
            and self.world_size in self._A2A_WORLD_SIZES
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
        "// infera: all-to-all.",
        [
            (
                "// ========== reduce_scatter kernel start ==========\n",
                KERNELS + "// ========== reduce_scatter kernel start ==========\n",
            ),
            (
                "    // only support gather first dim and gather last dim\n",
                DISPATCH + "    // only support gather first dim and gather last dim\n",
            ),
        ],
    ),
    "dist/device_communicators/custom_all_reduce.py": (
        "# infera: dispatchAllGather runs the all-to-all kernel",
        [("    def fused_ar_rms(\n", PY_METHODS)],
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
