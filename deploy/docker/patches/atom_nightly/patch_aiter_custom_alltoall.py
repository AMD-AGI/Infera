#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""An all-to-all on aiter's custom (IPC) all-reduce buffers, for the DCP merge.

ATOM's DCP decode merges the partial attention of every rank with one
all_to_all_single per layer (81 per GLM-5.2 step, 5.3 MB per rank at graph
size 40), on RCCL: ~40 us each in the decode graph, ~3.3 ms per step, and no
RCCL setting helps (NCCL_PROTO / channel counts, measured). aiter's custom
all-gather moves the same bytes per rank (its 1.3 MB input x 4 ranks) in about
half the time, so this adds the all-to-all next to it:

- ``alltoall_vec``: allgather_vec reading only the chunk addressed to this
  rank: rank r receives chunk r of every rank's input, in rank order (the
  all_to_all_single layout). Same start/end sync, same <= kMaxBlocks grid.
- ``dispatchAllGather`` runs it for gather_dim == -1000, so the existing
  all_gather_reg/unreg bindings, the copy-in pool and the CUDA-graph buffer
  registration all apply unchanged; no new pybind symbol.
- ``CustomAllreduce.custom_all_to_all(inp)``: the Python entry, None where the
  path does not apply (gfx1250 bindings, a size the vector kernel cannot split).

The patched module (module_custom_all_reduce) is rebuilt here, the way the
runtime JIT would; needs no GPU, the target archs come from GPU_ARCHS
(default PYTORCH_ROCM_ARCH). Written against aiter 9205e1059 in
rocm/atom-dev:nightly_202610131505. Strict: see anchor_patch.py.
"""

import importlib.util
import os
import sys

from anchor_patch import apply

TAG = "aiter-patch custom-alltoall"

KERNEL = r"""
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

"""

DISPATCH = """    // infera: all-to-all, see alltoall_vec.
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

PY_METHOD = '''    # infera: dispatchAllGather runs the all-to-all kernel for this "dim".
    _ALL_TO_ALL_DIM = -1000

    def custom_all_to_all(self, inp: torch.Tensor) -> torch.Tensor | None:
        """all_to_all_single over dim 0 of ``inp``: its world_size equal chunks
        go to ranks 0..N-1 and the result holds every rank's chunk for this
        rank, in rank order. Buffer and capture rules as custom_all_gather;
        None when this path does not apply (the caller falls back to RCCL)."""
        nbytes = inp.numel() * inp.element_size()
        if (
            self._ops_all_gather is not None
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

    def fused_ar_rms(
'''

PATCH_CSRC = {
    "../csrc/include/custom_all_reduce.cuh": (
        "// infera: all-to-all.",
        [
            (
                "// ========== reduce_scatter kernel start ==========\n",
                KERNEL + "// ========== reduce_scatter kernel start ==========\n",
            ),
            (
                "    // only support gather first dim and gather last dim\n",
                DISPATCH + "    // only support gather first dim and gather last dim\n",
            ),
        ],
    ),
    "dist/device_communicators/custom_all_reduce.py": (
        "# infera: dispatchAllGather runs the all-to-all kernel",
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
