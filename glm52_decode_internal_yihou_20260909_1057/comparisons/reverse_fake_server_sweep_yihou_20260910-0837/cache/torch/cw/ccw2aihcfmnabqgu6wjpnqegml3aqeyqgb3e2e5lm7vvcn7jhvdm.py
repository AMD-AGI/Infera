# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /torch-cache/mu/cmurqao7fuekp35ig5fe3kvwmvh4nuefvm6cj6w75fzrbfw4bqdk.py
# Topologically Sorted Source Nodes: [weights, unsqueeze, mul_1], Original ATen: [aten.mul, aten.unsqueeze]
# Source node to ATen node mapping:
#   mul_1 => mul_7
#   unsqueeze => unsqueeze
#   weights => mul_2
# Graph fragment:
#   %gemm_a16w16_default : Tensor "bf16[s77, 32][32, 1]cuda:2" = PlaceHolder[target=gemm_a16w16_default]
#   %arg4_1 : Tensor "f32[s77, 32, 1][32, 1, 1]cuda:2" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg5_1]
#   %mul_2 : Tensor "bf16[s77, 32][32, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gemm_a16w16_default, 0.1767766952966369), kwargs = {})
#   %unsqueeze : Tensor "bf16[s77, 32, 1][32, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_2, -1), kwargs = {})
#   %mul_7 : Tensor "f32[s77, 32, 1][32, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %arg4_1), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[s77, 32, 1][32, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %convert_element_type_default_1), kwargs = {})
#   return %mul_tensor
triton_poi_fused_mul_unsqueeze_0 = async_compile.triton('triton_poi_fused_mul_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=2, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = in_ptr2
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        s77 = arg0_1
        s27 = arg1_1
        assert_size_stride(arg2_1, (s77, s27), (s27, 1))
        assert_size_stride(arg3_1, (32, 6144), (6144, 1))
        assert_size_stride(arg4_1, (s77, 32, 1), (32, 1, 1))
        assert_size_stride(arg5_1, (), ())
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            # Topologically Sorted Source Nodes: [result], Original ATen: [aiter.gemm_a16w16]
            buf0 = torch.ops.aiter.gemm_a16w16.default(arg2_1, arg3_1, None, torch.bfloat16, scale_a=None, scale_b=None, scale_c=None)
            del arg2_1
            del arg3_1
            buf3 = buf0
            assert_size_stride(buf3, (s77, 32), (32, 1), 'torch.ops.aiter.gemm_a16w16.default')
            assert_alignment(buf3, 16, 'torch.ops.aiter.gemm_a16w16.default')
            del buf0
            buf4 = empty_strided_cuda((s77, 32, 1), (32, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [weights, unsqueeze, mul_1], Original ATen: [aten.mul, aten.unsqueeze]
            triton_poi_fused_mul_unsqueeze_0_xnumel = 32*s77
            stream2 = get_raw_stream(2)
            triton_poi_fused_mul_unsqueeze_0.run(buf3, arg4_1, arg5_1.item(), buf4, triton_poi_fused_mul_unsqueeze_0_xnumel, stream=stream2)
            del arg4_1
            del arg5_1
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 96
    arg1_1 = 6144
    arg2_1 = rand_strided((96, 6144), (6144, 1), device='cuda:2', dtype=torch.bfloat16)
    arg3_1 = rand_strided((32, 6144), (6144, 1), device='cuda:2', dtype=torch.bfloat16)
    arg4_1 = rand_strided((96, 32, 1), (32, 1, 1), device='cuda:2', dtype=torch.float32)
    arg5_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
