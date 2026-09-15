# AOT ID: ['5_inference']
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


# kernel path: /torch-cache/q7/cq7dxbjcsre5lsld7hzu3v334pxqysxfpbdejvgh2xzqpuzhthoo.py
# Topologically Sorted Source Nodes: [unsqueeze, view, expand_scores, flatten, max_1], Original ATen: [aten.unsqueeze, aten.view, aten.mul, aten.max]
# Source node to ATen node mapping:
#   expand_scores => mul_1
#   flatten => view_1
#   max_1 => getitem, max_1
#   unsqueeze => unsqueeze
#   view => view
# Graph fragment:
#   %arg1_1 : Tensor "f32[1, 1][1, 1]cuda:1" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1, 1][1, 1]cuda:1" = PlaceHolder[target=arg2_1]
#   %mul_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:1" = PlaceHolder[target=mul_1]
#   %unsqueeze : Tensor "f32[1, 1, 1][1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 2), kwargs = {})
#   %view : Tensor "f32[1, 1, 1][1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [-1, 1, 1]), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %view), kwargs = {})
#   %view_1 : Tensor "f32[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_1, [1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %getitem : Tensor "f32[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=operator.getitem](args = (%max_1, 0), kwargs = {})
#   return %mul_1,%getitem
triton_poi_fused_max_mul_unsqueeze_view_0 = async_compile.triton('triton_poi_fused_max_mul_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=1, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_mul_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_mul_unsqueeze_view_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp4, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /torch-cache/y6/cy6pmutxl63h2c3dkcopzaxadl3fcw4al37w7zavcukwes5jrj63.py
# Topologically Sorted Source Nodes: [flatten, max_1, gather], Original ATen: [aten.view, aten.max, aten.gather]
# Source node to ATen node mapping:
#   flatten => view_1
#   gather => gather
#   max_1 => max_1
# Graph fragment:
#   %arg3_1 : Tensor "i64[1, 1][1, 1]cuda:1" = PlaceHolder[target=arg3_1]
#   %view_1 : Tensor "f32[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_1, [1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %gather : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg3_1, 1, %getitem_1), kwargs = {})
#   return %gather
triton_poi_fused_gather_max_view_1 = async_compile.triton('triton_poi_fused_gather_max_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=1, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_max_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gather_max_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp1, None)
''', device_str='cuda')


# kernel path: /torch-cache/l2/cl2twn2hqae3udfvbgsuh647zb3hmo5y6wcvfshtfdn5jfputnx3.py
# Topologically Sorted Source Nodes: [flatten, max_1, flat_cs, floordiv, batch_offsets, repeat_interleave, selected_input_index, hidden_states], Original ATen: [aten.view, aten.max, aten.floor_divide, aten.arange, aten.unsqueeze, aten.add, aten.index]
# Source node to ATen node mapping:
#   batch_offsets => iota
#   flat_cs => view_4
#   flatten => view_1
#   floordiv => div
#   hidden_states => index
#   max_1 => max_1
#   repeat_interleave => unsqueeze_1, view_5
#   selected_input_index => add
# Graph fragment:
#   %arg5_1 : Tensor "bf16[1, s98][s98, 1]cuda:1" = PlaceHolder[target=arg5_1]
#   %view_1 : Tensor "f32[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_1, [1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %view_4 : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [1]), kwargs = {})
#   %div : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%view_4, 1), kwargs = {rounding_mode: floor})
#   %iota : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 1), kwargs = {})
#   %view_5 : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand, [1]), kwargs = {})
#   %add : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %view_5), kwargs = {})
#   %index : Tensor "bf16[1, s98][s98, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg5_1, [%add]), kwargs = {})
#   return %index
triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=1, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /torch-cache/y4/cy4ycirxbyy7ekgz7ycn32jfawivfa6nud55y2cwzniiofivgfq7.py
# Topologically Sorted Source Nodes: [flatten, max_1, add_2], Original ATen: [aten.view, aten.max, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_3
#   flatten => view_1
#   max_1 => max_1
# Graph fragment:
#   %view_1 : Tensor "f32[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_1, [1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %add_3 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1, %arg6_1), kwargs = {})
#   return %add_3
triton_poi_fused_add_max_view_3 = async_compile.triton('triton_poi_fused_add_max_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'ks0': 'i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=1, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_view_3(out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = ks0
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        s53 = arg0_1
        s98 = arg4_1
        s77 = arg6_1
        assert_size_stride(arg1_1, (1, 1), (1, 1))
        assert_size_stride(arg2_1, (1, 1), (1, 1))
        assert_size_stride(arg3_1, (1, 1), (1, 1))
        assert_size_stride(arg5_1, (1, s98), (s98, 1))
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf0 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
            buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze, view, expand_scores, flatten, max_1], Original ATen: [aten.unsqueeze, aten.view, aten.mul, aten.max]
            stream1 = get_raw_stream(1)
            triton_poi_fused_max_mul_unsqueeze_view_0.run(arg1_1, arg2_1, buf0, buf1, 1, stream=stream1)
            del arg1_1
            del arg2_1
            buf2 = empty_strided_cuda((1, 1), (1, 1), torch.int64)
            # Topologically Sorted Source Nodes: [flatten, max_1, gather], Original ATen: [aten.view, aten.max, aten.gather]
            stream1 = get_raw_stream(1)
            triton_poi_fused_gather_max_view_1.run(arg3_1, buf2, 1, stream=stream1)
            buf3 = empty_strided_cuda((1, s98), (s98, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flatten, max_1, flat_cs, floordiv, batch_offsets, repeat_interleave, selected_input_index, hidden_states], Original ATen: [aten.view, aten.max, aten.floor_divide, aten.arange, aten.unsqueeze, aten.add, aten.index]
            stream1 = get_raw_stream(1)
            triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2.run(arg5_1, buf3, s98, stream=stream1)
            del arg5_1
            buf4 = empty_strided_cuda((1, 1), (1, 1), torch.int64)
            # Topologically Sorted Source Nodes: [flatten, max_1, add_2], Original ATen: [aten.view, aten.max, aten.add]
            stream1 = get_raw_stream(1)
            triton_poi_fused_add_max_view_3.run(buf4, s77, 1, stream=stream1)
        return (reinterpret_tensor(buf2, (1, ), (1, ), 0), buf3, buf1, buf0, arg3_1, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 1
    arg1_1 = rand_strided((1, 1), (1, 1), device='cuda:1', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1), (1, 1), device='cuda:1', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1), (1, 1), device='cuda:1', dtype=torch.int64)
    arg4_1 = 6144
    arg5_1 = rand_strided((1, 6144), (6144, 1), device='cuda:1', dtype=torch.bfloat16)
    arg6_1 = 1
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
