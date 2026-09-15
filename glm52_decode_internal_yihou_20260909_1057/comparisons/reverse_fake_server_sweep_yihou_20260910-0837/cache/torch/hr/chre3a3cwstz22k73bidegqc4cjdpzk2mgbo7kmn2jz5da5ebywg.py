# AOT ID: ['6_inference']
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


# kernel path: /torch-cache/zs/czs6bjdnf7ayndsbnn7dsg2qhub4qqeqms7k5mxv7w4d6h6d5qg2.py
# Topologically Sorted Source Nodes: [unsqueeze, view, expand_scores, flatten, max_1], Original ATen: [aten.unsqueeze, aten.view, aten.mul, aten.max]
# Source node to ATen node mapping:
#   expand_scores => mul_3
#   flatten => view_1
#   max_1 => getitem, max_1
#   unsqueeze => unsqueeze
#   view => view
# Graph fragment:
#   %arg2_1 : Tensor "f32[s0, 1][1, 1]cuda:2" = PlaceHolder[target=arg2_1]
#   %arg4_1 : Tensor "f32[s0, 1][1, 1]cuda:2" = PlaceHolder[target=arg4_1]
#   %mul_3 : Tensor "f32[s0, 1, 1][1, 1, 1]cuda:2" = PlaceHolder[target=mul_3]
#   %unsqueeze : Tensor "f32[s0, 1, 1][1, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, 2), kwargs = {})
#   %view : Tensor "f32[s0, 1, 1][1, 1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg4_1, [-1, 1, 1]), kwargs = {})
#   %mul_3 : Tensor "f32[s0, 1, 1][1, 1, 1]cuda:2"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %view), kwargs = {})
#   %view_1 : Tensor "f32[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [%arg3_1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %getitem : Tensor "f32[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=operator.getitem](args = (%max_1, 0), kwargs = {})
#   return %mul_3,%getitem
triton_poi_fused_max_mul_unsqueeze_view_0 = async_compile.triton('triton_poi_fused_max_mul_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=2, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_mul_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_mul_unsqueeze_view_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /torch-cache/wz/cwz4j5zejo3bv3n6bxyqpbei2flxztqasnxxw5pwyctfblxp5ih2.py
# Topologically Sorted Source Nodes: [flatten, max_1, gather], Original ATen: [aten.view, aten.max, aten.gather]
# Source node to ATen node mapping:
#   flatten => view_1
#   gather => gather
#   max_1 => max_1
# Graph fragment:
#   %arg6_1 : Tensor "i64[s37, 1][1, 1]cuda:2" = PlaceHolder[target=arg6_1]
#   %view_1 : Tensor "f32[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [%arg3_1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %gather : Tensor "i64[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg6_1, 1, %getitem_1), kwargs = {})
#   return %gather
triton_poi_fused_gather_max_view_1 = async_compile.triton('triton_poi_fused_gather_max_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=2, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_max_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gather_max_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /torch-cache/n2/cn2ceoa3rbjbvqpk3mnsmstuw3t4qfgtzn3hxs766ndeukp7x3eh.py
# Topologically Sorted Source Nodes: [flatten, max_1, flat_cs, floordiv, batch_offsets, repeat_interleave, selected_input_index, hidden_states], Original ATen: [aten.view, aten.max, aten.floor_divide, aten.arange, aten.unsqueeze, aten.add, aten.index]
# Source node to ATen node mapping:
#   batch_offsets => iota
#   flat_cs => view_4
#   flatten => view_1
#   floordiv => div
#   hidden_states => index
#   max_1 => max_1
#   repeat_interleave => unsqueeze_1, view_5
#   selected_input_index => add_47
# Graph fragment:
#   %arg8_1 : Tensor "bf16[s0, s87][s87, 1]cuda:2" = PlaceHolder[target=arg8_1]
#   %view_1 : Tensor "f32[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [%arg3_1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %view_4 : Tensor "i64[s0][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg3_1]), kwargs = {})
#   %div : Tensor "i64[s0][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%view_4, 1), kwargs = {rounding_mode: floor})
#   %iota : Tensor "i64[s0][1]cuda:2"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg3_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:2, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 1), kwargs = {})
#   %view_5 : Tensor "i64[s0][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand, [%arg3_1]), kwargs = {})
#   %add_47 : Tensor "i64[s0][1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %view_5), kwargs = {})
#   %index : Tensor "bf16[s0, s87][s87, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg8_1, [%add_47]), kwargs = {})
#   return %index
triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=2, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /torch-cache/4w/c4wludnmp5iep5dy6pofdz5ouvg6fcm4wsvul23wipuhhua57rbe.py
# Topologically Sorted Source Nodes: [flatten, max_1, add_2], Original ATen: [aten.view, aten.max, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_53
#   flatten => view_1
#   max_1 => max_1
# Graph fragment:
#   %view_1 : Tensor "f32[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [%arg3_1, 1]), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_1, -1, True), kwargs = {})
#   %add_53 : Tensor "i64[s0, 1][1, 1]cuda:2"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1, %arg9_1), kwargs = {})
#   return %add_53
triton_poi_fused_add_max_view_3 = async_compile.triton('triton_poi_fused_add_max_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=2, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A26C22AC9188601F861B0885E260980AF55E28349F902C6DD0A908F1E1E0B3B7', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_view_3(out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = ks0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s53 = arg0_1
        s68 = arg1_1
        s0 = arg3_1
        s37 = arg5_1
        s87 = arg7_1
        s77 = arg9_1
        assert_size_stride(arg2_1, (s0, 1), (1, 1))
        assert_size_stride(arg4_1, (s0, 1), (1, 1))
        assert_size_stride(arg6_1, (s37, 1), (1, 1))
        assert_size_stride(arg8_1, (s0, s87), (s87, 1))
        with torch.cuda._DeviceGuard(2):
            torch.cuda.set_device(2)
            buf0 = empty_strided_cuda((s0, 1, 1), (1, 1, 1), torch.float32)
            buf1 = empty_strided_cuda((s0, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze, view, expand_scores, flatten, max_1], Original ATen: [aten.unsqueeze, aten.view, aten.mul, aten.max]
            stream2 = get_raw_stream(2)
            triton_poi_fused_max_mul_unsqueeze_view_0.run(arg2_1, arg4_1, buf0, buf1, s0, stream=stream2)
            del arg2_1
            del arg4_1
            buf2 = empty_strided_cuda((s0, 1), (1, 1), torch.int64)
            # Topologically Sorted Source Nodes: [flatten, max_1, gather], Original ATen: [aten.view, aten.max, aten.gather]
            stream2 = get_raw_stream(2)
            triton_poi_fused_gather_max_view_1.run(arg6_1, buf2, s0, stream=stream2)
            buf3 = empty_strided_cuda((s0, s87), (s87, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flatten, max_1, flat_cs, floordiv, batch_offsets, repeat_interleave, selected_input_index, hidden_states], Original ATen: [aten.view, aten.max, aten.floor_divide, aten.arange, aten.unsqueeze, aten.add, aten.index]
            triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2_xnumel = s0*s87
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2.run(arg8_1, buf3, triton_poi_fused_add_arange_floor_divide_index_max_unsqueeze_view_2_xnumel, stream=stream2)
            del arg8_1
            buf4 = empty_strided_cuda((s0, 1), (1, 1), torch.int64)
            # Topologically Sorted Source Nodes: [flatten, max_1, add_2], Original ATen: [aten.view, aten.max, aten.add]
            stream2 = get_raw_stream(2)
            triton_poi_fused_add_max_view_3.run(buf4, s77, s0, stream=stream2)
        return (reinterpret_tensor(buf2, (s0, ), (1, ), 0), buf3, buf1, buf0, arg6_1, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 1
    arg1_1 = 8
    arg2_1 = rand_strided((8, 1), (1, 1), device='cuda:2', dtype=torch.float32)
    arg3_1 = 8
    arg4_1 = rand_strided((8, 1), (1, 1), device='cuda:2', dtype=torch.float32)
    arg5_1 = 8
    arg6_1 = rand_strided((8, 1), (1, 1), device='cuda:2', dtype=torch.int64)
    arg7_1 = 6144
    arg8_1 = rand_strided((8, 6144), (6144, 1), device='cuda:2', dtype=torch.bfloat16)
    arg9_1 = 1
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
