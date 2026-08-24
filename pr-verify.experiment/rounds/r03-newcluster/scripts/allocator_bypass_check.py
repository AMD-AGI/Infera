"""Does routing HIP to alloc_with_pin_memory bypass a non-default allocator?

alloc_with_host_register calls allocator.allocate(...); alloc_with_pin_memory
calls torch.empty(..., pin_memory=True) and ignores the allocator argument
entirely. Every pool sets self.allocator = get_allocator_from_storage(...),
which is never None. So on ROCm the patch makes a configured storage allocator
(mooncake / mori / shm) inert.

This measures the concrete consequence for the DEFAULT allocator, which is the
case that matters for whether the PR is safe to land as-is.
"""
import torch
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator, alloc_with_host_register, alloc_with_pin_memory,
)

dims, dtype = (256, 512), torch.uint8
a = HostTensorAllocator()

hr = alloc_with_host_register(dims, dtype, "cpu", True, a)
pm = alloc_with_pin_memory(dims, dtype, "cpu", True, a)

for label, t in (("host_register(default alloc)", hr), ("pin_memory", pm)):
    print(f"  {label:<30} pinned={t.is_pinned()}  nbytes={t.numel()*t.element_size()}")
print()
print("  allocator state after host_register:", a.dims, a.dtype)
b = HostTensorAllocator()
alloc_with_pin_memory(dims, dtype, "cpu", True, b)
print("  allocator state after pin_memory:   ", b.dims, b.dtype, "<- untouched")
