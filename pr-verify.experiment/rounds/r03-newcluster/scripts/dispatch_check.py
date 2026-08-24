"""Which allocator does ALLOC_MEMORY_FUNCS actually hand back on this machine?

The repro exercises the two strategies directly. This checks the *dispatch* the
PR is really about: the pools key the table with a torch.device object, and
torch.device('cuda:0') is not dict-key-equal to 'cuda'.
"""
import torch
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS, alloc_with_host_register, alloc_with_pin_memory,
)
name = {id(alloc_with_host_register): "alloc_with_host_register",
        id(alloc_with_pin_memory): "alloc_with_pin_memory"}
for key in ["cuda", torch.device("cuda:0"), torch.device("cuda")]:
    f = ALLOC_MEMORY_FUNCS[key]
    print(f"  key={str(key):<16} -> {name.get(id(f), getattr(f,'__name__','?'))}")
