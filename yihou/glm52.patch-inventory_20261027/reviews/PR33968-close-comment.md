Closing this older allocator workaround as superseded by the merged #35233. I rechecked upstream main `449617bc94a5d4ae23ad62d0da2f842e711ca983`:

- `pool_host/common.py::make_kernel_ptr_table` resolves registered host allocations through `get_device_accessible_ptr` for the target GPU and fails closed on HIP when the API is unavailable.
- The MHA, MLA, and DSA host pools use those kernel pointer tables. CPU/storage consumers retain their original host addresses.
- #35233 includes registered-mmap transfer and graph regressions, so it addresses the address-domain root cause without bypassing custom host allocators.
- The later merged #42553 adds an opt-in GTT allocation path while preserving custom allocators. This PR's unconditional HIP allocator substitution is not needed on current main.

The original diagnosis and historical gfx950 results remain useful context, but are not a reason to merge an obsolete workaround. No new hardware test is claimed by this closure.
