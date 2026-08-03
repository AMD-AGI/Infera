#!/usr/bin/env python3
"""Fix the hicache host-pool allocator on ROCm: use hipHostMalloc, not
mmap + hipHostRegister.

THE BUG
-------
SGLang allocates every hierarchical-cache host pool through

    ALLOC_MEMORY_FUNCS[device] -> alloc_with_host_register(...)
        buffer = alloc_mmap(dims, dtype)          # anonymous mmap
        cudaHostRegister(buffer.data_ptr(), n, 0) # page-lock it

and then stores the resulting **host** virtual addresses in a device-side
pointer table that a GPU kernel dereferences:

    memory_pool_host.py :: DSAIndexerPoolHost.init_kv_buffer
        self.index_k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.index_k_data_refs],  # HOST VAs
            dtype=torch.uint64, device=<gpu>)
    ... -> transfer_kv_all_layer_mla(dst_layers=self.index_k_data_ptrs, ...)

On CUDA that is fine: cudaHostRegister maps the pages at the *same* address in
the device address space, so a host VA is directly dereferenceable from a kernel.

On ROCm it is not. hipHostRegister maps the pages at a **different** device
address, which you must obtain with hipHostGetDevicePointer. Measured on this
stack (MI355X / gfx950 / ROCm 7.2.0), 8 MiB buffer:

    [pin_memory]            host=0x7d7c2f600000  devPtr=0x7d7c2f600000  same=True
    [mmap + hipHostRegister]host=0x7d3bee790000  devPtr=0x7d3bede00000  same=False
    [+hipHostRegisterMapped]host=0x7d3bed600000  devPtr=0x7d3becc00000  same=False
    [+Portable|Mapped]      host=0x7d3bec400000  devPtr=0x7d3a97600000  same=False
    [MAP_PRIVATE]           host=0x7d3a96e00000  devPtr=0x7d3a96400000  same=False

So the kernel is handed an address that is not mapped on the device, and the
first write-back aborts the process with

    Memory access fault by GPU node-2 (Agent handle: ...) on address <host VA>.
    Reason: Unknown.

The fault address equals the host pointer exactly, which is the fingerprint.
gfx950 reports `xnack-`, so there is no page-migration path to paper over it.

THE FIX
-------
Register ROCm in ALLOC_MEMORY_FUNCS to use `alloc_with_pin_memory`, i.e.
torch's `pin_memory=True` (hipHostMalloc underneath), which returns memory whose
device pointer *is* the host pointer. Exactly what the "npu" and "musa" entries
already do for the same reason.

This is the smallest change that makes the existing pointer-table design correct
on ROCm. The alternative -- translate every host pointer through
hipHostGetDevicePointer before building the tables -- touches ~10 call sites in
memory_pool_host.py and is the right upstream shape, but it is not a patch to
apply blind to a running container.

Cost: the hugepage path in alloc_mmap (SGLANG_HUGEPAGE_SIZE) is bypassed on
ROCm. That env var is unset here.

VERIFIED
--------
Standalone repro of the exact kernel (`micro_writeback.py`), 78 layers, 7.33 GB
host indexer buffer, page ranges head / tail / last:
  * mmap+hipHostRegister -> Memory access fault, every variant
  * pin_memory           -> ALL OK, no fault

Idempotent and self-locating. Run inside the container, then delete stale .pyc.
"""
import importlib.util
import os
import re
import sys

MARKER = "GLM52_ROCM_HOST_ALLOC"


def find_common_py() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("cannot locate the sglang package")
    root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, "srt", "mem_cache", "pool_host", "common.py")
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}")
    return path


OLD_IMPORT = "from sglang.srt.mem_cache.mmap_allocator import alloc_mmap"
NEW_IMPORT = (
    "from sglang.srt.mem_cache.mmap_allocator import alloc_mmap\n"
    f"from sglang.srt.utils import is_hip  # {MARKER}"
)

OLD = """ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)"""

NEW = '''# {marker}: on ROCm, hipHostRegister maps host pages at a DIFFERENT device
# address than the host VA (hipHostGetDevicePointer != data_ptr). The hicache
# pools hand raw host data_ptr()s to GPU kernels via device-side pointer tables
# (see DSAIndexerPoolHost.init_kv_buffer -> transfer_kv_all_layer_mla), so those
# kernels dereference an address that is not mapped on the device and abort with
# "Memory access fault by GPU node-2 ... on address <host VA>".
#
# Measured on MI355X/gfx950/ROCm 7.2.0 (xnack-), 8 MiB:
#   pin_memory              devPtr == hostPtr   -> kernel OK
#   mmap + hipHostRegister  devPtr != hostPtr   -> fault (all flag combos,
#                                                  MAP_SHARED and MAP_PRIVATE)
#
# hipHostMalloc (torch pin_memory=True) returns memory whose device pointer IS
# the host pointer, which is what the pointer-table design requires. This is the
# same reason "npu" and "musa" are already routed here.
#
# NOTE the string constant below is deliberate: it is a real module-level
# literal (not a comment) so that `strings <common.cpython-*.pyc> | grep` can
# prove the patch reached the BYTECODE. A stale __pycache__ entry silently
# reverting a patch has already invalidated one full experiment on this stack.
{marker} = "applied"

_ALLOC_MEMORY_FUNCS_OVERRIDES = {{
    "npu": alloc_with_pin_memory,
    "musa": alloc_with_pin_memory,
}}
if is_hip():
    _ALLOC_MEMORY_FUNCS_OVERRIDES["cuda"] = alloc_with_pin_memory

ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_pin_memory if is_hip() else alloc_with_host_register,
    _ALLOC_MEMORY_FUNCS_OVERRIDES,
)'''.format(marker=MARKER)


def main() -> int:
    path = find_common_py()
    src = open(path).read()

    if MARKER in src:
        print(f"[patch] already applied: {path}")
        return 0

    if OLD not in src:
        print("[patch] ERROR: ALLOC_MEMORY_FUNCS block not found in the expected "
              "shape. Refusing to guess -- inspect the file:")
        print(f"        {path}")
        for m in re.finditer(r"ALLOC_MEMORY_FUNCS.*", src):
            print("        |", m.group(0))
        return 1

    # common.py does NOT import is_hip today (it only pulls in alloc_mmap), so
    # the import has to be added alongside the dispatch change.
    if "is_hip" not in src:
        if OLD_IMPORT not in src:
            print("[patch] ERROR: mmap_allocator import line not found; aborting.")
            return 1
        src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)

    open(path, "w").write(src.replace(OLD, NEW))
    print(f"[patch] applied to {path}")

    pyc = os.path.join(os.path.dirname(path), "__pycache__")
    n = 0
    if os.path.isdir(pyc):
        for f in os.listdir(pyc):
            if f.startswith("common."):
                os.remove(os.path.join(pyc, f))
                n += 1
    print(f"[patch] removed {n} stale .pyc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
