"""Back vLLM's UVA buffers (v1/worker/gpu/buffer_utils.UvaBuffer) with HIP VMM host memory instead of pinned memory.

UvaBuffer keeps a pinned host tensor (torch.zeros(..., pin_memory=True) -> hipHostMalloc) that GPU kernels read and
write directly. hipHostMalloc memory is KFD userptr (EXPERIMENT 4.11, kfd_migrate_probe.py): when the kernel migrates
or write-protects any page of it, KFD evicts every GPU queue of the worker and revalidates the whole range. The largest
one, RequestState.all_token_ids (max_num_reqs x max_model_len int32, 4 GiB at max_model_len 1M), made the engine
workers stall 17-40 s each when another job on the node drove compaction (R13, 16:11). HIP VMM host memory is TTM pages
that compaction never touches; vLLM's Engram tables already live there (VLLM_ROCM_ENGRAM_HIP_VMM). UvaBuffer already
keeps separate CPU (.cpu, .np) and GPU (._uva) views, so this swaps only where they point: the GPU view at the
hipMemMap'd device address, the CPU view at an mmap of the exported dmabuf FDs (the Engram helpers do both).

Loaded in every Python process through uva_vmm.pth; does nothing unless VLLM_UVA_VMM=1. Buffers smaller than
VLLM_UVA_VMM_MIN_BYTES (0: all) keep the original allocation; any failure falls back to it with one warning.
"""

import os
import sys

_ON = os.environ.get("VLLM_UVA_VMM") == "1"
_MIN = int(os.environ.get("VLLM_UVA_VMM_MIN_BYTES", "0"))
_TARGET = "vllm.v1.worker.gpu.buffer_utils"
_stats = {"vmm": 0, "vmm_bytes": 0, "kept": 0, "failed": 0}


def _vmm_views(num_bytes):
    """CPU and GPU uint8 views of num_bytes of zeroed HIP VMM host memory on this GPU's NUMA node."""
    from vllm.models.deepseek_v41.amd.ops import engram_host_table as eht

    handles, fds, padded = eht._create_table(num_bytes)
    try:
        cpu, gpu = eht._map_table(eht.ExportedHostTable(padded, ""), num_bytes, fds, handles)
    finally:
        # The mappings keep the allocations alive (as share_host_table does).
        eht._close(fds)
        eht._release(handles)
    cpu.zero_()
    return cpu, gpu


def _patch(m):
    import torch

    orig_init = m.UvaBuffer.__init__

    def __init__(self, size, dtype):
        shape = (size,) if isinstance(size, int) else tuple(size)
        num_bytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        if num_bytes < max(_MIN, 1):
            _stats["kept"] += 1
            return orig_init(self, size, dtype)
        try:
            cpu, gpu = _vmm_views(num_bytes)
            self.cpu = cpu.view(dtype).view(shape)
            self.np = self.cpu.numpy()
            self._uva = gpu.view(dtype).view(shape)
            _stats["vmm"] += 1
            _stats["vmm_bytes"] += num_bytes
            if num_bytes >= 1 << 26 or _stats["vmm"] == 1:
                sys.stderr.write(f"[uva_vmm] pid {os.getpid()}: UvaBuffer {shape} {dtype} ({num_bytes >> 20} MiB) "
                                 f"in HIP VMM host memory; {_stats}\n")
        except Exception as e:  # noqa: BLE001
            _stats["failed"] += 1
            if _stats["failed"] == 1:
                sys.stderr.write(f"[uva_vmm] pid {os.getpid()}: HIP VMM allocation failed ({e!r}); "
                                 f"falling back to pinned memory\n")
            orig_init(self, size, dtype)

    m.UvaBuffer.__init__ = __init__
    sys.stderr.write(f"[uva_vmm] patched {_TARGET}.UvaBuffer in pid {os.getpid()} (min {_MIN} bytes)\n")


if _ON:
    import importlib.abc

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            orig_exec_module = loader.exec_module

            def exec_module(module):
                orig_exec_module(module)
                try:
                    _patch(module)
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f"[uva_vmm] patch failed: {e!r}\n")

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
