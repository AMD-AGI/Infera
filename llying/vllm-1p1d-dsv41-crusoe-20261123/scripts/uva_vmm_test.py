#!/usr/bin/env python3
"""GPU test of instr/uva_vmm.py in the engine image (run with VLLM_UVA_VMM=1 and instr/ on sys.path).

A UvaBuffer and a StagedWriteTensor(uva_instead_of_gpu=True) must land in HIP VMM host memory, CPU writes must be
visible to the GPU and GPU writes to the CPU, and making the CPU range read-only and back (an MMU invalidation, as in
kfd_migrate_probe.py) must not evict this process's queues. Prints PASS / FAIL lines.
"""
import ctypes
import glob
import os
import sys

import torch

import uva_vmm  # noqa: F401  (installs the import hook)


def kfd_pids():
    return {os.path.basename(p) for p in glob.glob("/sys/class/kfd/kfd/proc/[0-9]*")}


def evicted_ms(pid):
    return sum(int(open(f).read().split()[0]) for f in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/stats_*/evicted_ms"))


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    return ok


def main():
    before = kfd_pids()
    torch.cuda.set_device(0)
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    new = kfd_pids() - before
    pid = new.pop() if len(new) == 1 else None

    from vllm.v1.worker.gpu import buffer_utils as bu

    ok = check("UvaBuffer patched", bu.UvaBuffer.__init__.__module__ == "uva_vmm")
    b = bu.UvaBuffer((64, 1 << 20), torch.int32)  # 256 MiB
    ok &= check("UVA view on the GPU", b._uva.is_cuda and b._uva.shape == (64, 1 << 20))
    ok &= check("CPU view is not pinned memory (not hipHostMalloc)", not b.cpu.is_pinned())
    ok &= check("zeroed", int(b._uva.abs().sum()) == 0)
    b.np[3, 5:10] = [1, 2, 3, 4, 5]
    ok &= check("CPU write visible on the GPU", b._uva[3, 5:10].tolist() == [1, 2, 3, 4, 5])
    b._uva[7, 0] = 42
    torch.cuda.synchronize()
    ok &= check("GPU write visible on the CPU", int(b.np[7, 0]) == 42)

    s = bu.StagedWriteTensor((32, 4096), torch.int32, torch.device("cuda", 0), uva_instead_of_gpu=True)
    s.stage_write(2, 10, [7, 8, 9])
    s.apply_write()
    torch.cuda.synchronize()
    ok &= check("StagedWriteTensor write applied", s.gpu[2, 10:13].tolist() == [7, 8, 9])

    if pid:
        libc = ctypes.CDLL(None, use_errno=True)
        e0 = evicted_ms(pid)
        addr, size = b.cpu.data_ptr(), b.cpu.numel() * 4
        for prot in (1, 3):
            libc.mprotect(ctypes.c_void_p(addr & ~4095), ctypes.c_size_t(size), prot)
        b._uva[1, 1] = 5
        torch.cuda.synchronize()
        ok &= check(f"no KFD eviction on mprotect of the CPU range (evicted_ms +{evicted_ms(pid) - e0})",
                    evicted_ms(pid) - e0 == 0)
    else:
        print(f"SKIP eviction check: cannot tell this process's KFD entry ({sorted(new)})")
    print(f"stats {uva_vmm._stats}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
