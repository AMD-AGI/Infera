#!/usr/bin/env python3
"""Does migrating pages of a GPU-mapped host buffer make KFD evict this process's queues?

For each way of allocating host memory for the GPU (EXPERIMENT 4.11):
  register  anonymous mmap + hipHostRegister(Mapped)            (UMBP's default pool, Engram before HIP VMM)
  malloc    hipHostMalloc(Portable | Mapped | NonCoherent)        (instr/umbp_pinned_pool.so, MORI UMBP_DRAM_PINNED)
  vmm       hipMemCreate on a host NUMA node, GPU via hipMemMap, CPU via mmap of the dmabuf FD (Engram HIP VMM)
it allocates SIZE_GIB, faults it in from the CPU, reads it from the GPU, then asks the kernel to move PAGES of its
own pages to another NUMA node and back (move_pages(2), no privilege needed for the caller's own pages; a no-op
on a one-node host), then makes the whole range read-only and writable again (mprotect: an MMU invalidation without
migration), and reads
/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms before and after. A control step waits without moving anything.
<pid> is the host PID of the KFD entry that appears when this process opens the GPU. move_pages needs CAP_SYS_NICE
under docker's default seccomp profile; without it those steps report the error and the mprotect step still runs.
One line per step:
  RESULT method=<m> step=<control|move|move-back|mprotect> moved=<n> errors=<errno:count,...> evicted_ms=<delta> gpu_read_s=<t>
usage: kfd_migrate_probe.py [SIZE_GIB=64] [PAGES=4096] [methods=register,malloc,vmm]
"""
import collections
import ctypes
import glob
import mmap
import os
import sys
import time

import torch

SIZE = int(sys.argv[1] if len(sys.argv) > 1 else 64) << 30
PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
METHODS = (sys.argv[3] if len(sys.argv) > 3 else "register,malloc,vmm").split(",")
PAGE = os.sysconf("SC_PAGE_SIZE")
COPY = 64 << 20
CHUNK = 2 << 30

hip = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
# Calls an LD_PRELOADed shim may interpose (instr/umbp_vmm_pool.so) resolve in the global scope, as a C++ caller's do;
# through the library's own handle they would skip the shim.
api = ctypes.CDLL(None)
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.syscall.restype = ctypes.c_long
SYS_move_pages = 279  # x86_64
MPOL_MF_MOVE = 2


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check(rc, what):
    if rc != 0:
        hip.hipGetErrorName.restype = ctypes.c_char_p
        raise RuntimeError(f"{what} failed: {hip.hipGetErrorName(rc).decode()}")


KFD_PID = None  # this process's host PID: the KFD entry that appears when it opens the GPU (no host PID namespace)


def kfd_pids():
    return {os.path.basename(p) for p in glob.glob("/sys/class/kfd/kfd/proc/[0-9]*")}


def evicted_ms():
    total = 0
    for f in glob.glob(f"/sys/class/kfd/kfd/proc/{KFD_PID}/stats_*/evicted_ms"):
        total += int(open(f).read().split()[0])
    return total


def mem_nodes():
    return sorted(int(p.rsplit("node", 1)[1]) for p in glob.glob("/sys/devices/system/node/node[0-9]*")
                  if int(open(p + "/meminfo").read().split("MemTotal:")[1].split()[0]) > 0)


def move_pages(addrs, nodes):
    n = len(addrs)
    pa = (ctypes.c_void_p * n)(*addrs)
    st = (ctypes.c_int * n)()
    nd = (ctypes.c_int * n)(*nodes) if nodes is not None else None
    rc = libc.syscall(SYS_move_pages, 0, ctypes.c_ulong(n), pa, nd, st, MPOL_MF_MOVE if nodes is not None else 0)
    err = ctypes.get_errno() if rc < 0 else 0
    return rc, err, list(st)


class Buf:
    def __init__(self, method):
        self.method, self.host, self.dev, self.extra = method, None, None, []
        if method == "register":
            h = libc.mmap(None, SIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
            assert h not in (None, ctypes.c_void_p(-1).value)
            ctypes.memset(h, 1, SIZE)
            check(api.hipHostRegister(ctypes.c_void_p(h), ctypes.c_size_t(SIZE), ctypes.c_uint(2)), "hipHostRegister")
            self.host = h
        elif method == "malloc":
            p = ctypes.c_void_p()
            check(hip.hipHostMalloc(ctypes.byref(p), ctypes.c_size_t(SIZE), ctypes.c_uint(0x80000003)), "hipHostMalloc")
            self.host = p.value
            ctypes.memset(self.host, 1, SIZE)
        elif method == "vmm":
            self._vmm()
            ctypes.memset(self.host, 1, SIZE)
        if self.dev is None:
            d = ctypes.c_void_p()
            check(api.hipHostGetDevicePointer(ctypes.byref(d), ctypes.c_void_p(self.host), ctypes.c_uint(0)),
                  "hipHostGetDevicePointer")
            self.dev = d.value

    def _vmm(self):
        class Loc(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

        class Prop(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", Loc),
                        ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", ctypes.c_uint32)]

        class Access(ctypes.Structure):
            _fields_ = [("location", Loc), ("flags", ctypes.c_int)]

        prop = Prop(type=1, requestedHandleTypes=1, location=Loc(3, mem_nodes()[0]))
        gran = ctypes.c_size_t()
        check(hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), ctypes.c_int(1)),
              "hipMemGetAllocationGranularity")
        addr = ctypes.c_void_p()
        check(hip.hipMemAddressReserve(ctypes.byref(addr), ctypes.c_size_t(SIZE), ctypes.c_size_t(gran.value), None,
                                       ctypes.c_ulonglong(0)), "hipMemAddressReserve")
        h = libc.mmap(None, SIZE, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)  # CPU address range
        handles, fds = [], []
        for off in range(0, SIZE, CHUNK):  # chunked like Engram's tables
            handle, fd = ctypes.c_void_p(), ctypes.c_int(-1)
            check(hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(CHUNK), ctypes.byref(prop),
                                   ctypes.c_ulonglong(0)), "hipMemCreate")
            check(hip.hipMemExportToShareableHandle(ctypes.byref(fd), handle, ctypes.c_int(1), ctypes.c_ulonglong(0)),
                  "hipMemExportToShareableHandle")
            check(hip.hipMemMap(ctypes.c_void_p(addr.value + off), ctypes.c_size_t(CHUNK), ctypes.c_size_t(0), handle,
                                ctypes.c_ulonglong(0)), "hipMemMap")
            c = libc.mmap(h + off, CHUNK, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | 0x10, fd.value, 0)
            if c != h + off:
                raise OSError(ctypes.get_errno(), "mmap of a dmabuf FD failed")
            handles.append(handle)
            fds.append(fd.value)
        acc = Access(location=Loc(1, torch.cuda.current_device()), flags=3)
        check(hip.hipMemSetAccess(addr, ctypes.c_size_t(SIZE), ctypes.byref(acc), ctypes.c_size_t(1)), "hipMemSetAccess")
        self.host, self.dev, self.extra = h, addr.value, [handles, fds]

    def free(self):
        if self.method == "register":
            api.hipHostUnregister(ctypes.c_void_p(self.host))
            libc.munmap(self.host, SIZE)
        elif self.method == "malloc":
            hip.hipHostFree(ctypes.c_void_p(self.host))
        else:
            handles, fds = self.extra
            libc.munmap(self.host, SIZE)
            hip.hipMemUnmap(ctypes.c_void_p(self.dev), ctypes.c_size_t(SIZE))
            hip.hipMemAddressFree(ctypes.c_void_p(self.dev), ctypes.c_size_t(SIZE))
            for handle in handles:
                hip.hipMemRelease(handle)
            for fd in fds:
                os.close(fd)


def gpu_read(buf, scratch, offsets):
    t = time.monotonic()
    for off in offsets:
        check(api.hipMemcpy(ctypes.c_void_p(scratch.data_ptr()), ctypes.c_void_p(buf.dev + off),
                            ctypes.c_size_t(COPY), ctypes.c_int(3)), "hipMemcpy D2D")
    check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
    return time.monotonic() - t


def main():
    global KFD_PID
    before = kfd_pids()
    torch.cuda.init()
    torch.ones(1, device="cuda").add_(1)  # this process now has KFD queues
    torch.cuda.synchronize()
    new = kfd_pids() - before
    if len(new) != 1:
        raise SystemExit(f"cannot tell this process's KFD entry: new entries {sorted(new)}")
    KFD_PID = new.pop()
    scratch = torch.empty(COPY, dtype=torch.uint8, device="cuda")
    nodes = mem_nodes()
    say(f"pid {os.getpid()} (host {KFD_PID}) size {SIZE >> 30} GiB pages {PAGES} memory nodes {nodes} "
        f"kfd stats files {len(glob.glob(f'/sys/class/kfd/kfd/proc/{KFD_PID}/stats_*/evicted_ms'))}")
    stride = max(PAGE, (SIZE // PAGES) // PAGE * PAGE)
    offsets = [i * (SIZE // 8) // COPY * COPY for i in range(8)]
    for method in METHODS:
        t = time.monotonic()
        try:
            buf = Buf(method)
        except Exception as e:  # report and go on with the other methods
            say(f"RESULT method={method} step=alloc error={e!r}")
            continue
        say(f"{method}: allocated and faulted in {SIZE >> 30} GiB in {time.monotonic() - t:.1f} s; "
            f"first GPU read {gpu_read(buf, scratch, offsets):.2f} s")
        addrs = [buf.host + i * stride for i in range(PAGES)]
        rc, err, where = move_pages(addrs, None)
        here = collections.Counter(where).most_common(3)
        src = where[0] if where and where[0] >= 0 else nodes[0]
        dst = next((n for n in nodes if n != src), src)
        for step in ("control", "move", "move-back", "mprotect"):
            e0 = evicted_ms()
            if step == "control":
                time.sleep(3)
                moved, errs = 0, {}
            elif step == "mprotect":  # read-only and back: an MMU invalidation without moving a page
                errs = collections.Counter()
                for a in (mmap.PROT_READ, mmap.PROT_READ | mmap.PROT_WRITE):
                    if libc.mprotect(ctypes.c_void_p(buf.host), ctypes.c_size_t(SIZE), a) != 0:
                        errs[f"mprotect{ctypes.get_errno()}"] += 1
                moved = 0
                time.sleep(1)
            else:
                target = dst if step == "move" else src
                rc, err, st = move_pages(addrs, [target] * len(addrs))
                moved = sum(1 for s in st if s == target)
                errs = collections.Counter(s for s in st if s < 0)
                if rc < 0:
                    errs[f"rc{-err}"] += 1
                time.sleep(1)
            r = gpu_read(buf, scratch, offsets)
            time.sleep(2)
            say(f"RESULT method={method} step={step} placement_before={here} moved={moved} "
                f"errors={dict(errs)} evicted_ms={evicted_ms() - e0} gpu_read_s={r:.2f}")
        buf.free()


if __name__ == "__main__":
    main()
