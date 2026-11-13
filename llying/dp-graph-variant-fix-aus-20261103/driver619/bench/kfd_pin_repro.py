#!/usr/bin/env python3
"""DRIVER619-20261113: minimal repro of the amdgpu KFD double count of RDMA-pinned VRAM, and the GEM-backed workaround.

One process, one GPU, no torch / Mooncake: allocate A GiB, register it with ionic RDMA NICs in 1 GiB MRs, then measure
how much more VRAM the driver lets us allocate and whether a new HIP stream can run a memset.

  --alloc kfd   hipMalloc (a KFD BO: counted in kfd.vram_used)
  --alloc gem   DRM_IOCTL_AMDGPU_GEM_CREATE on the GPU's render node, exported as a dma-buf and imported with
                hipImportExternalMemory (KFD imports are not counted in kfd.vram_used), then pinned at once with a
                one-page dma-buf MR so vram_pin_size carries it
  --reg dmabuf  hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr (Mooncake MOONCAKE_DISABLE_HIP_DMABUF=0)
  --reg peermem ibv_reg_mr through the amdgpu peer-memory client (Mooncake MOONCAKE_DISABLE_HIP_DMABUF=1); refused for
                gem: kfd_peerdirect.c amd_get_pages dereferences bo->kfd_bo, which is NULL for a GEM BO
  --reg none

Each step prints one JSON line: reported free (hipMemGetInfo, i.e. KFD available memory), real VRAM use
(mem_info_vram_used), this process's KFD VRAM (/sys/class/kfd/kfd/proc/<pid>/vram_*) and, with --headroom, the largest
hipMalloc the driver accepts (binary search, 256 MiB steps, never above the real free VRAM).
"""
import argparse
import ctypes as C
import fcntl
import glob
import json
import os
import struct
import sys
import time

GiB, MiB = 1 << 30, 1 << 20
p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--gib", type=float, default=40)
p.add_argument("--alloc", choices=["kfd", "gem"], default="kfd")
p.add_argument("--reg", choices=["dmabuf", "peermem", "none"], default="dmabuf")
p.add_argument("--nics", default="ionic_0")
p.add_argument("--chunk-gib", type=float, default=1)
p.add_argument("--headroom", type=int, default=1)
p.add_argument("--tag", default="")
a = p.parse_args()
if a.alloc == "gem" and a.reg == "peermem":
    sys.exit("refused: peer-memory registration of a GEM BO dereferences a NULL kfd_bo in amd_get_pages (kernel oops)")
T0 = time.time()


def out(step, **kw):
    d = dict(tag=a.tag, step=step, t=round(time.time() - T0, 2), gpu=a.gpu, alloc=a.alloc, reg=a.reg, gib=a.gib)
    d.update(kw)
    print(json.dumps(d), flush=True)


def load(name):
    for cand in (name, f"/opt/rocm/lib/{name}"):
        try:
            return C.CDLL(cand, use_errno=True)
        except OSError:
            pass
    sys.exit(f"cannot load {name}")


hip, hsa, ibv = load("libamdhip64.so"), load("libhsa-runtime64.so.1"), load("libibverbs.so.1")
vp, sz = C.c_void_p, C.c_size_t
hip.hipGetErrorString.restype = C.c_char_p
for f, args in [("hipMalloc", [C.POINTER(vp), sz]), ("hipFree", [vp]), ("hipMemGetInfo", [C.POINTER(sz), C.POINTER(sz)]),
                ("hipStreamCreateWithPriority", [C.POINTER(vp), C.c_uint, C.c_int]), ("hipStreamSynchronize", [vp]),
                ("hipMemsetAsync", [vp, C.c_int, sz, vp]), ("hipMemcpy", [vp, vp, sz, C.c_int]),
                ("hipImportExternalMemory", [C.POINTER(vp), vp]), ("hipExternalMemoryGetMappedBuffer", [C.POINTER(vp), vp, vp]),
                ("hipDeviceGetPCIBusId", [C.c_char_p, C.c_int, C.c_int]), ("hipSetDevice", [C.c_int]),
                ("hipDeviceSynchronize", [])]:
    getattr(hip, f).argtypes = args
    getattr(hip, f).restype = C.c_int
hsa.hsa_amd_portable_export_dmabuf.argtypes = [vp, sz, C.POINTER(C.c_int), C.POINTER(C.c_uint64)]
hsa.hsa_amd_portable_export_dmabuf.restype = C.c_int
hsa.hsa_amd_portable_close_dmabuf.argtypes = [C.c_int]
ibv.ibv_get_device_list.restype = C.POINTER(vp)
ibv.ibv_get_device_list.argtypes = [C.POINTER(C.c_int)]
ibv.ibv_get_device_name.restype = C.c_char_p
ibv.ibv_get_device_name.argtypes = [vp]
ibv.ibv_open_device.restype = vp
ibv.ibv_open_device.argtypes = [vp]
ibv.ibv_alloc_pd.restype = vp
ibv.ibv_alloc_pd.argtypes = [vp]
ibv.ibv_reg_mr.restype = vp
ibv.ibv_reg_mr.argtypes = [vp, vp, sz, C.c_int]
ibv.ibv_reg_dmabuf_mr.restype = vp
ibv.ibv_reg_dmabuf_mr.argtypes = [vp, C.c_uint64, sz, C.c_uint64, C.c_int, C.c_int]
ibv.ibv_dereg_mr.argtypes = [vp]
ACCESS = 1 | 2 | 4  # LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ


def herr(rc):
    return f"{rc}:{hip.hipGetErrorString(rc).decode()}" if rc else "ok"


def check(rc, what):
    if rc:
        out("error", what=what, rc=herr(rc))
        sys.exit(1)


check(hip.hipSetDevice(a.gpu), "hipSetDevice")
busid = C.create_string_buffer(64)
check(hip.hipDeviceGetPCIBusId(busid, 64, a.gpu), "hipDeviceGetPCIBusId")
bdf = busid.value.decode().lower()
pci = glob.glob(f"/sys/bus/pci/devices/{bdf}")[0]


def rd(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return -1


def kfd_proc_vram():
    return sum(rd(f) for f in glob.glob(f"/sys/class/kfd/kfd/proc/{os.getpid()}/vram_*"))


def mem():
    free, total = sz(), sz()
    rc = hip.hipMemGetInfo(C.byref(free), C.byref(total))
    used = rd(f"{pci}/mem_info_vram_used")
    vt = rd(f"{pci}/mem_info_vram_total")
    return dict(rep_free_gib=round(free.value / GiB, 2) if rc == 0 else herr(rc), total_gib=round(total.value / GiB, 2),
                real_used_gib=round(used / GiB, 2), real_free_gib=round((vt - used) / GiB, 2),
                proc_kfd_gib=round(kfd_proc_vram() / GiB, 2))


def headroom():
    if not a.headroom:
        return None
    vt, used = rd(f"{pci}/mem_info_vram_total"), rd(f"{pci}/mem_info_vram_used")
    lo, hi = 0, (vt - used) // (256 * MiB) + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        q = vp()
        rc = hip.hipMalloc(C.byref(q), mid * 256 * MiB)
        if rc == 0:
            hip.hipFree(q)
            lo = mid
        else:
            hi = mid - 1
    hip.hipDeviceSynchronize()
    return round(lo * 256 * MiB / GiB, 2)


def stream_probe():
    """New streams (HIP creates hardware queues for the first few) running a memset on a buffer allocated up front."""
    res = []
    for i in range(4):
        s = vp()
        rc = hip.hipStreamCreateWithPriority(C.byref(s), 0, 0)
        if rc:
            res.append(f"create {herr(rc)}")
            break
        rc = hip.hipMemsetAsync(scratch, i, 2 * MiB, s) or hip.hipStreamSynchronize(s)
        res.append(f"memset {herr(rc)}")
        if rc:
            break
    q = vp()
    rc = hip.hipMalloc(C.byref(q), 2 * MiB)
    res.append(f"hipMalloc 2 MiB {herr(rc)}")
    if rc == 0:
        hip.hipFree(q)
    return "; ".join(res)


out("start", bdf=bdf, driver=open("/sys/module/amdgpu/version").read().strip(), **mem())
scratch = vp()
check(hip.hipMalloc(C.byref(scratch), 2 * MiB), "hipMalloc scratch")
h0 = headroom()
out("before", headroom_gib=h0, **mem())

size = int(a.gib * GiB) // (2 * MiB) * (2 * MiB)
ptr, gem_fd = vp(), -1
ctxs = []
num = C.c_int()
devs = ibv.ibv_get_device_list(C.byref(num))
want = a.nics.split(",")
for i in range(num.value):
    name = ibv.ibv_get_device_name(devs[i]).decode()
    if name in want:
        ctx = ibv.ibv_open_device(devs[i])
        pd = ibv.ibv_alloc_pd(ctx) if ctx else None
        if not pd:
            out("error", what=f"open/alloc_pd {name}", errno=C.get_errno())
            sys.exit(1)
        ctxs.append((name, pd))
if a.reg != "none" or a.alloc == "gem":
    if len(ctxs) != len(want):
        out("error", what=f"NICs found {[n for n, _ in ctxs]} of {want}")
        sys.exit(1)

if a.alloc == "kfd":
    check(hip.hipMalloc(C.byref(ptr), size), f"hipMalloc {a.gib} GiB")
else:
    rn = glob.glob(f"{pci}/drm/renderD*")[0].rsplit("/", 1)[1]
    dfd = os.open(f"/dev/dri/{rn}", os.O_RDWR | os.O_CLOEXEC)
    req = bytearray(struct.pack("QQQQ", size, 2 * MiB, 0x4, 0))  # bo_size, alignment, AMDGPU_GEM_DOMAIN_VRAM, flags
    fcntl.ioctl(dfd, 0xC0206440, req)  # DRM_IOCTL_AMDGPU_GEM_CREATE
    handle = struct.unpack_from("I", req, 0)[0]
    prime = bytearray(struct.pack("IIi", handle, os.O_CLOEXEC | os.O_RDWR, -1))
    fcntl.ioctl(dfd, 0xC00C642D, prime)  # DRM_IOCTL_PRIME_HANDLE_TO_FD
    gem_fd = struct.unpack_from("IIi", prime, 0)[2]
    out("gem_created", render=rn, handle=handle, fd=gem_fd, **mem())
    # hipExternalMemoryHandleDesc: type @0, union{int fd; ...} @8 (16 bytes), size @24, flags @32, reserved[16] -> 104
    desc = (C.c_char * 104)()
    struct.pack_into("i", desc, 0, 1)  # hipExternalMemoryHandleTypeOpaqueFd; HIP owns the fd it is given
    struct.pack_into("i", desc, 8, os.dup(gem_fd))
    struct.pack_into("Q", desc, 24, size)
    ext = vp()
    rc = hip.hipImportExternalMemory(C.byref(ext), desc)
    via = "hipImportExternalMemory"
    if rc == 0:
        bdesc = (C.c_char * 88)()  # hipExternalMemoryBufferDesc: offset @0, size @8, flags @16, reserved[16]
        struct.pack_into("QQI", bdesc, 0, 0, size, 0)
        rc = hip.hipExternalMemoryGetMappedBuffer(C.byref(ptr), ext, bdesc)
    if rc:
        out("hip_import_failed", rc=herr(rc))
        agents = []
        AGENT_CB = C.CFUNCTYPE(C.c_int, C.c_uint64, vp)

        def cb(handle, _):
            t = C.c_int()
            hsa.hsa_agent_get_info(C.c_uint64(handle), 17, C.byref(t))  # HSA_AGENT_INFO_DEVICE
            if t.value == 1:  # HSA_DEVICE_TYPE_GPU
                agents.append(handle)
            return 0

        hsa.hsa_iterate_agents(AGENT_CB(cb), None)
        ag = (C.c_uint64 * 1)(agents[0])
        msz, meta, isz = sz(), vp(), sz()
        hsa.hsa_amd_interop_map_buffer.argtypes = [C.c_uint32, vp, C.c_int, C.c_uint32, C.POINTER(sz), C.POINTER(vp),
                                                   C.POINTER(sz), C.POINTER(vp)]
        rc = hsa.hsa_amd_interop_map_buffer(1, ag, os.dup(gem_fd), 0, C.byref(isz), C.byref(ptr), C.byref(msz),
                                            C.byref(meta))
        via = "hsa_amd_interop_map_buffer"
        if rc:
            out("error", what="hsa_amd_interop_map_buffer", rc=rc)
            sys.exit(1)
    out("gem_imported", via=via, ptr=hex(ptr.value), headroom_gib=headroom(), **mem())
    anchor = ibv.ibv_reg_dmabuf_mr(ctxs[0][1], 0, 4096, ptr.value, gem_fd, 1)
    if not anchor:
        out("error", what="dma-buf pin anchor", errno=C.get_errno())
        sys.exit(1)
    out("gem_pinned", headroom_gib=headroom(), **mem())
    pat = (C.c_uint8 * MiB)()
    C.memset(pat, 0x5A, MiB)
    back = (C.c_uint8 * MiB)()
    rc = hip.hipMemcpy(C.c_void_p(ptr.value + size - MiB), pat, MiB, 1) or \
        hip.hipMemcpy(back, C.c_void_p(ptr.value + size - MiB), MiB, 2)
    out("gem_rw", rc=herr(rc), ok=rc == 0 and bytes(back) == bytes(pat))
out("allocated", ptr=hex(ptr.value), **mem())

mrs, how = [], ""
t = time.time()
chunk = int(a.chunk_gib * GiB)
if a.reg != "none":
    for name, pd in ctxs:
        for off in range(0, size, chunk):
            n = min(chunk, size - off)
            if a.reg == "peermem":
                mr = ibv.ibv_reg_mr(pd, C.c_void_p(ptr.value + off), n, ACCESS)
            else:
                fd, foff = C.c_int(-1), C.c_uint64(0)
                rc = hsa.hsa_amd_portable_export_dmabuf(C.c_void_p(ptr.value + off), n, C.byref(fd), C.byref(foff))
                if rc == 0:
                    how = "hsa_amd_portable_export_dmabuf"
                    mr = ibv.ibv_reg_dmabuf_mr(pd, foff.value, n, ptr.value + off, fd.value, ACCESS)
                    hsa.hsa_amd_portable_close_dmabuf(fd.value)
                elif gem_fd >= 0:
                    how = f"gem fd (hsa export rc={rc})"
                    mr = ibv.ibv_reg_dmabuf_mr(pd, off, n, ptr.value + off, gem_fd, ACCESS)
                else:
                    out("error", what=f"hsa_amd_portable_export_dmabuf rc={rc}")
                    sys.exit(1)
            if not mr:
                out("error", what=f"{a.reg} MR {name} off {off >> 30} GiB", errno=C.get_errno(), **mem())
                sys.exit(1)
            mrs.append(mr)
out("registered", mrs=len(mrs), nics=[n for n, _ in ctxs], secs=round(time.time() - t, 1), dmabuf_via=how,
    headroom_gib=headroom(), **mem())
out("after", stream=stream_probe(), **mem())
for mr in mrs:
    ibv.ibv_dereg_mr(mr)
out("deregistered", headroom_gib=headroom(), **mem())
