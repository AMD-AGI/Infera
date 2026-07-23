#!/usr/bin/env python3
"""
Ionic GPU-VRAM RDMA registration probe (single node, no partner, no Mooncake).

Tests the two things that decide whether PD KV transfer works on this NIC stack:

  TEST A  plain ibv_reg_mr(VRAM)        <- what Mooncake/Mori do by default.
                                           OLD ionic 25.08 -> FAILS (errno 14 EFAULT / 12 ENOMEM)
                                           want on NEW ionic -> OK
  TEST B  ibv_reg_dmabuf_mr(VRAM)       <- the modern GPUDirect path, and how much
                                           extra VRAM it costs.
                                           OLD ionic 25.08 -> registers OK but DOUBLES VRAM (2x shadow)
                                           want on NEW ionic -> OK and ~1x (no shadow)

Env knobs:  PROBE_GIB=4 (buffer size for the cost test)   PROBE_DEV=ionic_0 (force a NIC)
Run inside a ROCm container that can see the ionic NICs (needs >= PROBE_GIB free VRAM).
"""
import ctypes, os, sys

GiB = 1 << 30
BUF = int(float(os.environ.get("PROBE_GIB", "4")) * GiB)   # dma-buf cost test
SMALL = 256 << 20                                          # bare reg_mr test
ACCESS = 1 | 2 | 4                                         # LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ
HIP_DMABUF = 1                                             # hipMemRangeHandleTypeDmaBufFd

def load(*names):
    for n in names:
        try:
            return ctypes.CDLL(n, use_errno=True)
        except OSError:
            continue
    sys.exit(f"FATAL: cannot load any of {names} (run inside a ROCm+RDMA container)")

hip = load("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so", "/opt/rocm/lib/libamdhip64.so.6")
ibv = load("libibverbs.so.1", "libibverbs.so")

for fn, res, args in [
    ("hipMalloc", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
    ("hipFree", ctypes.c_int, [ctypes.c_void_p]),
    ("hipMemGetInfo", ctypes.c_int, [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
    ("hipMemGetHandleForAddressRange", ctypes.c_int,
     [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_ulonglong]),
]:
    getattr(hip, fn).restype = res
    getattr(hip, fn).argtypes = args

ibv.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
ibv.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
ibv.ibv_get_device_name.restype = ctypes.c_char_p
ibv.ibv_get_device_name.argtypes = [ctypes.c_void_p]
ibv.ibv_open_device.restype = ctypes.c_void_p
ibv.ibv_open_device.argtypes = [ctypes.c_void_p]
ibv.ibv_alloc_pd.restype = ctypes.c_void_p
ibv.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
ibv.ibv_reg_mr.restype = ctypes.c_void_p
ibv.ibv_reg_mr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
ibv.ibv_reg_dmabuf_mr.restype = ctypes.c_void_p
ibv.ibv_reg_dmabuf_mr.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
                                  ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
ibv.ibv_dereg_mr.restype = ctypes.c_int
ibv.ibv_dereg_mr.argtypes = [ctypes.c_void_p]

def free_gib():
    fr, tot = ctypes.c_size_t(), ctypes.c_size_t()
    hip.hipMemGetInfo(ctypes.byref(fr), ctypes.byref(tot))
    return fr.value / GiB

# ---- pick an ionic device ----
n = ctypes.c_int(0)
lst = ibv.ibv_get_device_list(ctypes.byref(n))
names = [ibv.ibv_get_device_name(lst[i]).decode() for i in range(n.value)]
print("RDMA devices visible:", names or "(none)")
ionics = [x for x in names if x.startswith("ionic")]
want = os.environ.get("PROBE_DEV", "")
dev_name = want or (ionics[0] if ionics else (names[0] if names else None))
if not dev_name:
    sys.exit("FATAL: no RDMA device visible (are the ionic NICs mapped into the container?)")
if not ionics:
    print("WARNING: no ionic* device found; testing on", dev_name)
dev = next(lst[i] for i in range(n.value) if names[i] == dev_name)
pd = ibv.ibv_alloc_pd(ibv.ibv_open_device(dev))
if not pd:
    sys.exit(f"FATAL: cannot open/alloc_pd on {dev_name}")
print(f"testing on device: {dev_name}\n")

results = {}

# ================= TEST B: dma-buf registration + VRAM cost =================
print(f"=== TEST B: ibv_reg_dmabuf_mr on {BUF/GiB:.1f} GiB VRAM (+ measure VRAM cost) ===")
f0 = free_gib()
dptr = ctypes.c_void_p()
if hip.hipMalloc(ctypes.byref(dptr), ctypes.c_size_t(BUF)) != 0 or not dptr.value:
    sys.exit("FATAL: hipMalloc failed (not enough free VRAM? lower PROBE_GIB)")
f1 = free_gib()
fd = ctypes.c_int(-1)
rc = hip.hipMemGetHandleForAddressRange(ctypes.byref(fd), dptr, ctypes.c_size_t(BUF), HIP_DMABUF, 0)
if rc != 0 or fd.value < 0:
    print(f"  dma-buf export failed rc={rc} -> cannot run TEST B")
    results["dmabuf"] = "EXPORT_FAILED"
else:
    ctypes.set_errno(0)
    mrd = ibv.ibv_reg_dmabuf_mr(pd, 0, ctypes.c_size_t(BUF), ctypes.c_uint64(dptr.value), fd.value, ACCESS)
    ed = ctypes.get_errno()
    f2 = free_gib()
    print(f"  free VRAM: before alloc={f0:.2f}  after alloc={f1:.2f}  after reg={f2:.2f} (GiB)")
    print(f"  buffer={BUF/GiB:.2f}  alloc_cost={f0-f1:+.2f}  reg_extra_cost={f1-f2:+.2f}")
    if mrd:
        shadow = (f1 - f2) > 0.5 * (BUF / GiB)
        verdict = "2x SHADOW (VRAM doubled on registration!)" if shadow else "~1x IN-PLACE (no shadow)"
        print(f"  ibv_reg_dmabuf_mr: OK   ->  {verdict}")
        results["dmabuf"] = "OK_2x" if shadow else "OK_1x"
        ibv.ibv_dereg_mr(mrd)
    else:
        print(f"  ibv_reg_dmabuf_mr: FAIL errno={ed} ({os.strerror(ed) if ed else ''})")
        results["dmabuf"] = f"FAIL_{ed}"
hip.hipFree(dptr)

# ================= TEST A: bare ibv_reg_mr on VRAM (Mooncake default) =========
print(f"\n=== TEST A: plain ibv_reg_mr on {SMALL>>20} MiB VRAM (Mooncake/Mori default path) ===")
sp = ctypes.c_void_p()
if hip.hipMalloc(ctypes.byref(sp), ctypes.c_size_t(SMALL)) != 0 or not sp.value:
    print("  hipMalloc failed; skipping TEST A")
    results["bare"] = "ALLOC_FAILED"
else:
    ctypes.set_errno(0)
    mrb = ibv.ibv_reg_mr(pd, sp, ctypes.c_size_t(SMALL), ACCESS)
    eb = ctypes.get_errno()
    if mrb:
        print("  ibv_reg_mr(VRAM): OK  -> Mooncake works out-of-box, no code change")
        results["bare"] = "OK"
        ibv.ibv_dereg_mr(mrb)
    else:
        print(f"  ibv_reg_mr(VRAM): FAIL errno={eb} ({os.strerror(eb) if eb else ''})")
        results["bare"] = f"FAIL_{eb}"
    hip.hipFree(sp)

# ================= VERDICT =================
print("\n===================== VERDICT =====================")
print(f"  bare ibv_reg_mr(VRAM) : {results.get('bare')}")
print(f"  dma-buf reg(VRAM)     : {results.get('dmabuf')}")
print("  (OLD ionic 25.08 baseline on our cluster: bare=FAIL_14, dmabuf=OK_2x)")
print("  (want on NEW ionic 26.x:                  bare=OK,      dmabuf=OK_1x)")
