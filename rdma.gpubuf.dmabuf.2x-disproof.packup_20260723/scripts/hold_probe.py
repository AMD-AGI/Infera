#!/usr/bin/env python3
# Register a dmabuf MR and HOLD it, printing a PID + phase markers, so an external
# rocm-smi can read TRUE per-GPU VRAM bytes at each phase. Releases everything at end.
import ctypes, os, sys, time
GiB=1<<30; BUF=int(float(os.environ.get("PROBE_GIB","4"))*GiB)
HOLD=float(os.environ.get("HOLD_S","12"))
def load(*n):
    for x in n:
        try: return ctypes.CDLL(x,use_errno=True)
        except OSError: continue
    sys.exit("load fail")
hip=load("libamdhip64.so"); ibv=load("libibverbs.so.1")
for fn,r,a in [("hipMalloc",ctypes.c_int,[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]),
 ("hipFree",ctypes.c_int,[ctypes.c_void_p]),("hipInit",ctypes.c_int,[ctypes.c_uint]),
 ("hipSetDevice",ctypes.c_int,[ctypes.c_int]),
 ("hipMemGetInfo",ctypes.c_int,[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]),
 ("hipMemGetHandleForAddressRange",ctypes.c_int,[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_ulonglong])]:
    getattr(hip,fn).restype=r; getattr(hip,fn).argtypes=a
ibv.ibv_get_device_list.restype=ctypes.POINTER(ctypes.c_void_p); ibv.ibv_get_device_list.argtypes=[ctypes.POINTER(ctypes.c_int)]
ibv.ibv_get_device_name.restype=ctypes.c_char_p; ibv.ibv_get_device_name.argtypes=[ctypes.c_void_p]
ibv.ibv_open_device.restype=ctypes.c_void_p; ibv.ibv_open_device.argtypes=[ctypes.c_void_p]
ibv.ibv_alloc_pd.restype=ctypes.c_void_p; ibv.ibv_alloc_pd.argtypes=[ctypes.c_void_p]
ibv.ibv_reg_dmabuf_mr.restype=ctypes.c_void_p
ibv.ibv_reg_dmabuf_mr.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t,ctypes.c_uint64,ctypes.c_int,ctypes.c_int]
ibv.ibv_dereg_mr.restype=ctypes.c_int; ibv.ibv_dereg_mr.argtypes=[ctypes.c_void_p]
def fG():
    fr,t=ctypes.c_size_t(),ctypes.c_size_t(); hip.hipMemGetInfo(ctypes.byref(fr),ctypes.byref(t)); return fr.value/GiB
hip.hipInit(0); hip.hipSetDevice(0)
print(f"PID={os.getpid()} dev=GPU0 BUF={BUF/GiB:.1f}GiB", flush=True)
n=ctypes.c_int(0); lst=ibv.ibv_get_device_list(ctypes.byref(n))
names=[ibv.ibv_get_device_name(lst[i]).decode() for i in range(n.value)]
dev=next(lst[i] for i in range(n.value) if names[i].startswith("ionic"))
pd=ibv.ibv_alloc_pd(ibv.ibv_open_device(dev))
print(f"PHASE=baseline hipfree={fG():.2f}", flush=True); time.sleep(HOLD)
dptr=ctypes.c_void_p(); hip.hipMalloc(ctypes.byref(dptr),ctypes.c_size_t(BUF))
print(f"PHASE=after_malloc hipfree={fG():.2f}", flush=True); time.sleep(HOLD)
fd=ctypes.c_int(-1); hip.hipMemGetHandleForAddressRange(ctypes.byref(fd),dptr,ctypes.c_size_t(BUF),1,0)
print(f"PHASE=after_export hipfree={fG():.2f} fd={fd.value}", flush=True); time.sleep(HOLD)
mr=ibv.ibv_reg_dmabuf_mr(pd,0,ctypes.c_size_t(BUF),ctypes.c_uint64(dptr.value),fd.value,1|2|4)
print(f"PHASE=after_reg hipfree={fG():.2f} mr={'OK' if mr else 'FAIL'}", flush=True); time.sleep(HOLD)
if mr: ibv.ibv_dereg_mr(mr)
os.close(fd.value) if fd.value>=0 else None
print(f"PHASE=after_dereg hipfree={fG():.2f}", flush=True); time.sleep(HOLD)
hip.hipFree(dptr)
print(f"PHASE=after_free hipfree={fG():.2f}", flush=True)
