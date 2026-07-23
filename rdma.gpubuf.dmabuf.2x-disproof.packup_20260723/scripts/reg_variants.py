#!/usr/bin/env python3
# The 2x is at reg time, not export. Probe whether ibv_reg_dmabuf_mr ARGS change it:
# vary access flags (LOCAL only vs REMOTE), and check if the shadow is per-MR or
# tied to something reusable. Also register the SAME dmabuf twice to see if 2nd is free.
import ctypes, os, sys
GiB=1<<30; BUF=int(float(os.environ.get("PROBE_GIB","4"))*GiB)
def load(*n):
    for x in n:
        try: return ctypes.CDLL(x,use_errno=True)
        except OSError: continue
    sys.exit("load fail")
hip=load("libamdhip64.so"); ibv=load("libibverbs.so.1"); hsa=load("libhsa-runtime64.so.1")
for fn,r,a in [("hipMalloc",ctypes.c_int,[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]),
 ("hipFree",ctypes.c_int,[ctypes.c_void_p]),("hipInit",ctypes.c_int,[ctypes.c_uint]),
 ("hipMemGetInfo",ctypes.c_int,[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]),
 ("hipMemGetHandleForAddressRange",ctypes.c_int,[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_ulonglong])]:
    getattr(hip,fn).restype=r; getattr(hip,fn).argtypes=a
ibv.ibv_get_device_list.restype=ctypes.POINTER(ctypes.c_void_p);ibv.ibv_get_device_list.argtypes=[ctypes.POINTER(ctypes.c_int)]
ibv.ibv_get_device_name.restype=ctypes.c_char_p;ibv.ibv_get_device_name.argtypes=[ctypes.c_void_p]
ibv.ibv_open_device.restype=ctypes.c_void_p;ibv.ibv_open_device.argtypes=[ctypes.c_void_p]
ibv.ibv_alloc_pd.restype=ctypes.c_void_p;ibv.ibv_alloc_pd.argtypes=[ctypes.c_void_p]
ibv.ibv_reg_dmabuf_mr.restype=ctypes.c_void_p
ibv.ibv_reg_dmabuf_mr.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t,ctypes.c_uint64,ctypes.c_int,ctypes.c_int]
ibv.ibv_dereg_mr.restype=ctypes.c_int;ibv.ibv_dereg_mr.argtypes=[ctypes.c_void_p]
def fG():
    fr,t=ctypes.c_size_t(),ctypes.c_size_t();hip.hipMemGetInfo(ctypes.byref(fr),ctypes.byref(t));return fr.value/GiB
hip.hipInit(0)
n=ctypes.c_int(0);lst=ibv.ibv_get_device_list(ctypes.byref(n))
names=[ibv.ibv_get_device_name(lst[i]).decode() for i in range(n.value)]
dev=next(lst[i] for i in range(n.value) if names[i].startswith("ionic"))
pd=ibv.ibv_alloc_pd(ibv.ibv_open_device(dev))

def export(dptr):
    fd=ctypes.c_int(-1);hip.hipMemGetHandleForAddressRange(ctypes.byref(fd),dptr,ctypes.c_size_t(BUF),1,0);return fd.value

# 1) access-flag sweep
for label,acc in [("LOCAL_only",1),("LOCAL+REMOTE_WRITE",1|2),("LOCAL+REMOTE_READ",1|4),("ALL",1|2|4)]:
    dptr=ctypes.c_void_p();hip.hipMalloc(ctypes.byref(dptr),ctypes.c_size_t(BUF))
    fd=export(dptr); f1=fG()
    mr=ibv.ibv_reg_dmabuf_mr(pd,0,ctypes.c_size_t(BUF),ctypes.c_uint64(dptr.value),fd,acc)
    f2=fG()
    print(f"access={label:20s} reg_cost={f1-f2:+.2f} -> {'2x' if (f1-f2)>0.5*BUF/GiB else '1x'}  {'OK' if mr else 'FAIL'}")
    if mr: ibv.ibv_dereg_mr(mr)
    hip.hipFree(dptr)

# 2) register the SAME buffer twice — is the shadow per-registration?
print("\n-- double-register same buffer --")
dptr=ctypes.c_void_p();hip.hipMalloc(ctypes.byref(dptr),ctypes.c_size_t(BUF))
fd1=export(dptr); a=fG()
mr1=ibv.ibv_reg_dmabuf_mr(pd,0,ctypes.c_size_t(BUF),ctypes.c_uint64(dptr.value),fd1,1|2|4); b=fG()
fd2=export(dptr); mr2=ibv.ibv_reg_dmabuf_mr(pd,0,ctypes.c_size_t(BUF),ctypes.c_uint64(dptr.value),fd2,1|2|4); c=fG()
print(f"1st reg cost={a-b:+.2f}  2nd reg cost={b-c:+.2f}")
for m in (mr1,mr2):
    if m: ibv.ibv_dereg_mr(m)
hip.hipFree(dptr)
