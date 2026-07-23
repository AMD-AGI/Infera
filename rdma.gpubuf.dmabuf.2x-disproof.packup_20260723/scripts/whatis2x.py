#!/usr/bin/env python3
# Is the "2x" real VRAM duplication or P2PDMA/BAR accounting? 
# Register a buffer, hold it, and report BOTH hipMemGetInfo free AND note the addr,
# so an external rocm-smi can read true VRAM. Also try scaling: does Nth distinct
# buffer each cost +size, or is there a saturation (fixed P2P window)?
import ctypes,os,sys,time
GiB=1<<30; BUF=int(float(os.environ.get("PROBE_GIB","2"))*GiB)
def load(*n):
    for x in n:
        try:return ctypes.CDLL(x,use_errno=True)
        except OSError:continue
    sys.exit("load")
hip=load("libamdhip64.so");ibv=load("libibverbs.so.1")
for fn,r,a in [("hipMalloc",ctypes.c_int,[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]),
 ("hipFree",ctypes.c_int,[ctypes.c_void_p]),("hipInit",ctypes.c_int,[ctypes.c_uint]),
 ("hipMemGetInfo",ctypes.c_int,[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]),
 ("hipMemGetHandleForAddressRange",ctypes.c_int,[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_ulonglong])]:
    getattr(hip,fn).restype=r;getattr(hip,fn).argtypes=a
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
def exp(d):
    fd=ctypes.c_int(-1);hip.hipMemGetHandleForAddressRange(ctypes.byref(fd),d,ctypes.c_size_t(BUF),1,0);return fd.value
# scaling test: register N DISTINCT buffers, watch cumulative reg cost
print(f"buffer={BUF/GiB:.1f}GiB each; registering 4 DISTINCT buffers, cumulative reg cost:")
ptrs=[];mrs=[]
base=fG()
for i in range(4):
    d=ctypes.c_void_p();hip.hipMalloc(ctypes.byref(d),ctypes.c_size_t(BUF));ptrs.append(d)
    am=fG()
    fd=exp(d)
    mr=ibv.ibv_reg_dmabuf_mr(pd,0,ctypes.c_size_t(BUF),ctypes.c_uint64(d.value),fd,1|2|4);mrs.append(mr)
    ar=fG()
    print(f"  buf#{i}: afterMalloc_free={am:.2f} afterReg_free={ar:.2f} reg_cost={am-ar:+.2f}")
print(f"total free drop over 4 bufs = {base-fG():.2f} (pure buffers would be {4*BUF/GiB:.1f})")
for m in mrs:
    if m: ibv.ibv_dereg_mr(m)
for p in ptrs: hip.hipFree(p)
