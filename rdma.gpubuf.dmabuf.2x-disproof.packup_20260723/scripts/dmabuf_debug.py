#!/usr/bin/env python3
# Debug harness for the ionic dmabuf 2x-shadow. Isolates WHERE the extra VRAM
# appears: (a) hipMalloc, (b) dmabuf export, (c) ibv_reg_dmabuf_mr.
# Compares export via HIP (v1/NONE) vs HSA v2 with PCIE mapping flag.
import ctypes, os, sys
GiB = 1<<30
BUF = int(float(os.environ.get("PROBE_GIB","4"))*GiB)
ACCESS = 1|2|4
HIP_DMABUF = 1  # hipMemRangeHandleTypeDmaBufFd
HSA_MAP_NONE = 0
HSA_MAP_PCIE = 1

def load(*names):
    for n in names:
        try: return ctypes.CDLL(n, use_errno=True)
        except OSError: continue
    sys.exit(f"cannot load {names}")

hip = load("libamdhip64.so","/opt/rocm/lib/libamdhip64.so")
ibv = load("libibverbs.so.1")
hsa = load("libhsa-runtime64.so.1","/opt/rocm/lib/libhsa-runtime64.so.1")

for fn,res,args in [
 ("hipMalloc",ctypes.c_int,[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]),
 ("hipFree",ctypes.c_int,[ctypes.c_void_p]),
 ("hipMemGetInfo",ctypes.c_int,[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]),
 ("hipMemGetHandleForAddressRange",ctypes.c_int,[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_ulonglong]),
 ("hipInit",ctypes.c_int,[ctypes.c_uint]),
]:
    getattr(hip,fn).restype=res; getattr(hip,fn).argtypes=args

# HSA v1 + v2 export signatures
hsa.hsa_init.restype=ctypes.c_int; hsa.hsa_init.argtypes=[]
hsa.hsa_amd_portable_export_dmabuf.restype=ctypes.c_int
hsa.hsa_amd_portable_export_dmabuf.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint64)]
have_v2 = hasattr(hsa,"hsa_amd_portable_export_dmabuf_v2")
if have_v2:
    hsa.hsa_amd_portable_export_dmabuf_v2.restype=ctypes.c_int
    hsa.hsa_amd_portable_export_dmabuf_v2.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint64),ctypes.c_uint64]

ibv.ibv_get_device_list.restype=ctypes.POINTER(ctypes.c_void_p); ibv.ibv_get_device_list.argtypes=[ctypes.POINTER(ctypes.c_int)]
ibv.ibv_get_device_name.restype=ctypes.c_char_p; ibv.ibv_get_device_name.argtypes=[ctypes.c_void_p]
ibv.ibv_open_device.restype=ctypes.c_void_p; ibv.ibv_open_device.argtypes=[ctypes.c_void_p]
ibv.ibv_alloc_pd.restype=ctypes.c_void_p; ibv.ibv_alloc_pd.argtypes=[ctypes.c_void_p]
ibv.ibv_reg_dmabuf_mr.restype=ctypes.c_void_p
ibv.ibv_reg_dmabuf_mr.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t,ctypes.c_uint64,ctypes.c_int,ctypes.c_int]
ibv.ibv_dereg_mr.restype=ctypes.c_int; ibv.ibv_dereg_mr.argtypes=[ctypes.c_void_p]

def freeG():
    fr,tot=ctypes.c_size_t(),ctypes.c_size_t(); hip.hipMemGetInfo(ctypes.byref(fr),ctypes.byref(tot)); return fr.value/GiB

hip.hipInit(0); hsa.hsa_init()
n=ctypes.c_int(0); lst=ibv.ibv_get_device_list(ctypes.byref(n))
names=[ibv.ibv_get_device_name(lst[i]).decode() for i in range(n.value)]
ionics=[x for x in names if x.startswith("ionic")]
dev_name=os.environ.get("PROBE_DEV","") or (ionics[0] if ionics else names[0])
dev=next(lst[i] for i in range(n.value) if names[i]==dev_name)
pd=ibv.ibv_alloc_pd(ibv.ibv_open_device(dev))
print(f"device={dev_name}  have_v2={have_v2}\n")

def run(export_mode):
    # export_mode: 'hip', 'hsa_v1', 'hsa_v2_none', 'hsa_v2_pcie'
    print(f"--- export_mode={export_mode} ---")
    f0=freeG()
    dptr=ctypes.c_void_p()
    if hip.hipMalloc(ctypes.byref(dptr),ctypes.c_size_t(BUF))!=0 or not dptr.value:
        print("  hipMalloc failed"); return
    f1=freeG()
    fd=ctypes.c_int(-1); off=ctypes.c_uint64(0); rc=-1
    if export_mode=='hip':
        rc=hip.hipMemGetHandleForAddressRange(ctypes.byref(fd),dptr,ctypes.c_size_t(BUF),HIP_DMABUF,0)
    elif export_mode=='hsa_v1':
        rc=hsa.hsa_amd_portable_export_dmabuf(dptr,ctypes.c_size_t(BUF),ctypes.byref(fd),ctypes.byref(off))
    elif export_mode=='hsa_v2_none' and have_v2:
        rc=hsa.hsa_amd_portable_export_dmabuf_v2(dptr,ctypes.c_size_t(BUF),ctypes.byref(fd),ctypes.byref(off),HSA_MAP_NONE)
    elif export_mode=='hsa_v2_pcie' and have_v2:
        rc=hsa.hsa_amd_portable_export_dmabuf_v2(dptr,ctypes.c_size_t(BUF),ctypes.byref(fd),ctypes.byref(off),HSA_MAP_PCIE)
    else:
        print("  (mode unavailable)"); hip.hipFree(dptr); return
    f2=freeG()
    if rc!=0 or fd.value<0:
        print(f"  export FAILED rc={rc}"); hip.hipFree(dptr); return
    ctypes.set_errno(0)
    mr=ibv.ibv_reg_dmabuf_mr(pd,off.value,ctypes.c_size_t(BUF),ctypes.c_uint64(dptr.value),fd.value,ACCESS)
    e=ctypes.get_errno()
    f3=freeG()
    print(f"  VRAM(GiB): start={f0:.2f} afterMalloc={f1:.2f} afterExport={f2:.2f} afterReg={f3:.2f}")
    print(f"  cost: malloc={f0-f1:+.2f} export={f1-f2:+.2f} reg={f2-f3:+.2f}  offset={off.value}")
    if mr:
        total_extra=f1-f3
        print(f"  reg OK. extra-beyond-buffer(export+reg)={total_extra:+.2f} -> {'2x SHADOW' if total_extra>0.5*(BUF/GiB) else '~1x OK'}")
        ibv.ibv_dereg_mr(mr)
    else:
        print(f"  reg FAILED errno={e} ({os.strerror(e) if e else ''})")
    hip.hipFree(dptr)
    print()

for m in ['hip','hsa_v1','hsa_v2_none','hsa_v2_pcie']:
    try: run(m)
    except Exception as ex: print(f"  {m} raised {ex}\n")
