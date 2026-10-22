#!/usr/bin/env python3
"""Extended cross-container HIP IPC test.
exp <dir> <n_importers>: torch on visible dev 3; exports 3 interior/fragment pointers like mooncake
   registerLocalMemory does (hipIpcGetMemHandle(addr) on the registered addr):
   S = slice of a 256 MiB torch tensor at offset 37 MiB + 4 KiB (64 MiB long)
   T = 2nd 4 MiB torch tensor (caching-allocator sub-allocation inside a 20 MiB segment)
   F = raw hipMalloc(256 KiB) (ROCr fragment sub-allocator)
   Each filled with its own byte; surrounding memory 0x11. Writes handles.json, waits for N done files.
imp <dir> <variant>: variant in M0 (main thread, no hipSetDevice), M3 (main, hipSetDevice(3)),
   W0 (worker thread, no hipSetDevice), W3 (worker thread, hipSetDevice(3)). Opens all 3 handles,
   DtoH first/last 4 KiB of each, checks byte."""
import ctypes, json, os, sys, threading, time

role, d = sys.argv[1], sys.argv[2]
hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetErrorString.restype = ctypes.c_char_p
def es(rc): return hip.hipGetErrorString(rc).decode()
MODE = "legacy" if os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY") == "1" else "default"

class H(ctypes.Structure):
    _fields_ = [("r", ctypes.c_char * 64)]

if role == "exp":
    import torch
    n_imp = int(sys.argv[3])
    torch.cuda.set_device(3)
    big = torch.full((256 << 20,), 0x11, dtype=torch.uint8, device="cuda")
    off = (37 << 20) + 4096
    S = big[off:off + (64 << 20)]; S.fill_(0x5A)
    t1 = torch.full((4 << 20,), 0x11, dtype=torch.uint8, device="cuda")
    T = torch.full((4 << 20,), 0x6B, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    fp = ctypes.c_void_p()
    assert hip.hipSetDevice(3) == 0
    assert hip.hipMalloc(ctypes.byref(fp), ctypes.c_size_t(256 << 10)) == 0
    assert hip.hipMemset(fp, 0x7C, ctypes.c_size_t(256 << 10)) == 0
    assert hip.hipDeviceSynchronize() == 0
    bufs = {"S": (S.data_ptr(), S.numel(), 0x5A), "T": (T.data_ptr(), T.numel(), 0x6B),
            "F": (fp.value, 256 << 10, 0x7C)}
    print(f"[exp {MODE}] big={big.data_ptr():#x} S={S.data_ptr():#x} t1={t1.data_ptr():#x} T={T.data_ptr():#x} "
          f"(T-t1={T.data_ptr() - t1.data_ptr()}) F={fp.value:#x}", flush=True)
    out = {}
    for k, (ptr, n, b) in bufs.items():
        h = ctypes.create_string_buffer(64)
        rc = hip.hipIpcGetMemHandle(h, ctypes.c_void_p(ptr))
        print(f"[exp {MODE}] hipIpcGetMemHandle {k} ptr={ptr:#x} rc={rc} {es(rc)}", flush=True)
        base = ctypes.c_void_p(); sz = ctypes.c_size_t()
        rcr = hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(sz), ctypes.c_void_p(ptr))
        boff = ptr - (base.value or ptr)
        print(f"[exp {MODE}]   {k} hipMemGetAddressRange rc={rcr} base={(base.value or 0):#x} size={sz.value} ptr-base={boff}", flush=True)
        if rc == 0:
            out[k] = dict(h=h.raw.hex(), n=n, b=b, boff=boff)
    tmp = os.path.join(d, "handles.tmp"); json.dump(out, open(tmp, "w")); os.rename(tmp, os.path.join(d, "handles.json"))
    t0 = time.time()
    while len([f for f in os.listdir(d) if f.startswith("done.")]) < n_imp and time.time() - t0 < 300:
        time.sleep(0.2)
    print(f"[exp {MODE}] exit after {time.time() - t0:.1f}s", flush=True)
else:
    var = sys.argv[3]
    hs = json.load(open(os.path.join(d, "handles.json")))
    res = {}
    def work():
        if var.endswith("3"):
            res["setdev"] = hip.hipSetDevice(3)
        cur = ctypes.c_int(-1); hip.hipGetDevice(ctypes.byref(cur)); res["cur"] = cur.value
        hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), H, ctypes.c_uint]
        for k, e in hs.items():
            hh = H(); ctypes.memmove(ctypes.addressof(hh), bytes.fromhex(e["h"]), 64)
            q = ctypes.c_void_p()
            rc = hip.hipIpcOpenMemHandle(ctypes.byref(q), hh, 1)
            if rc != 0:
                res[k] = f"OPEN rc={rc} {es(rc)}"; continue
            host = (ctypes.c_ubyte * 4096)()
            ok = True
            for o in (0, e["n"] - 4096):
                rc2 = hip.hipMemcpy(host, ctypes.c_void_p(q.value + o), ctypes.c_size_t(4096), 2)
                if rc2 != 0:
                    res[k] = f"MEMCPY rc={rc2} {es(rc2)}"; ok = None; break
                if not all(x == e["b"] for x in host):
                    ok = False; res[k] = f"DATA_MISMATCH got0={host[0]:#x} want={e['b']:#x} at+{o}"; break
            if ok:
                res[k] = "OK"
            elif e.get("boff"):
                rc3 = hip.hipMemcpy(host, ctypes.c_void_p(q.value + e["boff"]), ctypes.c_size_t(4096), 2)
                res[k] += f" | at opened+boff({e['boff']}): rc={rc3} " + ("MATCH" if rc3 == 0 and all(x == e["b"] for x in host) else f"nomatch got0={host[0]:#x}")
            hip.hipIpcCloseMemHandle(q)
    if var.startswith("W"):
        th = threading.Thread(target=work); th.start(); th.join()
    else:
        work()
    print(f"RESULT {MODE} {var} " + json.dumps(res), flush=True)
    open(os.path.join(d, f"done.{var}"), "w").write("x")
