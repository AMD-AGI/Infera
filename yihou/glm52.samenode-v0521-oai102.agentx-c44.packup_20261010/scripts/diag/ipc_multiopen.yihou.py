#!/usr/bin/env python3
"""Multi-open of ONE allocation via several interior-pointer handles (KV-pool-like per-layer views).
exp <dir> <nviews>: torch dev 3, big = (nviews+1)*16 MiB tensor; view k = big[k*16MiB:(k+1)*16MiB]
   filled with byte 0x20+k. Exports handles for views 1..nviews (interior) then view 0 (== base).
imp <dir>: worker thread, hipSetDevice(3), opens handles in file order; for each: open rc, data at
   opened+0 and opened+boff; keeps all open until the end (like mooncake's remap cache)."""
import ctypes, json, os, sys, threading, time
role, d = sys.argv[1], sys.argv[2]
hip = ctypes.CDLL("libamdhip64.so"); hip.hipGetErrorString.restype = ctypes.c_char_p
es = lambda rc: hip.hipGetErrorString(rc).decode()
MODE = "legacy" if os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY") == "1" else "default"
class H(ctypes.Structure): _fields_ = [("r", ctypes.c_char * 64)]
MB16 = 16 << 20
if role == "exp":
    import torch
    nv = int(sys.argv[3]); torch.cuda.set_device(3)
    big = torch.empty(((nv + 1) * MB16,), dtype=torch.uint8, device="cuda")
    views = [big[k * MB16:(k + 1) * MB16] for k in range(nv + 1)]
    for k, v in enumerate(views): v.fill_(0x20 + k)
    torch.cuda.synchronize(); assert hip.hipSetDevice(3) == 0
    out = []
    for k in list(range(1, nv + 1)) + [0]:
        h = ctypes.create_string_buffer(64); p = views[k].data_ptr()
        rc = hip.hipIpcGetMemHandle(h, ctypes.c_void_p(p))
        print(f"[exp {MODE}] get view{k} ptr={p:#x} boff={p - big.data_ptr()} rc={rc} {es(rc)} h16={h.raw[:16].hex()}", flush=True)
        out.append(dict(k=k, h=h.raw.hex(), boff=p - big.data_ptr(), b=0x20 + k))
    json.dump(out, open(os.path.join(d, "h.tmp"), "w")); os.rename(os.path.join(d, "h.tmp"), os.path.join(d, "handles.json"))
    t0 = time.time()
    while not os.path.exists(os.path.join(d, "done")) and time.time() - t0 < 120: time.sleep(0.2)
    print(f"[exp {MODE}] exit", flush=True)
else:
    hs = json.load(open(os.path.join(d, "handles.json"))); rows = []
    def work():
        rows.append(f"setdev rc={hip.hipSetDevice(3)}")
        hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), H, ctypes.c_uint]
        host = (ctypes.c_ubyte * 4096)(); opened = []
        for e in hs:
            hh = H(); ctypes.memmove(ctypes.addressof(hh), bytes.fromhex(e["h"]), 64); q = ctypes.c_void_p()
            rc = hip.hipIpcOpenMemHandle(ctypes.byref(q), hh, 1)
            if rc:
                rows.append(f"view{e['k']}: OPEN rc={rc} {es(rc)}"); continue
            opened.append(q.value)
            def rd(o):
                r = hip.hipMemcpy(host, ctypes.c_void_p(q.value + o), ctypes.c_size_t(4096), 2)
                return f"rc{r}" if r else f"{host[0]:#x}"
            rows.append(f"view{e['k']}: open rc=0 q={q.value:#x} @+0={rd(0)} @+boff={rd(e['boff'])} want={e['b']:#x}")
        rows.append(f"distinct_q={len(set(opened))}/{len(opened)}")
        for q in opened: hip.hipIpcCloseMemHandle(ctypes.c_void_p(q))
    t = threading.Thread(target=work); t.start(); t.join()
    for r in rows: print(f"IMP {MODE} {r}", flush=True)
    open(os.path.join(d, "done"), "w").write("x")
