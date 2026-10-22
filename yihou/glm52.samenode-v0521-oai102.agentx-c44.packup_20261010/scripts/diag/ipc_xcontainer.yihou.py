#!/usr/bin/env python3
"""Cross-container HIP IPC test (no torch). usage: exp <dir> | imp <dir>
exp: hipMalloc 64 MiB, fill 0x5A, hipIpcGetMemHandle -> <dir>/handle.bin, wait for <dir>/done,
     then check whether importer wrote 0xA5 into the first 1 MiB.
imp: read handle, hipIpcOpenMemHandle(lazy peer), DtoH verify 0x5A, memset first 1 MiB 0xA5,
     hipIpcCloseMemHandle, write <dir>/done."""
import ctypes, os, sys, time

role, d = sys.argv[1], sys.argv[2]
hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetErrorString.restype = ctypes.c_char_p
N = 64 << 20
tag = f"[{role} pid={os.getpid()} HVD={os.environ.get('HIP_VISIBLE_DEVICES')} legacy={os.environ.get('HSA_ENABLE_IPC_MODE_LEGACY','unset')} agent={'on' if os.environ.get('HSA_TOOLS_LIB') else 'off'}]"

def chk(name, rc):
    print(f"{tag} {name} rc={rc} {hip.hipGetErrorString(rc).decode()}", flush=True)
    if rc != 0:
        print(f"RESULT {role} FAIL {name} rc={rc}", flush=True)
        if role == "imp":
            open(os.path.join(d, "done"), "w").write("fail")
        sys.exit(2)

chk("hipSetDevice", hip.hipSetDevice(0))
if role == "exp":
    p = ctypes.c_void_p()
    chk("hipMalloc", hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(N)))
    chk("hipMemset", hip.hipMemset(p, 0x5A, ctypes.c_size_t(N)))
    chk("hipDeviceSynchronize", hip.hipDeviceSynchronize())
    h = ctypes.create_string_buffer(64)
    chk("hipIpcGetMemHandle", hip.hipIpcGetMemHandle(h, p))
    tmp = os.path.join(d, "handle.tmp")
    open(tmp, "wb").write(h.raw); os.rename(tmp, os.path.join(d, "handle.bin"))
    socks = [l.split()[-1].rstrip("@") for l in open("/proc/net/unix") if ("@hsa%d@" % os.getpid()) in l]
    agent = any("rocm-debug-agent" in l for l in open("/proc/self/maps"))
    print(f"PROOF exp own_hsa_socket={bool(socks)} debug_agent_mapped={agent} handle16={h.raw[:16].hex()}", flush=True)
    t0 = time.time()
    while not os.path.exists(os.path.join(d, "done")) and time.time() - t0 < 90:
        time.sleep(0.2)
    host = (ctypes.c_ubyte * (1 << 20))()
    chk("hipMemcpyDtoH", hip.hipMemcpy(host, p, ctypes.c_size_t(1 << 20), 2))
    print(f"RESULT exp OK importer_write_seen={all(b == 0xA5 for b in host[:4096])}", flush=True)
else:
    raw = open(os.path.join(d, "handle.bin"), "rb").read()
    h = ctypes.create_string_buffer(raw, 64)
    q = ctypes.c_void_p()
    # hipIpcMemHandle_t passed BY VALUE (64-byte struct)
    class H(ctypes.Structure):
        _fields_ = [("r", ctypes.c_char * 64)]
    hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), H, ctypes.c_uint]
    hh = H(); ctypes.memmove(ctypes.addressof(hh), raw, 64)
    chk("hipIpcOpenMemHandle", hip.hipIpcOpenMemHandle(ctypes.byref(q), hh, 1))
    host = (ctypes.c_ubyte * (1 << 20))()
    chk("hipMemcpyDtoH", hip.hipMemcpy(host, q, ctypes.c_size_t(1 << 20), 2))
    ok = all(b == 0x5A for b in host)
    chk("hipMemset", hip.hipMemset(q, 0xA5, ctypes.c_size_t(1 << 20)))
    chk("hipDeviceSynchronize", hip.hipDeviceSynchronize())
    chk("hipIpcCloseMemHandle", hip.hipIpcCloseMemHandle(q))
    agent = any("rocm-debug-agent" in l for l in open("/proc/self/maps"))
    print(f"RESULT imp {'OK' if ok else 'DATA_MISMATCH'} agent_mapped={agent}", flush=True)
    open(os.path.join(d, "done"), "w").write("ok")
