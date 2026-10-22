#!/usr/bin/env python3
"""Discriminating test: ROCr dma-buf IPC export binds an ABSTRACT unix socket named
"\\0hsa<getpid()>" (runtime.cpp IPCCreate: snprintf "xhsa%i" then sun_path[0]=0, addrlen =
sizeof(sockaddr_un)). Abstract sockets live in the NETWORK namespace, PIDs in the PID
namespace -> two --network host containers whose processes share a PID collide.
Modes: control | prebind (squat our own name first, as another container's process would).
Run under HSA_ENABLE_IPC_MODE_LEGACY=0/1. Prints IPCGET rc + whether our name is bound."""
import ctypes, os, socket, sys

mode = sys.argv[1]
name = b"\0hsa%d" % os.getpid()
squat = None
if mode == "prebind":
    squat = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    squat.bind(name + b"\0" * (108 - len(name)))   # full sun_path, like ROCr
    squat.listen(1)
hip = ctypes.CDLL("libamdhip64.so")
assert hip.hipSetDevice(0) == 0
p = ctypes.c_void_p()
assert hip.hipMalloc(ctypes.byref(p), 2 << 20) == 0
h = ctypes.create_string_buffer(64)  # hipIpcMemHandle_t is 64 bytes
rc = hip.hipIpcGetMemHandle(h, p)
hip.hipGetErrorString.restype = ctypes.c_char_p
socks = [l.split()[-1].rstrip("@") for l in open("/proc/net/unix") if "@hsa" in l]
print(f"IPCGET mode={mode} legacy={os.environ.get('HSA_ENABLE_IPC_MODE_LEGACY','unset')} pid={os.getpid()} "
      f"rc={rc} ({hip.hipGetErrorString(rc).decode()}) hsa_sockets={socks}", flush=True)
