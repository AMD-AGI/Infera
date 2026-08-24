#!/usr/bin/env python3
"""MVP: can two mooncake TransferEngines on the SAME host talk over RDMA?

This is the premise the whole single-node TP4+TP4 PD plan rests on. If two
engines in two processes on one box cannot register buffers and transfer
between each other, the plan is dead and there is no point launching a
400 GB model to find that out.

Deliberately does NOT involve sglang: it exercises the same mooncake engine
sglang's MooncakeKVManager drives, so a failure here is unambiguous.

Run twice on one host, in two shells:
    python3 mvp_mooncake_loopback.py target   <local_ip> <device>
    python3 mvp_mooncake_loopback.py initiator <local_ip> <device> <target_session>
"""

import ctypes
import sys
import time

from mooncake.engine import TransferEngine

N_BYTES = 8 << 20


def make_engine(local_ip, device):
    e = TransferEngine()
    # hostname must be unique per process on the same box, hence the port suffix.
    rc = e.initialize(local_ip, "P2PHANDSHAKE", "rdma", device)
    if rc != 0:
        sys.exit(f"initialize failed rc={rc}")
    return e


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    role, local_ip, device = sys.argv[1], sys.argv[2], sys.argv[3]

    eng = make_engine(local_ip, device)
    sid = f"{local_ip}:{eng.get_rpc_port()}"
    print(f"[{role}] session_id = {sid}", flush=True)

    buf = ctypes.create_string_buffer(N_BYTES)
    addr = ctypes.addressof(buf)
    rc = eng.register_memory(addr, N_BYTES)
    print(f"[{role}] register_memory({addr:#x}, {N_BYTES}) rc={rc}", flush=True)
    if rc != 0:
        sys.exit(f"[{role}] register_memory FAILED rc={rc}")

    if role == "target":
        ctypes.memset(addr, 0xAB, N_BYTES)
        print(f"[target] buffer filled 0xAB at {addr:#x}; "
              f"pass this to the initiator:", flush=True)
        print(f"TARGET_SESSION={sid} TARGET_ADDR={addr:#x}", flush=True)
        print("[target] holding for 180s ...", flush=True)
        time.sleep(180)
    else:
        if len(sys.argv) < 6:
            sys.exit("initiator needs <target_session> <target_addr>")
        tgt_sid = sys.argv[4]
        tgt_addr = int(sys.argv[5], 16)
        ctypes.memset(addr, 0x00, N_BYTES)
        t0 = time.perf_counter()
        rc = eng.transfer_sync_read(tgt_sid, addr, tgt_addr, N_BYTES)
        dt = time.perf_counter() - t0
        print(f"[initiator] transfer_sync_read rc={rc} in {dt*1e3:.2f} ms", flush=True)
        if rc != 0:
            sys.exit(f"[initiator] TRANSFER FAILED rc={rc}")
        got = bytes(buf[:16])
        ok = all(b == 0xAB for b in bytes(buf[:N_BYTES]))
        gbps = (N_BYTES * 8 / 1e9) / dt
        print(f"[initiator] first16={got.hex()} all_0xAB={ok} ~{gbps:.1f} Gb/s",
              flush=True)
        print("RESULT: PASS" if ok else "RESULT: FAIL (data mismatch)", flush=True)


if __name__ == "__main__":
    main()
