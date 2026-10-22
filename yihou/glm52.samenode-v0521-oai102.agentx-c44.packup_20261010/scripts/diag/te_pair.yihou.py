#!/usr/bin/env python3
"""Real mooncake TransferEngine pair test (HIP transport, 0 HCAs), mirroring sglang PD.
dst <dir> <host> <dev>: 'decode' side. L per-layer torch.zeros(64 MiB) on cuda:<dev> + one INTERIOR region
    (a 64 MiB slice at +4 MiB+4 KiB of a 128 MiB tensor). batch_register_memory all; publish
    session + ptrs; wait; verify each region got the source pattern (and where it actually landed).
src <dir> <host> <dev>: 'prefill' side. Same-size source tensors filled 0x30+i; register; for each
    dst region transfer_sync_write(session, src, dst_ptr, len); report rc."""
import json, os, sys, time
import torch
from mooncake.engine import TransferEngine

role, d, host, dev = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
L, SZ = 8, 64 << 20
MODE = "legacy" if os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY") == "1" else "default"
torch.cuda.set_device(dev)
te = TransferEngine()
rc = te.initialize(host, "P2PHANDSHAKE", "rdma", os.environ.get("YIHOU_IB", "ionic_0"))
sess = f"{host}:{te.get_rpc_port()}"
print(f"[{role} {MODE}] init rc={rc} session={sess} dev={dev} HVD={os.environ.get('HIP_VISIBLE_DEVICES')}", flush=True)
if role == "dst":
    layers = [torch.zeros(SZ, dtype=torch.uint8, device="cuda") for _ in range(L)]
    holder = torch.full((128 << 20,), 0x11, dtype=torch.uint8, device="cuda")
    ioff = (4 << 20) + 4096
    interior = holder[ioff:ioff + SZ]; interior.zero_()
    regs = layers + [interior]
    torch.cuda.synchronize()
    ptrs, lens = [t.data_ptr() for t in regs], [t.numel() for t in regs]
    rc = te.batch_register_memory(ptrs, lens)
    print(f"[dst {MODE}] batch_register rc={rc} n={len(ptrs)} interior_ptr={ptrs[-1]:#x} holder_base={holder.data_ptr():#x}", flush=True)
    json.dump(dict(session=sess, ptrs=ptrs, lens=lens), open(os.path.join(d, "dst.tmp"), "w"))
    os.rename(os.path.join(d, "dst.tmp"), os.path.join(d, "dst.json"))
    t0 = time.time()
    while not os.path.exists(os.path.join(d, "src.done")) and time.time() - t0 < 180:
        time.sleep(0.2)
    torch.cuda.synchronize()
    for i, t in enumerate(regs):
        want = 0x30 + i
        ok = bool((t[:4096] == want).all() and (t[-4096:] == want).all())
        print(f"[dst {MODE}] region{i}{' (INTERIOR)' if i == L else ''} correct={ok} first={int(t[0])} want={want}", flush=True)
    # where did the interior write land? scan holder at 4 KiB granularity for the pattern
    want = 0x30 + L
    hits = (holder.view(-1, 4096)[:, 0] == want).nonzero().flatten()
    if hits.numel():
        print(f"[dst {MODE}] interior pattern found in holder at byte offset {int(hits[0]) * 4096} .. {int(hits[-1]) * 4096 + 4095} (registered at {ioff})", flush=True)
    print(f"[dst {MODE}] exit", flush=True)
else:
    info = None
    t0 = time.time()
    while info is None and time.time() - t0 < 240:
        try:
            info = json.load(open(os.path.join(d, "dst.json")))
        except Exception:
            time.sleep(0.2)
    srcs = [torch.full((n,), 0x30 + i, dtype=torch.uint8, device="cuda") for i, n in enumerate(info["lens"])]
    torch.cuda.synchronize()
    rc = te.batch_register_memory([s.data_ptr() for s in srcs], [s.numel() for s in srcs])
    print(f"[src {MODE}] batch_register rc={rc}", flush=True)
    for i, (s, p, n) in enumerate(zip(srcs, info["ptrs"], info["lens"])):
        t1 = time.time()
        r = te.transfer_sync_write(info["session"], s.data_ptr(), p, n)
        print(f"[src {MODE}] write region{i} rc={r} {(time.time() - t1) * 1e3:.1f} ms", flush=True)
    open(os.path.join(d, "src.done"), "w").write("x")
