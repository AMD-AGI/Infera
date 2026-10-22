#!/usr/bin/env python3
"""sglang PR #42553 smoke (one GPU): page_first MLA host pool from the PATCHED
pool_host/common.py (mounted over the image's) with HSA_USERPTR_FOR_PAGED_MEM=0 ->
hipHostMalloc (GTT). Device->host (transfer_kv_all_layer_mla_lf_pf) and host->device
(transfer_kv_per_layer_mla_pf_lf) on token slots near the START, MIDDLE and END of the pool,
the same AOT kvcacheio kernels the MLA host pool uses (mla.py:961-971, :713-722).
Prints alloc/free timings and GTT usage. usage: gtt_smoke.yihou.py <pool_GB> [layers] [item_bytes]
"""
import gc, glob, os, sys, time

import torch

GB = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
L = int(sys.argv[2]) if len(sys.argv) > 2 else 78          # layers
ITEM = int(sys.argv[3]) if len(sys.argv) > 3 else 656      # bytes per token per layer (fp8 MLA)
LAYOUT = ITEM * L                                           # page_first token stride
N = 512                                                     # tokens touched per region


def log(*a):
    print(*a, flush=True)


def gtt():
    out = []
    for d in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")):
        try:
            out.append(int(open(d).read()) / 1e9)
        except Exception:
            pass
    return f"gtt_used_GB(sum over {len(out)} cards)={sum(out):.1f}" if out else "gtt_used n/a"


log(f"HSA_USERPTR_FOR_PAGED_MEM={os.environ.get('HSA_USERPTR_FOR_PAGED_MEM')} pool={GB} GB L={L} item={ITEM}")
torch.cuda.set_device(0)
from sglang.srt.mem_cache.pool_host import common as hc  # patched file via mount
from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla_lf_pf, transfer_kv_per_layer_mla_pf_lf

log("patched common.py:", hasattr(hc, "_alloc_hip_host_tensor"), hc.__file__)
size = int(GB * 1e9) // LAYOUT
dims = (size, L, 1, ITEM)
log(f"dims={dims} bytes={size * LAYOUT} ({size * LAYOUT / 1e9:.2f} GB) {gtt()}")
t0 = time.time()
host = hc.ALLOC_MEMORY_FUNCS["cuda"](
    dims, dtype=torch.uint8, device="cpu", pin_memory=True,
    allocator=hc.HostTensorAllocator(), registration_granularity_bytes=64 * LAYOUT,
)
t_alloc = time.time() - t0
regs = getattr(host, hc._CUDA_HOST_REGISTERED_RANGES_ATTR, None)
log(f"ALLOC {t_alloc:.2f}s path={'hipHostMalloc' if regs == [] else 'hostRegister nreg=%s' % (len(regs) if regs else '?')} "
    f"ptr={host.data_ptr():#x} {gtt()}")

# device per-layer KV (lf), N*3 tokens each
dev_tokens = 3 * N
dev = [torch.empty((dev_tokens, 1, ITEM), dtype=torch.uint8, device="cuda") for _ in range(L)]
regions = {"start": 0, "middle": size // 2, "end": size - N}
host_idx = torch.cat([torch.arange(o, o + N) for o in regions.values()]).to(torch.int64)
dev_idx = torch.arange(dev_tokens, dtype=torch.int64)


def pattern(layer, tok_idx):  # byte value per (layer, host token), broadcast over ITEM
    return ((tok_idx * 7 + layer * 31 + 5) % 251).to(torch.uint8)


for l in range(L):
    dev[l].copy_(pattern(l, host_idx).view(-1, 1, 1).expand(-1, 1, ITEM))
ptrs = torch.tensor([t.data_ptr() for t in dev], dtype=torch.uint64, device="cuda")
torch.cuda.synchronize()

# device -> host
t0 = time.time()
transfer_kv_all_layer_mla_lf_pf(src_layers=ptrs, dst=host, src_indices=dev_idx.cuda(),
                                dst_indices=host_idx.cuda(), item_size=ITEM,
                                dst_layout_dim=LAYOUT, num_layers=L)
torch.cuda.synchronize()
t_d2h = time.time() - t0
ok_h = True
for name, o in regions.items():
    got = host[o:o + N]                                     # (N, L, 1, ITEM) on CPU
    for l in (0, L // 2, L - 1):
        want = pattern(l, torch.arange(o, o + N)).view(-1, 1)
        good = bool((got[:, l, 0, :] == want).all())
        ok_h &= good
        if not good:
            log(f"  D2H MISMATCH region={name} layer={l}")
log(f"D2H {t_d2h * 1e3:.1f} ms host_data_ok={ok_h}")

# host -> device (into zeroed device buffers)
for t in dev:
    t.zero_()
torch.cuda.synchronize()
t0 = time.time()
for l in range(L):
    transfer_kv_per_layer_mla_pf_lf(src=host, dst=dev[l], src_indices=host_idx.cuda(),
                                    dst_indices=dev_idx.cuda(), layer_id=l, item_size=ITEM,
                                    src_layout_dim=LAYOUT)
torch.cuda.synchronize()
t_h2d = time.time() - t0
ok_d = all(bool((dev[l][:, 0, :].cpu() == pattern(l, host_idx).view(-1, 1)).all()) for l in range(L))
log(f"H2D {t_h2d * 1e3:.1f} ms device_data_ok={ok_d}")

t0 = time.time()
del host
gc.collect()
t_free = time.time() - t0
log(f"FREE (last view dropped -> hipHostFree) {t_free:.2f}s {gtt()}")
log(f"RESULT pool_GB={GB} alloc_s={t_alloc:.2f} free_s={t_free:.2f} d2h_ok={ok_h} h2d_ok={ok_d} "
    f"path={'hipHostMalloc' if regs == [] else 'hostRegister'}")
