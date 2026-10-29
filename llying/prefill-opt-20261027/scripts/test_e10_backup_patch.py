"""E10 smoke test inside the production image, with the patched pool_host/mla.py and dsa.py mounted at their
production paths: the mounted source is what gets imported (not a stale .pyc), the env knob reaches the module, and
the real backup_from_device_all_layer methods (page_first, kernel io backend, the branch the prefill takes on ROCm)
copy one eviction's worth of KV and indexer pages byte-exactly, in the time the microbench predicts.
  SGLANG_HICACHE_BACKUP_BLOCK_QUOTA=16 python3 test_e10_backup_patch.py"""

import inspect
import os
import time
import types

import torch

import sglang.srt.mem_cache.pool_host.dsa as dsa_mod
import sglang.srt.mem_cache.pool_host.mla as mla_mod
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices

dev = torch.device("cuda")
PAGE, NODES, TOK = 64, 75, 7104
print(f"env SGLANG_HICACHE_BACKUP_BLOCK_QUOTA={os.environ.get('SGLANG_HICACHE_BACKUP_BLOCK_QUOTA')}; "
      f"mla {mla_mod.__file__} quota {mla_mod._BACKUP_BLOCK_QUOTA}; dsa {dsa_mod.__file__} quota {dsa_mod._BACKUP_BLOCK_QUOTA}")
for cls in (mla_mod.MLATokenToKVPoolHost, dsa_mod.DSAIndexerPoolHost):
    src = inspect.getsource(cls.backup_from_device_all_layer)
    print(f"{cls.__name__}.backup_from_device_all_layer passes block_quota: {'block_quota=_BACKUP_BLOCK_QUOTA' in src}")
probe = torch.arange(128, device=dev)
print(f"maybe_dcp_kernel_indices is the identity at dcp_size 1: {torch.equal(maybe_dcp_kernel_indices(probe, 1, 0), probe)}")

gen = torch.Generator().manual_seed(1)


def runs(num_pages_total, num_pages):
    pages = torch.randperm(num_pages_total, generator=gen)[:num_pages]
    return (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)


dev_tok, host_tok = 4 << 20, 2 << 20
src_all = runs(dev_tok // PAGE, NODES * TOK // PAGE)
dst_all = runs(host_tok // PAGE, NODES * TOK // PAGE)
src = [x.to(dev) for x in src_all.split(TOK)]
dst = [x.to(dev) for x in dst_all.split(TOK)]

# KV: 20 layers x 576 B per token, host rows of 11,520 B (page_first)
kv_layers = [torch.randint(0, 256, (dev_tok, 576), dtype=torch.uint8, device=dev) for _ in range(20)]
kv_ptrs = torch.tensor([b.data_ptr() for b in kv_layers], dtype=torch.uint64, device=dev)
kv_host = torch.zeros(host_tok * 11520, dtype=torch.uint8).pin_memory()
kv_self = types.SimpleNamespace(
    dcp_size=1, dcp_rank=0, _is_device_layer_sharded=lambda dp: False, device_data_ptrs=kv_ptrs,
    device_kv_buffers=kv_layers, layout="page_first", can_use_jit=False, can_use_write_back_jit=False,
    device_rows_packed=True, _require_packed_device_rows=lambda name: None, kv_buffer=kv_host,
    token_stride_size=576, layout_dim=11520, layer_num=20)

# indexer: 5 layers x 8,448 B per 64-token page, host rows of 42,240 B (page_first)
ix_layers = [torch.randint(0, 256, (dev_tok // PAGE, 8448), dtype=torch.uint8, device=dev) for _ in range(5)]
ix_ptrs = torch.tensor([b.data_ptr() for b in ix_layers], dtype=torch.uint64, device=dev)
ix_host = torch.zeros(host_tok // PAGE * 42240, dtype=torch.uint8).pin_memory()
ix_self = types.SimpleNamespace(
    page_size=PAGE, _is_device_layer_sharded=lambda dp: False, indexer_page_stride_size=8448, layout="page_first",
    can_use_write_back_jit=False, index_k_device_ptrs=ix_ptrs, index_k_with_scale_buffer=ix_host,
    indexer_layout_dim=42240, layer_num=5)
ix_self._get_indexer_page_indices = types.MethodType(dsa_mod.DSAIndexerPoolHost._get_indexer_page_indices, ix_self)

for name, fn, obj in (("KV", mla_mod.MLATokenToKVPoolHost.backup_from_device_all_layer, kv_self),
                      ("indexer", dsa_mod.DSAIndexerPoolHost.backup_from_device_all_layer, ix_self)):
    best = None
    for rep in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for s, d_ in zip(src, dst):
            fn(obj, None, d_, s, "kernel")
            torch.cuda.synchronize()
        el = time.perf_counter() - t
        best = el if best is None else min(best, el)
    print(f"{name}: one eviction through the real method, {NODES} victims: {best * 1e3:.1f} ms")

got = kv_host.view(host_tok, 20, 576).index_select(0, dst_all).to(dev)
exp = torch.stack([b.index_select(0, src_all.to(dev)) for b in kv_layers], dim=1)
print(f"KV destination rows exact: {torch.equal(got, exp)}")
mask = torch.ones(host_tok, dtype=torch.bool)
mask[dst_all] = False
print(f"KV rows outside the destinations untouched: {not bool(kv_host.view(host_tok, 11520)[mask].any())}")
sp, dp = src_all.view(-1, PAGE)[:, 0] // PAGE, dst_all.view(-1, PAGE)[:, 0] // PAGE
got = ix_host.view(host_tok // PAGE, 5, 8448).index_select(0, dp).to(dev)
exp = torch.stack([b.index_select(0, sp.to(dev)) for b in ix_layers], dim=1)
print(f"indexer destination pages exact: {torch.equal(got, exp)}")
pmask = torch.ones(host_tok // PAGE, dtype=torch.bool)
pmask[dp] = False
print(f"indexer pages outside the destinations untouched: {not bool(ix_host.view(host_tok // PAGE, 42240)[pmask].any())}")
