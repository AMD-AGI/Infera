"""E10 microbench: device -> host write-back bandwidth of sgl_kernel.transfer_kv_all_layer_mla_lf_pf (the HiCache
write_back eviction copy on ROCm) versus block_quota, in the production shapes of one GLM-5.2 prefill PP rank:
  KV       item = token, 576 B per layer, 20 layers, host page_first rows of 11,520 B
  indexer  item = page (64 tokens), 8,448 B per layer, 5 layers
One eviction = NODES calls of TOKENS_PER_NODE tokens (scattered 64-token pages on both sides), each followed by a
synchronize, as the eviction walk does per victim. Correctness for every setting: the device data is re-randomized,
then every destination row is compared byte for byte; at the end every host row outside the destinations must still
hold the sentinel (no stray writes).
  python3 bench_hicache_backup.py [--quotas 1 2 4 8 16 32 64] [--warps 16] [--nodes 75] [--tokens-per-node 7104]"""

import argparse
import time

import torch
from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla_lf_pf

ap = argparse.ArgumentParser()
ap.add_argument("--quotas", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
ap.add_argument("--warps", nargs="*", type=int, default=[16])
ap.add_argument("--nodes", type=int, default=75)
ap.add_argument("--tokens-per-node", type=int, default=7104)
ap.add_argument("--dev-tokens", type=int, default=4 << 20)
ap.add_argument("--host-tokens", type=int, default=2 << 20)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
dev = torch.device("cuda")
PAGE = 64
SENTINEL = 0x5A


def page_runs(num_pages_total, num_pages, gen):
    """Random distinct pages, expanded to token indices (64 consecutive tokens each)."""
    pages = torch.randperm(num_pages_total, generator=gen)[:num_pages]
    return (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)


def run_case(name, layers, item, items_per_node, dev_items, host_items, gen):
    layout = item * layers
    dev_bufs = [torch.empty((dev_items, item), dtype=torch.uint8, device=dev) for _ in range(layers)]
    ptrs = torch.tensor([b.data_ptr() for b in dev_bufs], dtype=torch.uint64, device=dev)
    t0 = time.time()
    host = torch.full((host_items * layout,), SENTINEL, dtype=torch.uint8).pin_memory()
    print(f"[{name}] pinned host {host.numel() / 1e9:.1f} GB in {time.time() - t0:.0f} s; "
          f"device {sum(b.numel() for b in dev_bufs) / 1e9:.1f} GB", flush=True)
    per_page = item == 8448
    total = a.nodes * items_per_node
    if per_page:
        src_all = torch.randperm(dev_items, generator=gen)[:total]
        dst_all = torch.randperm(host_items, generator=gen)[:total]
    else:
        src_all = page_runs(dev_items // PAGE, total // PAGE, gen)
        dst_all = page_runs(host_items // PAGE, total // PAGE, gen)
    src = [x.to(dev) for x in src_all.split(items_per_node)]
    dst = [x.to(dev) for x in dst_all.split(items_per_node)]
    nbytes = total * layout
    dst_all_dev = dst_all.to(dev)
    src_all_dev = src_all.to(dev)
    results = []
    for warps in a.warps:
        for q in a.quotas:
            for b in dev_bufs:
                b.random_(0, 256, generator=None)
            torch.cuda.synchronize()
            times = []
            for rep in range(a.reps):
                t = time.perf_counter()
                for s, d_ in zip(src, dst):
                    transfer_kv_all_layer_mla_lf_pf(src_layers=ptrs, dst=host, src_indices=s, dst_indices=d_,
                                                    item_size=item, dst_layout_dim=layout, num_layers=layers,
                                                    block_quota=q, num_warps_per_block=warps)
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            best = min(times)
            got = host.view(host_items, layers, item).index_select(0, dst_all).to(dev)
            exp = torch.stack([b.index_select(0, src_all_dev) for b in dev_bufs], dim=1)
            ok = torch.equal(got, exp)
            results.append((warps, q, best, ok))
            print(f"[{name}] warps {warps:2d} block_quota {q:3d}: one eviction ({a.nodes} calls, {nbytes / 1e9:.2f} GB) "
                  f"{best * 1e3:7.1f} ms = {nbytes / best / 1e9:5.1f} GB/s (reps {', '.join(f'{x * 1e3:.0f}' for x in times)} ms); "
                  f"all destination rows exact: {ok}", flush=True)
            del got, exp
    mask = torch.ones(host_items, dtype=torch.bool)
    mask[dst_all] = False
    rest = host.view(host_items, layout)[mask]
    stray = int((rest != SENTINEL).any(dim=1).sum())
    print(f"[{name}] host rows outside the destinations still holding the sentinel: "
          f"{rest.shape[0] - stray} / {rest.shape[0]} (stray writes: {stray})", flush=True)
    del host, dev_bufs, rest
    return results


gen = torch.Generator().manual_seed(0)
print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
run_case("KV", 20, 576, a.tokens_per_node, a.dev_tokens, a.host_tokens, gen)
run_case("indexer", 5, 8448, a.tokens_per_node // PAGE, a.dev_tokens // PAGE, a.host_tokens // PAGE, gen)
