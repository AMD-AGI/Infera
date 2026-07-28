# GLM-5.2-MXFP4 sglang mooncake PD on crsuse spur (mlx5+dmabuf) — results

Reproduced the vultr glm5.2 mooncake+DPA+MTP recipe on the **spur** cluster, swapping the
transport to mlx5 + dmabuf (spur has no peermem; mlx5 has ODP -> dmabuf dynamic-attach, no pin).

- Image: `infera.yihou.sglang.1.0` (= `Dockerfile.sglang.dmabuf`, mooncake rebuilt with
  USE_HIP_DMABUF + HIP-transport gate). Saved to `/home/yihou/infera.yihou.sglang.1.0.tar` (27G).
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
- Transport (verified): `installTransport type=rdma` on `mlx5_0` GID idx 3, 0 TCP fallback,
  0 KVTransferError. Forced via MC_MS_FILTERS=mlx5_0, MC_GID_INDEX=3, DMABUF=1.
- Topology: prefill=node069, decode=node321, TP8 each.

## Config A — mooncake RDMA + DP-attention (both legs symmetric DP8), no MTP
- Correctness: **4/4** (Paris/Beijing/4/Jupiter, coherent CoT) via router :8002
- conc=128 (1k/1k, 512 prompts): **512/512, 0 fail**
  - total 7218 tok/s, out 3609 tok/s, median TPOT 31.3ms (P99 32.3), median TTFT 3785ms, conc 126.3

## Config B — mooncake RDMA + MTP (EAGLE steps=3 on decode), TP8, no DPA
- Correctness: **4/4** via router; draft head loads clean (no 3072/6144 shape crash, nextn patch OK)
- Spec-dec active: accept len 2.80-2.98 (of 4), accept rate 0.60-0.66
- conc=128 (1k/1k, 512 prompts): **512/512, 0 fail**
  - total 8990 tok/s, out 4495 tok/s, median TPOT **19.2ms** (1.6x faster decode vs A), median TTFT 835ms

## Config C — infera kvd + kv-aware routing (single-node mix, node069)

Stack: etcd (v3 HTTP gateway) + infera-kvd daemon + one infera.engine.sglang worker (etcd
self-register, kv-events on) + infera.server router (`--router-policy kv-aware`). infera pkg
pip-installed into the container from repo source.

- **kv-aware routing: WORKS.** 4/4 correctness via router :8100. Prefix-reuse test (2029-tok shared
  prefix, sent twice): req1 cold 1.87s (0 cached) -> req2 warm **0.31s, cached_tokens=1984/2029**
  (6x faster TTFT). Router log proves the policy fired:
  `pick policy=kv-aware picked=<worker> cache_hits=31 request_blocks=31 w_overlap=1.00`.
  Worker self-registered in etcd (`/infera/workers/<ip>:30000`, `kv=yes`).
- **kvd daemon: WORKS.** L3 self-check WRITE 20.7 GB/s READ 43.7 GB/s; listening on UDS. Wired into
  sglang: all 8 TP ranks log `Creating dynamic storage backend 'infera-kvd'`.
- **kvd L3 write-back GPU-faults on gfx950.** With `--enable-hierarchical-cache` (auto-enabled by
  `--infera-kvd-socket`), the first real prefix-cache *write* triggers `Memory access fault by GPU
  node-N` and kills the worker -- with BOTH `kernel` and `direct` hicache io backends. So the kvd L3
  tier is not usable for GLM-5.2's DSA/MLA KV layout on this GPU (sglang 0.5.15.post1). kv-aware
  routing (needs only kv-events, not hicache) is unaffected -- hence Config C uses the engine's
  NATIVE radix prefix cache (`infera_worker_nokvd.sh`, `--enable-cache-report`) for the live cache
  demo, and separately shows the kvd daemon + backend-registration healthy.
- Also: a large hicache host allocation can wedge a spur node at kernel level (D-state on
  `__flush_workqueue`, un-killable) -- always cap with a small fixed `--hicache-size`.

## Note: DPA + MTP together does NOT work
Combining `--enable-dp-attention` with EAGLE MTP crashes the decode leg during draft-extend:
`RuntimeError: Expected lengths.size(0) == B` in the DSA indexer fast_topk_v2
(dsa_topk_backend.py). The DSA sparse-indexer topk kernel doesn't handle the DP-attention batch
layout in the EAGLE draft-extend path. The reference kits (06=MTP, 07=DPA) ran these SEPARATELY and
never fused them. So on spur they're validated separately, each passing conc=128.
