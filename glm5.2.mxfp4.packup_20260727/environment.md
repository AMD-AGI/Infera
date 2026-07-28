# Environment (shared across all 7 experiments)

Captured 2026-07-27 on the MI355X cluster via the jump host `root@149.28.124.225`.

## Hardware — nodes used

| Node | Role(s) | Data-plane IP (NIC enp193s0f1np1) | Kernel | ROCm drv | ionic kmod | libionic (host) |
|------|---------|-----------------------------------|--------|----------|-----------|-----------------|
| chi2879 | mix / PD decode (all phases) | 10.2.122.10/23 | 6.8.0-124 | 6.16.13 | 26.03.3.001 | libionic.so.1.1.54.0-187 |
| chi2878 | PD prefill (mori, mooncake) | 10.2.122.3/23 | 6.8.0-134 | 6.16.6 | 25.12.15.001 | libionic.so.1.1.54.0-184 |
| chi2832 | PD prefill (PD-MTP only) | 10.2.122.79/23 | 6.8.0-107 | 6.16.13 | 26.03.3.001 | libionic.so.1.1.54.0-187 |

- **GPU**: 8× AMD Instinct MI355X (gfx950), 288 GB HBM/card, per node.
- **CPU/RAM**: AMD EPYC 9575F 64-Core (2 sockets, 256 threads), ~3 TB RAM.
- **RDMA fabric**: 8× ionic RoCE-v2 NICs per node (ionic_0..7). Routable GID at **index 1**
  on all three nodes (idx2 empty). Single-rail ib_write_bw ≈ 335–338 Gb/s (healthy).
  - chi2878↔chi2879 ionic_0 = 337.98 Gb/s; chi2832↔chi2879 ionic_0 = 335.13 Gb/s.
- **Kernel RDMA-GPU-direct flags** (checked on all nodes): `CONFIG_PCI_P2PDMA=y`,
  `CONFIG_DMABUF_MOVE_NOTIFY=y` both present. `ib_peer_mem` loaded on chi2878/chi2879,
  NOT on chi2832. (NOTE: this differs from jiejing's chi2866 which lacked P2PDMA — see
  03_pd_mooncake/notes.md; we did NOT re-test mooncake RDMA on these nodes, only TCP.)

## Software

- **Image A (01/02/04/05)**: `rocm/infera:sglang-v0.1.0-rc6` — sglang **0.5.15.post1** (knows GLM-5.2
  head_dim=192; 0.5.12/0.5.14 cannot load GLM-5.2).
  - Base image digest: `sha256:58d21d3300e9502967f4d1cf5fefc129372087e6bbce6dad80d3efa7e8f57af6`
  - Local image ID: `1b63f3d6ccb5`. Pull: `docker pull rocm/infera:sglang-v0.1.0-rc6`.
- **Image B (03/06/07 — mooncake)**: `infera/engine-sglang:pd-unified` — sglang **0.5.15.post1**, but
  the bundled mooncake (#2682) rebuilt per **Infera PR #19** so PD KV-transport is runtime-decided:
  HIP transport OFF by default (`MC_DISABLE_HIP_TRANSPORT=1`) → cross-node RDMA; dma-buf compiled in
  but OFF by default (bare `ibv_reg_mr`+peermem). Local image ID `f8ec2d627392`, 78.6 GB.
  - **NOT on a public registry** — it's a local build (chi2798/chi2878 had it). Distribute node→node
    by streaming `docker save | ssh <dst> docker load` (NFS `docker save` is very slow for 78 GB).
  - Build source: Infera repo `deploy/docker/Dockerfile.sglang` on the PR #19 branch. The full method
    is in the reference packup `sglang_unified_pd_test.packup_20260727`.
- **Aux images**: `quay.io/coreos/etcd:v3.5.14`, `nats:2.10` (for the rc6 infera.server router path,
  02/05). The pd-unified mooncake path (03/06) uses `sglang_router` in-container — no etcd/nats.
- **Repo**: `infera.glm5.2.mxfp4`, branch `yihou.dev.glm5.2.mxfp4`, commit `2df2fed` (at packup time).
  The engine code that runs is INSIDE the image (`infera.engine.sglang` + sglang), not this repo —
  the repo provides the example kit under `examples/deepseek_v4/` that inspired these scripts.

## Model (external dependency — not in repo)

- **GLM-5.2-MXFP4**: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST mount, world-readable).
  `GlmMoeDsaForCausalLM`: MLA (kv_lora_rank 512) + DSA sparse-attn indexer (index_topk 2048),
  78 layers, 256 experts, head_dim 192, quark MXFP4 (fp4 group_size 32, e8m0 scale;
  dense layers 0-2 + all attn/indexer/lm_head + MTP layer 78 EXCLUDED → bf16). 282 shards, ~408 GiB.

## Secrets / access (names only — NOT values)

- **Cluster SSH**: via jump host `root@149.28.124.225` (chi2866), then `ssh <node>` from there.
  Arrange your own ProxyJump / key access.
- **Docker registry**: `rocm/infera` images are on public Docker Hub — no login needed for the
  rc6 image used here.
- **Slurm hold**: nodes were held via slurm (see slurm-cluster skill); ownership signalled by
  job name. Not required to reproduce if you already have the nodes.
