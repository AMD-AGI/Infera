# Environment — GLM-5.2-MXFP4 sglang mooncake PD on crsuse spur

Captured 2026-07-28 (UTC) on the AMD `crsuse2-m2m` (Spur) cluster.

## Hardware (nodes used)

| Node | Role | mlx5 IP (ens3) | ionic NICs | mlx5 |
|------|------|----------------|-----------|------|
| crsuse2-m2m-069 | PD prefill (DPA) | 10.245.155.58 | 8× (no ODP) | mlx5_0 ACTIVE (has ODP) |
| crsuse2-m2m-321 | PD decode / single-node kv-aware | 10.245.149.213 | 8× (no ODP) | mlx5_0 ACTIVE (has ODP) |
| crsuse2-m2m-243 | (discarded — bad node, torch.cuda.is_available()=False) | | | |

- GPU: 8× AMD Instinct MI355X (gfx950, CDNA4), ~288 GiB HBM/card, per node.
- CPU: 236 logical cores. RAM: ~2751 GB. Host kernel: 6.8.0-107-generic. Host ROCm: 7.2.0.
- Scheduler: Spur 0.5.1, single partition `amd-spur`. Access via `spur exec <job>` (ssh to compute
  nodes is banned). Nodes held with `-q amd-burst-qos -N1 -G8`.

## RDMA fabric — the crucial cluster fact

Each node: **8× AMD Pensando ionic (NO ODP) + 1× Mellanox mlx5_0 (HAS ODP)**. IP only on `ens3`
(the mlx5 netdev). **No peermem kernel module** → a bare `ibv_reg_mr` on a GPU pointer EFAULTs.
Therefore the ONLY GPUDirect path is **dma-buf via mlx5** (`ibv_reg_dmabuf_mr`; mlx5's ODP →
dynamic attach → no pin, no KV-pool doubling). ionic+dmabuf would pin the whole KV pool (no ODP)
and crash. mlx5 RoCEv2 routable GID = **index 3** (IPv4-mapped, v2). Verified transport:
`installTransport, type=rdma` on `mlx5_0` GID idx 3 (see results/transport_evidence.txt).

This is the opposite of the vultr cluster the original recipe was proven on (ionic+peermem, dmabuf
OFF, GID1) — see notes.md §"two clusters".

## Software

- **Image: `infera.yihou.sglang.1.0`** (local id `sha256:347bcd45da0d`, 108 GB).
  - Built on-node from `infera.yihou.dev` repo `deploy/docker/Dockerfile.sglang.dmabuf`.
  - Base: `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`
    (sha256:`40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d`).
  - Rebuilds bundled Mooncake (#2682) with `USE_HIP_DMABUF=ON` (self-verified:
    `DMABUF_COMPILED_IN=yes`, `LINKS_HSA=yes`) + the HIP-transport gate (installTransport("hip")
    behind `MC_ENABLE_HIP_TRANSPORT`, default off → cross-node stays RDMA).
  - Image ROCm 7.2.0, sglang 0.5.15.post1, torch 2.9.1+rocm7.2.0, Python 3.10 (/opt/venv).
  - Saved to NFS: `/home/yihou/infera.yihou.sglang.1.0.tar` (27 GB compressed).
  - For the kvd + kv-aware test, the **infera** Python package was pip-installed into the running
    container from the `infera.yihou.glm5.2.mxfp4` repo source (`amd-infera 1.0.0`; deps from pypi —
    the Rust router in `Dockerfile.sglang` can't build on-node because crates.io is blocked).
- **etcd**: `quay.io/coreos/etcd:v3.5.14` (kv-aware worker discovery via its v3 HTTP/JSON gateway).

## Repo / code

- `infera.yihou.dev` @ `6b50e13` branch `yihou.dev.sglang.mooncake.experiment`
  (has Dockerfile.sglang.dmabuf + build_mooncake_dmabuf.sh + mooncake_cpp patches).
- `infera.yihou.glm5.2.mxfp4` @ `c9b1f58` branch `yihou.dev.glm5.2.mxfp4.experiment`
  (has the infera package, Dockerfile.sglang, Dockerfile.kvd; this delivery lives here).

## Model (external dependency — not in git)

- **GLM-5.2-MXFP4**: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
  (also the HF cache `/shared_nfs/models--amd--GLM-5.2-MXFP4`, snapshot `386bd0e4…`).
  `GlmMoeDsaForCausalLM`: MLA + DSA sparse-attn indexer, 78 layers, head_dim 192, quark MXFP4,
  282 shards ~408 GiB. Shared NFS, mounted on every node.

## Secrets / access

- **None required.** Public base image on docker.io, model on shared `/shared_nfs`, spur cluster
  identity is automatic. Node→node image move via `docker save`→NFS tar→`docker load` (no ssh).

## Filesystem / mounts

| Mount | Note |
|-------|------|
| `/shared_nfs` | shared NFS — model here, on every node |
| `/home/yihou` | user NFS home — scripts, logs, image tar; mounted in containers |
