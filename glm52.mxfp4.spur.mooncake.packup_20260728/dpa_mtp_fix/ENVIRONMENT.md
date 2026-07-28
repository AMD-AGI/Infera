# Environment — GLM-5.2 DSA "DPA + MTP" fix (Bug 1)

Captured 2026-07-28 UTC on the AMD **crsuse2-m2m (Spur)** cluster, inside the running
container (not inferred — every value below was read off the live node).

## Hardware

| Item | Value |
|------|-------|
| Nodes used | `crsuse2-m2m-207` (single-node mix + PD decode), `crsuse2-m2m-197` (PD prefill) |
| GPU | 8× **AMD Instinct MI355X** (gfx950, CDNA4), card model `0x75a3`, ~288 GiB HBM each |
| CPU | 236 logical cores |
| RAM | 2751 GB total |
| Host kernel | 6.8.0-107-generic |
| amdgpu / KFD driver | **6.14.14** |
| ROCm (in image) | **7.2.0** (`/opt/rocm/.info/version`) |

Node IPs (ens3 = the mlx5 netdev, the only NIC with an IP):
- `crsuse2-m2m-207` → **10.245.156.172**
- `crsuse2-m2m-197` → **10.245.158.91**

## RDMA fabric (the critical cluster fact)

Each node exposes **9 RDMA devices**: `ionic_0..ionic_7` + `mlx5_0`.

| Item | Value |
|------|-------|
| KV NIC used | **`mlx5_0`** (the only one with ODP) |
| `mlx5_0` fw_ver | **28.43.3608** |
| port state | `PORT_ACTIVE (4)`, `link_layer: Ethernet` (RoCEv2) |
| GID index | **3** (RoCEv2 routable, IPv4-mapped) |
| peermem | **absent** → bare `ibv_reg_mr` on a GPU pointer EFAULTs |
| GPUDirect path | **dma-buf** (`ibv_reg_dmabuf_mr`) via mlx5's ODP → dynamic attach, no pin |

The 8 ionic NICs have **no ODP**; dma-buf there would pin (and double) the whole KV pool.
Mooncake is therefore forced onto mlx5 only. **Verified in both PD legs**: 8×
`installTransport, type=rdma`, `Device mlx5_0 port 1 is available`, **0** hip transport,
**0** tcp fallback, **0** ionic mentions, **0** `KVTransferError`
(see `evidence/transport_evidence.txt`).

## Software

| Item | Value |
|------|-------|
| Image | **`infera.yihou.sglang.1.0`** |
| Image ID | `sha256:347bcd45da0dee1bc87f10c348e41f20ed56e11d23f9fead164cdef4e51dc970` |
| Base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (`sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d`) |
| Image tar (NFS) | `/home/yihou/infera.yihou.sglang.1.0.tar` (27 GB) |
| OS in image | Ubuntu 22.04.5 LTS |
| **sglang** | **0.5.15.post1** |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |
| triton | 3.6.0 |
| Python | 3.10 (`/opt/venv`) |
| Mooncake | rebuilt with `USE_HIP_DMABUF=ON` + HIP-transport gate |

The image is the same one used by the parent packup; it is built from
`infera.yihou.dev` `deploy/docker/Dockerfile.sglang.dmabuf`. The dmabuf rebuild is what
makes `ibv_reg_dmabuf_mr` reachable at all — the stock base image silently compiles that
branch out (see the parent kit's `notes.md` ★1 and the DSv4 kit's `GOTCHAS.md` ★1).

## Repos / code state

| Repo | Branch | Commit |
|------|--------|--------|
| `infera.yihou.glm5.2.mxfp4` (this delivery) | `yihou.dev.glm5.2.mxfp4.experiment` | `5293a1b` |
| `infera.yihou.dev` (Dockerfile + mooncake patches) | `yihou.dev.sglang.mooncake.experiment` | `6b50e13` |

The patched file lives **inside the container** at
`/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
(sglang is an editable source checkout in the image, not a wheel).

## Model (external dependency — not in git)

- **GLM-5.2-MXFP4**: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
  (~408 GiB, 282 shards, shared NFS mounted on every node).
  `GlmMoeDsaForCausalLM`: MLA + DSA sparse-attention indexer, 78 layers, head_dim 192,
  quark MXFP4, `index_topk=2048`, `index_share_for_mtp_iteration=True`.

## Absolute paths / files NOT in git

| Path | What | Note |
|------|------|------|
| `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | model weights | shared NFS, every node |
| `/home/yihou/infera.yihou.sglang.1.0.tar` | 27 GB image tar | NFS; `docker load` per node |
| `/home/yihou/glm52_fix/` | **original experiment workspace** | NFS, preserved as-is (logs, sweep, patches) |
| `/home/yihou/glm52_spur/` | parent kit's workspace (scripts reused) | NFS, preserved |
| `/tmp/dockercfg` | writable DOCKER_CONFIG | per-node, recreated each session |

## Secrets / credentials

**None required.** The base image is public on docker.io, the model sits on shared
`/shared_nfs`, and Spur cluster identity is automatic (no token, no key). Node→node image
transfer goes through `docker save` → NFS tar → `docker load` because ssh between compute
nodes is banned — so no SSH key is needed either.

## Scheduler / access

- Spur 0.5.1, single partition `amd-spur`. Nodes held with
  `sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00 ~/hold_node.sh`.
- Access via `spur exec <jobid> <cmd>` — **ssh to compute nodes is blocked**.
- Jobs used in this run: `4540` (node 207), `4614` (node 197).
- `export DOCKER_CONFIG=/tmp/dockercfg` is required before every docker call.
