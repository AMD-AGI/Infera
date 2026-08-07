# Environment — GLM-5.2 MIX fixlen sweep

## Hardware

| item | value |
|---|---|
| Cluster | AMD crsuse spur (`crsuse2-m2m`, `amd-spur` partition) |
| Node | `crsuse2-m2m-036` (single node; slurm job **44901**) |
| Data-plane IP | `10.245.148.191` |
| NIC | `ens3` |
| GPU | 8 × AMD Instinct **MI355X** (gfx950) |
| Deployment | single-node aggregated (MIX) — **no PD, no mooncake, no RDMA** |

MIX is single-node: there is no KV transfer / RDMA fabric to capture. Only the
data-plane NIC (`ens3`) matters, and only because the worker binds
`SGLANG_LOCAL_IP_NIC` / `GLOO_SOCKET_IFNAME` to it.

## Software

| item | value |
|---|---|
| Repo branch | `dev.yihou.glm52.mix.experiment` |
| Repo commit | `d1a97b2b17732664580c6cbf7c5aa1ff4bef51bd` |
| Engine image | `infera/engine-sglang:final-pr` — **built on the node** from branch source |
| Dockerfile | `deploy/docker/Dockerfile.sglang` (in-tree at the above SHA) |
| Base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (ROCm 7.2.0, MI35x) |
| etcd | `quay.io/coreos/etcd:v3.5.14` (needs `--entrypoint /usr/local/bin/etcd`) |

> **Pin the base digest.** This packup records the base by its floating tag
> `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`; the `sha256` digest was not captured
> at build time. To make this fully reproducible, resolve and record it on the node:
> `docker inspect --format '{{index .RepoDigests 0}}' lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`

### Build-time patches baked into the image (in-tree, applied by the Dockerfile)

The image bakes the branch's DSA + ROCm-hicache patches; they are **not** duplicated
into this packup because they live in the repo at the pinned SHA. Paths:

- `deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`
  — DSA indexer DP-padded-rows patch. Carries the **P1V3 `_p1v2_rows` reconciliation**
  (`min(real, padded)`) for the reversed IDLE-rank case under MTP draft-extend. The
  `_p1v2_rows` **bytecode marker was verified at build** (identifier, not a comment).
- `deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py`
  — routes hicache `ALLOC_MEMORY_FUNCS` to `pin_memory` (`hipHostMalloc`); without it
  gfx950 (`xnack-`) hard-aborts with a GPU memory-access fault on the host VA.
- `deploy/docker/scripts/apply_sglang_dsa_patches.sh` — applies the DSA arm at build.

### Key engine config the numbers are sensitive to

TP8 · DPA **off** · EP8 · MTP EAGLE (steps 3 / topk 1 / draft 4) + `--disable-custom-all-reduce`
· kvd on (hicache-size 16; kvd daemon L2 32 GiB RAM + L3 file 64 GiB) · KV dtype
`fp8_e4m3` · context-length 262144 · chunked-prefill 8192 · `--mem-fraction-static 0.85`
· `--reasoning-parser glm45`.

DSA-ROCm env (mandatory on gfx950 — without it the model serves garbage):
`SGLANG_OPT_USE_TILELANG_INDEXER=1`, `SGLANG_OPT_USE_TOPK_V2=0`, `SGLANG_OPT_USE_JIT_NORM=0`,
`SGLANG_USE_AITER=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=0`. (Full env in `scripts/mix_worker.sh`.)

## Dependencies (absolute paths, not in repo)

| dep | path |
|---|---|
| Weights | `/shared_nfs/GLM-5.2-MXFP4` (HF cache `/shared_nfs/models--amd--GLM-5.2-MXFP4`) |
| Bind-mount | `/shared_nfs` mounted into the container at the same path |
| Served name | `glm5.2-mxfp4` (router/bench `--model`); tokenizer = the weights dir |

## Secrets (names + source only — never values)

- **Docker registry login** for pulling the base image / pushing the built image —
  from the operator's registry creds, written to `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`
  (crsuse `/tmp` is root-owned; `/var/tmp` is writable by `yihou`).
- **Cluster access** — spur allocation on `crsuse2-m2m` (see `spur-cluster-usage`);
  commands run via `spur exec <job> …`.
- No API keys / etcd auth / S3 creds needed — etcd runs unauthenticated on the node's
  own IP.
