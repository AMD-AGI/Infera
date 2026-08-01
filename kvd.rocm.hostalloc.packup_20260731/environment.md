# Environment

## Hardware

| | |
|---|---|
| cluster | crsuse **spur** — scheduler is Spur, not stock Slurm (`spur exec <job> <cmd>`); **ssh to compute nodes is banned** |
| partition / qos | `amd-spur` / `amd-burst-qos` |
| prefill node | job **19254**, host `crsuse2-m2m-034`, `ens3` = **10.245.153.38** |
| decode node | job **19255**, host `crsuse2-m2m-227`, `ens3` = **10.245.151.183** |
| GPUs | 8 x **AMD Instinct MI355X** per node |
| GPU arch | **`gfx950:sramecc+:xnack-`** |
| `amdgpu.noretry` | **-1** |

`xnack-` is load-bearing for this bug, not incidental: with no page-migration
capability the GPU cannot fault-in an unmapped host page, so a bad address aborts the
process instead of silently working. An `xnack+` part might mask the same defect.

**Gap:** CPU model and RAM were not captured. The jobs reached walltime TIMEOUT before
this packup was assembled and the nodes were released, so they can no longer be read
back. Neither value affects this result — the bug is a pointer-mapping property of the
ROCm host allocator, reproducible on any single gfx950 node.

## Software

| | |
|---|---|
| ROCm / HIP | **7.2.26015-fc0010cf6a** (ROCm 7.2.0) |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| sglang | **0.5.15.post1** |

### Image (pinned by digest — read out of the build log, not the floating tag)

    infera/engine-sglang:kvaware-kvd
      FROM docker.io/infera/engine-sglang:kvaware-kvd-base
           @sha256:7112f4f511a26dbfa7d6673fdded703579d15f2e66c7e32ae84472ed3f26c3ac
      which is FROM docker.io/lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
           @sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d

Dockerfile: `deploy/docker/Dockerfile.sglang.kvaware-kvd` in the `infera.kv.fix`
worktree. Its stage-2 self-check must print `kvaware+kvd self-check OK`; it imports the
real modules (`InferaKvdBackend`, `wire_infera_kvd_backend`, `attach_to_radix_cache`)
rather than listing files, so it also catches a base-image bump that moves or renames
`HiCacheStorage` — the failure mode where the image builds, starts, and serves with no
L3 at all.

**Gap:** the final built image's own `sha256` Id was not captured before the nodes were
released. The two base digests above are pinned and exact, and the build is
deterministic from them plus the git SHA below.

### Repo state

| worktree | branch | commit | clean? |
|---|---|---|---|
| `infera.kv.fix` (image source) | `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` | **`52d71195498f9caaf8b84bcca3276a366b1e8010`** | **clean** |
| `infera.yihou.glm5.2.mxfp4` (this kit) | `yihou.dev.glm5.2.mxfp4.experiment` | `1c6d318da227b545ea83b7289bcb54fb54f94e37` | 4 untracked |

The image was built from `infera.kv.fix` at a **clean** tree, so that SHA fully
determines the infera code in the image.

## Transport (spur, not vultr — they are configured oppositely)

    IBDEV=mlx5_0         MC_GID_INDEX=3        MOONCAKE_DISABLE_HIP_DMABUF=0
    MC_MS_AUTO_DISC=0    MC_MS_FILTERS=mlx5_0  NIC=ens3

Verified in-run: `MC_FORCE_TCP` hits **0**; `mlx5_0` appears 26x in the prefill log.
The 8 ionic NICs on these nodes lack ODP and are not used for KV. On vultr the reverse
holds (ionic x8, GID 1, dma-buf OFF, host `libionic` injection mandatory) — using the
wrong block silently drops to TCP, which still *works* and so hides itself.

Not relevant to this bug, which is host-memory mapping and never leaves the node — but
recorded because the engine will not boot in PD mode without it.

## External dependencies (absolute paths, not in the repo)

| what | where |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB, `GlmMoeDsaForCausalLM`, 78 layers, 256 experts, ships `chat_template.jinja`) |
| scratch / logs / bench artifacts | `/shared_nfs/yihou_agentbench/` |
| kvd L3 long tier | `/tmp/kvd-long` inside the container (`--long-bytes 512G`) |

`/home` must not be used for large artifacts — it has filled up and destroyed a 28 GB
image tar on this cluster before.

## Model shape that determines the faulting buffer

The micro-repro hardcodes these to rebuild the exact allocation:

    page_size            64
    layers               78
    index_head_dim       128
    quant_block_size     128
    size_per_token       128 + 128/128*4 = 132
    indexer page stride  132 * 64 = 8448 bytes
    host indexer buffer  7.33 GB across 78 layers

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| docker registry login | team registry account; `export DOCKER_CONFIG=/tmp/dockercfg` **before every docker call** (docker 29 buildx plugin discovery fails on the default path) |
| cluster access | Spur job allocation via `sbatch`; no SSH keys involved — `spur exec` only |

No API keys, tokens, S3 or etcd credentials are needed. etcd runs unauthenticated on
the prefill node's private data-plane IP. No secret value appears in any file in this
kit.
