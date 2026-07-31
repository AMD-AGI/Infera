# Environment — Exp 3a / 3c

Captured from the running nodes on 2026-07-30 / 2026-07-31. Where a value is
inherited rather than directly measured, it says so.

## Nodes — three pairs, because two died

This arm outlived two node pairs. All three are listed because the failure
reproduced identically on each, which is part of the result.

| round | role | spur job | node | data-plane IP | fate |
|---|---|---|---|---|---|
| 1–2 | prefill | 14320 | `crsuse2-m2m-074` | 10.245.154.156 | `NODE_FAIL` 2026-07-30 12:49 |
| 1–2 | decode | 14321 | `crsuse2-m2m-072` | 10.245.144.119 | `NODE_FAIL` 2026-07-30 12:49 |
| 3 | prefill | 14915 | `crsuse2-m2m-214` | 10.245.158.72 | lost with the same event |
| 3 | decode | 14914 | `crsuse2-m2m-030` | 10.245.147.58 | lost with the same event |
| 4–7 | prefill | 17443 | `crsuse2-m2m-099` | 10.245.152.84 | alive at time of writing |
| 4–7 | decode | 17444 | `crsuse2-m2m-227` | 10.245.151.183 | alive at time of writing |

All six passed the GPU health gate (`torch.cuda.is_available()` → `True`,
`device_count()` → `8`) before use. Spur has nodes that enumerate 8 GPUs and
report `False`; gate every freshly held node.

Router ran on the prefill node. Ports were bumped on each restart
(8140 → 8141 → 8142 → 8143 → 8144, prometheus 29140 → 29144) so a stale open
circuit breaker from a previous round could never be misread as a new failure.
This mattered: one conc=32 run returned 503 in **0.42 s**, which is the
circuit-breaker signature, not a backend fault. Restarting the router on a
fresh port turned it back into the real 12–23 s failure.

**On the NODE_FAIL:** five jobs (14316, 14317, 14318, 14320, 14321) all
transitioned to `NODE_FAIL / NodeDown` at 12:49:01, and `spurctld` then refused
connections for roughly 15 minutes. A cluster-side event, not something this
workload caused. Everything on the nodes was lost, including the container
image — see below.

## Hardware

**Provenance:** inherited from `crsuse2-m2m-118` (Exp-1 prefill node, same day,
same spur partition, same `-G8` request, same SKU and health gate). Not
re-measured per node. Re-run `collect_env.sh` if an exact per-node record
matters.

| item | value |
|---|---|
| GPU | **AMD Instinct MI355X** × 8 (gfx950) |
| CPU | AMD EPYC 9575F 64-Core × 2 sockets (236 logical CPUs) |
| RAM | 2751 GiB |
| kernel | 6.8.0-107-generic |

### RDMA fabric

| item | value |
|---|---|
| KV-transfer NIC | **`mlx5_0`**, RoCEv2, GID index **3** |
| GPUDirect path | **dma-buf** (`MOONCAKE_DISABLE_HIP_DMABUF=0`) |
| present, deliberately unused | `ionic_0 … ionic_7` |

Spur has no peermem and the ionic NICs lack ODP, so dma-buf over the single
mlx5 rail is the only working GPUDirect path. Mooncake is pinned with
`MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0` and `MC_DISABLE_HIP_TRANSPORT=1`.
Never enable the HIP transport for cross-node PD here.

## Software — the image had to be rebuilt

The image named by earlier kits, `/home/yihou/infera.yihou.sglang.1.0.tar`, **no
longer exists**: `/home` filled to 100 % and the tar was removed. After the
NODE_FAIL no surviving node held the image either, and it was never pushed to a
registry (it is a local build). It was therefore **rebuilt from git** on
2026-07-30 for rounds 3–7.

| item | value |
|---|---|
| container image | `infera.yihou.sglang.1.0:latest` |
| image ID (rebuilt) | `4155ff02bbab`, 108 GB |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| sglang | 0.5.15.post1 |
| ROCm | 7.2.0 |
| mooncake | rebuilt in-image with `USE_HIP_DMABUF` |

**Rebuild inputs** (all in this repo, so the image is reproducible):

| component | location |
|---|---|
| Dockerfile | `glm52.mxfp4.spur.mooncake.packup_20260728/patches/Dockerfile.sglang.dmabuf.copy` |
| build script | `git show 9cbed1f:deploy/docker/scripts/build_mooncake_dmabuf.sh` |
| C++ patches | `deploy/docker/patches/mooncake_cpp/` |

The build self-checks and refuses to produce a silently-degraded image. Its
verification line on this rebuild:

```
DMABUF_COMPILED_IN=yes (hsa_amd_portable_export_dmabuf ibv_reg_dmabuf_mr)
```

Without that, device-memory registration falls back to bare `ibv_reg_mr`, which
EFAULTs on this cluster and would break cross-node PD entirely.

The rebuild took ~30 s (base layers cached); `docker save` + `docker load` to
the second node took ~6 min for a 28 GB tar. The tar is kept at
`/shared_nfs/yihou_imgbuild/infera.yihou.sglang.1.0.tar` — on **shared** storage
this time, since a node-local copy does not survive a NODE_FAIL.

## Model

`/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`

## Repo state

Branch `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e374`.

## JIT caches

Moved off `/home` (which was full and would fail writes silently, showing up
only as a much longer boot):

```
TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_exp3way/inductor_cache
TRITON_CACHE_DIR=/shared_nfs/yihou_exp3way/triton_cache
```

Cold start with empty caches ~13 min; with warm caches ~7 min.

## Cluster access

Spur, not stock Slurm: `spur exec <job> <cmd>`; **ssh to compute nodes is
banned**. `export DOCKER_CONFIG=/tmp/dockercfg` before every docker call.
Never background a long docker client inside `spur exec` — the exec namespace
teardown kills it.
