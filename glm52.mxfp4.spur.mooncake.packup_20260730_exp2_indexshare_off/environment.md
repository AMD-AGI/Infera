# Environment — Exp 2

Everything below was captured **from the running nodes on 2026-07-30**, inside the
container that served the traffic, not reconstructed afterwards. Where a value could not
be captured it says so explicitly rather than guessing.

## Nodes

Held with `sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00` (Spur, not stock
Slurm — see "Cluster access" below).

| role | spur job | node | data-plane IP (`ens3`) | server port |
|---|---|---|---|---|
| prefill | 14317 | `crsuse2-m2m-234` | 10.245.152.243 | 30000 |
| decode | 14318 | `crsuse2-m2m-259` | 10.245.155.111 | 30001 |

Router ran on the **prefill** node, port **8120**, prometheus **29120**.
(Each arm of the three-way run got distinct ports so one arm's open circuit breaker could
never be mistaken for another arm's failure.)

Both nodes passed the GPU health gate before use:
`python3 -c "import torch; torch.cuda.is_available()"` → `True`, `device_count()` → `8`.
Spur has nodes that enumerate 8 GPUs but report `False`; gate every freshly held node.

## Hardware

**Provenance note:** the hardware table below was captured on `crsuse2-m2m-118` (the
prefill node of the sibling Exp-1 run on the same day), **not** on this arm's own nodes.
All six nodes of the 2026-07-30 three-way run were allocated from the same spur partition
with the same `-G8` request and are the same SKU, and all six passed the same GPU health
gate — but the CPU/RAM/firmware figures here are inherited, not directly measured on
`crsuse2-m2m-234` / `crsuse2-m2m-259`. Re-run `collect_env.sh` on those nodes if an exact
per-node record matters.

| item | value |
|---|---|
| GPU | **AMD Instinct MI355X** × 8 (gfx950) |
| CPU | AMD EPYC 9575F 64-Core × 2 sockets (236 logical CPUs) |
| RAM | 2751 GiB total |
| kernel | 6.8.0-107-generic |
| OS (host) | Ubuntu (spur image) |

### RDMA fabric — the part that matters most on this cluster

| item | value |
|---|---|
| KV-transfer NIC | **`mlx5_0`** — firmware `28.43.3608`, `PORT_ACTIVE`, `link_layer: Ethernet` (RoCEv2) |
| GID index | **3** |
| GPUDirect path | **dma-buf** (`MOONCAKE_DISABLE_HIP_DMABUF=0`) |
| also present, deliberately unused | `ionic_0 … ionic_7` (8 rails) |

Spur has **no peermem**, and the 8 ionic NICs lack ODP — so dma-buf over the single mlx5
rail is the only working GPUDirect path here. This is the **opposite** of the vultr
cluster (ionic + peermem). Mooncake is forced onto mlx5 with
`MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0` and `MC_DISABLE_HIP_TRANSPORT=1`; never enable
the HIP transport for cross-node PD here.

## Software

| item | value |
|---|---|
| container image | `infera.yihou.sglang.1.0:latest` |
| image ID | `sha256:347bcd45da0dee1bc87f10c348e41f20ed56e11d23f9fead164cdef4e51dc970` (108 GB) |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| image tar (for `docker load`) | `/home/yihou/infera.yihou.sglang.1.0.tar` |
| OS in container | Ubuntu 22.04.5 LTS |
| ROCm | **7.2.0** |
| torch | **2.9.1+rocm7.2.0.lw.git7e1940d4** (HIP `7.2.26015-fc0010cf6a`) |
| sglang | **0.5.15.post1** |
| sglang source commit (in image) | **`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`** |
| triton | 3.6.0+git42270451 |
| tilelang | 0.1.7.post3+cuda.gita55a8230 |
| mooncake | rebuilt in-image with `USE_HIP_DMABUF`; the wheel exposes no `__version__`, so no version string could be captured — it is pinned only by the image digest above |

`sgl-kernel` and `aiter` report no version via `importlib.metadata` in this image (they
are built in, not pip-installed). They are likewise pinned only by the image digest.

**This repo:** branch `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e37437c392b1b27ef61540038593df68e3951`.
The repo supplies the packup kits and patches; the server code comes from the image.

## External dependencies (absolute paths, not in this repo)

| what | path |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| scratch workspace this run used | `/shared_nfs/yihou_exp3way` |
| torch-inductor cache | `/shared_nfs/yihou_exp3way/inductor_cache` |
| triton cache | `/shared_nfs/yihou_exp3way/triton_cache` |

The JIT caches were deliberately moved onto `/shared_nfs`: `/home` (10 TB NFS) was **100 %
full** during this run, and a failed cache write there is silent — it shows up only as a
much slower boot. See `notes.md`.

## Required secrets (names and sources only — no values here)

| secret | needed for | where it comes from |
|---|---|---|
| docker registry login | pulling the base image, if you rebuild rather than `docker load` the tar | team registry credentials; **must** `export DOCKER_CONFIG=/tmp/dockercfg` first (docker 29 buildx plugin discovery fails on the default path) |
| spur cluster access | holding nodes, `spur exec` | existing cluster account; **ssh to compute nodes is banned** — dispatch only through `spur exec` |

No API keys, tokens, or object-store credentials are involved in this experiment.

## Cluster access notes

- Scheduler is **Spur**, not stock Slurm: dispatch with `spur exec <job> <cmd>`.
- Expect `JobHoldMaxRequeue` bounces when holding nodes; retry or `--exclude` bad nodes.
- Never background a long docker client inside `spur exec` — the exec namespace teardown
  kills it. Run work detached (`docker run -d` / `docker exec -d`) and poll the log file.
- Node→node image move: `docker save` → NFS tar → `docker load`.
