# Environment

Raw `collect_env.sh` output per node is in `env/env_<hostname>.txt`. This file
is the distilled version plus the things the script cannot know.

Captured 2026-09-18. Both nodes were identical in every field below.

## Hardware

| | crsuse2-m2m-135 | crsuse2-m2m-138 |
|---|---|---|
| role | prefill + control + builder | decode |
| GPU | AMD Instinct MI355X ×8 (`0x75a3`) | same |
| GPUs used | **2,3,4,5** | **2,3,4,5** |
| VRAM per GPU | 309,220,868,096 B (288 GiB) | same |
| CPU | AMD EPYC 9575F 64-Core ×2 sockets, 236 threads | same |
| RAM | 2.7 TiB | same |
| data-plane IP | `10.245.148.209` | `10.245.157.237` |

### Known hardware caveats on crsuse2-m2m-135

- **GPU[1] is occupied** by a root-owned Kubernetes pod (`pod5d84e491`,
  `vllm serve Qwen3-32B`, 22 days uptime) holding 96.94 GB. Not clearable
  without cluster-admin rights. This is why the run uses devices 2,3,4,5.
- **`ionic_7` is defective** — no netdev (`enP3p0s12` absent), GID index 1
  all-zero, so rail `0200` is missing on this node.

Neither applies to crsuse2-m2m-138, whose eight GPUs and eight NICs are healthy.

## Software

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| kernel | 6.8.0-107-generic |
| ROCm driver | **6.14.14** |
| container runtime | Docker (host network, `--device /dev/kfd /dev/dri /dev/infiniband`) |

### Images — pinned by digest, not tag

| | |
|---|---|
| built image | `infera-sglang:v0519-yihou-0917` |
| built image id | `sha256:4190c3a99d0ea8b195580008e7f37fb2f1dfdbe6254b019884d4bef5721041cd` |
| base image | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917` |
| base digest | `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32` |
| Dockerfile | `deploy/docker/Dockerfile.sglang` at the repo SHA below |

`issue.md` §3.2 documents a same-named tag silently drifting to different source
commits, so record the **in-image** identifiers, not the tag:

| | |
|---|---|
| sglang | `0.5.19.dev20260917+ga9fb1c3238` (the `+g…` suffix is the commit) |
| sglang base checkout HEAD | `7ccbf5fd04f7ee23095fc38e49e749d58dc18282` (tree dirtied by the patch set) |
| aiter | source install, git `4ad998328`, `v0.1.21.dev0-48-g4ad998328-dirty` |

The aiter `-dirty` is the base image's own state, not introduced by this build.
`aiter.__version__` is unreadable in a GPU-less container (importing aiter runs
`rocminfo`), so the git SHA is the GPU-free equivalent.

### Repo

| | |
|---|---|
| repo | `AMD-AGI/Infera` |
| branch | `dev/pd_opt/glm_5.2_agentx` |
| HEAD | `5a342acffe10e09729662ff40e81b22a4367fd75` |
| plus | the two patches in `patches/` (uncommitted at pack-up time) |

## RDMA fabric

Eight `ionic` RoCE NICs per node = 8 rails, plus `mlx5_0` on the management
network (not a rail). All ports `ACTIVE / LinkUp / 400 Gb/s (4X NDR)`, MTU 9000,
RoCE `active_mtu` 4096 — except 135's `ionic_7`, above.

**The rails are IPv6-only** (`fc01::/…` ULA). No ionic NIC has an IPv4 address;
only the `ens3` management NIC does. Consequences that silently break a run:

- **`MC_GID_INDEX=1` is mandatory**, not tuning. Index 0 is link-local `fe80::`
  and cannot route; preflight's `rdma-default` variant reproduces exactly that
  failure.
- Any code path that resolves NICs via IPv4-mapped GIDs finds nothing and
  no-ops — which is why `apply_mooncake_topology_default()` cannot help here.

**Rails are physically isolated**: `ionic_i` reaches only `ionic_i`. Every
off-diagonal pairing fails with `IBV_WC_RETRY_EXC_ERR`. There is no cross-rail
fallback, and equally no cross-rail contention.

### GPU ↔ NIC ↔ rail map

Rail id is the **2nd hextet of the GID** and is constant for a given `ionic_N`
across every node in the fabric. It is **not** in `ionic_N` order — never infer a
rail from the device index.

| GPU | NUMA / PCI domain | NIC | rail id | used here |
|---|---|---|---|---|
| 0 | 0 / `0002` | ionic_0 | `0800` | |
| 1 | 0 / `0002` | ionic_1 | `0700` | occupied by k8s pod on 135 |
| 2 | 0 / `0002` | ionic_2 | `0500` | **yes** |
| 3 | 0 / `0002` | ionic_3 | `0600` | **yes** |
| 4 | 1 / `0003` | ionic_4 | `0400` | **yes** |
| 5 | 1 / `0003` | ionic_5 | `0300` | **yes** |
| 6 | 1 / `0003` | ionic_6 | `0100` | |
| 7 | 1 / `0003` | ionic_7 | `0200` | dead on 135 |

Measured rail bandwidth in this run: 41.1 GB/s NUMA-local, ~31 GB/s cross-NUMA,
byte-verified. The 2026-09-17 fleet survey measured a 48.7 GB/s ceiling (97% of
line rate) with 2 QPs and 1 MiB messages.

## External dependencies (absolute paths)

| what | path | notes |
|---|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | shared NFS, read-only mount |
| host RDMA lib | `/lib/x86_64-linux-gnu/libionic.so` | injected into containers at `/host-libionic/libionic.so` |
| bulk scratch | `/mnt/m2m_nobackup` | local NVMe per node, ~7.5 TB free |
| AITER JIT cache | `/tmp/aiter-jit-<uid>/<image-id>` | per image id; first build is slow |
| InferenceX | fetched at commit `918524ff94045b3f091115f1051c22a8588edf2b` | pinned by the bench |

## Gaps in this capture

Recorded rather than guessed:

- The `ionic` **NIC firmware/driver version** was not captured; `collect_env.sh`
  reports the RDMA link state and GUIDs but not the vendor driver revision.
- Per-GPU HBM temperature / clock state during the run was not sampled; only
  VRAM occupancy was.
