# Environment — GLM-5.2-MXFP4 SGLang PD, same-node P4D4

The benchmark result in this kit was produced on **`crsuse2-m2m-137`**, running
both the prefill and the decode leg on that single node (P4 + D4, 8 GPUs) under
docker image **`infera-sglang:v0519-yihou-0917-nextnfix-hicache`**
(in-image `sglang 0.5.19.dev20260917+ga9fb1c3238`).

A second node, **`crsuse2-m2m-276`**, was the originally intended target. It
**segfaults this exact workload** while 137 runs it clean, so the experiment was
moved to 137. Its environment is captured here for the comparison record. 276 is
idle; nothing was run on it beyond the read-only probes below.

Snapshot taken 2026-09-22. All probes were read-only. The live deployment on 137
(`glm52-pd-yihou-sn-p4d4-*` containers) was not touched.

## The load-bearing finding: 137 and 276 are software-identical

Every captured software version is **identical** across the two nodes, yet 276
segfaults the workload and 137 does not. The difference is not in any field
below.

| Field | crsuse2-m2m-137 (works) | crsuse2-m2m-276 (segfaults) |
|---|---|---|
| Kernel (`uname -r`) | 6.8.0-107-generic | 6.8.0-107-generic |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS |
| amdgpu (`/sys/module/amdgpu/version`) | 6.14.14 | 6.14.14 |
| rocm-smi driver version | 6.14.14 | 6.14.14 |
| `rocm-core` | 7.0.1.70001-42~24.04 | 7.0.1.70001-42~24.04 |
| `hsa-rocr` | 1.18.0.70001-42~24.04 | 1.18.0.70001-42~24.04 |
| `amdgpu-dkms` | 1:6.14.14.30100100-2212064.24.04 | 1:6.14.14.30100100-2212064.24.04 |
| `libionic1` | 54.0-149.g3304be71 | 54.0-149.g3304be71 |
| `rdma-core` | 2410mlnx54-1.2410068 | 2410mlnx54-1.2410068 |
| `ibverbs-providers` | 2410mlnx54-1.2410068 | 2410mlnx54-1.2410068 |
| `perftest` | 4.5-327.gd9e28e4 | 4.5-327.gd9e28e4 |
| GPU | 8× AMD Instinct MI355X (gfx950) | 8× AMD Instinct MI355X (gfx950) |
| CPU | 2× AMD EPYC 9575F 64-Core (236 threads) | 2× AMD EPYC 9575F 64-Core (236 threads) |
| RAM | 2.7 TiB | 2.7 TiB |
| RDMA data rails | 8× ionic @ 400 Gb/s, all Active | 8× ionic @ 400 Gb/s, all Active |
| In-image sglang | 0.5.19.dev20260917+ga9fb1c3238 | 0.5.19.dev20260917+ga9fb1c3238 |
| Image RootFS DiffIDs | 72 layers, list-sha `82048ee5…870887cd` | 72 layers, list-sha `82048ee5…870887cd` (**identical**) |

### On the docker image "difference" that isn't one

The two nodes use different docker storage backends, so the **top-level image ID
legitimately differs** for byte-identical content — this is expected, not a
discrepancy:

| | 137 | 276 |
|---|---|---|
| storage driver | `overlay2` | `overlayfs` (containerd image store) |
| top-level image ID | `sha256:fd7220a5…51e91d35` | `sha256:291f148a…bff0fead` |
| **RootFS layer DiffIDs** | **identical (72 layers)** | **identical (72 layers)** |

The comparable field is the RootFS DiffID list, and it is **byte-identical**
between the two nodes (verified by diffing the full 72-line lists — same
sha256sum `82048ee56ced25f577dcd1dcd78400d53a0507b2fd5c748d229a5317870887cd`).
The image content is the same on both machines.

## Per-node detail

### crsuse2-m2m-137 (the result node)

**Host.** Ubuntu 24.04.4 LTS, kernel `6.8.0-107-generic`. Uptime ~26 days at
snapshot.

**CPU / RAM.** 2× AMD EPYC 9575F 64-Core (2 sockets, 59 cores/socket, 2
threads/core → 236 logical CPUs). 2.7 TiB RAM, 0 swap. 2 NUMA nodes: node0 =
CPUs 0-117, node1 = CPUs 118-235.

**GPU.** 8× AMD Instinct MI355X, GFX version gfx950, card model 0x75a3. Per-GPU
VRAM 309,220,868,096 B (≈288 GiB). rocm-smi driver 6.14.14. rocm-smi node IDs
2–9 map to GPU[0]–GPU[7]. (VRAM was ~85–92% used at snapshot because the live
benchmark deployment is running.)

**GPU ↔ NUMA.** 4 GPUs on NUMA node0 (PCI domain `0002`), 4 on node1 (PCI domain
`0003`):

| PCI BDF | NUMA | drm card |
|---|---|---|
| 0002:00:01.0 | 0 | card0 |
| 0002:00:02.0 | 0 | card8 |
| 0002:00:03.0 | 0 | card16 |
| 0002:00:04.0 | 0 | card24 |
| 0003:00:01.0 | 1 | card32 |
| 0003:00:02.0 | 1 | card40 |
| 0003:00:03.0 | 1 | card48 |
| 0003:00:04.0 | 1 | card56 |

**RDMA fabric.** 8 ionic data-plane rails @ 400 Gb/s (4X NDR), all `State:
Active / LinkUp`, plus one Mellanox `mlx5_0` @ 200 Gb/s (4X HDR) as the
frontend/management NIC. The ionic rails split cleanly across NUMA, matching the
GPU split (rails on domain `0002`/numa0 pair with the numa0 GPUs, `0003`/numa1
with the numa1 GPUs):

| device | PCI BDF | NUMA | netdev | state | rate | gid[1] |
|---|---|---|---|---|---|---|
| ionic_0 | 0002:00:09.0 | 0 | enP2p0s9 | Active | 400 Gb/s | fc01:0800:880d:2d5f:0690:81ff:fe44:4d69 |
| ionic_1 | 0002:00:0a.0 | 0 | enP2p0s10 | Active | 400 Gb/s | fc01:0700:870d:2d5f:0690:81ff:fe44:9f41 |
| ionic_2 | 0002:00:0b.0 | 0 | enP2p0s11 | Active | 400 Gb/s | fc01:0500:850d:2d5f:0690:81ff:fe43:c489 |
| ionic_3 | 0002:00:0c.0 | 0 | enP2p0s12 | Active | 400 Gb/s | fc01:0600:860d:2d5f:0690:81ff:fe44:d781 |
| ionic_4 | 0003:00:09.0 | 1 | enP3p0s9 | Active | 400 Gb/s | fc01:0400:840d:2d5f:0690:81ff:fe42:a771 |
| ionic_5 | 0003:00:0a.0 | 1 | enP3p0s10 | Active | 400 Gb/s | fc01:0300:830d:2d5f:0690:81ff:fe44:3719 |
| ionic_6 | 0003:00:0b.0 | 1 | enP3p0s11 | Active | 400 Gb/s | fc01:0100:810d:2d5f:0690:81ff:fe42:8731 |
| ionic_7 | 0003:00:0c.0 | 1 | enP3p0s12 | Active | 400 Gb/s | fc01:0200:820d:2d5f:0690:81ff:fe44:37c1 |
| mlx5_0 | 0000:00:03.0 | 0 | ens3 | Active | 200 Gb/s | fe80::8c9e:e2ff:feee:2387 |

`mlx5_0` gid[3] = `::ffff:0af5:99f7` (RoCEv2 IPv4-mapped, = 10.245.153.247).
No inactive rails and no all-zero GIDs on this node.

**Storage.**
- `/mnt/m2m_nobackup` — 28 TB LVM, 20 TB used, **8.6 TB free** (70%).
- `/shared_nfs` — 410 TB NFS (`172.27.255.2`), **2.5 TB free** (100% used).
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` — **exists, 408 GB**.

**Docker.** Server 29.6.1, storage driver `overlay2`. Running containers at
snapshot: `glm52-pd-yihou-sn-p4d4-router / -decode-0 / -prefill-0 / -etcd`
(the live deployment) plus an unrelated `crusoe-vector` sidecar.

**Image.** `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, top-level ID
`sha256:fd7220a5…51e91d35`, 72 RootFS layers (list-sha `82048ee5…870887cd`).
In-image `sglang 0.5.19.dev20260917+ga9fb1c3238`. Base image: **ubuntu 22.04**
(image carries `org.opencontainers.image.ref.name=ubuntu`,
`org.opencontainers.image.version=22.04` at the bottom of its history; no
explicit `base_image` label is set).

### crsuse2-m2m-276 (comparison node, segfaults this workload)

Reached via `spur exec 165913` (runs as `yihou`). Hardware and every software
version match 137 exactly — see the comparison table above. Differences from 137,
all of which are benign / expected:

- **Docker storage backend** is the containerd image store (`overlayfs`), so the
  top-level image ID is `sha256:291f148a…bff0fead` — different bytes at the
  top level, **but the 72 RootFS DiffIDs are identical to 137's**.
- **Node-local NIC identities differ** (as they must — different hardware
  instances): ionic gid[1] values are on GID prefix `…2d70…` vs 137's `…2d5f…`,
  and `mlx5_0` gid[1] = `fe80::fc6a:9aff:fe04:d5c2`, gid[3] = `::ffff:0af5:98f9`
  (= 10.245.152.249). All 8 ionic rails are Active @ 400 Gb/s; no inactive rails,
  no all-zero GIDs.
- **`/mnt/m2m_nobackup`** — 28 TB, only 2.5 TB used, **26 TB free** (9%). (137's
  is fuller at 8.6 TB free; not workload-relevant.)
- `/shared_nfs` and the model path are the same shared mount — model present,
  408 GB.

RDMA rail detail on 276 (same PCI/NUMA layout as 137):

| device | PCI BDF | NUMA | netdev | state | rate |
|---|---|---|---|---|---|
| ionic_0 | 0002:00:09.0 | 0 | enP2p0s9 | Active | 400 Gb/s |
| ionic_1 | 0002:00:0a.0 | 0 | enP2p0s10 | Active | 400 Gb/s |
| ionic_2 | 0002:00:0b.0 | 0 | enP2p0s11 | Active | 400 Gb/s |
| ionic_3 | 0002:00:0c.0 | 0 | enP2p0s12 | Active | 400 Gb/s |
| ionic_4 | 0003:00:09.0 | 1 | enP3p0s9 | Active | 400 Gb/s |
| ionic_5 | 0003:00:0a.0 | 1 | enP3p0s10 | Active | 400 Gb/s |
| ionic_6 | 0003:00:0b.0 | 1 | enP3p0s11 | Active | 400 Gb/s |
| ionic_7 | 0003:00:0c.0 | 1 | enP3p0s12 | Active | 400 Gb/s |
| mlx5_0 | 0000:00:03.0 | 0 | ens3 | Active | 200 Gb/s |

## Repo state (login node)

Repository `/home/yihou/dev/git/infera.glm52.pd`:
- branch: `dev/pd_opt/glm_5.2_agentx`
- HEAD: `83e0f6c86cce34718f369d8669b750fa812c62d0`
- working tree: **dirty** — modified `bench/glm5p2_pd/{config.full.sh,config.sh,engine.sh,launch.sh}`, several `rust/router/src/*.rs`, `.claude/CLAUDE.md`, and untracked `.serena/`. The bench-side changes (`engine.sh` `DECODE_EXTRA_ARGS` escape hatch, config edits) are uncommitted and required to reproduce.

## Not captured / caveats

- **kfd sysfs GPU→NUMA name path** was not readable via
  `/sys/class/kfd/.../topology/nodes/*/name`; the GPU→NUMA mapping above was
  instead derived from the DRM cards' PCI `numa_node`, and GPU model/gfx from
  `rocm-smi`. This is a complete substitute, not a gap.
- **Base image digest** is not pinned by a label; only the human-readable base
  (`ubuntu:22.04`) is recorded, read from the image history. The exact base
  digest is **not captured**.
- 276's GPU/CPU/RAM were read via the same probes as 137 and match; a couple of
  276's live-utilisation counters were not separately recorded (the node is idle,
  so they are not meaningful for reproduction).
