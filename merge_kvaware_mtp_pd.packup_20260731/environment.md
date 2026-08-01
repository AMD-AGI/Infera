# Environment

Captured on both nodes at 2026-07-31 14:00 UTC with the packup skill's
`collect_env.sh`, ~15 min after the last stress run. Full raw output:
`results/env_chi2879.txt`, `results/env_chi2867.txt`.

## When

2026-07-31, 12:22–13:40 UTC.

| step | time (UTC) |
|---|---|
| patch set applied, both nodes | 12:28–12:33 |
| G0 legs launched | 12:33 |
| G0 prefill ready / decode ready | 12:38 / 12:42 |
| G0 probe + prefix reuse | 12:44–12:47 |
| G0 prefill restart (kvd attribution) | 12:48–12:51 |
| patches 6 + 7 written and applied | 12:58–13:12 |
| G1 legs launched, both ready | 13:13 / 13:22 |
| G1 probe + cache-view + reuse | 13:24–13:29 |
| G2 needle (2 runs) | 13:32–13:33 |
| stress conc=16 / conc=128 / conc=128 re-run | 13:34 / 13:35 / 13:38 |

## Nodes

Vultr cluster, reached through a jump host — the nodes are not directly
routable. Both were held for this run; no slurm involved (these are the
long-lived `chi28xx` boxes, claimed by convention).

| role | host | data-plane IP | GPUs | port |
|---|---|---|---|---|
| prefill + etcd + router + kvd | `chi2879` | 10.2.122.10 | 0–7 (TP8) | 30000, router 8100 |
| decode + kvd | `chi2867` | 10.2.122.44 | 0–7 (TP8) | 30000 |

## Hardware

Identical SKU on both nodes; the figures below are from `chi2879` and matched on
`chi2867` except where noted.

| item | value |
|---|---|
| GPU | AMD Instinct **MI355X** × 8 (**gfx950**), model `0x75a3` |
| amdgpu driver | **6.16.13** |
| ROCm | 7.2.0 |
| CPU | AMD EPYC 9575F 64-Core × 2 sockets (**256** logical CPUs) |
| RAM | 3.0 TiB |
| OS | Ubuntu 24.04.3 LTS |
| kernel | `6.8.0-124-generic` (chi2879), `6.8.0-107-generic` (chi2867) |
| docker | 28.5.1 (chi2879), 28.4.0 (chi2867) |

The kernel and docker minor-version skew between the nodes did not matter here
and is recorded only so a reproducer is not surprised by it.

## RDMA fabric

| item | value |
|---|---|
| NICs | AMD Pensando **ionic**, RoCE v2 |
| rails | `ionic_0 … ionic_7` — **all 8 Active / LinkUp, 400 Gb/s**, both nodes |
| ionic_rdma driver | **26.03.3.001** |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187` |
| GID index | `MC_GID_INDEX=1` |
| HIP transport | **disabled** (`MC_DISABLE_HIP_TRANSPORT=1`) |
| dma-buf | **disabled** (`MOONCAKE_DISABLE_HIP_DMABUF=1`) |

**Two settings that are not tuning knobs on this fabric.** HIP intra-node P2P
transport cannot open a peer node's handle, so leaving it on breaks cross-node PD
outright. dma-buf registration on ionic (no ODP) duplicates the KV pool at pin
time — this is the *opposite* of the spur/mlx5 cluster, where dma-buf is the only
GPUDirect path because there is no peermem. Carrying a recipe across the two
clusters without flipping both will fail.

The container ships a libionic from a different release train than the host
kernel module. The image entrypoint (`infera-inject-host-ionic`) replaces it from
the `-v <host libionic>:/host-libionic/libionic.so:ro` bind mount. **Verified: 8
`PORT_ACTIVE` inside each container before anything else ran.** Without it
`ibv_get_device_list()` returns zero devices and mooncake silently drops to TCP —
the run would "work" and be meaninglessly slow.

`ip_local_port_range = 32768 60999` on both nodes. Relevant to
`free_tcp_port_block`, which scans *below* this range.

## Software

| item | value |
|---|---|
| image | `infera/engine-sglang:kvaware-kvd` |
| image digest | `sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80` |
| image built | 2026-07-31T09:57:03Z (on chi2879), 78.6 GB |
| **same digest verified on both nodes** | yes |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| sglang | **0.5.15.post1**, commit `0b3bb0cbe318` (release/v0.5.15) |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |
| built from | `deploy/docker/Dockerfile.sglang.kvaware-kvd` on branch `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` @ `da65cc7` |

### The image was patched in-container, not rebuilt

**This is the single most important caveat in this kit.** The base image above
carries kvaware + kvd only. The other two workstreams were applied by
`scripts/apply_all_in_container.sh` into the *running* containers, with every
patch verified present in the **bytecode** (not just the source) on both nodes.

`deliverable/deploy/docker/Dockerfile.sglang` is the merged image definition and
runs the same scripts in the same order — but **it has not been built**. A
reproducer has two paths, and they are not equivalent:

- **Reproduce what was measured**: follow `REPRODUCE.md`, which patches in-container.
- **Validate the deliverable**: build the Dockerfile, then re-run G0–G2 against it.
  This has not been done and is the outstanding work item.

### Source-side changes

Four infera files were edited (three fixes) plus three new test modules; they are
in `deliverable/infera_source_changes.diff` against branch
`yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` @ `da65cc7`. In the experiment they were
applied to the containers as scripts (`patches/patch_infera_*.py`); as a
deliverable they belong as source commits, picked up by the Dockerfile's existing
`COPY infera ./infera`.

Repos involved:

| repo | branch | commit | role |
|---|---|---|---|
| `AMD-AGI/Infera` | `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` | `da65cc7` | kvaware+kvd, the base image |
| `AMD-AGI/Infera` | `worktree-dsa-hip-dp-rows-fix.rebase` | `78cf750` | PR58 — the 3 DSA diffs |
| `AMD-AGI/Infera` | `llying/glm5p2_fp8_fixes` | PR56 | early-send + bigram (partially taken) |
| `AMD-AGI/Infera` | `main` | `8692fb4` | the base all three sit on |

## External dependencies (not in any repo)

| path | what | notes |
|---|---|---|
| `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | model weights | shared VAST NFS (`10.2.123.177:/aac-8634674/...`), mounted at `/mnt/vast` on both nodes; bind-mounted into the container |
| `/mnt/vast/c_huggingface/merge_20260731/` | this run's staging dir | scripts + logs; created by the run, kept |
| `/usr/lib/x86_64-linux-gnu/libionic.so.1` | host RDMA userspace lib | must match the host's `ionic_rdma` kmod; bind-mounted, see above |
| `quay.io/coreos/etcd:v3.5.14` | discovery | pulled on chi2879 |

## Secrets required (names and sources only — no values here or anywhere in this kit)

| what | where it comes from |
|---|---|
| Cluster SSH | jump host `root@149.28.124.225`, then `ssh chi2879` / `ssh chi2867`. Key-based; arrange your own access. |
| Docker registry | **not needed** — the image is already local on both nodes. Only needed if you rebuild and push. |
| etcd / router | no auth configured on this cluster. |
| Model weights | no credential; readable on the shared mount. |

No API keys, tokens, or S3 credentials are involved in this experiment.
