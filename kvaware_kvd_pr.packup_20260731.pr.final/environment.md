# Environment

Captured on both nodes at run time with `scripts/kvaware_env.sh` (2026-07-31 10:59 UTC).

## Git

| | |
|---|---|
| Repo | `AMD-AGI/infera` |
| Branch | `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` |
| Base | `8692fb4` (`origin/main`, "Merge pull request #42 from AMD-AGI/dev/limou/pd_ut") |
| HEAD | `da65cc7` — 5 commits, all in `patches/` |
| Worktree | `/home/yihou/dev/git.16-19/infera.glm5.2.mxfp4.offical` |

The branch originally sat on `2df2fed`, **35 commits behind `origin/main`** and with
zero commits of its own. It was reset to `origin/main` before any work: the winning
image needs the unified-Mooncake rebuild (`a546137`), which exists on `main` but not
on `2df2fed`. Building on the stale base would have produced an image without the
HIP-transport gate, breaking cross-node PD.

## Images

| | |
|---|---|
| Final image | `infera/engine-sglang:kvaware-kvd` |
| Digest | `sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80` |
| Built | 2026-07-31T09:57:03Z, on chi2879 |
| Size | 78.6 GB |
| Intermediate base | `infera/engine-sglang:kvaware-kvd-base`, `sha256:362a0d331ad9c70111f4f3e0b63b751cb60f8fddcd9ac25e6b9bcc4586822fd8` (from `deploy/docker/Dockerfile.sglang`) |
| Vendor base | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| SGLang | 0.5.15.post1 (verified inside the image) |

The same digest was verified present on **both** nodes before the run.

## Hardware — identical on both nodes

| | |
|---|---|
| CPU | AMD EPYC 9575F, 64 cores / 128 threads |
| RAM | 3023 GB |
| GPU | AMD MI355X (gfx950) × 8 |
| ROCm | 7.2.0 |
| amdgpu driver | 6.16.13 |

## Nodes

| Role | Host | Data-plane IP | GPUs |
|---|---|---|---|
| Prefill + etcd + router | `chi2879` | 10.2.122.10 | 0–7 (TP8) |
| Decode | `chi2867` | 10.2.122.44 | 0–7 (TP8) |

Kernel differs slightly and did not matter: chi2879 `6.8.0-124-generic`,
chi2867 `6.8.0-107-generic`. Docker 28.5.1 / 28.4.0.

Access is via a jump host (`root@149.28.124.225`); the nodes are not directly
reachable.

## RDMA fabric

| | |
|---|---|
| NICs | Pensando **ionic**, RoCE v2 |
| Active rails | `ionic_0 … ionic_7` — all 8 ACTIVE on both nodes |
| ionic_rdma driver | 26.03.3.001 |
| Host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187` |
| GID index | `MC_GID_INDEX=1` |
| dma-buf | **disabled** (`MOONCAKE_DISABLE_HIP_DMABUF=1`) — ionic/no-ODP hosts duplicate the KV pool at pin time |

The container's own libionic is from a different release train than the host's
kernel module. It is replaced at container start by the image entrypoint
(`infera-inject-host-ionic`) via the `-v <host libionic>:/host-libionic/libionic.so:ro`
bind mount. **Verified: 8 `PORT_ACTIVE` inside the container on both nodes.** Without
it `ibv_get_device_list()` returns zero devices and Mooncake silently drops to TCP.

`ip_local_port_range` = `32768 60999` on both nodes — relevant to the
`free_tcp_port_block` fix in `patches/0001`, which scans below this range.

## External dependencies (not in the repo)

| Path | What | Notes |
|---|---|---|
| `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | model weights, 816 GB | NFS mount `10.2.123.177:/aac-8634674/aac/shared/data`, 501 TB. `GlmMoeDsaForCausalLM`, MLA + DSA, 78 layers, 256 experts |
| `/mnt/vast/c_huggingface/kvaware_kvd_final/` | run workspace | scripts, logs, the 79 GB image tar used to move the image between nodes |
| `quay.io/coreos/etcd:v3.5.14` | discovery | pulled from the public registry |

## Secrets required

No credential values are recorded here. To reproduce you need:

- **SSH to the jump host** and onward to `chi2879` / `chi2867` — key-based, from
  the team's cluster access.
- **No registry credentials** — the images are built locally from this repo and
  the public `lmsysorg` base.
- **etcd** runs unauthenticated on the data-plane IP, as in the lab. Do not
  copy that choice to production.
