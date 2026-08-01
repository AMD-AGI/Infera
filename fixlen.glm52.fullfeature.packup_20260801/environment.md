# Environment

Full per-node snapshots (read-only, no secret values) are in `env/env_chi2879.txt` and
`env/env_chi2867.txt`, captured 2026-08-01 11:21 UTC with the packup skill's
`collect_env.sh`. This file is the digest of what matters.

## Cluster

| | |
|---|---|
| cluster | **vultr** (not spur — the two are configured oppositely, see Transport) |
| access | jump host `root@149.28.124.225` (= slurm login node chi2866), then `ssh chi2879` |
| partition | `k8s` |
| prefill node | **chi2879**, data-plane `10.2.122.10` |
| decode node | **chi2867**, data-plane `10.2.122.44` |
| slurm holder | job name `yeandy-debug` on **both** nodes — **not ours** |

> **The slurm hold belongs to another user.** We killed only our own sglang processes
> and containers and never ran `scancel`. Anyone reproducing this must do the same: on
> these nodes, `docker rm -f` your own container, never drop the node's slurm hold.

## Hardware (identical on both nodes)

| | |
|---|---|
| GPUs | 8 × **AMD Instinct MI355X**, `gfx950` |
| VRAM | 288 GB/card (309,220,868,096 bytes reported) |
| CPU | **AMD EPYC 9575F** 64-Core, 256 threads (1 thread/core, 2 sockets) |
| RAM | ~3.0 TB |
| GPU driver | **6.16.13** |
| kernel | `6.8.0-124-generic` (Ubuntu SMP PREEMPT_DYNAMIC, 2026-05-26) |
| root disk | `/dev/md0`, 838 GB — **see the kvd disk trap in `patches/README.md` §0003** |

### RDMA fabric — the part this experiment turns on

8 × **ionic** RoCE v2 NICs per node (`ionic_0` … `ionic_7`), all `PORT_ACTIVE`.

**`MC_GID_INDEX` differs per node** and must be discovered, not assumed:

| node | routable GID index | link-local (do not use) |
|---|---|---|
| chi2879 | **1** | 0 |
| chi2867 | **2** | 0 |

Both expose exactly two RoCE v2 GIDs per port — `fe80::` (link-local) and `fd93::`
(routable) — but at different indices. This is `patches/0001`, and getting it wrong
kills the decode leg on all 8 DP ranks at init.

Host `libionic` is bind-mounted into the container and must match the host's
`ionic_rdma` kmod: `/usr/lib/x86_64-linux-gnu/libionic.so.1` → `/host-libionic/libionic.so`.
A failed injection silently drops mooncake to TCP and the run "works" while measuring
nothing — hence the `PORT_ACTIVE: 8` gate in `scripts/reset_node.sh`.

## Software

| | |
|---|---|
| image | **`infera/engine-sglang:merged-e`** |
| digest, chi2879 | `sha256:bfcb6462fa306743e0bf43b32ac0263ce9094e13591f6f748263e5348bf97e41` |
| digest, chi2867 | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`<br>`sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| sglang | **0.5.15.post1** |
| ROCm | 7.2.0 |
| etcd | `quay.io/coreos/etcd:v3.5.14` (on chi2879) |
| engine entry | `python3 -m infera.engine.sglang` — the **infera wrapper**, not `sglang.launch_server` |
| router | `python3 -m infera.server --router-backend rust` → execs `/usr/local/bin/infera-router` |

> **The two node image ids differ and that is expected** — each node built the image
> independently from the same branch, so Rust objects and layer timestamps differ. Do
> not check for equal digests; check content equivalence
> (`glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/scripts/verify_built_image.sh`).

### Repo state

| | |
|---|---|
| repo | `AMD-AGI/Infera`, worktree `infera.merge.liying.kv.mtp` |
| branch | **`yihou.dev.glm52.merged.experiment`** |
| commit | **`b92a1e81380f7583ef030b2f4a56426149f9a412`** (`b92a1e8`) |
| tree | clean except untracked `mission.kv.liying.mtp.bench.md`, `work.bench_20260801/`, `CLAUDE.md`, and this packup |

The image predates the checkout by hours but is built from the same branch head — it is
the 31-commit `merged-e` build validated in `liying_rest_pr56.packup_20260801`.

## External dependencies (absolute paths, not in any repo)

| what | where | notes |
|---|---|---|
| model weights | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST NFS, mounted `/mnt/vast` on both nodes. `generation_config.json` (temp 1.0 / top_p 0.95) is **load-bearing** — see `notes.md` |
| tokenizer | same path (`tokenizer.json` + `chat_template.jinja`) | the Rust router loads both directly from disk |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1` | bind-mounted; must match host kmod |
| scratch / logs | `/mnt/vast/c_huggingface/bench_20260801/` | shared, world-writable |
| kvd L3 tier | `/tmp/kvd-long` **inside the container** | on the node's root disk, **not** `/mnt/vast` — this is the §0003 trap |

`/mnt/vast` is `10.2.123.177:/aac-8634674/aac/shared/data`, 501 TB, ~75 % used.

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| cluster SSH | key-based access to `root@149.28.124.225`, then node-to-node as root. Arrange your own; nothing in this kit contains a key. |
| docker registry | **not needed** — both the base image and `merged-e` were already present on both nodes. A cold node would need the team registry login to pull `lmsysorg/sglang`. |
| etcd | **unauthenticated** on the prefill node's private data-plane IP. No credential. |
| router / engine | no API key (`api_key=None`, `admin_api_key=None` in server args). |

No API keys, tokens, or S3 credentials are involved. **No secret value appears anywhere
in this kit** — the env snapshots and logs were checked.

## Deployment under test

    two-node PD over mooncake RDMA   ionic RoCE, MC_GID_INDEX discovered per node
    DP-attention 8/8 both legs       --dp-size 8 --enable-dp-attention --ep-size 8
    kv-aware routing                 ON, RUST backend, w_prefill 20.0 / w_decode 2.0
    kvd (infera HiCacheStorage)      prefill ON (--hicache-size 16), decode skipped by design
    MTP                              decode leg only, EAGLE steps=3 topk=1 draft=4
                                     + --disable-custom-all-reduce
    --context-length                 262144
    --chunked-prefill-size           65536      (= 8192/rank at dp8)
    --cuda-graph-max-bs              128
    --max-running-requests           2048       (engine reports 256 effective)
    --enable-cache-report            ON         (bench delta; else cache-hit reads 0)
    --kv-cache-dtype                 fp8_e4m3
    mem-fraction-static              prefill 0.80 / decode 0.85     <-- see patches/0002
    max_total_num_tokens             2,829,952 per rank at gmu 0.80 (was 3,260,672 at 0.88)

**`mem-fraction-static` changed mid-experiment.** The p50 rounds ran at prefill 0.88;
the p90 rounds at 0.80 after the OOM fix. The two pairs are therefore not a controlled
comparison of each other — stated in `notes.md` and `README.md`.

## Transport — vultr, not spur

The two clusters are configured oppositely and the wrong block silently drops to TCP:

    MC_GID_INDEX=<discovered: 1 on chi2879, 2 on chi2867>
    MC_DISABLE_HIP_TRANSPORT=1
    MOONCAKE_DISABLE_HIP_DMABUF=1
    disaggregation-ib-device = ionic_0..ionic_7   (all 8, auto-detected as ACTIVE+ionic)

Verified in-run: `MC_FORCE_TCP` hits **0**, mooncake init failures **0**, 194 ionic
mentions in the prefill log.

> The spur kit's `IBDEV=mlx5_0 / MC_GID_INDEX=3 / MC_MS_FILTERS=mlx5_0` block is **wrong
> here**. These nodes' mlx5 path is not the KV fabric.

## Gaps, stated rather than guessed

- **`accept len` was sampled from the p2 decode log (11 values, 1.52–2.67), not
  aggregated across the whole sweep.** The sweep rounds' decode log is in
  `logs/p3_decode.log.gz` if a fuller distribution is wanted.
- **No kvd-off A/B** was run, so no performance claim is made for kvd in this phase.
- The kvaware weight sweep (w ∈ {1.0, 20.0}) was **not** run — the OOM debugging
  consumed its budget.
