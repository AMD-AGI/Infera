# Environment

## Time

All runs 2026-10-13 UTC (per-run start/end in `results/<round>/<round>.chain.log` and `status.tsv`):

| round | node | bring-up start | A1 | A2 | A3 |
|---|---|---|---|---|---|
| x8d | crsuse2-m2m-026 | 06:28 | 07:01-08:27 | 08:27-08:41 FAIL (ECC) | — |
| x8p | crsuse2-m2m-274 | 06:36 | 07:03-08:29 | 08:29-09:51 | (C52 not run: driver killed by plan, 274 reused for pc48) |
| x8d2 | crsuse2-m2m-206 | 09:30 | 09:51-11:16 | 11:16-12:39 | C52 12:39-14:02 |
| x8pc | crsuse2-m2m-206 | 15:11 | 15:34-16:59 | 16:59-18:23 | — |
| x128p | crsuse2-m2m-085 | 16:49 | 17:16-18:40 | 18:40-20:03 | — |
| q1 (reference) | crsuse2-m2m-137 | 2026-10-12 10:54 | C44 11:21-12:43 | C44 12:43-14:04 | **C48 14:04-15:24 (the reference)** |

## Nodes / allocations

| node | data IP | allocation | state at pack-up |
|---|---|---|---|
| crsuse2-m2m-026 | 10.245.153.227 | spur oai job 211170 | **released** 2026-10-13 ~08:50 (GPU3 HBM ECC: UMC correctable 13,198,148, deferred 2, `amd-smi metric --ecc`; GPUs 0-2,4-7 = 0) |
| crsuse2-m2m-274 | 10.245.157.34 | 211171 | held |
| crsuse2-m2m-206 | 10.245.148.83 | 211763 | ended 2026-10-14 ~01:30 (JobHoldMaxRequeue) |
| crsuse2-m2m-085 | 10.245.153.145 | 211169 | held |
| crsuse2-m2m-137 | 10.245.153.247 | plain ssh (no job); `/shared_nfs` read-only → `NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg` | used by another user since 2026-10-13 |

All: `sbatch -A amd-oai -p amd-spur -q amd-oai-qos --gres=gpu:8 -c 160 -t 24:00:00`. Foreign container `crusoe-sgl-q36`
(guanchen, a `pool_mtp.sh` loop able to relaunch sglang on GPU0-5) was present on 085/274/206 and `docker stop`ped (not
removed) with user approval before use.

## Hardware (first-hand, `environment/collect_env.{085,274}.20261014.txt`, captured 2026-10-14 01:57 UTC)

| item | 085 and 274 (identical except IDs) |
|---|---|
| OS / kernel | Ubuntu 24.04.4 LTS, Linux 6.8.0-107-generic |
| CPU / RAM | AMD EPYC 9575F, 2 sockets (236 CPUs visible), 2.7 TiB RAM, no swap |
| GPU | 8x AMD Instinct MI355X (gfx950), 309,220,868,096 B VRAM each; rocm-smi driver version 6.14.14 |
| RDMA | `ionic_0..7` (400 Gb/s, Ethernet link layer, all Active/LinkUp) + `mlx5_0` (200 Gb/s); harness binds `ionic_0..3` for both legs |
| docker | 29.7.2 |

**Gaps:** 026 and 206 were not captured (026 released after ECC, 206's job ended before the capture); assumed the same
node class [inferred]. 137 (reference) is described in the sibling pack-up `glm52.v0521-cap-c44c48.packup_20261013/environment.md`.
The collect_env snapshots were taken while other stacks were up (VRAM use non-zero) — they are inventory, not run-time state.

## Software

| item | value |
|---|---|
| image | `infera-sglang:v0521-yihou-1010-mc4340`, ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` on 026/085/274/206 (docker load of the 102 tar, ID verified); 137: `sha256:e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b` (same tar, re-IDed by the store) |
| image tar | `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` (23,524,473,704 B) |
| base image | `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008` (`config.yihou.v0521-base.sh`); build recipe in the baseline kit |
| sglang source of the image (diff base) | `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (config.json sha256 `8b46225f7afd7181735c2dd97b925bcb70dfebb72b1a105d4ac0be7583a5c726`) |
| runtime mounts (both legs, all rounds) | `mounts.yihou.v2-regbase.tsv`: `kv_events_publisher.yihou-noblock.py`, `forward_mla_rocm.yihou-noabsorbcfg.py`, `conn.yihou-regbase.py` (copies in `scripts/harness/mounts-v0521/src/`); host `/lib/x86_64-linux-gnu/libionic.so` |
| per-round mounts | reduced leg only: `mounts.yihou.expert8.tsv` or `mounts.yihou.expert128.tsv` (see `patches/`) |
| full rendered argv + env per leg | `scripts/dryrun/dryrun.<round>.yihou.txt` |
| harness | `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (not in git; copied to `scripts/harness/`) |
| repo | `/home/yihou/dev/git/infera.glm52.pd`, branch `dev/pd_opt/glm_5.2_agentx` @ `a3ba39687e884201839f75a45374cc145d07e7fa` (dirty tree; the harness does not live in the repo) |
| benchmark | AgentX (aiperf) in the harness client container; dataset `semianalysisai/cc-traces-weka-062126` (393 entries), 3600 s, warmup 10 requests/lane |

## Secrets / credentials

None packed, none in the copied `runtime.env` files (grepped for token/key/password/secret: no hits).
Needed: spur cluster login (ssh to `spur006` = `crsuse2-slog-006`) with the `amd-oai` account/QoS; docker on the held node
(via `spur exec`). No registry or HF login (image from the NFS tar, model local).
