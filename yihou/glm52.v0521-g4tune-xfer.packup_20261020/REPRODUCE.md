# Reproduce — g4tune C56 rounds (G4 OffloaderV2 + xferfix2 transfer overlay)

All paths are on the spur cluster (`ssh spur006`, fallback spur007/spur005). Login hosts evict a user above ~4 GB and have
1 CPU: run drivers ON the node, keep login-host commands small.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness     # verbatim copies of what is needed in scripts/harness/
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
W=/shared_nfs/yihou_glm52/g4tune_20261019                # env.*, dryrun.*, launchers (copies in scripts/session/)
C=/shared_nfs/yihou_glm52/v0521dbg_20261010/cache        # JIT caches, key = last '-' field of the hostname
```

## 0. Prerequisites

As in `../glm52.v0521-offloaderv2-burst-c56c64.packup_20261019/REPRODUCE.md` §0: model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`,
image `infera-sglang:v0521-yihou-1010-mc4340` (ID `9d755dd6a7bf…`), the OffloaderV2 YIHOU-OFFLOAD-ATTRS patch
(`patches/mounts-v0521/offloader.yihou-attrs.py`, mounted by `mounts.yihou.offload-attrs.tsv`).

## 1. Node

Request through the machine pool (OAI is allowed again from 2026-10-19: `-A amd-oai -q amd-oai-qos -p amd-spur`; fallback
amd-spur-v2 burst, which is preemptible — burst holds often end at 20:00 and cannot cover the 25–35 min bring-up).
Then prepare it (image load + cache seed):

```bash
spur exec $J bash -c "cd /; setsid nohup bash $N/prep_node.yihou.sh <key> seed > /dev/null 2>&1 &"
ls $C/<key>/aiter/prefill/9d755dd6*/ | grep -c '\.so$'   # 16 (decode 14); lock_* files 0; KFD pids 0
```

## 2. Config chain (`scripts/harness/`)

`CONFIG=config.yihou.xferfix2.sh` → `XFERFIX_BASE_CONFIG=config.yihou.offload.sh` (OffloaderV2 `OFF_G/OFF_K/OFF_STEP`) →
`OFF_BASE_CONFIG=config.yihou.tokw2.sh` (decode tok4/detok4) → `TOKW2_BASE_CONFIG=config.yihou.mc4340n16.sh` (node table incl. m2m-250 and v3-*;
`n17` adds m2m-097 + `topology.yihou.097.tsv`) → `MC4340_BASE_CONFIG=config.yihou.E2-target.sh` → `config.yihou.v0521-base.sh`.
xferfix2 swaps the mounts to `mounts.yihou.v2-xferfix2.tsv` (4 engine overlays, `patches/`) and adds prefill env
`YIHOU_XFER_LOG / YIHOU_XFER_SHARD / YIHOU_NONFINAL_EVENT`.

| point | env (`scripts/session/`) | knob vs R1 |
|---|---|---|
| R1 baseline | `env.g4xb5-250.yihou.sh` | — (`YIHOU_XFER_SHARD=dest`, chunk 32768) |
| R2 mix (same node) | `env.g4xm4-250.yihou.sh` | `YIHOU_XFER_SHARD=mix` |
| R2 mix (burst) | `env.g4xm3-v3017.yihou.sh` | `YIHOU_XFER_SHARD=mix`, node v3-017 |
| R3 chunk | `env.g4xc1-097.yihou.sh` | `PREFILL_CHUNK_SIZE=65536`, node 097 (config n17) |
| R3b chunk | `env.g4xc2-250.yihou.sh` | `PREFILL_CHUNK_SIZE=65536` |

Common: `OFF_G=4 OFF_K=1 OFF_STEP=1 CONC=56 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61`.

For a new node: copy an env, change `YIHOU_NODE/YIHOU_IP/YIHOU_JOB/YIHOU_ROUND`, then dryrun + tokdiff (must show only node/IP or the intended knob):

```bash
( cd $H; source $W/env.<new>.yihou.sh; bash $H/run.v3.yihou.sh dryrun > $W/dryrun.<new>.yihou.txt )
bash $H/tokdiff.yihou.sh $W/dryrun.g4xb5-250.yihou.txt $W/dryrun.<new>.yihou.txt
```

## 3. Launch (on the node; short round = A1 fresh + A2 scored at C56)

```bash
spur exec $J bash -c "cd /; setsid nohup bash $W/launch_after_prep.yihou.sh $W/env.<round>.yihou.sh <round> <key> 56 56 > /dev/null 2>&1 &"
# = wait for prep_node, check img/.so/locks/KFD, then chain_cap2.yihou.sh <env> 56 56 + warmup_pyspy.yihou.sh; log $W/launch.<round>.yihou.log
```

If the bench wrapper dies but the A1 client container keeps running (as g4xm4 on EDQUOT): `bash $W/a2_after_orphan.yihou.sh <env> 56` on the node
(waits for the A1 client to exit, flushes both legs, runs A2). Between rounds: `( source <env>; bash $H/run.v3.yihou.sh stop )`, check `docker ps` and KFD 0.
Bring-up ~25–35 min; each 3600 s bench ~86–90 min wall.

## 4. Read-out (on a login host, from $R)

```bash
cd $R && python3 -B $N/tools_hits.yihou.py "g4xb5-A2=g4xb5/agentx-c56-A2/agentx_conc56.json"
python3 -I $N/hist.yihou.py g4xb5-A2 $R/g4xb5/agentx-c56-A2/aiperf_artifacts/server_metrics_export.csv     # -> results/hist.g4tune.jsonl
python3 -I $W/xfer_sum.yihou.py $R/g4xb5/server-logs/prefill-0.log                                         # YIHOU-XFER per DP / 5 min
```

Note: `tools_hits.yihou.py` takes `label=relative/path` arguments (the label must not contain `=`).

## 5. Machine microbenchmarks (`scripts/machine/`, results in `results/machine/`)

Run inside the production image on a held node, one `docker run --rm` per run. The one recorded invocation is
`scripts/machine/run_sdma.yihou.sh` (template for all mc_bw runs; it hard-codes job 225141 / IP 10.245.145.196 of v3-021):

```bash
docker run --rm --name yihou-mcbwN --network host --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined -e YX_IP=<node data IP> -e YX_LOADSIDE=send -e MC_TE_FILTERS=yihou_no_such_rdma_device \
  -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 -e MC_GID_INDEX=1 -e MC_LOG_LEVEL=INFO -v $W/machine:$W/machine \
  infera-sglang:v0521-yihou-1010-mc4340 python3 $W/machine/mc_bw4.yihou.py
```

- `mc_bw.yihou.py`: Mooncake `batch_transfer_sync_write` GPU s → s+4, block shapes, pairs 1 vs 4, threads per sender.
- `mc_bw2/3.yihou.py`: production-like registration (many buffers, `nbuf`) + optional bf16 matmul burn; `mc_bw4.yihou.py`: burn side via `YX_LOADSIDE=send|recv`.
  360 KiB blocks × 5825 = 2 GiB per call. Which flag set produced which log is identified by the log file name and its JSON fields.
- `p2p_bw.yihou.py`: torch peer copy GPU0-3 → GPU4-7, 4 GiB as 1 block vs 10000 blocks.
- `xgmi_load.yihou.py`: torch copies GPU g → g+4 (all 4 pairs at once) with load 0 / 1 (bf16 burn on sender); output `results/machine/xgmi_load.097.yihou.log` (m2m-097, 2026-10-20 01:28Z, complete: 8 JSON lines).
