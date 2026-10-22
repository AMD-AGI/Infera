# Reproduce — OffloaderV2 short rounds at C56 / C64 on burst nodes

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # verbatim copy in scripts/harness/
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
C=/shared_nfs/yihou_glm52/v0521dbg_20261010/cache            # JIT/client caches, key = last '-' field of the hostname
```

## 0. Prerequisites

Same as `../glm52.v0521-offloaderv2.packup_20261015/REPRODUCE.md` §0: model, image tar `9d755dd6`, the YIHOU-OFFLOAD-ATTRS patch (copy in `patches/`). Login hosts evict any user above ~4 GB: keep commands small, run drivers on the node.

## 1. Hold + prepare a node (amd-spur-v2; amd-spur holds never survived bring-up, see notes)

```bash
sbatch --parsable --export=ALL -A amd-burst -p amd-spur-v2 -q amd-burst-qos -N1 --gres=gpu:8 -c 160 -t 24:00:00 \
    -J yihou-ofs-burst-v2 ~/spur-hold/hold_node.sh
# on the node: docker stop (never rm) non-system containers holding GPUs (user-authorized), then
spur exec $J bash -c "cd /; setsid nohup bash $N/prep_node.yihou.sh <key> seed > /dev/null 2>&1 &"   # docker load + seed cache/<key> from cache/102
ls $C/<key>/aiter/prefill/9d755dd6*/ | grep -c '\.so$'    # 16 (decode 14); lock_* files: 0
```

## 2. Config chain and env (`scripts/harness/`)

`CONFIG=config.yihou.offload.sh` → `config.yihou.tokw2.sh` (decode tok4/detok4) → `config.yihou.mc4340n7.sh` (covers every `crsuse2-m2m-v3-*`; n8–n16 add the amd-spur nodes) → `config.yihou.E2-target.sh` (pf90: prefill mem 0.90, HiCache 450 GB, decode mem 0.95, max_running 96, sim 3.61).

| point | env | knobs that differ |
|---|---|---|
| G2 C56 | `env.ofg2h21-v3006.yihou.sh` | `OFF_G=2 CONC=56` |
| G3 C64 | `env.ofg3h14-v3006.yihou.sh` | `OFF_G=3 CONC=64` |
| G2 C64 | `env.ofg2h22-v3006.yihou.sh` | `OFF_G=2 CONC=64` |
| G4 C64 | `env.ofg4h5-v3006.yihou.sh` | `OFF_G=4 CONC=64` |
| G3 C56 | `env.ofg3h2-v3010.yihou.sh` | `OFF_G=3 CONC=56` (node v3-010) |

For another node: copy the env, change `YIHOU_NODE/YIHOU_IP/YIHOU_JOB/YIHOU_ROUND`, then

```bash
( source env.<new>.yihou.sh; bash run.v3.yihou.sh dryrun > $N/dryrun.<new>.yihou.txt )
bash tokdiff.yihou.sh $N/dryrun.<ref>.yihou.txt $N/dryrun.<new>.yihou.txt   # must show only node/IP (or the intended knob)
```

## 3. Launch (driver ON the node)

```bash
spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.ofg2h21-v3006.yihou.sh 56 56 > $R/ofg2h21.chain.log 2>&1 < /dev/null &"
spur exec $J bash -c "cd /; setsid nohup bash $H/warmup_pyspy.yihou.sh ofg2h21 006 <ISO-now> 300 > $R/ofg2h21.pyspy.log 2>&1 < /dev/null &"
```

Bring-up ~25–35 min, each 3600 s bench ~88–90 min wall → one round ≈ 3.5 h of node time. Between rounds on a held node: `( source env.<prev>; bash run.v3s.yihou.sh stop )`, check `docker ps` empty and KFD 0, then launch the next env.
If the driver dies but the stack is healthy: `bash $H/cont_bench.yihou.sh <env> - <C>` on the node (flush both legs → A2).

## 4. Read-out

```bash
cd $R && python3 -B $N/tools_hits.yihou.py "name=ofg2h21/agentx-c56-A2/agentx_conc56.json"
python3 -B scripts/session/results_csv_burst.yihou.py > results.csv
```
