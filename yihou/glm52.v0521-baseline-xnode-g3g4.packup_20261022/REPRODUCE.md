# REPRODUCE — 10-21 baseline, same-node knobs, cross-node 1P1D

Paths below are the live NFS paths used on 10-21 (`H` harness, `R` runs, `W` g4tune workspace, `N` session tools). The
files this packup copies from them are in `scripts/` (harness) and `scripts/envs/` (env files). Run everything from a spur
login host (`ssh spur006`, fallback spur007/005); login hosts evict a user above ~4 GB, so keep login-side commands tiny.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
W=/shared_nfs/yihou_glm52/g4tune_20261019
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
```

## 0. Prerequisites

- Image `infera-sglang:v0521-yihou-1010-mc4340` from the tar
  `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst`.
  `docker load` on a spur slurm node keeps ID `9d755dd6...`; on 135/136 (plain-ssh, overlay2 store) it becomes `e8e2838f...`
  (`config.yihou.mc4340n19.sh` lists both).
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (read-only NFS).
- Mount patches: `mounts.yihou.v2-xferfix2-gtt-idlefix.tsv` (sources under `$H/mounts-v0521/src/`, idlefix copy in
  `scripts/mounts-src/eager_runner.yihou-idlefix.py`) and prefill-only `mounts.yihou.offload-attrs.tsv` (OffloaderV2
  `is_shuffled` patch `/shared_nfs/yihou_glm52/v0521cap2_20261013/offload/offloader.yihou-attrs.py`).
- Nodes: spur slurm nodes only through the machine pool (`~/.claude/skills/machine-pool`, never sbatch yourself), reached with
  `spur exec <job> -- ...`; crsuse2-m2m-135/136 by plain ssh from the login host.

## 1. Node prep (once per node)

```bash
# slurm node:  spur exec <job> -- bash $N/prep_node.yihou.sh <key> seed     (docker load + seed aiter JIT cache from cache/102)
# 135 / 136:   ssh crsuse2-m2m-135 "setsid nohup bash $N/prep_node.yihou.sh 135 seed >/dev/null 2>&1 &"
#              ssh crsuse2-m2m-136 "setsid nohup bash $N/prep_node.yihou.sh 136x noseed >/dev/null 2>&1 &"
# 135/136 only: engine.sh keys the aiter JIT cache by the LOCAL image ID -> copy the seed under the e8e2838f key:
C=/shared_nfs/yihou_glm52/v0521dbg_20261010/cache/135
for r in prefill decode; do cp -a $C/aiter/$r/9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2 \
  $C/aiter/$r/e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b; done
```
Checks done before any launch (all passed 10-21): docker active; `/mnt/m2m_nobackup` mounted from the nvme LV and docker's
root dir on it; `/` < 92 %; NFS rw and NOT root-squashed (container `touch` test); 0 KFD processes; rails ACTIVE.

## 2. RDMA rail probe (cross-node only)

```bash
for d in ionic_0 ionic_1 ionic_2 ionic_3; do bash $W/xnode/rail_probe.yihou.sh $d crsuse2-m2m-135 crsuse2-m2m-136; done
# -> $W/xnode/preflight/<dev>-<UTC>/crsuse2-m2m-13{5,6}.json ; copies in results/xnode/preflight/
```

## 3. Same-node round (any held slurm node)

Every env file is `scripts/envs/env.<round>-<node>.yihou.sh`; one variable vs the baseline env (`env.idxf1b-268.yihou.sh`).
```bash
source $W/env.<round>-<key>.yihou.sh
bash $H/run.v2.yihou.sh dryrun > $W/dryrun.<round>-<key>.yihou.txt          # then token-diff vs the baseline dryrun:
diff <(tr ' ' '\n' < dryrun.idxf1b-268.yihou.txt | sed s/idxf1b/ROUND/g) <(tr ' ' '\n' < dryrun.<round>-<key>.yihou.txt | sed s/<round>/ROUND/g)
# fresh node:
spur exec <job> -- bash -c "cd /; setsid nohup bash $W/launch_after_prep.yihou.sh $W/env.<round>-<key>.yihou.sh <round> <key> 56 56 >/dev/null 2>&1 &"
# replace a finished round on the same node (stop old stack, wait 0 KFD, start chain_cap2 C1 C2):
spur exec <job> -- bash -c "cd /; setsid nohup bash $H/stop_then_chain.yihou.sh <old-env> <new-env> 56 56 > $W/stc.<old>-<new>.yihou.log 2>&1 &"
```
chain_cap2 = pipeline.v3 (up -> census -> smoke -> audit -> A1 @C1) -> flush both legs -> A2 @C2; stack left running;
stop with `bash $W/stop_round.yihou.sh <key> <round>` on the node (docker stop only).

hipxfer shim round (g4h1): `CONFIG=$H/config.yihou.xfertrace.sh` (adds `LD_PRELOAD=/opt/yihou/libhipxfer.yihou.so` to the
prefill only). Build the shim (gcc, ROCm 7.0 headers; glibc 2.35 like the image):
`gcc -O2 -Wall -I/opt/rocm-7.0.2/include -fPIC -shared -o libhipxfer.yihou.so scripts/mounts-src/hipxfer_trace.yihou.c -ldl -lpthread`.
Local smoke tests: `scripts/t_hipxfer.yihou.c`, `scripts/t_local.yihou.c` + `scripts/fakemc.yihou.c` (RTLD_LOCAL load).

## 4. Cross-node round (135 prefill / 136 decode)

```bash
source $W/env.g4x2-135x136.yihou.sh                 # or env.g4x1 / env.g3x1 (OFF_G=3)
bash $H/run.v3x.yihou.sh dryrun > $W/dryrun.g4x2-135x136.yihou.txt
ssh crsuse2-m2m-135 "cd /; setsid nohup bash $H/chain_x.yihou.sh $W/env.g4x2-135x136.yihou.sh 56 56 > $R/g4x2.chain.log 2>&1 < /dev/null &"
# conc sweep on the same live stack after the chain (A3 C60, A4 C64):
ssh crsuse2-m2m-135 "cd /; setsid nohup bash $H/conc_after_x.yihou.sh $W/env.g4x2-135x136.yihou.sh 3 60 64 > $R/g4x2.conc.log 2>&1 < /dev/null &"
# or chain old -> new in one go (waits for the old bench tag, stops the old stack, checks both nodes clean):
ssh crsuse2-m2m-135 "cd /; setsid nohup bash $H/after_x_then.yihou.sh $W/env.g4x2-135x136.yihou.sh agentx-c64-A4 $W/env.g3x1-135x136.yihou.sh 56 56 3 60 64 > $W/after.g4x2-g3x1.yihou.log 2>&1 < /dev/null &"
# stop: ssh crsuse2-m2m-135 "cd $H && source $W/env.<round>-135x136.yihou.sh && bash $H/run.v3x.yihou.sh stop"
```
The env must export `XNODE_DECODE_NODE/XNODE_DECODE_IP` AND carry them plus `NODE_RW_ROOT=` in `YIHOU_OVERRIDES` (chain_x checks).

## 5. Analysis

```bash
python3 -I $N/hist.yihou.py <label> <bench>/aiperf_artifacts/server_metrics_export.csv    # transfer-stage histograms
python3 -I $H/tools/xfer_summary.yihou.py $R/<round>/server-logs/prefill-0.log <start-iso> <end-iso>   # per-call Mooncake
bash scripts/summary.yihou.sh <stage-dir>          # results/summary.tsv (run on a compute node: reads all bench CSVs)
python3 analysis/tpot_proxy_binning.a.py            # TPOT vs busiest-DP-rank running (paths inside the script)
```
