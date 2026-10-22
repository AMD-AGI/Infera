# Reproduce — pf90 / pf92 (prefill mem 0.90 / 0.92), wts48 / wts64 (HiCache write_through_selective), pc64, cx90

Wall time per round on a node with the image loaded and JIT caches seeded: VRAM drain of a previous stack (5-30 min) +
gpuidle wait + bring-up 22-37 min (`status.tsv` start → bothlegs: pf88 22, pc64 21, wts48 23, pf90 35, pf92 36; pf90's prefill weight load took 257 s vs 45-127 s elsewhere) + A1 (~84 min incl. warm-up) + flush + A2 (~83 min).
Expected (single replicates): pf90 A1 ≈21.9k / A2 ≈23.1k; pf92 A1 ≈23.0k; wts48 A2 ≈21.2k (after a partial A1); pc64 A1 ≈10.5k tok/s/chip.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # harness (verbatim copy in scripts/harness/)
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs         # round outputs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013             # session workspace (shim, pcomplete mounts, metrics, dryruns, correctness tests)
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012            # metrics sampler, image tar, extracted sglang source
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010             # $WS/cache/<node-key>/{aiter,triton} JIT caches (oai nodes)
```
Scripts have these paths baked in. To run from this kit instead, copy `scripts/harness/*` to a new directory and `sed` `H`/`WS`
in the env files and configs (baseline kit REPRODUCE §2).

## 0. Prerequisites

- spur cluster login (`ssh spur006` = `crsuse2-slog-006`), account/QoS `amd-oai` / `amd-oai-qos`. **Secrets:** none beyond the cluster
  login (model + image tar on NFS; AgentX dataset public; no tokens in any env file — grepped).
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (256 experts, unmodified), served as `glm5.2-mxfp4`.
- Image `infera-sglang:v0521-yihou-1010-mc4340`: tar `$CAP/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst`
  (ID `sha256:9d755dd6…` after `docker load`; 089/085/274). Node 136 runs its own build `sha256:9872bbda…` (see environment.md).
  Rebuild recipe: `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/REPRODUCE.md` §1.

## 1. Node, image, JIT seed

Same as `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-prefill-complete.packup_20261014/REPRODUCE.md` §1-3
(`sbatch … hold`, `spur exec` checks: 0 KFD procs, < 400 MiB per GPU, ECC clean; `zstd -dc <tar> | docker load`; seed `$WS/cache/<K>`).
A new oai node needs a **new** config copy with `<node>) EXPECTED_IMAGE_ID=…` — today `config.yihou.mc4340n4.sh` = mc4340n3 + node 089
(diff is one comment + one `case` alternative). Node 136 (plain ssh) uses `config.yihou.mc4340n.sh` and node-local
`NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg`.

## 2. Env files (scripts/harness/env.<round>-<node>.yihou.sh; each header states the question)

| round | env | CONFIG | YIHOU_OVERRIDES difference vs q1 (`env.q1-137`) |
|---|---|---|---|
| pf90 | `env.pf90-089` | `config.yihou.mc4340n4.sh` | `PREFILL_MEM_FRACTION=0.90` |
| pf92 | `env.pf92-085` (was `env.pf92-206`) | `mc4340n4` | `PREFILL_MEM_FRACTION=0.92` |
| pf88 | `env.pf88-274` (rerun prepared as `env.pf88b-085`) | `mc4340n4` | none (0.88 control) |
| wts48 / wts64 | `env.wts48-136` | `config.yihou.mc4340n.sh` | `PREFILL_HICACHE_WRITE_POLICY=write_through_selective` (wts64: same stack, CONC 64) |
| pc64 | `env.pc64-089` | `config.yihou.pcomplete2.sh` (→ `PCOMPLETE_BASE_CONFIG=mc4340n4`) | pc48b config: `PREFILL_HICACHE_SIZE=370`, `DECODE_MEM_FRACTION=0.96`, policy `prefill_complete` both legs (forces write_back), attndp hook via `YIHOU_MOUNTS=mounts.yihou.v2-regbase-attndp.tsv`; `CONC=64` |
| cx90 | `env.cx90-089` | `mc4340n4` | pf90 + `DECODE_SIMULATE_ACC_LEN=` (empty → simulated acceptance OFF) |
| pf93 / pf94 | `env.pf93-136` / `env.pf94-136` | `mc4340n` | `PREFILL_MEM_FRACTION=0.93` / `0.94` |

Common overrides: `MC4340_BASE_CONFIG=config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 DECODE_MEM_FRACTION=0.95
ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000`.

## 3. Dryrun + token diff (must show only the intended change + node paths/IP)

```bash
cd $H && source env.pf90-089.yihou.sh && bash run.v3.yihou.sh dryrun > $N/dryrun.pf90-089.yihou.txt
bash $H/tokdiff.yihou.sh $H/dryrun.q1-137.yihou.txt $N/dryrun.pf90-089.yihou.txt
```
Recorded diffs: `results/dryruns/tokdiff.*.txt` (pf90 vs q1: `0.88→0.90` + paths/IP; wts48 vs q1: `write_through→write_through_selective` + paths;
cx90 vs pf90: only the 3 `SGLANG_SIMULATE_ACC_*` envs on decode).

## 4. Launch

oai node (driver ON the node, local ssh shim `$N/shimbin/ssh`):
```bash
spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.pf90-089.yihou.sh 48 48 52 > $R/pf90.chain.log 2>&1 < /dev/null &"
```
(chain = pipeline.v3 gpuidle → etcd → prefill → decode → router → census → smoke → audit v3 → A1 fresh C48 → flush decode :29357 + prefill :29101
→ A2 C48 → flush → A3 C52. Today the pf90/pf92 drivers were killed after A2 so no A3 ran.)
Hand a node over to the next round after an A2: `after_a2_then.yihou.sh <round> <C2> <old-env> <new-env> C1 C2` (→ `stop_then_chain`).

plain-ssh node 136 (driver on the login host, writes to node-local `/mnt`):
```bash
ssh spur006 "cd /; setsid nohup bash $H/chain_cap2l.yihou.sh $H/env.wts48-136.yihou.sh 48 48 52 > $R/wts48.chain.log 2>&1 < /dev/null &"
# continue on a live stack (no A1): flush -> A2 @C
ssh spur006 "cd /; setsid nohup bash $H/cont_benchl.yihou.sh $H/env.wts48-136.yihou.sh - 48 >> $R/wts48.cont.log 2>&1 < /dev/null &"
ssh spur006 "cd /; setsid nohup bash $H/cont_benchl.yihou.sh $H/env.wts48-136.yihou.sh - 64 >> $R/wts48.cont.log 2>&1 < /dev/null &"   # wts64
# then pf94 on the same node: after_wts64_pf94.yihou.sh (waits for wts64, stops wts48, runs chain_cap2l env.pf94-136 48 48)
```
pc64 (pipeline stopped at smoke, then): `cont_bench.yihou.sh $H/env.pc64-089.yihou.sh 64 64`.

Correctness round (node 089): `spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cx.yihou.sh $H/env.cx90-089.yihou.sh > $R/cx90.chain.log 2>&1 < /dev/null &"`
→ up → census → smoke → audit → `python3 $N/correctness_tests.yihou.py http://<ip>:28100 glm5.2-mxfp4 $N/runs/cx90` → stop.

## 5. Memory measurements

```bash
zgrep -aE 'DP0 TP0.*(Load weight end|KV Cache is allocated|Memory pool end|max_total_num_tokens)' logs/server/pf90/prefill-0.log.gz
spur exec $J bash -c "setsid nohup bash $N/vram_peak.yihou.sh $N/runs/<round>/vram.<K>.tsv 7200 > /dev/null 2>&1 &"   # rocm-smi every 5 s
```

## 6. Read the scores

```bash
ssh spur006 "python3 -B <kit>/scripts/tools/results_csv.yihou.py"     # = results/results.csv
ssh spur006 "python3 -B $N/tools_hits.yihou.py 'pf90 A2=pf90/agentx-c48-A2/agentx_conc48.json'"
```
Score = `request_metrics.throughput.per_gpu.total_tput_tps`.

## 7. Stop

`cd $H && source env.<round>.yihou.sh && PATH=$N/shimbin:$PATH bash run.v3.yihou.sh stop` (oai: inside `spur exec`; 136: from spur006 without the shim).
VRAM release after a HiCache 370-450 stack takes ~10-20 min; the next pipeline's gpuidle gate waits on it.
