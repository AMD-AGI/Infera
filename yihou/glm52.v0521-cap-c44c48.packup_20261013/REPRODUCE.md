# Reproduce — best run q1 A3 (C48, 22,306/chip) and the Track B patch rounds

Everything builds on the baseline kit `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012` (read its REPRODUCE §0-2
for allocation, image, harness copy). Harness here = `scripts/harness/` (verbatim copy of
`/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` at pack-up time; absolute paths baked in — run from the original
location or sed the WS path as in the baseline REPRODUCE §2).

## 1. Image on a new node

`infera-sglang:v0521-yihou-1010-mc4340` (102 ID sha256:9d755dd6…). A docker-save is on NFS:
`/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` (23.5 GB).
`zstd -dc <file> | docker load`. The loaded ID may change with the node's docker store (174/029 kept 9d755dd6; 137 got
e8e2838f…): add `<node>) EXPECTED_IMAGE_ID=<id>` to `scripts/harness/config.yihou.mc4340n.sh` (new file; the original
`config.yihou.mc4340.sh` only knows 102/136). Seed JIT caches from `$WS/cache/102/{aiter,triton}` (cp -a) to save ~30 min.

## 2. q1 recipe (C44 → C48 on one stack)

Env `scripts/harness/env.q1-137.yihou.sh` = e2g recipe + `DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88`, HiCache 450,
decode mem 0.95, CONC 44; plain-ssh node → `NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg` (NFS read-only on node).
Retarget YIHOU_NODE / YIHOU_IP / YIHOU_JOB for your node (spur node: set YIHOU_JOB, NODE_RW_ROOT empty).
```bash
cd $WS/harness && source env.q1-137.yihou.sh && bash run.v2.yihou.sh dryrun   # compare with results/dryrun.q1-137.yihou.txt
setsid nohup bash chain_q1.yihou.sh > ../runs/q1.chain.log 2>&1 < /dev/null &      # A1 (fresh C44) -> flush -> A2 (warm C44)
setsid nohup bash a3_q1c48.yihou.sh > ../runs/q1.A3c48.log 2>&1 < /dev/null &      # waits for q1 chain done -> flush -> C48 (A3)
```
The C48 run (A3) is the 3rd AgentX run on the same stack; CONC is client-side only (server args identical).
Expected: A2 ≈ 21.9k, A3 ≈ 22.3k (`results/q1-A3-C48/agentx_conc48.json`).

## 3. Track B patch rounds (decode in-flight KV cap)

- v1: `CONFIG=config.yihou.runshare.sh` (decode mount `decode.yihou-runshare.py`, env `YIHOU_DECODE_RUNNING_KV_SHARE=0.5`).
- v2: `CONFIG=config.yihou.runshare2.sh` (`decode.yihou-runshare2.py`, + `YIHOU_RUNSHARE_MAX_WAIT_S=5`).
- Rounds: `env.pc40c-102.yihou.sh` / `chain_pc40c.yihou.sh` (v1), `env.pc40d-102.yihou.sh` / `chain_pc40d.yihou.sh` (v2):
  up → census → smoke → audit → 64-prompt long burst (must log `YIHOU-RUNSHARE block`, keep decode "pre-allocated usage"
  ≤ 0.5) → C40 A1 → flush → A2. Burst tool: `scripts/tools/longprompt_burst.yihou.py URL 64 200000 16`.
- Verify live: `strings decode-0.a1.log | grep -a 'YIHOU-RUNSHARE block'` (logged every 100th block; fields nonrunning /
  cand / cap / running / xfer / waited), decode log `pre-allocated usage`.

## 4. Telemetry and analysis

- `/metrics` sampler: `scripts/tools/metrics_sampler.yihou.py OUT.tsv DURATION 5 decode=http://IP:29357 prefill=http://IP:29101`.
- Little's-law phases: `scripts/tools/residence.yihou.py TSV 2026-10-12 HH:MM:SS HH:MM:SS QPS` (window = "Phase profiling"
  start/complete in benchmark.log; QPS = `request_metrics.qps.mean`).
