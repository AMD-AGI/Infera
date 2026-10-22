# Task: GLM-5.2 v0.5.21 1P1D P4D4 — more prefill+decode capacity, AgentX CONC 44→48, > 23k tok/s/chip

Goal (user, 2026-10-12): combine the HiCache-450 GB success with the decode KV-capacity report
(`decode.kvcache.report.md`) to raise prefill and decode capacity, raise AgentX CONC to 44 (must score ≥ the C40
result, 22,114/chip) then 48, and push throughput **above 23k tok/s/chip**. In parallel, research and then apply a
decode-side policy so **running requests always keep ≥ half of the decode KV pool** (likely by capping requests in
prealloc/transfer). Two machines in parallel: oai node 102 + crsuse2-m2m-137.

Task of record: `glm52.v0521-cap-c44c48-23k_20261012/mission.md` (re-injected every 10 min). Plan: `…/PLAN.md`.
Debug ledger `…/debug/working_process.md`, poll ledger `…/debug/poll.md`, research `…/research/`.

## Workspaces

- Docs: `glm52.v0521-cap-c44c48-23k_20261012/`.
- Runtime: `/shared_nfs/yihou_glm52/v0521cap_20261012/` (runs, logs, `src/image-9d755dd6/sglang` = sglang source
  extracted from the image for code reading, `images/` = docker-save of the image for 137).
- Previous phase (harness, runs, caches): `/shared_nfs/yihou_glm52/v0521dbg_20261010/` (harness `$WS/harness`).
- Baseline kit: `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/` (22,114/chip recipe, REPRODUCE.md).
- `/home/yihou/dev/git/infera.glm52.pd` is **READ-ONLY**.

## Key facts (first-hand)

- Baseline e2g A2 = 22,114/chip at C40 under the **warm-stack protocol** (score the 2nd run after `/flush_cache`;
  fresh-stack first runs are ~10 % slower, mechanism open).
- Image `infera-sglang:v0521-yihou-1010-mc4340` (102: sha256:9d755dd6…). Prefill: TP4 DP4, mem 0.85, HiCache 450 GB
  (`SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`), chunked 8192, max_running 64. Decode: mem 0.95 (2,762,176 tok/rank),
  max_running 64, cuda-graph bs 64, EAGLE 5/1/6, `num_reserved_decode_tokens 512`,
  `disaggregation_decode_allocation_policy 'early'`, `disaggregation_decode_host_receive_threshold 0.0`.
- Decode capacity levers (report): mem-fraction linear ≈ 54.7k tok per 0.01 (0.95 validated; 0.96/0.97 capacity
  only); decode HiCache is a no-op in PD; index-K elision blocked under PD.
- C40 is decode-bound (decode busy 93-96 %); admission never bound at C40 in R041 (prealloc>0 0 %, retracted 0).
- Nodes: 102 via spur shim (`SPUR_SSH_JOBS=crsuse2-m2m-102:207495`, shim dir
  `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin`), job ends 2026-10-13 05:26 UTC. 137 via plain ssh
  (MI355X ×8, 2.75 TB RAM, ionic ×8, IP 10.245.153.247; docker root /mnt/m2m_nobackup).

## Core principles

1. Measure, don't assume; rank evidence; suspend, don't conclude. Each experiment tests several hypotheses.
2. Work in containers; host actions are docker / ssh / spur shim only. serena/LSP for code.
3. Never delete a file whose path lacks `yihou`. Never disturb other users' containers/GPUs.
4. English for work, Chinese for reports.
5. Leader polls every 20 min; watchdog every 10 min while a service/bench runs. Ultracode: workflows.

## Other notable details

- RCCL same-node init race: retry decode only, never restart prefill.
- VRAM release after teardown 10-40 min: gate launches on all GPUs idle.
- Logs contain binary bytes: `strings <log> | grep`.
- Pass config deltas via the CONFIG file so `agentx_bench.sh` matches `launch.sh`.
- Never edit a running bash script in place; drivers detached with `setsid nohup … < /dev/null &`.
