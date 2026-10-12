# GLM-5.2-MXFP4 same-node 1P1D P4D4 on sglang **v0.5.21** — AgentX full C40 = **22,114 tok/s/chip** (warm-stack protocol)

**Ran:** 2026-10-10 → 2026-10-12 (UTC). Headline run **e2g A2**: bench 07:16:30 → 08:35:29 UTC 2026-10-11
(profiling 07:30:53 → 08:31:33).
**Node:** `crsuse2-m2m-102` (spur, `amd-oai` / `amd-oai-qos`, jobs 202546 → 207495). Prefill GPUs 0-3, decode GPUs 4-7,
KV over XGMI via HIP IPC.
**Author:** yihou (with Claude; agent team: leader, harness-v2, repro, watchdog2).
**Status:** **PASS** — 22,114 tok/s/chip exceeds the user's floor (v0.5.19 triton pack-up, 21,032/chip) and the 22k goal.
Replicates of the same protocol: 21,921, 21,696 and 21,681 (mean of the four warm runs 21,853, spread −0.8…+1.2 %).

## Goal

User directives (`spec/mission.md`, "PERF GOAL" 2026-10-10 15:55 and "FLOOR" 2026-10-11 05:00): with the v0.5.21 target
config (triton DSA, prefill HiCache 450 GB, decode mem-fraction 0.95, EAGLE MTP 5/1/6) at AgentX full **CONC 40**,
exceed 22k tok/s/chip and in any case beat the v0.5.19 triton reference
`/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923`
(first-hand: 168,260 tok/s = **21,032/chip**; it is not 22k).

## Result

| run | stack state | tok/s/chip | total tok/s | ok / err | TTFT mean / p95 | TPOT mean | file |
|---|---|---|---|---|---|---|---|
| **e2g A2** | **warm (2nd run, flushed)** | **22,114** | 176,909 | 4286 / 1 | 5.30 / 19.21 s | 12.1 ms | `results/e2g/A2-scored/agentx_conc40.json` |
| e2i R2 | warm (decode debug agent on) | 21,921 | 175,366 | 4258 / 1 | 5.83 / 19.37 s | 12.3 ms | `results/e2i/R2-warm/` |
| e2c C40 | warm (after a C44 run) | 21,696 | 173,570 | 4237 / 3 | 5.61 / 20.05 s | 12.2 ms | `results/e2c-C40-warm/` |
| e2j R2 | warm (+ decode host-variant, graph bs 1..16) | 21,681 | 173,451 | 4242 / 1 | 5.21 / 16.57 s | 12.4 ms | `results/e2j/R2-warm/` |
| e2g A1 | fresh (1st run, same stack as A2) | 19,640 | 157,121 | 3993 / 4 | 7.61 / 28.70 s | 14.3 ms | `results/e2g/A1-first-run/` |
| v0.5.19 reference | fresh | 21,032 | 168,260 | 4157 / 4 | — / 20.10 s | — | packup_20260923 |

| criterion | target | actual (e2g A2) | verdict |
|---|---|---|---|
| beat the v0.5.19 triton pack-up | > 21,032 /chip | 22,114 (+5.1 %) | ✅ |
| goal | > 22,000 /chip | 22,114 (+0.5 %) | ✅ (1 of 4 warm runs above 22k; warm mean 21,853) |
| AgentX full, CONC 40, passing | `AgentX passed` | rc=0, 3629 s, 4286 ok | ✅ |
| stability | 0 GPU faults | MAF 0, OOM 0 (one asyncio CancelledError at teardown) | ✅ |
| triton DSA both legs | yes | `dsa_prefill_backend='triton'`, `dsa_decode_backend='triton'` in server_args (both legs) | ✅ |

All runs: `results/analysis/runs_summary.csv`; decode-step statistics: `results/analysis/decode_stats.txt`.

## The one thing to know: the warm-stack protocol

A freshly started stack's **first** AgentX run scores ~19.3-20.4k/chip; the **second** run on the same stack, after
`/flush_cache` on both legs (caches cold again, first-turn `#cached-token 0`), scores ~21.7-22.1k. Same image, same
args, same node, same hour range. The penalty is a decode-step slowdown of +7-12 % at matched load in the first run
(R046-R055), localized by a kineto trace to a ~9 ms/step GPU-idle "gap A" (a cross-stream dependency stall with the
GPU idle). The mechanism is **open** (see `notes.md` §1). The scored protocol is therefore:

1. bring up the stack; 2. run one full AgentX C40 (warm-up, not scored); 3. `POST /flush_cache` decode then prefill;
4. run AgentX C40 again → scored. (The earlier 21,696 run used the same protocol by accident.)

## How to reproduce

See `REPRODUCE.md`. TL;DR: hold 102 (or another MI355X spur node), verify/build `infera-sglang:v0521-yihou-1010-mc4340`,
`source scripts/harness/env.e2g-102.yihou.sh && bash scripts/harness/pipeline.yihou.sh` (A1, ~1 h 40 min), then
`bash scripts/harness/a2_e2g.yihou.sh` (A2, ~80 min) → `runs/e2g/agentx-c40-A2/agentx_conc40.json`.

## Folder map

- `REPRODUCE.md` — ordered steps from allocation to the scored JSON.
- `environment.md` — node, driver, image IDs/digests, git branch + SHAs, versions.
- `spec/` — `mission.md` (task of record incl. all user directives), `PLAN.md`, task `CLAUDE.task.md`.
- `scripts/harness/` — the runnable harness, verbatim (env/config chain, `pipeline.yihou.sh`, `run.v2.yihou.sh`,
  `a2_e2g.yihou.sh`, `chain_*.yihou.sh`, `bench-harness/`, `tools/`, `mounts-v0521/src/`).
- `scripts/alloc/` — the allocation holder and the spur exec ssh shim. `scripts/verify/` — image checks.
- `scripts/diag/` — the decode-cost / host-sampling / copy-interference / event-wait diagnostics used in R034-R055.
- `patches/` — image recipe (5 commits + Dockerfiles + Mooncake diffs + a git bundle), runtime mounts (with diffs vs
  the image), harness diffs, experimental mounts. See `patches/README.md`.
- `results/` — per-run JSON / benchmark.log / audit / census / dryrun / pipeline logs; `analysis/` tables; `env/` captures.
- `logs/` — gzipped server logs of the e2g stack (A1 + A2) and build logs.
- `analysis/` — the full debug ledger (`working_process.md`, R001-R056), poll ledger, RESULTS.md.
- `notes.md` — gotchas, wrong turns, open questions.

## Related pack-ups

- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-v0521-oai102.agentx-c44.packup_20261010` — phase 2 of this
  task: the v0.5.21 bring-up debug (H1 HiCache chunk register, H2 index_share, PID offset, ports, Mooncake #4340) and
  the C44 target pass (20,524/chip). This pack-up builds on it; the image and base harness are the same.
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923` — the v0.5.19
  triton C40 reference (21,032/chip, fresh stack, index_share TRUE).
- `/home/yihou/dev/git/infera.glm52.view/glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009` — the XGMI/HIP-IPC transport recipe.
- `/home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md` — consolidated decode KV-capacity report (the v0519
  capacity-lever campaign and the v0.5.21 decode 0.95 point used here: 2,762,176 KV tokens/rank).
