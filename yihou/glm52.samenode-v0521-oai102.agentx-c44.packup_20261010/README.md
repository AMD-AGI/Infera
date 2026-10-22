# GLM-5.2-MXFP4 same-node 1P1D P4D4 on sglang **v0.5.21** — AgentX full CONC 44, target config PASS

**Ran:** 2026-10-09 (phase 1, rounds c44a-c44e, all failed) → 2026-10-10 (phase 2, debug + passes;
target pass e2c at 13:17:30 UTC).
**Node:** `crsuse2-m2m-102` (spur job 202546, `amd-oai` / `amd-oai-qos`). Prefill GPUs 0-3, decode GPUs 4-7.
**Author:** yihou (with Claude; agent team: leader, harness-v2, repro, sanity, watchdog2).
**Status:** **PASS.** The mission target config passed AgentX full mode at CONC 44 on v0.5.21. Two
relaxed configs on the same chain also passed.

## Goal

The user's spec (`spec/mission.md`): debug until GLM-5.2-MXFP4 same-node 1P1D P4D4 runs on an infera
image built on `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`. First relax the config to get
a pass, then step back to the target one group at a time, localizing the triggers.

**Target:** triton DSA, prefill HiCache 450 GB, decode mem-fraction 0.95, EAGLE MTP, KV over
XGMI/HIP IPC (no TCP, no NIC), AgentX full (3600 s, warmup 10/lane) at CONC=44, passing.

## Result

| round | config | CONC | verdict | ok / err | total tok/s | TTFT p50 / p95 | ITL p50 |
|---|---|---|---|---|---|---|---|
| e1c | relaxed: hicache ratio 1.5 (~154 GB), decode 0.85, fused indexer OFF | 40 | ✅ PASS | 4174 / 2 | 168,389 | 2.85 / 17.6 s | 13.2 ms |
| e2b4 | target minus fused indexer | 44 | ✅ PASS | 4138 / 1 | 162,774 | 5.41 / 30.5 s | 14.0 ms |
| **e2c** | **target** (fused DSA indexer ON = v0521 default) | **44** | ✅ **PASS** | **4179 / 2** | **164,194** (20,524/GPU) | **4.63 / 28.65 s** | **14.7 ms** |

| target criterion | e2c |
|---|---|
| triton DSA both legs | ✅ `--dsa-{prefill,decode}-backend triton` (audit PASS) |
| prefill `--hicache-size 450` | ✅ 4 × 450.00 GB host pool, filled to ~100 % |
| decode mem-fraction 0.95 | ✅ 2,762,176 KV tokens/rank |
| EAGLE MTP | ✅ 5/1/6 (simulated acceptance 3.61, as the 0923 AgentX reference) |
| KV over XGMI / HIP IPC | ✅ 4× `Found 0 HCAs`, HIP transport ×4 per leg, 0 transfer failures |
| AgentX full C44 passing | ✅ `AgentX passed`, 3630 s, 4179 ok, 2 error-dropped |
| stability | ✅ Memory access fault 0, Traceback 0 on both legs over the whole run |

Details: `results/SUMMARY.md`. All numbers are single runs (no replicates).

## What it took (root causes found and fixed)

Details are in `notes.md` and `analysis/working_process.md`.
1. **H1:** a HiCache host pool larger than 256 GB is registered in chunks, but the kernels use one base
   pointer, so prefill faults → `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`.
2. **H2:** `index_share_for_mtp_iteration=true` → decode MAF → set to false.
3. **R012/R020/R022:** same-node IPC PID collisions (the old "RCCL race" and a new 201 from CLR's same-PID
   guard). `--pid host` is denied on spur nodes, so the decode container's PIDs are offset by 30000.
4. **R015:** fixed ports moved below the ephemeral range (< 30000).
5. **R018:** Mooncake HIP IPC interior-registration misplacement → image with Mooncake PR #4340.
6. **v0.5.21-only:** a blocking kv-events PUSH (R001) and a bad absorb-bmm config (R003), fixed with mounts.

## How to reproduce

See `REPRODUCE.md`. TL;DR: hold 102 (or another MI355X spur node). Build or verify the mc4340 image.
Then run `source scripts/harness/env.e2c-102.yihou.sh && bash scripts/harness/pipeline.yihou.sh`.
Expect `AgentX passed` about 1 h 50 min later.

## Folder map

- `REPRODUCE.md`: ordered steps from allocation to the result JSON.
- `environment.md`: node, driver, image IDs and digests, sglang/aiter/mooncake versions.
- `spec/`: `mission.md` (task of record), `PLAN.md`, the task CLAUDE.md.
- `scripts/`:
  - `harness/`: the runnable harness, verbatim. Includes the env files for e1c/e2b4/e2c, configs, the
    pipeline, and `bench-harness/`.
  - `diag/`: no-model repro tests (PID guard, PID offset, te_pair, H1 chunk, debug agent, PR #42553).
  - `shimbin/ssh`: the spur exec shim.
  - `hold_inf.yihou.sh`: the allocation holder.
  - `verify_v0521.sh`, `verify_mc4340.sh`: image checks.
- `patches/`: image recipe commits, runtime mounts with diffs, harness diffs, and the optional
  PR #42553. See `patches/README.md`.
- `results/`: per-round `agentx_conc*.json`, census, audit, dryrun, pipeline, benchmark.log; env
  captures; side analyses; small diag outputs.
- `logs/`:
  - `e2c/`: the full e2c server, engine and router logs, gzipped.
  - `excerpts/`: key lines for e1c/e2b4/e2c and for the failing rounds (c44c/c44d MAF, e1a 201,
    e2b3 pid-host denied).
  - `build/`: the mc4340 image build log.
- `analysis/`: the debug ledger (R001-R027) and the 20-min poll ledger.
- `notes.md`: gotchas, wrong turns, open items.

## Related pack-ups

- `../glm52.samenode-triton-idxshare.c40-full.packup_20260923`: the v0.5.19 AgentX C40 reference.
- `../glm52.samenode-p4d4.agentx-c40.packup_20260922`: the v0.5.19 same-node origin (ENGINE_PORT_STRIDE, start gate).
- `/home/yihou/dev/git/infera.glm52.view/glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009`: the v0.5.19 oai
  same-node recipe (the XGMI transport recipe).
