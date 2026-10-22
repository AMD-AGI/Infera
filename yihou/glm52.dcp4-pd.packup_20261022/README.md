# GLM-5.2-MXFP4 + DCP + SGLang on MI355X, PD-disaggregated (Infera) and mixed — best PD = **18,321 tok/s/chip** (target 20k NOT met)

**Ran:** 2026-10-21 12:38Z .. 2026-10-22 (pack-up at ~02:30Z 10-22; r5b still running). **Nodes:** spur MI355X
`crsuse2-m2m-054` (PD r1-r4), `-v3-045` (mixed), `-v3-060` (microbenchmarks, PD r5/r5b), `-v3-070` (PD E2) (`environment.md`).
**Author:** yihou (with Claude; agent team main, pdrun, mixrun, benchrun, analyst, coder2, watchdog).
**Status:** **target NOT met**; r5b config: A1 19,400 (not scored), scored A2 pending (r6b on OAI 105, see `ANALYSIS.md` §1).

- **Handoff: how to run the current best DCP PD config → `RUN_BEST_DCP.md`.**
- **Handoff update (r5b results, deltas vs r4, merge with infera-glm52-pd-5f optimizations) → `HANDOFF_r5b.md`.**
- **Explanations (run status, weight memory, TPOT vs step, DCP vs DPA output, transfer, TOPK_V2, infra/guard, plan) → `ANALYSIS.md`.**

## What and why

Task of record `spec/mission.md`: run GLM-5.2 with decode context parallelism (DCP) on AMD MI355X under SGLang, first
mixed and then PD-disaggregated through Infera (prefill without DCP, decode with DCP), and reach 20k+ tok/s/chip on
AgentX C40. Upstream sglang has no PD + DSA DCP path, so this kit builds one on top of PR #44322 (triton sparse-MLA DCP
for ROCm) — 31 commits on branch `dev/yihou/glm52_dcp_pd` (`patches/`).

## Headline

**Target 20k tok/s/chip: NOT met.** Best DCP PD run is **pdBeperf-r4 A2 = 18,321 tok/s/chip** (146,569 tok/s total on
8 GPUs; TPOT mean 19.7 ms; TTFT mean 3.93 s; 3828 ok / 2 err) — 0.916x the target and 0.828x the non-DCP PD baseline
(22,114 tok/s/chip, TPOT 12.1 ms) (`results/RESULT.md` §0).

| Run | Layout | Scoring | tok/s/chip | total tok/s | TPOT mean ms | TTFT mean s | ok / err | Source |
|---|---|---|---|---|---|---|---|---|
| **PD r4 A2 (best DCP PD)** | P: TP4+DPA4, no DCP; D: TP4/EP4 + DCP4, EAGLE 5/1/6, graph, custom AR on | /8 | **18,321** | 146,569 | 19.7 | 3.93 | 3828 / 2 | `results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json` |
| PD r4 A1b (fresh stack) | same | /8 | 17,473 | 139,786 | 21.9 | 3.21 | 3710 / 2 | `results/pd/pdBeperf-r4/agentx-c40-A1b/agentx_conc40.json` |
| PD r5b (kv_b bf16 + split triton indexer, DEBUG_AGENT=0, v3-060) | same + knobs | /8 | **pending** | | | | | `results/pd/pdBeperf-r5b/` (setup files only) |
| Non-DCP PD baseline e2g A2 | P: TP4+DPA4; D: TP4+DPA4 | /8 | 22,114 | 176,909 | 12.1 | 5.30 | 4286 / 1 | `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/results/analysis/runs_summary.csv` row e2g-A2 (`results/RESULT.md` §0) |
| Mixed DCP4 c40 (one 4-GPU server) | TP4/EP4 + DCP4 | **/4** | 26,277 | 105,108 | 36.9 | 1.30 | 3025 / 1 | `results/mixed/full1-dcp4-c40/agentx_conc40.json` |

- The mixed number is total / **4** GPUs (one server, InferenceX non-multinode scoring); the PD numbers are total / **8**
  GPUs. **The per-chip columns are not comparable**; compare totals (105,108 on 4 GPUs vs 146,569 on 8 GPUs).
- All perf runs use simulated acceptance (`SGLANG_SIMULATE_ACC_LEN=3.61`); they measure speed, not correctness.

## Configuration of the best PD run (pdBeperf-r4 A2)

Node `crsuse2-m2m-054` (amdgpu 6.14.14, SDMA on), image `infera-sglang:v0522-1020-yihou-dcp-mc4340`
(`sha256:5b6565ef...7376`), sglang overlay `ae529c96bf`. Sources: rendered commands
`scripts/harness/dryrun.pdBeperf-r4-054.yihou.txt` (D = dryrun), live server-arg audit `results/pd/pdBeperf-r4/audit.txt`
(A), client env `results/pd/pdBeperf-r4/agentx-c40-A2/runtime.env` (R), env file `scripts/harness/env.pdBeperf-r4-054.yihou.sh`
+ `PD_EXTRA` in `scripts/harness/pd_r4.yihou.sh` (E). The r5b column is the current main line (same layout), from
`scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt` and `results/pd/pdBeperf-r5b/audit.txt`.

| Item | Prefill leg (GPUs 0-3) | Decode leg (GPUs 4-7) | r5b difference | Source |
|---|---|---|---|---|
| Parallelism | TP4, DP4 + DP attention, EP1, no DCP | TP4, EP4, DP1, no DP attention, `--dcp-size 4` | none | D, A |
| DSA backends | `--dsa-prefill-backend triton --dsa-decode-backend triton` | same | none | D, A |
| `DSA_TOPK_BACKEND` | empty (flag dropped) | empty | none | R; `config.yihou.v0521-base.sh` l.226-229 |
| KV cache dtype | fp8_e4m3 | fp8_e4m3 | none | A |
| mem-fraction-static | 0.85 | 0.85 | none | A |
| max-running-requests / chunked prefill | 64 / 32768 | 64 / 32768 | none | D |
| Speculative (EAGLE MTP) | none | EAGLE, steps 5, topk 1, draft tokens 6 | none | A |
| CUDA graph | `--cuda-graph-max-bs-prefill 64` | on, `--cuda-graph-max-bs-decode 64` | none | D |
| Custom all-reduce | on (`disable_custom_all_reduce=False`) | **on** (`DECODE_CUSTOM_AR=1`); the audit's one FAIL (`want True`) is a stale expectation | none | A, E |
| aiter all-reduce fusion | on | on | none | A |
| HiCache | on, 450 GB/rank, ratio 1.5, write_through, io kernel, layout page_first | off (`hicache_size=0`) | none | A |
| Decode radix cache | n/a | off (`disable_radix_cache: True`) | none | `notes.md` §13 |
| index_share_for_mtp_iteration | false (`--json-model-override-args`) | false | none | A |
| KV transfer | Mooncake, same node; `MC_TE_FILTERS=yihou_no_such_rdma_device`, `MC_DISABLE_HIP_TRANSPORT=1`, `MOONCAKE_DISABLE_HIP_DMABUF=0`, `MC_LOG_LEVEL=INFO`; Mooncake metrics off | same | none | D; `results/RESULT.md` §2.2 |
| Router | Infera rust router, policy kv-aware, overlap weights prefill 20.0 / decode 2.0, `INFERA_PD_DP_RANK_AFFINITY=false` | | none | D |
| `SGLANG_DSA_DCP_*` flags | none set | **none set** | decode: `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`, `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1`, `SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1` (+ `SGLANG_LOG_DECODE_SEQ_LENS=1`, diagnostic) | D |
| `SGLANG_DSA_DCP_PD_DROP_MTP_SEED` | not set | not set | not set | D |
| `SGLANG_OPT_USE_TOPK_V2` | false | false (not read on decode under the split knobs, `notes.md` §12) | none | A |
| `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK` | 0 | 0 | none | D |
| ROCm debug agent (`DEBUG_AGENT`) | **1** (`HSA_TOOLS_LIB=librocm-debug-agent.so.2`, `HSA_ENABLE_DEBUG=1`) | **1** | **0** on both legs | D, A |
| `HSA_ENABLE_SDMA` | not set (SDMA on) | not set | `0` on both legs (v3 node) | D |
| Other env | `HSA_NO_SCRATCH_RECLAIM=0`, `SGLANG_DP_USE_GATHERV=1`, `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`, `SGLANG_USE_AITER=1` | `HSA_NO_SCRATCH_RECLAIM=1`, `SGLANG_DP_USE_GATHERV=0`, `SGLANG_USE_AITER=1`; PID offset 30000 | none | D |
| NCCL debug | off | off (r2 had `NCCL_DEBUG=INFO`) | none | D; `results/RESULT.md` §2 |
| Simulated acceptance | n/a | `SGLANG_SIMULATE_ACC_LEN=3.61`, `match-expected`, `real-draft-token` | none | D, A |
| sglang code | overlay `src/sglang-ae529c96bf` | same | `src/sglang-93034eba91` | D |
| Client | AgentX C40, 3600 s, warm-up 10/lane, seed 42, InferenceX `918524ff` (clean), aiperf 0.12.0 | | | R; `benchmark_command.txt`; `environment.md` |

## Scope limits (read before quoting a number)

1. **Simulated acceptance waives correctness.** Every perf score above ran with `SGLANG_SIMULATE_ACC_LEN=3.61` on the
   decode. Correctness evidence is separate and sim-off only: r4 code `pdBe-r1` (eager, 5-request T4a PASS) and r5 code
   `pdBe-r5g` (graph, T4a PASS). A **40-concurrent sim-off check of the DCP4 PD decode in graph mode is pending**
   (`results/RESULT.md` §6).
2. **v3 vs 054 are not comparable.** v3 nodes (amdgpu 6.16.13) need `HSA_ENABLE_SDMA=0` and ran r5/r5b with
   `DEBUG_AGENT=0`; 054 (6.14.14) ran SDMA on and `DEBUG_AGENT=1`. r5b vs r4 therefore changes node, driver, SDMA,
   debug agent and three decode knobs at once (`environment.md`, `notes.md` §1, §6).
3. **E2 (PD with a DP2 x DCP2 decode) garbles answers in graph mode** — 3/200, 13/240, 27/320 bad across rounds
   (eager 0/40). Recorded as a known risk by user decision; not bisected (`results/RESULT.md` §6, `notes.md` §11).
4. **r5b is pending** (A1 running at 02:29Z 10-22; A1/A2 ETA ~03:15Z / ~04:40Z per `WS/poll.md` 01:46Z; `WS` = the local workspace, `environment.md`). The E2 perf
   round pdE2-p1 was stopped without a score.
5. Mixed (/4) and PD (/8) per-chip numbers are different conventions (`notes.md` §14).
6. The transfer-bound reading of the C40 -> C52 queue depths is **not confirmed** (no transfer-time measurement;
   `results/RESULT.md` §2.2).
7. The AgentX JSON field `decode_dcp_size` reads 1 even for DCP4 runs; DCP4 is evidenced by the server args.

## Navigation

| Path | What |
|---|---|
| `REPRODUCE.md` | Ordered steps: image, node prep, sglang overlay from patches, PD r4 / r5b (gate -> A1 -> flush -> A2), mixed DCP4 c40, decode microbenchmarks, scoring |
| `results/RESULT.md` | All numbers with sources: PD rounds, decode step time, queue depths, mixed runs, microbenchmark A/Bs, E2 correctness |
| `results/pd/<round>/` | AgentX JSONs, runtime.env, audit / census / decode_stats / pdcheck per PD round |
| `results/mixed/`, `results/decodebench/`, `results/mixed_checks/`, `results/pdq/` | Mixed AgentX, microbenchmark and check outputs, PD queue-depth readouts |
| `environment.md` | Nodes, drivers, image IDs, sglang shas per run, client versions, external paths, secrets (none besides spur ssh) |
| `scripts/` (`scripts/README.md`) | Verbatim PD harness (`harness/`, `harness/pd-tools/`), mixed / microbench scripts (`mixed/`), microbench pair drivers + `wavesteps.py` (`bench/`), analysis tools (`tools/`) |
| `patches/` (`patches/README.md`) | sglang branch as 0000 merge diff + 31 format-patches + full diff, with per-commit what/why/verification |
| `spec/` | `mission.md` (task + user directives), `answers/dcp_q1_q2.md`, `design/*.md`, `research/*.md` |
| `working_process.md` | Timeline, phases 0-8, every round including failed ones |
| `notes.md` | Gotchas (SDMA, debug agent, spur exec, pd_seq bug, decode radix off, ...) and 16 open questions with the measurement that answers each |

## Related pack-ups

- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012` — the non-DCP PD
  baseline (22,114 tok/s/chip, e2g A2) whose prefill leg and AgentX protocol this kit reuses.
