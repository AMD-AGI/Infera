# RESULT — GLM-5.2-MXFP4 + DCP + SGLang on MI355X, PD-disaggregated (Infera) and mixed, 2026-10-21..22

Every number cites its source. Paths in this file:
- `results/...` = copies inside this pack-up (verbatim, copied from the cluster; each file < 4 MB).
- `PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd`, `MIX=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun`,
  `BENCH=/shared_nfs/yihou_glm52/dcp_pd_20261021/bench` = cluster originals.
- `WS=/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021` = local workspace (ledgers, design notes).
- `BASE=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012` = 22k non-DCP pack-up.

"pending" = not measured yet. "est." = estimate. Mechanism statements appear only where a ledger records a code read or an experiment.

## 0. Headline

**Target 20k tok/s/chip was NOT met.** Best PD result: r4 A2, 18,321 tok/s/chip (TPOT mean 19.7 ms, TTFT mean 3.93 s).
The non-DCP PD baseline scores 22,114 tok/s/chip (TPOT mean 12.1 ms).

| Run | Layout | Scoring | tok/s/chip | total tok/s | TPOT mean / p95 ms | TTFT mean / p50 / p95 s | ok / err | Source |
|---|---|---|---|---|---|---|---|---|
| **PD r4 A2 (best DCP PD)** | P: TP4+DPA4 (no DCP), D: TP4/EP4 + DCP4, EAGLE 5/1/6, graph, custom AR on | /8 | **18,321** | 146,569 | 19.7 / 24.4 | 3.93 / 1.61 / 15.90 | 3828 / 2 | `results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json` |
| PD r4 C52 (same stack) | same, client C52 | /8 | 18,070 | 144,556 | 21.7 / 27.7 | 14.44 / 5.95 / 56.62 | 3984 / 3 | `results/pd/pdBeperf-r4/agentx-c52-C52/agentx_conc52.json` |
| 22k baseline e2g A2 (no DCP) | P: TP4+DPA4, D: TP4+DPA4 | /8 | 22,114 | 176,909 | 12.1 / 17.9 | 5.30 / 3.33 / 19.21 | 4286 / 1 | `BASE/results/analysis/runs_summary.csv` row e2g-A2 |
| Mixed DCP4 c40 (one 4-GPU server) | TP4/EP4 + DCP4 | /4 | 26,277 | 105,108 | 36.9 / 67.7 | 1.30 / 0.47 / 4.71 | 3025 / 1 | `results/mixed/full1-dcp4-c40/agentx_conc40.json` |

Gap of best DCP PD to the baseline: 18,321 / 22,114 = 0.828 (−17.2 %). Gap to target: 18,321 / 20,000 = 0.916.

## 1. Scoring conventions

- PD: total tok/s (input + output) / 8 GPUs (4 prefill + 4 decode), the `per_gpu.total_tput_tps` field of the AgentX
  JSON (`num_prefill_gpu` 4 + `num_decode_gpu` 4). Same convention as the 22k baseline.
- Mixed: total tok/s / 4 GPUs (one 4-GPU server; InferenceX non-multinode convention). **Not comparable to /8 PD numbers**
  without re-scaling; compare totals instead (mixed DCP4 105.1K on 4 GPUs vs PD r4 A2 146.6K on 8 GPUs).
- Protocol: AgentX C40, 3600 s profiling window, 444 warm-up requests. "A1"/"A1b" = first run on a fresh stack;
  "A2"/"A2b" = scored warm run after `/flush_cache` on both legs (same as the 22k kit's A1/A2).
- All perf runs use `SGLANG_SIMULATE_ACC_LEN=3.61` on the decode. Simulated acceptance waives correctness; correctness
  comes only from the separate sim-off gates in §5.
- JSON metadata note (observation): the AgentX JSONs record `decode_dcp_size: 1` for the DCP4 decode runs
  (e.g. `results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json`). The DCP4 setting is evidenced by the server
  args (`results/pd/pdBeperf-r4/audit.txt`, `census.txt`) and `WS/ledger.pdrun.md` P015, not by that JSON field.

## 2. PD runs (node crsuse2-m2m-054, image `infera-sglang:v0522-1020-yihou-dcp-mc4340`, Mooncake same node)

Setup (from `WS/RESULTS.md` §1, `WS/ledger.pdrun.md` P010-P015): prefill = the 22k e2g leg (TP4 + DPA4, no DCP,
no EAGLE, HiCache 450), decode = TP4/EP4 + DCP4, EAGLE 5/1/6, CUDA graph (max bs 64), max-running 64,
index_share false, DEBUG_AGENT=1 on both legs (per `WS/design/pd_vs_mixed_decode_diff.md` §1 #2).

| Run | Change vs previous | Bench | tok/s/chip | TPOT mean / p95 ms | TTFT mean / p50 / p95 s | GPU cache hit | ok / err | Source |
|---|---|---|---|---|---|---|---|---|
| pdBe-r1 | eager, sim off, real draft | correctness gate | — | — | — | — | math 29 ×2, passkey 8K/34K/96K PASS | `results/pd/pdBe-r1/t4/`, `WS/ledger.pdrun.md` P010 |
| pdBeperf-r1 | graph + sim 3.61 | C8 600 s (not comparable) | 3,989 | 9.8 / 14.7 | 1.18 / 0.61 / 4.25 | — | 123 / 1 | `results/pd/pdBeperf-r1/agentx-c8-short/agentx_conc8.json` |
| pdBeperf-r2 | HiCache 450, C40, custom AR off, NCCL_DEBUG=INFO | A1 fresh | 16,670 | 23.7 / 30.0 | 3.24 / 1.50 / 12.83 | 96.3 % | 3593 / 1 | `results/pd/pdBeperf-r2/agentx-c40-A1/agentx_conc40.json` |
| pdBeperf-r2 | same | A2b warm (scored) | 17,053 | 21.5 / 26.4 | 4.92 / 1.68 / 23.07 | 94.6 % | 3642 / 1 | `results/pd/pdBeperf-r2/agentx-c40-A2b/agentx_conc40.json` |
| pdBeperf-r2 | same, profiler on (600 s) | prof (not a score) | 12,654 | 15.6 / 22.8 | 6.07 / 2.77 / 30.31 | 85.3 % | 620 / 0 | `results/pd/pdBeperf-r2/agentx-c40-prof/agentx_conc40.json` |
| r3 | decode-only restart, custom AR on, NCCL off | C40 | **aborted** by client (empty-content errors 15/142 profiling, 132 warm-up); partial JSON not a result | — | — | — | 127 / 147 | `results/pd/pdBeperf-r2/agentx-c40-r3/agentx_conc40.json`, `WS/ledger.pdrun.md` P014 |
| pdBeperf-r4 | fresh full stack, custom AR on, NCCL off | A1b fresh | 17,473 | 21.9 / 28.8 | 3.21 / 1.45 / 11.79 | 96.4 % | 3710 / 2 | `results/pd/pdBeperf-r4/agentx-c40-A1b/agentx_conc40.json` |
| pdBeperf-r4 | same | **A2 warm (scored)** | **18,321** | 19.7 / 24.4 | 3.93 / 1.61 / 15.90 | 94.7 % | 3828 / 2 | `results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json` |
| pdBeperf-r4 | same stack, client C52 | C52 warm | 18,070 | 21.7 / 27.7 | 14.44 / 5.95 / 56.62 | 83.1 % | 3984 / 3 | `results/pd/pdBeperf-r4/agentx-c52-C52/agentx_conc52.json` |
| pdBeperf-r5 (v3-060) | r4 + KEEP_KV_B_BF16 + split triton indexer + SDMA=0, code 93034eba91 | A1 | **crashed** in warm-up (prefill ROCm debug-agent abort) | — | — | — | — | `WS/ledger.pdrun.md` P016 |
| pdBeperf-r5b (v3-060) | r5 + DEBUG_AGENT=0 | A1 / A2 | **pending** (client running from 01:55Z; A1 ETA ~03:15Z, A2 ~04:40Z per `WS/poll.md` 01:46Z) | | | | | `results/pd/pdBeperf-r5b/` (setup files only) |

r2 → r4 (custom AR off → on, NCCL debug INFO → off; one run each): A1 16,670 → 17,473; A2 17,053 → 18,321.

### 2.1 PD decode step time (decode TP0 log, profiling window; `tools/decode_stats.dcp.yihou.py`)

| Run | ms/step | mean running | gen tok/s | KV usage | transfer-in > 0 (% lines) | prefill new / cached tokens | Source |
|---|---|---|---|---|---|---|---|
| r2 A1 | 75.3 | 21.38 | 994.9 | 0.432 | 67.3 | 16.4M / 464.5M | `results/pd/pdBeperf-r2/decode_stats.r2.txt` |
| r2 A2b | 66.5 | 19.37 | 1001.5 | 0.441 | 72.0 | 16.6M / 477.5M | same |
| r4 A1b | 69.4 | 20.49 | 1031.3 | 0.467 | 68.3 | 16.3M / 489.0M | `results/pd/pdBeperf-r4/decode_stats.r4.txt` |
| r4 A2 | 61.2 | 18.75 | 1057.7 | 0.455 | 71.3 | 17.9M / 510.8M | same |
| r4 C52 | 59.8 | 18.31 | 1024.4 | 0.671 | 89.9 | 30.3M / 492.9M | same |
| 22k e2g A2 (per DP, DPA4) | 40.9 | 3.57 per DP × 4 | 320.3 per DP | 0.323 | 54.9 | 25.6M / 612.9M | `BASE/results/analysis/decode_stats.txt` |

### 2.2 PD queue depths (skill llm-pd-bottleneck-finder; decode = TP0, prefill = all 4 DP ranks)

| state (mean / p95 / max) | r2 A2b C40 | r4 A2 C40 | r4 C52 | Source |
|---|---|---|---|---|
| prefill #queue-req (per rank) | 0.3 / 2 / 10 | 0.3 / 2 / 6 | 1.2 / 6 / 18 | `results/pdq/prefill.allranks.txt` |
| prefill #inflight-req (per rank) | 1.7 / 5 / 14 | 1.8 / 6 / 12 | 3.6 / 12 / 16 | same |
| decode #prealloc-req | 0 / 0 / 0 | 0 / 0 / 0 | 0.0 / 0 / 6 | `results/pdq/decode.stats.txt` |
| decode #transfer-req | 6.0 / 24 / 36 | 4.9 / 21 / 33 | 18.2 / 45 / 51 | same |
| decode #running-req (cap 64) | 19.4 / 33 / 46 | 18.9 / 31 / 44 | 18.5 / 40 / 52 | same |
| decode retracted / queue | 0 / 0 | 0 / 0 | 0 / 0 | same |

- Mooncake metrics were off (MC_TE_METRIC off); no transfer-error signatures in the windows (`results/pdq/*.stats.txt`).
- C40 → C52 on the same stack: decode running mean 18.9 → 18.5, decode transfer-in 4.9 → 18.2, prefill in-flight
  1.8 → 3.6 per rank. Per the skill's rule this is the transfer-bound pattern; **not confirmed** by a transfer-time
  measurement (`WS/ledger.pdrun.md` P017).

## 3. Mixed (non-PD) runs (node crsuse2-m2m-v3-045, `HSA_ENABLE_SDMA=0`, sim 3.61, one 4-GPU server, /4)

| Config | Conc | tok/s/GPU | TTFT mean / p50 / p90 s | TPOT mean / p90 ms | GPU cache hit | ok / err | Source |
|---|---|---|---|---|---|---|---|
| DCP4 TP4/EP4 (ae529c96bf) | c40 | **26,277** | 1.30 / 0.47 / 2.16 | 36.9 / 48.3 | 93.3 % | 3025 / 1 | `results/mixed/full1-dcp4-c40/agentx_conc40.json` |
| ref TP4/EP4, no DCP | c40 | 5,130 | 136.55 / 152.29 / 285.68 | 81.7 / 182.4 | 26.8 % | 678 / 3 | `results/mixed/full1-ref-c40/agentx_conc40.json` |
| DCP4 + KEEP_KV_B_BF16 (5aacce1f1c, fresh stack) | c40 | 26,152 | 1.56 / 0.46 / 2.45 | 37.0 / 47.7 | 94.5 % | 3010 / 2 | `results/mixed/full-kvb-dcp4-c40/agentx_conc40.json` |
| DP2 + DCP2 (5aacce1f1c) | c40 | no score: warm-up 221/444 after 2,404 s, stopped | | | prefill cached 64 % | | `WS/ledger.mixrun.md` (22:40-23:23Z); setup files `results/mixed/full-kvb-dp2-c40/` |
| DCP4, max-running 96 | c48 | 13,012 | 6.32 / 0.66 / 21.90 | 136.9 / 318.3 | 79.1 % | 1726 / 3 | `results/mixed/full1-dcp4-c48/agentx_conc48.json` |
| #44322 author (other node/image) | c40 / c48 | 27,087 / 13,252 | | | | | PR #44322 body (second-hand) |

Smoke runs (300-600 s window, warm-up 1/lane, `--unsafe-override`: **not valid scores**, `WS/ledger.mixrun.md` lines 144-212):

| Smoke | tok/s/GPU | TTFT mean / p50 / p90 s | TPOT mean / p90 ms | ok / err | Source |
|---|---|---|---|---|---|
| smoke-dcp4-c40 (300 s) | 19,213 | 0.54 / 0.36 / 0.84 | 17.9 / 23.7 | 272 / 0 | `results/mixed/smoke-dcp4-c40/agentx_conc40.json` |
| smoke-ref-c40 (300 s) | 16,496 | 1.77 / 0.32 / 4.70 | 28.6 / 44.6 | 253 / 2 | `results/mixed/smoke-ref-c40/agentx_conc40.json` |
| smoke600 DCP4 custom AR on | 17,102 | 4.31 / 0.43 / 3.72 | 71.5 / 47.5 | 417 / 0 | `results/mixed/smoke600-arOn-c40/agentx_conc40.json` |
| smoke600 DCP4 custom AR off | 20,948 | 1.48 / 0.47 / 2.31 | 44.2 / 49.6 | 483 / 1 | `results/mixed/smoke600-arOff-c40/agentx_conc40.json` |

KV capacity (max_total_num_tokens; `WS/RESULTS.md` §2, mixrun logs): DCP4 8,894,208; no DCP 2,860,224 (3.1×);
DCP4 with custom AR off 9,764,352.

Mixed correctness (`WS/RESULTS.md` §2, `WS/ledger.mixrun.md` R2): DCP4 and ref pass check.py (math 29 ×2, passkey
8K/34K/96K) eager and graph; 40-concurrent 40/40; partial eager GSM8K (sgl-eval 0.1.3, GLM-5.2-FP8 preset, stopped
early) ref 194/197 = 0.9848, DCP4 193/196 = 0.9847, same wrong set.

## 4. Decode step time, PD vs mixed (`WS/design/pd_vs_mixed_decode_diff.md` §2)

Step = 1000 × running × accept / gen_tput per TP0 decode log line, binned by running count.

| running | PD r4 A2 median ms (n) | mixed DCP4 c40 "clean" lines (no prefill since previous decode line) median ms (n) | mixed all lines median ms (n) |
|---|---|---|---|
| 16-19 | 61.4 (262) | 58.6 (10) | 70.1 (69) |
| 20-23 | 68.5 (301) | 63.2 (8) | 79.1 (88) |
| 24-27 | 74.1 (230) | 69.8 (12) | 88.0 (168) |
| 28-31 | 78.9 (126) | 77.5 (6) | 96.3 (174) |

- PD decode-only steps are +1.4..+5.3 ms over mixed decode-only steps at matched running; the mixed clean bins are small
  (n 6-12) and the context distributions are not matched.
- PD r4 A2, transfers in flight vs none at matched running: 16-19: 60.8 (n 190) vs 62.1 (n 72); 20-23: 68.2 (n 206) vs
  68.9 (n 95).
- Server-arg / env differences PD r4 decode vs mixed (same doc §1): `SGLANG_OPT_USE_TOPK_V2` false (PD) vs true (mixed);
  ROCm debug agent on (PD) vs off (mixed, 22k); decode radix off (PD); index_share false (PD) vs true; max-running 64 vs 80.
  None A/B-measured in PD. Later code read (`WS/poll.md` 02:27Z): with the split-indexer knobs (r5/r5b) TOPK_V2 is not
  read on the decode.

## 5. Decode microbenchmarks (mixed server, cached prompts, waves of N, sim 3.61, graph; median ms/step at running = N)

Raw files: `results/decodebench/` (copy of `BENCH/results`) and `results/mixed_checks/` (copy of `MIX/results`).

| A/B | Context | N=8 | N=16 | N=24 | N=32 | Δ (B − A) | Source |
|---|---|---|---|---|---|---|---|
| custom AR ON vs OFF | 34K | 33.0 vs 37.0 | 43.4 vs 47.1 | 52.7 vs 56.8 | 58.9 vs 62.7 | −3.5..−3.9 ms | `WS/ledger.mixrun.md` "Custom-AR decode microbench" |
| KEEP_KV_B_BF16 vs off | 34K | 32.9 vs 33.35 | 42.2 vs 43.8 | 51.35 vs 53.1 | 57.6 vs 59.3 | −0.4 / −1.6 / −1.7 / −1.7 ms | `WS/ledger.benchrun.md` T10 (step medians, from server logs); `results/decodebench/decodebench.kvb*.txt` = wave timings only |
| index_share true vs false (sim off) | 34K | 32.8 vs 33.05 | 43.3 vs 43.35 | 53.05 vs 53.35 | 59.35 vs 59.5 | ≈ 0 (≤ 0.6 ms) | `WS/ledger.benchrun.md` T11 (step medians, from server logs); `results/decodebench/decodebench.is{t,f}*.txt` = wave timings only |
| split indexer (torch) vs ref | 34K | 33.5 vs 33.25 | 49.5 vs 43.5 | 59.2 vs 53.0 | 65.7 vs 59.2 | +6 ms at N ≥ 16 | `WS/ledger.benchrun.md` T7 |
| split (torch) vs ref | 150K | 35.2 vs 35.0 | 50.8 vs 49.1 | 60.8 vs 60.9 | — | parity | `WS/ledger.benchrun.md` T9 |
| split triton, sort merge vs ref | 34K | equal | 44.4 vs 42.1 | 53.8 vs 51.2 | 60.2 vs 57.9 | +2.0..+2.6 ms | `WS/RESULTS.md` §3 (benchrun graph A/B); per-cell medians `WS/ledger.benchrun.md` T12 (mean of the two halves) |
| same | 150K | equal | 46.5 vs 47.7 | 56.2 vs 59.1 | — | −1.2 / −2.9 ms | same |
| split triton, radix merge (93034eba91) vs ref | 200K | — | — | — | — | −4.4 / −6.7 / −9.1 ms (N=16/24/32; deltas only) | `WS/RESULTS.md` §3 (benchrun pair 6); per-cell medians `WS/ledger.benchrun.md` T16 (P6/P7) |
| same | 34K / 150K | — | — | — | — | 34K +0.5..+1.5 ms; 150K N=16 −2.6..−3.1, N=24 −4.1..−4.2 ms | `WS/ledger.benchrun.md` T15 |
| long context, AR on: 131K vs 34K | — | 34.8 vs 33.0 | 47.8 vs 43.6 | 59.8 vs 53.0 | — | +5 / +10 / +13 % | `WS/ledger.mixrun.md` "Long-context decode baseline" |
| DP2+DCP2 (per DP at N/2) vs DCP4 at N | 34K | — | equal within 1 ms | equal within 1 ms | — | ≈ 0 | `WS/ledger.mixrun.md` R25+; `results/mixed_checks/decodebench.dp2*.txt` |
| same | 131K | — | 45.2 vs 48.1 | 56.0 vs 59.6 | — | −6 % | same |

## 6. E2: PD with a DP2 × DCP2 decode (node crsuse2-m2m-v3-070, v3/SDMA=0) — correctness

Sim off, 40-concurrent check (20 math, 10 passkey 8K, 10 passkey 34K) with fresh seeds; "bad" = garbled answer
(first token(s) right, then junk). Source `WS/ledger.mixrun.md` lines 420-459; per-seed files `results/mixed_checks/check40.e2r*.txt`.

| Round | Decode mode | Config of note | Bad / requests | Source |
|---|---|---|---|---|
| pdE2-r1 | eager | custom AR on, index_share false | 0 / 40 (×1 round), check.py 5/5 | `results/mixed_checks/check40.e2r1-eager.txt` |
| pdE2-r2 | graph | index_share false, kv_b on, split-triton on | **3 / 200** | `results/mixed_checks/check40.e2r2-*.txt` |
| pdE2-r3 | graph, custom AR off | — | decode failed at graph capture (hipErrorCapturedEvent) | `WS/ledger.mixrun.md` |
| pdE2-r5 | graph | index_share **true** | **13 / 240** | `results/mixed_checks/check40.e2r5-*.txt` |
| pdE2-r6 | graph | split indexer **off** | **27 / 320** | `results/mixed_checks/check40.e2r6-*.txt` |

- Rates per stack are not separable from round-to-round variance (`WS/ledger.mixrun.md` line 455).
- User decision (10-22 ~02:05Z, `WS/mission.md`): garbling is a recorded known risk; not bisected further.
- The DCP4 (no DP) PD decode gates were 5-request checks only (`WS/poll.md` 01:15Z). A 40-concurrent sim-off check on
  the DCP4 PD decode in graph mode is **pending**.
- E2 perf round pdE2-p1 (AgentX C40, sim 3.61): started 02:19Z, then stopped for a decodebench task (`WS/poll.md`
  02:35Z); no score.

## 7. Files in this directory

| Dir | Content | Copied from |
|---|---|---|
| `pd/<round>/<bench>/` | `agentx_conc*.json`, `runtime.env`, `benchmark_command.txt`, `workload_distribution_summary.txt`, `client.container-exit-code` | `PD/runs/<round>/<bench>/` |
| `pd/<round>/` | `workers.json`, `decode_stats*.txt`, `spec_accept*.txt`, `census*.txt`, `audit*.txt`, `pdcheck.txt`, `smoke*.txt`, `status.tsv`, `pyspy*.txt`, `t4/check.*` | `PD/runs/<round>/` (rounds pdBe-r1, pdBeperf-r1, -r2, -r4, -r5b) |
| `mixed/<name>/` | same AgentX files | `MIX/agentx/<name>/` |
| `pdq/` | queue-depth readouts (`*.stats.txt`, `*.json`, `decode.stats.txt`, `prefill.allranks.txt`); the extracted `*.log` windows (0.4-1.8 MB) were not copied | `WS/pdrun/pdq/` |
| `decodebench/` | decodebench, check, conc, ctxload results | `BENCH/results/` (all non-log files, 600 KB) |
| `mixed_checks/` | mixed and E2 check / check40 / decodebench results | `MIX/results/` (all non-log files) |

Not copied: server logs, `aiperf_artifacts/`, `audit/*.json` (container inspect), plots. They remain at the cluster paths.
