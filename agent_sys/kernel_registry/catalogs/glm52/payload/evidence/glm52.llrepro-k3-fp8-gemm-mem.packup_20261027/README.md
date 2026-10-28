> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# GLM-5.2 PD 4+4 (REDACTED_NODE -> REDACTED_NODE): ~47.9 k with K3 + draft FP8 + aiter GEMM retune + decode mem 0.95 (2026-10-26/27)

Follows `registry/glm52.llrepro-k3-mtp.packup_20261026` (llr27, K3, 46,565). Same topology and harness: prefill
compute-node-03 GPU4-7 (PP4xTP1), decode compute-node-04 GPU4-7 (TP4/DP4 DPA), contributor 4+4 kit + our hooks, AgentX
3600 s, CONC=96, A1 only. Goal = throughput only. Acceptance is SIMULATED (`SGLANG_SIMULATE_ACC_LEN=2.99` at K3):
**no correctness claim**. Master table: `OPT_LEDGER.md`.

## Result: current stack = llr35 config, 47,843 / 47,950 (replicate), mean 47,897 (+7.8 % vs llr24 44,432)

| run | change (one variable vs the parent) | parent | score | vs parent | qps | TPOT ms | step p50 ms | verdict |
|---|---|---|---|---|---|---|---|---|
| llr27 | MTP K3 (previous pack-up) | llr24 | 46,565 | +4.8 % | 3.041 | 15.4 | 44.6 | kept |
| llr29 | MTP draft experts online FP8 (`SGLANG_GLM_NEXTN_MOE_PTPC=1`) | llr27 | 47,014 | +1.0 % | 3.063 | 14.9 | 43.3 | kept |
| llr31 | decode per-rank CUDA graph cap 64 -> 80 | llr27 | 46,406 | -0.3 % | 3.041 | 15.4 | 44.3 | neutral (per-rank running p90 is 18, cap never binds) |
| llr32 | aiter BF16 GEMM retune csv #1 (75 shapes: o_proj N=6144 K=16384 etc.) | llr29 | 47,447 | +0.9 % | 3.104 | 14.7 | 42.1 | kept |
| llr33 | decode mem 0.93 -> 0.95 | llr32 | **47,957** | +1.1 % | 3.114 | 14.8 | 42.75 | kept (KV +4.2 %, prealloc-wait 0.44 -> 0.36, no HSA errors) |
| llr34 | draft MXFP4 (our patch) instead of FP8 | llr32 | 47,282 | -0.3 % | 3.090 | 14.7 | 42.4 | neutral (FP8 kept) |
| llr35 | aiter BF16 GEMM retune csv #2 (49 shapes that appear only with the FP8 draft) | llr33 | 47,843 | -0.2 % | 3.125 | 14.8 | 42.3 | neutral, kept in stack (0 untuned BF16 GEMMs left) |
| llr35r | replicate of llr35 | — | 47,950 | — | 3.126 | 14.7 | 42.3 | spread of the stack ±0.1 % |
| llr36 | decode mem 0.95 -> 0.96 | llr35 | 47,861 | +0.04 % | 3.111 | 14.7 | 42.4 | neutral (KV +2 % no longer converts to score) |

At pack-up time: llr37 = llr35 at CONC 104 running; llr38 = llr35 + prefill PP4 chunk 16384 queued.

## Decode state at the current stack (llr35r, `results/llr/llr35r-c96/bneck.txt`)
Per rank: running mean 12 / p90 18, token usage mean 0.86 (max 1.00), prealloc waits in 35 % of samples, step p50 42.3 /
p90 51 ms. Decode step went 55.3 (llr24) -> 42.3 ms: K3 -10.7, draft FP8 -1.3, GEMM retune -1.2 (per-run deltas above).

## Layout
`results/llr/<run>/` (agentx json, hist, bneck, qseries, isl_series, key log lines, guard/run logs), `sites/`
(llr29-llr38 + llr35r site files), `aiter_retune/` (the two tuned csvs, untuned inputs, tune scripts, `mk_untuned`),
`scripts/`, `OPT_LEDGER.md`, `REPRODUCE.md`, `notes.md`, `environment.md`.
