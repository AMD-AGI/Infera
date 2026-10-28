> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# GLM-5.2 PD 4+4 (REDACTED_NODE -> REDACTED_NODE): concurrency sweep on the ~47.9 k stack: C104 = 48,713 (best, single run) (2026-10-27)

Follows `registry/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027` (stack = llr35: contributor 4+4 kit + KF decode kernels + K3 +
draft FP8 + 2 aiter BF16 GEMM csvs + decode mem 0.95; C96 mean 47,897). Same topology/harness; AgentX 3600 s, A1 only.
Acceptance is SIMULATED (2.99 at K3): **no correctness claim**. Master table: `OPT_LEDGER.md`.

**Correctness warning (contributor, 10-27, branch commits c9ea79ce..19a9c82d):** with REAL acceptance the llr35 stack
garbles (probe 10/200, long-context needle 13/64 at 16k, 2/16 at 86k); his bisect points mainly at
`SGLANG_PP_COMM_OVERLAP=1` (prefill env in this stack), decode custom all-reduce adds short-prompt garbling, long
contexts still degrade without either (round 2 running at pack-up time). Every score here includes these settings.

| run | change (one variable vs llr35) | score | vs C96 mean | qps | TTFT p50/p90 s | decode KV usage / prealloc-wait | verdict |
|---|---|---|---|---|---|---|---|
| llr35 / llr35r | C96 (previous pack-up) | 47,843 / 47,950 | — | 3.126 | 5.0 / 13 | 0.86 / 0.35 | base |
| **llr37** | **CONC 96 -> 104** | **48,713** | **+1.7 %** | 3.209 | 6.0 / 15.0 | 0.92 / 0.58 | **best (replicate llr37r running at pack-up)** |
| llr38 | prefill PP4 chunk 8192 -> 16384 (C96) | 46,862 | -2.2 % | 3.048 | 6.5 / 15.0 | 0.86 / 0.35 | rejected: prefill PP0 batches 1.93 -> 1.34/s, KV transfer p50 2.0 -> 3.0 s |
| llr39 | CONC 104 -> 112 | 45,668 | -4.7 % (-6.3 % vs C104) | 3.066 | 10.0 / 23.4 | 0.95 / 0.90 | rejected: decode KV saturated, GPU prefix hit 0.43 -> 0.39 |

Throughput peaks between C104 and C112 on this stack. Queued after llr37r: llr40 (decode `--enable-dp-lm-head`),
llr41 (decode `--enable-dp-attention-local-control-broadcast`), both on llr37 (site files in `sites/`).

## NFS cleanup (user-approved 10-27, `nfs_cleanup/`)
`REDACTED_LOCAL_PATH` 939 GB -> 189 GB (du). Audit `REPORT.md`; validated path lists (`*.list.gz`); `validate.registry.sh`
(prefix / depth / protected / live-mount checks, dry run); `gen_cmds.registry.py` turns a validated list into literal per-path
commands; `cleanup_run.registry.sh` runs steps 1-4 (aiperf dataset copies, stale caches, gzip of large json/logs, root-owned
regenerable caches removed from a short-lived container); `cleanup_run.log` has the per-step counts. Image tars kept.
`llrepro_20261023/xferlat.registry/extract.registry.py` now also reads `.gz` inputs.

## Layout
`results/llr/<run>/` (agentx json, hist, bneck, qseries, isl_series, key log lines, run/guard logs), `sites/`, `scripts/`,
`nfs_cleanup/`, `OPT_LEDGER.md`. Reproduce exactly as in the previous pack-up's REPRODUCE.md with the site files here.
