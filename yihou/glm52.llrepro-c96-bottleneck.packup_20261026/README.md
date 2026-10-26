# GLM-5.2 PD 4+4 (135 -> 136): C96 = current best (44,432) + bottleneck analysis (2026-10-26)

Follows `yihou/glm52.llrepro-ppcomm-c88.packup_20261025`. Config = llr21 (llying 4+4 kit + KF decode kernels + decode
custom AR + TOPK_V2=false + mem 0.93 + stream-interval 30 + CLIP 8 + prefill pp-async-batch-depth 1 + prefill
SGLANG_PP_COMM_OVERLAP=1) at CONC=96; AgentX 3600 s, sim acceptance 3.61 (**no correctness claim**). Goal: throughput
only (user 10-26). Master table of all optimizations: `OPT_LEDGER.md` (also on NFS
`/shared_nfs/yihou_glm52/llrepro_20261023/OPT_LEDGER.md`).

## Concurrency (llr21 config)

| | C80 (llr21r; mean of 3 = 43,399) | C88 (llr23) | C96 (llr24) |
|---|---|---|---|
| score | 43,716 | 44,233 | **44,432** |
| completed req/s | 2.875 | 2.931 | 2.941 |
| TTFT p50/p90 s | 2.99 / 6.62 | 4.09 / 11.96 | 5.51 / 16.27 |
| TPOT ms | 16.2 | 16.9 | 17.6 |
| decode running (4 ranks) | 46.4 | 49.1 | 51.7 |
| decode KV usage mean | 0.75 | 0.85 | 0.90 |
| decode prealloc waits (frac of time) | 0.11 | 0.35 | 0.59 |
| decode step p50 ms | 52.5 | 53.3 | 55.3 |
| prefill PP0 batch/s | 2.25 | 2.07 | 1.96 |
| prefill bootstrap-req (waiting for decode allocation) | 0.8 | 3.1 | 7.2 |

## Bottleneck at C96: decode (compute and KV capacity)
Probe 05:12-05:15Z (`analysis/c96.*`): decode GPU4-7 use 100 % (min 98), VRAM 286-288 of 288 GiB, 843-881 W; one
decode process at 100 % CPU = the DP controller busy-poll (`data_parallel_controller.py:906`, NOBLOCK recv loop);
prefill GPU4-7 use 61-81 % (idle samples), VRAM 240-256 GiB, no core above 50 %. Raising concurrency 80 -> 96 (+20 %)
gave +2.3 % req/s; the extra requests wait for decode KV (prealloc / bootstrap).

Within a run (`results/llr/llr24-c96/qseries.txt`, `analysis/isl_series.*.txt`): decode running peaks ~66 at 20-40 min
then falls to ~36 while KV usage stays 0.92-0.94, because later AgentX turns are longer (ISL mean ~100k in the first
30 min vs 147k-169k in the last 20 min; p90 up to 326k) - fewer requests fit the fixed KV pool. Same shape at C88.

## Layout
`results/llr/llr24-c96/` (as previous pack-ups), `analysis/` (VRAM/docker-stats/RDMA/CPU probes, ISL series),
`scripts/` (site files llr24/llr25, isl_series.yihou.py, probe_mem.yihou.sh, run/guard scripts), `OPT_LEDGER.md`.
