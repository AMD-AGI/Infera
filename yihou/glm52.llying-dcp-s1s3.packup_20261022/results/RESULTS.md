# Results — AgentX C80, 1800 s, one run per stage (2026-10-22)

Source: each run dir's `analysis.txt` (made by `scripts/kit/analyze_ll.yihou.sh`) and `agentx/agentx_conc80.json`.
Per chip = total tok/s / 8. The JSON `per_gpu` is total / 5 for all four runs (notes §1) and is wrong.

| Run dir | Stage | total tok/s | **tok/s/chip** | JSON per_gpu (wrong) | ok / err (all err in warm-up) |
|---|---|---|---|---|---|
| `S1-219_20261022T121047Z` | S1 DPA4 decode | 276,074 | **34,509** | 55,215 | 4507 / 4 |
| `S2-039_20261022T121510Z` | S2 DCP4 decode | 207,525 | **25,941** | 41,505 | 3362 / 6 |
| `S3-210_20261022T123116Z` | S3 DCP4 + OffV2 prefill | 168,201 | **21,025** | 33,640 | 2671 / 4 |
| `S3-028_20261022T123245Z` | S3-aux (burst, SDMA=0; not comparable) | 131,777 | 16,472 | 26,355 | 2065 / 3 |

## Latency
| Run | TTFT mean / p50 / p90 / p95 (s) | TPOT mean / p50 / p90 / p95 (ms) |
|---|---|---|
| S1 | 5.50 / 3.85 / 8.62 / 13.26 | 18.31 / 19.13 / 20.81 / 21.33 |
| S2 | 17.69 / 13.00 / 32.70 / 47.49 | 20.32 / 21.21 / 23.73 / 24.57 |
| S3 | 34.35 / 29.20 / 66.18 / 77.41 | 18.59 / 19.05 / 22.10 / 23.04 |
| S3-aux | 56.38 / 54.43 / 103.48 / 119.20 | 16.29 / 16.08 / 20.45 / 21.36 |

## Queue readout (means over steady-state log lines; llm-pd-bottleneck-finder method, `extract_pd_stats.py`)
| Run | prefill input | handshake | prefill in-flight (KV send) | decode running | decode transfer-in | decode admission (prealloc) | retracted |
|---|---|---|---|---|---|---|---|
| S1 | 2.0 | 0.4 | 6.8 | 9.9 per DP rank (x4) | 3.0 | 0.2 | 0 |
| S2 | 0.7 | 11.5 | 17.2 | 30.8 (cap 96) | 22.0 | 12.8 (max 56) | 0 |
| S3 | 0.9 | 15.6 | 30.7 | 22.9 | 35.1 | 16.9 (max 55) | 0 |
| S3-aux | 0.9 | 20.6 | 40.6 | 14.2 | 45.6 | 22.2 (max 65) | 0 |

Client (aiperf) server-metrics during profiling reported decode `kv_usage` 96-99 % in S2/S3 (e.g. S3-210 at 13:42Z:
98.0 %). S1: no stage backed up. S2/S3: admission and transfer-in queues deep while running stays under the cap.
**Why admission backs up is not established** (candidate measurements: notes §4).

## Decode step p50 (ms) by running requests (TP0 / DP0 log lines)
| running | 1-8 | 9-16 | 17-24 | 25-32 | 33-40 | 41-64 |
|---|---|---|---|---|---|---|
| S1 (per DP rank, DP0) | 42.8 | 67.3 | 72.4 | 79.5 (n=3) | - | - |
| S2 (whole DCP4 group) | 28.4 | 44.6 | 52.6 | 63.1 | 71.5 | 78.9 |
| S3 | 32.4 | 45.3 | 53.8 | 62.7 | 71.5 | 75.6 |
| S3-aux | 28.6 | 45.7 | 53.6 | 63.0 | 71.8 | 77.2 |

S1 counts requests on one DP rank (4 ranks run in parallel); S2/S3 count the whole decode group. Same bucket is not
the same node load. p90 and per-rank tables: `analysis.txt`.

## Faults
Decode logs: 0 Traceback lines in all runs. Prefill logs: 5/5/1/5 Traceback-related lines (S1/S2/S3/S3-aux); in S1/S2 one
is the start-up `/freeze_gc` connection refused; the rest were not inspected. AgentX rc=2 on both S3 runs is a
client teardown `OSError` (`/ax-tmp/pymp-...` not empty) after the result was written and validated.

## Not included (size; on NFS)
`agentx/aiperf_artifacts/{profile_export.jsonl,*timeslices*,server_metrics_export.*}` (15 MB - 1.5 GB each),
`logs-live/{prefill0,decode}{,.win}.log` (0.3-17 MB): under
`/shared_nfs/yihou_glm52/llying_dcp_20261022/work/<S>-<NK>/runs/<RUN_ID>/`.
