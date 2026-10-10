# AUS request-level 阶段报告

生成时间：2026-10-10T04:21:11.202912+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 7990, "joined_prefill": 7990, "joined_decode": 7990, "paired": 7990, "sent_by_runner": 8020, "unexported_requests": 30, "paired_fraction_of_sent": 0.9962593516209476}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 7990 | 10512.44 | 1.49 | 41430.57 | 88439.77 |
| decode/bootstrap_ms | 7990 | 0.18 | 0.16 | 0.24 | 0.39 |
| decode/generation_ms | 7990 | 18120.20 | 8532.02 | 38739.85 | 150993.73 |
| decode/queue_ms | 7990 | 0.11 | 0.09 | 0.15 | 0.28 |
| decode/transfer_wait_ms | 7990 | 2718.74 | 1547.17 | 4065.67 | 23147.29 |
| prefill/bootstrap_ms | 7990 | 10511.26 | 2.58 | 41415.76 | 88567.23 |
| prefill/forward_envelope_ms | 7990 | 1336.35 | 941.19 | 2213.31 | 8769.46 |
| prefill/queue_ms | 7990 | 817.24 | 1.27 | 487.10 | 17216.49 |
| prefill/transfer_tail_ms | 7990 | 558.76 | 459.72 | 1093.98 | 2222.19 |

发生 allocation 阻塞的请求：`{"kv_budget": 1427}`

缓存统计：`{"input_tokens": 910727855, "cached_device": 867624320, "cached_host": 5116928, "cached_storage": 0, "miss_tokens": 37986607, "hit_rate": 0.9582898373082044}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
