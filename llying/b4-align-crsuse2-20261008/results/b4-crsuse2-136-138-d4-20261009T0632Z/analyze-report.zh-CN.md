# AUS request-level 阶段报告

生成时间：2026-10-09T08:16:00.387011+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 7287, "joined_prefill": 7287, "joined_decode": 7287, "paired": 7287, "sent_by_runner": 7318, "unexported_requests": 31, "paired_fraction_of_sent": 0.9957638699098115}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 7287 | 15858.21 | 348.27 | 52321.98 | 97393.50 |
| decode/bootstrap_ms | 7287 | 0.17 | 0.16 | 0.22 | 0.33 |
| decode/generation_ms | 7287 | 17497.80 | 8149.67 | 37598.28 | 154969.17 |
| decode/queue_ms | 7287 | 0.11 | 0.09 | 0.16 | 0.28 |
| decode/transfer_wait_ms | 7287 | 2724.54 | 1496.74 | 4238.37 | 25502.37 |
| prefill/bootstrap_ms | 7287 | 15879.73 | 716.54 | 52408.90 | 97488.28 |
| prefill/forward_envelope_ms | 7287 | 1340.73 | 927.60 | 2179.77 | 8935.35 |
| prefill/queue_ms | 7287 | 861.23 | 1.27 | 583.06 | 18699.35 |
| prefill/transfer_tail_ms | 7287 | 526.65 | 443.08 | 1025.58 | 1856.24 |

发生 allocation 阻塞的请求：`{"kv_budget": 1597}`

缓存统计：`{"input_tokens": 831660245, "cached_device": 790602944, "cached_host": 5102464, "cached_storage": 0, "miss_tokens": 35954837, "hit_rate": 0.9567673972440512}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
