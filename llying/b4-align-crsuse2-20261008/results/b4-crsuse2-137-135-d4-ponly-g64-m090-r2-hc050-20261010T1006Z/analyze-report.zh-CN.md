# AUS request-level 阶段报告

生成时间：2026-10-12T08:27:53.867986+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 8817, "joined_prefill": 8817, "joined_decode": 8817, "paired": 8817, "sent_by_runner": 8841, "unexported_requests": 24, "paired_fraction_of_sent": 0.997285374957584}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 8817 | 3448.94 | 2.56 | 11789.98 | 30183.26 |
| decode/bootstrap_ms | 8817 | 0.18 | 0.16 | 0.25 | 0.45 |
| decode/generation_ms | 8817 | 19756.72 | 9367.73 | 42399.02 | 162963.16 |
| decode/queue_ms | 8817 | 0.12 | 0.10 | 0.17 | 0.28 |
| decode/transfer_wait_ms | 8817 | 2695.50 | 1631.08 | 4447.98 | 19801.35 |
| prefill/bootstrap_ms | 8817 | 3433.49 | 1.49 | 11778.77 | 30164.38 |
| prefill/forward_envelope_ms | 8817 | 1351.32 | 983.05 | 2192.54 | 8885.10 |
| prefill/queue_ms | 8817 | 747.11 | 1.35 | 887.81 | 14572.77 |
| prefill/transfer_tail_ms | 8817 | 583.45 | 477.53 | 1120.61 | 2131.31 |

发生 allocation 阻塞的请求：`{"kv_budget": 2048}`

缓存统计：`{"input_tokens": 989761216, "cached_device": 940799680, "cached_host": 7740672, "cached_storage": 0, "miss_tokens": 41220864, "hit_rate": 0.958352718480333}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
