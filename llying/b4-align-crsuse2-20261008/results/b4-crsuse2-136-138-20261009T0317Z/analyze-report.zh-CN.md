# AUS request-level 阶段报告

生成时间：2026-10-09T05:03:51.162492+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 10644, "joined_prefill": 10644, "joined_decode": 10644, "paired": 10644, "sent_by_runner": 10655, "unexported_requests": 11, "paired_fraction_of_sent": 0.9989676208352886}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 10644 | 41.65 | 1.36 | 2.71 | 6.70 |
| decode/bootstrap_ms | 10644 | 0.16 | 0.16 | 0.22 | 0.27 |
| decode/generation_ms | 10644 | 13554.12 | 6222.35 | 30157.92 | 112129.63 |
| decode/queue_ms | 10644 | 0.12 | 0.09 | 0.16 | 0.32 |
| decode/transfer_wait_ms | 10644 | 3082.11 | 1757.03 | 4960.58 | 25560.24 |
| prefill/bootstrap_ms | 10644 | 131.03 | 0.27 | 433.17 | 1298.17 |
| prefill/forward_envelope_ms | 10644 | 1546.60 | 1101.40 | 2530.69 | 9525.40 |
| prefill/queue_ms | 10644 | 940.28 | 1.40 | 919.55 | 20111.27 |
| prefill/transfer_tail_ms | 10644 | 610.39 | 541.76 | 1076.42 | 1829.73 |

发生 allocation 阻塞的请求：`{"kv_budget": 40}`

缓存统计：`{"input_tokens": 1297647457, "cached_device": 1238013376, "cached_host": 9880512, "cached_storage": 0, "miss_tokens": 49753569, "hit_rate": 0.961658639462043}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
