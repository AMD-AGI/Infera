# AUS request-level 阶段报告

生成时间：2026-09-24T04:05:30.597489+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 9774, "joined_prefill": 9774, "joined_decode": 9774, "paired": 9774, "sent_by_runner": 9796, "unexported_requests": 22, "paired_fraction_of_sent": 0.9977541853817885}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 9774 | 58.68 | 0.56 | 0.85 | 1.64 |
| decode/bootstrap_ms | 9774 | 0.11 | 0.11 | 0.14 | 0.20 |
| decode/generation_ms | 9774 | 13342.33 | 6249.87 | 29190.46 | 110469.97 |
| decode/queue_ms | 9774 | 0.13 | 0.10 | 0.15 | 0.27 |
| decode/transfer_wait_ms | 9774 | 9123.29 | 3894.25 | 22803.39 | 60075.56 |
| prefill/bootstrap_ms | 9774 | 103.77 | 0.22 | 0.63 | 1582.21 |
| prefill/forward_envelope_ms | 9774 | 2988.07 | 2167.03 | 4421.30 | 18261.61 |
| prefill/queue_ms | 9774 | 5457.58 | 2.99 | 17652.68 | 52672.32 |
| prefill/transfer_tail_ms | 9774 | 910.79 | 869.26 | 1291.67 | 1804.11 |

发生 allocation 阻塞的请求：`{"kv_budget": 33}`

缓存统计：`{"input_tokens": 1191147019, "cached_device": 1134073920, "cached_host": 1846848, "cached_storage": 0, "miss_tokens": 55226251, "hit_rate": 0.9536360750444022}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
