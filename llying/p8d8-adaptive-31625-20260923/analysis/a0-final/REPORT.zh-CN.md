# AUS request-level 阶段报告

生成时间：2026-09-23T17:32:42.159695+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 9727, "joined_prefill": 9727, "joined_decode": 9727, "paired": 9727, "sent_by_runner": 9740, "unexported_requests": 13, "paired_fraction_of_sent": 0.9986652977412731}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 9727 | 25.89 | 0.57 | 0.86 | 1.57 |
| decode/bootstrap_ms | 9727 | 0.12 | 0.11 | 0.15 | 0.22 |
| decode/generation_ms | 9727 | 13356.82 | 6243.18 | 29164.50 | 114241.31 |
| decode/queue_ms | 9727 | 0.13 | 0.10 | 0.16 | 0.28 |
| decode/transfer_wait_ms | 9727 | 8609.41 | 3843.46 | 21096.50 | 58251.19 |
| prefill/bootstrap_ms | 9727 | 58.14 | 0.22 | 0.58 | 1135.78 |
| prefill/forward_envelope_ms | 9727 | 3038.20 | 2153.31 | 4529.96 | 19752.56 |
| prefill/queue_ms | 9727 | 4857.12 | 2.94 | 15631.78 | 50779.79 |
| prefill/transfer_tail_ms | 9727 | 904.82 | 862.63 | 1280.20 | 1852.45 |

发生 allocation 阻塞的请求：`{"kv_budget": 32}`

缓存统计：`{"input_tokens": 1161686094, "cached_device": 1097429312, "cached_host": 6266432, "cached_storage": 0, "miss_tokens": 57990350, "hit_rate": 0.950080877872676}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
