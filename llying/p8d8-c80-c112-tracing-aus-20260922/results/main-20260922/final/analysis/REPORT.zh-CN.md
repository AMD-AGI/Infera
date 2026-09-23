# AUS request-level 阶段报告

生成时间：2026-09-22T22:15:59.172057+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 9651, "joined_prefill": 9651, "joined_decode": 9651, "paired": 9651, "sent_by_runner": 9662, "unexported_requests": 11, "paired_fraction_of_sent": 0.998861519354171}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 9651 | 34.09 | 0.57 | 0.85 | 1.66 |
| decode/bootstrap_ms | 9651 | 0.12 | 0.11 | 0.15 | 0.22 |
| decode/generation_ms | 9651 | 13132.53 | 6090.28 | 28699.16 | 111289.11 |
| decode/queue_ms | 9651 | 0.13 | 0.10 | 0.16 | 0.29 |
| decode/transfer_wait_ms | 9651 | 9420.72 | 4052.41 | 24079.75 | 60113.59 |
| prefill/bootstrap_ms | 9651 | 70.85 | 0.22 | 0.59 | 1184.50 |
| prefill/forward_envelope_ms | 9651 | 3119.94 | 2223.07 | 4684.16 | 19962.59 |
| prefill/queue_ms | 9651 | 5575.70 | 3.19 | 17619.32 | 54289.45 |
| prefill/transfer_tail_ms | 9651 | 927.39 | 880.73 | 1317.09 | 1961.46 |

发生 allocation 阻塞的请求：`{"kv_budget": 38}`

缓存统计：`{"input_tokens": 1157823951, "cached_device": 1091984192, "cached_host": 6933568, "cached_storage": 0, "miss_tokens": 58906191, "hit_rate": 0.9491233611559656}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## C112

关联覆盖：`{"client_records": 9321, "joined_prefill": 9321, "joined_decode": 9321, "paired": 9321, "sent_by_runner": 9354, "unexported_requests": 33, "paired_fraction_of_sent": 0.9964720974983964}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 9321 | 793.20 | 0.59 | 0.98 | 28630.48 |
| decode/bootstrap_ms | 9321 | 0.12 | 0.11 | 0.14 | 0.21 |
| decode/generation_ms | 9321 | 12513.20 | 5478.77 | 27270.56 | 108600.28 |
| decode/queue_ms | 9321 | 0.13 | 0.11 | 0.16 | 0.28 |
| decode/transfer_wait_ms | 9321 | 28028.63 | 12283.57 | 77161.09 | 164920.71 |
| prefill/bootstrap_ms | 9321 | 844.59 | 0.23 | 3.10 | 28832.16 |
| prefill/forward_envelope_ms | 9321 | 4594.37 | 2933.16 | 6305.17 | 41211.41 |
| prefill/queue_ms | 9321 | 22409.10 | 6432.36 | 68698.61 | 157972.25 |
| prefill/transfer_tail_ms | 9321 | 1222.43 | 1120.66 | 1751.82 | 2788.97 |

发生 allocation 阻塞的请求：`{"kv_budget": 268}`

缓存统计：`{"input_tokens": 1047702056, "cached_device": 942076608, "cached_host": 29092928, "cached_storage": 0, "miss_tokens": 76532520, "hit_rate": 0.9269520188857967}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
