# AUS request-level 阶段报告

生成时间：2026-10-10T08:21:41.080684+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 8881, "joined_prefill": 8881, "joined_decode": 8881, "paired": 8881, "sent_by_runner": 8918, "unexported_requests": 37, "paired_fraction_of_sent": 0.9958510876878224}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 8881 | 3281.65 | 1.61 | 10571.97 | 29737.77 |
| decode/bootstrap_ms | 8881 | 0.96 | 0.15 | 0.23 | 0.42 |
| decode/generation_ms | 8881 | 19347.98 | 9195.19 | 41169.77 | 156541.33 |
| decode/queue_ms | 8881 | 0.11 | 0.09 | 0.15 | 0.26 |
| decode/transfer_wait_ms | 8881 | 2854.36 | 1648.14 | 4512.89 | 21701.03 |
| prefill/bootstrap_ms | 8881 | 3271.80 | 3.31 | 10576.96 | 29653.00 |
| prefill/forward_envelope_ms | 8881 | 1426.28 | 995.51 | 2354.70 | 8525.38 |
| prefill/queue_ms | 8881 | 833.42 | 1.37 | 834.34 | 16361.34 |
| prefill/transfer_tail_ms | 8881 | 586.44 | 481.20 | 1102.93 | 2074.71 |

发生 allocation 阻塞的请求：`{"kv_budget": 2131}`

缓存统计：`{"input_tokens": 995664752, "cached_device": 941764608, "cached_host": 11969856, "cached_storage": 0, "miss_tokens": 41930288, "hit_rate": 0.9578871423179597}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
