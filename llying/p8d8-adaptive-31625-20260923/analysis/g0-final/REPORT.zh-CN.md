# AUS request-level 阶段报告

生成时间：2026-09-23T19:07:43.595958+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 10033, "joined_prefill": 10033, "joined_decode": 10033, "paired": 10033, "sent_by_runner": 10049, "unexported_requests": 16, "paired_fraction_of_sent": 0.9984078017713205}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 10033 | 23.03 | 0.59 | 0.90 | 1.56 |
| decode/bootstrap_ms | 10033 | 0.12 | 0.12 | 0.16 | 0.22 |
| decode/generation_ms | 10033 | 13059.07 | 6110.90 | 28410.62 | 112421.52 |
| decode/queue_ms | 10033 | 0.13 | 0.10 | 0.16 | 0.29 |
| decode/transfer_wait_ms | 10033 | 6253.16 | 3079.87 | 13687.71 | 47718.62 |
| prefill/bootstrap_ms | 10033 | 59.39 | 0.22 | 0.52 | 1109.24 |
| prefill/forward_envelope_ms | 10033 | 2880.29 | 2037.81 | 4236.58 | 19702.28 |
| prefill/queue_ms | 10033 | 2676.90 | 1.79 | 7556.06 | 39964.77 |
| prefill/transfer_tail_ms | 10033 | 896.28 | 867.22 | 1310.29 | 1874.56 |

发生 allocation 阻塞的请求：`{"kv_budget": 25}`

缓存统计：`{"input_tokens": 1225115004, "cached_device": 1158794816, "cached_host": 10566912, "cached_storage": 0, "miss_tokens": 55753276, "hit_rate": 0.9544913940177325}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
