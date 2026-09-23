# AUS request-level 阶段报告

生成时间：2026-09-23T11:33:06.507140+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 9584, "joined_prefill": 9584, "joined_decode": 9584, "paired": 9584, "sent_by_runner": 9597, "unexported_requests": 13, "paired_fraction_of_sent": 0.9986454100239658}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 9584 | 63.45 | 0.56 | 0.83 | 34.90 |
| decode/bootstrap_ms | 9584 | 0.12 | 0.11 | 0.14 | 0.20 |
| decode/generation_ms | 9584 | 12971.84 | 6106.52 | 28435.90 | 103129.62 |
| decode/queue_ms | 9584 | 0.14 | 0.10 | 0.16 | 0.28 |
| decode/transfer_wait_ms | 9584 | 10232.35 | 5795.95 | 23322.43 | 54723.52 |
| prefill/bootstrap_ms | 9584 | 87.62 | 0.24 | 0.62 | 1954.92 |
| prefill/forward_envelope_ms | 9584 | 4504.22 | 3616.75 | 6460.71 | 19936.47 |
| prefill/queue_ms | 9584 | 4483.93 | 2.14 | 15298.12 | 46732.98 |
| prefill/transfer_tail_ms | 9584 | 1693.14 | 1649.56 | 2382.10 | 3308.88 |

发生 allocation 阻塞的请求：`{"kv_budget": 37}`

缓存统计：`{"input_tokens": 1133393191, "cached_device": 1069157376, "cached_host": 7396480, "cached_storage": 0, "miss_tokens": 56839335, "hit_rate": 0.949850294274443}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
