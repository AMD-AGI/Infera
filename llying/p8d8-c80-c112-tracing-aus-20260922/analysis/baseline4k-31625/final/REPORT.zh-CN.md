# AUS request-level 阶段报告

生成时间：2026-09-23T14:12:11.299522+00:00

本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。

## C80

关联覆盖：`{"client_records": 9782, "joined_prefill": 9782, "joined_decode": 9782, "paired": 9782, "sent_by_runner": 9793, "unexported_requests": 11, "paired_fraction_of_sent": 0.9988767486980497}`

| 阶段 | 样本数 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| decode/alloc_wait_ms | 9782 | 21.94 | 0.56 | 0.84 | 1.47 |
| decode/bootstrap_ms | 9782 | 0.12 | 0.11 | 0.15 | 0.21 |
| decode/generation_ms | 9782 | 13344.62 | 6199.07 | 29268.40 | 111570.31 |
| decode/queue_ms | 9782 | 0.13 | 0.10 | 0.16 | 0.28 |
| decode/transfer_wait_ms | 9782 | 8620.83 | 3970.12 | 21746.64 | 54169.73 |
| prefill/bootstrap_ms | 9782 | 50.15 | 0.22 | 0.55 | 1061.46 |
| prefill/forward_envelope_ms | 9782 | 3045.90 | 2163.06 | 4579.96 | 19786.57 |
| prefill/queue_ms | 9782 | 4733.43 | 2.92 | 16291.78 | 46170.44 |
| prefill/transfer_tail_ms | 9782 | 929.90 | 875.66 | 1302.14 | 1881.59 |

发生 allocation 阻塞的请求：`{"kv_budget": 20}`

缓存统计：`{"input_tokens": 1186970342, "cached_device": 1124691968, "cached_host": 5837568, "cached_storage": 0, "miss_tokens": 56440806, "hit_rate": 0.9524496914515157}`

逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。

## 解释边界

- Tracing overhead is not established by comparison with historical runs.
- Host acknowledgement is CPU observation, not pure DMA duration.
- Cross-host clocks require independent uncertainty bounds.
- Prefill and Decode waits overlap; do not add their percentiles.
- Long-window balanced token counts do not exclude transient imbalance.
