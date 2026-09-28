# C80 campaign-b4-triton 与 RB_selected_stack 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | RB_selected_stack | campaign-b4-triton | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 10186 | 10728 | +5.32% |
| Total tokens/s/GPU | 21481 | 22756 | +5.94% |
| Output tokens/s/GPU | 166.33 | 171.95 | +3.38% |
| 输出吞吐 (token/s) | 2661.4 | 2751.2 | +3.38% |
| 输入吞吐 (token/s，含命中) | 3.4103e+05 | 3.6134e+05 | +5.96% |
| TTFT mean (s) | 6.5358 | 4.6542 | -28.79% |
| TTFT p50 (s) | 4.235 | 3.2108 | -24.18% |
| TTFT p90 (s) | 11.529 | 7.811 | -32.25% |
| TTFT p95 (s) | 19.107 | 11.982 | -37.29% |
| ITL mean (s) | 0.01485 | 0.01474 | -0.74% |
| 实际平均输入 tokens | 1.2149e+05 | 1.2222e+05 | +0.60% |
| 实际平均输出 tokens | 948.13 | 930.56 | -1.85% |
| AIPerf GPU cache hit（聚合诊断值） | 0.99287 | 0.99574 | +0.29% |
| AIPerf Host cache hit（聚合诊断值） | 0.02531 | 0.02007 | -20.70% |
| 请求级 device hit rate | 0.9369 | 0.94234 | +0.58% |
| 请求级 host hit rate | 0.025102 | 0.020084 | -19.99% |
| 请求级 miss rate | 0.037994 | 0.037578 | -1.10% |
| prefill/queue_ms mean | 1894.5 | 985.27 | -47.99% |
| prefill/queue_ms p50 | 1.5497 | 1.2228 | -21.09% |
| prefill/queue_ms p90 | 4356.4 | 1073.6 | -75.36% |
| prefill/queue_ms p99 | 30834 | 18982 | -38.44% |
| prefill/forward_envelope_ms mean | 2548.8 | 1590.9 | -37.58% |
| prefill/forward_envelope_ms p50 | 1875.3 | 1164.3 | -37.91% |
| prefill/forward_envelope_ms p90 | 3765.4 | 2582.2 | -31.42% |
| prefill/forward_envelope_ms p99 | 15755 | 10220 | -35.13% |
| decode/alloc_wait_ms mean | 3.1929 | 1.8249 | -42.84% |
| decode/alloc_wait_ms p50 | 1.1896 | 1.1676 | -1.85% |
| decode/alloc_wait_ms p90 | 2.3568 | 2.2677 | -3.78% |
| decode/alloc_wait_ms p99 | 4.5878 | 4.0993 | -10.65% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
