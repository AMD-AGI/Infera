# C80 R1-P-session-affinity 与 G0-R1-only 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | G0-R1-only | R1-P-session-affinity | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 10033 | 10252 | +2.18% |
| Total tokens/s/GPU | 21264 | 21665 | +1.88% |
| Output tokens/s/GPU | 168.7 | 172.33 | +2.15% |
| 输出吞吐 (token/s) | 2699.2 | 2757.3 | +2.15% |
| 输入吞吐 (token/s，含命中) | 3.3753e+05 | 3.4388e+05 | +1.88% |
| TTFT mean (s) | 7.5792 | 6.5151 | -14.04% |
| TTFT p50 (s) | 4.5482 | 4.0691 | -10.53% |
| TTFT p90 (s) | 15.606 | 12.296 | -21.21% |
| TTFT p95 (s) | 23.828 | 19.242 | -19.25% |
| ITL mean (s) | 0.01348 | 0.01438 | +6.68% |
| 实际平均输入 tokens | 1.2211e+05 | 1.216e+05 | -0.41% |
| 实际平均输出 tokens | 976.49 | 975.02 | -0.15% |
| AIPerf GPU cache hit（聚合诊断值） | 0.93384 | 0.99272 | +6.31% |
| AIPerf Host cache hit（聚合诊断值） | 0.00758 | 0.01451 | +91.42% |
| 请求级 device hit rate | 0.94587 | 0.94619 | +0.03% |
| 请求级 host hit rate | 0.0086252 | 0.015545 | +80.23% |
| 请求级 miss rate | 0.045509 | 0.03826 | -15.93% |
| prefill/queue_ms mean | 2676.9 | 1991.1 | -25.62% |
| prefill/queue_ms p50 | 1.7923 | 1.6189 | -9.67% |
| prefill/queue_ms p90 | 7556.1 | 4580.8 | -39.38% |
| prefill/queue_ms p99 | 39965 | 32830 | -17.85% |
| prefill/forward_envelope_ms mean | 2880.3 | 2546.9 | -11.58% |
| prefill/forward_envelope_ms p50 | 2037.8 | 1875.1 | -7.99% |
| prefill/forward_envelope_ms p90 | 4236.6 | 3917.6 | -7.53% |
| prefill/forward_envelope_ms p99 | 19702 | 15783 | -19.89% |
| decode/alloc_wait_ms mean | 23.029 | 13.646 | -40.74% |
| decode/alloc_wait_ms p50 | 0.58658 | 0.55965 | -4.59% |
| decode/alloc_wait_ms p90 | 0.90081 | 0.83921 | -6.84% |
| decode/alloc_wait_ms p99 | 1.5628 | 1.401 | -10.35% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
