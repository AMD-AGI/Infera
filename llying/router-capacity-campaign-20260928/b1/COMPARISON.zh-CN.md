# C80 campaign-b1-dynamo-p 与 P_session_baseline 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | P_session_baseline | campaign-b1-dynamo-p | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 10252 | 10074 | -1.74% |
| Total tokens/s/GPU | 21665 | 21259 | -1.88% |
| Output tokens/s/GPU | 172.33 | 168.11 | -2.45% |
| 输出吞吐 (token/s) | 2757.3 | 2689.8 | -2.45% |
| 输入吞吐 (token/s，含命中) | 3.4388e+05 | 3.3745e+05 | -1.87% |
| TTFT mean (s) | 6.5151 | 7.155 | +9.82% |
| TTFT p50 (s) | 4.0691 | 4.2373 | +4.13% |
| TTFT p90 (s) | 12.296 | 12.331 | +0.28% |
| TTFT p95 (s) | 19.242 | 21.518 | +11.83% |
| ITL mean (s) | 0.01438 | 0.01446 | +0.56% |
| 实际平均输入 tokens | 1.216e+05 | 1.2156e+05 | -0.03% |
| 实际平均输出 tokens | 975.02 | 968.98 | -0.62% |
| AIPerf GPU cache hit（聚合诊断值） | 0.99272 | 0.99344 | +0.07% |
| AIPerf Host cache hit（聚合诊断值） | 0.01451 | 0.01052 | -27.50% |
| 请求级 device hit rate | 0.94619 | 0.94715 | +0.10% |
| 请求级 host hit rate | 0.015545 | 0.011468 | -26.23% |
| 请求级 miss rate | 0.03826 | 0.041383 | +8.16% |
| prefill/queue_ms mean | 1991.1 | 2482.6 | +24.68% |
| prefill/queue_ms p50 | 1.6189 | 1.6042 | -0.91% |
| prefill/queue_ms p90 | 4580.8 | 5017.3 | +9.53% |
| prefill/queue_ms p99 | 32830 | 38992 | +18.77% |
| prefill/forward_envelope_ms mean | 2546.9 | 2651.7 | +4.12% |
| prefill/forward_envelope_ms p50 | 1875.1 | 1914 | +2.08% |
| prefill/forward_envelope_ms p90 | 3917.6 | 3873 | -1.14% |
| prefill/forward_envelope_ms p99 | 15783 | 17267 | +9.40% |
| decode/alloc_wait_ms mean | 13.646 | 28.756 | +110.72% |
| decode/alloc_wait_ms p50 | 0.55965 | 0.56309 | +0.61% |
| decode/alloc_wait_ms p90 | 0.83921 | 0.85767 | +2.20% |
| decode/alloc_wait_ms p99 | 1.401 | 1.4834 | +5.88% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
