# C80 campaign-b2-radix 与 P_session_baseline 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | P_session_baseline | campaign-b2-radix | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 10252 | 10295 | +0.42% |
| Total tokens/s/GPU | 21665 | 21716 | +0.23% |
| Output tokens/s/GPU | 172.33 | 170.53 | -1.04% |
| 输出吞吐 (token/s) | 2757.3 | 2728.6 | -1.04% |
| 输入吞吐 (token/s，含命中) | 3.4388e+05 | 3.4472e+05 | +0.24% |
| TTFT mean (s) | 6.5151 | 6.0946 | -6.45% |
| TTFT p50 (s) | 4.0691 | 4.1177 | +1.19% |
| TTFT p90 (s) | 12.296 | 10.796 | -12.20% |
| TTFT p95 (s) | 19.242 | 18.566 | -3.51% |
| ITL mean (s) | 0.01438 | 0.01464 | +1.81% |
| 实际平均输入 tokens | 1.216e+05 | 1.2153e+05 | -0.06% |
| 实际平均输出 tokens | 975.02 | 961.95 | -1.34% |
| AIPerf GPU cache hit（聚合诊断值） | 0.99272 | 0.99195 | -0.08% |
| AIPerf Host cache hit（聚合诊断值） | 0.01451 | 0.0157 | +8.20% |
| 请求级 device hit rate | 0.94619 | 0.9452 | -0.11% |
| 请求级 host hit rate | 0.015545 | 0.016812 | +8.15% |
| 请求级 miss rate | 0.03826 | 0.037993 | -0.70% |
| prefill/queue_ms mean | 1991.1 | 1580.7 | -20.62% |
| prefill/queue_ms p50 | 1.6189 | 1.5145 | -6.45% |
| prefill/queue_ms p90 | 4580.8 | 3526.1 | -23.02% |
| prefill/queue_ms p99 | 32830 | 29417 | -10.40% |
| prefill/forward_envelope_ms mean | 2546.9 | 2403.2 | -5.64% |
| prefill/forward_envelope_ms p50 | 1875.1 | 1787.6 | -4.66% |
| prefill/forward_envelope_ms p90 | 3917.6 | 3558.4 | -9.17% |
| prefill/forward_envelope_ms p99 | 15783 | 15029 | -4.78% |
| decode/alloc_wait_ms mean | 13.646 | 22.21 | +62.75% |
| decode/alloc_wait_ms p50 | 0.55965 | 0.89841 | +60.53% |
| decode/alloc_wait_ms p90 | 0.83921 | 1.6715 | +99.18% |
| decode/alloc_wait_ms p99 | 1.401 | 3.0729 | +119.34% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
