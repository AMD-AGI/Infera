# C80 campaign-rb-rebaseline 与 P_session_baseline 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | P_session_baseline | campaign-rb-rebaseline | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 10252 | 10186 | -0.64% |
| Total tokens/s/GPU | 21665 | 21481 | -0.85% |
| Output tokens/s/GPU | 172.33 | 166.33 | -3.48% |
| 输出吞吐 (token/s) | 2757.3 | 2661.4 | -3.48% |
| 输入吞吐 (token/s，含命中) | 3.4388e+05 | 3.4103e+05 | -0.83% |
| TTFT mean (s) | 6.5151 | 6.5358 | +0.32% |
| TTFT p50 (s) | 4.0691 | 4.235 | +4.08% |
| TTFT p90 (s) | 12.296 | 11.529 | -6.24% |
| TTFT p95 (s) | 19.242 | 19.107 | -0.70% |
| ITL mean (s) | 0.01438 | 0.01485 | +3.27% |
| 实际平均输入 tokens | 1.216e+05 | 1.2149e+05 | -0.09% |
| 实际平均输出 tokens | 975.02 | 948.13 | -2.76% |
| AIPerf GPU cache hit（聚合诊断值） | 0.99272 | 0.99287 | +0.02% |
| AIPerf Host cache hit（聚合诊断值） | 0.01451 | 0.02531 | +74.43% |
| 请求级 device hit rate | 0.94619 | 0.9369 | -0.98% |
| 请求级 host hit rate | 0.015545 | 0.025102 | +61.48% |
| 请求级 miss rate | 0.03826 | 0.037994 | -0.70% |
| prefill/queue_ms mean | 1991.1 | 1894.5 | -4.86% |
| prefill/queue_ms p50 | 1.6189 | 1.5497 | -4.27% |
| prefill/queue_ms p90 | 4580.8 | 4356.4 | -4.90% |
| prefill/queue_ms p99 | 32830 | 30834 | -6.08% |
| prefill/forward_envelope_ms mean | 2546.9 | 2548.8 | +0.08% |
| prefill/forward_envelope_ms p50 | 1875.1 | 1875.3 | +0.01% |
| prefill/forward_envelope_ms p90 | 3917.6 | 3765.4 | -3.89% |
| prefill/forward_envelope_ms p99 | 15783 | 15755 | -0.18% |
| decode/alloc_wait_ms mean | 13.646 | 3.1929 | -76.60% |
| decode/alloc_wait_ms p50 | 0.55965 | 1.1896 | +112.56% |
| decode/alloc_wait_ms p90 | 0.83921 | 2.3568 | +180.84% |
| decode/alloc_wait_ms p99 | 1.401 | 4.5878 | +227.47% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
