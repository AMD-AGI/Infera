# C80 G0 与 A0 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | A0 | G0 | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9727 | 10033 | +3.15% |
| Total tokens/s/GPU | 20166 | 21264 | +5.44% |
| Output tokens/s/GPU | 161.66 | 168.7 | +4.36% |
| 输出吞吐 (token/s) | 2586.5 | 2699.2 | +4.36% |
| 输入吞吐 (token/s，含命中) | 3.2007e+05 | 3.3753e+05 | +5.45% |
| TTFT mean (s) | 9.8598 | 7.5792 | -23.13% |
| TTFT p50 (s) | 5.5083 | 4.5482 | -17.43% |
| TTFT p90 (s) | 22.25 | 15.606 | -29.86% |
| TTFT p95 (s) | 32.999 | 23.828 | -27.79% |
| ITL mean (s) | 0.01414 | 0.01348 | -4.67% |
| 实际平均输入 tokens | 1.1943e+05 | 1.2211e+05 | +2.24% |
| 实际平均输出 tokens | 965.11 | 976.49 | +1.18% |
| AIPerf GPU cache hit（聚合诊断值） | 0.82803 | 0.93384 | +12.78% |
| AIPerf Host cache hit（聚合诊断值） | 0.00471 | 0.00758 | +60.93% |
| 请求级 device hit rate | 0.94469 | 0.94587 | +0.12% |
| 请求级 host hit rate | 0.0053943 | 0.0086252 | +59.90% |
| 请求级 miss rate | 0.049919 | 0.045509 | -8.84% |
| prefill/queue_ms mean | 4857.1 | 2676.9 | -44.89% |
| prefill/queue_ms p50 | 2.9447 | 1.7923 | -39.14% |
| prefill/queue_ms p90 | 15632 | 7556.1 | -51.66% |
| prefill/queue_ms p99 | 50780 | 39965 | -21.30% |
| prefill/forward_envelope_ms mean | 3038.2 | 2880.3 | -5.20% |
| prefill/forward_envelope_ms p50 | 2153.3 | 2037.8 | -5.36% |
| prefill/forward_envelope_ms p90 | 4530 | 4236.6 | -6.48% |
| prefill/forward_envelope_ms p99 | 19753 | 19702 | -0.25% |
| decode/alloc_wait_ms mean | 25.893 | 23.029 | -11.06% |
| decode/alloc_wait_ms p50 | 0.57071 | 0.58658 | +2.78% |
| decode/alloc_wait_ms p90 | 0.86233 | 0.90081 | +4.46% |
| decode/alloc_wait_ms p99 | 1.5667 | 1.5628 | -0.25% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
