# C80 R1+R4 与 historical-G0-R1 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | historical-G0-R1 | R1+R4 | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 10033 | 7167 | -28.57% |
| Total tokens/s/GPU | 21264 | 13958 | -34.36% |
| Output tokens/s/GPU | 168.7 | 115.87 | -31.32% |
| 输出吞吐 (token/s) | 2699.2 | 1853.9 | -31.32% |
| 输入吞吐 (token/s，含命中) | 3.3753e+05 | 2.2148e+05 | -34.38% |
| TTFT mean (s) | 7.5792 | 26.93 | +255.32% |
| TTFT p50 (s) | 4.5482 | 17.593 | +286.81% |
| TTFT p90 (s) | 15.606 | 60.776 | +289.44% |
| TTFT p95 (s) | 23.828 | 86.86 | +264.53% |
| ITL mean (s) | 0.01348 | 0.01244 | -7.72% |
| 实际平均输入 tokens | 1.2211e+05 | 1.1211e+05 | -8.18% |
| 实际平均输出 tokens | 976.49 | 938.48 | -3.89% |
| AIPerf GPU cache hit（聚合诊断值） | 0.93384 | 0.72228 | -22.65% |
| AIPerf Host cache hit（聚合诊断值） | 0.00758 | 0.01418 | +87.07% |
| 请求级 device hit rate | 0.94587 | 0.87372 | -7.63% |
| 请求级 host hit rate | 0.0086252 | 0.017032 | +97.47% |
| 请求级 miss rate | 0.045509 | 0.10924 | +140.05% |
| prefill/queue_ms mean | 2676.9 | 19098 | +613.43% |
| prefill/queue_ms p50 | 1.7923 | 9828.9 | +548310.55% |
| prefill/queue_ms p90 | 7556.1 | 51111 | +576.42% |
| prefill/queue_ms p99 | 39965 | 1.1363e+05 | +184.33% |
| prefill/forward_envelope_ms mean | 2880.3 | 5955.3 | +106.76% |
| prefill/forward_envelope_ms p50 | 2037.8 | 3406.9 | +67.19% |
| prefill/forward_envelope_ms p90 | 4236.6 | 10250 | +141.94% |
| prefill/forward_envelope_ms p99 | 19702 | 51078 | +159.25% |
| decode/alloc_wait_ms mean | 23.029 | 103.23 | +348.27% |
| decode/alloc_wait_ms p50 | 0.58658 | 0.55245 | -5.82% |
| decode/alloc_wait_ms p90 | 0.90081 | 0.82604 | -8.30% |
| decode/alloc_wait_ms p99 | 1.5628 | 34.561 | +2111.51% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
