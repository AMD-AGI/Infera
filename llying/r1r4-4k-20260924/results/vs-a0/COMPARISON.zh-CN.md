# C80 R1+R4 与 historical-A0 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | historical-A0 | R1+R4 | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9727 | 7167 | -26.32% |
| Total tokens/s/GPU | 20166 | 13958 | -30.79% |
| Output tokens/s/GPU | 161.66 | 115.87 | -28.32% |
| 输出吞吐 (token/s) | 2586.5 | 1853.9 | -28.32% |
| 输入吞吐 (token/s，含命中) | 3.2007e+05 | 2.2148e+05 | -30.80% |
| TTFT mean (s) | 9.8598 | 26.93 | +173.13% |
| TTFT p50 (s) | 5.5083 | 17.593 | +219.39% |
| TTFT p90 (s) | 22.25 | 60.776 | +173.15% |
| TTFT p95 (s) | 32.999 | 86.86 | +163.22% |
| ITL mean (s) | 0.01414 | 0.01244 | -12.02% |
| 实际平均输入 tokens | 1.1943e+05 | 1.1211e+05 | -6.12% |
| 实际平均输出 tokens | 965.11 | 938.48 | -2.76% |
| AIPerf GPU cache hit（聚合诊断值） | 0.82803 | 0.72228 | -12.77% |
| AIPerf Host cache hit（聚合诊断值） | 0.00471 | 0.01418 | +201.06% |
| 请求级 device hit rate | 0.94469 | 0.87372 | -7.51% |
| 请求级 host hit rate | 0.0053943 | 0.017032 | +215.74% |
| 请求级 miss rate | 0.049919 | 0.10924 | +118.84% |
| prefill/queue_ms mean | 4857.1 | 19098 | +293.19% |
| prefill/queue_ms p50 | 2.9447 | 9828.9 | +333688.23% |
| prefill/queue_ms p90 | 15632 | 51111 | +226.97% |
| prefill/queue_ms p99 | 50780 | 1.1363e+05 | +123.77% |
| prefill/forward_envelope_ms mean | 3038.2 | 5955.3 | +96.02% |
| prefill/forward_envelope_ms p50 | 2153.3 | 3406.9 | +58.22% |
| prefill/forward_envelope_ms p90 | 4530 | 10250 | +126.27% |
| prefill/forward_envelope_ms p99 | 19753 | 51078 | +158.59% |
| decode/alloc_wait_ms mean | 25.893 | 103.23 | +298.68% |
| decode/alloc_wait_ms p50 | 0.57071 | 0.55245 | -3.20% |
| decode/alloc_wait_ms p90 | 0.86233 | 0.82604 | -4.21% |
| decode/alloc_wait_ms p99 | 1.5667 | 34.561 | +2106.01% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
