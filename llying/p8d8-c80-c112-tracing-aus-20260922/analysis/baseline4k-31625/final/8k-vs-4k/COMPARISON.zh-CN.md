# C80 current-nodes-8K 与 current-nodes-4K 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | current-nodes-4K | current-nodes-8K | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9782 | 9584 | -2.02% |
| Total tokens/s/GPU | 20602 | 19681 | -4.47% |
| Output tokens/s/GPU | 163.04 | 158.56 | -2.74% |
| 输出吞吐 (token/s) | 2608.6 | 2537 | -2.74% |
| 输入吞吐 (token/s，含命中) | 3.2703e+05 | 3.1236e+05 | -4.48% |
| TTFT mean (s) | 10.009 | 11.497 | +14.86% |
| TTFT p50 (s) | 5.7754 | 7.2304 | +25.19% |
| TTFT p90 (s) | 23 | 24.797 | +7.81% |
| TTFT p95 (s) | 33.257 | 35.411 | +6.48% |
| ITL mean (s) | 0.01432 | 0.01374 | -4.05% |
| 实际平均输入 tokens | 1.2134e+05 | 1.1826e+05 | -2.54% |
| 实际平均输出 tokens | 967.9 | 960.51 | -0.76% |
| GPU cache hit rate | 0.94766 | 0.94319 | -0.47% |
| Host cache hit rate | 0.0049 | 0.0065 | +32.65% |
| prefill/queue_ms mean | 4733.4 | 4483.9 | -5.27% |
| prefill/queue_ms p50 | 2.9212 | 2.1357 | -26.89% |
| prefill/queue_ms p90 | 16292 | 15298 | -6.10% |
| prefill/queue_ms p99 | 46170 | 46733 | +1.22% |
| prefill/forward_envelope_ms mean | 3045.9 | 4504.2 | +47.88% |
| prefill/forward_envelope_ms p50 | 2163.1 | 3616.8 | +67.21% |
| prefill/forward_envelope_ms p90 | 4580 | 6460.7 | +41.06% |
| prefill/forward_envelope_ms p99 | 19787 | 19936 | +0.76% |
| decode/alloc_wait_ms mean | 21.937 | 63.453 | +189.26% |
| decode/alloc_wait_ms p50 | 0.56221 | 0.55733 | -0.87% |
| decode/alloc_wait_ms p90 | 0.84401 | 0.83309 | -1.29% |
| decode/alloc_wait_ms p99 | 1.4661 | 34.903 | +2280.71% |

详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
