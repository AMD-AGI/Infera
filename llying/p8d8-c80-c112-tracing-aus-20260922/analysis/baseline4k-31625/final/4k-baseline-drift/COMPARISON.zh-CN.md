# C80 current-nodes-4K 与 historical-4K 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | historical-4K | current-nodes-4K | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9651 | 9782 | +1.36% |
| Total tokens/s/GPU | 20116 | 20602 | +2.42% |
| Output tokens/s/GPU | 160.52 | 163.04 | +1.57% |
| 输出吞吐 (token/s) | 2568.4 | 2608.6 | +1.57% |
| 输入吞吐 (token/s，含命中) | 3.1928e+05 | 3.2703e+05 | +2.43% |
| TTFT mean (s) | 10.654 | 10.009 | -6.05% |
| TTFT p50 (s) | 5.676 | 5.7754 | +1.75% |
| TTFT p90 (s) | 25.087 | 23 | -8.32% |
| TTFT p95 (s) | 36.654 | 33.257 | -9.27% |
| ITL mean (s) | 0.01418 | 0.01432 | +0.99% |
| 实际平均输入 tokens | 1.1997e+05 | 1.2134e+05 | +1.14% |
| 实际平均输出 tokens | 965.05 | 967.9 | +0.30% |
| GPU cache hit rate | 0.94298 | 0.94766 | +0.50% |
| Host cache hit rate | 0.00597 | 0.0049 | -17.92% |
| prefill/queue_ms mean | 5575.7 | 4733.4 | -15.11% |
| prefill/queue_ms p50 | 3.1866 | 2.9212 | -8.33% |
| prefill/queue_ms p90 | 17619 | 16292 | -7.53% |
| prefill/queue_ms p99 | 54289 | 46170 | -14.96% |
| prefill/forward_envelope_ms mean | 3119.9 | 3045.9 | -2.37% |
| prefill/forward_envelope_ms p50 | 2223.1 | 2163.1 | -2.70% |
| prefill/forward_envelope_ms p90 | 4684.2 | 4580 | -2.22% |
| prefill/forward_envelope_ms p99 | 19963 | 19787 | -0.88% |
| decode/alloc_wait_ms mean | 34.092 | 21.937 | -35.65% |
| decode/alloc_wait_ms p50 | 0.5656 | 0.56221 | -0.60% |
| decode/alloc_wait_ms p90 | 0.85456 | 0.84401 | -1.23% |
| decode/alloc_wait_ms p99 | 1.6578 | 1.4661 | -11.56% |

详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
