# C80 8K 与历史 4K 自动比较

以下为已完成运行的自动汇总。Prefill 节点发生变化，且闭环轨迹进度可能不同；单次历史对比不能独立证明 chunk-size 收益。

| 指标 | 历史 4K | 当前 8K | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9651 | 9584 | -0.69% |
| 输出吞吐 (token/s) | 2568.4 | 2537 | -1.22% |
| 输入吞吐 (token/s，含命中) | 3.1928e+05 | 3.1236e+05 | -2.17% |
| TTFT mean (s) | 10.654 | 11.497 | +7.91% |
| TTFT p50 (s) | 5.676 | 7.2304 | +27.39% |
| TTFT p90 (s) | 25.087 | 24.797 | -1.16% |
| TTFT p95 (s) | 36.654 | 35.411 | -3.39% |
| ITL mean (s) | 0.01418 | 0.01374 | -3.10% |
| 实际平均输入 tokens | 1.1997e+05 | 1.1826e+05 | -1.43% |
| 实际平均输出 tokens | 965.05 | 960.51 | -0.47% |
| GPU cache hit rate | 0.94298 | 0.94319 | +0.02% |
| Host cache hit rate | 0.00597 | 0.0065 | +8.88% |
| prefill/queue_ms mean | 5575.7 | 4483.9 | -19.58% |
| prefill/queue_ms p50 | 3.1866 | 2.1357 | -32.98% |
| prefill/queue_ms p90 | 17619 | 15298 | -13.17% |
| prefill/queue_ms p99 | 54289 | 46733 | -13.92% |
| prefill/forward_envelope_ms mean | 3119.9 | 4504.2 | +44.37% |
| prefill/forward_envelope_ms p50 | 2223.1 | 3616.8 | +62.69% |
| prefill/forward_envelope_ms p90 | 4684.2 | 6460.7 | +37.93% |
| prefill/forward_envelope_ms p99 | 19963 | 19936 | -0.13% |
| decode/alloc_wait_ms mean | 34.092 | 63.453 | +86.12% |
| decode/alloc_wait_ms p50 | 0.5656 | 0.55733 | -1.46% |
| decode/alloc_wait_ms p90 | 0.85456 | 0.83309 | -2.51% |
| decode/alloc_wait_ms p99 | 1.6578 | 34.903 | +2005.43% |

详细匹配分层、请求关联覆盖率和错误计数见 `chunk8k-comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量判断是否应在同两节点回测 4K。
