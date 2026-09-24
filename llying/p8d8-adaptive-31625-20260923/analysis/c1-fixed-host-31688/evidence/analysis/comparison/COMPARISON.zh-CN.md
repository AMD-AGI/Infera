# C80 c1-fixed-host-31688 与 reference 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | reference | c1-fixed-host-31688 | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9727 | 9774 | +0.48% |
| Total tokens/s/GPU | 20166 | 20702 | +2.66% |
| Output tokens/s/GPU | 161.66 | 161.86 | +0.12% |
| 输出吞吐 (token/s) | 2586.5 | 2589.7 | +0.12% |
| 输入吞吐 (token/s，含命中) | 3.2007e+05 | 3.2865e+05 | +2.68% |
| TTFT mean (s) | 9.8598 | 10.37 | +5.17% |
| TTFT p50 (s) | 5.5083 | 5.5257 | +0.32% |
| TTFT p90 (s) | 22.25 | 24.017 | +7.94% |
| TTFT p95 (s) | 32.999 | 34.37 | +4.16% |
| ITL mean (s) | 0.01414 | 0.01449 | +2.48% |
| 实际平均输入 tokens | 1.1943e+05 | 1.2187e+05 | +2.04% |
| 实际平均输出 tokens | 965.11 | 960.33 | -0.50% |
| AIPerf GPU cache hit（聚合诊断值） | 0.82803 | 0.83674 | +1.05% |
| AIPerf Host cache hit（聚合诊断值） | 0.00471 | 0.00132 | -71.97% |
| 请求级 device hit rate | 0.94469 | 0.95209 | +0.78% |
| 请求级 host hit rate | 0.0053943 | 0.0015505 | -71.26% |
| 请求级 miss rate | 0.049919 | 0.046364 | -7.12% |
| prefill/queue_ms mean | 4857.1 | 5457.6 | +12.36% |
| prefill/queue_ms p50 | 2.9447 | 2.988 | +1.47% |
| prefill/queue_ms p90 | 15632 | 17653 | +12.93% |
| prefill/queue_ms p99 | 50780 | 52672 | +3.73% |
| prefill/forward_envelope_ms mean | 3038.2 | 2988.1 | -1.65% |
| prefill/forward_envelope_ms p50 | 2153.3 | 2167 | +0.64% |
| prefill/forward_envelope_ms p90 | 4530 | 4421.3 | -2.40% |
| prefill/forward_envelope_ms p99 | 19753 | 18262 | -7.55% |
| decode/alloc_wait_ms mean | 25.893 | 58.681 | +126.63% |
| decode/alloc_wait_ms p50 | 0.57071 | 0.55692 | -2.42% |
| decode/alloc_wait_ms p90 | 0.86233 | 0.84744 | -1.73% |
| decode/alloc_wait_ms p99 | 1.5667 | 1.636 | +4.42% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
