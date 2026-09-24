# C80 C1G1 与 C1 自动比较

以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。

| 指标 | C1 | C1G1 | 变化 |
|---|---:|---:|---:|
| 成功请求数 | 9774 | 10171 | +4.06% |
| Total tokens/s/GPU | 20702 | 21490 | +3.81% |
| Output tokens/s/GPU | 161.86 | 172.07 | +6.31% |
| 输出吞吐 (token/s) | 2589.7 | 2753.2 | +6.31% |
| 输入吞吐 (token/s，含命中) | 3.2865e+05 | 3.4109e+05 | +3.79% |
| TTFT mean (s) | 10.37 | 7.1125 | -31.41% |
| TTFT p50 (s) | 5.5257 | 4.4485 | -19.49% |
| TTFT p90 (s) | 24.017 | 14.185 | -40.94% |
| TTFT p95 (s) | 34.37 | 22.001 | -35.99% |
| ITL mean (s) | 0.01449 | 0.01406 | -2.97% |
| 实际平均输入 tokens | 1.2187e+05 | 1.2171e+05 | -0.13% |
| 实际平均输出 tokens | 960.33 | 982.36 | +2.29% |
| AIPerf GPU cache hit（聚合诊断值） | 0.83674 | 0.93993 | +12.33% |
| AIPerf Host cache hit（聚合诊断值） | 0.00132 | 0.00263 | +99.24% |
| 请求级 device hit rate | 0.95209 | 0.95273 | +0.07% |
| 请求级 host hit rate | 0.0015505 | 0.0029849 | +92.51% |
| 请求级 miss rate | 0.046364 | 0.044288 | -4.48% |
| prefill/queue_ms mean | 5457.6 | 2315.3 | -57.58% |
| prefill/queue_ms p50 | 2.988 | 1.6863 | -43.57% |
| prefill/queue_ms p90 | 17653 | 6514.2 | -63.10% |
| prefill/queue_ms p99 | 52672 | 35071 | -33.42% |
| prefill/forward_envelope_ms mean | 2988.1 | 2872.2 | -3.88% |
| prefill/forward_envelope_ms p50 | 2167 | 2059.4 | -4.97% |
| prefill/forward_envelope_ms p90 | 4421.3 | 4217.4 | -4.61% |
| prefill/forward_envelope_ms p99 | 18262 | 19423 | +6.36% |
| decode/alloc_wait_ms mean | 58.681 | 10.246 | -82.54% |
| decode/alloc_wait_ms p50 | 0.55692 | 0.56843 | +2.07% |
| decode/alloc_wait_ms p90 | 0.84744 | 0.85621 | +1.03% |
| decode/alloc_wait_ms p99 | 1.636 | 1.416 | -13.45% |

AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。
详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。
需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。
