# 相同 trace turn 配对比较

共同身份 9665；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9091。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 120511.0508 | 120510.9153 | -0.00% |
| output_tokens | 975.0906 | 975.0906 | +0.00% |
| device_tokens | 113960.8809 | 113887.9331 | -0.06% |
| host_tokens | 566.4257 | 963.2524 | +70.06% |
| miss_tokens | 5983.7443 | 5659.7297 | -5.41% |
| ttft_ms | 9933.2688 | 7685.3763 | -22.63% |
| prefill_queue_ms | 4910.5571 | 2760.5518 | -43.78% |
| prefill_bootstrap_ms | 60.1809 | 60.9370 | +1.26% |
| prefill_forward_envelope_ms | 3044.4875 | 2915.7289 | -4.23% |
| prefill_transfer_tail_ms | 905.8180 | 899.3354 | -0.72% |
| decode_queue_ms | 0.1351 | 0.1322 | -2.10% |
| decode_bootstrap_ms | 0.1201 | 0.1226 | +2.11% |
| decode_alloc_wait_ms | 27.6450 | 24.3239 | -12.01% |
| decode_transfer_wait_ms | 8672.4606 | 6370.7661 | -26.54% |
| decode_generation_ms | 13507.9973 | 13049.4695 | -3.39% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
