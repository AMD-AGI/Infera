# 相同 trace turn 配对比较

共同身份 10126；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 8995。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 122110.0266 | 122110.2280 | +0.00% |
| output_tokens | 929.0091 | 929.0091 | +0.00% |
| device_tokens | 114948.2690 | 115544.6466 | +0.52% |
| host_tokens | 2620.5421 | 2040.7640 | -22.12% |
| miss_tokens | 4541.2155 | 4524.8175 | -0.36% |
| ttft_ms | 6526.8704 | 4656.4486 | -28.66% |
| prefill_queue_ms | 1914.6949 | 1000.4012 | -47.75% |
| prefill_bootstrap_ms | 48.8659 | 40.7574 | -16.59% |
| prefill_forward_envelope_ms | 2532.6673 | 1593.9371 | -37.06% |
| prefill_transfer_tail_ms | 874.2845 | 598.7922 | -31.51% |
| decode_queue_ms | 0.1162 | 0.1113 | -4.25% |
| decode_bootstrap_ms | 0.1267 | 0.1301 | +2.68% |
| decode_alloc_wait_ms | 3.4212 | 1.8315 | -46.47% |
| decode_transfer_wait_ms | 5170.9588 | 3169.2064 | -38.71% |
| decode_generation_ms | 13650.2277 | 13459.9786 | -1.39% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
