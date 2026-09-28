# 相同 trace turn 配对比较

共同身份 10167；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9399。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 121050.1186 | 121050.0217 | -0.00% |
| output_tokens | 981.4987 | 981.4987 | +0.00% |
| device_tokens | 114479.8893 | 115536.6213 | +0.92% |
| host_tokens | 1899.7902 | 897.8249 | -52.74% |
| miss_tokens | 4670.4391 | 4615.5755 | -1.17% |
| ttft_ms | 6584.5613 | 6296.4848 | -4.38% |
| prefill_queue_ms | 2047.2564 | 1780.8507 | -13.01% |
| prefill_bootstrap_ms | 82.0450 | 44.5113 | -45.75% |
| prefill_forward_envelope_ms | 2562.0613 | 2464.7328 | -3.80% |
| prefill_transfer_tail_ms | 867.1809 | 874.6948 | +0.87% |
| decode_queue_ms | 0.1297 | 0.1149 | -11.35% |
| decode_bootstrap_ms | 0.1208 | 0.1250 | +3.48% |
| decode_alloc_wait_ms | 14.1998 | 10.2285 | -27.97% |
| decode_transfer_wait_ms | 5313.4756 | 5009.7037 | -5.72% |
| decode_generation_ms | 13838.5451 | 13760.5828 | -0.56% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
