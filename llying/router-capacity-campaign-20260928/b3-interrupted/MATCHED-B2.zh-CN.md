# 相同 trace turn 配对比较

共同身份 10143；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9307。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 121347.4337 | 121346.9873 | -0.00% |
| output_tokens | 972.6909 | 972.6909 | +0.00% |
| device_tokens | 114648.8931 | 115840.8664 | +1.04% |
| host_tokens | 2023.9321 | 864.9386 | -57.26% |
| miss_tokens | 4674.6085 | 4641.1822 | -0.72% |
| ttft_ms | 6115.2696 | 6299.5363 | +3.01% |
| prefill_queue_ms | 1598.0191 | 1779.1552 | +11.34% |
| prefill_bootstrap_ms | 59.3464 | 44.1877 | -25.54% |
| prefill_forward_envelope_ms | 2424.0575 | 2467.9871 | +1.81% |
| prefill_transfer_tail_ms | 870.3304 | 871.4619 | +0.13% |
| decode_queue_ms | 0.1119 | 0.1152 | +2.91% |
| decode_bootstrap_ms | 0.1255 | 0.1251 | -0.31% |
| decode_alloc_wait_ms | 23.9704 | 10.2053 | -57.43% |
| decode_transfer_wait_ms | 4780.2879 | 5008.2238 | +4.77% |
| decode_generation_ms | 14180.5836 | 13634.4035 | -3.85% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
