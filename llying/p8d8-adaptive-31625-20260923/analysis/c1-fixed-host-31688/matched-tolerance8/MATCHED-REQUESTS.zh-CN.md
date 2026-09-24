# 相同 trace turn 配对比较

共同身份 9542；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 8897。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 120327.0950 | 120326.9672 | -0.00% |
| output_tokens | 962.2016 | 962.2016 | +0.00% |
| device_tokens | 113738.1787 | 114366.3023 | +0.55% |
| host_tokens | 585.8550 | 197.3519 | -66.31% |
| miss_tokens | 6003.0613 | 5763.3129 | -3.99% |
| ttft_ms | 9898.7493 | 10500.8220 | +6.08% |
| prefill_queue_ms | 4886.1153 | 5566.9850 | +13.93% |
| prefill_bootstrap_ms | 58.4585 | 107.8628 | +84.51% |
| prefill_forward_envelope_ms | 3051.2036 | 3013.4896 | -1.24% |
| prefill_transfer_tail_ms | 905.0411 | 911.7479 | +0.74% |
| decode_queue_ms | 0.1357 | 0.1310 | -3.48% |
| decode_bootstrap_ms | 0.1201 | 0.1147 | -4.47% |
| decode_alloc_wait_ms | 26.8287 | 62.7971 | +134.07% |
| decode_transfer_wait_ms | 8650.8986 | 9256.9349 | +7.01% |
| decode_generation_ms | 13336.0156 | 13365.1800 | +0.22% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
