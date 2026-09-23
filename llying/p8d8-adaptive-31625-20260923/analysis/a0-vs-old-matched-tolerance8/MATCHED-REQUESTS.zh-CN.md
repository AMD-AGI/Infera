# 相同 trace turn 配对比较

共同身份 9574；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 8903。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 119764.0885 | 119764.1801 | +0.00% |
| output_tokens | 964.2740 | 964.2740 | +0.00% |
| device_tokens | 113324.1954 | 113177.1960 | -0.13% |
| host_tokens | 645.4418 | 581.1470 | -9.96% |
| miss_tokens | 5794.4513 | 6005.8370 | +3.65% |
| ttft_ms | 10095.4826 | 9923.1069 | -1.71% |
| prefill_queue_ms | 4827.2558 | 4896.5248 | +1.43% |
| prefill_bootstrap_ms | 51.8029 | 59.8197 | +15.48% |
| prefill_forward_envelope_ms | 3053.8469 | 3054.8645 | +0.03% |
| prefill_transfer_tail_ms | 929.9862 | 905.3665 | -2.65% |
| decode_queue_ms | 0.1353 | 0.1355 | +0.10% |
| decode_bootstrap_ms | 0.1185 | 0.1201 | +1.30% |
| decode_alloc_wait_ms | 23.4928 | 27.3084 | +16.24% |
| decode_transfer_wait_ms | 8716.5852 | 8668.1082 | -0.56% |
| decode_generation_ms | 13292.5088 | 13361.5084 | +0.52% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
