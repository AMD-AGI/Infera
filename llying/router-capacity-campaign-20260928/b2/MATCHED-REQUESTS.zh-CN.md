# 相同 trace turn 配对比较

共同身份 10148；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9453。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 121138.9736 | 121139.3302 | +0.00% |
| output_tokens | 974.5075 | 974.5075 | +0.00% |
| device_tokens | 114495.9323 | 114388.9068 | -0.09% |
| host_tokens | 1904.9766 | 2084.8983 | +9.44% |
| miss_tokens | 4738.0646 | 4665.5250 | -1.53% |
| ttft_ms | 6557.3996 | 6080.4976 | -7.27% |
| prefill_queue_ms | 2015.0458 | 1566.9609 | -22.24% |
| prefill_bootstrap_ms | 80.4397 | 59.0195 | -26.63% |
| prefill_forward_envelope_ms | 2564.5664 | 2411.8270 | -5.96% |
| prefill_transfer_tail_ms | 860.6925 | 866.5476 | +0.68% |
| decode_queue_ms | 0.1295 | 0.1115 | -13.87% |
| decode_bootstrap_ms | 0.1211 | 0.1257 | +3.84% |
| decode_alloc_wait_ms | 13.8888 | 23.5049 | +69.24% |
| decode_transfer_wait_ms | 5280.9453 | 4735.0830 | -10.34% |
| decode_generation_ms | 13713.0328 | 14150.7704 | +3.19% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
