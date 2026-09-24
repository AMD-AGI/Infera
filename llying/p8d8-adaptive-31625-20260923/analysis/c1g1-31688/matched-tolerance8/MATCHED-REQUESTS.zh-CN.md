# 相同 trace turn 配对比较

共同身份 9703；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 8962。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 123839.0637 | 123839.0860 | +0.00% |
| output_tokens | 965.5243 | 965.5243 | +0.00% |
| device_tokens | 117881.2729 | 118105.1944 | +0.19% |
| host_tokens | 188.3937 | 316.5008 | +68.00% |
| miss_tokens | 5769.3971 | 5417.3909 | -6.10% |
| ttft_ms | 10378.6459 | 7218.3064 | -30.45% |
| prefill_queue_ms | 5425.3196 | 2393.0074 | -55.89% |
| prefill_bootstrap_ms | 108.9049 | 57.5368 | -47.17% |
| prefill_forward_envelope_ms | 3016.8460 | 2899.7240 | -3.88% |
| prefill_transfer_tail_ms | 911.7641 | 907.5208 | -0.47% |
| decode_queue_ms | 0.1314 | 0.1302 | -0.92% |
| decode_bootstrap_ms | 0.1149 | 0.1178 | +2.50% |
| decode_alloc_wait_ms | 62.4874 | 9.5819 | -84.67% |
| decode_transfer_wait_ms | 9123.4591 | 5977.9992 | -34.48% |
| decode_generation_ms | 13408.6369 | 13207.0769 | -1.50% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
