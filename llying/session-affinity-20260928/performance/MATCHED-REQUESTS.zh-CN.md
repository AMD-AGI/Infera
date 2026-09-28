# 相同 trace turn 配对比较

共同身份 9980；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9394。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 122727.2004 | 122727.3129 | +0.00% |
| output_tokens | 980.8635 | 980.8635 | +0.00% |
| device_tokens | 116090.1818 | 116166.5608 | +0.07% |
| host_tokens | 1000.2367 | 1829.9613 | +82.95% |
| miss_tokens | 5636.7819 | 4730.7908 | -16.07% |
| ttft_ms | 7677.0837 | 6626.9747 | -13.68% |
| prefill_queue_ms | 2744.9039 | 2064.6846 | -24.78% |
| prefill_bootstrap_ms | 60.6067 | 82.4815 | +36.09% |
| prefill_forward_envelope_ms | 2904.3366 | 2578.0842 | -11.23% |
| prefill_transfer_tail_ms | 897.9027 | 866.1828 | -3.53% |
| decode_queue_ms | 0.1319 | 0.1298 | -1.66% |
| decode_bootstrap_ms | 0.1230 | 0.1208 | -1.80% |
| decode_alloc_wait_ms | 24.5479 | 14.2088 | -42.12% |
| decode_transfer_wait_ms | 6347.4295 | 5349.0669 | -15.73% |
| decode_generation_ms | 13105.4242 | 13823.8314 | +5.48% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
