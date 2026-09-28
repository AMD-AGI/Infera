# 相同 trace turn 配对比较

共同身份 10014；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9381。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 120795.0027 | 120794.9341 | -0.00% |
| output_tokens | 977.3213 | 977.3213 | +0.00% |
| device_tokens | 114205.4655 | 114333.8953 | +0.11% |
| host_tokens | 1859.1519 | 1333.2753 | -28.29% |
| miss_tokens | 4730.3852 | 5127.7635 | +8.40% |
| ttft_ms | 6596.8636 | 7166.4557 | +8.63% |
| prefill_queue_ms | 2052.8314 | 2476.6473 | +20.65% |
| prefill_bootstrap_ms | 81.7368 | 60.9237 | -25.46% |
| prefill_forward_envelope_ms | 2571.4139 | 2678.0185 | +4.15% |
| prefill_transfer_tail_ms | 862.2959 | 858.4535 | -0.45% |
| decode_queue_ms | 0.1296 | 0.1316 | +1.54% |
| decode_bootstrap_ms | 0.1209 | 0.1203 | -0.55% |
| decode_alloc_wait_ms | 14.2256 | 27.2916 | +91.85% |
| decode_transfer_wait_ms | 5323.3092 | 5850.7452 | +9.91% |
| decode_generation_ms | 13756.3194 | 13776.6998 | +0.15% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
