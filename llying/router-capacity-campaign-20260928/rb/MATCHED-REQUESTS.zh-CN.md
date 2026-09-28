# 相同 trace turn 配对比较

共同身份 10075；输入容差≤8 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 9232。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 121466.7482 | 121466.6443 | -0.00% |
| output_tokens | 963.3333 | 963.3333 | +0.00% |
| device_tokens | 114963.6187 | 114182.7314 | -0.68% |
| host_tokens | 1822.2946 | 2665.3241 | +46.26% |
| miss_tokens | 4680.8348 | 4618.5888 | -1.33% |
| ttft_ms | 6556.3342 | 6564.2132 | +0.12% |
| prefill_queue_ms | 2019.7172 | 1928.0342 | -4.54% |
| prefill_bootstrap_ms | 81.2795 | 48.1068 | -40.81% |
| prefill_forward_envelope_ms | 2560.4058 | 2553.6251 | -0.26% |
| prefill_transfer_tail_ms | 862.3304 | 875.3588 | +1.51% |
| decode_queue_ms | 0.1301 | 0.1154 | -11.28% |
| decode_bootstrap_ms | 0.1210 | 0.1266 | +4.59% |
| decode_alloc_wait_ms | 13.2310 | 3.3621 | -74.59% |
| decode_transfer_wait_ms | 5281.3607 | 5205.8304 | -1.43% |
| decode_generation_ms | 13550.1888 | 14110.6347 | +4.14% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
