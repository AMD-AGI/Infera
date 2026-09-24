# 相同 trace turn 配对比较

共同身份 9542；输入容差≤0 tokens 且≤0.1%、实际输出长度相同、两端关联的配对 2634。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 101293.3910 | 101293.3910 | +0.00% |
| output_tokens | 842.0862 | 842.0862 | +0.00% |
| device_tokens | 95505.6887 | 95845.5642 | +0.36% |
| host_tokens | 755.6082 | 53.8436 | -92.87% |
| miss_tokens | 5032.0942 | 5393.9833 | +7.19% |
| ttft_ms | 9534.9571 | 11080.3385 | +16.21% |
| prefill_queue_ms | 4858.2210 | 6327.5639 | +30.24% |
| prefill_bootstrap_ms | 60.1795 | 99.0097 | +64.52% |
| prefill_forward_envelope_ms | 2834.7611 | 2961.9753 | +4.49% |
| prefill_transfer_tail_ms | 903.3983 | 916.7497 | +1.48% |
| decode_queue_ms | 0.1103 | 0.1055 | -4.32% |
| decode_bootstrap_ms | 0.1196 | 0.1143 | -4.41% |
| decode_alloc_wait_ms | 36.6963 | 59.6941 | +62.67% |
| decode_transfer_wait_ms | 8362.3328 | 9942.9735 | +18.90% |
| decode_generation_ms | 11748.0856 | 11719.0116 | -0.25% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
