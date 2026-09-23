# 相同 trace turn 配对比较

共同身份 9574；输入/实际输出长度相同且两端关联的配对 3800。

| 指标 | 对照 mean | 处理 mean | 变化 |
|---|---:|---:|---:|
| input_tokens | 103697.9311 | 103697.9311 | +0.00% |
| output_tokens | 901.3845 | 901.3845 | +0.00% |
| device_tokens | 97788.9516 | 97442.7116 | -0.35% |
| host_tokens | 478.8211 | 531.0484 | +10.91% |
| miss_tokens | 5430.1584 | 5724.1711 | +5.41% |
| ttft_ms | 10447.0839 | 9897.2605 | -5.26% |
| prefill_queue_ms | 5303.2458 | 4921.9079 | -7.19% |
| prefill_bootstrap_ms | 51.7276 | 61.8852 | +19.64% |
| prefill_forward_envelope_ms | 2996.5515 | 3030.0507 | +1.12% |
| prefill_transfer_tail_ms | 941.8873 | 925.6595 | -1.72% |
| decode_queue_ms | 0.1513 | 0.1526 | +0.86% |
| decode_bootstrap_ms | 0.1164 | 0.1184 | +1.67% |
| decode_alloc_wait_ms | 24.6152 | 24.4431 | -0.70% |
| decode_transfer_wait_ms | 9117.9817 | 8675.3925 | -4.85% |
| decode_generation_ms | 12519.8028 | 12580.4992 | +0.48% |

这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。
