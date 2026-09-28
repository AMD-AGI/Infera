# B2：只开启D radix，完整C80结果

正式窗口：2026-09-28 15:43:23～16:43:23 UTC。10,295条正式完成，11条边界取消，0正式请求错误。导出中另有1条warmup InvalidInferenceResultError（无有效内容）；runner的warmup错误计数为0，两种口径均保留。取消credit的10秒收尾超时触发了强制phase完成，不能隐去这一边界。

当前为legacy P评分+R1+P亲和；D radix on，D会话亲和off，D KV事件不参与Router评分，模拟接受长度3.61。P计算内核和后端不变。

| 指标 | 原P亲和基线 | B2 | 变化 |
|---|---:|---:|---:|
| 输出tokens/s/GPU | 172.33 | 170.53 | −1.04% |
| 输出tokens/s | 2757.26 | 2728.56 | −1.04% |
| 平均TTFT | 6.515s | 6.095s | −6.45% |
| TTFT p95 | 19.242s | 18.566s | −3.51% |
| 平均ITL | 14.38ms | 14.64ms | +1.81% |
| P平均排队 | 1.991s | 1.581s | −20.62% |
| P miss/input | 3.826% | 3.799% | 基本相近 |
| D平均分配等待 | 13.65ms | 22.21ms | +8.56ms |

D本地prefix事件覆盖全部10,295条正式完成请求，无重复分配事件；3,653条有本地复用。总输入1,251,166,186 tokens，复用前缀295,468,608 tokens（约23.62%），协议需传输的逻辑tokens约955,697,578。它是逻辑payload计数，不是直接测量的RDMA bytes。

9,453对匹配请求中，TTFT −7.27%、P queue −22.24%、D等KV区间 −10.34%；D生成区间 +3.19%。D等KV包括等待P，不能把全部改善归因为纯网络传输。

**结论：这一组改善TTFT，但未显示吞吐收益。继续B3验证D亲和是否提高局部性，不在此时把radix或D亲和宣称为已获吞吐加速。** P/D配对、room与引擎身份检查通过。导出器的counter-reset警告不等同于重启：scheduler PID和startup记录未变；主要结论使用客户端和逐请求数据。

数据：[比较](COMPARISON.zh-CN.md)、[匹配请求](MATCHED-REQUESTS.zh-CN.md)、[D本地复用](decode-local-prefix.json)、[质量检查](REVIEW.json)、[负载分析](balance.json)。单次历史对照、完成请求交集与边界取消的限制仍适用。
