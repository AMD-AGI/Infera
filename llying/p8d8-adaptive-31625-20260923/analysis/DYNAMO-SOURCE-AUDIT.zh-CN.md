# Dynamo 当前源码核验（补充离线调研）

2026-09-23 通过公开 GitHub 仓库取得 commit `ff3ac59e83c73e03b98a5d0ec192ec28847130f7`。以下仅陈述已读取的实现，不把上游功能存在等同于本次 AMD/SGLang 后端已兼容或有性能收益。

## 与当前实验最相关的事实

1. [reference scorer](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/lib/kv-router/src/scheduling/selector/reference.rs#L284) 分开处理 device、host、disk 和共享 cache credit；并不是当前 Infera 的单一 `-w*hits+active+recent` 公式。
2. 同一 scorer 会按相对最轻候选的 `active_prefill_tokens` backlog，对 device overlap credit 做有界衰减；随后将 prefill cost、decode cost 和 active request cost 相加。它依赖负载数据的可用性和配置，不能只抄一个权重常数。
3. 代码明确说明通常的 disaggregated Decode 路由通过请求覆盖将 overlap credit 设为0；非通常的 conditional-disagg 路径可以保留 credit。不能说所有 D 路由都固定同一规则，也不能由此证明当前系统的全部 D budget 已被覆盖。
4. scorer 的 DEBUG 候选行带 request_id、worker_type、worker_id/DP rank，以及评分组成。这是当前 Infera 缺少全候选证据时值得借鉴的观测方式，不能据选中 rank 一行日志计算 oracle。
5. [request lifecycle](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/lib/llm/src/kv_router/routing_host/request_guard.rs#L730) 有 `mark_prefill_completed_if_booking` 的单独生命周期处理。它说明上游显式区分 Prefill 完成与整个请求完成；还需沿具体 backend 路径核实何时触发，不能仅凭函数名认定每种后端均正确释放。
6. [worker metrics](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/lib/kv-router/src/protocols.rs#L1100) 提供可选的 active decode blocks / active prefill tokens。字段存在不保证当前引擎已提供数据，也不是引擎真实 admission 成功的保证。

## 对本轮选择的影响

优先构建可复核的容量/缓存对照，随后依据 miss 和服务时间决定是否值得补全候选审计。暂不迁移整个推理框架，不把 cache、路由、生命周期和传输组件一起替换。D 扩容和 D overlap 权重不作为当前 C80 的第一项优化。

本文件是源码研究成果，不是 Dynamo 性能对照或已完成移植的声明。克隆位于 `/tmp/dynamo-source-20260923`，关键文件哈希见 dynamo-source-manifest.json。
