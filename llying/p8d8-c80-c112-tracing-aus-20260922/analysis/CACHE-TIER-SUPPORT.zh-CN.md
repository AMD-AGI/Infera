# Infera 是否支持 GPU/host 分层路由？Unified Radix 与 Dynamo 的关系

调研日期：2026-09-24。范围：本地 Infera `c85d65d7f1e3d5b2e53012fb966ae231699e6184` 的 Rust/Python KV-aware Router、已有实验镜像源码快照，以及 Dynamo `ff3ac59e83c73e03b98a5d0ec192ec28847130f7`。本次是源码核查，未修改服务或执行性能实验；不泛指 Infera 未来或其他分支版本。

> 开发更新：以下结论针对上述开发前版本。当前分支已新增默认关闭的 Rust R3 分层目录与评分，见 [开发说明](../../router-r234-20260924/README.zh-CN.md)。Python 路径未变；实际 Unified Radix 事件链路和性能仍待机器验收。

## 结论

**当前 Infera 有缓存感知路由，但没有 GPU/host 分层感知路由。** 即使 SGLang 提供了带存储层的事件，当前 Router 也不能分别维护 GPU、host 命中并给它们不同的权重。这个结论已由源码确认，不再只是“待核实”。

| 能力 | 当前 Infera | 核查版本 Dynamo |
|---|---|---|
| 根据前缀缓存选择 worker/DP rank | 支持 | 支持 |
| 接收缓存 Stored/Removed/Cleared 事件 | 支持 | 支持 |
| 分别维护 GPU 与 host 驻留目录 | 当前 KV-aware 路径不支持 | 支持，依赖有效分层事件 |
| 分别给 GPU/host 命中计分 | 不支持，仅一个 hits 信号 | 支持 |
| 自动知道本次 Unified Radix 的真实 host 状态 | 不能据此声称支持 | 也必须验证引擎事件链路 |

## 为什么能确认 Infera 的缺口

Rust 路径：

- `rust/router/src/kv_event.rs:148`：内部 `Event::Stored`/`Removed` 没有 medium 字段。
- 同文件 `:1006` 附近的 `parse_event` 接受带 medium 的消息，但不保存它；map 格式处理同样不保留分层信息。
- 同文件 `:33` 的 `RankViews` 是每 rank 一个 `HashSet<u64>`，没有 GPU/host 维度；`:836` 起插入单一集合，`:892` 起从单一集合删除。
- 同文件 `:1239` 的既有测试明确说明 decoder 忽略尾部 medium；这是现有行为的证据，不是本次新增或运行的测试。
- `rust/router/src/policy.rs:219` 起描述评分为 `-overlap_weight × hits + load`，没有独立 host hits 项。
- `rust/router/src/kv_event_nats.rs:236` 将消息交给同一个 `apply_encoded_batch`；更换为 NATS 传输不补充分层目录。

Python 路径：

- `infera/router/kv_event/events.py:93` 起的 SGLang event schema **有** medium 字段。
- 但 `infera/router/kv_event/client.py:310` 起的 `_handle_event` 和 `:357` 起的 `_on_block_stored` 不按 medium 区分，仍修改同一个 rank view/map。
- `infera/router/policy/kv_event_aware.py:660` 查询这个单一 view。
- NATS client 继承相同的 cache-view 维护逻辑。

因此，准确说法不是“所有 Infera decoder 都不识别 medium”，而是 **Rust 解码丢掉它，Python 解码保留却未用于目录和评分，两者都未实现分层路由。**

一个按代码推演的例子：如果同一 block hash 依次收到 `store(GPU)`、`store(CPU_PINNED)`、`remove(GPU)`，当前单一集合会先插入、再次插入、最终删除；无法表达“GPU 已删除，但 host 副本仍存在”。反过来，host 副本被删而 GPU 仍存在时，也无法独立保留另一层。这不是对本次真实事件流已经发生该序列的声明。

## Unified Radix 是什么

它是 **SGLang 内部的一种前缀缓存管理实现**，对应 `UnifiedRadixCache`。前缀可以想成多条请求共同的“开头”：例如两个请求都有同一段系统提示词，就可以复用这部分已经计算出的 KV。

Radix tree 用树组织共同前缀，便于找到最长可复用部分。Unified 的含义可以理解为：用统一的树和组件机制管理不同模型的缓存状态，例如 full attention、sliding-window attention、Mamba，并协调锁定、淘汰、回读等操作。它不是跨所有 worker 共享显存的意思。

HiCache 则是 SGLang 的分层缓存能力：GPU 放不下的内容可以保留在 host，按配置还可以接外部存储。**Unified Radix 与 HiCache 并不互斥。** 本次镜像的 UnifiedRadixCache 通过 `attach_hybrid_pool_to_unified_cache` 接入相关存储池，并调用 `tree_core.set_hicache_enabled()`。

镜像源码依据：[unified_radix_cache.py](source-evidence/mem_cache/unified_radix_cache.py)，重点为 `:161` 的类、`:177` 起的组件初始化、`:204` 起的 TreeCore、`:457` 起的分层池接入。来源及哈希见 [manifest.json](source-evidence/manifest.json)。

这说明引擎能够管理 GPU/host 缓存，不等于已经证明所有 host 迁移事件都正确发给 Router。当前快照中的顶层类将事件管理委托给 TreeCore；本次未完整提取底层事件实现或采集运行时事件，所以不声称已完成 Unified Radix host 事件完整性验收。

## Dynamo 用的又是什么

Dynamo 不是固定使用另一种 SGLang 缓存。它可以接多个推理后端；当接 SGLang 时，实际 KV 仍由所部署版本、模型和配置选中的 SGLang 缓存实现管理。

核查版本的 Dynamo HiCache 文档以 **SGLang HiRadixCache** 为分层事件示例：GPU 与 CPU_PINNED 的 Stored/Removed 分开报告。这个示例不代表 Dynamo 强制用 HiRadixCache，也不能用来证明我们镜像中的 UnifiedRadixCache 具有完全相同的事件行为。

Dynamo Router 自己也有 radix/prefix 索引，但那是“哪个 worker 的哪一层有哪段缓存”的目录，不存放模型的真实 KV 张量。Dynamo 的 KVBM 是另一项缓存管理能力；不能把它当作所有 SGLang 部署默认替换 Unified Radix 的实现。

固定 Dynamo 源码依据：

- [SGLang publisher](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/components/src/dynamo/sglang/publisher.py#L354)：订阅 SGLang 各 DP rank 的 KV 事件。
- [事件处理](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/lib/llm/src/kv_router/publisher/event_processor.rs#L48)：将存储层带入事件处理。
- [分层索引](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/lib/kv-router/src/indexer/lower_tier_indexers.rs#L273)：按不同层扩展连续前缀查询。
- [评分](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/lib/router-plugins/builtin/src/default/scorer.rs#L171)：device 与 host overlap 分别计入 credit。
- [HiCache 文档](https://github.com/ai-dynamo/dynamo/blob/ff3ac59e83c73e03b98a5d0ec192ec28847130f7/docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/hicache.md)：引擎侧示例及事件时序。

## 对本次优化的意义

已经确认一个可借鉴的具体方向：**给 Infera 补齐分层目录和评分，而不必先更换 SGLang 缓存实现。**

实施前先检查本次 Unified Radix 的 GPU/host Stored、Removed 和回读事件：若已完整发布，就主要改 Router；若事件缺失，还需补引擎发布。Router 应按 worker、rank、tier 维护独立状态，避免一层的删除误删另一层，并避免重复计算两层共有的前缀。

之后先对现有请求离线比较“原选择”和“分层评分选择”，判断有多少真实决策会改变，再开展单变量性能验证。当前证据证明功能缺口存在，但尚未量化它对长尾 miss、TTFT 或吞吐的贡献。
