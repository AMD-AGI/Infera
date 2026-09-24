# PR #173：改动与 P8D8 性能优化的关系

分析日期：2026-09-24 UTC。静态审阅 PR 完整 diff，重点核查启动、Rust/Python PD 请求生命周期与本地实验补丁；未运行 PR 测试或性能实验，未合入代码。

## 1. 结论与版本

[PR #173](https://github.com/AMD-AGI/Infera/pull/173) 的核心是 **PD 启动可靠性、异常请求回收和流式诊断**。标题中的 “prefill wait for decode” 指启动阶段：P/D 并行加载，P 在找到兼容 D、真实 KV 探针通过之后才注册。它不意味着每个请求都先完成 D admission 再开始 P，也没有改变 D 提前分配输入 KV 的调度机制。

GitHub API 显示已于 2026-09-23 06:59:39 UTC 合并，32 个文件，新增 5925 行、删除 350 行，包含大量测试。PR 描述仍是模板，因此以下以代码为依据。

- base：`625a950b109371aaf8ebedfdc05757b57ac32eab`
- head：`5caa437f9348b2577b33f0d231759e998d8cbadb`
- merge：`0d2cbe9ff47a579b6d14a2c92e5ca671a385aa3e`
- 本地实验仓库 HEAD：`c85d65d7`；本地生产源码仍是旧的共用 guard 路径，G0 优化保存在实验补丁中，不能仅凭当前日期认定已有 PR 的改动。

## 2. 做了哪些工作

| 部分 | 实际行为 | 对本次工作的意义 |
|---|---|---|
| Prefill 启动 barrier | 默认在 P 开启；跳过假 bootstrap warmup；等匹配 model/engine/protocol 的 D，再做真实 `/generate` KV 探针，最后注册 P | 减少健康接口正常但 PD 无法传输的启动假阳性 |
| 发现与退出 | K8s 清理过期注册、要求 Pod Ready、按 deployment 隔离；etcd 依赖独立 prefix；等待时处理退出信号和引擎死亡 | 防止错误 peer 或残留进程；Slurm 环境主要借鉴探针及退出监督 |
| 请求回收 | P/D 使用一致的 SGLang rid；客户端断开、流不完整等情况下向两端发送 `/abort_request`；覆盖 HTTP/NATS 和多样本 rid | 降低已被客户端放弃的请求继续占据 inflight/KV 的风险 |
| 正常结束判定 | PD SSE 按 `[DONE]` 或 Responses 的 `response.completed` 判断结束，避免把截断 EOF 当成功 | 提升请求完整性，利于解释无效输出和尾部故障 |
| 流诊断 | Rust 新增首字节前/流中停滞日志，携带 worker、rid、path；默认阈值 240/60 秒，仅告警 | 为长尾定位提供关联信息，不直接缩短排队 |
| 超时和重试 | HTTP body idle timeout 默认关闭，NATS 维持 900 秒；Decode HTTP 自动重试限于连接错误 | 避免正常长排队被过早截断或已送达请求被重试 |
| SSE 与 API | 加 `X-Accel-Buffering: no`；修复 Anthropic SSE 最后一行无换行丢失及残缺尾事件处理 | 有 nginx/Anthropic 路径时有用；不能泛化为 GPU 吞吐收益 |
| 镜像平台 | GLM-5.3 Dockerfile 新增 MI325X/gfx942 分支，保留 MI355X/gfx950；集中平台相关 Mooncake/libionic/patch 参数 | 本次 MI355X GLM-5.2 实验不宜借此整体换镜像 |
| 其他 | 加 anyio 依赖、更新文档和测试；projection tuning 部分仅格式调整 | 不包含新的自动调参算法 |

真实 KV 探针向 P/D 并发发送 4 个输入 token、生成 1 token 的请求；检查 `max(P DP size, D DP size)` 个 rank 对，失败时双端 abort、换 room 重试。它覆盖双方每个 rank，但不是所有 P×D 组合，更不证明大 KV 传输的带宽或稳定性。probe 成功主要根据请求是否报错/HTTP 状态，不能替代正式性能验收。

P+Mooncake 路径默认补充 `SGLANG_ENABLE_FAILED_SESSION_PROBE=1` 和 `SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=5`，采用 set-if-unset。是否有效仍取决于本次镜像中的 SGLang 实现，不能只设置变量就宣称修复失败 session。

## 3. 两个不能误读的生命周期

### 3.1 PR 没有实现 G0 的 P guard 提前释放

PR head 的 Rust `dispatch` 仍用一个 `ActiveGuard` 同时记录 P/D，streaming 由 Decode 的 `GuardedStream` 持有。异常 abort 与正常 P 负载账本释放是不同的改动。

本次 G0 的 [router-prefill-guard.patch](../../p8d8-adaptive-31625-20260923/patches/router-prefill-guard.patch) 在 HTTP streaming 路径拆出 P guard，交给 Prefill drain task，在 P 响应体读完后释放。这才对应已经观测到的 P 负载记账、队列和缓存选择变化。

已有 [A0/G0 结果](../../p8d8-adaptive-31625-20260923/analysis/RESULTS.zh-CN.md)：total tokens/s/GPU +5.44%，TTFT mean -23.13%，P queue mean -44.89%。它们属于 G0 的实验结果，不能归给 PR #173；已有时序/热身限制也仍成立。

若以后移植 PR 的回收逻辑，需将 G0 的 P guard 所有权合并进新的 drain/controller 结构，核查成功、客户端断开、D 打开失败、P task 被取消各路径恰好释放一次。两者都改 `disagg.rs` 的相同区域，不能假设补丁可以直接机械叠加。

### 3.2 Rust 的 300 秒不是整个请求的上限

`watch_prefill_after_decode` 先等 `StreamEnd`：

- `Incomplete`：取消 P drain，并尝试双端 abort。
- `Complete`：此时才开始最多 300 秒等待剩余 P drain；设为 0 时禁用这个 cap。
- D 仍未结束：尚未进入这个 300 秒 timeout。

因此不能说它保证所有卡住的 P 在发出 300 秒内都被清理。HTTP idle timeout 默认也是 0；240/60 秒的告警不主动结束请求。代码注释/参数帮助比真实控制流更笼统，应以控制流为准。

此外，发送 abort 成功不等于物理 KV 已回收；需要引擎完成/abort 事件和容量指标确认。Rust HTTP Prefill drain 对 `resp.bytes()` 的错误仍未单独分类，回收机制也不能理解为完整的传输故障检测。

## 4. 哪些值得应用，按什么顺序

1. **先借鉴诊断和验收。** 保持既有 workload、镜像、路由参数，用真实 PD 探针检查 rank 可用性；将 rid、P/D task 结束、abort 原因与引擎 inflight/KV 指标关联。探针在正式缓存重置/热身之前进行，避免悄悄改变对照条件。长尾首字节仍须拆成 P queue、forward、KV transfer、D admission，不能仅由 240 秒告警归因。
2. **有孤儿请求证据时移植回收逻辑。** 先确认实际 API 接受 `rid`/`request_id`/`x-override-rid`，且 abort 命中 scheduler 中的请求；再核查 client disconnect、截断 SSE、n>1、D 连接失败。统计残留请求数、回收延迟、错误率和有效输出长度，避免把提前终止请求造成的表面提速当收益。不能声称已证明本次长尾 miss 源自孤儿请求。
3. **保留 G0 为独立路由优化。** 它与本 PR 互补；不同时叠加新 abort 策略、容量和 guard 开关来解释净收益。沿用已有基线，不追加重复 A。
4. **维持 cache/准入研究方向。** 本 PR 没有改路由 overlap 评分、GPU/host cache 目录、P KV 容量或 D 物理 token admission，不能替代全候选缓存命中审计、D 输入需求对账和固定 host 的 P GPU KV 扩容验证。

工作区另有 [C1 启动计划](../../p8d8-adaptive-31625-20260923/analysis/c1-fixed-host-31688/PLAN.zh-CN.md)，记录 02:21 UTC 开始启动、目标 P GPU KV 340 万且 host 不变。此处未查询实时进度或中断该实验；不应为纳入 PR 而中途升级其 router/镜像。

启动 barrier 不能解释或修复发生在引擎启动中的 HIP stream 原生段错误。节点驱动差异仍是需单独验证的线索。Dockerfile 原本已有 libionic ABI 检查，PR 主要把它参数化到不同 GPU 平台；不能把它描述为本 PR 首次解决本次节点驱动问题。

## 5. 固定版本源码入口

以下均固定到 PR head，避免 main 后续变化：

- [decode_barrier.py](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/infera/engine/decode_barrier.py)：匹配、真实 probe、恢复默认值。
- [SGLang main](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/infera/engine/sglang/__main__.py)：加载、barrier、注册和退出监督次序。
- [Rust disagg.rs](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/rust/router/src/disagg.rs)：共用 guard、drain controller、双端 abort、连接重试。
- [Rust proxy.rs](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/rust/router/src/proxy.rs)：SSE 结束判定、stall watch、禁用代理缓冲。
- [Rust config.rs](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/rust/router/src/config.rs)：超时和告警默认值。
- [Rust protocol.rs](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/rust/router/src/protocol.rs)：rid 和 Responses request_id。
- [Python disagg.py](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/infera/router/disagg.py)：Python 路由端对应实现与 header 差异。
- [GLM-5.3 Dockerfile](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/deploy/docker/Dockerfile.sglang.glm53)：平台分支。
- [Rust functional tests](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/rust/router/tests/functional.rs) 和 [barrier tests](https://github.com/AMD-AGI/Infera/blob/5caa437f9348b2577b33f0d231759e998d8cbadb/tests/unit/engine/test_decode_barrier.py)：PR 提供的测试，不代表本次已运行或在 MI355X 上验收。
