# R1–R4 代码审查与修正

2026-09-24。审查对象为初版 `8cb2914e`，并直接修正已确认的问题。编译和测试位于 `dccs-1334-slurm`，并行度 4；没有使用 GPU 计算节点。

## 结论

原实现确有需要修正的正确性和实验隔离问题，不能只根据初版测试通过就认为完成验收。以下问题现已修复并加入回归检查。原评分公式、默认关闭策略及旧缓存目录保留。

| 问题 | 影响 | 修正 |
|---|---|---|
| 评分时反复读取同一候选状态 | 一次排序可能混用不同时间的状态，日志与实际选目标也可能不一致 | 一次读取候选缓存和负载快照，使用同一份数据排序和记录 |
| 需求锁内查询缓存并写日志 | 延长串行区间，日志开销可能影响请求派发 | 锁内仅需求快照、选择和预订；其他工作移到锁外 |
| R4 shadow 因持有 reservation 而提前读取非流式 P 响应体 | shadow 不仅观察，还改变 I/O 顺序和完成时机 | 只有明确 R1 completion 或 R4 on 才提前 drain；shadow 将账本随原响应体读取路径保留 |
| 以单一链 hash 集合管理多个引擎 hash 的别名 | 删除一个别名会误删另一份有效缓存 | 按层引用计数，重复 store/remove 幂等；hash 身份变化使旧分层状态失效 |
| 非字符串 medium 被误认为旧式 GPU 事件 | 错把未知层当作 GPU 命中 | 缺省/nil 与非法类型分开，非法类型记为 unknown |
| NATS 流错误与回放期间仍可能使用旧/历史分层命中 | 重连和追赶期间可能选到已经不存在的缓存 | 流错误立即失效，回放追平后才开放分层命中 |
| R1 completion 在 round-robin 配置下显示开启却未生效 | 配置与实际行为不一致 | 启动校验明确拒绝该组合 |
| 输入模板已确认不一致仍使用估算长度 | 将已知不可靠的 token 数加入新账本 | 对对应 worker 标记 unknown，保守退回当前控制策略 |
| R3/R4 同时旁路只有一个 suggested 标记 | 无法辨别建议来自哪个模型 | 分别记录 tier_suggested 与 demand_suggested |
| profile 名称错误前已重置环境 | 拼写错误会留下意外配置 | 先验证名称，再写环境变量 |

## 简化

统一候选排序与 tie-break，合并重复缓存读取，删除仅供测试使用的 reservation 取出接口，缩短触及代码中的长注释。生产路径仍在同一份开发目录，功能通过开关组合；未为“独立”新增多个实现副本。

新增测试验证 snapshot/tie-break、配置生效、shadow 非流式时序、模板失败降级、重复层事件、别名删除、非法 medium、hash 身份变化以及回放阶段的命中隔离。

## NATS 原测试的修正

实际 broker 补跑首次出现一项历史断言失败：原测试要求“只由 bucket 快照填充的目录，永不被空快照清除”。开发前已有 `seed_rank_view` 实现明确允许更新 snapshot-only view；只有通过 rooted live events 建立的 view 才不被 bucket 覆盖。此次没有为通过测试而改回生产逻辑。

测试现分别覆盖这两个分支：snapshot-only view 可清空；live-event view 不被空 bucket 覆盖。测试 worker ID 使用唯一标识，避免重复运行时被旧 broker 数据污染。首次失败输出保留在 [review-nats-initial.txt](validation/review-nats-initial.txt)。

## 验证与边界

常规回归加临时 NATS broker 测试合计 **321 项通过**：274 单元、25 HTTP、4 ZMQ、14 模板探测、4 NATS。常规命令中的 NATS ignored 条目已通过单独 `--ignored` 命令执行；不能重复计数。

[常规测试](validation/review-tests.txt)、[NATS 测试](validation/review-nats-tests.txt)、[构建记录](validation/review-build.txt)、[受测源码哈希](validation/review-manifest.json)。

本次验证的是 Router 代码与本机 mock/消息服务行为。R2 仍是输入需求估计，R4 仍是有效工作量估计，不等于后端物理 KV 或实际剩余执行时间。实际 Unified Radix 事件完整性、真实取消回收、Router CPU/内存开销及重启 P/D 的性能 A/B 仍待 GPU 机器验收。没有把本地测试通过当作性能收益。
