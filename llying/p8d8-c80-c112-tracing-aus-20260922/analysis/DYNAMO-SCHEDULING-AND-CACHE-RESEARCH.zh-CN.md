# NVIDIA Dynamo 调度与缓存调研：C80/C112 联合优化方案

日期：2026-09-23。

**状态：本地证据分析与 debug／验证方案已完成；Dynamo 当前版本的在线源码取证受阻，本文是待补充源码核验的调研初稿，不是已完成的 Dynamo 实现审计。**

本次访问 GitHub 时，默认沙箱报 DNS 解析失败；申请下载公开仓库的网络权限时，自动审批服务因 `codex-auto-review` 部署不存在而失败，下载未执行。没有绕过审批，也没有取得 Dynamo commit。因此下文严格区分“Dynamo 架构背景／待核实问题”“本次已验证事实”和“为当前系统提出的设计”。不编造 Dynamo 的公式、参数默认值、版本支持或性能数据。

相关资料：

- [原 C80/C112 分析报告](RECOVERY-AND-FINDINGS.zh-CN.md)。
- [调度链详细问答](SCHEDULING-QA.zh-CN.md)。
- [新增离线证据](joint-cache-evidence.json)与[复算脚本](../scripts/analyze_joint_cache.py)。
- [已完成 8K chunk 实验复核](chunk8k-31625/final/REVIEW.zh-CN.md)。

## 目录

1. [核心判断与建议顺序](#conclusion)
2. [Dynamo 应如何分层理解](#dynamo)
3. [本次系统与 Dynamo 调研的对照](#comparison)
4. [新增实验数据审计](#data)
5. [先把长尾 miss 分成六类](#miss)
6. [最小 debug 数据闭环](#debug)
7. [可实施的调度与 cache 联合方案](#solutions)
8. [验证矩阵与收益判据](#experiments)
9. [源码取证清单与剩余工作](#sources)

<a id="conclusion"></a>
## 1. 核心判断与建议顺序

**联合调整调度和缓存值得优先投入，但应先证明长尾 miss 中有多少可以避免，再选择策略。** 现有证据支持“长尾 miss 增加 P 服务需求并引发连锁等待”，尚未证明这些 miss 主要由路由选错位置或缓存容量不足导致。

需要把两个问题分开：

- 定位瓶颈：少量请求贡献大量 miss，横跨多轮 Prefill，影响请求供给和等待。
- 找到可消除的工作：这些请求在到达时，是否本来就有正确、可复用、可及时取回的 prefix KV？

如果多数是从未计算过的新前缀，提高 cache affinity 不会消除它们的首次计算。如果其他 rank 已有完整前缀，路由修正可能直接省掉多轮 Prefill；如果同 rank 的前缀刚被驱逐，保留／容量策略才是关键；如果 prefix 已在 host，却未被正确识别或回读，扩 GPU 容量未必是第一步。

建议顺序：

1. 核对实际 token 前缀、缓存事件和命中记账的正确性。
2. 为 miss≥32K 请求记录所有候选 P rank 的可兑现命中与预计等待，先做 shadow 审计。
3. 根据 miss 分类，只实施一个最有证据的改动：纠正缓存视图、改善前缀放置，或减少不必要驱逐。
4. 加入短时间尺度的真实 P 剩余工作与 D 分配压力反馈，验证联合效果。
5. 若 D 提前预分配仍显著限制系统，再研究两阶段 credit／reservation，不能直接延后现有 D prealloc 调用。

按用户决定，**不安排“给新短请求保留份额／改变 chunk 续跑公平性”的实验**。优先减少真实 miss 和服务需求，而不是重新分配等待。

本地已有 8K 实验：长 miss 组 chunk 数由 18.80 降至 10.15，但未观察到总体吞吐收益。Prefill 节点不同，不能把差异全部归因于 chunk；这至少说明“轮数减少必然提速”不成立，不应把尚未执行的 4K→8K 当作唯一首选下一步。

<a id="dynamo"></a>
## 2. Dynamo 应如何分层理解

### 2.1 架构背景与当前核验边界

Dynamo 的公开架构背景涉及分布式推理编排、KV-aware routing、P/D 解耦、KV 传输及资源规划。其生态还涉及 NIXL 和分层 KV 管理。这些名称可用来组织调研，但**本次未读取其当前官方文档或源码，不能据此断言每个后端、版本或部署方式都同时具备并默认开启这些能力**。

Dynamo 并非一个单独决定所有 GPU batch 的调度器。调查时至少要分开以下层次：

| 层次 | 决定的问题 | 必须查证的具体内容 |
|---|---|---|
| 请求路由 | 请求交给哪个 worker／rank | 候选粒度、prefix overlap、活动负载、反馈来源、实际 cost、tie-break |
| P/D 编排 | 请求何时进入 P 或 D，谁触发远端 Prefill | P-first／D-first／其他路径、bootstrap、异步队列、取消与失败处理 |
| 引擎 batch 调度 | 下一轮运行哪些请求、多少 token | 具体后端的 chunk、预算、抢占、decode 优先级；不能仅看 Dynamo router |
| KV 位置与生命周期 | KV 在 GPU、host 或其他层，何时保留／回读／驱逐 | 分层索引、锁、可用性、事件、逐后端支持 |
| KV 传输 | 如何将可用数据送到目标 | NIXL 或后端 connector 的提交、完成、同步、传输拓扑与成本 |
| 资源规划 | 给 P、D 分配多少 worker／副本 | 扩缩容周期、服务目标、负载预测、缓存冷启动成本 |

固定 P8D8 场景下，最相关的是前五层。增加实例的 Planner 策略不能直接替代单请求 miss 和 rank 调度优化。

### 2.2 Prefill 调度要问什么

对 KV-aware router，不能只确认“支持 cache-aware”，需要核实：

1. 命中来源是真实 KV 事件、历史请求推断，还是两者结合？
2. 匹配粒度是整 worker、DP rank，还是具体 cache partition？
3. 只看到 GPU KV，还是也能看到 host 等慢层？不同层命中是否赋予不同成本？
4. 已在计算但尚未就绪的前缀，会不会被当成命中？如何等待并处理生产者失败？
5. 负载是请求数、输入 tokens、剩余 miss、运行 token 量，还是预计完成时间？
6. P 工作完成后何时从负载中扣除，是否一直等到 D 输出结束？
7. 选中目标后，后端 admission 失败会留队、重试还是改路由？

这些问题直接对应当前系统已发现的风险。**本文不将 `-overlap_weight × hits + active + recent` 宣称为 Dynamo 的公式；它是当前 Infera 仓库里的公式。**

### 2.3 Decode 调度要问什么

Decode 必须区分“路由到哪个 D”与“所选 D 当前是否能分配输入 KV”。即使 Dynamo 有更多遥测，也不能用一个路由评分替代后端实际分配检查。

要查证的内容包括：

- 是否按已分配 KV、待分配输入需求、活动请求数、输出增长估计做负载判断。
- D 不复用 prefix 时是否禁用无实际收益的 cache overlap，或自动退回其他策略。
- 完整输入 KV 何时预分配；等待 P 的请求计入何种负载与容量指标。
- 是否有跨 P/D 的 admission credit 或限流；适用于哪些 connector／backend。
- 请求分配后能否安全改选 D；哪些阶段已有接收地址或进行中的写入，不能直接改目标。

**不能仅因 Dynamo 支持 disaggregation，就认定它已经解决“D 提前占满输入 KV 等 P”的问题。** 需要沿真实请求路径核查。

### 2.4 分层 KV 管理与传输能解决什么

分层 KV 管理的目标通常是扩大可复用工作集、管理 GPU 与较慢介质之间的数据移动；传输组件负责数据可达与搬运。它们的价值取决于重用时间、传输成本和正确性，而非层数越多越好。

需核实的边界：缓存目录知道 host 有数据，不代表 router 会把它计入命中；传输库能搬数据，不代表系统会自动从任意其他 P rank 拉取前缀；支持某种后端，不代表支持本次 attention DP=8 和 Unified Radix 的实际 layout。

NVIDIA 环境的结果也不能直接当成当前 AMD／ionic／Mooncake 环境的预测。可以借鉴路由、记账和协议设计；组件替换必须单独验证 accelerator、内存注册、KV layout 与后端集成。

<a id="comparison"></a>
## 3. 本次系统与 Dynamo 调研的对照

| 问题 | 当前实验已知事实 | 值得从 Dynamo 核查／借鉴的内容 |
|---|---|---|
| P 命中视图 | Router 基于 KV 事件和前缀 hash 评分，P 实际用 Unified Radix；全候选快照未保存 | 缓存目录一致性、分层位置信息、缓存事件丢失恢复 |
| P 负载 | active 受 P/D 共用 guard 生命周期影响，recent 为派单历史 | 分角色负载生命周期、剩余计算反馈、完成时间估计 |
| P 内部调度 | 每 rank 4K chunk，旧 chunk 先续跑 | 必须落到所用后端看 batch 逻辑，而非套用 router 文档 |
| D 路由 | kv-aware 配置，但无 block metadata；21,151 次 D pick 的 hits/active/request blocks 全为零 | 无 cache 信息时的 fallback、KV 容量遥测和 reservation |
| D 准入 | 本地检查池和 KV 预算，提前分配输入空间等 P | P/D 交接时序、背压、credit 是否真实实现 |
| P cache | GPU resident 接近满，但主要可驱逐；host 也接近满 | 工作集留存、复用价值估计、慢层读取与预取 |
| 传输 | 缓存 prefix 和新算 KV 都可能需要发给 D | 从实际传输量／时间建模，避免只按 miss 估传输 |
| 工作负载 | AgentX 多轮、fan-out、闭环轨迹；不同 CONC 改变执行进度 | 会话局部性、多分支重复前缀与热点复制 |

“采用 Dynamo”与“修正当前实现”是两个不同范围的工程选项。目前没有证据要求先迁移整个推理栈；先得到可避免 miss 的账本，才能判断哪项能力值得引入。

<a id="data"></a>
## 4. 新增实验数据审计

### 4.1 长尾计算量与 D 驻留不应混为一谈

本轮从原始 joined 请求明细复算，口径为有效 profiling cohort，尾组是实际 miss≥32,768：

| 指标 | C80 | C112 |
|---|---:|---:|
| 有效请求 | 9,651 | 9,321 |
| 长 miss 请求 | 332 | 450 |
| 长 miss 组占全部 miss | 41.58% | 57.03% |
| 长 miss 组平均 P forward | 19.13 秒 | 30.94 秒 |
| 全体 D 等待输入 KV 驻留代理 | 310.13 万 token | 863.40 万 token |
| 长 miss 组自身 D 等待驻留代理 | 30.33 万 token | 109.28 万 token |
| 长 miss 组自身占 D 等待驻留代理 | 9.78% | 12.66% |

驻留代理为 `Σ(input_tokens × D_transfer_wait_seconds) / 3600`，不是瞬时页数；未裁剪请求尾部到发送窗口，不含输出增长和页对齐。

**少量长 miss 请求贡献大量计算，但 D 等待 KV 的占用主要在其他请求。** 这与“重尾 P 服务拖慢其他高命中、长输入请求”的机制相容；它不是逐请求阻塞因果证明，还需同 rank 时序与 batch 关联。优化时应同时观察被拖慢的其他请求，而非只观察长 miss 组自己的 D 内存。

这也细化了“不做短请求保留份额”这一判断：短 miss 请求可能有非常长的总输入，占用很多 D KV；不能简单用 miss 长短代表内存大小。当前仍按用户决定不做公平性实验，重点是消除多余计算。

### 4.2 同一 source position 不保证同一实际输入

使用 `(source_trace_id, source_outer_idx, source_inner_idx, source_kind)` 对齐：

- C80/C112 各自 source position 无重复、无缺失。
- 两档共同位置 7,227 个。
- 只有 1,905 对实际 input token 数相同。
- 这 1,905 对中，1,742 对选择了不同 P rank。
- 相同长度子集的总输入均为 251,332,701 token，但 miss 从 11,263,837 增至 16,100,381。

**不能因此宣称“换 rank 导致 miss 增加”。** 相同位置和长度都不证明 token ID 一样；两档顺序运行、缓存历史不同，rank 改变本身也可能是合理的路由响应。这个审计的价值是：下一次必须记录实际 token-prefix digest，才能做可靠的配对和 replay。

### 4.3 已有 8K 实验对下一步的影响

本地 [8K 复核](chunk8k-31625/final/REVIEW.zh-CN.md) 记录：

- 输出吞吐 -1.22%，完成请求 -0.69%。
- P queue 均值 -19.58%，P forward 均值 +44.37%。
- TTFT mean +7.91%，p50 +27.39%。
- 22 个共同 context/miss/host 分组统一加权 forward +48.48%。
- 长 miss 组 chunk 数由 18.80 降至 10.15，chunk envelope 均值由 1.018 秒升至 2.042 秒。

这不是严格同节点因果实验，也不证明 8K 永远无益；但不能继续用“轮数少一半”预测吞吐会提升。原复核提到的同节点 4K 对照是否已完成，本轮未核查其后续状态；本文不启动或重复该实验。

<a id="miss"></a>
## 5. 先把长尾 miss 分成六类

分类应按请求的前缀区间进行，一个请求可以同时含多类 miss。不得把整条请求粗暴归到唯一原因，或对同一段重复记账。

| 类别 | 定义 | 证明需要什么 | 优先处理 |
|---|---|---|---|
| 首次访问／内容变化 | 到达前从未产生过相同有效前缀 | 实际 token 链、历史已完成 KV；warmup 之前缺失时标 unknown | 接受必要计算；优化前缀稳定性须保持语义 |
| 放置／路由损失 | 选中 P 无缓存，其他 P 有可兑现前缀 | 到达时所有候选 rank 的缓存状态、锁和可用性 | cache-aware placement、有限会话亲和 |
| 保留／容量损失 | 历史曾计算，但需要时已不在可用缓存层 | 插入、驱逐、host 副本、重用间隔和容量 | 留存策略、限制污染、工作集容量 |
| 目录／匹配错误 | 实际有 KV，但事件索引、hash 或元数据不一致 | engine 与 router token/block 对照、事件序列与快照 | 先修正确性，不先调权重 |
| 慢层不可兑现 | host 有数据但无法及时回读、被门限拒绝或未接入 | host match、load submit/finish、拒绝原因、bytes | 回读路径、预取、容量和 I/O 成本 |
| 并发重复计算 | 另一请求正计算相同前缀，尚未发布为可用缓存 | prefix digest、producer 生命周期、batch/完成事件 | 有界等待、前缀生产者协调，谨慎避免重复算 |

只有拥有完整到达前历史，才能称为“compulsory miss”。只有在单 rank 相同访问序列下进行保留策略对照，才适合把损失细分为严格容量 miss 或替换策略 miss。有限采样无法支撑时统一写“未知／待区分”。

### 5.1 三个可计算的 oracle

1. **实际可兑现的全 rank oracle**：到达时哪一个可接纳 P 的 GPU／host 前缀能以最低预计成本使用？用于估计路由机会，不允许访问未来缓存。
2. **历史无限容量 oracle**：只保存该前缀首次完成后的缓存历史，不驱逐，估计历史重用上限；必须包括 warmup，或明确截断历史。
3. **当前容量的不同保留策略 replay**：使用相同访问顺序、页粒度、锁、层级和传输限制，比较保留价值，而不是简单累加所有命中潜力。

Oracle 1 的 GPU 命中增量可写为 `max_rank GPU_prefix_hit - chosen_GPU_prefix_hit`，但它仅是 token 机会。真实可优化量还要考虑该 rank 是否能准入、排队成本和事件滞后。Host hit 需按“最终可复用总前缀”统一计算，不能与 GPU hit 重复累加。

离线换路由会改变后续缓存和排队历史。固定原轨迹的 shadow 只估计当时机会，不能直接宣称整小时可节省多少总 token 或吞吐；需要闭环 replay／实际对照确认。

<a id="debug"></a>
## 6. 最小 debug 数据闭环

### 6.1 关联键与时间

优先复用现有 RID、source position 和 DP rank。新增各组件本地单调时钟、事件序号、router decision ID 及接收时刻；不使用不同机器 wall clock 的亚秒差值推导因果。事件应能从 router pick 关联至 P admission、chunk、KV transfer 和 D prealloc。

在 engine 实际 tokenize 完成后生成 token-prefix 链式 digest，与 Router 的 tokenizer、模板版本、block size、模型/cache 身份一致。只对长度或渲染文本做 hash 不足以验证实际 token 相同；模型版本、位置编码等会影响 KV 有效性。

### 6.2 需要新增的最小字段

| 位置 | 必要记录 | 目的 |
|---|---|---|
| Router 每次 pick | 全候选 P/D rank、命中前缀长度／层、完整评分分量、有效权重、候选是否可接纳、索引事件序号与年龄 | 判断是否有更好且可用的候选；当前只记选中 rank 不够 |
| P 实际匹配 | 实际 input digest、GPU/host/storage 连续前缀、最终承诺复用长度 | 检查 router 命中是否兑现 |
| P admission | 候选 miss/context、预算前后值、选中 extend 范围、停止原因、是否已加入 batch | 区分空间不足、chunk 用完和回读问题 |
| P 每轮执行 | batch ID、各 rank context/miss、chunk 范围、CPU 提交与 GPU event 时间 | 分离 kernel、同步、调度间隙 |
| Unified cache | insert/evict/load/writeback 的 node/block digest、层、bytes、锁、原因、submit/finish | 判定缓存丢在哪里、回读是否值得 |
| D prealloc | input、完整两门槛、free、增长预留、slots、waiting-for-P token 总量 | 不能只记录 required 门槛一 |
| 请求收尾 | P 最后计算、P transfer 完成、D ready、D completion、guard 释放 | 修正负载生命周期和驻留账本 |

缓存事件与 token digest 的覆盖必须足够完整，才能判断某个 prefix “不存在”。可将昂贵的 device profile 限定在短代表窗口；不能采样掉关键 cache 历史后，把未看到当成从未存在。全候选 pick 日志可限于短窗口或异常触发，但需记录覆盖范围。

本次旧 host 插桩在 hiradix，而实际用 Unified Radix。新增 host 日志必须落在实际 `_load_back_transfers / init_load_back / loading_check` 路径，不能重复使用不生效的事件。

### 6.3 先做四个一致性检查

1. Router 认为命中 H 个 GPU blocks，P 实际 GPU hit 是否接近 `H × block_size`？差异需排除对齐／边界处理，再查事件滞后和驱逐。
2. Router 看不到某前缀时，其他 rank engine snapshot 是否真的也没有？快照要带事件游标，避免时间不一致。
3. cache 已报告 load 完成后，是否兑现了预期 prefix，后续 miss 是否相应下降？有 bytes 不等于被请求实际复用。
4. `used + evictable + free = capacity` 是否保持，锁状态能否解释在途传输和回读？

若发现 hash／事件正确性问题，先修它，不继续用权重 sweep 掩盖错误。

### 6.4 建议的 debug 输出

每个 miss≥32K 请求生成一行：源位置、实际 digest、总输入、GPU/host hit、miss、P rank、最佳可行候选、可避免 token 区间、等待、chunk 数、host bytes、D token-seconds、分类及证据覆盖度。

然后按“可避免 miss tokens”与“预计可节省 P 服务时间”排序。二者都应保留，因为长 context 下每 token 成本不同。不要按 TTFT 最大直接排序后就将所有尾延迟归为计算 miss。

<a id="solutions"></a>
## 7. 可实施的调度与 cache 联合方案

以下公式与策略是**针对当前系统提出的设计，不是 Dynamo 当前实现的复述**。

### 7.1 先修正路由的缓存视图与 P 负载生命周期

若目录有误，修复 rank 标识、block hash、快照／增量衔接和事件丢失恢复。不要用周期性清空热缓存作为正常“修复”手段，那会主动制造冷 miss。

P 的负载应在实际计算／交接完成的合适阶段分别扣除，不能一直沿用到 D 输出结束的共同 guard。建议拆开“剩余计算负载”“在途发送负载”“端到端请求引用”：它们有不同的释放时刻，取消／失败也要恰好清理一次。

计算耗尽后可不再计入 P compute work，但仍在 transfer 的请求会占 KV 与传输资源，不能一律认为 P 负载归零。此改动先 shadow 新旧估计差，不直接改线上评分。

### 7.2 按预计可完成时间选择 P，而非无限追逐命中

一个可校准的评分原型为：

```text
J_P(r) = 预计排队等待(r)
       + 预计未命中部分服务时间(context, miss, batch shape)
       + host 回读／其他缓存获取对关键路径的增量
       + P→D 传输对关键路径的增量
       + 缓存状态不确定性惩罚
```

各项统一为时间单位，并处理重叠：不能把已经包含在历史服务模型中的回读或通信再加一遍，也不能把所有传输串行相加。用实际测量拟合关键路径，而非假定 miss/峰值 token/s 就是服务时间。

`预计排队等待` 优先使用当前 rank 剩余 chunk/context 工作与最近真实完成速率，结合 rank 间协同；不只用请求数量。校准时看预测误差分布，在数据不足时保守回退，避免不稳定策略频繁摆动。

对两个候选，缓存较好的 A 应当优先，当且仅当其可节省的预计重算／回读时间超过额外排队与传输成本，并且 admission 可行。这样既利用长尾缓存收益，也避免热前缀把某 rank 持续压满。

### 7.3 会话／分支前缀放置：软亲和与有限复制

对 AgentX 多轮轨迹，优先用“实际共享前缀”而非仅 session ID 做 affinity。一个会话可以产生不同分支，多个会话也可能共享系统／工具前缀。

可以先给同一活跃分支维持 P home rank，在 home 预计等待明显过高或缓存已不存在时才允许溢出。阈值应按可节省的服务时间设定，并保留等待上限，避免永久粘住热点。

对高复用公共前缀，可评估少量副本，分摊热点；副本数受容量和实际需求限制。把所有前缀复制到全部 8 个 rank 会缩小有效工作集，可能增加长尾 miss。

需要记录 home 命中收益、溢出造成的新增 miss、热点等待和副本占用。相同 source 在 C80/C112 选了不同 rank，不足以直接证明需要硬绑定。

### 7.4 用复用价值保留缓存，避免一次性长前缀污染

先测 prefix 重用间隔和被驱逐后的重算量，再比较保留策略。候选价值函数可以使用：

```text
保留价值 ≈ 预计近期复用次数 × 每次避免的服务时间 / 占用字节
```

实际 radix 节点共享祖先，成本要算边际新增字节，不能让祖先和每个子节点重复领取整段收益。host 与 GPU 价值也不同，保留在 host 的收益要扣未来回读成本。

优先保护高复用祖先、活跃分支近期会再用的前缀；对一次性长尾 suffix 可降低 GPU 留存优先级或偏向 host。不能仅按前缀长度保留，长且不再用的数据可能挤掉大量高复用前缀。

保护最好是有期限的软优先级，而非无限硬 pin；pin 会把可驱逐缓存变为不可驱逐占用，反而制造 admission 压力。算法先离线 replay，不能在未证明重用时盲目扩容。

### 7.5 有界 host 预取与回读成本决策

若审计确认长尾 prefix 已在 host，且下一步请求可预测，可在不改变主执行顺序的前提下提前准备。Prefetch 有独立 bytes／请求数预算，可取消、可超时，并避免占用全部 GPU 空闲空间。

是否回读取决于“回读对关键路径的成本”与“重算成本”，不能以 host hit 数越多越好。预取未被使用时应计浪费；读写竞争使其他请求变慢时也要纳入净收益。

如果现有 prefix 未及时回读是因为 GPU 预算或锁限制，先查这些条件；增加 host 容量不会自动解决回读门槛。

### 7.6 并发重复前缀：先观察，再考虑 producer 协调

Fan-out 的多个请求可能同时携带相同长前缀。若缓存尚未发布，可能在多个 rank 重复计算。需要记录实际 digest 与各 producer 的开始／完成，才能确认，而非看到同一会话就认定重复。

可考虑：后续请求对同一个正在生成的前缀做有界等待，待安全发布后共享；超过等待阈值允许自行计算。KV 未完整就绪不能当普通 cache hit，生产者失败、取消和不同模型/cache 身份必须处理。

这不是合并多个请求的随机输出，也不是给短请求切片额度；只协调相同前缀的生产和复用。实现复杂度较高，若观测到的重复计算很少，则不做。

### 7.7 D KV-aware 选择与等待 P 的 token credit

本次 D `hits=active_blocks=0`，调 overlap 权重无效。第一步应补每 rank 的实际分配与 pending 输入需求、slots 和增长余量，并复用 D 完整 admission 规则做 shadow 判定。

对 D 不复用前缀的配置，新请求需求近似跟完整输入而不是 P miss 成正比。联合路由应避免把输入很大且预计 P 等待很长的请求继续压到已有大量 waiting-for-P KV 的 rank。

D 容量反馈宜使用“可分配预算 + pending reservation”，而非 free gauge；多个并发 pick 需要原子或有界误差的 credit，避免同时看到同一份空闲空间后超卖。出现容量不足时要定义重试和取消路径，不可依赖路由快照保证。

如果进一步延后 D 的物理输入分配，必须重新设计两阶段协议：先获得可撤销的容量意向／credit，在 P 接近可执行时落实 D 地址与分配，再开始依赖目标地址的发送。现有 optimistic=0 路径需要 D bootstrap，不能直接将 prealloc 往后挪，否则可能互等，也可能损失计算传输重叠。

### 7.8 暂不建议做的事

- 不做已移除的 chunk 公平性实验。
- 不只调大 P overlap weight；不知道错在哪里时可能把队列热点放大。
- 不调 D overlap weight 来解决本次空间偏斜。
- 不把 decode reserve=512 降到 0 来换容量。
- 不因 P resident≈100% 就断言必须扩物理池。
- 不将 Dynamo、NIXL、cache policy、memory fraction 和 chunk 一起替换，无法归因。

<a id="experiments"></a>
## 8. 验证矩阵与收益判据

### 8.1 第一步：无需改调度行为的审计

先采集覆盖 warmup 与代表性稳定窗口的 prefix/cache 历史和短窗口全候选 pick，统计六类 miss 的 token 占比与时间成本。诊断记录要单独评估开销，不能用高开销 tracing 的结果直接做吞吐承诺。

当窗口较短、初始缓存状态未知时，只报告窗口内可验证的路由机会与重复计算，首次访问／驱逐原因保留 unknown。

### 8.2 按证据选择实验，不全面扫参数

| 触发证据 | 单项实验 | 预期机制 | 停止／否定信号 |
|---|---|---|---|
| Router/engine prefix 经常不一致 | 修事件／hash／rank 视图 | 命中预测与实际一致，长尾 miss 下降 | 修复只改日志，实际 miss 不变 |
| 其他可行 P 常有更多有效前缀 | 软 affinity 或新评分之一 | 避免 miss 和 chunk 服务时间 | 命中提高但队列、完成率更差 |
| 近期重用前缀反复被驱逐 | 仅改保留策略或仅改 P 容量 | 重算量下降且工作集成本可解释 | GPU 占用增加但重算不降 |
| host 命中存在但回读晚／未兑现 | 仅改有界预取或单一回读参数 | 有效回读提前、P 关键路径缩短 | 未使用预取、写回竞争和锁定增长 |
| fan-out 同前缀重复计算占比高 | 有界 producer 协调 | 降低重复 miss 与服务量 | 等待依赖反而扩大尾部 |
| D 阻塞时其他 rank 实际可接纳 | D 容量-aware 路由 | 降 allocation 尾部与等待 token-seconds | 多数替代 rank 也不满足预算 |

保留策略和 P 扩容应分开测；改变 P mem_fraction 会随 hicache_ratio 改变 host 容量，需要固定 host 绝对值。路由模型更新与 guard 记账更新也要分别观察其影响。

### 8.3 工作负载控制

算法机会评估优先 replay 固定实际 token 序列与相同到达顺序，保存模型、tokenizer、chat template、cache identity、warmup 与初始缓存状态。AgentX 闭环业务测试另做一组，报告轨迹进度变化，不能与固定 open-loop replay 混成同一口径。

对有生成历史的请求，要保存实际已构建输入以便 token 级配对。改路由后生成和轨迹进度可能变化，source position 相同不足以控制输入工作量。

选定最有证据的策略后做匹配的 A/B 或 A/B/A，并按可用预算决定是否短试。本文不启动实验，也不将一次结果当成稳定收益。

### 8.4 成功不能只看 hit rate

主要结果应同时包括：

- 完成请求／轨迹速率、输出 token/s，以及输入／输出的实际工作量。
- TTFT p50/p90/p99、P 首次 queue、按 context/miss 分层的服务时间。
- 可避免 miss 总量、miss≥32K 组数量及贡献、重复计算量。
- D waiting-for-P 的 input token-seconds、allocation 阻塞、最热 rank 的真实预算。
- GPU/host 缓存实际占用、驱逐后再访问、有效回读 bytes、未使用预取。
- 错误、取消、retraction、请求永久等待和诊断开销。

逻辑输入 token/s 包含缓存命中，不能作为实际 GPU 计算吞吐。命中率提升但完成率不升，可能是 host/传输/排队的代价抵消了重算收益。

在相同逻辑输入下，C80 miss 比例约 5.0877%，hit 提高 1 个百分点对应约 19.66% 的 miss 工作减少，这是算术敏感性而非可实现承诺。实际服务成本不与 miss 线性，不能直接换算成 24.5% 系统吞吐增长。

<a id="sources"></a>
## 9. 源码取证清单与剩余工作

### 9.1 Dynamo 官方取证入口（本轮未成功访问）

- 官方仓库：<https://github.com/ai-dynamo/dynamo>
- 官方文档入口：<https://docs.nvidia.com/dynamo/latest/>
- NIXL 仓库：<https://github.com/ai-dynamo/nixl>

这些链接是待访问入口，不是已经阅读并支持本文具体实现断言的引文。网络恢复后，先固定 commit 或 release tag，并记录获取时间；不能以浮动 latest 说明版本行为。

### 9.2 网络恢复后的逐项交付

1. 找到 KV router 的实际评分、负载统计、prefix index、事件订阅、调度返回和失败路径，按 commit permalink 引用。
2. 分别追踪相关后端的 P/D 请求入口、rank 选择、D 预分配、远端 prefill 调用、传输完成与资源释放。
3. 验证分层 KV 管理与 router 的信息接口：GPU／host 可见性、inflight 数据是否可路由、驱逐传播和版本限制。
4. 验证 Planner 的 P/D 资源调整与固定拓扑的适用差异。
5. 将第 2、3 节待核实问题替换为“代码事实／不支持／版本限制”，给出与本次 SGLang/Unified Radix/AMD 环境的兼容边界。
6. 依据验证结果判断是借鉴设计、移植单个组件还是需要另建 Dynamo 对照部署；不预设迁移一定优于局部修正。

### 9.3 本轮已经完成的内容

新增 [analyze_joint_cache.py](../scripts/analyze_joint_cache.py) 仅读取原始 joined cohort，校验请求唯一性、有效请求数和非负 miss，记录输入 SHA256，输出 [joint-cache-evidence.json](joint-cache-evidence.json)。复算命令（实验目录下）：

```bash
python3 scripts/analyze_joint_cache.py \
  /perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922 \
  analysis/joint-cache-evidence.json
```

已完成：本地瓶颈证据审计、source position 配对检查、长 miss 与 D 驻留分组、8K 后续材料核对，以及分阶段 debug／优化方案。

未完成：Dynamo 当前实现在线源码核验、全候选缓存机会量化、token 级 replay、新增插桩与优化 A/B。当前不能给出可信的 Dynamo 迁移收益或 cache 策略吞吐百分比。
