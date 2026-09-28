# Infera 对齐 InferenceX Dynamo-SGLang 的分析与实施计划

日期：2026-09-28。目标是 GLM-5.2 GB200/GB300 配方指定的单 Router 行为；不引入 Nginx。本文件是实现计划，尚未修改生产路由代码。独立的 overlap 权重实验配置已准备，等待用户确认后运行。

## 1. 先明确拥塞逃逸：框架有路径，目标配方没有开启相关负载阈值

固定审计版本为 Dynamo `71eb001e17fa73c742f0afe1a6ed96836cb135fd`。这里要区分三种行为：

| 行为 | 固定版本源码 | InferenceX GB200/GB300 目标配方 |
|---|---|---|
| 绑定 worker 不存在、rank 不合法、目标不可用后重选 | 存在，自动会话绑定选择失败可失效并重试；显式指定目标另有规则 | 适用的失效恢复路径 |
| 绑定 worker 达到过载阈值，被判为不合格后重选 | 存在，目标校验/选择可返回过载错误，自动绑定外层可解除绑定并重新选择 | 三项负载阈值均为 `None`，不能认为此机制已启用 |
| 绑定仍有效，仅因别的 rank 分数更低而迁移 | pin 路径不做全候选自由竞争 | 没有证据支持这种主动迁移 |

三项阈值为 `active-decode-blocks-threshold`、`active-prefill-tokens-threshold`、`active-prefill-tokens-threshold-frac`。`LoadThresholdConfig` 明确说明三项全为 None 时禁用该过载判定。队列准入也是另一项独立机制：`router_queue_threshold` 默认 None，不能把“排队”“过载拒绝”和“主动迁移”当成同一项功能。

因此，对用户问题的准确回答是：**Dynamo 支持以过载失效为条件的重新选择，但我们核对的 InferenceX 配方没有开启这组三项过载阈值。不能说其公开成绩已使用了主动拥塞逃逸。** 运行时覆盖项仍需启动日志证明，配方审计不等于逐请求运行审计。

此前“第一版不加入拥塞逃逸”的建议应表达为：目标配方的阈值默认关闭；实现中保留清晰的可选过载处理接口，不能遗漏 Dynamo 框架已有的失效重选能力，也不能擅自为对齐实验开启自定义拥塞阈值。

## 2. 未绑定会话的实际选择逻辑

没有 session ID、首次出现的 session、空闲 TTL 过期或旧绑定失效，都会进入正常选择。先过滤模型、worker/rank 有效性和相关可用性约束，再对候选评分。目标配方未覆盖的参数按固定版本默认值处理；没有证据时不把其他 Dynamo 版本的默认值套入。

### 2.1 评分不是只看缓存命中，也不是只看请求数

用 block 统一单位，在正常命中量不超过输入量、没有共享远端缓存的情况下，可将源码的主公式写为：

```text
score(j) = prefill_scale × max(0, raw_prefill_blocks(j) - cache_credit(j))
           + potential_active_blocks(j)
           + active_request_weight × active_requests(j)

raw_prefill_blocks(j) ≈ (在途有效 Prefill tokens(j) + 本次输入 tokens) / block_size
cache_credit(j) = GPU_weight × GPU命中blocks
                  + host_weight × host延伸命中blocks
                  + disk_weight × disk延伸命中blocks
potential_active_blocks(j) = 已在途活跃blocks(j) + 本次请求新增活跃blocks(j)
```

分数越低越优先；默认温度 0 选择最小值。精确实现需要保留源码中的饱和扣减、block 取整和候选并列处理，不把这段解释性公式直接当作全部实现。

固定版本的相关默认参数：GPU overlap credit=1，host=0.75，disk=0.25，prefill scale=1，active-request weight=0，overlap-credit-decay=0，output-block tracking=false。目标 GB300 recipe 没有显式覆盖这些参数；host/disk 项只有收到对应层级的有效命中信息才有实际作用，不能据默认权重宣称历史运行已经用到了 host 命中。

`potential_active_blocks` 是 Router 的在途账本投影，不是 GPU 当前物理 KV 使用量。新增活跃 blocks 与 cache miss 也不是同一概念：某前缀即便在闲置缓存中存在，本次启用它仍可能增加活跃工作集；与其他在途请求共享的块则可能不重复计入。

### 2.2 普通分离部署的 P 与 D

**P：** 保留输入工作量和缓存抵扣，并使用相应活跃块投影。不是“队列长度减缓存”，也不能简化成现有 R4 的一个未命中 token 账本。当前 R4 没有完整复现上述组合项。

**D：** `build_decode_router_override` 设置 `overlap_score_credit=0`、`assume_kv_reuse=false`、`track_prefill_tokens=false`。因此普通分离部署主要按候选的在途 block 需求选择，不依赖 D radix cache，也不使用 P 的缓存抵扣项。当前新请求的完整输入需求通常对所有同规格候选相同，加入这一项不一定改变本次候选排序；它被记入后，会影响后续请求的选择。

当前 R2 以完整输入 tokens 记账，是方向相近的近似。完整对齐还要核对 block 边界、预登记的原子性、释放时点和输出块追踪配置，不能直接将 R2 重命名为 Dynamo 模式。

Prefill 在途工作量的扣减/估算同样需要对齐实际配置。固定源码支持可选模型估算的衰减，但没有估算时不能凭空假定按 chunk 实时递减，也不应在第一版添加未经校准的速度常数。

### 2.3 绑定后的路径

会话绑定到具体 worker/rank，P 和 D 分别管理。固定目标仍经过有效性/准入检查和正常记账，不是直接绕过 Router。空闲 TTL=3600 秒，活跃租约避免正在处理的会话因定时清理被删除；没有在途请求并空闲超时后才过期。

## 3. Infera 复现范围与模块设计

目标分成两层：**先对齐会话路由语义，再对齐首次选择和负载账本**。两者都有完成条件；仅实现第一层时应称“会话亲和版本”，不能宣称完全复现 Dynamo。

| 模块 | 已有基础 | 计划改动与验收 |
|---|---|---|
| 请求上下文 | `handlers.rs`、`proxy.rs`；部分请求体 cache hints 有 session 字段 | 从 HTTP 接入 `X-Dynamo-Session-ID`，显式传入路由上下文；不把内部路由信息混进 prompt 或影响 token hash。明确 header 与已有 body hints 的优先级，不把 `prompt_cache_key` 自动等同于会话 |
| 会话表 | 尚无目标行为 | 新增独立模块，按模型/会话/角色隔离；保存 worker ID、rank、有效性身份、空闲期限和活跃租约。支持首次选择协调、失效和有界清理 |
| 首次并发请求 | 现有策略有部分原子选择/预约能力 | 同会话首次请求只建立一个有效绑定；其他请求等待该选择完成或接管失败初始化。锁不能跨整个推理请求持有，也不能阻塞其他会话 |
| 固定目标选取 | `RouteTarget` 已能描述 worker/rank | 在 policy 的选择阶段施加 pin；选中目标依然生成正常 `Pick`、blocks 和 reservation。不能先调用普通 pick 记账，再把目标换掉 |
| 生命周期 | `ActiveGuard`、R1 和 reservation | 绑定租约、P/D 工作账本分别管理。P 继续按 R1 在响应体完成时释放，D 完成/取消时释放；取消的初始化不能留下永久 pending；旧失败回调不能删掉已更新的新绑定 |
| 错误恢复 | discovery、breaker、现有 PD 请求重试 | 派单前失效可重选，明确 worker 重启/rank 变化行为；P/D 已启动后遵守现有 bootstrap/重试约束，不对已输出请求随意改目标或重放 |
| 过载处理 | breaker 不是负载阈值检测器 | 将健康故障与负载过载分开。目标配方默认关闭过载阈值；若实现可选阈值，需明确采样新鲜度、单位、回落条件和全体繁忙时的行为 |
| 未绑定 P 评分 | 现有 KV 目录、R3 分层查询、R4 预约 | 新建有明确名称的可选评分模式；按固定源码加入有效 P 工作、缓存抵扣、活跃 block 投影及参数默认值，保留原模式 |
| 未绑定 D 评分 | R2 输入需求账本 | 对齐无缓存复用的 block 需求、原子预登记和释放；明确完整/尾部 block 的处理。默认不添加输出预测 |
| 缓存事件 | Infera 已有 GPU/host 事件接入与旁路证据 | 验证真实 GPU→host、移除、重启事件与前缀连续性；没有信息时显式标明不可用，不能将 unknown 当成真实零命中。SGLang Unified Radix 是引擎能力，不替代 Router 目录正确性 |
| 客户端 | 已有固定 AIPerf 客户端 | 亲和实验启用 session header，并核对同会话稳定性、不同会话隔离性；不要同时改数据集或会话生成规则 |
| 观测 | 已有路由日志与引擎采样 | 计数 pin hit/new/expired/invalid/overloaded、重选和账本残留；抽样记录选择原因。避免 session ID 作为高基数指标标签；观测失败不杀服务 |

建议可选模式为 `off / prefill / both`，空闲 TTL 默认 3600 秒；评分模式单独区分 legacy 与对齐模式。它们是计划接口，尚未成为实际 CLI/env 名称。

保持单 Router、进程内会话表，不增加 Nginx、分布式状态存储或多 Frontend 同步。Router 重启导致重新建立绑定是需要记录的行为。父子会话、显式 target、session-final 等扩展按固定客户端实际发送内容核对后纳入范围，不能猜测父 session header 就意味着自动继承父绑定。

## 4. 实施阶段与完成条件

### 阶段 A：固定行为规格和离线参照

保存目标配方、Dynamo SHA、默认参数、输入 token/block 定义，建立小型候选输入参照。覆盖无缓存/全命中/部分命中/仅 host、多 rank、并列、活跃共享块和输入长度边界。源码期望值和 Infera 结果逐项比较，不以测试只是重复本地公式作为正确性证明。

本阶段把会话与首次选择两个对齐目标写清楚，过载默认关闭；不假定此前没有日志证明的 InferenceX KV 事件路径已被闭环验证。

### 阶段 B：会话亲和，保留 legacy 评分

实现显式上下文、绑定表、固定目标和生命周期。用受控时钟验证 TTL；用并发请求验证单次初始化；覆盖目标下线、取消、请求失败、P 完成但 D 未结束、过期后重选和不同模型同 session ID。

首先给出 `prefill` 和 `both` 的可开关实现，默认 off。验收重点是正确性和未启用时原行为保留，不在此阶段顺便调整评分权重或拥塞策略。

### 阶段 C：未绑定评分和事件能力对齐

实现固定版本的 block 账本、P/D 评分和参数语义；复用已有 R2/R3/R4 的合适基础，而不是强行叠加三个开关。新模式需要独立说明，与旧实验模式的组合关系应明确，避免重复预约或互相覆盖。

加入与参照的候选选择、账本及释放验证。检查真实引擎事件能力；新能力未经验证时不能隐藏成“完整对齐”。可选过载阈值的单元测试不需要 GPU，也不代表目标性能实验会开启该阈值。

### 阶段 D：短时实机验证与性能实验

新开发版本先做关闭 HiCache 的短时 smoke：相同 session 复用 P/D rank、不同 session 正常分配、P-only 与 both 开关、TTL 测试配置、正常完成和取消后无账本泄漏；再验证需要 HiCache 的事件/回读路径。

正式性能按单变量推进：优先 R1＋legacy评分＋P会话亲和，再按证据选择 both 或完整首次评分对齐。P-only 是为了隔离缓存收益，both 才覆盖所研究配方的 P/D 绑定。各轮以历史 G0 或已有有效对照为依据，配置确认后运行；不预先承诺把每种组合都跑一遍。

评价输出吞吐、TTFT、P queue/miss、D 有效并发和 KV 热点，同时统计会话迁移/绑定命中。需要确认收益来自缓存局部性，而非不同负载、客户端或采样口径。

## 5. 现在的两台机器：建议先做 P overlap 20→40 实验

已只读确认 allocation `31999`，qos=batch，节点 `smci355-ccs-aus-n03-33` 与 `smci355-ccs-aus-n10-29`。拟沿用历史做过 P 的 n10-29 作为 P，n03-33 作为 D；尚未检查节点当前进程或显存，也未启动或清理服务。

**值得做一组，但不要期待“必然显著提升”。** 当前 legacy P 评分为：

```text
score = active_distinct_blocks + recent_blocks - weight × cached_prefix_blocks
```

例如一个缓存更好的候选比另一个多 ΔH 个命中块，但负载高 ΔL，权重 20 下需要 `ΔL < 20×ΔH` 才偏向它；权重 40 把这个容忍范围加倍。它是软缓存偏好，不是固定会话绑定。

G0 未命中比例已经约 4.55%，提高权重可能只影响少量尾部请求，也可能集中到热 rank，反而增加排队。R4 退化证明削弱缓存偏好有风险，不能反推权重越大越好。先做 40 是单次、容易解释的增强，不一次铺开 40/80/160 扫描。

| 配置 | 建议值 |
|---|---|
| 对照 | 已有完整 G0：R1-only、P weight=20，不重跑 |
| 新实验唯一策略变化 | P weight=40 |
| 其他路由 | R1=completion；R2/R3/R4 off；D weight=2；会话亲和不启用 |
| 负载 | C80，884 条预热＋3600 秒正式窗口 |
| 引擎 | P/D 各 8 GPU，TP8/DP8/EP1、DP attention；有效 chunk 均 4096（launcher 32768/DP8） |
| 缓存 | P HiCache ratio1.5、write_through/kernel/page_first；D HiCache off；KV fp8_e4m3 |
| 内存/请求上限 | mem_fraction 0.85，max_running256，graph max BS256 |
| MTP | D EAGLE：5 steps、topk1、draft6、simulate acceptance3.61；P 不新增 MTP |
| 种子 | P823508857，D19197414 |
| 镜像 | `infera-sglang:aus-0922-reqtrace`，沿用已验证 image ID |
| Router | 已验证 release 二进制，SHA256 `0cf48cbf96df48caf8c03949fc5edce1c97d26bebb293c71d5409dbe77cafe6d` |
| 数据与客户端 | 固定历史 revision；详细字段见 JSON |

完整可审核配置：[overlap40-proposed-config.json](overlap40-proposed-config.json)。当前状态为待用户确认，不是已执行。

这次是已验证代码的参数实验，只需启动检查和少量真实请求，不再为同一代码重复整套 C80 功能验证。新的亲和代码仍必须经过阶段 D 的 smoke。正式运行过程中周期性检查请求进度、P 队列、miss、D running 和错误；若发生真实故障或持续明显退化，由人工分析决定处理，不用辅助元数据 gate 自动杀服务。

## 6. 实验与开发如何并行

权重实验只使用固定旧二进制和冻结配置；新开发在当前源码目录中进行，产物另存，不能覆盖在运行的二进制、脚本或配置。CPU 编译/离线测试在控制机进行，不占实验 GPU 或与推理服务抢 CPU。

这样无需等待新策略开发就可以利用已分配节点。权重结果回答“增强现有软缓存偏好是否有收益”，开发验证回答“会话绑定和 Dynamo 式首次选择是否有收益”，两者相关但不互相替代。

## 7. 源码证据与限制

- [目标 GB300 配方](https://github.com/SemiAnalysisAI/InferenceX/blob/b384c40/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/gb300-fp4/agentic/glm5.2-agentx.yaml)：会话 TTL 和三项 None 阈值。
- [过载阈值语义](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/discovery/worker_monitor.rs)：`LoadThresholdConfig`。
- [绑定选择失败与重选](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/kv_router/push_router.rs)：`select_with_affinity`。
- [候选公式、pin 分支与最小值选择](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/kv-router/src/scheduling/selector.rs)。
- [活跃 block 投影](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/kv-router/src/sequences/prompt_registry.rs)与[默认参数](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/kv-router/src/scheduling/config.rs)。
- [D override](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/kv_router/prefill_router/mod.rs)、[会话租约和空闲 TTL](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/session_affinity/coordinator.rs)。
- [前次公开成绩关联审计](../router-followup-20260928/REPORT.zh-CN.md)。

上述是代码与配方证据，不是 NVIDIA 运行日志中的事件计数。实施可以对齐明确的语义、默认值和参照用例；性能和缓存事件端到端行为必须由后续实测确认。
