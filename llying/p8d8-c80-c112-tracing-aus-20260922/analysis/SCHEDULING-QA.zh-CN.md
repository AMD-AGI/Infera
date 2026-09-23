# P8D8 C80/C112 调度链解读与问答

整理日期：2026-09-23。

本文整理围绕 [原分析报告](RECOVERY-AND-FINDINGS.zh-CN.md) 第 3 节的多轮问答，结合本次镜像源码、逐请求记录和 Router 运行日志说明术语与执行过程。后续日志核查得到的新结论统一写入本文，避免把早先的一般机制解释误当成本次实验事实。

**最重要的修正：本次 D 路由虽然配置为 `kv-aware`，但全部已保存 Decode pick 的 `request_blocks`、`cache_hits`、`active_blocks` 都为 0。因此 D 实际按近期派单次数均衡，overlap weight=2 没有影响缓存评分。“distinct blocks 去重低估 D KV”是算法的一般风险，不是本次已经观测到的具体机制。**

## 目录

- [1. Prefill admission 是什么意思？](#q1)
- [2. 已有 chunk 优先续跑是什么意思？](#q2)
- [3. Router P/D guard 是什么？](#q3)
- [4. Router cost 公式与 P/D overlap weight](#q4)
- [5. 本次实验 D rank 实际怎么选，权重有没有影响？](#q5)
- [6. 已经选好了 D rank，为什么还需要 D admission？](#q6)
  - [6.1 两层决策的区别](#q61)
  - [6.2 一个请求的完整过程](#q62)
  - [6.3 为什么 D 要提前分配完整输入 KV？](#q63)
  - [6.4 预算 B 怎么计算？](#q64)
  - [6.5 新请求要通过哪两个门槛？](#q65)
  - [6.6 和 C80/C112 性能退化的关系](#q66)
- [7. 原报告 3.3：P admission、cache 和 chunk 的完整过程](#q7)
  - [7.1 P 收到请求后，先等什么？](#q71)
  - [7.2 缓存匹配到底匹配什么？](#q72)
  - [7.3 P admission 如何组建一轮 batch？](#q73)
  - [7.4 host cache 如何回读？](#q74)
  - [7.5 chunk 如何续跑，为什么短请求会等？](#q75)
  - [7.6 这段流程怎样解释实验数据？](#q76)
- [8. 原报告 3.4：P handoff、传输完成与解锁](#q8)
  - [8.1 handoff 交接的是什么？](#q81)
  - [8.2 缓存前缀、中间 chunk、最后 chunk 怎么发送？](#q82)
  - [8.3 为什么算完还不能立刻释放 KV？](#q83)
  - [8.4 D 何时开始生成，P 何时解锁？](#q84)
  - [8.5 解锁、驱逐、释放 guard 有什么区别？](#q85)
  - [8.6 transfer tail 与 D transfer wait 怎么读？](#q86)
- [9. 证据来源与边界](#evidence)

<a id="q1"></a>
## 1. Prefill admission 是什么意思？

Admission 可以译为“准入”。这里指：**请求已经来到 P 端，调度器判断它能否进入本轮 Prefill batch，以及本轮处理它的多少个 token。**

它和请求进入等待队列是不同的步骤：

| 步骤 | 含义 |
|---|---|
| 进入 waiting queue | 请求在等候调度 |
| 通过本轮 admission | 资源和预算允许，请求的一段工作被选入本轮 batch |

“匹配 GPU / host cache”是这个决策的输入。调度器需要知道哪些前缀 KV 已在 GPU、哪些可以从 host 回读、哪些还要计算，再评估 KV 空间、本轮输入 token 预算、chunk 预算和请求槽位等约束。

例如，一个请求有 100K 输入 token，其中 80K 的前缀 KV 已在 GPU，剩余 20K 需要计算。即使 KV 空间足够，也不代表这 20K 全部在本轮执行。本次每 rank 的 chunk 预算实际为 4,096 token，本轮可能只接纳其中约 4K。

源码中，`_select_prefill_admission` 先选择本轮处理范围，随后才按需要调用 `init_load_back` 发起 host 回读，再继续提交请求。见 [schedule_policy.py](source-evidence/managers/schedule_policy.py)。

报告中的“匹配 GPU / host cache 并选择 Prefill admission”，可以改读为：

> 检查可复用的 GPU/host 前缀缓存，并根据本轮资源预算，决定接纳哪些请求及其计算范围。

**P 仍有可回收 KV，却存在排队，并不矛盾。** KV 容量只是准入条件之一；本轮 chunk 预算已被占用，也会让新请求继续等待。

<a id="q2"></a>
## 2. 已有 chunk 优先续跑是什么意思？

用户的理解基本正确，但应将“优先进入队列”精确为：**未完成的 chunk 请求，在下一轮组建 Prefill batch 时优先获得执行预算；并不是把所有剩余 chunk 一次性插入队列。**

“已有 chunk 请求”特指当前 rank 上已经开始 Prefill、但 prompt 尚未处理完，保存在 `self.chunked_req` 中的请求。

实际顺序是：

```text
组建本轮 Prefill batch
  → 先尝试加入未完成请求的下一段 chunk
  → 扣减本轮预算
  → 再遍历 waiting queue，尝试加入新请求
  → 执行选出的 batch
```

[scheduler.py](source-evidence/managers/scheduler.py) 中，先调用 `add_chunked_req`，再遍历 `waiting_queue`。

简化例子：A 总共还有 12K token 要计算，B 只有 1K，B 在 A 第一轮之后到达。假设每轮可用预算为 4K，忽略其他限制：

| 轮次 | A 执行的工作 | B 的情况 |
|---|---|---|
| 第 1 轮 | 第一个 4K | 尚未到达 |
| 第 2 轮 | 优先续跑第二个 4K | 预算用完，等待 |
| 第 3 轮 | 优先续跑最后一个 4K | 预算用完，等待 |
| 第 4 轮 | A 已完成 | B 获得执行机会 |

如果 A 最后一轮只剩 2K，剩余预算就可能让 B 在同一轮进入 batch。因此，这不是规定“长请求结束前绝不运行其他请求”；但长请求持续吃满预算时，实际效果会接近这样。续跑仍受内存预算等约束，并非无条件执行。

**新请求只是命中了以前留下的 prefix cache，并不会因此成为这里的“已有 chunk 请求”。** 后者特指正在跨轮执行的未完成请求。

从本次逐请求记录和 chunk span 明细复算，miss≥32K 的请求为：

| 指标 | C80 | C112 |
|---|---:|---:|
| 请求数 | 332 | 450 |
| 每请求平均 chunk 数 | 18.80 | 24.57 |
| 平均 Prefill forward 时间范围 | 19.13 秒 | 30.94 秒 |

长 miss 请求确实横跨很多轮执行，续跑优先提供了它们影响后续请求等待的具体路径。不过，这些时间包含轮间调度、同步等等待，并非纯 GPU kernel 时间；统计也不能单独量化每个短请求被某个长请求挡了多久。

<a id="q3"></a>
## 3. Router P/D guard 是什么？

这里的 guard 是 Rust 中的 `ActiveGuard`，可以理解成“一张请求负载登记单”：

1. Router 选好 P rank 和 D rank。
2. 创建一个 guard，把请求涉及的 blocks 分别登记到 P、D 的 active 统计中。
3. 正常流式响应结束，或错误、断连等退出时，guard 被销毁，自动撤销登记。

创建时调用 `on_request_started`，销毁时调用 `on_request_finished`。实现见 [policy.rs](../../../rust/router/src/policy.rs) 和 [disagg.rs](../../../rust/router/src/disagg.rs)。

KV-aware 策略维护 block hash 的引用计数，并使用仍有引用的不同 block 数估计 active 负载。多个请求涉及同一个 block，引用可以有多份，但 distinct block 只计一份。**若该侧请求没有 block 信息，登记不会产生有效的 active blocks；本次 D 就属于这种情况。**

Guard 本身不负责锁住 GPU KV，不等于服务端的内存预留，也不是把整个请求串行化的锁。它更新的是 Router 后续选 rank 所使用的负载信息。

报告关注其生命周期与 P 实际工作时间不一致。例如，以下是示意时间线：

```text
t=0s   分配给 P2 / D5，登记两边 active
t=3s   P2 完成 Prefill 和传输，服务端可以解锁相关 KV
t=15s  D5 完成流式输出，Router guard 才撤销登记
```

第 3 秒到第 15 秒，该请求已不需要 P2 再做 Prefill，但 Router 的 P2 active 账目仍保留其 blocks。因此 Router 估计的 P 负载与实际 P 等待/执行负载会有偏差。

一般而言，D 关闭 prefix reuse 时，用去重后的 active blocks 估计空间也可能低估真实分配需求。**但本次实验 D active_blocks 始终为 0，具体应按第 5 节解释，而不能归因为去重。**

<a id="q4"></a>
## 4. Router cost 公式与 P/D overlap weight

原报告 3.1 正是在解释：Router 使用“缓存收益 + 估计负载”选 rank，但评分不等于实时排队时间或实际 KV 占用。

### 4.1 三个量的含义

```text
cost(rank) = -w × cached_prefix_blocks
             + distinct_active_blocks
             + recent_blocks
```

**选择 cost 最小的 rank。** Cost 是比较评分，不是毫秒数或实际内存大小，负数正常。

| 量 | 含义 | 影响 |
|---|---|---|
| cached_prefix_blocks | 当前请求在该 rank 上估计能命中的最长连续前缀 block 数 | 越多越倾向选它 |
| distinct_active_blocks | 已分配到该 rank、尚未结束的请求涉及的不同 block 数，共享 block 去重 | 越多越倾向避开它 |
| recent_blocks | 最近分配到该 rank 的估计 miss blocks 的衰减累计值 | 最近派发的新工作越多，越倾向避开它 |

`cached_prefix_blocks` 针对当前请求，并不是 rank 缓存总大小。缓存再多，与当前 prompt 无关也没有加分。

当前代码每次选择后更新 recent：

```text
所有 rank：recent ← recent × 0.97
选中 rank：recent ← recent + 本次估计 miss blocks
```

历史贡献约经过 23 次选择衰减一半，按选择次数而非秒数衰减。它保留“最近给谁派了工作”的记忆，避免请求不重叠、active 归零时反复选同一个 rank。

如果 `request_blocks=0`，无法估计 miss，则给选中 rank 加保底的 1 分。本次 D 使用了这一分支。

### 4.2 P=20、D=2 表示什么？

Overlap 指请求前缀与缓存前缀的重合，不是计算和通信重叠。

- P 权重 20：多命中 1 个 prefix block，评分减 20。
- D 权重 2：多命中 1 个 prefix block，评分减 2。

P、D 分别选 rank。20/2 不是流量比例，也不表示 P 收到 D 的十倍请求。

记 `L = distinct_active_blocks + recent_blocks`。若 A 比 B 多命中 `ΔH` 个 blocks，同时负载高 `ΔL`，则：

```text
当 ΔL < w × ΔH 时，选择 A。
```

权重规定“为了多一点缓存命中，愿意容忍多少额外负载评分”。

假设：

| 候选 rank | 命中 blocks | 负载 L |
|---|---:|---:|
| A | 100 | 1,000 |
| B | 60 | 300 |

| 权重 | A cost | B cost | 选择 |
|---|---:|---:|---|
| 20 | -1,000 | -900 | A |
| 2 | 800 | 180 | B |

这是基本公式的示例。当前代码允许缓存保留提示等调整有效权重，精确到某次选择应读 pick 日志的 `w_overlap`。本次已保存 Decode pick 全部为 2.0。

### 4.3 对性能的影响

P 权重较大，可能减少 miss、chunk 数和重算，提高完成率；也可能把共享前缀的请求集中到较忙的 rank，让新请求等待长请求续跑。

P 权重较小，可能缓解局部排队；但命中下降会增加 miss 和服务需求，最终也可能让整体更慢。即使权重降为 0，负载仍是代理指标，并不会自动变成按真实等待时间或剩余 KV 路由。

本次 C80→C112，cache hit 从 94.912% 降为 92.695%，实际 miss tokens 从约 5,891 万增到 7,653 万，增加约 29.92%。这说明不能忽略缓存损失的工作量代价，但两档实验不是严格匹配工作负载的权重对照，不能据此归因于权重。

C112 各 rank 平均 queue 都升高，整小时 miss-token 分布反而更均匀，也不支持简单解释为“缓存亲和性把请求全压到一个 rank”。

本次 D 权重的实际效果应看下一节：由于命中项为零，权重没有作用。现有实验没有证明 20/2 最优，也没有测出修改权重的吞吐收益。

<a id="q5"></a>
## 5. 本次实验 D rank 实际怎么选，权重有没有影响？

**配置为 kv-aware，实际按近期派单次数均衡；D overlap weight=2 被加载，但没有影响本次缓存评分。**

启动日志明确记录：

```text
router_policy: "kv-aware"
kv_prefill_overlap_weight: Some(20.0)
kv_decode_overlap_weight: Some(2.0)
```

Decode worker 注册信息为：

```text
dp_size: 8
kv_events_endpoint: null
kv_block_size: null
```

Router 知道有 8 个 D rank，但没有 D 的 KV 事件端点和 block 大小，不能像 P 一样建立请求与缓存 blocks 的匹配。

保存的 Router 日志共有 **21,151 次 Decode pick**，时间覆盖 2026-09-22 18:48:37 至 22:09:40 UTC，包括探测、warmup 和正式实验等阶段，并非仅报告的有效完成请求。全量统计如下：

| 字段 | 全部 Decode pick 的取值 |
|---|---:|
| cache_hits | 0 |
| request_blocks | 0 |
| active_blocks | 0 |
| w_overlap | 2.0 |
| mm_images | 0 |

所以 D 评分简化为：

```text
cost_D(rank) = -2 × 0 + 0 + recent_blocks(rank)
             = recent_blocks(rank)
```

没有 block 信息时，每选中一次，给该 rank 的 recent 加 1。因此，本次实际是优先选择近期派单较少的 D rank，每个请求同样计 1 分，没有按输入长度、实际 KV 占用、队列等待或预计输出时长区分请求。

实际分配：dp0、dp1、dp2、dp3、dp4、dp5、dp7 各 2,644 次，dp6 为 2,643 次。效果接近轮流派单，但不是显式 round-robin，实际顺序也非严格 0→1→…→7。

在这些观测条件下，仅将 D overlap weight 从 2 改为 0 或 20，不改变该评分结果。

**对原报告的修正：本次 D 路由不是 active blocks 去重后低估 KV，而是没有获得有效 block 负载信息。** 各 rank 请求数几乎相等，仍可因输入长度、上游等待和输出寿命不同而出现明显 KV 偏斜。优化重点是补充实际已分配 KV、pending 需求等信息，而不是单独调 D overlap 权重。

<a id="q6"></a>
## 6. 已经选好了 D rank，为什么还需要 D admission？

> **Router 选 D rank，回答“把请求交给谁”；D admission 回答“请求已经交给我了，我现在有没有资源接纳它”。**

Router 选中 D3，不代表 D3 已经分配了 KV，更不代表请求可以立刻生成。尤其本次 D 路由基本按近期派单次数均衡，没有提前保证目标 rank 的空间足够。

<a id="q61"></a>
### 6.1 两层决策的区别

假设 Router 给 D3 分配了一个 100K 输入 token 的请求。D3 可能请求数较少，却因为已有几个长请求，KV 接近满了。它必须检查本地真实状态。

| 步骤 | 执行位置 | 决定什么 |
|---|---|---|
| Router 选 rank | Router 进程 | 请求发给哪个 D rank |
| D admission | 被选中的 D rank 内部 | 现在能否分配资源，开始接收 KV |
| D generation 调度 | D rank 内部 | KV 就绪后，何时加入生成 batch |

D admission 是本地资源保护机制。即使 Router 将来能读取 KV 占用，也仍需它：路由信息可能滞后，多个请求可能接连到达，分配时必须再次检查。

<a id="q62"></a>
### 6.2 一个请求的完整过程

```text
Router 选好 P rank 和 D rank，并向两端发请求
                  ↓
D：请求进入预分配等待队列
                  ↓
D admission：检查槽位和 KV 预算
       ┌──────────┴──────────┐
     不通过                 通过
       ↓                     ↓
留在本地队列等待       分配输入 KV 空间
后续调度轮再检查       向 P 提供接收位置
                             ↓
                    D 等待 P 产生并传来 KV
                             ↓
                         KV 接收完成
                             ↓
                    进入 generation 调度
                             ↓
                      逐步生成输出 token
```

本次 `optimistic_prefill_attempts=0`，P 需要等待 D 准备好接收信息，才能进入正常 Prefill 调度路径。因此 D admission 等待可能让 P 停在 bootstrap 阶段。

三个关键区别：

- Admission 通过时，是空间准备好了，KV 数据尚未全部到达。
- 空间从分配完成起就已占用，即使 D 尚未生成。
- 普通容量不足让请求留队等待，不是立即向客户端报错，也不是自动换另一个 D rank。

源码 `pop_preallocated` 依次检查 request pool、metadata pool、适用的额外资源池和 KV 预算。遇到 KV 不足会 `break`，停止本轮后续分配。因此当前放不下的大请求，可能让后面较小请求也等。这叫队头阻塞。

这些资源不是一回事：请求槽位用来维护请求与 token 的映射等状态，metadata 槽位服务于传输等控制信息，KV token 槽位存放模型上下文 KV。任何适用资源不足，都可能阻止本轮准入。

<a id="q63"></a>
### 6.3 为什么 D 要提前分配完整输入 KV？

D 生成后续 token 时，需要使用整个上下文的 KV。

假设输入为 100K，P 命中其中 96K，只需新计算 4K：

```text
P 要新计算的内容：约 4K token
D 要持有的上下文 KV：约 100K token
```

**P 高缓存命中节省重算，不会自动减少 D 的上下文存储需求。**

本次 D 关闭 prefix reuse，不会因为其他请求已有相同前缀，就共享其 KV。它要单独准备完整输入 KV 的接收空间，P 再将缓存前缀和新计算部分分段传过去。计算和传输可以重叠，图示不是所有操作严格串行。

所以“容量检查在 generation 之前”的重点是：**D KV 不仅被正在生成的请求占用，也被已经分配空间、正在等 P 的请求占用。**

<a id="q64"></a>
### 6.4 预算 B 怎么计算？

这里预算以 **KV token 槽位数**计量，可以理解为“还能存放多少个 token 的 KV”，与 Router 的 block 评分不同。

检查分两步：先算 rank 允许拿出多少空间给新请求，再算新请求必须满足多大的空间门槛。

第一步的简化式：

```text
B ≈ free_KV
     - max(512 × active_requests, single_request_growth_reserve)
     - 其他调整项
```

| 项 | 含义 |
|---|---|
| free_KV | 当前 rank 的 KV allocator 实际剩余空间 |
| 512 × active_requests | 为已接纳请求保留后续生成的增长余量 |
| single_request_growth_reserve | 为已有 running 请求中的较大单请求保留继续推进所需的余量 |
| 其他调整项 | 为撤回待恢复请求、待恢复缓存，以及尚未计入常规队列的特定调度状态等补记预算 |
| B | 扣除保护性余量后，用于新 admission 的预算 |

这里 active_requests 是 **D 服务端的 running、等待 KV transfer、KV-ready waiting 等请求数**，不是 Router guard 的 active blocks。

为什么扣增长余量？输出增加通常会增加 KV。如果把空闲空间全部分给新请求输入，已有请求可能没有空间继续生成。

为什么取 max 而不相加？这是两种覆盖部分相同需求的空间保护条件，代码取更严格的一项。

512 是准入预算的增长预留参数，不是输出长度上限，也不等于准入时已经实际分配或写入 512 个输出 token 的 KV。

进一步对应源码：令 T 为当前 running 请求的输入与已生成输出 token 总量，则单请求保护项按 running 请求中的最大 `输入长度 + 截断后的最大输出长度 - T` 估算；没有 running 请求时为 0。它考虑必要时撤回 running 请求可回收的空间，不是把所有请求的最大输出长度相加预留。

具体实现见 [decode.py](source-evidence/disaggregation/decode.py) 的 `_need_space_for_single_req`、`_active_reserved_tokens` 和 `_allocatable_token_budgets`。上式为解释用简化式，不能替代包含页对齐、恢复等分支的完整代码。

<a id="q65"></a>
### 6.5 新请求要通过哪两个门槛？

对本次普通新请求，D 不复用前缀，忽略页对齐等细节，可写成：

```text
门槛一 = 本次需要分配的输入 KV + 512

门槛二 = 输入长度
         + 经过截断的最大输出长度估计
         - 当前 running 请求可通过撤回回收的 token 量

要求：max(门槛一, 门槛二) ≤ B
```

两个门槛分别保护：

- 门槛一：当前分配不能把基本增长余量挤掉。
- 门槛二：不能只把输入放进去，却缺少后续继续生成的空间。

“撤回”指必要时暂停请求、回收其设备 KV、之后再恢复的机制。代码把当前 running 请求的 token 量作为可回收量纳入估算；这不表示本次 admission 会立即执行撤回。等待接收 KV 的请求不能简单按同样方式视为可回收，所以不能将所有已分配 KV 都用于抵扣。

以下是纯示意例子，单位均为 KV token 槽位：

```text
可接纳预算 B          = 20,000
新请求输入长度        = 12,000
输出长度估计          = 16,000
可撤回回收的 token 量  =  4,000

门槛一 = 12,000 + 512             = 12,512
门槛二 = 12,000 + 16,000 - 4,000 = 24,000
```

虽然输入放得下，12,512 < 20,000，但第二门槛不满足，24,000 > 20,000，所以仍需等待。

报告中的 **“required < budget，但仍被拒绝”** 就可能来自这里：诊断事件只记录门槛一，而实际判断取两者最大值。“拒绝”在这里准确说是“本轮不予接纳”，不是请求失败。

<a id="q66"></a>
### 6.6 和 C80/C112 性能退化的关系

```text
P 排队和处理变慢
    ↓
D 已分配输入 KV，但更长时间等不到完整 KV
    ↓
这些请求尚未开始生成，却一直占着 D 空间
    ↓
某些 D rank 的 admission 更难通过
    ↓
后续请求在 D 等预分配，P 也可能等 bootstrap
```

本次 D allocation 平均等待由 C80 的 0.034 秒增至 C112 的 0.793 秒；等待上游期间的输入 KV 驻留代理从约 310 万增至 863 万 token。

“驻留代理”来自 `Σ(输入 token 数 × 阶段时长) / 3600`，估计该阶段的平均占用贡献，并非某一时刻精确分配页数，也不包含所有输出增长和页对齐等影响。

因此：**Router 派单数量均匀，不保证各 D rank 能及时通过本地资源检查；D 开始生成前的等待，也会占用大量 KV。** 这解释了为什么 D running 没有增加，KV 却更满了。

<a id="q7"></a>
## 7. 原报告 3.3：P admission、cache 和 chunk 的完整过程

> **P admission 决定“本轮让哪些请求计算多少 token”；cache 决定“哪些前缀可以复用”；chunk 决定“尚需计算的部分如何分多轮完成”。**

这三个机制一起决定请求何时真正开始 Prefill、要执行多少轮，以及占用多少资源。它们发生在 P 服务端，与 Router 选 P rank 是不同层次的决策。

<a id="q71"></a>
### 7.1 P 收到请求后，先等什么？

本次普通请求的概念流程如下：

```text
Router 选择 P rank，向 P 发送请求
    ↓
P 建立该请求的 KV sender，进行 bootstrap
    ↓
等待 D 完成接收资源准备、提供接收信息
    ↓
请求进入 P waiting queue
    ↓
每轮组 batch：先考虑旧 chunk 请求，再考虑 waiting queue
    ↓
匹配缓存、检查预算、必要时安排 host 回读、提交本轮计算范围
    ↓
执行本轮 Prefill
    ├─ prompt 尚未处理完 → 记录进度，后续轮续跑
    └─ prompt 处理完 → 完成剩余发送，等待传输确认
```

这是理解调度依赖的流程，不代表 host 回读、GPU 执行与 P→D 传输全部严格串行。实现会通过异步提交和完成事件保证数据依赖，并尽量重叠工作。

本次 `optimistic_prefill_attempts=0`，没有依靠 optimistic 路径在 D 未准备好时提前计算。因此要区分：

- **P bootstrap 等待**：D 接收信息等前置条件尚未就绪。
- **P waiting queue 等待**：进入可调度队列后，尚未首次执行。
- **后续 chunk 间隙**：已经执行过第一轮，仍在跨轮完成 prompt，计入 forward envelope，而非首次 P queue。

<a id="q72"></a>
### 7.2 缓存匹配到底匹配什么？

Prefill 的输入不只是“总共有多少 token”，还包括“已有多少前缀 KV 可复用”。本次 P 使用 Unified Radix Cache，缓存按前缀关系组织，调度时根据本地实际缓存状态更新匹配结果。

Router 的命中估计用于选 rank；P 本地缓存匹配用于真正执行。两者不必完全一致，因为缓存可能在路由之后被驱逐、回读或被其他请求占用。

假设一个 24,576-token 输入，在本地匹配后为：

| 输入区间 | 状态 | P 要做什么 |
|---|---|---|
| 前 12,288 token | GPU 上已有可复用 KV | 直接引用已有 KV，无需重算这部分 |
| 接下来的 4,096 token | host 上有可复用 KV | 安排回读到 GPU，再复用 |
| 最后 8,192 token | 没有可复用 KV | 执行新的 Prefill 计算 |

在这三个连续区间的简化例子中：

```text
可复用前缀 = GPU hit + 可兑现的 host hit = 16,384 token
剩余计算量 = 总输入 - 可复用前缀       =  8,192 token
```

这不是任意位置相同的 token 都能抵扣；它依赖有效的前缀匹配。实际实现还可能涉及辅助状态、页对齐、最后 token 的处理等，表格只是说明主流程。

**host hit 不等于数据已经在 GPU，也不等于零成本。** 它省下的是重算，但需要搬运、GPU 接收空间、锁和同步等操作。

复用已有前缀也不意味着后续计算不再读取它。计算新 token 的 attention 仍依赖已有上下文；同样 4K 个新 token，在不同上下文长度下，服务成本不一定相同。

<a id="q73"></a>
### 7.3 P admission 如何组建一轮 batch？

每轮调度会新建一个 `PrefillAdder`，管理这轮的候选列表和资源预算。可以把它理解成“本轮 batch 的组装器”。

**第一步：先考虑上一轮留下的 chunk 请求。**

如果 `self.chunked_req` 不为空，先更新它的下一轮输入范围，再调用 `add_chunked_req`。若资源允许，就把下一段加入本轮 batch，并扣减预算；若暂时无法放入，仍保留未完成状态。优先不等于保证执行。

**第二步：遍历 waiting queue 中的新候选。**

本次策略为 FCFS。对候选请求，调度器检查请求槽位等限制，调用 `init_next_round_input(self.tree_cache)` 更新缓存匹配，再调用 `add_one_req`。

**第三步：先选择执行形状，再提交有副作用的操作。**

`add_one_req` 临时锁住匹配节点，避免把即将复用的前缀仍算成可随意驱逐的容量，然后调用 `_select_prefill_admission`。该函数决定：

- 从哪个 token 位置开始新计算，即 `prefix_len`。
- 本轮新计算多少 token，即 `extend_len`。
- 本轮之后 prompt 是否仍未完成，即 `is_chunked`。

选择阶段本身不把 host KV 真正搬回来，也不直接分配请求槽位。选定后，才按需要调用 `init_load_back`，随后通过 `_commit_prefill_admission` 记录执行范围、增加请求持有的锁、加入 `can_run_list` 并扣减预算。后续 batch 准备阶段再完成实际执行所需的分配和映射。

主要限制可分为：

| 限制 | 回答的问题 | 不满足的后果 |
|---|---|---|
| request slot / batch 请求数限制 | 本轮还能接入更多请求吗？ | 停止接入或等待 |
| KV 容量预算 | 锁住要复用的前缀后，还有足够空间支撑本次接纳吗？ | 本轮不能接纳 |
| input token 预算 | 本轮输入工作量额度是否还有余量？ | 停止继续组 batch |
| chunk token 预算 | 本轮还允许处理多少新 token？ | 截断成较小 chunk，或额度耗尽后停止 |
| 其他适用 gate | 实现中的 tile 等约束是否允许当前形状？ | 按分支停止或重试 |

这些是不同约束，不能用“GPU 还有空闲页”代替全部判断。本次已确认每 rank 的 chunk 预算为 4,096，不能把命令行的 32,768 当成每 rank 预算；也不能把源码中所有可选 gate 的存在当成它们在实验里都触发过。

**第四步：处理返回结果，完成 batch。**

如果返回 `CONTINUE`，继续看后面的候选；若返回其他结果，这条候选遍历路径会停止本轮继续接纳。注意，停止可能发生在“候选没被加入”时，也可能发生在“候选已经加入，恰好耗尽预算”之后，不能只看返回值就断定该请求被拒绝。

最后，被接纳的新请求移出 waiting queue，形成 Prefill batch；其余请求留队。源码也有对特定未就绪条件执行 `continue` 的分支，因此“遇到任何障碍都绝不跳过”也不是准确概括。

<a id="q74"></a>
### 7.4 host cache 如何回读？

对需要 host 回读的候选，关键过程是：

```text
确认需要回读的缓存区间和关联状态
    ↓
检查回读额度、锁住必要的源数据和节点
    ↓
检查 GPU 空闲空间
    ├─ 足够 → 准备接收
    └─ 不足 → 尝试驱逐可回收缓存，再检查
    ↓
提交 host→GPU 拷贝，登记目标 device indices 和 ongoing load
    ↓
通过完成事件保证依赖该 KV 的操作不会过早读取
    ↓
loading_check 轮询完成 ack、进行必要的跨 rank 协调和事件同步
    ↓
撤销回读期间的临时保护，更新缓存状态
```

`_load_back_transfers` 会在空闲空间不足时调用 `evict_for_alloc`，再通过 `cache_controller.load` 提交回读。如果空间仍不足等条件不能满足，回读可能无法提交；不能因为早先匹配到 host hit，就认为它必然已经可用。

`init_load_back` 返回目标索引，不代表 CPU 返回时所有字节已经到达。异步拷贝的实际就绪依靠事件等同步机制。`loading_check` 中可见 ack 检查、all-reduce 和 `finish_event.synchronize()`。

要区分两类锁：回读操作为了保护源和目标数据而持有的临时锁，以及请求为后续计算、传输仍需持有的锁。回读完成释放前者，不意味着请求使用的所有 KV 都立即可驱逐。

本次还配置了 `write_through`，会产生 GPU→host 的缓存备份工作。这与 host→GPU 回读方向相反，不能用 backup 速率代替回读速率。

**没有 host hit 的请求，也可能间接受 host 工作影响。** 回读、写回、驱逐和同步会使用共享资源。不过，源码只证明这些操作存在，不能据此认定它们是本次主要耗时来源。

<a id="q75"></a>
### 7.5 chunk 如何续跑，为什么短请求会等？

延续前面的例子：缓存复用后还需计算 8,192 token，假设本轮完整 4,096-token 预算可用，忽略其他限制：

| 轮次 | 本轮新计算区间 | 轮后状态 |
|---|---|---|
| 第 1 轮 | 第 16,384 至 20,479 个位置 | prompt 未完成，保留为 chunked_req |
| 第 2 轮 | 第 20,480 至 24,575 个位置 | prompt 完成，转入传输完成等待 |

位置按从 0 开始编号。第一轮计算出的 KV 会被保留，第二轮在前一轮结果上继续，不会把第一轮重新算一遍。

一个请求剩余 8K 要分两轮，并不代表“每个请求每轮都可领 4K”。**4K 是该 rank 本轮共享的 chunk 预算。** 如果旧 chunk 已用完它，新请求即使很短，也可能进不了本轮 batch。若旧请求只剩 2K，剩余额度才可能让其他请求加入。

FCFS 主要约束 waiting queue 中候选的顺序；已有 chunk 请求在遍历该队列前就被考虑，因此不等于“每轮把所有新旧请求重新公平轮转”。较长 miss 请求可以连续占用多轮预算。

此外，attention DP rank 虽有独立请求状态，TP/DP 执行仍可能协同。不能把某 rank 本地队列为空直接理解成对应 GPU 完全空闲，更不能把系统当成 8 个互不影响的服务台。

<a id="q76"></a>
### 7.6 这段流程怎样解释实验数据？

| 观测 | 应如何理解 |
|---|---|
| C80 P queue 均值 5.576 秒，C112 为 22.409 秒 | 首次执行前的等待显著增加；不能全部归成纯 kernel 时间 |
| C80 有排队的同 rank 样本，可回收 KV 平均仍约 299.5 万 token | 不支持普遍因活跃 KV 把物理池占满而停工；仍可能受 chunk 等预算和服务进度限制 |
| miss≥32K 组平均 18.80/24.57 个 chunk | 长请求确实跨多轮，续跑优先为影响新请求等待提供了机制 |
| 长 miss 组 forward 均值 19.13/30.94 秒 | 包含多轮服务、调度与同步范围，不是纯 GPU kernel 累加时间 |
| C80/C112 host-hit 请求仅约 0.84%/3.62% | 不支持所有慢请求都在直接回读；不能排除共享回读/写回开销 |

当前证据定位了 P 服务与排队路径，但没有逐次 admission 停止原因、完整 Unified host 回读时序和 device profile，因此尚不能精确分摊 chunk 固定成本、计算、collective、缓存操作与调度间隙。

<a id="q8"></a>
## 8. 原报告 3.4：P handoff、传输完成与解锁

> **Handoff 指把 P 已得到的上下文 KV 和必要的请求状态交给 D，让 D 接着生成；P 必须在确认传输完成后，才能撤销该请求对发送源 KV 的保护。**

这里不是把 P 的同一块 GPU 内存“过户”给 D。两端有各自的 KV 空间：D 预先分配目标位置，传输将 P 的 KV 写入 D 的目标空间。

<a id="q81"></a>
### 8.1 handoff 交接的是什么？

主要包括 prompt 对应的 KV，以及 Decode 继续请求所需的元数据和适用的状态。最后一段发送会处理相关 metadata；带 speculative 等配置时还可能有配套状态，不能把交接理解为只有一段原始字节拷贝。

本次 D 不复用前缀，`decode_prefix_len=0`，所以即使 P 上 prompt 大部分是缓存命中，D 仍需要接收完整输入对应的 KV。P cache hit 高，主要减少重算，不等于同比减少 P→D 的输入 KV 传输量。

```text
P 端：GPU 命中的前缀 + host 回读的前缀 + 新计算的后缀
                         ↓ 发送
D 端：已预分配的完整输入 KV 接收空间
```

<a id="q82"></a>
### 8.2 缓存前缀、中间 chunk、最后 chunk 怎么发送？

`send_kv_chunk` 使用 `start_send_idx` 记录已经推进到的发送位置，结合本次可以发送的 `end_idx`，选取尚未提交的区间，转换为传输需要的页索引，再调用 sender。这个游标表示发送提交进度，不能当成 D 已确认收到的数据量。

**第一类：缓存前缀。**

代码提供 `maybe_send_cached_prefix_chunk`，在相应 early-send 开关打开、bootstrap 就绪等条件满足时，可以提前发送已在 GPU 的缓存前缀。它不会把尚未回读完成的 host-only 前缀直接当成 GPU 数据读取。

这是一条受配置控制的优化路径。现有核查确认代码支持它，但没有据此确认本次每个请求都发生了独立的 early-prefix send。如果未独立提前发，尚未发送的前缀仍可随后续 chunk 发送覆盖。

**第二类：中间 chunk。**

某轮新计算完成、prompt 尚未处理完时，已就绪的 KV 可以分段交给传输模块。下一轮计算可与先前 KV 传输重叠，不必等整个 prompt 完成才开始全部搬运。

在 overlap 执行分支，源码会记录 GPU 完成事件，并把事件交给 sender，避免传输模块在 GPU 尚未写完该段 KV 时就读取源内存。发送边界也会考虑页对齐；不能把每个计算 chunk 和每次底层传输调用机械地一一对应。

**第三类：最后 chunk。**

prompt 处理完成后，代码调用 `send_kv_chunk(last_chunk=True)`，提交剩余区间及必要的最终元数据/状态，并把请求放入 P 的 inflight transfer 队列等待完成确认。即使之前已发送大量前缀和中间 chunk，这个最后阶段仍要正确完成交接。

例如，一个请求有 8K GPU 缓存前缀、8K miss，每轮算 4K。在允许提前发送并能重叠的示意流程中：

```text
发送缓存前缀 8K ──────────┐
计算第一个新 4K → 发送这段 ├─ D 逐步接收
计算第二个新 4K → 最终发送 ┘
```

如果未启用独立的前缀提前发送，第一次发送可以同时覆盖尚未发送的缓存前缀和已计算后缀。无论如何，本次 D 需要的是完整输入 KV，而不是只有 8K miss 的 KV。

<a id="q83"></a>
### 8.3 为什么算完还不能立刻释放 KV？

因为 GPU 计算完成、发送调用返回和传输完成是三件不同的事：

| 事件 | 能说明什么 | 还不能说明什么 |
|---|---|---|
| 最后一个 Prefill chunk 完成 | P 已完成本次输入处理所需的计算 | D 已收到所有 KV |
| sender.send 返回 | 该段发送已提交给传输路径 | 异步读取和网络写入全部完成 |
| sender poll 返回 Success | P 观察到传输后端报告成功 | Router 的整个流式请求也已结束 |

如果发送还在读取 P 的源 KV，P 就把这些页释放并复用于另一个请求，传输可能读到被覆盖的数据。因此，完成计算后仍需保护 KV，直到传输成功或失败清理路径安全地结束其使用。

源码中的 P inflight 队列会轮询 sender 状态；`WaitingForInput` 或 `Transferring` 继续等，`Success` 才走正常解锁和清理，失败则走错误处理路径。

<a id="q84"></a>
### 8.4 D 何时开始生成，P 何时解锁？

两端分别观察传输状态：

```text
                   KV 传输完成
                 /             \
P 轮询 sender 得到 Success      D 轮询 receiver 得到 Success
          ↓                               ↓
release_kv_cache                  提交接收结果和请求状态
清理 sender                              ↓
结束该请求的 P 传输阶段           进入 KV-ready waiting / prebuilt 接入
                                          ↓
                                  加入后续 generation
```

P 的 `process_disagg_prefill_inflight_queue` 在 `KVPoll.Success` 后调用 `release_kv_cache`，完成 cache 请求收尾、清理 sender，并记录传输完成观察时间。D 在 receiver 成功后调用 `_commit_transfer_to_req`，再将请求推进到后续调度。

**不能把两端写成“P 先解锁，D 才开始”或“D 一定先开始，P 才解锁”。** 它们有各自的轮询与调度节奏；D 也无需等 Router guard 释放。没有精确跨机时序证据时，不应推断亚秒级的严格先后。

同样，收到第一个中间 chunk 不代表普通请求已经能开始持续生成。D 要等本请求所需的输入 KV 和相关状态就绪，再进入生成路径；本次使用的是这个常规 PD 交接流程。

<a id="q85"></a>
### 8.5 解锁、驱逐、释放 guard 有什么区别？

| 操作 | 所在层次 | 实际含义 |
|---|---|---|
| P 请求释放/解锁 KV | P 服务端 | 该请求不再需要保护这些 KV，释放请求相关资源和引用 |
| 缓存成为可驱逐 | P 缓存管理 | 若无其他锁/引用，可以在需要时回收，但数据可能仍在 GPU |
| 驱逐缓存 | P 缓存管理 | 实际移除相应 GPU 缓存驻留，回收页；host 副本可能仍保留 |
| D 请求完成释放 KV | D 服务端 | 输出结束后释放本请求的 D 资源；本次 D 不保留用于 radix reuse 的前缀 |
| Router guard 销毁 | Router | 撤销路由侧 active 登记，不直接操作 P/D 的 GPU 页 |

**P 解锁不等于删除缓存，更不等于 GPU free 数量必然立即增加。** 对已纳入前缀缓存的 KV，如果不再被其他请求或缓存操作保护，常见状态变化是从“不可驱逐”变为“可驱逐”；数据仍 resident，后续请求可以命中。未纳入缓存的资源则可按相应路径回收。

若另一个请求仍引用同一前缀，当前请求结束也不会把其他请求的引用一起撤销；对应 KV 可能仍受保护。

这就是为什么报告要分开统计：

```text
capacity = 非可驱逐占用 + 可驱逐缓存 + 空闲
resident = 非可驱逐占用 + 可驱逐缓存
可回收量 = 可驱逐缓存 + 空闲
```

P resident 接近 100%，可以同时存在大量可回收缓存，并不等于活跃请求把池占满。另一方面，等待传输确认确实可能延长锁定时间，使部分页暂时无法驱逐。

<a id="q86"></a>
### 8.6 transfer tail 与 D transfer wait 怎么读？

这两个指标起点不同：

| 指标 | 大致起止 | 包含的主要内容 |
|---|---|---|
| P transfer tail | 最后阶段进入 P transfer 队列 → P 观察传输完成 | 剩余传输、后端完成通知、轮询/调度延迟及资源持有时间 |
| D transfer wait | D 已分配空间进入 transfer 队列 → 输入 KV 就绪进入 waiting | 等 P 首次调度、多轮 Prefill、必要的 KV 搬运及完成观察 |

特别注意：P 的 transfer tail 从最后阶段开始计，之前的缓存前缀和中间 chunk 可能已经在发送；它不是该请求所有传输操作的完整耗时。

以下时间线只是示意，不能用来判断本次真实跨机先后：

```text
t=0.0s  D 分配完输入 KV，开始等 P
t=0~5s  P 等待首次执行
t=5~8s  P 分轮计算，期间已发送部分 KV
t=8.0s  P 提交最后发送，进入 transfer tail
t=8.8s  D 观察输入 KV 就绪
t=9.0s  P 下一次轮询观察传输成功，解锁
```

这个例子中 D transfer wait 约 8.8 秒，P tail 约 1 秒，但不能相加成“网络花了 9.8 秒”。D 的 8.8 秒大部分在等上游，P tail 与它还可能覆盖相同的物理传输过程。

本次实际均值为：

| 指标 | C80 | C112 |
|---|---:|---:|
| P queue | 5.576 秒 | 22.409 秒 |
| P forward envelope | 3.120 秒 | 4.594 秒 |
| P transfer tail | 0.927 秒 | 1.222 秒 |
| D transfer wait | 9.421 秒 | 28.029 秒 |

C80 的 P queue + forward 已约 8.696 秒，与 D transfer wait 的 9.421 秒在量级上接近，支持 D 大量时间在等 P。不同阶段有重叠和边界差异，不能把差额直接认作网络耗时，更不能用 `输入字节 / D transfer wait` 计算 RDMA 带宽。

这些数据没有完全排除网络、通知或轮询开销；要进一步区分，需要记录实际发送提交、传输引擎完成、字节数和 CPU 观察时间。现有证据支持优先查 P 服务路径，不能仅凭 P tail 接近 1 秒就宣布 RDMA 是主瓶颈。

<a id="evidence"></a>
## 9. 证据来源与边界

- 主报告：[RECOVERY-AND-FINDINGS.zh-CN.md](RECOVERY-AND-FINDINGS.zh-CN.md)。
- 镜像源码：[scheduler.py](source-evidence/managers/scheduler.py)、[schedule_policy.py](source-evidence/managers/schedule_policy.py)、[decode.py](source-evidence/disaggregation/decode.py)、[prefill.py](source-evidence/disaggregation/prefill.py)、[unified_radix_cache.py](source-evidence/mem_cache/unified_radix_cache.py)。镜像来源与哈希见 [manifest.json](source-evidence/manifest.json)。
- Chunk 汇总：[chunk-evidence.json](chunk-evidence.json)。本轮问答另从逐请求及 span 明细复算了 miss≥32K 组的请求数、chunk 数和 forward 均值，结果一致。
- 当前仓库 Router 源码：[policy.rs](../../../rust/router/src/policy.rs)、[disagg.rs](../../../rust/router/src/disagg.rs)。
- 原始实验根目录：`/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922`。
- Router 运行证据：[server-logs/router.log](/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922/server-logs/router.log)、[launch/workers.json](/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922/launch/workers.json)。
- 请求与 span 明细：该实验根目录下 `analysis/joined-requests.jsonl`、`analysis/span-timelines.jsonl`。

运行日志直接确认了本次 Router 策略、权重和所选 D rank 的字段统计；recent 的具体衰减和 guard 实现按当前仓库源码解释，未完成运行二进制与源码的逐字节核验。本文未新增服务运行、干预实验或权重对照实验，不能量化修改路由或 admission 策略的吞吐收益。
