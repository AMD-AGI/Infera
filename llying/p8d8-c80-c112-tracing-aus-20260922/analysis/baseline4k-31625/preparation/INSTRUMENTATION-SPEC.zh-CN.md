# 第 3/5/6 项诊断补充规格（不部署到本轮基线）

从已完成 8K run 的全部 diagnostics 检查了 115,586 个事件，共 11 类。字段清单见 observed-event-schema.json。当前 4K 保留相同诊断代码，以便与 8K 对照。

## 现有证据能支持什么

- P/D request_summary 有 RID/room/rank、input/output、device/host/storage cache 计数及阶段时间，支持 miss/context/host 分层、服务时间和驻留账目。
- D admission 有 reason、required、budget、free_requests/free_metadata、running/transfer/waiting/prealloc 计数，支持已观察到的直接阻塞归类。
- 没有 P admission 停止原因事件，没有 Unified host_load 提交/完成事件，没有 router 每次 pick 的全 rank 候选快照。

## Unified host 路径：第 3 项

应在实际 Unified 路径记录 request/node 标识、rank、提交/完成观察的 monotonic 时间、bytes、方向、eviction/backup 标记。CPU 观察完成与设备执行时间分别存储，不通过增加 device synchronize 获取计时。现有 host-hit 标记不能代替回读耗时；D transfer wait 也不能代替 RDMA 或 host DMA 时间。

如补充 device event，需评估事件采样率与开销，并在相同构建上采对照；不得把带新插桩的处理组直接与未插桩基线比较细微吞吐差异。

## P admission / 公平性：第 5 项

每轮候选检查需关联 scheduler iteration、RID、rank、queue age、input/miss/context、已有 chunk 的剩余工作、剩余 input/chunk/KV/req-slot 预算；记录接受、停止遍历、继续遍历及原因。

先记录现有顺序，再实现单一策略变更。需明确给新短请求的保留份额及长请求防饥饿规则，保持 chunk、总预算和 cache policy 不变。不能把 GPU 繁忙或 request count 均衡当作调度公平性证据。

## D 反事实 shadow：第 6 项

当前 admission 的 required 只涵盖完整判定的一部分，且缺少候选 max_new、增长/回收预留和同一时刻其他 rank 的状态。

至少需要：

1. 候选 input、clipped max_new、retracted/restore 状态与完整两个条件的需求值。
2. 同次决策下各 rank 的 KV budget、request/metadata slots、pending reservations、健康/可路由状态。
3. 原 pick、shadow pick、每个候选拒绝原因、状态采样时间和实际 reservation 后的预算变化。
4. 保留原路由决策，仅记录 shadow 结果；不发起重复请求，不实际占用候选 rank 的资源。

先核验当前 router 二进制与拟修改源码的对应关系。跨节点非原子 scrape 只能给近似候选判断，不能宣称精确可接纳或潜在吞吐收益。
