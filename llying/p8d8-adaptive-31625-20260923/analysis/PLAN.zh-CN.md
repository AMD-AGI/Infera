# 自主验证计划：置信度优先、按结果裁剪

2026-09-23 用户授权在约十小时审阅窗口内自主选择、裁剪和增加实验，阶段性分析并提交。沿用 job 31625，Prefill n03-33 / Decode n02-21，C80，4K，acceptance 3.61。保留原始数据和所有失败尝试。

## 决策顺序

1. 验证复用服务协议：完全无在途请求时 flush 两端缓存，检查 8 ranks 的 GPU/host 逻辑缓存与队列为空；重新建立 router/collector，记录 P/D PID 和容量保持不变。若无法验证，改为重新 launch。
2. 用相同 workload/warmup 做一次复用协议的 4K 控制，检验与刚完成的新启动基线的漂移。这也提供后续处理组的相邻对照。
3. 优先检验 P 缓存工作集/容量：P fraction 0.85→0.90，host 绝对 tokens 固定。通过独立无负载容量标定获得实际 GPU token 容量，再计算精确 ratio；正式处理组重新 launch P，并再次核对容量。D 可复用，但必须用已验证的清空协议。
4. 只有 miss/服务时间和吞吐同时给出收益证据才做重复/回退确认；若无改善，停止继续放大容量。
5. 按新的诊断证据决定路由/缓存视图或 host 路径实验；不全面扫参数，不同时改变容量、后端和路由。

## 暂不投入的方向

- D 扩容：4K C80 仅 20 条直接 KV 阻塞、allocation mean 约22ms，预期总吞吐信息收益低，降级。
- 短请求份额/公平性：沿用用户在调研中的排除决定，不执行。
- D overlap weight：D 无缓存块 metadata，现有记录 hits/active/request blocks 为0，调该权重不能检验真实容量感知。
- 扩 host 或换 backend：先有可兑现 host-hit/复制成本证据；不得只因 host resident 满就扩大。

## 判据

主要看完成率、output/total tokens/s/GPU、TTFT p50/p90/p99、按 context/miss/host 分层的 P forward 与实际 miss 总量；检查错误、取消、D驻留和采样缺口。3%以下的单次差异通常不足以支持改默认；更大变化也必须有工作量/机制解释和重复确认，阈值不是统计置信区间。

所有复用尝试必须显式标记 reuse_control/reuse_treatment，保存清空前后指标、进程身份、router 状态及新的采集边界。若不能证明缓存初始化等价，不与新启动基线直接作因果结论。

## 分配变更

14:50:32 UTC job 31625 被抢占，两节点 SSH 失效；复用准备脚本未运行，原 P/D 未被本轮清空或停止。已申请同规格 2 节点替代 job 31641（batch QOS，10小时），等待资源。实际节点分配后重新建立 A0 控制；历史不同节点结果只作参考，不作处理组的直接因果基线。新增的独立日志边界和缓存清空验证仍用于后续同分配内服务复用。

## 15:40 决策更新：优先做路由生命周期对照

31641 因其他用户长期 GPU 占用不可用，已释放；31644 获得 n10-29(P)/n02-21(D)。重新建立 A0。两节点 GPU 空闲，n02-21 的旧训练 CPU 等待进程无 GPU 分配，测试期间由所有权监测器检查额外 GPU 活动；若出现干扰则使该轮无效并停止本轮客户端。

已从诊断镜像的 rust COPY layer 提取源码，确认 policy.rs、disagg.rs、Cargo.toml、Cargo.lock 与当前仓库逐字节一致。新增候选 G：HTTP 流式 P active guard 在完整 P 响应体 drain 后释放；默认 decode 模式完全保留原释放时机。两组使用同一新二进制，仅环境开关不同；公共观测日志在两组都开启。非流式和 NATS 路径不改变。

先 A0(decode) → G(completion) → A1(decode)，每轮清空 GPU/host 逻辑缓存并重新建立 router/collector、统一 warmup。P/D 服务可复用的前提是 PID/容量不变、缓存/队列全空、无外部 GPU 活动。依据结果再决定重复 G 或进入 P 容量测试，避免为低价值参数反复等待 HiCache 释放。

预言：减少 P 已完成但仍占 active 记账的时间，若这种偏差实际损害路由，应减少 miss/服务时间或提高完成率。否定：仅改变记账日志，miss/服务/吞吐无改善，或更强亲和造成排队恶化。不能只凭账目修正就宣称性能修复。

## 容量候选的低重启实现（仅准备，未发压）

已核对实际镜像的 KVCacheConfigurator._profile_available_bytes 和 _apply_token_constraints：mem_fraction 提供 KV 预算上限，max_total_tokens 对最终 token capacity 取 min；随后按页对齐。当前 P 的权重约73.645 GiB、KV约161.674 GiB、startup available约41.01 GiB。

候选 C 可显式设 P max_total_tokens=3,400,000（64 对齐，比3,143,424约+8.16%），P mem_fraction_static=0.90；通过 plan_host_capacity 的 ratio 精确保持 host=4,715,200 tokens。实际全rank容量必须等于目标，否则不开始负载。D、chunk、max-running、host backend/layout和路由模式保持相应控制组一致。P graph本来禁用，token cap远大于显式graph batch上限，不触发其额外裁剪。

这是一项 P KV 容量干预，配置中的 budget/cap/ratio 是共同实现及补偿手段；额外占用GPU内存会减少workspace余量，这是容量干预的成本，必须监控OOM、retraction与服务时间。不能仅凭profile预算计算就认定没有内存风险。

此方案可避免一轮无HiCache标定启动，仍需重新launch P。是否执行及是否重复，取决于 G0/回退控制的收益、剩余时间和资源稳定性。

## 18:04 采集与下一次重启准备

G0 完成884条预热，无取消/错误，18:00:36 UTC 开始完整3600秒 profiling。旧4K/A0共同请求存在−4到+5 token的输入差，与AgentX强制benchmark-specific cache-bust前缀一致；配对容差口径已在G0结果出来前固定，见 REQUEST-MATCHING.zh-CN.md。

`retire_stack.py` 仅准备容量重启：要求上一轮完成且审计通过、无INVALID、当前job和节点/用户匹配、离审阅/清理至少3小时、8 ranks运行/队列为0、没有本次benchmark客户端，且P/D/router容器ID与最终记录一致。保存具体停止计划，默认dry-run；只有显式--execute才停止并重命名本次栈，随后等待每张GPU VRAM≤2%、busy≤5%。etcd使用独立固定镜像hash。无GPU reset、无其他用户进程操作；遇到检查失败停止复核。

脚本编译通过，并对正在运行的G0调用dry-run验证其在首个“case未完成”检查即拒绝，未执行任何停止。实际执行前仍须做针对最终case的dry-run并检查计划。当前G0不受该准备影响。
