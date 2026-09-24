# R1/R2/R3/R4 Router 开发与后续验收

开发日期：2026-09-24。代码位于同一份 Infera 源码 `rust/router/src/`，功能可单独切换或叠加，不需要多个开发目录。目标是本次使用的 **Rust Router**；Python Router 的算法未修改。

## 原有逻辑与默认值

原 KV-aware 评分、recent/active 账本、单层事件目录和 round-robin 保留。以下开关默认关闭；默认不会维护第二份分层目录，也不会登记新需求账本。`shadow` 保留原派单，只记录候选策略的成本和建议。实验前使用同一二进制切换模式。

| 环境变量 | 默认 | 可选值与含义 |
|---|---|---|
| `INFERA_PD_PREFILL_GUARD_RELEASE` | `decode` | `completion` 将旧 P block guard 交给 P 响应消费任务 |
| `INFERA_R2_DECODE_DEMAND` | `off` | `shadow` 记账和旁路观察；`on` 按在途输入需求选择 D |
| `INFERA_R3_CACHE_TIERS` | `off` | `shadow` 建立分层目录但维持原评分；`on` 用分层目录选 P |
| `INFERA_R3_HOST_WEIGHT` | `0` | `[0,1]`，host 相对 GPU 的命中折扣；非零要求 R3 非 off |
| `INFERA_R4_PREFILL_WORK` | `off` | `shadow` 记录 P 工作量；`on` 按积压＋新增有效工作量选 P |

R1 completion 和 R2/R3/R4 需要 `--router-policy kv-aware`，无效配置启动时报错。没有自动修改引擎启动参数、容量、chunk、客户端输出预算或生产默认行为。

R1 已整合到源码，仍使用旧开关。**编译这份源码时，不要再应用 `router-prefill-guard.patch`。** 该历史补丁保留作为 A0/G0 的复现证据。新实现同时覆盖 HTTP streaming、HTTP unary 和 NATS P 响应消费的生命周期，不能将旧实验收益直接归于所有新增路径。

## 实现内容

### R2：D 输入需求

输入长度由请求分词获得，按 worker 的模板变体处理，不再依赖 D 是否发布 `kv_block_size` 或启用 Radix Cache。第一版单位为 tokens，不声称等于物理 KV 页数。相同前缀的请求分别计入；OpenAI `n` 多样本在 D 侧按份数计费。

选择和预订需求在同一个短临界区完成；请求分词在锁外。RAII reservation 在 pick 未使用、请求出错、结束或被取消时释放；D 等 P 期间仍占账。未知长度、多模态请求及已确认模板不一致的 worker 标记 unknown；候选 D 有未知在途负载时，保守使用原策略，避免把未知需求当作零。

账本表示 Router 接受的在途请求，不包含所有后端输出增长和回收延迟。客户端断开使 Router 生命周期结束，并不能证明引擎已经回收 KV；与实际引擎的对账属于后续验收。

### R3：分层目录

保留原目录，并新增 worker/rank 下的 GPU 与 host 目录。读取 `medium`，支持旧事件未携带 medium 时按 GPU 处理；`CPU_PINNED`、`CPU`、`CPU_TIER1` 作为 host，未知层不用于新目录。

同一块可以同时驻留两层；删除一层不删除另一层。查询为 GPU 连续前缀＋host 连续延伸，不重复加分。host 权重 0 可单独验证分层目录修正。

ZMQ 可见序号缺口、publisher 重启、不可解码批次会使相应分层目录失效；NATS 重连重放、可见 JetStream 序号缺口或无法解码事件也清理分层目录。回放尚未追平时，分层命中暂不用于评分；消息流错误立即使分层目录失效。没有层信息的旧 KV bucket 快照不用于填充分层目录。目录从后续有根事件恢复；不能恢复引擎或 relay 在发布前已经丢失的事件。

候选日志中的 `tier_stats` 包含 GPU/host store、未知事件和拒绝 store 的计数。**当前 Unified Radix 的实际事件覆盖仍待有机器时检查。** 本次没有凭空补造引擎 host 事件，也没有声称已经完成 GPU→host→GPU 端到端验收。

### R4：P 工作量

以未完成有效输入工作量＋当前请求有效输入工作量选 P。GPU 命中减去相应 token 工作量；R3 on 时还可应用 host 折扣。R3 off 时可单独使用原目录的前缀命中估计，但会继承原目录无法区分存储层的限制。

R4 的工作账本独立于旧 active blocks。P 响应消费任务持有并释放它，不等 D 全部生成完成，也不在刚收到 P HTTP 响应头时释放。没有逐 chunk 进度预测；该估计会包含 P 任务等待传输的阶段。

比较 R4 与原 P 评分时，双方应固定相同 R1/R2/R3 设置。R4 shadow 保持当前控制配置的 I/O 顺序，不因新增账本提前读取 P 响应体。旁路的是旧派单所形成的真实轨迹，不是另一套完整的反事实运行。

## 日志与开关组合

启动记录 `router experiment controls`。实验候选日志 `routing experiment candidate` 包含同一次 pick 的 `decision_id`、role、target、模式、输入长度、已加当前请求的 demand cost、原评分、分层评分、GPU/host hits、实际 selected、原策略 legacy_selected、R2/R4 的 demand_suggested 和 R3 的 tier_suggested。每个候选只读取一次缓存快照；负载锁内只进行需求快照、选择和登记，日志在锁外输出。原 pick 日志带同一个 decision_id；不记录原始 prompt。

新策略只负责选 worker/rank，不增加硬性 D admission 门槛，也不改变 SGLang 的 chunk 或 KV allocator。

可使用 [config/profile.sh](config/profile.sh) 设置完整环境，避免继承上轮残留开关。例如：

```bash
source llying/router-r234-20260924/config/profile.sh r2-shadow
# 然后用原有完整配置启动新编译的 Rust Router。
```

profile 只设置环境，不启动服务、不申请 GPU、不清空缓存；拼错 profile 名称不会改动原环境。除 legacy 外，模板都显式开启 R1，因此 R2/R3/R4 的对照双方必须保持相同 R1 设置，不能把 legacy 与某个 shadow 模板直接当作“只加观察”的对照。`combined` 用于后续跑通组合功能，不作为单变量收益对照。host 示例权重 0.5 只是待校准起点。

## 本地检查和待做工作

编译和 CPU/mock 检查可用：

```bash
cargo test --manifest-path rust/Cargo.toml -p infera-router
cargo build --manifest-path rust/Cargo.toml -p infera-router --release
```

具体本地测试结果见 [VALIDATION.zh-CN.md](VALIDATION.zh-CN.md)。测试覆盖加和记账、同前缀不共享、无 D KV 元数据、未知输入、并发选择、drop 释放、P/D 分开释放、P 响应体时序、GPU/host 副本、层间淘汰、rank 隔离、清空、shadow 不改变选择及新评分选目标。

有机器后依次进行：

1. 固定源码和二进制哈希，检查启动日志中的模式；不要沿用旧流程重复打 R1 补丁。
2. 用实际 tokenizer/template 对账 D 输入长度，验证 rank 路由与请求结束/取消后的账本。
3. 检查本次 Unified Radix 的各层事件、实际 GPU/host 驻留及恢复行为；若 host 事件不完整，先保持 R3 shadow/关闭 host credit。
4. 验证 HTTP/NATS 实际使用路径、错误与取消、R1–R4 组合跑通及 Router CPU/内存开销。
5. 正式 A/B 每轮重启 P/D，固定容量、4K chunk、C80、884 warmup 和 3600 秒窗口；遵循已有可比启动要求，不恢复旧的默认复用 P/D 方案。

尚未执行 GPU 跑通或性能测试，尚无这份实现的吞吐收益结论。
