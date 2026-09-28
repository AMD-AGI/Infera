# Decode 有效并发与 InferenceX 实际 Router 配方核查

日期：2026-09-28。本文补充 R1–R4 总结。只读取已有采样、公开结果与源码，没有启动 GPU 实验。

## 1. Decode 有效并发

这里把有效并发定义为：**采样时 D 引擎正在运行的生成请求数，先对 8 个 DP rank 求和，再对正式窗口取平均**。指标为 `sglang:num_running_reqs`，不包含尚在等待 KV 传输或预分配的请求。它是引擎 running batch 的观测值，不代表每一个瞬间都有同样数量的请求执行 GPU kernel。

C80 是客户端并发配置，不能直接当作 D 有效并发。下表的“每 rank”是总并发除以 8，不是将 TP 的同一个请求重复计算。采样约每 2 秒一次，缺失采样不补零。

| 策略 | 正式统计窗口 | D 平均有效并发，总计 | 平均每个 D rank | 等待 KV 的平均请求数，总计 |
|---|---|---:|---:|---:|
| A0 原策略 | 完整 60 分钟 | 36.32 | 4.54 | 23.31 |
| R1-only，G0 | 完整 60 分钟 | 36.62 | 4.58 | 17.46 |
| R1+R4 | 完整 60 分钟 | 23.61 | 2.95 | 50.70 |
| A0 原策略，R2 对照窗口 | 前 43 分钟 | 34.79 | 4.35 | 22.10 |
| R2-on | 前 43 分钟 | 35.13 | 4.39 | 21.79 |
| R3-on | 无可比正式性能实验 | — | — | — |
| R4-only | 未单独做性能实验 | — | — | — |

不要直接拿 R2 的 43 分钟均值与其他策略的一小时均值解释收益。R3 只有旁路观察，当时没有改变实际派单，且 P chunk 为 8K，不能伪装成 R3-on / 4K 性能数据。

这些数值进一步说明：

- R1 的有效并发只略增，但等待 KV 的平均请求数明显下降。R1 的收益不能概括成“D batch 大幅增大”；P 等待和端到端延迟改善同样重要。
- R2 有效并发基本相同。它改善的是 KV 在不同 rank 间的分布与少数分配长尾，没有显著增加正在生成的请求数。
- R1+R4 相对 R1-only 的有效并发减少约 35.5%，等待 KV 的请求数则大增，支持“P 供给恶化，D 更多时间在等 P”的解释。

“等待 KV”包含 P 排队、P 计算和传输相关等待，不能全部算成网络耗时。两列也不是严格守恒拆分，不能要求相加等于 C80。各轮仍是不同节点/时间的历史对照。

精确数值、样本数、分位数与原始文件路径见 [decode-concurrency.json](decode-concurrency.json)。

## 2. InferenceX 不能概括为一种 NVIDIA Router 策略

本次按本 case 相关的 **GLM-5.2 AgentX** 查找，而不是把所有 NVIDIA 模型和后端混在一起。核查链路为：公开结果 API → 对应 GitHub Actions 运行 → 运行页面所示提交 → 该提交的 master config 和 recipe → 固定 Dynamo 源码中的路由逻辑。

公开结果返回的身份信息已保存在 [published-glm52-identities.json](published-glm52-identities.json)。配方提取字段、源链接与 SHA256 见 [inferencex-recipes.json](inferencex-recipes.json)。共读取 28 个相关配方；下表区分已公开成绩与仅在配置中存在的分支。

| 已公开 GLM-5.2 成绩 | 对应运行与源码版本 | 配方指定的策略 |
|---|---|---|
| GB300 / Dynamo-SGLang，2026-08-16，含分离部署 C45/C48/C128/C192 | [31825558603 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31825558603/attempts/1)，`b384c40` | `router-mode=kv`；Router 会话亲和 TTL=3600 秒；多 Frontend；Nginx 按 `X-Dynamo-Session-ID` 保持会话；客户端发送该 header |
| GB200 / Dynamo-SGLang，2026-08-19，含分离部署 C45/C48/C128 | [32207758126 attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32207758126/attempts/2)，`6130a8b` | 同样是 KV 模式＋3600 秒会话亲和，客户端提供 session header；实际安装配方固定 Dynamo `71eb001e…` |
| B200 / Dynamo-SGLang，2026-09-15，分离部署 C48/C64 | [34580793856 attempt 4](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34580793856/attempts/4)，`e937ff6` | KV 模式＋3600 秒会话亲和；使用镜像内 Dynamo，配方 `install=false`，master 标记 `1.5.0.dev20260909` |
| H200 / Dynamo-SGLang，2026-09-18，当前返回的 C2/C4/C8/C12/C16 为聚合部署 | [35177419428 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35177419428/attempts/1)，`f8fbcc2` | 聚合配方同样启用 KV 模式＋3600 秒会话亲和；镜像内 Dynamo，master 标记 `1.5.0.dev20260913` |
| GB300 / Dynamo-TensorRT-LLM，2026-09-11，分离部署 | [34413290524 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34413290524/attempts/1)，`3353a10` | KV 模式，但显式 `no-kv-events=true`；温度 0；Router 会话亲和 TTL=14400 秒；客户端发送 session header，后端还配置 conversation affinity |

另有 H200 分离部署配方把 `DYN_ROUTER_TEMPERATURE` 设为 `10000000`，与零温度最小分数选择不同；这轮公开 API 返回的 H200 成绩是聚合部署，**不能用这份分离部署配方解释这些 H200 成绩**。

版本标记也不能直接当作安装事实：GB200 对应历史 master 的部分条目标记 Router `1.2.1`，实际 recipe 却安装 `71eb001e…`；GB300 TensorRT-LLM 的 master 标记 wheel `1.4.0.dev20260807`，大多数对应 recipe 实际改为安装 `2cbbdc86…`，C1 recipe 才保留该 wheel。本文优先报告 recipe 的安装指令，未把它冒充实机二进制哈希。

## 3. 与我们最相关的 SGLang 策略怎么工作

**核心是 KV 模式加会话绑定，不是每一轮请求都重新在所有 P/D rank 中自由选择。**

GB200/GB300 配方固定的 Dynamo 为 `71eb001e17fa73c742f0afe1a6ed96836cb135fd`。核对该版本后，逻辑为：

1. 客户端通过 `X-Dynamo-Session-ID` 标识会话。Nginx 会话亲和用于把请求送回同一个 Frontend；它与 Router 内部绑定 worker/rank 是两个层次。
2. 未建立绑定时，Router 使用 KV 调度选择目标。默认温度为 0，选择最小成本候选；P 成本涉及在途 prefill 工作、新请求和缓存抵扣，普通分离部署的 D 选择则按负载，关闭缓存 overlap 抵扣及 prefill 工作项。
3. 已有会话绑定时，该 worker 和 DP rank 作为 pin 进入选择，优先沿用目标。正常情况下不会仅因另一个 rank 分数更低就重新分配；目标失效等错误有失效/重选逻辑。因此不能把绑定描述为永不变化。

源码依据：

- [会话绑定查询、选择与失败重试](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/kv_router/push_router.rs)：`select_with_affinity`。
- [将 affinity 合并为 worker/rank pin](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/kv_router/push_router/selection.rs)：`select_worker`。
- [评分与 pinned candidate 分支](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/kv-router/src/scheduling/selector.rs)。
- [普通分离部署 D 的 override](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/llm/src/kv_router/prefill_router/mod.rs)：`build_decode_router_override` 将 overlap credit 设为 0、assume_kv_reuse=false、track_prefill_tokens=false。
- [Router 默认参数](https://github.com/ai-dynamo/dynamo/blob/71eb001e17fa73c742f0afe1a6ed96836cb135fd/lib/kv-router/src/scheduling/config.rs)：温度 0、overlap credit 1、prefill scale 1，额外 active-request weight 0。
- [公开成绩对应 GB300 配方](https://github.com/SemiAnalysisAI/InferenceX/blob/b384c40/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/gb300-fp4/agentic/glm5.2-agentx.yaml)。

这些 SGLang 分离部署配方还显式设置 D `disable-radix-cache=true`。因此“D 选择考虑输入引起的负载”不要求 D 开启 radix cache；会话 pin 同样不等于 D 保存了可复用前缀。

## 4. 这次确认了什么，还不能确认什么

已确认公开成绩对应的配置不是只写了框架名称：具体指定了 KV 模式、会话亲和参数和客户端 session header；GB200/GB300 所固定源码也确实存在会话 pin 路径。过去仅研究 Dynamo 的通用评分公式，遗漏了 InferenceX 配置中的会话绑定层，这是需要补正的地方。

但本次没有取得上述运行的生成后启动命令、二进制哈希及逐请求 Router 日志。GitHub Actions API 访问受到未认证限流，运行身份改用公开运行页面与 raw 源文件核对。因此本文是**与公开成绩关联的配置和源码审计**，不是运行日志级证明。不能声称已测得会话 pin 的命中比例，或确认所有启动覆盖项。

尤其不能仅凭 `router-mode=kv` 就断言引擎真实缓存事件已正确回传：GB200/GB300 若干历史 recipe 未显式填写 backend `kv_events_config`，还依赖 srt-slurm 生成行为；GB300 launcher 当时使用浮动 `main`。Dynamo Router 默认订阅 KV 事件，也不能替代引擎 publisher 的证明。这部分需要生成命令或运行日志才能闭环。TensorRT-LLM 配方则明确关闭 KV 事件，更不能描述为依赖实时事件目录的路由。

## 5. 对后续优化顺序的影响

之前的方向“保护缓存亲和，再改善 P 排队”仍成立，但应增加一个更具体的候选：**会话级 P 亲和，结合受约束的拥塞逃逸**。原因是 InferenceX 相关配置确实重视同一会话的局部性，而我们的当前 R4 会逐请求按工作量重新权衡，已经观察到大段缓存损失。

这不代表照搬绑定必然更快。固定会话可能造成长短会话负载不均，也可能绑住拥塞 rank；本次没有隔离实验能把 NVIDIA 的性能归因为会话亲和。下一步应先离线统计同一会话跨 P rank 的迁移、缓存损失和队列变化，再决定是否实现 P 会话亲和及其逃逸条件。D 会话亲和应单独评估，不能因上游启用了它就放弃 R2 对 KV 热点的已观测改善。
