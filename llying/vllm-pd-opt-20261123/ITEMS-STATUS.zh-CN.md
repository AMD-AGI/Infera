# 优化项状态表（持续维护；11-25 05:30 建）

来源：计划 `../vllm-pd-opt-plan-20261123/PLAN.zh-CN.md`（下表"计划"列写节号）、执行中发现的项（"执行"）、上游调研（"上游"）。
状态四类：**已采用**（在推荐配置里）、**已试不采用**、**判断不做**（理论或实测判断收益小 / 不允许）、**待尝试**（预计有收益，还没做）。
收益按"TTFT 每秒约 3% 吞吐"、"decode 步长变化 × 0.3–0.6"估；成绩都是单次，成对比较见 `README.zh-CN.md`。推荐配置 = a25 配置（`SUMMARY-20261125.zh-CN.md`）。

## 1. decode 调度层（计划阶段 1）

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| H3 元数据里的主机同步（sync skip 覆盖层） | 计划 1C / 2.2 | **已采用** | GPU 忙碌 83–92% → 99.7% | a1 +9.1% |
| H2 抢占后整段重算 → decode 前缀缓存 | 计划 1B-3 | **已采用** | 重算超出时间 10.8% → 0.2% | a2 +4.2% |
| DP 引擎均衡（U28） | 计划 3 节 | **已采用** | 最忙 / 平均 1.28 → 1.13 | a3 +1.1% |
| KV 感知 DP 选 rank（W = 14） | 计划 1B-2 | **已采用** | KV 满的时间 53% → 47%、抢占减半 | a5 +1.2%（含 GEMM 表） |
| H1 新请求首步不均匀 / 首步均匀化 | 计划 1A | 判断不做 | 逐步日志 FULL 100%，不成立 | 0 |
| decode 准入余量 `--watermark` | 计划 1B-1 | 已试不采用 | 测具 b0 里与前缀缓存一样消掉重算，P4D4 用前缀缓存 | — |
| 抢占交回路由重走 prefill | 计划 1B-3 | 判断不做 | 抢占已降到每小时 2–14 次 | < 0.1% |
| ROCm sparse MLA 的 `update_draft_decode_metadata` | 计划 1C | 判断不做 | GPU 已 99%+ 忙，异步调度掩住 host 开销 | < 0.3% |
| DP 均衡权重按 Triton 后的新步长拟合重调（W 14 → 约 5） | 执行 | 判断不做 | 不均现主要是请求数（约 1.2 ms / 步），能省约 0.2 ms | 约 0.15% |
| decode 按会话路由到持有上一轮 KV 的 rank | 执行 | **待尝试** | decode 前缀命中约 10%，可减拉取量；与 DP 均衡冲突，KV 周转约 9 s 与思考间隔相当 | 0.3–0.5%（不确定） |
| decode DCP（上下文并行） | 计划"不做"、用户问 | 判断不做（第 6 节） | 要求 TP > 1（DP4 × TP1 不能开）；v0.34 的 ROCm 稀疏 MLA 不支持；公开数据 DCP 比不开慢 2–9% | 上限 2.8% 步长，开销更大 |

## 2. decode kernel（计划阶段 2）

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| dense BF16 GEMM 调优表（FlyDSL） | 计划 2A | **已采用** | a12 加 hipBLASLt 只再快约 0.1 ms，接近上限 | a5 里 |
| MoE 分发前量化成 MXFP4（FP4 dispatch） | 执行（替代计划 2B） | **已采用** | hidden all-gather 20–29 → 5–8 µs | a7 +0.6% |
| 两次 4 字节 all-gather（topk ids / weights）打包 | 计划 2B | 判断不做（单独拼接 / 拆分） | 拼接 / 拆分的小 kernel 抵消收益 | — |
| 把 topk ids / weights 写进 FP4 量化 kernel 的同一缓冲区，三次 AG 合成一次 | 执行 | **待尝试** | 两次小 AG 现在每步约 0.9 ms（约 2% 步长，trace 每次 6.1 µs × 150） | 0.3–0.5% |
| 长上下文 indexer top-k（Triton 第三版） | 计划 2D | **已采用** | 每次 105–186 → 约 70 µs；生产 716 万行逐行精确 | a18 +1.4%，与阈值叠加 a22 +4.5% |
| 稀疏 MLA 的局部性（Triton 输出时每次比 aiter 多约 6 µs） | 执行 | **待尝试（先查）** | v3-037：32.9 对 26.7 µs / 次，每步约 0.15 ms | 约 0.1–0.2% |
| Triton top-k 在异常节点上回退 aiter | 执行 | **待尝试（稳健性）** | v3-071 上 Triton 约 130 µs、aiter 105 µs | 防坏节点 |
| 采样 `_topk_topp_kernel` | 计划 2C | 判断不做 | 生产每步约 0.21 ms（0.5% 步长） | < 0.2% |
| KF MoE gemm1 | 计划 2E | 判断不做 | MoE 已在带宽上限（约 6 TB/s），KF 无上游 | < 0.3% |
| gluon paged MQA logits（U12）、aiter AG/RS 进 FULL graph（U13） | 计划 2E | **已采用**（覆盖层 / fork） | — | 上游化待做 |
| KF sparse MLA（#47278）、融合 indexer（#47276） | 计划"不做" | 判断不做 | A 线比现有实现慢 | — |
| EP / MoRI / DBO（单节点） | 计划 4 节 | 判断不做 | A 线单节点都比默认慢 | — |

## 3. decode KV 容量（计划阶段 3）

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| decode `--max-num-batched-tokens 2048` | 计划 3A | **已采用** | KV +6.3%，KV 满的时间 47% → 35% | a6 +1.0% |
| decode `--gpu-memory-utilization 0.97` | 执行 | 已试不采用 | KV +4.1%，成对 +0.2%（波动内） | — |
| 目标模型 dense 层在线 FP8 | 计划 3B | **待尝试**（11-25 用户：做，按实测正确性与性能决定是否采用；执行表 9b） | 改数值；InferenceX 规则上允许（eval 通过、草稿全排除，第 9 节） | 每 rank 权重约 −12 GiB → KV 约 +9% |
| PTPC FP8 草稿 MoE | 计划 3 节 | 判断不做 | InferenceX Check 13 | — |

## 4. prefill 与调度（计划原判断"P4D4 prefill 不是瓶颈"——**这个判断错了**）

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| prefill `--long-prefill-token-threshold`（4096 → 配分块 4096 时 2048） | 执行 | **已采用** | 准入排队 1.15 → 0.70 s；超长 prefill 不再饿死小请求 | a19 +1.6% |
| prefill `--max-num-batched-tokens 4096` | 执行 | **已采用** | PP4 流水线 0.82 → 0.58 s | a20 +0.8%；a25 合计 +5.2% |
| 超长 prefill 按上下文长度限分块（调度覆盖层） | 执行 | 判断不做（现在） | 推荐配置下 prefill 段 > 20 s 的请求 17 个、占 prefill 总时间 2.1%（a8 配置 164 个、19.1%）；95 万 token 那个窗口已与相邻窗口持平 | ≤ 0.2% |
| 小请求优先（SPF / 按未命中 token 排序准入；上游 #66705、#66558、#65339，见第 7 节） | 计划 3 节（sglang +15%） | **待尝试** | 未命中 < 2,048 的请求占 54%，prefill 段均值 1.36 s，计算只要约 0.1 s；平均准入排队 0.65 s | 0.3–0.6% |
| 小请求单独组 microbatch（减 PP4 流水线延迟） | 执行 | 待评估 | 与大分块同批时每级时长由大分块决定 | 约 0.3% |
| CPU 层 1.9 TB（精确锁页）+ 缓存感知准入 | 计划 3 节 | **已采用** | prefill 命中与 sglang 持平 | — |

## 5. PD 与 KV 传输

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| Mooncake 串行下发（路由等 prefill 回复后再发 decode） | 计划 / U16 | **已采用** | 防并发预留死锁 | — |
| Mooncake 并行度 / 完成信号（`MC_WORKERS_PER_CTX` 现 2、`MC_RDMA_SIGNAL_INTERVAL=32`、`MC_MAX_WR=4096`；再往后 MC#5680 并行提交） | 执行 + 上游 | **待尝试（只改配置）** | 8 块网卡平均各约 1.9 GB/s，远没到带宽；单段约 8 GB/s、每描述符 4–6 µs，像是处理速率受限 | KV 拉取 0.48 s 的一部分，≤ 1% |
| 描述符合并（prefill / decode 分配连续块） | 执行 | **待尝试（改代码）** | 每段约 3.3 万个 44 KB 的描述符 | 同上，与上一项不叠加 |
| prefill 按分块边算边推 KV（上游 #65613） | 执行 + 上游 | **待尝试（改代码，大）** | 把传输藏到 prefill 计算后面 | 约 1% |
| 同节点 HIP IPC（XGMI）传输 | 执行 | 判断不做（第 10 节） | 部署两次没起来；更关键的是之前的实测：发送端 GPU 有计算时 HIP 只剩 0.36–0.4 GB/s，比 RDMA 慢得多 | — |
| 跨节点 P4D4 | 用户问 | 判断不做 | 网卡没饱和（见上），跨节点不改变单段处理速率，还失去 XGMI 的可能，且与 sglang 单节点对标不同口径 | — |
| Mooncake 重传 / ack 超时（`MC_IB_TIMEOUT=18` 约 1 s 一次） | 执行 | 待查 | a28 A：ionic_0 77 次、ionic_1 18 次 ack 超时 | 只影响尾部 |

## 6. decode DCP 调研（11-25 05:15）：**判断不做**

- **上游支持**：DCP 已正式支持（MLA 首个 PR [#23734](https://github.com/vllm-project/vllm/pull/23734)；a2a 通信 [#34883](https://github.com/vllm-project/vllm/pull/34883)；
  query 复制 [#45964](https://github.com/vllm-project/vllm/pull/45964)，GLM 默认开 [#50382](https://github.com/vllm-project/vllm/pull/50382)）。CUDA 的稀疏 MLA（DSA）支持 DCP
  （[#46076](https://github.com/vllm-project/vllm/pull/46076)、[#46514](https://github.com/vllm-project/vllm/pull/46514)）。
  **ROCm 的 `ROCM_AITER_MLA_SPARSE` 在 v0.34.0 里 `supports_dcp=False`**，main 从 [#65685](https://github.com/vllm-project/vllm/pull/65685)（11-12）起才支持
  （限 gfx950、`fp8_ds_mla` KV）；GLM 走的 AMD 路径还要 [#68935](https://github.com/vllm-project/vllm/pull/68935)（draft，不加时启动报错或 GSM8K 0.01）。
  ROCm 稀疏路径上 MTP + DCP 没有（[#63859](https://github.com/vllm-project/vllm/pull/63859) 草稿明确不支持）。P2P `MooncakeConnector` 在 main 和我们的 fork 里都没有 DCP 逻辑。
- **硬约束**：DCP 复用 TP 的 rank，要求 `tp % dcp == 0`，**DP4 × TP1 不能开**，必须改成 TP4 × DCP4 × EP4（放弃 DP attention）。
- **公开数据都是负的或持平**：MI355X Kimi-K3 TP8 下 DCP8 比 DCP1 的 TPOT 高 8–9%（[#61435](https://github.com/vllm-project/vllm/pull/61435)）；sglang ROCm GLM TP4 + DCP4 比 TP4
  交互 −9%、吞吐 −2%（[sglang#42618](https://github.com/sgl-project/sglang/pull/42618)）；分片 indexer 在 3.4 万上下文时每步慢 0.5–1.5 ms，15 万以上才变快。
  DCP 的好处是 KV 不重复存（DP4 本来就不重复）和单个超长请求分摊到多卡。
- **账**：收益上限是现在的不均 1.2 ms / 步（2.8% 步长），DCP 每层多出 query 收集、输出 / LSE 交换、indexer top-k 合并和 o_proj 归约，按上面的数据在 42 ms 步长下约 1–4 ms，
  超过收益。要用还得回移植 ROCm 稀疏 DCP、改 KV 格式、自己补 MTP × DCP、修 FULL graph 下 aiter LSE 的问题、改 MooncakeConnector，几个人周、正确性风险高。
- 结论：不做。不均的剩余部分（主要是请求数）留给 DP 选 rank（已有 KV 感知，见第 1 节），超长请求（1M 级）可以考虑单独限额或打散。
- 调研记录：[vLLM DCP support research](c0b1a437-7dae-4132-9ccc-62a6f91f2a7d)。

## 7. 上游可移植项（11-25 05:20 调研：[Upstream vLLM optimization survey](d2956fc0-c063-407b-a531-2ce07c1e887b)）

v0.34.0 是 11-08 从 main 分出的，下表标"合入"的都**不在**我们的 fork 里；收益数字是 PR 作者自报的，不是我们测的。
"数值"列：会不会改变模型数值（要过门禁 / 合规审批）。镜像里的 Mooncake 是 v0.3.14（`e1ea5d31` + 补丁）。

| 优先 | PR | 状态 | 做什么 | 对应我们的瓶颈 | 作者数据 | 改动 / 数值 |
|---|---|---|---|---|---|---|
| 1 | [#68560](https://github.com/vllm-project/vllm/pull/68560) | open | PP 下单个长 prefill 也切块，使它的分块能跨 stage 流水 | prefill 计算 / 流水线 0.69 s、超长 prefill | MI355X PP4 × TP1 agentic（10 万命中 + 1 万新 token，阈值 4096）：单请求 TTFT 433 → 268 ms；2 × PP2 在 3–7 req/s 下 TTFT p50 −22% 至 −38% | +94 行 / 不改数值 |
| 1 | [#68614](https://github.com/vllm-project/vllm/pull/68614) | 合入 11-25 | PP>1、DP=1 时空步不占住 batch queue；按积压把 prefill 切成约 pp_size 份填满各 stage | 同上 + 准入 | PP4 × TP1、180k prompt（162k 命中）：中等负载 TTFT −42% 至 −60%，接近饱和 −14% 至 −22% | 12 个文件，改 async scheduler / 不改数值 |
| 2 | Mooncake 环境变量 `MC_RDMA_SIGNAL_INTERVAL=32 MC_MAX_WR=4096`（[MC#4542](https://github.com/kvcache-ai/Mooncake/pull/4542)，RFC [#63006](https://github.com/vllm-project/vllm/issues/63006)） | 已在 v0.3.14 | 每 32 个 WR 才发一次完成信号，减少 CQE 处理 | KV 拉取 0.48 s（描述符多） | RFC 实测用这组参数 | 零代码 / 不改数值 |
| 2 | [#67656](https://github.com/vllm-project/vllm/pull/67656) | 合入 11-16 | MooncakeConnector 用数组接口提交描述符（不转 Python list） | 同上（vLLM 侧提交开销） | 无单列数字 | 8 个文件 / 不改数值 |
| 2 | [MC#5680](https://github.com/kvcache-ai/Mooncake/pull/5680) | 合入 11-24（不在 v0.3.14） | TE 把大批描述符切成子批，线程池并行提交（`MC_BATCH_SUBMIT_THREADS`） | 同上 | TE 带宽最多 +70%，vLLM PD 的 KV 带宽最多 +106% | 重编 Mooncake / 不改数值 |
| 3 | [#61985](https://github.com/vllm-project/vllm/pull/61985) | open | gfx950 上 GLM-5 / DSV3.2 的 indexer K cache 用 MXFP4（MTP 草稿也用） | decode 步长（indexer）+ 传输字节 | GLM-5.2-MXFP4、MI355X TP4：TPOT −5.4% / −5.9%，MTP3 −6.8%，TTFT −7% 至 −8% | 7 个文件 / **改数值**（indexer 量化，要审批与 eval），会绕开 U12 |
| 4 | [#65519](https://github.com/vllm-project/vllm/pull/65519) | open | CPU→GPU 加载每次提交的描述符设上限，不长时间占住 worker 线程 | prefill CPU 层读回、排队 | GLM-5.2 AgentX + CPU 层（B200 TP8 c32）：706 → 859 output tok/s，TTFT 23.8 → 17.4 s | 3 个文件 / 不改数值；要与精确锁页、`hipMemcpyBatchAsync` 一起测 |
| 4 | [#69540](https://github.com/vllm-project/vllm/pull/69540)、[#68845](https://github.com/vllm-project/vllm/pull/68845) | open | CPU LRU 去掉 O(n) 堆重建停顿；合并相邻命中区间 | prefill 停顿 / 准入 | 1 TiB 层每小时 54 次、共 110 s 的停顿消失；140k 命中准入 25 → 16 ms | 小 / 不改数值 |
| 5 | [#66558](https://github.com/vllm-project/vllm/pull/66558) + [#66361](https://github.com/vllm-project/vllm/pull/66361)，对 [#66705](https://github.com/vllm-project/vllm/pull/66705) | open | 缓存感知准入把 CPU 层命中算进去（+ 最长等待）；或按剩余未缓存长度排序（SPF） | 准入排队 0.65 s | 无端到端数字 | 中 / 不改数值；两种 A/B |
| 5 | [#65339](https://github.com/vllm-project/vllm/pull/65339) | open | 运行中的 chunked prefill 不吃满每步预算，给已有缓存的等待请求留预算 | 准入排队 | 无 | 2 个文件 / 不改数值 |
| 6 | [#66799](https://github.com/vllm-project/vllm/pull/66799) | 合入 11-14 | DP / graph 填充行的 expert id 设为 −1，MoE kernel 跳过 | DP 不均下的填充行 | CUDA DSV3.2 DP4 +3% 至 +8%（每 rank 33–128 行时 −5.6%） | 6 个文件 / 不改真实行；先确认 aiter 融合共享专家路径和对 −1 的处理 |
| 7 | XGMI HipTransport（RFC [MC#5041](https://github.com/kvcache-ai/Mooncake/issues/5041)） | RFC open | 同节点 P/D 走 XGMI，小拷贝合成 batched kernel | KV 拉取 | 8 × MI355X、GLM-5.2 形状：1→4 拷贝 11.6 → 5.8 ms，大块约 200 GB/s | 高：要 Mooncake main + 未合入 PR，IPC 模式与 GEM / VMM 池冲突（即 a30 卡住的地方） |
| 8 | [#63079](https://github.com/vllm-project/vllm/pull/63079) | open | KV 紧时只缺草稿块的 decode 本步不带草稿，不抢占别人 | decode 抢占 | 只有模拟器数据 | +52 行 / 不带草稿的行可能让步落出 FULL graph |
| — | [#65613](https://github.com/vllm-project/vllm/pull/65613) | open | Mooncake 按层 / 按分块边算边发（`transfer_mode`） | KV 拉取藏到 prefill 后面 | 无 GPU 数据 | 约 3k 行，PP4 / DSA / MTP 未验证 |
| — | [#63425](https://github.com/vllm-project/vllm/pull/63425) | open | 多轮 P/D 中 prefill 从 decode 拉上一轮输出的 KV（`hold_kv`） | 每轮少算约 1k token | — | 中；decode 要多留 KV |
| — | [#69701](https://github.com/vllm-project/vllm/pull/69701) | open | async 调度下提早上报 KV 已到达 | decode 排队 | 每请求 −3 至 −15 ms | 中 / 收益小 |
| — | [#67897](https://github.com/vllm-project/vllm/pull/67897) | open | MTP 草稿只算部分行的 logits（不读整个 1.9 GB 的 LM head） | dense GEMM | 估计每步约 0.7–1 ms | 25 个文件 / **改草稿**（要确认 InferenceX 对草稿的规定） |
| — | [#63847](https://github.com/vllm-project/vllm/pull/63847)、[#59421](https://github.com/vllm-project/vllm/pull/59421)、[#63878](https://github.com/vllm-project/vllm/pull/63878) | 合入 / open / 合入 | ROCm 的 persistent / sampled top-k；放开 indexer native decode 的 next_n ≤ 2；indexer prefill 不重复 gather K | top-k（已有 Triton）、indexer、prefill | #63878：512K prefill −7% | 小；#59421 有正确性 bug #64855 |

- 看过不推荐：[#60889](https://github.com/vllm-project/vllm/pull/60889)（GLM-5.2 MonoKernel 只支持 TP8）、[#69327](https://github.com/vllm-project/vllm/pull/69327)（DP≠1 时关闭）、
  [#66771](https://github.com/vllm-project/vllm/pull/66771)（MXFP8 dispatch 不如已有的 MXFP4）、[#69522](https://github.com/vllm-project/vllm/pull/69522)（只在 dense 是 MXFP4 时有用）。
- 已列项的上游状态：U17 / U19 已开成 [#69616](https://github.com/vllm-project/vllm/pull/69616) / [#69617](https://github.com/vllm-project/vllm/pull/69617)；
  [#67232](https://github.com/vllm-project/vllm/pull/67232)（open）也删掉了 `ROCM_AITER_MLA_SPARSE` 重算后的 `synchronize()`。

## 8. 下一阶段执行表（按顺序做，做完一项更新一行；实验细节照旧写在 `README.zh-CN.md`）

基线：a25 配置，v3-037 上两次 56,342 / 56,448（平均 56,395），同节点轮间波动约 0.2%。每项默认在 v3-037 只跑 B 侧对比这个基线，差值 < 0.5% 按波动看待。
"数值"列：不改 = 只改调度 / 传输 / 实现；改 = 需要门禁 + InferenceX eval（第 9 节）。所有改代码的项最终都要进 vLLM / Mooncake 上游（InferenceX 不允许引擎补丁）。

| 顺序 | 项 | 类型 | 数值 | 针对 | 估计收益 | 状态 | 结果 / 记录 |
|---|---|---|---|---|---|---|---|
| 1 | Mooncake 环境变量 `MC_RDMA_SIGNAL_INTERVAL=32 MC_MAX_WR=4096`（prefill、decode 都设） | 配置 | 不改 | KV 拉取 0.48 s | ≤ 0.7% | **进行中**（a31，310832，v3-181 成对：B = 本项，A = a25 配置；v3-037 / v3-239 被他人占用） | **B 作废**：07:32 起 prefill PP2 的 Mooncake 发送反复 30 s 超时（141 次），请求失败、prefill 卡满，47,530 无效；失败前 300–1800 s 窗口 59,556（其他节点的 a25 配置 60.8–61.9k）。等 A 侧判断是参数还是节点 |
| 2 | 再加 `MC_WORKERS_PER_CTX=4`（视第 1 项结果） | 配置 | 不改 | 同上 | 同上，不叠加 | 待做 | — |
| 3 | 回移植 #68560 + #68614（PP4 prefill 调度：长 prefill 跨 stage 流水、按积压切分） | 改 vLLM 调度器 | 不改 | prefill 计算 / 流水线 0.69 s、准入 | 0.5–1.5% | **代码与单测就绪**（`scripts/make_ppsched_overlay.py`，env 开关；a32 待提交） | README"执行表第 3 / 5 项的覆盖层"节 |
| 4 | 回移植 #67656 + 重编 Mooncake（MC#5680 并行提交） | 改 connector / 重编 | 不改 | KV 拉取 | ≤ 0.7%（与 1、2 不叠加） | 待做 | — |
| 5 | CPU 层加载路径 #65519、#69540、#68845 | 改 offload | 不改 | prefill CPU 层读回、停顿 | 0.2–0.5% | **#69540 + #68845 代码与单测就绪**（`scripts/make_kvoffload_overlay.py`）；#65519 不做 | #65519 默认每步 8192 个描述符，会把我们约 4.8 万描述符的读回拆到约 6 步 |
| 6 | 准入 #66558（+ #66361）对 #66705（SPF），或 #65339 | 改调度器 | 不改 | 准入排队 0.65 s | 0.3–0.6% | 待做 | — |
| 7 | #66799 MoE 填充行跳过（先确认 aiter 路径） | 改 MoE 路由 | 不改真实行 | DP 不均的填充行 | 0.3–1%（不确定） | 待做 | — |
| 8 | FP4 量化 kernel 里顺带 all-gather topk ids / weights（三次 AG 合一） | 改覆盖层 / 上游 | 不改 | 每步约 0.9 ms | 0.3–0.5% | 待做 | — |
| 9 | #61985 MXFP4 indexer K cache（只用于目标模型层，草稿层保持默认） | 改 vLLM | **改** | indexer、传输字节 | 作者数据 TPOT −5–7% | **待尝试**（11-25 用户：做，按门禁 / eval 正确性 + 成对性能决定是否采用） | 第 9 节；采用后要进上游、写进 recipe |
| 9b | 目标模型 dense 层在线 FP8（草稿 / MTP 层全排除） | 改 vLLM / 覆盖层 | **改** | KV 容量（权重约 −12 GiB / rank） | KV +9%；sglang 上吞吐持平，vLLM 受 KV 限可能不同 | **待尝试**（11-25 用户：按门禁 / eval + 成对性能决定） | 第 9 节 |
| 10 | 按层 / 按分块边算边发 #65613 | 改 connector（大） | 不改 | KV 拉取藏到 prefill 后面 | 约 1% | 后放 | — |
| 11 | decode 按会话路由 | 改路由 / LB | 不改 | 拉取量、KV 周转 | 0.3–0.5%（不确定） | 后放 | — |
| — | 同节点 HIP IPC（XGMI）传输 | — | 不改 | — | — | **判断不做** | 第 10 节：发送端 GPU 有计算时只剩 0.36–0.4 GB/s |
| — | #67897 草稿稀疏 LM head | — | 改草稿 | — | — | **不允许** | 第 9 节：减少草稿 FLOPs 违反 Check 13 |

## 9. InferenceX 对"改数值"的规则（11-25 06:30，`SemiAnalysisAI/InferenceX` main：`CONTRIBUTING.md`、`inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md`、`.github/codeowner-signoff-verify-prompt.md`）

- **目标 / verifier 降精度：允许，但要过 eval**。Check 8："Target/verifier FLOPs at lower precision are fine when evals pass under Check 2's cohort comparison"；
  拿精度换速度的手段（`online_quant_config`、为抵消别处损失而开的 FP8 KV、indexer 跳过 / 共享等）必须有 eval 覆盖，"上游已合入"不能抵消精度损失。
  总原则："只用对精度敏感的客户在生产里也会用的优化"。
- **不允许减少模型结构的 FLOPs**（Check 8(a)）：例如没有原生支持时跳过某些层的 indexer、裁层 / 裁专家 / 裁头。"同样的计算跑得更快"可以。
- **草稿必须"按发布的样子跑"**（Check 13）：草稿权重 / 激活 / 计算不能低于发布精度；不能"以作者没发布的方式减少草稿 FLOPs"；不能换草稿 checkpoint；
  不能为草稿打补丁。草稿 KV cache 的存储 dtype 只能是框架默认、或 bf16 / fp16 / fp32 / fp8 / fp8_e4m3 / fp8_e5m2（要通过启动参数设置并记录在 golden AL 身份里）；
  int8 / int4 / MXFP8 / FP4 / NVFP4 / MXFP4 只有是框架默认时才行。合成接受下的 AL 不能作为证据。
- **引擎必须原样运行**：不允许任何引擎补丁（`.patch`、改 site-packages、monkey-patch、覆盖容器文件、装改过的 wheel），只有附 waiver 的例外；
  vLLM 提交要用上游镜像（main 构建或获批的分支），所有生产有用的参数要写进已**合入**的 vLLM recipe。AgentX 的 stream interval ≤ 60（我们是 30）。
- **对我们各项的结论**：

  | 项 | 判断 |
  |---|---|
  | #61985 MXFP4 indexer K cache | 目标模型层：规则上允许，前提是进上游镜像、写进 recipe、eval 通过；**草稿层不允许**（MXFP4 不在草稿 KV 允许列表里，PR 现在也作用于 MTP 草稿，要拆开） |
  | #67897 草稿稀疏 LM head | **不允许**（减少草稿 FLOPs） |
  | 目标模型 dense 层在线 FP8 | 规则上允许（eval 通过、`exclude_layer` 覆盖整个草稿、上游支持）；11-25 用户：做，按实测正确性与性能决定是否采用 |
  | PTPC FP8 草稿 MoE | 不允许（确认） |
  | 已采用的 sync skip、KV 感知 DP 均衡、FP4 dispatch、Triton top-k、prefill 两个参数 | 数值不变或门禁干净，但前四项现在是覆盖层，**正式提交前必须进 vLLM main**；两个 prefill 参数要写进 recipe |
  | 合成接受长度 2.99 | 要与 InferenceX golden AL 曲线里 GLM-5.2、对应 thinking 模式、草稿长度 3 的值一致（待核对） |

## 10. KV 传输是不是瓶颈（11-25 06:30）

- **不是主瓶颈，是 TTFT 里较大的一段**：推荐配置下 KV 拉取约 0.48 s，占 TTFT 2.81 s 的 17%；通道周期 104 / 3.662 req/s ≈ 28.4 s，KV 拉取约占 1.7%。
  全部去掉按"TTFT 每秒约 3%"估上限约 +1.4%，减半约 +0.7%。周期里 decode 约占一半多、客户端间隔约三分之一，决定吞吐的仍是 decode 步长。
  RDMA 传输由网卡 DMA 完成，不占 decode GPU 的算力，不影响 ITL。网卡平均各约 1.9 GB/s（8 块均匀），远没饱和；慢在单段的描述符处理（约 3.3 万个 44 KB）。
- **HIP（XGMI）传输在生产负载下比 RDMA 慢得多**（`AMD-AGI/Infera` 分支 `dev/pd_opt/glm_5.2_agentx`，`yihou/glm52.v0521-g4tune-xfer.packup_20261020/README.md`
  与 `analysis/ledger/poll.md` 10-19 条，v3-021 上 Mooncake `batch_transfer_sync_write`、生产 MC 环境）：两卡空闲时每对约 47–51 GB/s，
  **发送端 GPU 在跑 bf16 矩阵乘、≥ 2 对同时传时降到 0.36–0.4 GB/s（约慢 50 倍）**，接收端有负载不受影响；`MC_HIP_STREAM_PRIORITY=high`、`HSA_ENABLE_SDMA=1` 无效，
  `HSA_ENABLE_SDMA=0` 挂住；机制没查清。sglang 同节点 P4D4 走 HIP 时 prefill 的 transfer_kv p50 7.31 s、平均 9.98 s，KV 速度 p50 0.86 GB/s。
  prefill GPU 一直在算，正是这个最坏情况；我们现在 RDMA 回环每段约 8 GB/s、约 0.2 s。**所以同节点 HIP IPC 不做**（a30 两次没起来也就不再排查），
  上游 RFC MC#5041 的 XGMI 数据也是空闲时的。另见同目录 `glm52.samenode-p4d4.agentx-c40.packup_20260922/analysis/mooncake_samehost.yihou.md`：
  同节点时 Mooncake 默认优先选 HIP（优先级 4 > rdma 2），我们用 `MC_DISABLE_HIP=1` 强制走 RDMA 是对的。
