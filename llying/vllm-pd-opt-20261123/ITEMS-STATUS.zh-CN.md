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
| 目标模型 dense 层在线 FP8 | 计划 3B | 判断不做 | 改数值、InferenceX 合规未批 | — |
| PTPC FP8 草稿 MoE | 计划 3 节 | 判断不做 | InferenceX Check 13 | — |

## 4. prefill 与调度（计划原判断"P4D4 prefill 不是瓶颈"——**这个判断错了**）

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| prefill `--long-prefill-token-threshold`（4096 → 配分块 4096 时 2048） | 执行 | **已采用** | 准入排队 1.15 → 0.70 s；超长 prefill 不再饿死小请求 | a19 +1.6% |
| prefill `--max-num-batched-tokens 4096` | 执行 | **已采用** | PP4 流水线 0.82 → 0.58 s | a20 +0.8%；a25 合计 +5.2% |
| 超长 prefill 按上下文长度限分块（调度覆盖层） | 执行 | 判断不做（现在） | 推荐配置下 prefill 段 > 20 s 的请求 17 个、占 prefill 总时间 2.1%（a8 配置 164 个、19.1%）；95 万 token 那个窗口已与相邻窗口持平 | ≤ 0.2% |
| 小请求优先（SPF / 按未命中 token 排序准入） | 计划 3 节（sglang +15%） | **待尝试** | 未命中 < 2,048 的请求占 54%，prefill 段均值 1.36 s，计算只要约 0.1 s；平均准入排队 0.65 s | 0.3–0.6% |
| 小请求单独组 microbatch（减 PP4 流水线延迟） | 执行 | 待评估 | 与大分块同批时每级时长由大分块决定 | 约 0.3% |
| CPU 层 1.9 TB（精确锁页）+ 缓存感知准入 | 计划 3 节 | **已采用** | prefill 命中与 sglang 持平 | — |

## 5. PD 与 KV 传输

| 项 | 来源 | 状态 | 依据 | 收益 / 估计 |
|---|---|---|---|---|
| Mooncake 串行下发（路由等 prefill 回复后再发 decode） | 计划 / U16 | **已采用** | 防并发预留死锁 | — |
| Mooncake 并行度（`MC_WORKERS_PER_CTX` 现 2、`MC_NUM_QP_PER_EP`） | 执行 | **待尝试（只改配置）** | 8 块网卡平均各约 1.9 GB/s，远没到带宽；单段约 8 GB/s、每描述符 4–6 µs，像是处理速率受限 | KV 拉取 0.48 s 的一部分，≤ 1% |
| 描述符合并（prefill / decode 分配连续块） | 执行 | **待尝试（改代码）** | 每段约 3.3 万个 44 KB 的描述符 | 同上，与上一项不叠加 |
| prefill 按分块边算边推 KV | 执行 | **待尝试（改代码，大）** | 把传输藏到 prefill 计算后面 | 约 1% |
| 同节点 HIP IPC（XGMI）传输 | 执行 | 已试失败两次，待排查 | docker-authz 拒绝 `--pid=host`；`--pid container:` 可以，但 prefill 预热时报 HIP 错误 209 | 约 1% |
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

## 7. 上游（vLLM upstream）可移植项（待子任务结果补充）
