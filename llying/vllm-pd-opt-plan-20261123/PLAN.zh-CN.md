# GLM-5.2 AgentX P4D4（Infera + vLLM）：48k 现状的瓶颈分析与继续优化的计划（2026-11-23，待评审）

本文只做分析和计划，没有跑新的 GPU 实验；用到的数据都来自本分支已有的结果（`../vllm-p4d4-crusoe-20261117/`、`../vllm-decode-trackA-20261118/`、
`../upstream-vllm-20261117/`）和 sglang recipe 分支 `dev/pd_opt/glm_5.2_agentx`（`llying/pd-repro-crusoe-20261116/`、`llying/dp-graph-variant-fix-aus-20261103/`）。
vLLM 源码行号指 v0.34.0（`/home/liyingli/bench_agentx/baseline/src-snapshots/vllm-v0.34.0`）。若无说明，成绩都是单次运行。

## 0. 结论

- **差距全在 decode，而且主要不在 kernel，而在 PD decode 的调度层。** 同样"最忙 rank 约 17 个请求"的负载下，生产里的有效步长（ITL × 接受长度）
  vLLM 约 49.8 ms、sglang 约 42.8 ms（+16%）；而 vLLM 的纯 decode（A 线聚合测法，每 rank 16 个请求）只有 41.3 ms，按上下文长度校正后约 43–44 ms，
  与 sglang 只差 1–2 ms。**生产里多出来的约 6 ms / 步（约 12%）是 PD 场景特有的开销**，按读代码得出的机制，候选有三个，都还没被量化
  （11-23 更正：第 1 条大概率不成立，剩下第 2、3 条；若它们解释不了 6 ms，要另找原因，阶段 0a 的逐步日志就是为此）：
  1. **新请求进 decode 的第一步是不均匀批次**（**11-23 更正：大概率不成立**）：vLLM decode 对每个新请求重算最后 1 个 prompt token（`scheduler.py:3664`），
     但 v0.34 的调度器会给这样的请求补 3 个占位草稿（`scheduler.py:1358-1380`，"Pad new decode requests to uniform spec decoding size to preserve full
     cudagraph"），worker 把它当 decode 跑（`gpu/model_runner.py:1426-1438`），批次仍是均匀的 1 + 3。只有同一步里另有 prefill 块（主要是抢占后的重算）时
     才不补齐，这时 DP rank 间取最小 graph 模式（`v1/worker/dp_utils.py:115-187`），4 个 rank 一起降级——这一部分并入第 2 条。阶段 0a 的逐步日志会直接确认。
  2. **抢占后整段重算**：decode 只在第一次拉 KV（`mooncake_connector.py:1244` 之后 `do_remote_prefill=False`），又关了前缀缓存，被抢占的请求要在 decode
     本地把 10–75 万 token 重新 prefill；MoE 跨 4 个 rank 做 AG/RS，这些 8K 大块会拖住全部 rank。m10 / J3b 每小时 22 / 19 次抢占。现有指标看不到这部分
     （m10 decode 的 `prompt_tokens_by_source{local_compute}` 为 0，只在首 token 记一次）。
  3. **host 侧开销**：decode 4 个 EngineCore 主线程约 99% CPU；`ROCM_AITER_MLA_SPARSE` 没有实现 `update_draft_decode_metadata`，草稿每步在 Python 里重建注意力
     元数据（`spec_decode/speculator.py:450-465`；indexer、CUDA 的 sparse MLA、DeepSeek V4 AMD 后端都实现了）；异步调度 + DP + MoE 时每步的 DP 同步走 CPU（gloo）。
- **kernel 层落后约 2–4 ms**：dense GEMM（约 1–3 ms，分类口径待核对）、MoE 分发多 gather 两个小张量（约 0.7 ms）、MoE GEMM（约 0.9 ms）、采样 kernel（0.57 ms）；
  sparse MLA / indexer 反而比 sglang 快（64K 时快约 3 ms，长上下文下优势缩小），两相抵消后就是上面纯 decode 的 1–2 ms。
- **KV 容量与抢占**：decode 每 rank 2,955,712 token（0.95），比 sglang 少约 3%；KV 用量 p50 0.84–0.87、p90 0.98。decode 慢 8% 时吞吐掉 10%（m1 对 s13），
  所以 decode 的改进对吞吐的弹性约 1–1.25。**追平 sglang 需要有效步长降约 7–9%（3.5–4.5 ms）**；把第 1 条的三项和 kernel 项做完，有机会超过。
- **prefill、路由、KV 传输都不是瓶颈**：prefill 现算 3.92% ISL（sglang 3.9%），排队非空 45–52%（sglang 61–66%），TTFT 均值 5.20 s 已好于 sglang 的 5.65 s。
  DP rank 不均（最忙 / 平均 1.28）两边相同，不是 vLLM 特有的问题。
- **计划**（第 5 节）：阶段 0 先用一次带插桩的 P4D4 运行 + 一个 decode-only 测具（vLLM 自带的 `DecodeBenchConnector`）把这 6 ms 拆开；阶段 1 做 PD 调度层
  （首步均匀化、抢占、host 开销）；阶段 2 做 kernel；阶段 3 做 decode KV 容量；阶段 4 是 P8D8 和上游 / 合规。预计阶段 1 收益最大（+4–8%，推算）。

## 1. 现状与对标口径

| | sglang P4D4 C104（crusoe，3600 s） | vLLM m10（v0.34.0 + 覆盖层） | vLLM J3b（main + PR 代码） |
|---|---|---|---|
| tok/s/GPU | 53,351 | 48,572（91.0%） | 48,417（90.8%） |
| req/s | 3.469 | 3.297 | 3.285 |
| 交互 p50 / p90 | 69.9 / 60.8 | 59.4 / 48.5 | 60.0 / 54.8 |
| ITL p50 / p90（ms） | 14.3 / 16.4 | 16.9 / 20.6 | 16.7 / 18.2 |
| TTFT p50 / p90 / 均值（s） | 4.20 / 9.23 / 5.65 | 2.88 / 10.97 / 5.20 | 2.48 / 9.72 / 6.37 |
| prefill 现算 / 占 ISL | 16,631 tok/s / 3.9% | 15,170 / 3.92% | 15,224 / — |
| decode 最忙 rank 在跑 p50 / p90 | 16 / 24（AUS scoreC） | — | 17 / 23 |
| decode KV / rank | 3,093,376（PTPC 开） | 2,955,712（0.95） | 同左 |
| decode 抢占 / h | — | 22 | 19 |

- **同口径目标**：sglang 的 53,351 含 PTPC FP8 草稿 MoE（AUS 上关掉 −1.6%：52,830 对 53,666），而 InferenceX Check 13 禁止对草稿做任何量化，vLLM 侧也不做
  （11-17 评审决定）。不含 PTPC 的同口径目标约 **52.5k**；53,351 仍作为最终目标。sglang 的数里还含 KF kernel overlay 和两份 aiter GEMM 调优表，都没有上游。
- **测量噪声**：同配置两台节点之间差 7–10%（m1 对 s13、f1 对 f2），J3 与 J3b 差 12%（节点不同）；每轮测量期的头 5 min 固定约 22K（测量方式造成）。
  单次运行里小于约 10% 的差别不当结论（第 6 节的方法就是为此设计的）。

## 2. 瓶颈分析

### 2.1 decode 有效步长：生产比纯 decode 多约 6 ms

| | 步长 | 来源 |
|---|---|---|
| sglang 生产（最忙 rank 16–17） | 41–42.8 ms | 日志步长 @16 41 ms（crusoe）、@17 42 ms（AUS E11）；ITL 14.3 × 2.99 = 42.8 |
| **vLLM 生产（最忙 rank p50 17）** | **约 49.8 ms** | J3b ITL 16.7 × 接受长度 2.98 |
| vLLM 纯 decode，每 rank 16 / 26 个请求（64K 上下文，均匀） | 41.3 / 49.8 ms | A 线 r8（推荐配置） |
| vLLM 纯 decode，按每 rank 17 个、AgentX 上下文校正 | 约 43–44 ms | 插值 42.1 ms + 上下文校正约 1.4 ms（推算，下注） |

- 上下文校正用的是 sglang 的步长拟合（每 1M 最忙 rank 上下文 +1.66 ms，AUS RESULT 4.5）：每 rank 16 × 64K ≈ 1.05M，AgentX 约 16 × 12 万 ≈ 1.9M。
  vLLM 的上下文敏感度未单独测过（它的 indexer 打分比 sglang 快，但 aiter radix top-k 随长度增长），所以这一行是推算。
- 结论：**vLLM 纯 decode 的 kernel 与 sglang 只差 1–2 ms，生产里额外多出约 6 ms**。sglang 生产步长与它的纯 decode 基本一致（decode 每步都是均匀的 verify，
  GPU 空闲每步 0.2–0.8 ms：REANALYSIS 1.2、RESULT 4.1）。

### 2.2 这 6 ms 可能在哪（按代码机制推断，均待测）

| # | 机制 | 证据 | 粗估 |
|---|---|---|---|
| H1 | **新请求首步不均匀 → 4 个 rank 一起退出 FULL graph**（**11-23 更正：大概率不成立**）。decode 收完 KV 后 `num_computed_tokens = num_tokens - 1`（`scheduler.py:3664-3665`），但调度器会给只剩 1 个 token 的新请求补 `num_spec_tokens` 个占位草稿（`scheduler.py:1358-1380`），worker 按 decode 跑（`gpu/model_runner.py:1426-1438`），批次均匀。只有同一步有 prefill 块（`prefill_scheduled`）时不补齐，见 H2 | 代码；待 0a 逐步日志确认 | 约 0（不含 H2 的部分） |
| H2 | **抢占后在 decode 本地整段重算**。Mooncake 只拉一次（`mooncake_connector.py:1244`），decode 关前缀缓存；m12 卡死前 DP0 在推理中 JIT 编译 `_localize_aiter_prefill_topk_kernel`（decode 走了 prefill 路径）。重算块所在的步不均匀，4 个 rank 一起退出 FULL graph，且这一步新请求也不补齐 | m10 22 次 / h、J3b 19 次 / h；m10 抢占所在的 10 s 窗口里 decode 生成速度 20 次中约 13 次从 3,400–4,200 掉到 1,600–2,200 tok/s（本文用 `samples.jsonl` 算，粒度 10 s，混有负载波动，只算方向性证据） | 每次重算 10 万+ token、拖住 4 个 rank 数秒到十几秒；总量未知，1–3 ms 等效 |
| H3 | **host 侧开销**：草稿元数据每步 Python 重建（ROCm sparse MLA 不支持原地更新，`speculator.py:450-465`）；DP 同步走 gloo（`config/vllm.py:1639-1650`）；EngineCore 主线程约 99% CPU | 纯 decode 下被异步调度掩住（A 线 trace 里 GPU 忙碌 50.1 ms / 步 ≈ 非 trace 步长 49.77 ms），PD 负载下没测过 | 0–2 ms |
| H4 | DP 不均造成的补齐 | 最忙 / 平均 1.28（两边相同）；sglang PP8 上补齐行占 42% | 两边共有，不计入差距 |

H1（11-23 更正后只剩"重算块所在的步"这一部分）/ H2 有一个共同点：它们都只在 PD decode 上出现，A 线的聚合 bench（一次性发 104 个请求、不抢占）测不到，这也解释了为什么 A 线的 −6.3% 在 P4D4 上
没有完全兑现成吞吐。

### 2.3 kernel 同口径（每步，ms）

A 线 trace（r7 / r8，每 rank 26 个请求、64K）对 sglang TP4 C104 生产 trace（AUS RESULT 4.1 / 4.6）。负载形状不同，下表只用来找方向：

| 类别 | vLLM | sglang | 说明 |
|---|---|---|---|
| MoE GEMM（主干 MXFP4） | 约 17.5（moe1 150 µs + gemm2 83 µs，×75） | 16.6–17.7（KF gemm1 + atomic gemm2） | decode 下 MoE 读全部专家，与 M 基本无关；差约 0.9 |
| MTP 层 BF16 MoE（3 次草稿） | 约 2.2 | 在 draft graph 4.3 里 | 两边都 BF16（PTPC 不可用），接近带宽上限 |
| dense GEMM | 12.45（含 lm_head × 4 共 1.25、router gate 0.71） | 7.89（verify） | 去掉 lm_head / gate 后，按 kernel 名粗分的投影类（o_proj 51 µs × 79 = 4.04 及 4 个小 GEMM）约 7–9，sglang 注意力投影 5.6–6.2（absorb 另计）；差约 1–3，需同口径核对 |
| MoE 通信 | 6.10（hidden AG 34.5 µs + topk ids / weights 两次 AG 各 4.8 µs + RS 34 µs） | 5.15–5.37 | 多出的两次小 AG 约 0.7 |
| sparse MLA | 4.04 | 6.82 | vLLM 默认 asm 持久化 kernel 快 |
| indexer（打分 + top-k） | 1.47 + 约 0.7 | 3.36 | 64K 对 AgentX 上下文，不可直接比；长上下文下 vLLM 优势缩小 |
| 采样 `_topk_topp_kernel` | 0.57 | — | 每步一次 |

落后的几项合计约 2–4 ms，减掉 attention 侧的优势后净差 1–2 ms（与 2.1 节一致）。**这一层值得做，但单靠它追不平。**

### 2.4 KV 容量与抢占

- decode 每 rank：权重 130.8 GiB（其中每 rank 一份的 BF16 注意力 24 GiB、MTP 专家 4.5 GiB），激活峰值 6.7 GiB（按 `--max-num-batched-tokens 8192` 预留），
  graph 1.1 GiB，KV 133.3 GiB = 2,955,712 token。
- KV 用量 p50 0.84–0.87、p90 0.98；m10 有 60% 的时间某个 rank 超过 95%。vLLM 贪心准入、KV 满了再抢占（LIFO），被抢占的请求整段重算（H2）；
  sglang decode 按预分配准入（等预分配 2.4–2.9 s），把这部分代价变成排队。
- 0.95 下有三次 decode 崩溃 / 卡死（m9b、m12、m13，U29），原因未定（显存余量或第一版精确锁页）；之后用 `hipHostAlloc` 锁页的 J3 / J3b 在 0.95 下都正常。
  在提高显存目标之前，要先把 decode 的非 torch 显存量清楚。

### 2.5 吞吐对 decode 的弹性

- m1 对 s13（同配置、两台节点）：decode 生成慢 8% → KV 更满、decode 排队 1.30 → 3.38 s → 吞吐 −10%。KV 满的时候，decode 变快同时释放 KV、减少排队，
  弹性 ≥ 1。
- AgentX 的 lane 周期 = 工具 / 思考间隔（约 50%，服务端改不了）+ TTFT（约 16%）+ decode（约 34%）。vLLM 的 TTFT 已比 sglang 短，余下的差距由 decode 承担。

## 3. 对照：sglang recipe 有、vLLM 现在没有的

| sglang recipe 的项 | 作用（sglang 上实测） | vLLM 现状 | 本计划 |
|---|---|---|---|
| PD 下 prefill 传草稿信息，decode 首步即 verify（每步均匀） | decode 永远在 graph 里 | 不传草稿，但调度器给新请求补占位草稿，首步也是均匀的（11-23 更正） | 阶段 1A 视 0a 结果取消 |
| decode 预分配准入（`early` 预留 + 估计输出 `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`） | 不靠抢占腾 KV | 贪心准入 + 抢占重算（H2） | 阶段 1B |
| graph 内 aiter reduce-scatter（E11，`SGLANG_USE_ROCM700A=0`） | 每步约 −3 ms | U13 修好后 FULL graph 内生效（fork PR #1，未上游） | 跟进上游 |
| KF overlay：indexer 打分（#47897）/ top-k（#47872）/ sparse MLA（#47278）/ MoE | 每步约 3 ms（E11 时期） | #47897 = U12（上游 #68596 在审）；#47278 / KF sparse MLA 比 vLLM 默认慢，不移植；#47872 只在长上下文快；KF MoE 无 PR | 阶段 2D / 2E |
| 两份 GLM-5.2 aiter GEMM 调优表 | — | vLLM decode / prefill 日志有 "not found tuned config" | 阶段 2A |
| HiCache 写回 250 GB × 4 + SPF | prefill 命中 96%，SPF 比 FCFS +15% | 写穿 CPU 层 1.9 TB（精确锁页）+ 缓存感知准入，命中已持平；P4D4 prefill 不是瓶颈 | 只在 P8D8 再看（阶段 4） |
| router DP 均衡权重 40 万 | 吞吐 −0.7%、交互 p90 +6.6%（帕累托右移） | `INFERA_VLLM_DP_ENGINE_LB=1`（U28），吞吐结论不明 | 阶段 1B 改成 KV 感知 |
| PTPC FP8 草稿 MoE（+#45800） | +1.6% | 不做（InferenceX Check 13 禁止） | 不做 |
| dense 层在线 FP8 | sglang 上吞吐持平、KV +7.7% | 没有 Quark 排除层的在线 FP8 开关 | 阶段 3B（vLLM 受 KV 限，收益可能不同） |
| mem 0.95、stream-interval 30、MTP K3 | — | 已对齐 | — |

## 4. vLLM 自身可用的 feature / 优化

| feature | 状态 | 用途 |
|---|---|---|
| `--cudagraph-metrics`（`config/observability.py:84`） | 未开 | 阶段 0：直接统计 decode 各 graph 模式的步数，量 H1 |
| `DecodeBenchConnector`（`kv_connector/v1/decode_bench_connector.py`） | 未用 | 阶段 0：不起 prefill，用假 KV 模拟 PD decode（同样重算最后 1 个 token），在 4 卡上快速迭代 decode 侧改动 |
| 异步调度 | 默认开（`config/vllm.py:1610-1637`） | 确认 PD + MTP 下确实生效；DP 同步因此走 gloo |
| `update_draft_decode_metadata` | indexer 已实现，`ROCM_AITER_MLA_SPARSE` 未实现 | 阶段 1C：草稿步整体进 graph |
| decode 前缀缓存 + Mooncake 只拉缺的块（`mooncake_connector.py:1188-1196`） | decode 关着 | 阶段 1B：被抢占的请求可命中自己尚未被复用的块；同会话续轮少传 |
| vLLM DP 引擎均衡（按 waiting × 4 + running） | U28 开关已有 | 阶段 1B：加 KV 空闲量 |
| `--max-num-batched-tokens`（decode 激活预留） | 8192，预留 6.7 GiB | 阶段 3A：抢占重算消除后可降，换 KV |
| aiter AG/RS（U13）、gluon logits（U12）、精确锁页（U26 / #60909） | 覆盖层 / fork PR | 跟进上游，迁移前提 |
| EP / MoRI、DBO | A 线：单节点都不比默认快；DBO 需要 all2all 后端 | P4D4 不用，P8D8 再看 |

## 5. 计划

每项写明：做什么、依据、怎么验证、落在哪里（部署 / Infera / vLLM 上游 / aiter）。收益都是推算，不能简单相加。

### 阶段 0：把 decode 的 6 ms 拆开（约 2 个单节点作业，先做）

- **0a. 插桩的 P4D4 运行**（J3b 配置，3600 s，跑完即退）：decode 加 `--cudagraph-metrics`；一个只写日志的覆盖层，每步记录 `(t, rank, num_reqs, num_tokens,
  max_query_len, cg_mode, 步长)` 和每次抢占的 `(req, num_computed_tokens, 重算耗时)`；稳态（最忙 rank 15–20）时在 decode 上抓两次 3 s torch trace；
  py-spy dump EngineCore 与 worker（按进程名匹配，m12 时 `pgrep -f python` 漏掉了改过名的 EngineCore）；每 10 s 记 `mem_get_info` 与 torch reserved。
  产出：有新请求的步占比与这些步的步长（H1）；抢占重算的 token 数、持续时间和这期间其他 rank 的步长（H2）；生产负载下每步 GPU 空闲（H3）；
  AgentX 形状下的分类 kernel 时间，与 sglang 4.6 节的表逐项对照；decode 非 torch 显存峰值。
- **0b. decode-only 测具**：`DecodeBenchConnector` + DP4，GPU 0–3 和 4–7 各起一套同时跑（A 线 `round.sh` 的做法），请求的 ISL / OSL 取自 AgentX 数据集的分布，
  按 3.3 req/s 持续到达，目标每 rank 15–20 个在跑。两种到达方式对比：持续到达 vs 一次性到达后的纯 decode 窗口，差值就是 H1 的代价。以后阶段 1 / 2 的
  decode 改动都先在这里 A/B（约 30 min / 轮，不受 prefill 和节点间波动影响），再上 P4D4。
- **判据**：0a / 0b 的结果决定阶段 1 三项的顺序；若 H1 + H2 + H3 合计不到 3 ms，转为先做阶段 2 / 3。

### 阶段 1：PD decode 调度层（预计收益最大，+4–8%）

- **1A. 新请求首步均匀化（H1）**（**11-23 更正**：v0.34 调度器已经给新请求补占位草稿，见 2.2 节 H1；0a 若确认新请求所在的步都是 FULL，本项取消；
  若仍有不均匀的步，再按下面的做法处理。另：按每步约 0.165 个新请求算，"DP 对齐准入"要攒约 1 s 才能把降级步压到 5% 左右，TTFT 平均多约 0.5 s，代价偏大），三种做法，按改动从小到大：
  1. DP 对齐的准入：新到（KV 已就绪）的请求每 N 步统一进入一次，且 4 个 rank 同一步进入，把降级步从约 15% 压到 1/N 以下；代价是 TTFT 最多多 N × 50 ms。
     改 vLLM 调度器（可放在 DP 协调里），先做原型测收益。
  2. 首步补齐：新请求首步带 3 个占位草稿 token 并强制全部拒绝（含模拟接受长度的路径），批次保持 1 + 3 均匀；decode 受带宽限制，多算的 3 行几乎不花时间。
     改调度器和 rejection sampler，要过正确性门禁（真实接受长度）。
  3. 与 sglang 同形态：prefill 算出首 token 和 MTP 草稿，随 `kv_transfer_params` 传给 decode，decode 首步直接 verify，也省掉重算最后 1 个 token。
     改动跨 connector / 调度器 / Infera 路由，走 vLLM RFC。
  验证：0b 测具里降级步占比与平均步长；P4D4 稳态吞吐与交互 p90。落点：vLLM 上游（做法 1 / 2 可以先做覆盖层验证）。
- **1B. 抢占（H2）与 decode 准入**：
  1. decode 准入余量：远端 KV 请求准入时，除了 prompt 的块，还要给在跑请求的增长留出余量（参照 sglang 的预分配和输出估计），把抢占换成排队。
     **（11-23 补充）v0.34 已有 `--watermark <比例>`**：准入 WAITING / PREEMPTED 请求时保留这一比例的 KV 块（`v1/core/kv_cache_manager.py:207-210, 539-569`），
     只改部署参数即可；需要扫比例（与 KV 利用率此消彼长），0b 第二轮先试 0.03。
  2. KV 感知的 DP 选 rank：vLLM DP 引擎均衡现在只看 waiting / running，加上各 rank 的空闲 KV 块（或在 Infera 路由里按 decode 的 KV 用量选 rank）。
     U28 的数据显示只按请求数均衡时在跑数更均匀、但 KV 占满的比例没变。
  3. 让抢占变便宜：decode 开前缀缓存（被抢占请求释放的块在被复用前仍可命中；同会话续轮落在同一 rank 时还能少传 KV）；更彻底的做法是 decode 抢占时
     把请求交回路由重新走 prefill（prefill 有 1.9 TB CPU 层、现在只忙一半，只需算已生成的部分）。
  验证：每小时抢占数、重算 token 数、decode 排队、KV 占满比例；P4D4 吞吐。落点：vLLM 上游（1、3）、Infera（2 的路由版、3 的回退）。
- **1C. host 开销（H3，依 0a 的 GPU 空闲决定优先级）**：给 `ROCM_AITER_MLA_SPARSE` 实现 `update_draft_decode_metadata`（只发 graph 安全的操作、写进已有的持久张量，
  参照 indexer 和 DeepSeek V4 AMD 后端的实现），让 3 次草稿整体进 graph；按 py-spy 结果处理其余热点。落点：vLLM 上游（ROCm only）。

### 阶段 2：decode kernel（落后项约 2–4 ms，+2–4%）

- **2A. dense GEMM**：从 0a 的 trace 导出 decode 的 BF16 GEMM 形状（o_proj 6144 × 16384、q_b 16384 × 2048、fused_qkv_a、indexer wq_b / wk、lm_head；
  M = 4–100 行），与 sglang 同形状的 kernel 选择比对；用 aiter / hipBLASLt 调优工具生成调优表，经 aiter 的配置环境变量加载（sglang recipe 带的两份表先直接试）。
  落点：aiter 上游（调优表）或部署配置。预计 1–3 ms（先按 0a 的 trace 做同口径分类再定）。
- **2B. MoE 分发只 gather hidden states**：现在每层多两次 all-gather（topk ids / weights，各 4.8 µs × 75 层 ≈ 0.7 ms）；若 router 放到 dispatch 之后、在 gather 后的
  全部 token 上算（gate 是复制的，6144 × 256 很小），这两次可以省掉。先读 `naive_dp_ep` / AG/RS manager 确认可行，再在 0b 上 A/B。落点：vLLM 上游。
- **2C. 采样**：`_topk_topp_kernel` 每步 0.57 ms。查 AgentX 请求实际带的采样参数，看能否走 aiter / 更快的采样 kernel。
- **2D. 长上下文 indexer**：#47872 单遍 top-k 已移植（A 线 `scripts/kf/vllm_triton_topk_decode.py`），40K–200K 时快 1.33 倍，64K 时慢；在 0b 的 AgentX 形状上
  端到端 A/B，再决定是否接入。
- **2E. 跟进与低优先**：U12 ↔ #68596、U13（fork PR #1）上游化，是迁移的前提；KF MoE gemm1 与 aiter 在 M = 256–512 上对比（差约 0.9 ms，KF 无上游 PR，低优先）。

### 阶段 3：decode KV 容量（+1–3%）

- **3A. 显存账与激活预留**：用 0a 的非 torch 显存峰值给出可证明安全的显存目标；阶段 1B 之后 decode 不再整段重算，每步最多约 64 × 4 = 256 个 token，
  `--max-num-batched-tokens` 可降到 2048（约 +5 GiB / rank，KV +4%）。m9b 用 1024 崩过，当时的归因已撤回，要在 0b 上单独压测。
- **3B. 目标模型 dense 层在线 FP8**（ATOM 式 `ptpc_fp8`，排除 MTP 层）：每 rank 权重约 −12 GiB → KV 约 +9%。sglang 上吞吐持平（小 M 的 FP8 GEMM 不快、
  激活量化另加 1 ms），但 vLLM 受 KV 限更明显，收益可能不同；先算 KV 增量对抢占 / 排队的影响，再决定是否实现。需要 Quark 排除层走在线 FP8 的补丁和 eval。
  落点：vLLM 上游；InferenceX 允许（目标模型、过 eval）。

### 阶段 4：P8D8 与上游 / 合规

- P8D8（16 卡，sglang 54,468）：跨节点建连已打通（`MC_RDMA_NOTIFY_ENABLED=0`），上次测量中被外部 SIGKILL，用同一配置重提拿第一个成绩。sglang 的 P8D8
  受 prefill 限（单个 PP8 92% 时间排队），vLLM 上到那时再看 prefill 侧：按未命中 token 排序的准入（U3）、写回式 CPU 层（上游 #65681）。
- InferenceX 要求引擎原样运行：上面所有 vLLM 改动都要进上游 main，部署侧只留参数和 Infera 路由。迁移的必需项清单：U12（#68596）、U13、U26（#60909）、
  U11 / U19（如果用到 FS 层或共享区）、阶段 1 / 2 落地的项；Infera 侧 #246–#251。

### 不做 / 暂缓

| 项 | 原因 |
|---|---|
| EP / MoRI（单节点） | A 线：EP + 融合共享专家 53.40 ms、MoRI 修完 54.85 ms，都不如默认 49.77 ms |
| PTPC FP8 草稿 MoE | InferenceX Check 13 |
| decode TP4 + DCP4 | sglang 上比 DP4 慢 4.3%；DSA indexer 要按序列切分，vLLM ROCm 未支持 |
| P4D4 的 prefill 侧优化（SPF、chunk、写回） | prefill 只忙一半，不是瓶颈 |
| 短增量请求在 decode 上 prefill | 往瓶颈上加活 |
| KF sparse MLA（#47278）、#47276 融合 indexer | A 线：比 vLLM 现有实现慢 |

## 6. 实验方法与资源

- **两级 A/B**：decode 侧改动先在 0b 测具上比（两套 4 卡同时跑，看步长和降级步占比，噪声 < 0.1 ms），有收益才上 P4D4；P4D4 同节点、同作业里前后跑，
  比较稳态（300–3600 s）吞吐，并按最忙 rank 请求数分桶比较有效步长；关键结论重复一次。
- **每个改代码的项**过门禁：smoke、GSM8K 200、16k 长上下文、真实接受长度 ≥ 3（K3）。
- 资源规则沿用主线交接（11-19 起）：只用跑完即退的作业，不用 hold 作业占机器；每人同时约 2 个作业 / 2 台节点；分析在计算节点的作业里做完；作业开头检查
  GPU 空闲显存（v3-168 那种被占满的节点直接放弃）。
- 资源估计：阶段 0 约 2 个单节点作业（0a 约 2 h，0b 约 1.5 h）；阶段 1 / 2 每项 0b 上 1 轮 + P4D4 1–2 轮。

## 7. 风险

1. **0.95 下 decode 偶发崩溃（U29）**：阶段 1 / 3 都会改变 decode 显存和批次形状，每次改动都要看 hipBLASLt / `OUT_OF_RESOURCES` / KFD 驱逐计数。
2. **首步均匀化与正确性**：占位草稿必须在真实接受和模拟接受两条路径上都被拒绝；prefill 传草稿要保证与 decode 的 MTP 状态一致。
3. **抢占策略的副作用**：准入余量过大，KV 利用率下降、排队变长（sglang 上 DP 均衡权重过大时 TTFT +22% 是同类现象）。
4. **节点波动**：单次 P4D4 小于约 10% 的差别不可靠，要靠 0b 测具和同节点 A/B。
5. **上游节奏**：迁移依赖 U12 / U13 / U26 等进 vLLM main；阶段 1 的调度器改动在上游可能需要 RFC，周期较长。

## 8. 预期（推算，阶段之间不能简单相加）

| 阶段 | 有效步长 | 预期吞吐 |
|---|---|---|
| 现在 | 约 49.8 ms | 48.4–48.6k |
| 阶段 1（H2 + H3 收回一半到大部分；11-23 起 H1 不再计入，6 ms 里未解释的部分待 0a） | −3 到 −5 ms（若 0a 找到其余来源） | +4–8% → 50.5–52.5k |
| 阶段 2（kernel） | −1.5 到 −2.5 ms | +2–4% |
| 阶段 3（KV +4–9%） | — | +1–3% |
| 合计 | 约 42–45 ms（sglang 41–43） | 52.5–55k（同口径目标 52.5k，最终目标 53.4k） |

## 附：本文用到的数据与代码位置

- sglang：`dev/pd_opt/glm_5.2_agentx` 分支 `llying/pd-repro-crusoe-20261116/README.zh-CN.md`（53,351）、`llying/dp-graph-variant-fix-aus-20261103/RESULT.zh-CN.md`
  （4.1 / 4.5 / 4.6 trace 与步长拟合、4.3 PTPC、7.x）、`REANALYSIS-20261112.zh-CN.md`（GPU 空闲、补齐行、indexer）。
- vLLM：`../vllm-p4d4-crusoe-20261117/README.zh-CN.md`（m10、f1 / f2、1q / 1k / 1m）、`../upstream-vllm-20261117/results/main-300563/`（J3b 的
  `engine-config.env`、`samples_summary.txt`、`decode_pressure.txt`）、`../vllm-decode-trackA-20261118/README.zh-CN.md` 与 `results/a2-r7/trace-cat/`。
- 抢占前后生成速度：`/home/liyingli/bench_agentx/vllm-pd-20261117/runs/p4d4-m10-v3comm-render4-pgem088-exactpin-cpu1900-3600/samples.jsonl`（22 MB，decode 端点
  每 10 s 一条）。
- vLLM v0.34.0：`v1/core/sched/scheduler.py:3616-3666`、`v1/worker/dp_utils.py:115-263`、`v1/worker/gpu/cudagraph_utils.py:558-602`、
  `distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1160-1244`、`v1/worker/gpu/spec_decode/speculator.py:450-465`、
  `config/vllm.py:1589-1650`、`config/observability.py:84`、`distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`。
