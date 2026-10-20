# GLM-5.2-MXFP4 aggregated（不做 PD 分离）AgentX：单机 MI355X，TP4 + DP4 attention

目标：在只有 crsuse2-m2m-138 可用期间，用 infera（sglang v0.5.21 + infera rust router）复现 InferenceX
上 Dynamo + SGLang B300 aggregated 配置（4 卡一个实例，C48 约 42k tok/s/GPU，见
`../inferencex-glm52-b300-dynamo-agg-20261015/`），并在这个场景下尽量推高 tok/s/GPU。

## 配置

一个 mixed worker，138 GPU 0–3（`config/config.agg.sh`；每轮的差异在 `config/round.<ROUND>.sh`）：

| 项 | 值 | 与 B300 的差异 |
|---|---|---|
| 并行 | TP4，`--dp-size 4 --enable-dp-attention`，`--enable-dp-lm-head`，EP1 | 同 |
| MTP | EAGLE 5/1/6，模拟接受长度 3.61（match-expected，real-draft-token） | 同 |
| 调度 | chunk 32768，max-prefill-tokens 8192，max-running 128，decode graph 最大 bs 64，`--enable-prefill-delayer --prefill-delayer-max-delay-passes 60` | 同 |
| 显存 | mem-fraction-static 0.85，KV fp8_e4m3 | 同 |
| HiCache | 每 rank 270 GB 主机池；io-backend kernel + page_first | B300 为 write_back + direct + page_first_direct；ROCm 上用 3P1D prefill 已跑通的 kernel + page_first |
| tokenizer | 1 个 worker，stream-interval 1 | 同（默认值） |
| DSA | triton prefill / decode 后端 | ROCm 专有 |
| router | infera kv-aware，`--kv-overlap-weight 1.0`，`INFERA_SESSION_AFFINITY=both`（TTL 3600 s） | B300 为 Dynamo KV router（overlap 权重默认 1.0），session affinity TTL 3600 s |

镜像、模型、AgentX 客户端（agentx-harness 1.0.9，InferenceX `918524ff`）和 harness 与 3P1D 相同
（`../3p1d-tp4-tp8-v0521-20261012/`）。每 GPU 吞吐按 InferenceX 单机口径：GPU 数 = TP × PP × PCP = 4。

测量协议与 InferenceX 的差别：InferenceX 每个并发点重新部署；这里引擎常驻，每点之前 `/flush_cache` 并重启 router
（与 3P1D 调优轮相同）。

## 复现

```bash
bash scripts/launch_agg.sh ROUND=<tag>              # etcd -> worker -> router -> 冒烟
bash scripts/agg_round.sh ROUND=<tag> VARIANT=<name> CONC=<n> [DURATION=<s>]
python3 scripts/agg_report.py results/<name>/c<n>   # 客户端 + 每 DP rank 的引擎均值
```

harness 改动（mixed 角色）：`../3p1d-tp4-tp8-v0521-20261012/scripts/bench-harness/tools/topology.py`、
`tools/agentx_env.py`（拓扑全为 mixed 时生成单机口径的 runtime.env：`IS_MULTINODE=false`、`DISAGG=false`、`TP=4`）。

## 轮次

| ROUND | 内容 | 状态 |
|---|---|---|
| a1 | HiCache write_back 270 GB | 参数校验通过后停掉（按要求先不开 HiCache 调通流程） |
| a2 | 不开 HiCache，调通流程（预热每 lane 2 个请求） | 完成：C16 / 600 s 全流程通过（不计分） |
| a3 | HiCache write_through 270 GB / rank | HiCache 功能检查通过；C48 = 31,605；C64 = 30,027 |
| a4 | write_back + mem 0.92 | HiCache 功能检查通过；C64 = 31,221 |
| t8a / t8b | TP8 + DP8 attention（8 卡），write_back + mem 0.92，HiCache 200 GB / rank | OOM（chunk 8192/rank）/ AgentX 中止（单 tokenizer） |
| t8c | t8b + tokenizer 4 个 worker | C128 = 23,366 |
| t8d | + DP spec/prefill coordination，关 prefill delayer | 未跑（coordination 收益受逐层同步限制） |
| a5 | a4 + delayer 上限 20 步 + tokenizer 4 个 worker | C64 = 29,810（a4 为 31,221） |
| a6 | a4 + delayer 上限 120 步 + tokenizer 4 个 worker | 预热中 DP2 报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 后挂起（mem 0.92） |
| a6b | a6 + mem 0.90 | C64 = 32,228（TP4 最好） |
| a7 / a7b | a6b + DP spec/prefill coordination + 每 rank 2048 块（mem 0.90 / 0.85） | 负载下 decode 检查 32/32 通过；C64 预热 2–3 分钟 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 崩溃 |
| a7c | a7b + `HSA_NO_SCRATCH_RECLAIM=0` | C64 = 27,607 |
| a7d | a7c，块恢复 8192 | C64 = 29,307 |

全部实验（含失败和未跑的轮次）的配置与数据见 [`results/summary/agg_rounds.md`](results/summary/agg_rounds.md)。

### a2 调通结果（不计分）

`results/a2-debug2/c16/summary.txt`：12,580 tok/s/GPU，ITL p50 11.1 ms，TTFT p50 2.9 s，GPU 前缀命中
96.1%（理论 96.0%），无 retract，接受长度 ≈ 3.6。启动（无 HiCache）约 7 分钟；KV 池每 rank 2,377,856 token
（105.6 GB，fp8 MLA ≈ 44 KB/token），4 rank 共约 950 万 token。

工作负载与 B300 相同（`semianalysisai/cc-traces-weka-062126`，393 条；这里实际输出均值 874 token、输入 126k，
B300 各点约 1,000 / 140k）。有效并发 5.0（C16 的 31%），每 lane 约 40 s 一个请求；B300 C16 为约 25 s
（2,327 个 / 3600 s / 16）。600 s 窗口、每 lane 2 个预热请求太短，不能和 3600 s 的点比，只用于验证流程。

### B300 aggregated 对照（`../inferencex-glm52-b300-dynamo-agg-20261015/evidence/api-glm52-b300-rows.json`）

| 配置 | C | tok/s/GPU | 每 lane 周期 (s) | ITL p50 (ms) | TTFT p50 (s) | GPU / 主机命中 |
|---|---|---|---|---|---|---|
| TP4 无 DPA | 8 | 13,666 | 22.0 | 4.4 | 0.22 | 94.0% / 2.6% |
| TP4 无 DPA | 16 | 22,862 | 24.8 | 6.6 | 0.29 | 66.8% / 29.0% |
| TP4 + DPA4 | 24 | 29,726 | 28.9 | 10.6 | 1.30 | 94.7% / 1.3% |
| TP4 + DPA4 | 32 | 37,390 | 32.6 | 11.5 | 1.41 | 87.0% / 8.5% |
| TP4 + DPA4 | 48 | 42,047 | 39.7 | 15.5 | 3.60 | 52.3% / 42.5% |

每 lane 周期随并发变长（trace 自身的间隔之外还有排队和单请求延迟），吞吐增长随之放缓：C32 → C48 只多 12%。
C48 时主机命中占到 42.5%，说明那里 GPU 池已经装不下活跃会话的前缀，HiCache 是关键。

### a3 HiCache 功能检查（write_through）

`results/a3-wt/hicache_check.json`（`scripts/hicache_check_agg.sh`，即 3P1D 的 `hicache_check.py`）：**pass**。
16 个检索探针（各约 40k token）先新算，再用 110 个填充请求（约 1150 万 token，1.4 倍 GPU 池）把它们挤出 GPU，
再问一次：首 token 16/16 一致，top-1 logprob 最大差 0.0013；第二次从主机载回 557,056 token（探针的 87%），
主机路径确实走到了。主机池每 rank 5,933,568 token（270 GB）+ DSA indexer 61.9 GB，MTP draft 层打包在同一个池里；
开 HiCache 后 GPU 池每 rank 降到约 206 万 token（不开时 238 万）。`/flush_cache` 也清空主机池
（`hicache_host_used_tokens` 在空闲时不刷新，第一次调度后从约 300 万掉到 1.6 万）。

### a3 C48（write_through，3600 s）：31,605 tok/s/GPU，B300 的 75%

`results/a3-wt/c48/summary.txt`

| | a3 C48（MI355X） | B300 C48 |
|---|---|---|
| tok/s/GPU | 31,605 | 42,047 |
| 完成请求 / 每 lane 周期 | 3,436 / 50.3 s | 4,349 / 39.7 s |
| ITL p50 / p90 | 24.8 / 48.1 ms | 15.5 ms / — |
| TTFT p50 / p90 | 4.8 / 14.5 s | 3.6 s / — |
| 引擎侧前缀命中 | 96.2%（理论 97.1%） | — |

- 差距主要在 decode：每请求约 1,000 输出 token × 24.8 ms ≈ 25 s，占 50 s 周期的一半（B300 约 15 s）。
  新算 prefill 约 4.7k tok/s（4 rank 合计），占去约 20–25% 的 GPU 时间；EAGLE + DP attention 下任一 rank 在
  prefill 时所有 rank 都不能 decode（`scheduler.py`："prefill and decode batches will not be mixed"）。
- TTFT 主要来自 prefill delayer：每 rank 3600 s 内约 470 次推迟、累计约 2,250 s（平均 4.8 s/次）；执行的
  prefill 中约 219 次等齐、约 250 次等满 60 步超时放行。delayer 把各 rank 的 prefill 对齐到同一步，换来
  decode 少被打断，代价是 TTFT。
- 缓存不是瓶颈：命中贴近理论值；GPU KV 均值 0.59–0.69、峰值 0.99，每 rank 约 1 个请求排队等 KV；主机池
  89–98% 满；无 retract。在跑请求每 rank 5.9–8.6，不均衡（每步等最忙的 rank）。
- 开跑后前约 17 分钟（预热段）每 rank 积压约 200 万 token 新算 prefill，decode 完全停住，积压消化后才进入
  稳态；计分窗口在预热之后。

可调的杠杆（按预期收益）：GPU KV（mem-fraction 0.85 → 0.92，或 TP8/DP8 用满 8 卡、每卡权重 121 → 约 61 GB）；
write_back（主机不再重复存 GPU 上的页，有效缓存约 590 万 → 800 万 token/rank，并发更高时才有意义）；
`--prefill-delayer-max-delay-passes`（60 → 30，TTFT 与 decode 打断次数的权衡）；`--prefill-decode-interval`。

### a3 C64 与 a4 C64（write_back + mem 0.92）

| | a3 C48 | a3 C64 | a4 C64 |
|---|---|---|---|
| tok/s/GPU | 31,605 | 30,027 | 31,221 |
| 完成请求 | 3,436 | 3,672 | 3,752 |
| ITL p50 / p90 (ms) | 24.8 / 48.1 | 31.9 / 60.8 | 37.2 / 69.2 |
| TTFT p50 / p90 (s) | 4.8 / 14.5 | 9.2 / 51.3 | 8.5 / 29.6 |
| 引擎侧总命中（理论） | 96.2%（97.1%） | 92.8%（96.3%） | 95.6%（96.2%） |
| 新算 prefill (tok/s) | 4,679 | 8,090 | 5,404 |
| 每 rank 在跑 / 排队 | 7.0 / 1.3 | 9.2 / 4.3 | 11.1 / 2.8 |
| GPU KV 池 / rank | 2.06M | 2.06M | 2.43M |
| 输出 tok/s（4 卡合计） | 914 | 约 980 | 1,001 |

a3 C64 是登录节点上的驱动在 10-15 约 19:05 消失后，从 138 上客户端的本地输出找回的（`results/a3-wt/c64/RECOVERED.txt`）；
它的命中构成是用引擎累计计数器相减得到的（含预热）。a4 的 HiCache 功能检查（write_back）通过：首 token 16/16 一致，
top-1 logprob 最大差 0.0005，从主机载回 557,056 token。tokenizer 在 a4 窗口内平均 0.31 核，没有饱和。

结论：容量杠杆按预期起作用（命中追回到理论值附近，新算 prefill −33%，TTFT p90 51 → 30 s），但吞吐只比 a3 C64
高 4%，仍在 C48 水平。KV 多了，每 rank 就多跑请求（9.2 → 11.1），单步也随之变慢（ITL 31.9 → 37.2 ms），每 rank
输出仍是约 340 tok/s。瓶颈是每 rank 的 decode 单步耗时：约 134 ms/步（11 个在跑，含 prefill 打断），而 3P1D 的
TP8/DP8 decode 在 11.5 个在跑时为 66.7 ms/步（纯 decode）。decode 主要受读权重的时间限制，TP4 每卡读 121 GB 权重，
TP8 只有约 61 GB，所以下一步是 TP8/DP8（`config/round.t8a.sh`，`operations/t8a.steps`）。

### TP8/DP8（t8a → t8c，8 卡，write_back + mem 0.92）C128：23,366 tok/s/GPU，TP4 的 74%

| 轮次 | 改动 | 结果 |
|---|---|---|
| t8a | chunk 8192/rank | 预热第 16 分钟 HIP OOM：MoE 第二阶段缓冲 6.75 GiB（MoE 处理 8 个 rank 汇总的 65,536 token，TP4 是 32,768） |
| t8b | chunk 4096/rank，max-prefill-tokens 4096 | 预热的"缓存压力"段一次放出约 200 个约 18 万 token 的请求，单个 tokenizer 进程占满 1 核，worker HTTP 约 30 s 无响应，router 502，AgentX 中止 |
| t8c | + tokenizer 4 个 worker | 跑完：23,366 tok/s/GPU（总 186,928 tok/s，TP4 C48 的 1.48 倍），ITL p50/p90 47.2/107.8 ms，TTFT p50/p90 20.3/108.1 s |

每卡显存：权重 75.8 + draft 6.1 GB（DP attention 下每个 rank 都持有完整 attention 权重，只有 MoE 随 TP 切分），
GPU KV 每 rank 3,317,760 token（TP4 mem 0.92 为 2,432,128，+36%）。停栈后显存约 25 分钟才释放完（2 TB 锁页主机内存）。

t8c 的瓶颈是 prefill 调度，不是算力：新算 prefill 合计约 8.7k tok/s（每卡约 1.1k，单卡能力约 5k），但平均 70.9 个
请求停在首 token 之前；每 rank 63–92% 的时间都有 prefill 被 delayer 推迟，执行的 prefill 中 75–80% 等满 60 步
（TP4 约 50%）；8 个 rank 很少同时都有 prefill。rank 间不均衡：DP3/DP7 平均排队 16–19、命中率 10–14%，DP0 排队 1、
命中率 72%。每 rank decode 输出 194–373 tok/s，和 TP4 相近，平均每步约 170 ms（3P1D TP8 纯 decode 在相同在跑数下
45–70 ms），TP8 单步更快的优势被 prefill 停顿和 delayer 吃掉了。tokenizer（4 个 worker）均值 0.53 核。

下一步：`SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1`（v0.5.21 实验开关，默认关）。EAGLE + DP attention 下，
默认只要一个 rank 有新 prefill，其他 rank 这一步就只跑 idle 批次；打开后没有 prefill 的 rank 在同一步继续 draft/verify
（这些异构步全部 eager，prefill rank 要等 decode rank 的 5 步 eager draft），delayer 也就不必再等对齐（关掉）。
`scripts/decode_check.py`：在后台持续 prefill 负载下发 32 个检索问题（生成最多 16 token，答案须含口令），
t8c（开关关）基线 32/32 通过（`results/t8c-wb-m092-ck4k-tok4/decode_check.json`），打开后先过这个检查再计分。
候选：`operations/t8d.steps`（TP8 C128）、`operations/a5.steps`（TP4 C64）。

### a5（TP4，a4 + prefill delayer 上限 20 步 + tokenizer 4 个 worker）C64：29,810 tok/s/GPU，比 a4 低 4.5%

| | a4 C64（delayer 60） | a5 C64（delayer 20） |
|---|---|---|
| tok/s/GPU | 31,221 | 29,810 |
| ITL p50 / p90 (ms) | 37.2 / 69.2 | 43.7 / 81.0 |
| TTFT p50 / p90 (s) | 8.5 / 29.6 | 5.2 / 30.5 |
| 首 token 前在途 | 14.3 | 11.8 |
| 每 rank 在跑 / 排队 | 10.6–11.7 / 1.9–4.5 | 10.8–13.9 / 1.4–3.4 |

delayer（每 rank，60 分钟窗口）：a5 等待 435–627 次、累计 1,020–1,752 s，执行的 prefill 中等齐 69 次、等满上限 360–552 次
（超时约 85–89%）；a3（60 步）约 219 次等齐、约 250 次超时（约 53%）。TP4 上有相当一部分对齐要等 20–60 步，上限砍到 20
步后 TTFT p50 少 3.3 s，但对齐大多丢失，prefill 打断 decode 的长步变多，ITL +17%，每请求约 1,000 个输出 token 下 ITL 的
损失大于 TTFT 的收益。TP4 上 delayer 保持 60 步（或更大）。窗口内每 5 分钟吞吐在 2.2 万–3.7 万之间波动，两轮之差有一部分
可能是波动，但 ITL 与对齐成功率的变化方向一致。

对齐时间模型（解释 a5 为什么变差）：delayer 要等到其他每个 rank 都至少来一个新请求。每 rank 请求间隔 τ（a4 C64 约 3.8 s）、
每步约 134 ms 时，等待 T 内对齐的概率约 (1 − e^(−T/τ))^(其他 rank 数)：TP4 上 20 步约 17%（实测 14%）、60 步约 67%（实测 57%）、
120 步约 95%；TP4 平均约 7 s（约 50 步）才能等齐，TP8（7 个其他 rank）约 11.7 s（约 70 步）。之前按 t8c 的分布（20 步内对齐
约 8%）外推到 TP4 是错的。对齐比例 57% → 14% 使每小时单独打断 decode 的 prefill 长步约 950 → 1,750 个，ITL +6.5 ms（每请求约
+6.5 s），TTFT 均值只少 1.7 s（13.9 → 12.2 s）。TP8 应加大而不是减小等待上限，同时解决 max-prefill-tokens 4096 的准入限速。
采样器从 a6 之后的轮次起记录 `prefill_delayer_wait_forward_passes` 分布。

为什么 coordination 不作为主手段：DP attention 下 MoE 权重按 TP 切在所有 rank 上，每层都有跨 rank 集合通信，各 rank 逐层同步；
异构步里 decode rank 要陪 prefill rank 走完约 0.6–1.2 s 的一步，只多出一个 decode 步（约挽回被打断时间的 5–10%），且该步全部 eager。

PD 与 mixed 的 prefill 路由均衡（各自 60 分钟窗口，按 rank 统计服务端 prefill 批次）：3P1D C192 的 12 个 prefill rank 新算量
最多/最少 1.98 倍、排队 0.5–3.2（prefill 卡平均利用率约 36%，且不影响 decode）；mixed TP4 a4 C64 为 1.26 倍、排队 2.1–3.4；
mixed TP8 t8c C128 为 3.26 倍、排队 1.4–22.7（准入受 delayer 等待和 max-prefill-tokens 4096 限速，负载稍高的 rank 队列持续堆积）。

### a6b（TP4，write_back + mem 0.90，delayer 120 步）C64：32,228 tok/s/GPU

ITL p50/p90 32.8/57.8 ms，TTFT p50/p90 14.5/45.6 s；与 a4 每秒完成请求数相同（1.045 对 1.042），+3% 主要来自窗口内平均输入更长
（123.4k 对 119.8k），属波动范围。C64 下请求速率与总生成速度（约 1,000 tok/s）在 a4/a5/a6b 间几乎不变，"每输出 token 耗时 / 在跑数"
约为常数：delayer 只在"排队"与"在跑"之间挪请求。

### GPU 时间构成（a6b 栈，C64 负载下 torch profiler，10-16 23:13 / 23:23 两次抓取）

`scripts/trace_breakdown.py`（按 GPU 侧 `scheduler.run_batch` / `step[VERIFY|EXTEND|IDLE]` 注解归集 kernel）。完整性：按阶段抓取时
各 rank 起止不同，cap1 无 EXTEND 文件、cap2 只有两个 rank 有 EXTEND，cap2 TP-2/3 DECODE 缺批次注解；只用干净窗口和计数器下结论。

- 两次抓取之间 15 分钟（计数器）：每 rank decode 步 6,754（4 个 rank 相同，逐步同步），按 72.7 ms/步约占 54% 时间；其余约 46% 为 eager 步
  （某 rank prefill、其他 rank 跑 IDLE 批次）；各 rank 自己的 prefill 步 317 / 250 / 143 / 92，多数 prefill 步只有一个 rank 在算。
- decode 校验步（cap2 TP-0/1，各 29 步，72.7 ms/步，每 rank bs 8–13）：注意力/DSA 31–35%，MoE 28%，通信 15–19%，普通 GEMM 16–17%。
- 单 rank 长 prefill（cap1，同一 8.4 s 窗口）：TP-1 连续 8 个 prefill 批次（665 ms/个，一个长 prompt 的分块续算，delayer 不管续算），
  构成为注意力/DSA 48%、普通 GEMM 25%、通信 13%、MoE 11%；其余 3 个 rank 这期间在通信 kernel 里空等（TP-0 通信占 78%），该窗口
  decode 只占约 15%。
- 10-19 更正：最初的分类把 DSA indexer 的 top-k（`sgl::dsa_topk::coop_topk_kernel`，被 `topk_` 匹配进 MoE）和 logits
  （`_gluon_fp8_mqa_logits_kernel`，被 `logits` 匹配进采样）算错了类别，当时报的是 decode MoE 35–36%、注意力/DSA 24–26%，prefill
  注意力/DSA 42%、MoE 14%。修正后 decode 中占比最大的是注意力/DSA。重算结果：`results/prof-a6b/breakdown.txt`（`breakdown.json` 供画图）。
- prefill 用到的 bf16 普通 GEMM 有 4 类 (N, K)、约 5,500 种 M 未调优（aiter 退回默认实现）；decode 尺寸的已调优。

### a7 / a7b / a7c：DP spec/prefill coordination + 2048-token chunks（TP4 C64）

- a7（mem 0.90）、a7b（mem 0.85）：负载下 decode 正确性检查 32/32 通过（189 个后台 prefill），但都在 C64 预热 2–3 分钟时
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`（队列错误回调）abort，当时 KV 只用 4–9%、a7b 每卡空闲约 40 GB。
- a7c 改 `HSA_NO_SCRATCH_RECLAIM=0`（恢复 ROCr 回收 scratch 后重试）后跑完全程、无错误：判断是 kernel 申请 scratch 失败、
  NSR=1 下不回收直接 abort（a6 无 coordination 时也遇到过一次）。
- a7c 结果 27,607 tok/s/GPU（a6b 32,228）：decode 每输出 token 35.9 → 29.9 ms（−17%，coordination 有效），但新算 prefill
  4,961 → 4,219 tok/s（−15%，步数 4 倍、每步固定开销与等 decode rank 的 eager draft），排队 19.6 → 37.4、TTFT 均值
  21.6 → 46.8 s、在跑 36.5 → 25.4，每秒完成请求 1.045 → 0.819。下一步 a7d：保留 coordination，块恢复 8192。

### a7d：coordination + 8192-token chunks（mem 0.85，NSR=0）C64：29,307 tok/s/GPU

| 轮次 | tok/s/GPU | 每秒请求 | decode ms/token | TTFT 均值 | 新算 prefill | 在跑 / 排队 |
|---|---|---|---|---|---|---|
| a6b（无 coordination，mem 0.90，NSR=1） | 32,228 | 1.045 | 35.9 | 21.6 s | 4,961/s | 36.5 / 19.6 |
| a7c（coordination，2048 块） | 27,607 | 0.819 | 29.9 | 46.8 s | 4,219/s | 25.4 / 37.4 |
| a7d（coordination，8192 块） | 29,307 | 0.945 | 35.8 | 29.7 s | 4,825/s | 33.3 / 25.4 |

8192 块下 coordination 对 decode 没有改善，TTFT 与排队变差；2048 块下 decode 快 17% 但 prefill 损失更大。a7d 与 a6b 另有
mem（0.85 对 0.90，GPU 命中 27% 对 48%）与 NSR（0 对 1）两处不同，但无论怎么分摊，coordination 在 TP4 这个场景下没有净收益，
此方向收尾。TP4 aggregated 的最好成绩仍是 a6b 32,228 / a3 C48 31,605，约 3.1–3.2 万为当前软件栈下的平台。

### prefill bf16 GEMM 调优的准备（未执行）

a6b 日志中未调优的 bf16 GEMM：5,542 种 (M, N, K)，几乎都是 prefill 尺寸，4 类 (N, K)：(6144, 16384)、(2624, 6144)、(32, 6144)、
(4096, 2048)，M 为 257–8192（清单：`analysis/untuned_bf16_gemm_shapes.a6b.txt`）。aiter `tuned_gemm` 先按原始 M 精确查表，
再按 `get_padded_m(M, N, K, gl)` 补齐后查：gl=1 时每类 (N, K) 只有 5 个档位（512/1024/2048/4096/8192），gl=0 时 104 个。
调 4 × 5 = 20 个形状即可让所有 prefill GEMM 命中调优配置。普通 GEMM 约占 prefill 步 25%、prefill 步约占时间 46%，
调优若快 10–30%，总耗时约降 1–3%。

### 可视化与实验总表（10-19）

`results/summary/`，由 `scripts/plot_agg_rounds.py` 生成：

| 文件 | 内容 |
|---|---|
| `agg_rounds.md` / `.csv` / `.json` | 每行一个实验（含失败和未跑的轮次，另加 B300 C24/C32/C48）：生效配置与测试数据 |
| `agg_rounds.png` / `.svg` | 各计分轮与 B300 C48 并排：总吞吐、输出吞吐、每秒请求与平均输入、ITL、TTFT、缓存命中、有效并发、每 rank 在跑 / 排队 |
| `agg_vs_concurrency.png` / `.svg` | 吞吐、decode 每 token 耗时、TTFT p50、每秒请求随并发变化（B300 曲线 + 各轮散点） |
| `agg_pareto_interactivity.png` / `.svg` | InferenceX 视角：tok/s/GPU 对 interactivity p90 / p50（B300 曲线、各轮散点、我们的前沿） |
| `decode_vs_prefill_c48_b300_a3.txt` | B300 C48 与 a3 C48 的纯 decode 单步耗时、decode / eager 时间占比（`pass_timeslices.py`） |
| `gpu_time_breakdown.png` / `.svg` | a6b profiling：decode 步 / eager 步时间占比、各 rank prefill 步数、kernel 类别构成 |

配置按启动脚本的方式 source `config/config.agg.sh` 与 `config/round.<ROUND>.sh` 得到，并与结果目录的 `round.env` 逐项核对；
客户端与引擎数据沿用 `agg_report.py`。命中率用 AIPerf 的 API 计入命中：AgentX 记录的服务端 GPU / 主机拆分在 a3 C48 与 a4 两者
之和超过 100%，拆分只在表中作参考。重绘分两步（表格需 Python ≥ 3.12，画图需要 matplotlib；登录节点上没有 matplotlib，
图在 138 上用 a6b 镜像的一次性容器画，不挂 GPU、不联网）：

```bash
python3 -B scripts/plot_agg_rounds.py --table-only
ssh crsuse2-m2m-138 "docker run --rm --network none -u $(id -u):$(id -g) -e MPLCONFIGDIR=/tmp/mpl \
  -v $PWD/..:$PWD/.. -w $PWD --entrypoint python3 infera-sglang:v0521-llying-1012-mc4340-destpin \
  scripts/plot_agg_rounds.py --plot-only"
```

### InferenceX 视角：吞吐与 interactivity p90（10-19）

InferenceX 上要的是 tokens/s/GPU 尽量大、同时 interactivity（每请求 tok/s/user）的 p90 也尽量大。
`results/summary/agg_pareto_interactivity.png` 画的是 tok/s/GPU 对 interactivity p90 / p50。

| 点 | tok/s/GPU | interactivity p50 / p90 |
|---|---|---|
| a3 C48 | 31,605 | 40.4 / **20.8** |
| a6b C64 | 32,228 | 30.5 / 17.3 |
| a4 C64 | 31,221 | 26.9 / 14.5 |
| a7d C64 | 29,307 | 32.6 / 16.8 |
| t8c C128（8 卡） | 23,366 | 21.2 / 9.3 |
| B300 C32 / C48 | 37,390 / 42,047 | 86.6 / 44.9，64.5 / 38.7 |

两个指标一起看，目前最好的是 **a3 C48**：吞吐比 a6b C64 低 2%（在 ±3% 的波动内），interactivity p90 高 20%。C64 的各轮吞吐没有
提升，只是把延迟变差。我们的点都在 interactivity p90 ≤ 21 的区间，B300 在 39–267；要上 InferenceX，至少要把同等 interactivity
下的吞吐做出来，而不仅是单点吞吐。

### 与 B300 C48 的差距拆解（10-19）

同为 C48：B300 42,047，a3 31,605，差 1.33 倍（数据见 `results/summary/agg_rounds.md`）。

| | B300 C48 | a3 C48 |
|---|---|---|
| 每秒完成请求 | 1.200 | 0.947（−21%） |
| 平均每请求 token（输入 + 输出） | 140.2k | 133.6k（−5%） |
| 端到端延迟均值 | 24.8 s | 37.2 s |
| TTFT 均值 / p50 | 6.4 / 3.6 s | 7.5 / 4.8 s |
| decode 阶段每输出 token（（端到端 − TTFT）/ 输出） | 18.5 ms | 30.6 ms |
| ITL 均值 / p50（比值） | 19.0 / 15.5 ms（1.23） | 44.1 / 24.8 ms（1.78） |
| 缓存命中 | 94.8% | 96.1% |

- 1.33 ≈ 1.27（请求速率）× 1.05（每请求 token 数），约 85% 的差距来自请求速率；缓存命中不是原因。
- 请求速率的差距几乎都在 decode 阶段：TTFT 只差 1.1 s，decode 阶段每请求差约 11 s。
**实测拆分（10-19，`scripts/pass_timeslices.py`，输出 `results/summary/decode_vs_prefill_c48_b300_a3.txt`）。** 两边 AIPerf 的
`server_metrics_export.json` 都有 1 秒粒度的 `sglang:cuda_graph_passes`（按 rank、按 decode / prefill 模式）。所有 rank 都没有
prefill 的 1 秒片里，"1 s / decode 步数"就是纯 decode 单步耗时；再用 decode 总步数 × 单步耗时得到 decode 占墙钟时间的比例，其余为
eager（prefill）步。B300 数据来自 CI 产物 `11798041008`。

| 每个 DP rank，测量窗口 60 分钟 | B300 C48 | a3 C48 |
|---|---|---|
| decode 步速率 | 15.2 步/s | 9.6 步/s |
| 纯 decode 单步（纯 decode 片占比） | 47.5 ms，在跑 5.6（53%） | 65.6 ms，在跑 7.1（52%） |
| decode / eager 占墙钟时间 | 72% / 28% | 63% / 37% |
| prefill 步（4 个 rank 之和） | 4,119（917 / 968 / 871 / 1,363） | 3,485（756 / 851 / 1,059 / 819） |
| 每次 rank prefill 摊到的 eager 时间 | 246 ms | 387 ms |
| 平均每个 decode 步摊到的 eager 时间 | 18.3 ms | 38.6 ms |

- 平均每个 decode 步的墙钟时间：B300 65.8 ms（47.5 纯 decode + 18.3 eager），a3 104 ms（65.6 + 38.6），差 38 ms。其中约 18 ms 来自纯
  decode 更慢（65.6 对 47.5 ms，a3 每 rank 多跑 1.5 个请求，有一部分是因为慢而积压），约 20 ms 来自 prefill 打断更多（2.1 倍 = 每次 rank
  prefill 慢 1.6 倍 × 每个 decode 步摊到的 prefill 次数多 1.3 倍）。两项大致各占一半。
- 折成每输出 token：B300 13.2 + 5.1 ms，a3 18.2 + 10.7 ms，与客户端的 18.5 / 30.6 ms 一致。
- **更正**：这一节最初按 a6b C64 的 profiling（72.7 ms/步）推断"去掉打断后我们的纯 decode 已接近 B300 的整体水平、差距主要在打断"。
  实测 C48 下纯 decode 也慢约 1.4 倍（同 batch 下估计 1.2–1.3 倍），打断约占差距的一半，不是绝大部分。
- B300 也有同样的 delayer 现象：各 rank 的判定以 "mixed" 为主，等待 59 步（上限 60）的占多数，prefill 步也不均衡（DP3 比其他多约 45%）。
  它的打断代价小，主要是因为每次 prefill 快。

### 单 rank prefill 空等：机制、量化与可选方案（10-19）

**机制（读 a6b 镜像中的 v0.5.21 源码；v0.5.22 tag 的这几处与之相同，`prefill_delayer.py` 逐字节一致）：**

- `scheduler.py` `get_next_batch_to_run`：spec + DP attention 且未开 coordination 时，只要有一个 rank 有新的 prefill 批次，其他 rank
  经 `maybe_prepare_mlp_sync_batch` 拿到 IDLE 批次（"prefill and decode batches will not be mixed"）。
- `schedule_policy.py` `add_chunked_req`：正在分块续算的 rank 向 delayer 报 prefillable，并且无论结论如何本步都执行续算。
- `prefill_delayer.py` `_negotiate_should_allow_prefill_pure`：所有 rank 都 prefillable（"all"）才放行，部分 rank prefillable（"mixed"）
  就推迟，最多 `max_delay_passes` 步。
- 合起来：A 在续算、B / C 有可接纳的请求、D 没有时，结论是 "mixed"，B / C 被推迟，这一步只有 A 在算、B / C / D 跑 IDLE。这一步
  本来就是 eager 步，B / C 跟着 prefill 不会多出 eager 步。

**量化（a6b C64 测量窗口，`scripts/prefill_step_timeline.py`，输出 `results/prof-a6b/prefill_timeline.a6b.txt`）：** 服务端日志里各 rank
的 `Prefill batch` 行按 30 ms 间隔归成步（同一步内各 rank 相差 < 20 ms，相邻步 ≥ 200 ms）。

| 指标 | 值 |
|---|---|
| prefill（eager）步 | 1,606 个 / 60 分钟，平均每步 1.92 个 rank 在算 |
| 每步在 prefill 的 rank 数：1 / 2 / 3 / 4 | 55.0% / 15.0% / 12.8% / 17.2% |
| 单 rank 步中，该 rank 在续算上一步的满块 | 869 / 883 |
| 单 rank 步中，至少一个空等的 rank 有排队请求 | 827 / 883（按该 rank 最近一次记录的 `#queue-req`，含暂时不可接纳的请求，偏高） |
| 各 rank 新算 token（DP0 / 1 / 2 / 3） | 3.12M / 6.02M / 2.91M / 5.80M |
| 各 rank prefill 步（DP0 / 1 / 2 / 3） | 581 / 984 / 579 / 944 |
| router 会话亲和 | 3,868 次按亲和定位，634 次重新选择 |

**可选方案（预计收益按 eager 步约占 45.5% 时间换算，均为上限或估计）：**

| 方案 | 机制 | 预计 | 代价 / 风险 |
|---|---|---|---|
| 1. delayer 放行搭便车 prefill | 任一 rank 在续算时，其他可 prefill 的 rank 同步放行（all-gather 多带一个续算标志，"mixed" 分支据此放行），约二十行 | eager 步下限 1,606 → 984（最忙 rank），−39%，上限约 +20%；合并后单步变长（MoE、通信随总 token 增加），估计 +5–12% | 不增加 eager 步、decode 不走 eager，不涉及 coordination 的 scratch 问题；TTFT 应改善 |
| 2. router 按 prefill 负载选 rank | 新会话选 rank 时计入各 rank 后续的新算负担（DP1 / DP3 新算量约为 DP0 / DP2 的 2 倍，且由会话亲和固定） | 与 1 叠加后 eager 步下限 772，上限约 +30% | 改 router；会话一旦绑定就不再纠正，效果不确定 |
| 3. DSA prefill CP | 把单个长 prompt 的 prefill 拆到多个 rank | — | `attn_cp_size = tp_size / attn_dp_size`，TP4 + DPA4 下为 1，要 CP 就得放弃 DP attention；官方注明实验特性、只在 Hopper 验证。近期不可行 |
| 4. prefill kernel | prefill 步中注意力/DSA 48%、普通 GEMM 25% | GEMM 调优 1–3%；注意力/DSA 需要 kernel 工作 | kernel 层面 |

候选的下一个实验是方案 1（尚未执行，待讨论）：负载下 decode 正确性检查通过后测 C64，与 a6b 比较每秒完成请求、eager 步数
（`prefill_step_timeline.py`）和 TTFT。

### b1（crsuse2-m2m-140）：delayer 搭便车补丁，a3 C48 配置（10-19）

按上一节的方案 1 实现，运行时挂载（`config/mounts.piggyback.tsv`），用 `SGLANG_PREFILL_DELAYER_PIGGYBACK=1` 打开，默认关：

- `patches/runtime-mounts/prefill_delayer.llying-piggyback.py`（对 v0.5.21 镜像原文件的 diff：`*.diff`，26 行）：跨 rank all-gather
  多带一个 `mid_chunk` 字段；判定为 "mixed" 时，只要有 rank 在续算，就放行（原因记为 `piggyback`，指标里可见）。
- `patches/runtime-mounts/schedule_policy.llying-piggyback.py`（1 行）：`add_chunked_req` 协商时带上 `local_mid_chunk=True`。
- `patches/runtime-mounts/test_piggyback_delayer.py`：在镜像里用 CPU 跑判定函数（all-gather 换成固定张量），5 项全过：续算时放行、
  开关关时照旧推迟、无人续算时照旧推迟、全部可 prefill 时照旧放行、等满上限时照旧超时放行。

节点：138 当时在做 v0.5.22 镜像升级，b1 在 140 上跑（`config.agg.sh` 新增 `AGG_NODE`，round 文件里设置；140 的拓扑在
`config/topology.agg.140.tsv`，AITER JIT 缓存从 138 拷贝）。b1 = a3 C48 的全部设置（write_through 270 GB、mem 0.85、delayer 60、
1 个 tokenizer）+ 补丁。负载下 decode 正确性检查 32/32 通过（180 个后台 prefill，0 错误），C48 于 17:07:49 开始。

| | a3 C48（138，10-15） | b1 C48（140，10-19，piggyback） | 变化 |
|---|---|---|---|
| tok/s/GPU | 31,605 | 33,677 | +6.6% |
| 完成请求 / 每秒请求 | 3,409 / 0.947 | 3,637 / 1.002 | +5.8% |
| interactivity p50 / p90 | 40.4 / 20.8 | 40.8 / 19.5 | p90 −6% |
| TTFT p50 / p90 (s) | 4.8 / 14.5 | 3.7 / 9.2 | |
| ITL p50 / p90 (ms) | 24.8 / 48.1 | 24.5 / 51.3 | |
| eager 步数 / 单 rank prefill 占比 | 1,856 / 61.3% | 1,796 / 48.0% | |
| 有其他 rank 排队的单 rank 步 | 658 | 396 | −40% |

（prefill 步统计来自 `scripts/prefill_step_timeline.py`：`results/b1-a3-piggyback/c48/prefill_timeline.txt` 与
`results/prof-a6b/prefill_timeline.a3c48.txt`。）补丁按设计生效：两个 rank 同步 prefill 的步从 12.3% 升到 27.6%，"一个 rank 续算、
别的 rank 有请求在排队却空等"的步少了 40%，排队的请求更早开始，TTFT 明显下降、完成请求多 5.8%。代价是慢的一端 decode 被 prefill
打断得更多，interactivity p90 从 20.8 降到 19.5（ITL p90 48.1 → 51.3 ms）。**这是跨节点比较**，需要同节点对照。

**b0（140，b1 去掉开关的对照）无效。** b0 预热 18:55 开始、计分窗口 19:22-20:22；另一位用户的两个作业 19:03（atom，GPU 4）和
19:10（sglang v0.5.21，GPU 7，`--ipc host`）起在同一节点上，整段计分窗口重叠；20:24 时 GPU 4 占用 100%、主机内存只剩 37 GB、
load average 12-17。b0 只有 23,069 tok/s/GPU、2,538 个请求（预热 1,604 s，b1 为 1,120 s），远低于 a3 和 b1，判为受干扰，
`results/b0-a3-140/c48/FAILED.txt` 记下了经过，总表不计分。140 之后不再空闲，按要求暂停 140 上的实验，栈已撤下。
同节点对照改为：138 空出后在 138 上跑"a3 + piggyback"（与 a3 同节点、同镜像）——b2（`config/round.b2.sh`、`operations/b2.steps`），
10-19 23:00 在 138 GPU 0-3 启动，期间 138 不跑其他东西。10-20 00:08 起 140 空闲（他人容器已撤、8 卡显存 0、load 1.5），
00:09 在 140 上重跑 b0（b0r，`config/round.b0r.sh`，与 b0 完全相同），给 b1 补上同节点对照。b2 之后在 138 上跑 b3
（a3 去掉 DP attention，纯 TP4，`config/round.b3.sh`）：正确性排查表明并发下只有纯 TP4 输出正确，b3 量它在 C48 的代价。

**b2（138，b1 的设置放到 a3 的节点上，10-19 23:31-00:31）：同节点上 piggyback 不是改进。**

| | a3 C48（138，无 piggyback） | b2 C48（138，piggyback） | 变化 |
|---|---|---|---|
| tok/s/GPU | 31,605 | 31,004 | −1.9% |
| 完成请求 / 每秒请求 | 3,409 / 0.947 | 3,432 / 0.946 | 持平 |
| interactivity p50 / p90 | 40.4 / 20.8 | 39.7 / 18.6 | p90 −11% |
| TTFT p50 / p90 (s) | 4.8 / 14.5 | 3.9 / 11.0 | |
| ITL p50 / p90 (ms) | 24.8 / 48.1 | 25.2 / 53.9 | |
| eager 步数 / 单 rank prefill 占比 | 1,856 / 61.3% | 1,824 / 53.3% | |
| 有其他 rank 排队的单 rank 步 | 658 | 468 | −29% |

补丁照样生效（两 rank 同步的 prefill 步 12.3% → 24.1%，空等的步减少），但 prefill 总步数几乎不变（1,856 → 1,824），每步
prefill 量也差不多：多个 rank 一起 prefill 时，这一步的耗时由最慢的 rank 决定，decode 被打断的总时间没有减少，只是排队的请求
更早开始——TTFT 降、吞吐不变、慢的一端 ITL 变差。按 InferenceX 的两个轴（tok/s/GPU、interactivity p90）它是退步。b1 相对 a3 的
+6.6% 主要来自节点或日间差异，b0r（140 上的同节点对照）会给出最终结论。

同一份统计里更大的杠杆是 **DP rank 间 prefill 负载不均**：b2 中 DP0 做了 595 万新 token、1,029 个 prefill 步，DP2 只有 317 万、
726 步；若各 rank 均衡，eager 步数下限 809，现在 1,824（a3 也是 DP2 偏重：569 万对 329 万）。session 亲和性把会话钉在 rank 上
是主要原因之一，下一步的方向是 router 按各 rank 待 prefill 的 token 量分配新会话（`prefill_timeline.txt` 在
`results/b2-a3-piggyback-138/c48/`）。

**b3（138，纯 TP4，chunk 8192）预热中手动中止，b3b 重跑。** 我把 b3 的 chunk 设成 8192，理由写成"一步的 prefill 与 DP4 一个 rank
相当"——这是错的：TP4 一步里每个 token 由 4 张卡分摊，32768 token 的一步对每张卡的开销才相当于 DP4 一个 rank 的 8192。按 8192
跑，TP4 的 prefill 能力只有 DP4 的 1/4：预热 9.5 分钟只回来 47 / 531，引擎待 prefill 168 万 token、约 1.17 万 tok/s。01:12 停掉，
`round.b3.sh` 保留当时的配置并注明中止；b3b（`config/round.b3b.sh`）把 chunk 与 max-prefill 都设为 32768，01:14 启动（先等 b3
栈的显存回收）。纯 TP4 的 GPU KV 池是 249 万 token（MLA latent KV 在 TP rank 间复制），DP4 是每 rank 206 万、共 823 万。

顺带修了 `scripts/stop_stack.sh`：原来等节点上所有 GPU 都空才返回，共享节点上别人的作业会让它等满 30 分钟报错、卡住后续步骤；
现在只看本轮 `GPUS_mixed_0` 的卡。

### 10-16 12:50 /home 卷写满后的运行方式

共享 `/home`（10 TB CephFS）写满（剩余 0 GB，新建目录报 Disk quota exceeded），git 提交与 `/home` 下的日志都写不进去。
3P1D 的 `runtime/runs/r2`（1,491 个文件、7.40 GB，逐字节核对）已迁到
`/shared_nfs/liyingli/bench_agentx/3p1d-tp4-tp8-v0521-20261012/runtime-home-r2-20261016/`，腾出的空间随即被占用。
之后的实验从 `/shared_nfs/liyingli/bench_agentx/agg-tp4dpa-138-v0521-20261015-kit/`（本目录的副本，节点上只读、只用来读脚本）
运行：`RUNS`、结果、步骤日志都写在 138 本地盘 `/mnt/m2m_nobackup/liyingli_3p1d/agg-runs`、`agg-results`，
监视脚本经 ssh 在 138 上运行；结果定期同步到 `/shared_nfs`，`/home` 恢复后再提交到 git。

## 计划

1. a2：C16 / 600 s 调通全流程（flush → router → 采样 → AgentX → 发布 → 汇总）。
2. 开 HiCache（write_through）正式测 C32、C48（3600 s，预热每 lane 10 个请求）。
3. 若 write_through 下并发上不去（KV 或主机命中受限），尽快换 write_back（B300 的做法）。
4. 之后按每点的瓶颈分析决定是否加并发、调 mem-fraction / max-running / 调度参数。
