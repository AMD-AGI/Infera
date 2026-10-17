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
- decode 校验步（cap2 TP-0/1，各 29 步，72.7 ms/步，每 rank bs 8–13）：MoE 35–36%，注意力/DSA 24–26%，通信 15–19%，普通 GEMM 16–17%。
- 单 rank 长 prefill（cap1，同一 8.4 s 窗口）：TP-1 连续 8 个 prefill 批次（665 ms/个，一个长 prompt 的分块续算，delayer 不管续算），
  构成为注意力/DSA 42%、普通 GEMM 25%、MoE 14%、通信 13%；其余 3 个 rank 这期间在通信 kernel 里空等（TP-0 通信占 78%），该窗口
  decode 只占约 15%。
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
