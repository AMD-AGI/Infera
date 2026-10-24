# prefill 提效与 custom all-reduce（10-23 起）

目标（用户 10-23 17:56）：在 PR #210 的 PD example（PP4 prefill + decode 均衡）基础上，做 prefill 的 profiling 与配置实验，并解决 custom
all-reduce 的正确性问题；decode kernel 由同事负责。镜像 `infera-sglang:v0522-ex-3ad7527c`（138 / 140 都有）。前一阶段见
`../2p1d-v0522-20261020/README.zh-CN.md` 末节。

## 结论（持续更新）

（进行中）

## 已知事实（开始前）

- **PP4 prefill 的上限是"负载下每秒新算约 12k–13k token"**（4+8 C88–C112 合 3.0 req/s；2P C192 每实例约 2.3 req/s，因为每请求新算更多）；冷 prompt 时
  25.8k，负载下的小增量请求要对十几万 token 的缓存上下文做 DSA indexer 与 attention。
- **PP4 各级耗时**（`../2p1d-v0522-20261020/scripts/pp_stage_times.py`，4+8 C88 均衡 `c88-1023T0854` 的计分窗口，满 8k batch，中位缓存上下文 12.6 万、3 个请求）：
  相邻级时间差 PP0→1 422 ms、PP1→2 434 ms、PP2→3 356 ms，四级节拍都是 445 ms（p90 约 1,160 ms）。前两级比第三级慢约 20%；最后一级耗时日志里看不出，
  需要 profile。
- **custom all-reduce**（`../mix-v0522-correctness-20261019/`）：聚合 TP4 + DP4 attention 在持续并发下乱码，必要条件是 DP attention + decode CUDA graph，
  乱码条数与 EXTEND 步数相当；关 custom all-reduce（`ca_comm` 不初始化）或关 graph 都干净；单独关 custom all-gather / reduce-scatter、
  `SGLANG_AITER_CAPTURE_COPY_IN=1`、QuickReduce、融合算子、MTP、HiCache、delayer、GatherV、DP lm-head、DSA 后端 / top-k 都无效。
  PD 的 decode 节点只跑 decode 步（没有 EXTEND），v0.5.19 的 3P1D 开着 custom all-reduce GSM8K 0.97；我们的 2P1D 一直沿用 mixed 的结论关着它，没有在 PD decode 上验证过。
  PP4 prefill 每级是 TP1，没有 TP 集合通信，与 custom all-reduce 无关。

## 计划与用卡

| 步骤 | 用卡 | 内容 |
|---|---|---|
| 1 | 138 GPU 4–7 + 140 | PD 4+8，decode 开 custom all-reduce，真实接受率：smoke、GSM8K 全量、200 题 / 并发 64 的乱码探测（经 router） |
| 1' | 138 GPU 0–3（并行） | mixed 最简配置（关 MTP、关 HiCache）：新镜像复现 → 关 custom all-gather + reduce-scatter 只留 all-reduce → 整体关 custom all-reduce（对照） |
| 2 | 同 1 | prefill profile：4+8 C88（均衡开）25 分钟，预热后 10 分钟抓 prefill 各级 30 步；若 1 通过，decode 开 custom all-reduce，顺带比 decode 每步 |
| 3 | 同 1 | prefill 配置 A/B：4+8 C112（prefill 饱和，均衡开）——层切分、`--pp-async-batch-depth`、prefill 显存比例 |
| 4 | 16 卡 | 最好的组合在 2P1D C176 / C192 上确认高吞吐端 |

性能 A/B 计分期间不在同一节点上跑 mixed。

## 复现

- 实验 kit `../../../2p1d-kit/ex-exp`（= PR head `8859b247` 的 example 加三个开关：`DECODE_CUSTOM_AR`、`PP_LAYER_PARTITION`、`PREFILL_MEM_FRACTION`；
  `../../../2p1d-kit/ex-8859b247` 是未改的快照）。
- 正确性：`../2p1d-v0522-20261020/scripts/pd_correct_chain.sh`（`PROBE_PROMPTS=<lm-eval GSM8K samples>` 时在 GSM8K 之后跑 200 题探测）。
- mixed：`scripts/car_bisect_chain.sh config/site-bisect-138.env "NAME KEY=VALUE ..." ...`（调用 `../mix-v0522-correctness-20261019/scripts/engine_bisect.sh`）。
- profile：`scripts/prefill_profile_chain.sh KIT SITE LOG_DIR CONC [DURATION] [PROFILE_AFTER] [NUM_STEPS]`；`scripts/prof_breakdown.py TRACE...`。

## 进度

- 18:02 步骤 1 起栈（`config/../../2p1d-v0522-20261020/config/site-exp-1p-car.env`，138 `$W/2p1d/exp/`）；18:03 步骤 1' 开始（138 `$W/bisect/car-*`）。

### 步骤 1：PD decode 开 custom all-reduce 的正确性——通过

4+8，decode TP8 / DP8 attention 开 custom all-reduce（prefill PP4 不变），真实接受率，`car-correct-1023T1802`：

| 检查 | decode 开 custom all-reduce | 同镜像关（example 正确性，`correct-ex-1023T1034`） |
|---|---|---|
| smoke | 6/6 | 6/6 |
| GSM8K（1,319，并发 64）strict / flexible | 0.9659 / 0.9674 | 0.9666 / 0.9674 |
| 200 题 / 并发 64 探测（经 router） | 0.985，乱码 0，截断 2（doc 119、182，逐条看是连贯的反复推敲；119 是每次都推敲到上限的歧义题） | — |
| 真实接受长度 | 中位 4.64，p10 4.18 | 4.62 |
| 长上下文（`prof-c88` 起栈后，128 个约 13.9k token 请求、并发 32） | 捞针 64/64；长前缀 GSM8K 62/64，乱码 0，截断 1（连贯） | 以前 mixed 关 custom all-reduce：64/64、62/64 |

decode 日志里的 8 行 Traceback 都在 18:13:40 撤栈时（SIGTERM 后的退出）。与"乱码需要 EXTEND 步与 decode graph 交替"一致：PD 的 decode 只有 decode 步。

### 步骤 1'：mixed 定位到 custom all-reduce 本身（最简配置：关 MTP、关 HiCache，200 题 / 并发 64 直连引擎）

| 变体 | correct | 乱码 | 截断 |
|---|---|---|---|
| car-new-mtp0hc0：新镜像，开 custom all-reduce | 0.690 | 50 | 58 |
| car-agrs0-mtp0hc0：`SGLANG_DP_USE_REDUCE_SCATTER=0` + `SGLANG_USE_AITER_AG=0`（custom all-gather / reduce-scatter 都关，只留 all-reduce） | 0.710 | **50** | 51 |
| car-dcar-mtp0hc0：`--disable-custom-all-reduce`（对照） | 0.990 | 0 | 1 |

- 元凶是 custom all-reduce 这个操作本身。
- 代码（v0.5.22）：ROCm 上 `dispatch_custom_allreduce` 返回 aiter 的 `CustomAllreduce`，`enable_register_for_capturing = not SGLANG_MEMORY_SAVER_CUDA_GRAPH`，
  即 graph 捕获时走"注册 graph 池张量"路径（`registered_input=True`）。sglang 在 `parallel_state.py` 对 custom all-gather / reduce-scatter 的注释写明这条路径
  "makes peers read stale data over IPC on replay (garbage rows / NaN under DP attention + TP MoE on MI355X)"，并提供 `SGLANG_AITER_CAPTURE_COPY_IN=1` 让它们
  拷进预注册的 IPC 池，但 all-reduce 没有对应开关；aiter 自己对 gfx1250 已强制走拷入。
- 验证（`patches/custom_all_reduce.py`：新增 `SGLANG_AITER_AR_CAPTURE_COPY_IN`，置 1 时 `enable_register_for_capturing=False`，挂进容器，镜像不变）：
  V1 只让 all-reduce 拷入、V2 三者都拷入，18:40 起。
- `../mix-v0522-correctness-20261019/scripts/engine_bisect.sh` 加了 `EXTRA_MOUNTS`（教训：正在被 bash 执行的脚本不能原地改写，bash 按偏移边读边执行；
  18:20 误改后已原样写回同一 inode，等那条串行跑完才替换）。

V1（`car-arcopy-mtp0hc0`，只让 all-reduce 捕获时拷入，custom all-gather / reduce-scatter 仍走注册路径）：**0.975，乱码 0**，截断 4（doc 85、87、119、182，
逐条看是连贯的反复推敲）；引擎日志 4 个 rank 都打印了 `all-reduce capture copy-in`，且仍是 `Using AiterCustomAllreduce`（另一条
`All-reduce call path: NCCL (custom AR disabled)` 在乱码的基线里同样有 4 条，是只有 1 个 rank 的组；整体关的对照是 8 条）。V2（三者都拷入）0.975 / 0 / 4。
开 MTP 再验（`car-arcopy-mtp1hc0`）：0.980 / 0 / 3。**根因：MI355X 上 aiter custom all-reduce 在 CUDA graph 捕获时注册 graph 池里的输入张量；聚合部署里
eager 的 EXTEND 步与 decode graph 回放交替时，对端回放读到旧数据。all-reduce 捕获时改为拷进预注册的 IPC 池即可，custom all-reduce 保留。**

### 步骤 2：prefill profile（`prof-c88-1023T1822`，真实接受率，4+8 C88，预热结束 10 分钟后每级 30 步）

trace 在 138 `$W/2p1d/exp/traces/prefill-185742-p0/`，`scripts/prof_breakdown.py` 的输出：

| | PP0 | PP1 | PP2 | PP3 |
|---|---|---|---|---|
| GPU 忙碌（30 步，折每步） | 11.37 s（≈379 ms） | 11.80 s（≈393 ms） | 11.59 s（≈386 ms） | 9.23 s（≈308 ms） |
| DSA 稀疏 MLA attention（triton `_sparse_mla_fwd_split_dim_kernel`） | 18% | 21% | 22% | 25% |
| DSA indexer（`_gluon_fp8_mqa_logits` + `topk_packed`） | 5% | 10% | 9% | 7% |
| GEMM（BF16 hipBLASLt `Cijk_*MT256x256`、MXFP4 `gemm2_a4w4`、a16w16、batched a8w8） | 14% | 18% | 19% | 21% |
| MoE（`mfma_moe1_*afp4_wfp4`） | 4% | 5% | 6% | 8% |
| NCCL（级间收发） | 17% | 27% | 26% | 13% |
| 拷贝 / 其他（HtoD、direct copy、Fill 等） | 32% | 19% | 14% | 24% |
（占 kernel 时间的比例；kernel 时间含并行流，所以大于忙碌时间。）

- **最大的单项是 DSA 稀疏 attention 的 triton kernel**（每级每步约 100 ms），DSA 合计约占计算的 30%–40%；MoE 只有约 5%。负载下 prefill 慢在长上下文的
  attention / indexer。
- **PP0 是瓶颈**：采样时段 88% 忙，且比其他级多约 2 s / 30 步的输入准备拷贝（HtoD 1.94 s、direct copy 0.99 s、Fill 0.56 s）；PP 关了 overlap 调度，
  CPU 准备 batch 时 GPU 空着（约 12%）。379 ms 的 kernel 加空隙约 430 ms，与日志节拍吻合。
- **NCCL 时间多是空泡**：级间每次只传约 100–200 MB 激活，按带宽几毫秒；中间两级 26%–27% 的收发时间主要是在等上游。
- 日志对齐后的各级时间差（`pp_stage_times.py` 已改为按 (请求数, 新 token, 缓存 token) 对齐，窗口切在流水线中间时不再错位）：

  | 运行 | PP0→1 | PP1→2 | PP2→3 | 节拍 |
  |---|---|---|---|---|
  | `prof-c88`（真实接受率，中位缓存 10.6 万） | 361 | 372 | 320 | 384 ms |
  | `c88-1023T0854`（强制 3.61，12.6 万） | 422 | 434 | 356 | 445 ms |

  前两级稳定比第三级慢 13%–20%，最后一级最轻。

**配置实验候选**：层切分 19/19/20/20（估计节拍 −5%）；`--pp-async-batch-depth 1`（让下一个 batch 的准备与计算重叠，针对 PP0 的空隙）；DSA prefill 后端
`aiter`（v0.5.22 可选 `aiter`、`tilelang`、`triton_sparse_mla` 等；先过正确性，含长上下文）。

### DSA prefill 后端 `aiter`：长上下文 OOM，暂不能用

聚合单机（关 custom all-reduce、关 MTP / HiCache，138 GPU 0–3）：

| 变体 | 200 题 GSM8K | 长上下文（128 个约 13.9k token，并发 32） |
|---|---|---|
| `--dsa-prefill-backend aiter` | 0.985，乱码 0，截断 1 | **128 个全部报错**：四个 rank 同时 `torch.OutOfMemoryError: Tried to allocate 64.00 GiB` |
| `triton`（现用，对照） | — | 捞针 64/64，长前缀 GSM8K 63/64，乱码 0 |

OOM 在 `dsa_backend._forward_aiter_extend` → `aiter.mla.mla_decode_fwd` 的 `logits = torch.empty(...)`：extend 借用 decode kernel，按"token 数 × split 数 ×
头数 × 维度"预分配 logits，16k token 一次要 64 GiB；PD prefill 用 8k chunk 时约 32 GiB，PP4 每卡 KV 分配后剩 52–67 GB，余量太小。要用需改这块分配
（或降 chunk 到 4k，影响效率），列为待议，不进今晚的 A/B。

### 步骤 2'：decode 开 custom all-reduce 的性能——PD 4 卡 decode 上收益很小

4+4 C80，强制接受率 3.61，权重 0，decode 走默认注册路径（PD 已验证干净），`c80-1023T1941`（计分 20:03–21:03），产物 `../2p1d-v0522-20261020/results/exp-d4-car-c80/`：

| 4+4 C80 | tok/s/GPU | 每秒请求 | intvty p50 / p90 | ITL p50 / p90 (ms) | TTFT p50 / p90 (s) |
|---|---|---|---|---|---|
| custom all-reduce 关（kit，10-22） | 37,310 | 2.553 | 49.2 / 44.3 | 20.3 / 22.6 | 3.75 / 10.83 |
| custom all-reduce 开 | 37,400（+0.2%） | 2.563 | 50.1 / 45.0（+1.6%） | 19.9 / 22.2 | 3.74 / 11.75 |

同等最忙负载下每步只快 1–2 ms（@6 43 / 44、@9 55 / 56、@11 63 / 65、@13 65 / 67、@17 70 / 71、@20 73 / 74、@24 77 / 78 ms），约 1.5%–3%；
4+4 C80 本身又受 KV 限制。TP4 + DP attention 的 decode 里通信不是大头。结论：PD decode 开 custom all-reduce 正确但收益小，A/B 的 decode 保持关；
修复对聚合部署（mixed，以前关它 p90 −15%）更有意义。

### 步骤 3：prefill 配置 A/B（排在 4+4 C80 之后，`scripts/perf_plan.sh`，计划 138 `$W/2p1d/ab/plan.txt`）

4+8 C112（prefill 饱和，均衡开，`site-exp-1p.env`）：基线 → 层切分 19/19/20/20 → `--pp-async-batch-depth 1`。decode 保持关 custom all-reduce。
比较的量：prefill 新算 token/s、每秒请求、满 8k batch 的节拍（`pp_stage_times.py`），以及 tok/s/GPU、p90 交互。

| 组 | tok/s/GPU | 每秒请求 | intvty p50 / p90 | TTFT p50 / p90 (s) | 新算 tok/s | 节拍中位 / p90 (ms) | PP0→1 / 1→2 / 2→3 (ms) | 排队均值 / 有排队的 10 s |
|---|---|---|---|---|---|---|---|---|
| ab-base（20/20/20/18，`c112-1023T2116`，计分 21:43–22:43） | 30,071 | 2.990 | 79.2 / 57.6 | 5.41 / 11.35 | 12,980 | 481 / 1,162 | 459 / 467 / 386 | 30.6 / 94% |
| ab-split（19/19/20/20，`c112-1023T2257`，计分 23:25–00:25） | 29,758（−1.0%） | 2.951 | 81.9 / 66.8 | 5.44 / 11.89 | 12,614（−2.8%） | 485 / 1,210 | 457 / 472 / 412 | 32.2 / 94% |

- **层切分不改善 prefill**：前两级各减一层，时间差几乎不变（同一 42 分钟窗口 429 → 431、436 → 447 ms），节拍不降反升；这一组中位缓存上下文更长（13.5 万
  对 12.9 万）会略加重每个 batch，但解释不了前两级完全不变。前几级的时间主要不在层计算上，更像是每级"收 → 算 → 发"串行的调度开销（PP 关 overlap，
  PP0 还要准备输入）。另外最后一级多两层后 prefill KV 池 1,074 万 → 1,010 万 token（−5.9%）。p90 交互 57.6 → 66.8 是闭环连带（prefill 略慢 → decode
  最忙 rank p90 10 → 9），不是层切分本身的收益。不采用。

（中位缓存上下文 12.9 万、每 batch 3 个请求；decode 最忙 rank 中位 / p90 7 / 10。对照 kit 测的 4+8 C112 R2 29,540 @ 49.8、新算 13.1k；C88 均衡 31,088 @ 71.7、13.0k。）

### 操作上的两点

- `pd_perf_chain.sh` 等的是两台机器全部 GPU 空闲；同节点另一个任务卡在 HiCache 回收（D 状态）时，回收结束的一刻可能所有卡都空，性能测试会与之同时起栈。
  所以排队的性能测试改为显式等前一个任务的进程（19:18 停掉了一个这样的等待，没有起栈）。
- 等待要盯 `pd_perf_chain.sh` 本身的进程号，不要盯外层启动 shell（19:50 改正，同样没有起栈）。

### HiCache 撤栈对同节点的影响

18:13 撤下 PD 栈后，同节点 GPU 0–3 上新起的 mixed 引擎停在 D 状态、没有日志，直到 GPU 4–7 回收完（约 9 分钟）才继续——锁页主机池回收期间新进程的 KFD
初始化被挡住（以前记录过）。之后正确性类测试 prefill 关 HiCache，性能 A/B 保持开。
