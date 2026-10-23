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

### HiCache 撤栈对同节点的影响

18:13 撤下 PD 栈后，同节点 GPU 0–3 上新起的 mixed 引擎停在 D 状态、没有日志，直到 GPU 4–7 回收完（约 9 分钟）才继续——锁页主机池回收期间新进程的 KFD
初始化被挡住（以前记录过）。之后正确性类测试 prefill 关 HiCache，性能 A/B 保持开。
