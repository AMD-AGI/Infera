# KERNEL-PLAN-20261109 问答（11-09）

针对 Infera `dev/pd_opt/glm_5.2_agentx` 分支 `llying/dp-graph-variant-fix-aus-20261103/KERNEL-PLAN-20261109.zh-CN.md`
（最后修改 7021848b，11-09 08:07 UTC）的问答。同目录的其他文档（`KERNEL-PROGRESS-20261109`、`TP8-KERNEL-HANDOFF`、`RESULT`、`QA-20261106` 等）
作为背景。本地工作副本：`/perf_apps/liyingli/bench_agentx/kplan-qa-20261109/`。

约定：只解答，不执行任务、不占 GPU、不改队列。需要实测或需要原 agent 补充的，会在回答里单独标出"需要执行 / 需要问原 agent"。

---

## 问 1（09:29）：trace 里每个 scheduler batch 都有 draft、verify、draft_extend 三段，各做什么？一个 scheduler batch 是不是一步？verify 为什么要 42 ms 左右？里面那么多 all_gather，是不是因为有 78 层？

### 简答

- **一个 scheduler batch 就是一步**（文档里"ms/步"的"步"）。trace 里 CPU 侧的 span 叫 `scheduler.run_batch`（TP8 trace 里没有叫 `scheduler_batch` 的名字，你看到的应该就是它），
  每次调用一次 `forward_batch_generation`，里面按顺序跑 draft → verify → draft_extend。这是 SGLang 的 EAGLE（MTP）投机解码，
  我们的配置是 K3：`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（`recipe-e10.env`）。
- **draft 和 draft_extend 跑的是 1 层的 NextN 草稿模型，verify 跑的是 78 层的目标模型**，所以 verify 占了步长的 90%。
- **all-gather 多，是的，因为有 78 层**：DP attention + TP MoE 的结构下，每层固定 1 次 all-gather + 1 次 reduce-scatter，verify 每步各 78 次。
- **42 ms 的构成**：78 层平均每层约 0.54 ms。光读权重的下限约 11 ms，剩下的时间来自 kernel 带宽效率低（MoE、absorb BMM）、随上下文长度增长的注意力 / indexer、
  通信，以及约 2,080 个 kernel 里大量卡在启动地板上的小 kernel。

### 1. 三段各做什么

代码：`python/sglang/srt/speculative/eagle_worker_v2.py` 的 `forward_batch_generation`（生产镜像 trace 里也是这个函数，结构相同），
`spec_stage_span("draft")` / `spec_stage_span("draft_extend")` 就是 trace 里的这两个注解；verify 的注解是 `step[VERIFY bs=N]`。

| 段 | 跑的模型 | 每请求 token 数 | 做什么 | TP8 C208 GPU 时间 | kernel 数 / 次 |
|---|---|---|---|---|---|
| draft | NextN（GLM-5.2 自带的 MTP 层，1 层 MoE） | 1，**跑 2 次** | 自回归地生成第 2、3 个草稿 token（第 1 个草稿 token 已在上一步的 draft_extend 里算好，所以 3 个草稿只需 2 次 forward） | 1.58 ms | 86–88 |
| verify | 目标模型，78 层 | 4 = 上一步确认的最后 1 个 token + 3 个草稿 | 一次 forward 算出 4 个位置的 logits；逐个比对草稿，接受匹配的前缀，再加 1 个目标模型自己的 token（bonus）；写目标模型的 KV | 41.9 ms | 约 2,080 |
| draft_extend | NextN，1 次 | 4（固定按 `num_draft_tokens` 宽度补齐） | 用 verify 输出的目标模型 hidden states 和确认后的 token，给草稿模型补写 KV，并算出**下一步**的第 1 个草稿 token | 1.71 ms | 52 |

- 每步每个请求产出 1–4 个 token。计分时用 `SGLANG_SIMULATE_ACC_LEN=2.99` 模拟接受（kit 的 `worker.sh`），平均约 3 个；所以每用户约 3 / 0.046 s ≈ 65 tok/s，
  与交互度 p50（70–86）同一量级。
- 草稿只占约 3.3 ms，是因为 NextN 只有 1 层（每卡约 4.6 GB 权重，目标模型约 74 GB）。
- 顺带一个小更正：`RESULT` 第 5 节估带宽下限时写"草稿 3 步与 draft_extend 共 4 次 NextN"，按代码和 kernel 数应是 **draft 2 次 + draft_extend 1 次 = 3 次**
  （`draft_forward` 循环 `speculative_num_steps` 次，但最后一次在 forward 之前就 `break`，只做选 token）。对下限的影响约 1 次 NextN 读权重，0.7–1 ms，不改变结论。

### 2. 一个 scheduler batch = 一步

- 每次 profile 采 40 步：TP8 rank 0 的 trace 里 `scheduler.run_batch`、`draft`、`draft_extend`、`step[VERIFY …]` 各 40 次
  （grep 计数是 80，因为 CPU 和 GPU 轨道各记一份注解）。
- TP8 步长中位 **46.2 ms = verify 41.9 + draft 1.6 + draft_extend 1.7 + 约 1 ms**（graph 之外的 eager kernel 和空闲，`kdeep_tp8_083302.txt`）。
- 看耗时要看 GPU 轨道：CPU 上的 span 只表示发射和等待（每段是一次 `hipGraphLaunch`，整张 graph 的 kernel 一次性提交），长度不等于 GPU 执行时间。
- DP attention 下 8 个 rank 各有一个 scheduler、各跑自己的 `run_batch`，但它们是**锁步**的：同一步里所有 rank 用同一个 graph batch（按最忙的 rank 补齐，
  例如各 rank 实际 `[10, 14, 12, 14, 20, 21, 10, 16]` 个请求，都按 22 跑），每层的 all-gather 又把它们同步一次。

### 3. 为什么有这么多 all-gather

decode 是 TP8 + DP8 attention（EP1）：

- **attention 部分是数据并行**：每个 rank 只算自己的请求，注意力权重每个 rank 各有完整的一份（所以 attention 侧 GEMM 在 TP4 / TP8 下都是约 6.3 ms，不随 TP 变小）。
- **MoE（以及前 3 层的 dense MLP）是张量并行**：每个专家的中间维 2048 切成 8 份，每个 rank 握着**每个专家**的 1/8（256 维）。

所以每层都是这样一个循环（以 graph bs 22 为例，每 rank 22 × 4 = 88 行）：

    attention（本 rank 的 88 行）
      → all-gather：把 8 个 rank 的 hidden 拼成 8 × 88 = 704 行（每个 rank 都要全部 token，因为它握着每个专家的一片）
      → MoE：704 行 × 本 rank 的 1/8 专家权重，得到 704 行的部分和
      → reduce-scatter：8 个 rank 的部分和相加，并把各自的 88 行发回
      → 下一层

- 计数（kdeep）：verify 每步 `allgather_vec` 78 次、`reduce_scatter_split_first_dim` 78 次，末尾 lm_head 附近还有 1–2 次小通信。
  draft / draft_extend 里的 NextN 也是一层 MoE，按同样的结构也各带一对（我没有单独核对 kernel 名）。
- 每次 all-gather 搬约 704 × 6144 × 2 B ≈ 8.7 MB，用时普通层 30 µs、indexer 层之后 52.7 µs；多出来的部分是在**等最慢的 rank**
  （indexer 的工作量与上下文长度成正比，各 rank 不同，见 KERNEL-PLAN 2.3 第 6 条）。
- 通信合计 5.6 ms/步（verify 的 13%），其中搬数据约 4.4 ms、等待约 1.2 ms。

### 4. verify 为什么要 42 ms

**按层拆**（kdeep 的层签名，TP8 C208）：

| 层类型 | 层数 | 每层 µs | ms/步 |
|---|---|---|---|
| 普通 MoE 层 | 57 | 约 470 | 26.9 |
| 带 indexer 的 MoE 层 | 18 | 约 720 | 12.9 |
| 前 3 层 dense（也带 indexer） | 3 | 约 580 | 1.7 |
| 末尾（lm_head 等） | | | 0.4 |
| 合计 | 78 | 平均约 540 | **41.9** |

所以"因为有 78 层"是对的：每层约 0.5 ms，78 层叠起来就是 42 ms。

**按 kernel 类拆**（`kcat_summary.txt`，TP8）：

| 类 | ms/步 | 占比 | 说明 |
|---|---|---|---|
| MoE（router、排序 / 量化、gemm1、gemm2、reduction） | 16.0 | 38% | 读 48.3 GB 专家权重，只到 3.5–4.7 TB/s |
| attention 侧投影 GEMM（fused_qkv_a、q_b、o_proj） | 6.3 | 15% | 读 23.5 GB，2.2–4.2 TB/s；DP attention 下每 rank 一整份 |
| MLA absorb BMM | 3.2 | 8% | 只读 1.15 GB，0.36 TB/s，效率很低 |
| sparse MLA | 4.8 | 11% | 随请求数、每行 top-2048 的 KV gather 增长 |
| indexer（21 层） | 4.6 | 11% | 对整个上下文打分，随上下文长度增长（AgentX 上下文 10 万–18 万 token） |
| 通信 | 5.6 | 13% | 第 3 小节 |
| norm 等小 kernel | 1.5 | 4% | 每次约 4.4 µs，基本是启动地板 |

**为什么比读权重的下限慢这么多**：verify 每步读约 73 GB 权重，按可达约 6.5 TB/s 算下限约 11 ms，实测 42 ms，差在：

1. **kernel 带宽效率**：MoE 只到 3.5–4.7 TB/s（TP4 用 KF 能到 6 TB/s 以上），absorb BMM 只到 0.36 TB/s。
2. **不是读权重的工作**：sparse MLA + indexer 约 9.4 ms，与上下文长度、请求数有关，和权重大小无关。
3. **通信**：156 次集合通信，5.6 ms。
4. **小 kernel**：约 2,080 个 kernel 里将近一半是 4–6 µs 的小 kernel（norm、top-k、排序、量化、indexer 前处理等），时间主要花在启动和收尾上，合计约 5 ms（与上表各类有重叠，是换个角度看）。

两个附带的点，有助于读 KERNEL-PLAN：

- **MoE 基本上每层都把全部专家读一遍**：704 行 × top-9 ≈ 6,300 个（token, 专家）对，分到 257 个专家上平均每个 24 个，几乎所有专家都被激活。
  所以在这个负载下，MoE 时间几乎不随 batch 变化，主要取决于读权重的效率。这也是 KERNEL-PLAN 把 TP8 MoE 列为第一项的原因。
- **TP8 的 verify（41.9 ms）并不比 TP4（40.7 ms）快**：TP8 每卡的 MoE 权重减半，但 (a) TP8 的 MoE 没走 KF、效率低；(b) attention 侧是数据并行，不随 TP 减少；
  (c) 两组负载不同（C208 对 C104）。KERNEL-PLAN 第二、三节就是在拆这个差距。

---

## 问 2（10:05）：读权重的带宽是怎么算的？是不是因为 decode 是 memory bound，就把 kernel 的完成时间当作读内存的时间？

### 简答

- **公式**：等效带宽（TB/s）= 这类 kernel 每步**必须读的权重字节** ÷ 它在 trace 里的 **GPU 耗时**（`kcat.py`，1 TB = 10¹² B）。
- **方向对，但含义要反过来理解**：不是"假设 kernel 时间全在读内存"，而是"因为 memory bound，**理想**耗时 ≈ 权重字节 ÷ 可达带宽（约 6.5 TB/s）"。
  于是 等效带宽 ÷ 6.5 = 理想耗时 ÷ 实际耗时，衡量的是**离读权重下限还有多远**。实际耗时里包含启动、计算、写回、同步等所有东西；
  等效带宽低，恰恰说明时间花在了读权重以外的地方，具体是什么要逐个 kernel 查。
- **只对"理想情况下由读权重决定"的 kernel 有意义**（MoE GEMM、稠密投影）。absorb BMM 的 0.36 TB/s 只说明它离下限很远，不说明它卡在带宽上；
  attention（sparse MLA、indexer）主要读 KV cache，不用这个指标。

### 1. 分子：每类 kernel 要读的权重（`kcat.py` 的 `weights()`，TP8）

| kernel | 权重形状（每 rank） | 类型 | 每次 MB | 每步次数 | 每步 GB |
|---|---|---|---|---|---|
| MoE gemm1 | 257 专家 × (2 × 256) × 6144 | MXFP4：每元素 0.5 B + 每 32 元素 1 B scale | 429.5 | 75 | 32.2 |
| MoE gemm2 | 257 × 6144 × 256 | MXFP4 | 214.7 | 75 | 16.1 |
| router | 256 × 6144 | BF16 | 3.1 | 75 | 0.24 |
| fused_qkv_a | 2624 × 6144 | BF16 | 32.2 | 78 | 2.5 |
| q_b_proj | 16384 × 2048 | BF16 | 67.1 | 78 | 5.2 |
| o_proj | 6144 × 16384 | BF16 | 201.3 | 78 | 15.7 |
| absorb BMM（W_kc + W_vc） | 64 头 × (192 × 512 + 512 × 256) | FP8 | 14.7 | 78 | 1.15 |

例：gemm1 每步 32.2 GB ÷ 6.90 ms = 4.67 TB/s。TP4 的 MoE 中间维是 512，MoE 两行的字节翻倍。

分子里的两个假设：
- **只算权重**：不算激活输入、输出、中间结果和 KV。多数 GEMM 里这些很小，但有例外（见第 4 小节的 gemm2）。
- **MoE 按 257 个专家全读**：M 704 时，一个专家一次都没被选中的概率是 (1 − 8/256)^704 ≈ 2 × 10⁻¹⁰，所以确实全读；TP4 的 M 约 288 时也约 10⁻⁴。
  但小 M 时不成立（M 32 时约 36% 的专家不会被读），那时这个公式会高估字节、高估带宽。

### 2. 分母：trace 里的 GPU 耗时

- 该类 kernel 每步的 GPU 时间之和，各 rank 取平均。包含 kernel 的整个生命周期：启动、爬坡、计算、写回、收尾。
- 前提是 trace 用 `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` 采集，否则 graph 内的逐 kernel 时间不准（`KERNEL-PROGRESS` 第二节做过校准）。
- **不同文档的分母不一样，数字就不一样**，对同一份 TP8 MoE 权重：
  - gemm1 / gemm2 各自算：4.67 / 3.49 TB/s（`kcat`，KERNEL-PLAN 3.1）；
  - 交接文档的"MoE GEMM 4.18 TB/s"：48.3 GB ÷ (gemm1 + gemm2) 的 11.5 ms，不含 reduction；
  - KERNEL-PLAN 2.4 的"TP8 MoE 部分 3.0 TB/s"：644 MB ÷ 整条 MoE 链 212 µs（含 router、排序、量化、reduction）。
- 文档里的"峰值百分比"是对 8 TB/s（MI355X 标称）算的：4.18 / 8 = 52%、5.81 / 8 = 73%。估"能省多少"时用的是可达约 6.5 TB/s，或 KF 实测的约 6 TB/s。

### 3. 为什么说这些 kernel 是 memory bound：两种下限比一下

计算下限 = FLOPs ÷ 峰值算力（MI355X 标称 dense：BF16 约 2.5、FP8 约 5、MXFP4 约 10 PFLOP/s）；访存下限 = 权重字节 ÷ 6.5 TB/s。
行数取 TP8 C208 的典型值（每 rank 88 行，MoE 704 行）：

| kernel | 计算下限 | 访存下限 | 实测（graph 内） |
|---|---|---|---|
| MoE gemm1（6,336 个 token-专家对） | 39.9 GFLOP → 4.0 µs | 429.5 MB → 66 µs | 91.7 µs |
| o_proj | 17.7 GFLOP → 7.1 µs | 201.3 MB → 31 µs | 42.5 + 5.1 µs |
| q_b_proj | 5.9 GFLOP → 2.4 µs | 67.1 MB → 10.3 µs | 18.8 µs |
| fused_qkv_a | 2.8 GFLOP → 1.1 µs | 32.2 MB → 5.0 µs | 14.8 µs |
| absorb BMM（两个合计） | 2.6 GFLOP → 0.5 µs | 14.7 MB → 2.3 µs | 39.7 µs |

- 每一行的访存下限都是计算下限的 4–17 倍，所以理想耗时由读权重决定，"memory bound"成立。经验规则：BF16 GEMM 每读 1 字节权重只做约 M 次运算，
  而 MI355X 要约 300 次以上才轮到算力成为瓶颈；MoE 按每个专家分到的 token 算（约 25 个），离得更远。
- 但实测都比访存下限慢 1.4–17 倍：这部分时间**不是在读权重**，下面举几个例子。

### 4. 等效带宽低时，时间花在哪里（指标本身说不出来，要逐个查）

- **MoE gemm2（reduce 版，3.5 TB/s）**：除了 214.7 MB 权重，还要把 6,336 × 6144 的 BF16 中间结果（约 78 MB）写出去，再由单独的 `moe_reduction`（13.4 µs）读回来相加。
  把中间结果算上，gemm2 的真实流量约 293 MB，约 4.8 TB/s，所以它并不太"慢"，而是这种结构多了一份流量。换成 atomic 版不写中间结果，
  但变成 9 路 BF16 atomic 加回，受写回限制（`KERNEL-PROGRESS` 3.2 节：3.2 TB/s）。
- **absorb BMM（0.36 TB/s）**：每层只读 14.7 MB，访存下限 2.3 µs，实测 40 µs。原因是 tile 划分让 256 个 CU 用不满（W_vc 只有 128 个 workgroup），
  再加上 RoPE / 拼接 / 写 KV cache 的附带工作和两次启动。只换 Triton 配置，W_vc 就从 16 µs 降到 9 µs（`KERNEL-PROGRESS` 第四节），和带宽无关。
- **小 kernel 的地板**：每个 kernel 约 4 µs 起步，所以读不到约 26 MB（4 µs × 6.5 TB/s）的 kernel 不可能跑到带宽上限。fused_qkv_a（32 MB，14.8 µs，2.2 TB/s）
  这种中等偏小的矩阵最吃亏。
- **o_proj**：hipBLASLt 用 8 路 split-K，再由 `PostGSU8` 归约，多一次启动（5.1 µs）。

### 5. 用这个指标时要注意的

1. **只适用于理想上由读权重决定的 kernel**。sparse MLA / indexer 读的是 KV cache 和 index-K cache，字节数随上下文长度变化，而且是按索引的不规则访问，
   所以 `kcat` 没给它们算 TB/s，而是用"每次调用 µs""每行 µs"来比。
2. **"可达约 6.5 TB/s"是经验值**：标称 8 TB/s，KF 在 TP4 MoE 上实测到 6.1–6.4 TB/s，说明 6.5 这个量级可达；这几份文档里没有用 stream 类基准单独测过。
3. **单算子测试要防缓存命中**：反复用同一份权重会命中 L2 / Infinity Cache（MI355X 256 MB），算出的 TB/s 会虚高。所以 `moe_bench.py` 轮换 4 组权重、
   `absorb_bench.py` 轮换 48 组，AKLR 每次刷 L2。graph 内每层权重不同、每步读约 74 GB，不存在复用。
4. **这是"按应读字节"的估计，不是实测流量**。要知道 kernel 实际读写了多少字节、显存带宽实际有多忙，要用硬件计数器（rocprofv3 的 `FETCH_SIZE` / `WRITE_SIZE`、
   `MemUnitBusy` 等）。文档里目前没有做。**需要执行**：如果你想核实某个 kernel（比如 gemm2、absorb BMM）的真实流量，可以在现有单算子 bench 上加一次计数器采集，单卡几分钟。
5. 步级的"带宽下限"（`RESULT` 第 5 节）也是同一个思路：每步要读的权重总字节 ÷ 6.5 TB/s（TP8 约 88 GB → 13.6 ms），再和步长拟合的截距（17.8 ms）比。
