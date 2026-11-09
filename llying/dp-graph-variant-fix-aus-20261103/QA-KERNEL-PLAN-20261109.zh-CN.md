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
