# DCP4 decode 跑不稳、跑不赢的原因，与 ATOM 的实现对比，以及改进方案（10-23）

背景：我们在 4+4（138 PP4 prefill → 140 DCP4 decode，跨节点）上换 DCP4 decode，正确性通过，但 C80 两次在计分约 6 分钟时 decode 崩溃
（`../2p1d-v0522-20261020/README.zh-CN.md` "4+4 的 decode 由 DPA4 换成 DCP4"一节）。本报告综合三份材料：同事 10-23 的两份实验包、
ATOM（InferenceX PR #4045 的 DCP4 decode）源码、我们 overlay（同事的 sglang `93034eba91`）源码。源码结论均已抽查核对，路径行号如下引用：
ATOM 为 `ROCm/ATOM` @ `2c948339`（与 PR 镜像 `nightly_202610131249` 对应，本地 `ref/ATOM`）；sglang 为 `2p1d-kit/v7-dcp4/overlay/sglang/`。
本报告只做分析，没有上机。

## 0. 结论

1. **我们的崩溃有两层原因。** 直接原因是 PyTorch 缓存分配器的预留在负载下多涨 15–17 GiB（活跃张量只涨 0.8 GiB），把显存吃光后 HIP 运行时自己的
   分配失败（同事 gmu 实验包实测）。根源是 decode 在 graph 外的 eager 路径按 `req_to_token` 全宽（约 105 万列）一次性申请 page table 和 indexer
   logits，没有预算也不分块，形状随 batch 变化，释放后留在缓存里。
2. **"突发预分配不节流"不是 DCP 的 bug，是 sglang PD 的 `early` 分配策略**：握手时就按完整输入长度预留 decode KV，prefill 排队、计算期间一直占着。
   sglang 有晚分配策略 `prefill_complete`，但 DCP decode 和 PP prefill 都被禁用。ATOM 的路由器等 prefill 算完才把请求发给 decode，decode 那时才分配。
3. **DCP4 容量不如 DPA4，是因为 index-K 在每个 DCP rank 上存完整一份**（约 21 GB/卡）。ATOM 的 index-K 和 MLA KV 一样按序列切分，decode 的 indexer
   只在本 rank 的分片上打分、局部 top-2048、再合并。sglang 里把 index-K 改成切分是可行的（第 4 节）。
4. **同事的跨节点传输修复（prefill 侧 DCP 打包 78 ms → 0.6 ms、传输队列按请求哈希分片）对我们直接适用**，我们 C80 时 47–52 个等传输的请求很可能
   部分来自它。但修完后同一套环境里 DCP4 仍比 DPA4 低 4.3%（C56、KV 不紧张，DCP 每步慢约 10%）。
5. **改进分三步**：A 先稳定（移植已有补丁 + 留在 graph 内 + 拆分 indexer），拿到第一个有效的 DCP4 C80；B 提容量（index-K 切分 + 草稿层 KV 计数修正，
   预计 0.92 下 4 卡约 13.3M token，比 DPA4 多约 11%）；C 改分配时机。**C 的晚分配我们已实测过、收益约为零**（10-23 04:40 更正，见第 5 节 C），
   只保留准入节流这一小项。

## 1. 我们两次崩溃的经过（证据：`../2p1d-v0522-20261020/results/dcp4-c80*/incident-decode-oom.txt`）

| | 显存 0.92（10-22） | 显存 0.90（10-23） |
|---|---|---|
| KV 容量 / 捕获后剩余 | 11.28M / 5.42 GB | 10.86M / 11.02 GB |
| 突发 | 16:02:45 → 52 的 7 秒内 KV 0.19 → 0.96（等传输 2 → 47） | 03:01:52 → 56 的 4 秒内 KV 0.16 → 0.98（等传输 0 → 52） |
| 崩溃 | 16:03:15 TP0 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | 03:02:37 TP1 同样报错，"Available Free mem: 0 MB" |
| 崩溃时在跑 | 31（小于 graph 上限 128） | 37 |

两次都在计分约 6 分钟时、突发把 KV 一次占满之后崩溃；余量翻倍没用，说明运行期占用会继续涨。

## 2. 同事 10-23 的两份实验包

### 2.1 `yihou/glm52.dcp4-decode-gmu.packup_20261023`（单机，显存比例 0.85 以上为何崩、如何修）

- 定位：给 decode 加显存打印（每条 Decode batch 行打 `mem_get_info` 与 torch allocated / reserved）。0.88 未修复时，计分期间 allocated +0.8 GiB、
  reserved +15.7 GiB，设备剩余 16.36 → 0.92 GiB，2 秒后 abort。`expandable_segments`、`garbage_collection_threshold` 都无效；只在 TP0 释放缓存时 TP0
  恢复、其余 rank 照样崩；0.92 下 KV 只用 17% 时也崩过（与 KV 是否满无关）。
- 修复：每个 decode rank、每 8 次事件循环，设备剩余 < 6 GiB 时 `torch.cuda.empty_cache()`（`patches/03-ec2-allrank.decode.diff`，改 `decode.py` 两个事件循环，
  各 4 行；基于他们的本地提交 `ab7a898b77`，移植到 `93034eba91` 只需手工加这两处）。
- 结果（单机、C80、1800 s、各一次）：0.85 未修复 25,941 / 25,433；**0.88 修复后 27,685**；0.92 修复后能跑完但只有 22,540（原因未测）。
- 评价：根因证据扎实，但修复是治标——没有找出是哪些形状的分配造成碎片，也没测 `empty_cache` 的代价（0.92 变慢可能就是这个）。
- 他们的代码分析还找到：custom AR 在 DCP 下每卡占 14 GiB（我们关着）；index-K ×4 约 17 GiB；草稿层 KV 被多算 4 倍（3.57 GiB）。

### 2.2 `yihou/glm52.dcpx-pack-fix-best.packup_20261023`（跨节点 135 → 136，传输瓶颈）

- 定位：在传输路径加只写日志的计时，DCP 下 prefill 发送前在 GPU 上打包 KV（`try_pack_dcp_src`）每次 78 ms、占发送时间 91%，RDMA 本身不慢；
  同一 kernel 空闲 GPU 上 0.66 ms，多出来的是排队。
- 修复 1：打包的元数据缓存在 GPU 上（原来每次可分页内存同步 H2D），打包用高优先级流、不等默认流；每次 78 → 0.6 ms。
- 修复 2：Mooncake 传输队列按"目的端口之和"分片，DCP decode 只有一个 DP 组时这个值不变，4 个 worker 只用 1 个；改为按 `bootstrap_room` 哈希分片
  （与上游 sglang #42163 同思路）。
- 结果（C56、3600 s、各一次）：KV 传输 p50 12.3 → 0.93 s，DCP 19,997 → 24,753；同环境最好 **DPA4 26,321（显存 0.95，12.2M）vs DCP4 25,197（0.85，
  8.87M）**，TPOT 16.7 vs 18.4 ms。
- 适用性：补丁文件基于 `93034eba91`，与我们的 overlay 相同，可直接用（`patches/conn.yihou-dcpshard2.py`、`patches/dcp_pack.yihou-nowait2.py`）。
- 风险（同事自己指出）：PP prefill（不开 overlap）下"不等默认流"可能让前缀缓存部分在计算完成前发出。我们是 PP4 prefill，**打上后必须重做正确性验证**。

## 3. ATOM 与我们 overlay 的逐项对比

| 方面 | ATOM（`2c948339`） | 我们的 overlay（sglang `93034eba91`） |
|---|---|---|
| MLA KV 布局 | 按位置轮转，`interleave=1`；一个块表项是 16 × 4 = 64 个逻辑 token 的"虚拟块"，各 rank 存自己的 16 个（`atom/distributed/dcp_layout.py:12-46`，`aiter_mla.py:2515-2531`） | 按位置条带化，同样不复制 |
| **index-K** | **按序列切分**，与 MLA 在同一个分页池、同一套 slot 映射（`deepseek_v2.py:1441-1448` 注释："index cache 只存本 rank 轮转分到的 1/W"） | **每个 rank 一份完整副本**：池子计算 `indexer_cell_size *= dcp_size`（`model_executor/pool_configurator.py:401-402`），分配跨整个逻辑空间（`mem_cache/memory_pool.py:5346-5352`）；每槽多约 7.8 KB，约 21 GB/卡 |
| decode indexer | 本 rank 分片打分 → 局部 top-2048 → 只 all-gather 分数 → 合并，结果精确（除并列）；logits 宽度固定 `max_model_len/W`，graph 内静态（`model_ops/dcp_ops.py:1124-1307`） | 默认每个 rank 对完整 index-K 打分；拆分 indexer（默认关）只在本 rank 拥有的页上打分再合并（`layers/attention/dsa/dsa_dcp_split_indexer.py`），logits 宽度降到 1/4 |
| indexer 显存预算 | prefill 路径按 `ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB`（默认 2048）按行分块，并强制 aiter 2 GiB 硬上限（`model_ops/sparse_indexer_chunk.py:46-57`）；decode 由 graph 静态缓冲决定，不随上下文增长 | 预算只用于 prefill 的 ragged 路径；decode / 验证 / 草稿 extend 的 `aiter_paged_mqa_logits` 每次 `torch.empty((rows, max_seq_len), fp32)`（`kernels/ops/attention/dsa/paged_mqa_logits.py:110-114`），**eager 时宽度取 `req_to_token` 全宽**（`layers/attention/dsa_backend.py:1143`） |
| **decode 何时分配 KV** | **prefill 算完后**：路由器 await prefill 响应，再用 `kv_transfer_params` 构造 decode 请求（`atom/mesh/src/routers/http_pd_router.rs:1087-1147`） | **握手时**（`early`）；`prefill_complete` 策略存在，但 decode 开 DCP 或 prefill 开 PP 都被禁止（`arg_groups/pd_disaggregation_hook.py:205`、`disaggregation/common/conn.py:2329-2332`） |
| 准入限制 | 在跑 + 在途 ≤ `max_num_seqs`；块不够时先进先出阻塞（`model_engine/scheduler.py:1992-2012`）；无增长预留 | 可用 KV − 每个活跃请求 512 token 的预留；**每轮不限请求数和 token 数**（`disaggregation/decode.py:488-497`、`1313-1605`、`2072-2074`） |
| KV 用满时 | 抢占最新的在跑请求，在 decode 上本地重算（`scheduler.py:2342-2398`） | 回退（retract），DCP 下备份只能走 `cpu_tensor`，同步拷到主机，840K 请求约 11 GB/rank（`mem_cache/kv_cache_builder.py:214-227`） |
| 传输 | 每个 token 的 MLA 和 index-K 只发给拥有它的 rank；decode 侧每 rank 256 MiB 落地池（`kv_transfer/disaggregation/sharded_transfer.py`、`mooncake/mla_landing.py`） | MLA 条带化发送；index-K 发给每个 rank（`disaggregation/decode.py:1672-1682`）；prefill 侧打包（同事已修慢的问题） |
| 4 卡 KV 容量 | 约 196K 块 × 64 ≈ 12.5M token（显存 0.94，开 QREP；`recipes/Agentic-GLM-5.2.md:74-81`） | 0.92 时 11.28M；DPA4 为 12.0M |

## 4. 我们 overlay 的问题清单（按影响排序）

1. **eager 路径的全宽分配**（崩溃的根源）：graph 内（bs ≤ 128）每步新分配很少，但 eager 步骤按约 105 万列申请 `[6·bs, W]` 的 int32 page table 和 fp32
   logits（bs=40 时各约 1 GB，bs=128 时各约 3.2 GB），加上草稿每次 graph replay 都新分配 `[bs, W]` int32
   （`layers/attention/dsa/dsa_backend_mtp_precompute.py:123-132`）。形状随 bs 变化，释放后留在 PyTorch 缓存里，就是同事测到的"预留多涨 15–17 GiB"。
   进入 eager 的条件：bs > 128（我们 `--max-running-requests 256` 允许）、某个新请求没有 MTP seed（一个就会让整个 batch 的草稿走 eager，
   `speculative/eagle_info.py:257-270`）。静态部分还有约 8.6 GB 的宽 page table 和 3.2 GB 的 graph 内 logits。
2. **index-K 复制**：每卡约 21 GB，是 DCP4 容量不如 DPA4 的原因。
3. **`early` 分配 + 不节流**：突发一次把池子占满，在 prefill 排队的请求白占 decode KV；DCP 的共享池让一次突发能占得更多。
4. **草稿层 KV 多算 4 倍**（ROCm 上草稿 MLA 已按 DCP 切分，但池子计算仍 `draft_kv_size *= dcp_size`，`pool_configurator.py:303`），约损失 4% 槽位。
5. **跨节点传输慢**（prefill 侧打包、队列分片），同事已修。

## 5. 改进方案

### A. 先稳定，拿到第一个有效的 DCP4 C80（低风险，可直接做）

| 改动 | 依据 | 做法 |
|---|---|---|
| 所有 decode rank 在显存不足时 `empty_cache` | 同事实测有效（0.88 稳定） | 把 `03-ec2-allrank.decode.diff` 的两处 4 行移植到我们 overlay 的 `decode.py` |
| 传输修复（打包 + 哈希分片） | 同事实测传输 p50 12.3 → 0.93 s | 挂同事的 `dcp_pack.yihou-nowait2.py`、`conn.yihou-dcpshard2.py`（基于同一 `93034eba91`） |
| `--max-running-requests 128`（= graph 上限） | 避免 bs > 128 进入 eager 全宽路径 | worker.sh 一行；C80 时 decode 在跑最多约 80，不受影响 |
| 拆分 indexer（`SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1` + `_TRITON=1`） | indexer logits 宽度降到 1/4（graph 内与 eager 都是）；同事长上下文实测每步省 3–9 ms | 两个环境变量；与原计划的 A/B 合并 |

验证：因为打包修复在 PP prefill 下有顺序风险，先做 smoke（含 66k 长前缀）+ GSM8K + 真实接受长度；通过后 C80（显存 0.90 起，稳定后再试 0.92），
与 DPA4 C80（37.3k @ 44.3）对比。远端 guard 已就位，崩溃会自动停掉、不空转。预期：不再崩溃，等传输数下降；吞吐能否超过 DPA4 仍取决于容量（B）。

### B. 提容量（要改代码，预计 1–2 天）

| 改动 | 预期 | 改哪里（第 3、4 节的引用） |
|---|---|---|
| index-K 按序列切分（仿 ATOM） | 每槽 52.2 → 约 44.4 KB，0.92 时 4 卡约 13.3M token（比 DPA4 多约 11%，比 ATOM 的 12.5M 也多） | 池子计算去掉 `*= dcp_size`（`pool_configurator.py:283、401-402`）；分配改为本地页（`memory_pool.py:5346-5352`），写入时只写拥有的 token；拆分 indexer 读本地页（`dsa_dcp_split_indexer.py:18-20`），禁用读完整 index-K 的路径；PD 传输每个 rank 只发自己的页（`decode.py:1672-1682`，仿 `common/conn.py:550-582`）；retract 备份按本地页 |
| 草稿层 KV 计数修正 | 约 +4% 槽位 | `pool_configurator.py:303-304` 在 ROCm 草稿已切分时不再乘 `dcp_size` |

前提：拆分 indexer 成为 DCP decode 的唯一 index-K 读取方（ATOM 的做法）。验证要覆盖 smoke、长上下文 passkey、GSM8K、接受长度。

### C. 改分配时机

- **晚分配（10-23 04:40 更正：已实测，收益约为零，不作为方向）**：decode 等 prefill 算完才分配 KV。初稿把它写成"比 DCP 本身更大的杠杆"，
  忽略了我们之前的实测：
  - AUS 2P1D C192（10-20，E9 `a1-pcwb`，`../2p1d-aus-v0521-20261019/RESULT.zh-CN.md` §2.10–2.12）：KV 释放如设计（每 rank 等传输 7.6 → 2.2、
    KV>90% 的时间 62% → 10%、在跑 +13%），但**整段 KV 传输进入关键路径**（early 时 chunked prefill 边算边传，晚分配要算完、decode 分好后才传整段
    6.7 GB，传输 2.6 → 4.7 s 全部加到 TTFT），省下的约 4.8 s 被抵消大半。实测 32,339（early C192 33,698、early C176 35,377）；按模型扣除当时的
    "decode 冷启动"后估约 34,100，≈ early C192 +1%、early C176 −4%。那次的"冷启动"后来查明是 prefill 没开 MTP（§2.13），扣除后的估算本就按热态，
    结论不变。之后 v0.5.22 配置明确保留 early（`../2p1d-v0522-20261020/README.zh-CN.md` 配置表）。
  - 3P1D C192（10-15，`t8-pcwb`）：HiCache 一致性检查未过（检查本身有设计缺陷，未能定性），随后节点不可用，没有吞吐数据。
  - 唯一可能改变结论的因素是传输速度：AUS 那次单请求约 1.4 GB/s；同事的跨节点打包修复后 DCP 传输 p50 0.93 s。传输快很多时晚分配的代价会变小——
    推测，未实测。ATOM 采用晚分配，也同样承担整段传输在关键路径上的代价。
- **准入节流（较小的改动）**：每轮预分配设请求数 / token 数上限和池子水位，预留量随草稿 token 数放大（`decode.py:1460-1605`、`2072-2074`，
  参数放在 `num_reserved_decode_tokens` 旁边）。能削平突发，但不减少空占。

### D. 后续性能项

eager 宽度按调度器给出的最大序列长度收窄（`dsa_backend.py:1143`，需保证各 rank 一致）；草稿 precompute 复用缓冲；`KEEP_KV_B_BF16`（多占约 5 GB，
容量受限时不宜开）；custom AR（DCP 下每卡 14 GiB，保持关闭）。

## 6. 建议的顺序

1. A（约半天）：拿到第一个有效的 DCP4 C80，和 DPA4 C80 正面比较，确认传输、稳定性问题都已排除。
2. 视 A 的结果决定 B：若 DCP4 C80 的瓶颈确认在 KV 容量（KV 用满、等预分配多），做 index-K 切分。
3. C 只保留准入节流（削平突发）；晚分配已实测收益约为零，除非传输明显变快，不再作为方向。
