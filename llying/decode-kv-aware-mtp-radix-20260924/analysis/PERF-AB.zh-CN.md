# decode radix cache 与 decode HiCache 的性能对比（AgentX C40）

日期：2026-09-28。数据于 2026-09-24 采集。正确性验证见 [GATE-RESULTS.zh-CN.md](GATE-RESULTS.zh-CN.md) 和 [DECODE-HICACHE.zh-CN.md](DECODE-HICACHE.zh-CN.md)。

## 结论

- **decode radix（B）比同镜像的基线（A）吞吐高 3.0%。** 收益来自 TTFT 这一侧：decode 复用前缀后，要从 prefill 传的 KV 少了，预分配排队也短了。TTFT 均值降低 3.9%，但 p50、p90 略差。ITL 基本不变。
- **之前"B 比 t2f 高 8.1%"的结果，大部分来自镜像差异**，不是 decode radix 的贡献。
- **在 B 的基础上再开 decode HiCache（C，write_through），吞吐下降 4.5%，ITL 升高 6.2%。** C40 下这是净负收益。
  - TTFT 侧有收益但很小：主机命中让等待传输的时间缩短约 345 ms/请求，但预分配等待增加约 195 ms/请求。
  - 主要损失在 decode 单步：负载相同时，每步多约 1.7 ms（约 +3.7%）。decode 变慢后，同时停留在 decode 的请求增加约 5%，batch 变大又让单步更慢，最终 ITL +6.2%。
- **每步多出的时间分为两部分。**
  - 约 1–1.5 ms 与 HiCache 流量无关：流量最低的时段也存在。
  - 约 0.5–1 ms 随写入、加载和淘汰量增加。
  - 写入带宽只有每卡约 110 MB/s，拷贝本身占不了多少 GPU。
  - py-spy 显示 HiCache 代码只占 decode 调度线程约 0.3% 的时间，调度线程约 89% 的时间在等 GPU。所以多出的时间落在 GPU 时间线上，不是 CPU 开销。随流量增长的部分可以用拷贝 kernel 抢资源、rank 间互相等待来解释；固定部分还没有找到机制，其中可能有一部分是单次运行的波动。
- **换写策略解决不了主要问题。** write_back 和 write_through_selective 只改变写主机的时机和写入量，影响不到与流量无关的那部分开销。write_back 还会把写入变成调度线程上的同步操作。分析见"三种写策略"一节。
- **prefill 首次编译 tilelang kernel 造成的卡顿，对结果影响可以忽略。** 采样期间受影响的请求不到 1%，TTFT 增加 0.3%–1%。
- **三组采样期间都没有出错。** A 组 3 个、C 组 5 个错误，都是 warmup 阶段（max_tokens=1）的 `InvalidInferenceResultError`。

## 测试条件

- **部署：** 135 跑 prefill，138 跑 decode，各用 GPU 2–5，TP4/DP4/DPA，Mooncake 传输，`PD_DP_RANK_AFFINITY=1`。
- **镜像：** `infera-sglang:v0519-yihou-0917-nextnfix-hicache-decrad`。三组 prefill 都开 HiCache（write_through，ratio 1.5）。
- **负载：** AgentX C40 full 模式。warmup 每条 lane 10 个请求（约 21 分钟，max_tokens=1），然后正式采样 3600 秒。MTP 使用模拟接受率 3.61，与 yihou 的 t2f 相同。
- **运行顺序：** 同一天依次运行，每组只跑一次。B 13:27–14:53，C 17:34–19:00，A 19:09–20:38（UTC）。
- **各组配置**（`scripts/bench-harness/results/decrad/config.decrad.<组>-perf.sh`）：

| 组 | decode 缓存 | 与上一组的差别 |
|---|---|---|
| A | ChunkCache（关闭 radix） | 同镜像基线 |
| B | radix（`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`，开启 KV 事件） | 开启 decode radix |
| C | radix + HiCache（ratio 1.5，write_through，`kernel` io，`page_first`） | 再开 decode HiCache。HiCache 参数通过 `DECODE_EXTRA_ARGS` 传入，AgentX 这一步额外传 `DECODE_HICACHE=1`，以通过启动前检查 |

## 端到端结果

数据：[evidence/perf/summary.csv](evidence/perf/summary.csv)。t2f 来自 yihou 的 T2 扫描，使用不带 nextnfix 的基础镜像。

| 指标 | A | B | C | t2f |
|---|---|---|---|---|
| 总吞吐（tok/s） | 166,229 | 171,279 | 163,539 | 158,390 |
| 每卡吞吐（tok/s） | 20,779 | 21,410 | 20,442 | 19,799 |
| TTFT 均值 / p50 / p90 / p95（s） | 6.64 / 3.73 / 12.19 / 23.23 | 6.38 / 4.01 / 12.79 / 22.99 | 6.36 / 4.02 / 12.70 / 22.30 | 7.41 / 4.17 / 14.80 / 26.89 |
| E2E 均值（s） | 19.04 | 18.76 | 19.72 | 21.17 |
| ITL 均值 / p95（ms） | 13.77 / 19.54 | 13.58 / 18.22 | 14.42 / 20.44 | 13.58 / 18.47 |
| 采样期完成请求数 | 4143 | 4197 | 4108 | 4025 |
| 平均输出长度（token） | 937.6 | 942.2 | 961.9 | 1040.9 |

| 对比 | 吞吐 | TTFT 均值 | E2E 均值 | ITL 均值 / p95 |
|---|---|---|---|---|
| B vs A（decode radix 的收益） | **+3.0%** | −3.9% | −1.5% | −1.4% / −6.8% |
| C vs B（decode HiCache 的收益） | **−4.5%** | −0.3% | +5.1% | +6.2% / +12.2% |
| C vs A | −1.6% | −4.2% | +3.6% | +4.7% / +4.6% |
| B vs t2f | +8.1% | −13.9% | −11.4% | 0.0% / −1.4% |

- **B 与 t2f 不能直接比较。** nextnfix 修改了 MTP draft 的计算方式。模拟接受率模式下输出的正是 draft 生成的 token，所以两边的输出长度不同：942 对 1041。同镜像的 A 组才是公平的基线。
- **TTFT 分位数有混合变化。** B 的 TTFT 均值和 p95 下降，但 p50、p90 上升。每组只跑一次，3% 左右的差距最好再重复一次确认。

## 引擎侧指标

路由命中数据：[evidence/perf/router-hits.txt](evidence/perf/router-hits.txt)，统计范围为采样期。decode 端各阶段耗时和计数：[evidence/perf/decode-engine-metrics.txt](evidence/perf/decode-engine-metrics.txt)。decode 端的数据取自 AgentX 前后两次 `/metrics` 快照的差值，**包含 warmup**。

| 指标 | A | B | C |
|---|---|---|---|
| prefill 命中：有命中的请求 / 按块复用 | 91.4% / 94.7% | 91.5% / 94.5% | 91.6% / 94.5% |
| decode 命中：有命中的请求 / 按块复用 | 0 / 0 | 89.0% / 86.4% | 89.2% / 88.0% |
| decode 预分配等待（`decode_bootstrap`，bootstrap 完成到 KV 分配完成），ms | 113 | 68 | 262 |
| decode 等传输（`decode_transferred`，含等待 prefill 计算），ms | 11,181 | 10,045 | 9,700 |
| decode 端 ITL（服务端统计），ms | 13.23 | 13.14 | 13.90 |
| 平均生成 token 数（服务端，含 warmup） | 851 | 855 | 873 |
| 显存淘汰 token 数 | 0 | 7218 万 | 6710 万 |
| 写主机 token 数 / 字节数 | — | — | 4083 万 / 2.28 TB |
| 从主机加载 token 数 / 字节数 / 次数 | — | — | 3506 万 / 1.96 TB / 171 次，平均每次 0.25 s |

- **decode radix 的收益来自 TTFT 侧。** B 比 A 等传输的时间少约 1.1 s/请求，预分配等待少约 45 ms/请求。
- **decode HiCache 在 TTFT 侧的净收益很小。** C 比 B 等传输少约 345 ms/请求，因为主机命中的部分不必再从 prefill 传。但预分配等待多了约 195 ms/请求，还需要再查它来自主机加载还是淘汰。
- **主机池几乎一直是满的。** 采样结束时，4 个 rank 合计已用 1328 万 token，总容量 1336 万 token。

## C 组为什么慢

脚本和输出列在文末"复现"一节。

### 1. 慢在 decode 的每一步

**客户端视角：** 流式输出中每个 chunk 对应一个 decode step（约 3.5 个 token，与接受长度一致）。chunk 间隔的统计见 [evidence/perf/chunk-latency.txt](evidence/perf/chunk-latency.txt)：

| 组 | p50（ms） | p90（ms） | 均值（ms） | 间隔 >50 ms 的时间占比 | 间隔 >100 ms 的时间占比 |
|---|---|---|---|---|---|
| A | 42.5 | 66.5 | 47.67 | 58.7% | 30.5% |
| B | 43.1 | 59.9 | 47.35 | 59.2% | 33.0% |
| C | 45.2 | 64.6 | 50.08 | 66.9% | 32.4% |

C 的均值比 B 高 5.8%，与 ITL 升高的幅度吻合。变化集中在分布主体（50–100 ms 这一段）。100 ms 以上的长卡顿占比，B 和 C 差不多。所以 C 是每一步都慢了几毫秒，而不是被偶发的长阻塞拖慢。

**负载相同时：** 从 decode 日志估算每个 rank 的单步耗时，即 running × 接受长度 ÷ 生成吞吐，再按该 rank 的 running 请求数和上下文长度分桶。完整表格见 [evidence/perf/decode-step-equal-load.txt](evidence/perf/decode-step-equal-load.txt)，节选如下：

| running / rank | 上下文 / rank | B p50（ms） | C p50（ms） | 差值 |
|---|---|---|---|---|
| 1–2 | 1.0–1.5M | 42.0 | 43.6 | +1.6 |
| 3–4 | 0.5–1.0M | 43.2 | 45.1 | +1.9 |
| 3–4 | 1.0–1.5M | 46.5 | 47.7 | +1.2 |
| 5–6 | 1.0–1.5M | 49.6 | 50.9 | +1.3 |
| 7–8 | 1.0–1.5M | 53.9 | 56.5 | +2.6 |

负载相同时，C 每步多 1–3 ms。此外，C 每个 rank 的平均 running 请求数是 3.82，B 是 3.64（+5%）。这是闭环负载下 decode 变慢造成的：请求在 decode 停留得更久，同时在跑的请求就更多。C 的平均输出长度也长了 2%。两者叠加，把"同负载 +3.7%"放大成了 ITL +5.8%。

同一张表中还有一个现象：running 为 1–2 时，B 比 A 每步慢 2–7 ms；running ≥ 3 时两者基本持平。说明 decode radix 本身在小 batch 下也有一点单步开销，但它在 TTFT 侧省下的时间更多。

### 2. 写入流量本身不大

C 的 AgentX 运行约 86 分钟，共写主机 2.28 TB，折合每卡约 110 MB/s；写入操作共 8262 次，约每秒 1.6 次。从主机加载共 171 次，每次约 11.5 GB，耗时约 0.25 s。拷贝在单独的流上异步进行，写入和加载的耗时累加起来约 157 GPU·秒，不到 4 张卡总时长的 1%，不足以让每一步都慢 3–4%。

### 3. 与流量无关的部分和随流量增长的部分

用 AgentX 每秒抓取的服务端指标，得到每个 rank 每秒的单步耗时，以及同一秒的写入、加载和淘汰量。然后以 B 在同一负载分桶内的中位数为基线，计算残差。数据见 [evidence/perf/hicache-step-overhead.txt](evidence/perf/hicache-step-overhead.txt)。

- **找不到完全没有流量的时段。** 3640 秒里，所有 rank 前后 2 秒内都没有 HiCache 流量的只有 15 秒。所以改为按流量大小分档比较。
- **需要排除请求进出的影响。** 请求完成和到达越频繁，单步本来就越慢，B 也是如此。这里用淘汰量作为请求进出频繁程度的代理指标。

| 分档（整机 ±1 s 内的量） | B 相对自身基线 | C 相对 B 基线（按淘汰量） | C 相对 B 基线（按写入 + 加载量） |
|---|---|---|---|
| Q1（最少） | −0.50 ms | +1.03 ms | +0.76 ms |
| Q2 | −0.72 ms | +1.25 ms | +1.65 ms |
| Q3 | +0.88 ms | +1.84 ms | +1.98 ms |
| Q4（最多） | +0.47 ms | +2.51 ms | +2.64 ms |

表中数值为残差中位数。

- 在流量最低的一档，C 已经比 B 多 1–1.5 ms：这是开启 decode HiCache 后就有的固定开销。
- 从 Q1 到 Q4，C 的残差比 B 多涨约 1 ms：这部分随流量增长。
- Q4 的残差均值，C 为 +15～18 ms，B 为 +11 ms，说明两组都有长卡顿，C 稍多。

### 4. 最初的假设：CPU 侧开销

- **每轮调度的固定工作。** 开启 decode HiCache 后，调度器每一轮都会调用 `UnifiedRadixCache.check_hicache_events()`（注释："Called per scheduler step"）：先 `flush_pending_backups()`，再在 tp group（4 个 DP rank）上做一次 CPU all_reduce，汇总各 rank 已完成的写入和加载数（`_sync_hicache_ready_counts`），然后处理确认。无论有没有流量、用哪种写策略，这一步每轮都会执行。
- **加载完成检查。** 恢复流程中，`is_load_back_event_done()` 在本地加载完成时会调用 `loading_check()`，而后者内含 all_reduce。这部分只在有请求从主机加载时才会发生。
- **写入时调度线程上的 CPU 工作。** write_through 每次插入都要分配主机内存并发起拷贝。主机池满了以后，还要先在主机上淘汰旧节点。这些都在调度线程上执行。
- **radix 树变大。** 只存在于主机上的节点也留在树里，所以匹配、淘汰和更新叶子状态时要遍历更多节点。

### 5. py-spy 定位结果（2026-09-28）：CPU 侧的假设都不成立

实验：C-diag（配置同 C，采样 900 秒），在采样开始后第 3 分钟和第 9 分钟，对 4 个 decode 调度进程各采样 90 秒。py-spy 参数为 `--idle --nonblocking`，100 Hz，共 72k 个主线程样本。汇总见 [evidence/diag/c-diag-pyspy-summary.txt](evidence/diag/c-diag-pyspy-summary.txt)，原始数据在 `evidence/diag/c-diag-pyspy/`。

| 调度主线程的时间去向 | 占比（8 个文件的范围） |
|---|---|
| `run_batch` → `resolve_seq_lens_cpu` → `synchronize`（等 GPU） | 67–75% |
| `_resolve_attention_variant` → `DsaGraphVariants.select`（`seq_lens.max().item()`，也在等 GPU） | 12–14% |
| DP attention 的 `all_gather_into_tensor` | 3–10% |
| 全部 HiCache 相关代码：`check_hicache_events`、`_sync_hicache_ready_counts`、`writing_check`、`loading_check`、`init_load_back`、`_process_hicache_local_restores`、`evict_host` | 0.07–0.49%，合计约 0.3% |

- **调度线程约 89% 的时间在等 GPU 或集合通信。** 真正忙于 CPU 计算的时间只有约 11%。
- **HiCache 在调度线程上的开销，按每步 45–50 ms 算，只有约 0.15 ms。** 这远小于要找的 1–1.5 ms，所以第 4 节列出的 CPU 侧来源都不是主因。
- **因此，多出的时间落在 GPU 时间线上。** overlap 调度下，调度线程在 `resolve_seq_lens_cpu` 里等 forward stream 完成，所以单步耗时约等于 GPU 时间线的长度。这段时间包括 kernel 计算，也包括 GPU 的等待：跨 stream 的事件等待，以及 DP attention 和 MoE 中各 rank 之间的集合通信等待。**"GPU 算得变慢"只是其中一种可能**，目前的数据还分不开。
- **有机制可以解释的部分（随流量增长）：**
  - **拷贝 kernel 与 decode kernel 抢资源。** `kernel` io 后端配 `page_first` 布局时，写主机和从主机加载都由 GPU kernel 完成（`transfer_kv_all_layer_mla_lf_pf`、`transfer_kv_per_layer_mla_pf_lf`），不走 DMA 引擎。它们跑在单独的 stream 上，与 decode 并发，会占用 CU 和 HBM 带宽。
  - **一个 rank 慢，所有 rank 都要等。** HiCache 的拷贝是按 rank 发生的，而 DP attention 和 MoE 要求各 rank 每一层同步。
  - **这些机制的上限有限。** 按耗时累加，拷贝只活跃了约 157 GPU·秒，不到总时长的 1%。即使算上跨 rank 等待的放大效应，也只能解释随流量增长的那部分，以及 Q4 的长尾，解释不了约 1–1.5 ms 的固定部分。
- **固定部分目前没有找到机制。** 候选有两个：
  - **KV 页分布更分散。** 从主机加载回来的 KV 落在任意空闲页上，淘汰和写入也更频繁，同一请求的页因此更分散，DSA indexer 和稀疏注意力的逐 token gather 可能遇到更多 TLB miss。这一点还是推测。
  - **单次运行的波动。** 每组只跑了一次，而且运行时段不同，单步耗时的运行间波动从来没有测过。同负载下，B 在 running 为 1–2 时比 A 慢 2–7 ms，而 decode radix 并不改变 forward 用的 kernel。这说明单步耗时上存在这个量级的差异来源，可能是页分布，也可能就是波动。
- **要区分这些可能，需要**在 GPU 层面对比 B 和 C（用 torch profiler 比较每步的 kernel 时间线：是同样的 kernel 整体变慢，还是多了集合通信等待或并发的拷贝 kernel），同时重复跑 B，估计运行间的波动。
- **一个与 HiCache 无关的发现：** EAGLE verify 路径上 `forward_batch.seq_lens_cpu` 为空，`DsaGraphVariants.select` 每步都要走回退分支 `seq_lens.max().item()`，做一次 D2H 同步，占调度线程 12–14% 的时间。B 组应该也有这个问题；如果能从 CPU 侧的镜像值拿到 max_kv_len，就能省掉这次同步。
- C-diag 的端到端数字（吞吐 140k tok/s，ITL 15.07 ms）不能和 C 组直接比较：采样只有 900 秒，而且紧跟在 warmup 之后，同时还开着 py-spy。

## 三种写策略

以下行为依据镜像中 SGLang 实际走的代码路径。decode 日志显示 `impl=UnifiedRadixCache hicache_attached=True`，对应 `mem_cache/unified_radix_cache.py` 和 `mem_cache/unified_cache/unified_tree_core.py`（`_inc_hit_count_and_check`、`evict` 末尾的 `writing_check(write_back=True)`）。旧的 `hiradix_cache.py` 中逻辑相同，但这次运行没有使用它。

| 策略 | 什么时候写主机 | 是否阻塞调度 | 主机上存什么 |
|---|---|---|---|
| write_through（C 组所用） | 每次插入 radix 树时立即写（阈值 1） | 否，在单独的拷贝流上异步执行 | 所有插入过的前缀，与显存中的内容重复 |
| write_through_selective | 同一节点第二次被插入或命中时才写（阈值 2） | 否，异步 | 只存被复用过的前缀 |
| write_back | 插入时不写。显存淘汰时，把未备份的块写回主机 | 是：每次淘汰结束时调用 `writing_check(write_back=True)`（注释写明 "Blocking"），在调度线程上逐个 `synchronize` 拷贝完成事件 | 只存被淘汰的块，与显存中的内容不重复 |

结合上面的分析，对 C40 下的预期如下：

- **write_back：** 插入时不再写入。但采样期显存淘汰了 5770 万 token，比 write_through 写入的 2810 万还多。除了从主机加载回来、本来就有副本的部分，其余都要在淘汰时同步写回。淘汰发生在预分配路径上，所以预分配等待（已经多了约 195 ms）会进一步变长，ITL 也会出现尖峰。固定开销不变。**预期比 C 差。**
- **write_through_selective：** 写入仍是异步的，写入量会减少。但 AgentX 的前缀复用率高（decode 端 88% 的块被复用），能省掉的主要是从未被复用的尾部。另外，一个节点在第二次命中之前如果被淘汰，就会直接丢弃，主机命中因此变少。它只能减少随流量增长的那部分开销中的一部分，同时削减 TTFT 侧的收益。**预期介于 B 和 C 之间，很难超过 B。**
- **两种策略都影响不到与流量无关的约 1–1.5 ms。**
- **即使开销全部消除，C40 下 decode HiCache 的收益上限也不高。** TTFT 侧每请求净省约 150 ms，TTFT 均值为 6.4 s，所以整体大致与 B 持平或略好。decode HiCache 可能要在并发更高、上下文更长、decode 显存更紧张时才有意义；上游 [#38292](https://github.com/sgl-project/sglang/pull/38292) 的 L2-Only 方案正是针对这种场景。

## prefill tilelang 编译卡顿的影响

prefill 第一次遇到某个 query token 档位时，要现场编译 tilelang 稀疏注意力 kernel，编译期间整个 prefill 停约 11 秒。原因分析见 [PREFILL-JIT-STALL.zh-CN.md](PREFILL-JIT-STALL.zh-CN.md)。

| 组 | 采样期编译次数 | 受影响的请求 | 增加的 TTFT（总和 / 占 TTFT 总和） |
|---|---|---|---|
| A | 2 | 22（0.53%） | 176.8 s / 0.64% |
| B | 1 | 14（0.33%） | 72.9 s / 0.27% |
| C | 2 | 36（0.88%） | 259.9 s / 0.99% |

编译事件对三组的影响都在 1% 以内，不改变上面的结论。输出文件为 `evidence/perf/<组>-perf-compile-impact.txt`。

## 收尾决定（2026-10-06）

**decode 只开 radix，不开 HiCache。这条排查到此结束。**

- **不追根因的理由：** C40 下即使开销全部消除，C 也只比 B 快约 0.5–1%。TTFT 侧的净收益只有约 150 ms/请求，decode 端的块复用率只从 86.4% 提高到 88.0%。decode HiCache 只有在 decode 显存成为瓶颈时才可能有明显收益。
- **C56 判断实验没有完成，不作为结论依据。** 原计划在 C56 下用 900 秒采样对比 B 和 C（`config.decrad.{b,c}-diag.sh`，`CONC=56`，输出在 `evidence/c56/`），以检验高并发下 decode HiCache 是否有收益。
  - **B 组两次都没跑成：** 第一次在 warmup 中因 `/home` 共享卷写满（`Disk quota exceeded`）中断，产物在 `evidence/c56/b-diag-attempt1-quota-failed/`；重跑排在登录节点的 tmux 里，随会话终止而丢失。
  - **只有 C 组的数据**（[evidence/c56/summary.csv](evidence/c56/summary.csv)、[evidence/c56/router-hits.txt](evidence/c56/router-hits.txt)）：采样期完成 666 个请求，吞吐 85,356 tok/s，TTFT 均值 50.7 s，ITL 14.05 ms，decode 端按块复用 58.0%（C40 时为 88.0%）。没有同条件下的 B 组，这组数字不能说明 HiCache 的好坏。
  - **10-06 用户决定不再补跑。**
- **upstream：** 不自行提交，持续跟踪 [#40857](https://github.com/sgl-project/sglang/pull/40857) 和 [#38292](https://github.com/sgl-project/sglang/pull/38292)。
- **如果以后要再评估：**
  - B 与 A 之间约 3% 的差距，需要每组再重复一次确认。
  - decode HiCache 的固定开销，需要在 GPU 层面（torch profiler）对比 B 和 C 才能定位。

## 复现

在仓库根目录下运行。`E=llying/decode-kv-aware-mtp-radix-20260924/analysis/evidence/perf`，`S=llying/decode-kv-aware-mtp-radix-20260924/scripts`。

```bash
python3 $S/perf_summary.py $E yihou/glm52-agentx-t2sweep-simacc.packup_20260920/results/t2-summary.csv > $E/summary.csv
python3 $S/router_hits.py $E a-perf b-perf c-perf
python3 $S/prom_decode_summary.py $E a-perf b-perf c-perf
python3 $S/chunk_latency.py $E/{a,b,c}-perf-agentx-c40/aiperf_artifacts/profile_export.jsonl
python3 $S/decode_step_equal_load.py \
    19:33:25 20:34:17 $E/a-perf-launch/server-logs/decode-0.log \
    13:49:11 14:51:12 $E/b-perf-launch/server-logs/decode-0.log \
    17:55:29 18:56:11 $E/c-perf-launch/server-logs/decode-0.log
# server_metrics_export.json.gz 解压后约 1.4 GB，在计算节点上解析
for a in a b c; do python3 $S/server_metrics_extract.py \
    $E/$a-perf-agentx-c40/aiperf_artifacts/server_metrics_export.json.gz $a-perf $E/server-metrics-$a-perf.csv; done
python3 $S/hicache_step_overhead.py $E/server-metrics-b-perf.csv $E/server-metrics-c-perf.csv
python3 $S/compile_stall_impact.py $E/<组>-perf-agentx-c40/aiperf_artifacts/profile_export.jsonl \
    $E/<组>-perf-launch/server-logs/prefill-0.log
```

性能组的运行：`scripts/run_perf.sh a-perf b-perf c-perf`，需要 135 和 138 的 GPU 2–5 空闲。
