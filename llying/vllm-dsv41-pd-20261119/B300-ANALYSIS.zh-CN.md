# B300 dynamo + vLLM disagg（DSv4.1-Flash FP4，AgentX）为什么吞吐高：拆解与可移植项

对照：InferenceX `configs/nvidia-master.yaml` 的 `dsv41flash-fp4-b300-dynamo-vllm-agentic-dspark-disagg`，recipe
`benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/b300-fp4/agentx/disagg-variants.yaml`，计分 run
[43769338888](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43769338888)（PR [#4585](https://github.com/SemiAnalysisAI/InferenceX/pull/4585)，11-16）。
原始数据放在工作目录 `analysis/ixref/`：各点结果 JSON（`b300-run-43769338888/a*/`）、c768 一点的引擎日志（`srvlogs-c768/`，
log_stats 每秒一行）。脚本：`scripts/ax_summary.py`（结果 JSON 一行汇总）、`scripts/logstats.py`（引擎 log_stats 分位数）。

## 1. B300 各点的成绩（AgentX 正式计分，3600 s，golden AL 3.51）

| 并发 | 布局（P 卡 : D 卡） | GPU | tok/s/GPU | 输出 tok/s/GPU | 交互度 p50 / p90 | TTFT p50 / p90（s） | ITL p50（ms） | req/s | prefill 现算占输入 |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 2×DEP2 : DEP8（4:8） | 12 | 154,000 | 1,029 | 292 / 275 | 0.22 / 0.63 | 3.42 | 12.1 | 2.9% |
| 768 | 2×DEP2 : DEP8（4:8） | 12 | 420,200 | 2,946 | 224 / 214 | 0.63 / 1.59 | 4.46 | 34.4 | 3.1% |
| 1280 | 3×DEP2 : DEP8（6:8） | 14 | 576,600 | 4,046 | 192 / 175 | 0.75 / 1.99 | 5.21 | 55.1 | 3.1% |
| 2048 | 4×DEP2 : DEP8（8:8） | 16 | 738,100 | 5,263 | 134 / 121 | 0.92 / 2.74 | 7.44 | 82.5 | 3.2% |
| 3584 | 5×DEP2 : DEP8（10:8） | 18 | 840,400 | 6,265 | 88 / 85 | 7.82 / 12.95 | 11.34 | 111.3 | 3.7% |

（交互度 = 1 / ITL，tok/s/user；"prefill 现算占输入" = 服务端 computed / prefill 侧 prompt token。）

## 2. 吞吐从哪里来

1. **计分口径决定了吞吐几乎全是"命中缓存的输入"**：平均输入 14.6 万 token、输出约 1,020（c768：145,765 / 1,029），
   total tok/s 里 99.3% 是输入；而输入里 96.9% 命中 prefill 的前缀缓存。所以 **tok/s/GPU ≈ req/s × 14.6 万 ÷ GPU 数**，
   req/s 由 decode 出 token 的速度（每请求约 1,020 个）和 TTFT 决定。
2. **固定并发下，GPU 越少、延迟越低，tok/s/GPU 越高**：AgentX 的每条回放线有工具调用 / 思考的空闲（上限 300 s），
   c768 时服务端平均只有 185 个请求在途（decode 154、prefill 31，`profile_export_console.txt`）。B300 c768 的 prefill
   很空（下表），用了 12 卡；同样的 req/s 如果 8 卡做到，每卡就是 1.5 倍。
3. **B300 c768 稳态（05:00–05:53，log_stats 每秒一行，`srvlogs-c768/b300-c768-logstats.txt`）**：

   | | 在跑 p50 / p90 | 排队 p50 / p90 | 吞吐均值 | KV 占用 | 前缀命中 |
   |---|---|---|---|---|---|
   | decode（DEP8，1 个引擎） | 162 / 186（每 rank 约 20） | 5 / 11 | 生成 36,700 tok/s（每卡 4,590） | 1.9% | —（外部 100%） |
   | prefill w0（DEP2） | 4 / 9 | 1 / 15 | prompt 80,300 tok/s | 1.3% | 96.4% |
   | prefill w1（DEP2） | 4 / 9 | 1 / 12 | prompt 78,500 tok/s | 1.3% | 96.4% |

   decode 每步约 3.51 × 4.46 ms ≈ 15.7 ms（每 rank 约 20 个请求 × 6 token，全 EP 组 960 token）；接受长度 3.50。
4. **prefill 算力**：预热第一段（冷缓存，04:41–04:49）B300 每个 prefill 引擎（DEP2，2 卡）平均 207,000 prompt tok/s
   （p50 141k），我们 33d 同一阶段每个 prefill 引擎（DP2 + MoE TP2，2 卡）平均 70,800（p50 74k）、排队 308：
   **单卡冷 prefill 约慢 2.9 倍**。B300 稳态只用了 prefill 能力的约 40%（c768 时 4 卡现算约 156k tok/s）。
5. **前缀缓存容量**：B300 prefill 每 rank KV 88.5 GiB = 4,789 万 token（`kv-cache-dtype fp8`、显存 0.95，约 1.94 KB/token）；
   我们 60.5 GiB = 2,632 万 token（默认 dtype、0.9，约 2.30 KB/token），只有 55%。B300 的 70 GiB CPU 层（SimpleCPUOffload lazy）
   在 c768 只贡献了约 0.6% 的命中（prefill 日志 External hit 均值 0.6%）——**命中主要靠 GPU 前缀池**。

## 3. 分项拆解（B300 recipe vs 我们现在）

| 方面 | B300 dynamo + vLLM | 我们（infera + vLLM，32c / 33d / 35 的配置） | 判断 |
|---|---|---|---|
| router | Dynamo KV router（`router-mode kv`，prefill `kv_events: true`，router 对 prefill 缓存有精确视图）；session affinity TTL 4 h；不设 busy 阈值（不在 router 排队）；`DYN_ROUTER_TEMPERATURE=0`；frontend 用 fastokens + 8 GB tokenizer 缓存（长对话增量分词） | Rust kv-aware（KV 事件链没接上，退化成只看负载，`router.log` 里 `degenerated to load-only`）+ prefill session affinity（按 DP rank 粘住会话）；引擎各自渲染 / 分词（prefill 和 decode 各做一次） | 命中率靠 affinity 也能接近；分词 / 渲染在高 req/s 下可能成瓶颈（GLM5.2 的教训，renderer 线程） |
| 并行 | prefill DEP2（DP2 attention + EP2），每 worker 2 卡；decode DEP8 + EPLB（16 个冗余专家，`enable-ep-weight-filter`），一个 decode 引擎 8 卡 | prefill DP2 + MoE TP2；decode DP4 + MoE TP4（单节点只剩 4 卡给 decode） | decode 8 卡一个引擎让 MoE 每步聚合 960 token；我们 DP4 只有一半。跨节点 P/D 才能 DP8 |
| kernel | `FLASHMLA_MEGA_ATTN_DSV41`（融合 sparse MLA）、`deep_gemm_mega_moe`（EP dispatch + grouped GEMM + combine 一个 kernel，NVLink 对称内存）、`deep_select` indexer top-k、flashinfer autotune；decode `FULL_DECODE_ONLY` 图，捕获大小是 6 的倍数；prefill FULL_AND_PIECEWISE 到 2046 | aiter MoE、MXFP4 indexer、Triton sparse MLA（**P/D 下 `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` 因 KV connector 被忽略**，decode 日志 `rocm.py:103`）、breakable cudagraph、capture 到 1024、DP 下不能用 monokernel | 单卡 prefill 慢 2.9 倍、decode 每步慢多少待测——这是最大的差距，不是配置能补的 |
| P / D 配比 | 4:8（c256 / c768）→ 6:8 → 8:8 → 10:8（c3584）；prefill 稳态只用约 40% | 4:4（2P1D）或 2:4（1P1D） | 我们 prefill 慢，P 卡不能少；但 2:4 的分母小，要测 |
| KV offload | prefill：MultiConnector（NIXL push + SimpleCPUOffloadConnector lazy 70 GiB / worker）；decode 无前缀缓存、无 offload | 无 | B300 的 CPU 层几乎没贡献（0.6%）；我们 GPU 池只有 55%，CPU 层可能更有用（GLM5.2：写穿 OffloadingConnector + 精确锁页） |
| vLLM 调度 | prefill：`max-num-seqs 64`、`max-num-batched-tokens 16384`、`long-prefill-token-threshold 8192`、`prefill-schedule-interval 2`、`prefix-cache-retention-interval 32768`、显存 0.95、kv fp8；decode：`max-num-seqs` 每 rank 18 / 40 / 72 / 132 / 168（按并发档）、`max-num-batched-tokens` = 10 × max-num-seqs、显存 0.95、`VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=4`、`VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`；NIXL push + lease 120 s | prefill / decode 都是 `max-num-batched-tokens 16384`、显存 0.9、kv 默认 dtype、无 max-num-seqs 上限 | 可直接移植：prefill 的 long-prefill 阈值 / 显存 / kv fp8（要过正确性）；decode 的 max-num-seqs 上限是 pareto 旋钮（换交互度），不加吞吐 |

注：`VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` 在我们的镜像（`9c1d45c2`，比 B300 的 `96710245` 新）里已经不存在。

## 4. 可移植到我们实验里的配置（按预期收益排序）

1. **prefill 前缀缓存容量**：显存 0.9 → 0.95（+14 GiB / rank，约 +24% token）；`kv-cache-dtype fp8`（每 token 2.30 → 约 1.94 KB，
   再 +18%）。两者都要过正确性（真实接受率的 GSM8K / 16k）。
2. **prefill 调度**：`long-prefill-token-threshold 8192`（长 prompt 切成 8k 块，短的能插进来，改善 TTFT 分布）、`max-num-seqs 64`。
3. **DSpark K=3（golden AL 2.90）**：GLM5.2 那边试过的 golden 2.99（MTP K=3）在 DSv4.1 上对应 DSpark K=3 / 2.90。每步每请求
   4 个 token（K=5 是 6 个），每个计算 token 的产出 0.725 对 0.585；decode 吞吐受算力限制时可能更高，交互度可能低一点。
   InferenceX 按（模型、方法、K、thinking）查 golden，K=3 合规。
4. **P / D 配比 2:4（1P1D，6 卡计分）**：分母从 8 变 6；prefill 能否扛住要看命中率和 req/s。
5. **CPU KV 层**（prefill）：B300 的 SimpleCPUOffload lazy 在它那边几乎没用；我们池子小，可能有用。ROCm 锁页取整（U26）要用精确锁页覆盖层。
6. **decode `max-num-seqs` 上限 + 6 的倍数的 cudagraph 捕获**：用来在 pareto 上换交互度（低并发点），不是吞吐项。
7. **不能直接移植**：mega MoE / FlashMLA mega attention / deep_select（CUDA 专用）；DEP8 decode 要两节点；NIXL push（我们用 Mooncake）。
   EP 在 GLM5.2 的 MI355X 上不比 MoE TP 快（A 线结论），DSv4.1 未测。

## 5. 与我们数据的对比

我们 11-19 的三个正式点（32c c512、33d c768、35 c1024）都没有成绩：32c / 33d 在 AgentX 预热中 decode DP4 卡死（README 第 4.4 节），
35 因作业被抢占中断。能比的只有预热阶段的引擎指标（上面第 2 节第 4、5 条）。正式成绩出来后补这一节。
