# GLM-5.2-MXFP4 P8+DPA / D8+DPA AgentX 实验报告

## 实验目标与固定口径

- 模型：`GLM-5.2-MXFP4`
- 部署：1 Prefill + 1 Decode
- Prefill：TP8 / EP1 / attention-DP8，137 的 GPU 0-7
- Decode：TP8 / EP1 / attention-DP8，138 的 GPU 0-7
- Prefill/Decode HiCache：关闭
- Decode MTP：EAGLE，5 steps / 6 draft tokens / top-k 1
- AgentX 模拟接受长度：`3.61`
- 并发点：8、16、32、48、64、96、128、160、192
- 每点计分窗口：1200 秒
- 每次并发变化都执行完整 stop → HBM idle → fresh launch
- 每点两端的 max running 与 CUDA graph max batch 均等于该点并发
- 不跨点复用 prefix cache
- 本阶段只运行 AgentX，不重复 correctness

历史运行通过命令行把 `config.sh` 的 P4:D8 默认值覆盖为 P8D8。为防止漏传后静默
退回 P4D8，现已新增固定拓扑配置 `config.p8d8.sh`；它将 P/D GPU 列表、TP、DP、
DPA 和容器前缀固定为 P8D8。每个性能点仍必须显式传入四个与目标并发相等的
`PREFILL/DECODE_{MAX_RUNNING,GRAPH_MAX_BS}`，并在 benchmark 前从 live container
命令再次校验。

## 数据采用口径与路径索引

本报告中的相对路径均以本报告所在目录为根：
`/home/liyingli/bench_agentx/Infera/bench/glm5p2_pd/results/`
`20260913_p8dpa_d8dpa_c8-192_d1200_137_138/`。每个结果目录内：

- `agentx_conc<C>.json` 是 QPS、吞吐、延迟、cache hit 和请求计数的汇总来源；
- `aiperf_artifacts/profile_export.jsonl` 是逐请求原始记录；
- `aiperf_artifacts/server_metrics_export.csv` 是 Prefill/Decode queue、running、
  KV cache 和 transfer 指标来源；
- `benchmark_command.txt`、`runtime.env` 和 `service/*-server-info.json` 用于核对
  benchmark 参数、镜像和实际 engine max-running/CUDA graph 配置；
- `prefill.log`、`decode.log`、`router.log` 和 `prefill-hbm.jsonl`（若存在）只用于
  故障归因和 HBM 机制验证，不代替性能汇总。

当前 intended-commit 数据分成两组。原始 warmup10 历史曲线只有以下三个可采用点：

- 原始 sweep C8：`agentx/c8/bench-prefill-scratch-reclaim/agentx_conc8.json`；
- 原始 sweep C16：`agentx/c16/bench-scratch0-mem070/agentx_conc16.json`；
- 原始 sweep C32：`agentx/c32/bench/agentx_conc32.json`，包含 8.674% profiling
  空响应，只能作为历史 goodput 样本；

修复镜像后、统一 warmup1 且计分窗口均为 1,200 秒的当前 P8D8 性能曲线为：

- C32：`retest-pinned-6ff85f4a-warmup1/c32/bench/agentx_conc32.json`；
- C48：`retest-pinned-6ff85f4a-warmup1/c48/bench/agentx_conc48.json`；
- C64：`retest-pinned-6ff85f4a-warmup1/c64/bench/agentx_conc64.json`。

三点均 fresh launch，P/D max-running 与 CUDA graph max batch 分别严格为
32/48/64。它们可彼此横向比较并作为当前性能数据；因 warmup1 与原始曲线的
warmup10 不同，不将两组吞吐做纯代码 A/B。

以下路径是明确的机制对照或无效产物，不纳入 intended-commit 最终性能比较：

- 原始 sweep C48/C64/C96/C128/C160/C192：`agentx/c<C>/bench/`；
- C48 allocator no-op 对照：`debug-oor/c48-allocator-gc080/bench/`；
- max-running=8 的 active-GC C48/C96 机制样本：
  `debug-oor/c48-allocator-gc080-active/bench/` 和
  `debug-oor/c96-allocator-gc080-active/bench/`；
- C192 warmup1 失败：
  `retest-final/c192-affinity-gc080-warmup1/bench/`，profiling 未启动；
- digest `34909eb3...` 的 C32/C48/C64/C96：
  `retest-sequence-warmup1/c32/bench/`、`retest-sequence-warmup1/c48/bench/`、
  `retest-final/c64-affinity-gc080/bench/` 和
  `debug-oor/c96-max96-allocator-gc080-active/bench/`。这些点可用于定位新栈的
  batch-size cliff 和 active-GC 机制，但其实际 SGLang/AITER 为
  `402df1e/2c71811`，与 tag 后缀不符，不能进入 intended-commit 性能曲线；
- 修复后 300 秒 C32 验证：
  `debug-itl/c32-pinned-6ff85f4a-smoke/bench/agentx_conc32.json`。该点仅验证
  ITL 修复，不替代统一 warmup 口径下的 1,200 秒正式结果。

根目录的 `results.csv` 已同时保留原始 C8/C16/C32，并加入当前 pinned-image
warmup1 C32/C48/C64；`Point` 列用于区分两组，单点 JSON 仍是最终事实来源。

## 节点与镜像

- Prefill：`crsuse2-m2m-137`，`10.245.153.247`
- Decode：`crsuse2-m2m-138`，`10.245.157.237`
- 镜像：`infera/engine-sglang:glm52-v518-c29bd17-b02ab81`
- 历史 intended-commit Image ID：`sha256:8bef14c08ba4764698b5b9c82f41e1a3e5c0f7bc3c4a22d161cd00dbf1feb456`
- 被污染的同名 tag Image ID：`sha256:34909eb390cc5133aa0c6c2dfef7965ca28ec96a7ca1415629c2f5c477299b87`
- 修复后当前两节点 Image ID：`sha256:6ff85f4a43ae4b2773f59e827f230e23a657ae86ebf3d00a73b75bf94d5f9695`
- 当前镜像已在两节点容器内复核：
  SGLang `c29bd17d3584ccc9fa7ef2fc5510ddb8874f4e12`、
  AITER `b02ab811e5fea6deda9b56f546731f5285739d68`
- HiCache：关闭
- 隔离端口：etcd client 12379、peer 12380、router 18000
- Router backend：Rust；HTTP PD 路径直接透传 Decode SSE 字节，不经过 Python
  `infera/router/disagg.py` 的流处理

## 瓶颈判定口径

每点同时读取 AgentX 汇总和
`aiperf_artifacts/server_metrics_export.{json,csv}`。多 DP rank histogram 按
count/sum/buckets 合并，不平均各 rank 的 P90。

- Prefill：Prefill scheduler queue 堆积且 TTFT 同步变差，Decode queue 低
- Decode：Decode running 接近上限或 queue 持续堆积，ITL 变差且吞吐增益收敛
- KV transfer：transfer/prealloc queue 堆积，或 transfer latency 显著上升
- 未饱和：两端 queue 均低且吞吐仍随并发增长

每点将给出完整性能、瓶颈、P:D 比例建议和可独立 A/B 的优化方向。

以 C8 queue 为例，原始文件为
`agentx/c8/bench-prefill-scratch-reclaim/aiperf_artifacts/server_metrics_export.csv`。
筛选 `Metric=sglang:queue_time_seconds`，按 Prefill/Decode 分组，把 8 个 `dp_rank`
行末的 histogram bucket count 相加，再找累计比例首次达到 90% 的 bucket；不能平均
8 个 rank 各自的 P90。C8 Prefill 共 424 个样本，其中 389 个不超过 10 ms
（91.7%）；Decode 共 424 个样本，其中 401 个不超过 1 ms（94.6%），因此得到
下文的 pooled P90 上界。

## Pinned-image C32/C48/C64 正式复测

三点使用修复后 Image ID `6ff85f4a...`，镜像内 SGLang/AITER 为
`c29bd17/b02ab81`；均为 warmup1、1,200 秒 profiling、相同 trace/seed 和每点 fresh
launch。

请求计数口径：汇总 JSON 的 `num_requests_total` 是 warmup 与 profiling 的记录总数，
`num_requests_successful` 只是计分窗口内的请求数，二者之差是被丢弃的 warmup 记录，
不是失败请求。分阶段计数直接取自
`aiperf_artifacts/profile_export.jsonl` 的 `metadata.benchmark_phase`；由此确认三点
的 `InvalidInferenceResultError` 全部落在 warmup 阶段，profiling 错误率均为 0%。
汇总 JSON 的 `records_error_dropped` 统计的也是这些 warmup 阶段错误，所以它不与
`records_warmup_dropped` 相加，这解释了此前各点出现的记账差异。性能结果如下：

- C32：计分请求 1,394，warmup 丢弃 66，warmup 与 profiling 错误均为 0；平均 QPS
  1.13807，总吞吐 130,769.14 tok/s（8,173.07 tok/s/GPU）；TTFT P50/P90/P95 为
  2.254/7.061/9.653 秒，ITL P50/P90/P95 为 9.91/11.63/12.50 ms，
  P90 interactivity 85.96 tok/s/user；GPU cache hit 95.019%，KV 使用率 50%。
- C48：计分请求 1,732，warmup 丢弃 99（含 2 条 warmup-only
  `InvalidInferenceResultError`），profiling 错误 0；平均 QPS
  1.41109，总吞吐 183,767.75 tok/s（11,485.48 tok/s/GPU）；TTFT
  P50/P90/P95 为 2.448/9.064/13.647 秒，ITL P50/P90/P95 为
  10.47/12.50/13.98 ms，P90 interactivity 79.97 tok/s/user；GPU cache hit
  95.789%，KV 使用率 67%。
- C64：计分请求 2,036，warmup 丢弃 131（含 1 条 warmup-only
  `InvalidInferenceResultError`），profiling 错误 0；平均 QPS
  1.66041，总吞吐 227,141.49 tok/s（14,196.34 tok/s/GPU）；TTFT
  P50/P90/P95 为 3.199/12.820/18.155 秒，ITL P50/P90/P95 为
  11.81/15.11/18.46 ms，P90 interactivity 66.16 tok/s/user；GPU cache hit
  95.626%，KV 使用率 96%。

C32→C48→C64 总吞吐持续增长，分别为 130.77k、183.77k、227.14k tok/s；代价是
P90 TTFT 从 7.06 秒升至 12.82 秒、P90 ITL 从 11.63 ms 升至 15.11 ms。C64 已达到
96% GPU KV 使用率，是当前曲线的容量边界点；继续提高并发需要优先防止 KV 容量和
Prefill/transfer 排队恶化。

## Wrong-image 顺序复测与 ITL 修复验证

### 作废的 fresh C32（digest `34909eb3...`）

- 诊断汇总：`retest-sequence-warmup1/c32/bench/agentx_conc32.json`
- 逐请求记录：
  `retest-sequence-warmup1/c32/bench/aiperf_artifacts/profile_export.jsonl`
- 服务指标：
  `retest-sequence-warmup1/c32/bench/aiperf_artifacts/server_metrics_export.csv`
- 启动验收：
  `retest-sequence-warmup1/c32/launch/server-info/{prefill-0,decode-0}.json`
- Prefill/Decode `max-running` 与 CUDA graph max batch 均已验收为 32/32；
- warmup1：66/66 完成、错误 0，耗时 577.18 秒；
- profiling 完成并成功 806/806、错误 0；发送窗口结束后取消 17 条在途请求；
- 有效统计窗口：1,215.891 秒，平均 QPS 0.66337；
- 总吞吐 67,043.30 tok/s，每 GPU 4,190.21 tok/s；
- TTFT P50/P90/P95：2.10182/11.21700/15.63215 秒；
- E2E P50/P90/P95：14.02647/65.24053/108.85378 秒；
- ITL P50/P90/P95：43.00/47.40/48.48 毫秒；
- P90 interactivity：21.10 tok/s/user；
- GPU cache hit / 理论值：93.878% / 96.007%，GPU KV cache 使用率 52%。

该点证明 digest `34909eb3...` 在 C32 的完整 1,200 秒窗口内没有 profiling 空响应，
但不能作为 intended-commit 性能点。它使用每 lane 1 条附加 warmup，而原始 C32
使用每 lane 10 条；AgentX 会把 warmup 结束时的 live trajectory 状态交给
profiling，所以两轮的请求密度、输入分布和 cache 状态不同，67,043.30 与原始
150,211.00 tok/s 不是纯代码 A/B。

同一错误 digest 的其他证据给出明确的 per-DP batch cliff：

- C48 的 `max/graph=8` 机制样本在 DPA8 下每 rank 仅捕获 BS1，ITL P50 为
  8.82 ms；路径为
  `debug-oor/c48-allocator-gc080-active/bench/agentx_conc48.json`；
- C32/C48/C64 严格使用全局 `max/graph=32/48/64`，每 rank 捕获到 BS4/6/8，
  ITL P50 分别为 43.00/46.09/46.51 ms；
- C96 全局 `max/graph=96`，每 rank 捕获到 BS12，ITL P50 又恢复到
  12.36 ms；
- C64 已使用完整 warmup10 仍为 46.51 ms，排除“仅因 warmup1 未热透”的解释。

服务端指标进一步排除客户端统计假象：用
`sum(num_running_reqs)/generation_token_rate` 计算的 time-per-output，fresh
C32/C48 分别为 43.15/48.76 ms，历史 C32 与 wrong-image C96 分别为
11.80/14.31 ms，与请求侧 TPOT 同步。四轮 MTP accept length 均约 3.55--3.60；
按 ISL 分桶后异常点仍稳定在 42--47 ms；Decode forward 均走
`decode_cuda_graph`，没有 eager fallback。故慢值来自服务端 Decode 输出能力下降，
不是 AIPerf 展示口径、MTP acceptance、请求长度或 graph fallback。

因此 ITL 异常的已闭合根因边界是：**同名 tag 被覆盖为
SGLang `402df1e` + AITER `2c71811` 后，新 Decode 栈在 DPA8 中小 batch 负载下进入
慢模式。** per-DP graph tier 4--8 与慢模式强相关，tier 1/12 样本正常；但现有历史
样本同时改变了 max-running、graph tier 或并发，未做 wrong-image 上完全受控的
单变量 graph A/B，所以不能把“具体 graph 阈值”写成已闭合的底层根因。也没有做
SGLang/AITER 交叉镜像矩阵，不能虚构为某一个具体 commit。行动修复以回退并固定
已知良好的完整源码组合为准。

修复采用显式完整 SHA 重建，并在每次 launch 前同时校验所有节点 Image ID 与容器内
`git rev-parse`。修复后 digest 为 `6ff85f4a...`。C32 使用 P8D8、
P/D `max-running=graph-max=32`、rank affinity、Prefill active-GC 和 warmup1 做
300 秒最小验证：

- 预检：`debug-itl/c32-pinned-6ff85f4a-smoke/preflight-p8d8/validation.json`，
  1 条 P→D edge 的 8/8 GPU WRITE paths 全部通过；
- 启动验收：
  `debug-itl/c32-pinned-6ff85f4a-smoke/launch-pinned/server-info/`；
- 汇总：
  `debug-itl/c32-pinned-6ff85f4a-smoke/bench/agentx_conc32.json`；
- profiling 227/227 成功、错误 0；
- ITL P50/P90/P95：8.25/9.44/9.83 ms，较 wrong-image C32 的
  43.00/47.40/48.48 ms 恢复；
- TTFT P50/P90：0.832/3.420 秒，总吞吐 64,187.08 tok/s。

最后一项吞吐仅是 300 秒 warmup1 轨迹样本，不能与历史 1,200 秒 warmup10 的
150,211.00 tok/s 作最终比较。最终 C32→C48→C64 必须统一 warmup 口径后 fresh
launch；warmup1 只能作为快速筛查，不能替代可与历史曲线比较的认证结果。

## 原始 sweep 进度（历史基线）

本节及紧随其后的九个 `AgentX C*` 小节记录首次完整 sweep；修复后的正式结果和替代
路径以上面的“数据采用口径与路径索引”为准。

- C8：完成
- C16：完成
- C32：完成
- C48：Prefill 显存容量失败，无有效性能结果
- C64：warmup 完成；profiling 错误率 10.076% 且窗口提前结束，无有效 1200 秒结果
- C96：Prefill DP0 显存容量失败，无有效性能结果
- C128：warmup 完成；profiling 错误率 17.188% 且窗口提前结束，无有效 1200 秒结果
- C160：warmup 完成；profiling 错误率 10.5% 触发在线门禁，无有效 1200 秒结果
- C192：Prefill DP5 显存容量失败，无有效性能结果
- 全部并发点均已执行；每点均完成匹配参数 stop 和下一点 fresh launch

## AgentX C8

- 配置：`CONC=8`
- 本节数据源：`agentx/c8/bench-prefill-scratch-reclaim/`
- Prefill/Decode max running：8 / 8
- Prefill/Decode CUDA graph max batch：8 / 8
- Prefill scratch reclaim：开启（`PREFILL_HSA_NO_SCRATCH_RECLAIM=0`）
- Decode scratch reclaim：保持基线关闭（`DECODE_HSA_NO_SCRATCH_RECLAIM=1`）
- launch 命令总耗时：3,839.285 秒；其中包含等待外部任务释放 GPU 的时间
- warmup：87/87 成功，耗时 452.55 秒
- 实际统计窗口：1,229.841 秒
- 计分请求：337
- warmup 请求：87
- 错误请求：0
- 平均 QPS：0.27362
- 总吞吐：32,851.28 tok/s
- 每 GPU 总吞吐：2,053.21 tok/s
- 每 GPU 输入吞吐：2,037.23 tok/s
- 每 GPU 输出吞吐：15.98 tok/s
- P50 TTFT：0.75641 秒
- P90 TTFT：3.85857 秒
- P95 TTFT：9.69213 秒
- P50 E2E：3.76628 秒
- P90 E2E：23.95173 秒
- P95 E2E：34.04052 秒
- P50 ITL：6.77 毫秒
- P90 ITL：7.64 毫秒
- P90 interactivity：130.90 tok/s/user
- GPU cache hit rate：97.491%
- 理论 cache hit rate：97.698%
- GPU KV cache 使用率：25%
- CPU HiCache 使用率：不适用

该点的 AgentX 结果通过门禁：profiling 337/337 成功、错误率 0，TTFT 和 ITL
覆盖率均为 100%。1200 秒发送窗口结束后使用 30 秒 grace period 完成所有在途请求；
`grace_period_timeout=True` 表示 drain 等待到期时已无在途 credit，不是请求失败。

### C8 瓶颈判断与下一步

结论：**C8 尚无持续的 Prefill、Decode 或 KV transfer 饱和；Decode 相对更忙，
当前首先受负载不足限制。**

证据：

- 合并 8 个 DP rank 后，Prefill scheduler queue P90 不超过 10 毫秒，Decode
  scheduler queue P90 不超过 1 毫秒；两端都没有持续排队。
- Prefill 8 个 rank 的平均 running request 合计约 0.11；Decode 8 个 rank合计约
  5.17。Decode 请求生命周期明显更长，但每 rank 最大 running request 仍只有 1。
- P90 ITL 7.64 毫秒、P90 interactivity 130.90 tok/s/user，Decode 在当前负载下仍
  保持较好的单用户生成速度。
- Prefill 侧记录 338 次 KV transfer，按 count 加权平均 transfer latency 为
  96.09 毫秒、平均速度为 90.82 GB/s；Decode transfer queue P90 为 0、最大仅 1，
  因而 transfer 不是持续瓶颈。
- GPU cache hit 为 97.491%，绝大多数长 prompt token 命中 Prefill GPU cache，
  Prefill 实算负载很低。
- 与 P4:D8 C8 相比，总吞吐 32,851.28 vs 32,944.39 tok/s，基本持平；P8:D8
  使用 16 GPU 后每 GPU 吞吐从 2,745.37 降至 2,053.21 tok/s。这证明 C8 增加
  Prefill GPU 没有吞吐收益，符合 Prefill 未饱和的 queue 证据。

P/D 比例与优化建议：

- 若只服务 C8，P8:D8 资源过配；P4:D8 已有相同性能且 GPU 效率高约 33.7%。
- 若目标是最低延迟，P8 将 P50/P90 TTFT 从 P4 的 0.763/4.263 秒小幅改善到
  0.756/3.859 秒，但差异不足以证明值得增加 4 张 Prefill GPU。
- 当前不应增加 Decode，也没有证据需要调 Mooncake。继续扫描更高 concurrency，
  用吞吐拐点及两端 queue 决定是增加 Prefill 还是 Decode 副本。

## AgentX C16

- 配置：`CONC=16`
- 本节数据源：`agentx/c16/bench-scratch0-mem070/`
- Prefill/Decode max running：16 / 16
- Prefill/Decode CUDA graph max batch：16 / 16
- Prefill scratch reclaim：开启（`PREFILL_HSA_NO_SCRATCH_RECLAIM=0`）
- Prefill 静态 KV 比例：0.70
- Decode scratch reclaim/静态 KV 比例：基线 1 / 0.85
- 成功 A/B launch 耗时：806.238 秒
- warmup：177/177 成功，耗时 682.13 秒
- 实际统计窗口：1,206.979 秒
- 计分请求：599
- warmup 请求：177
- profiling 错误请求：0
- 结果收集器另记录 2 条 `InvalidInferenceResultError` error-dropped 记录；它们未计入
  599 条计分成功记录，AIPerf 最终门禁仍为 0/599
- 平均 QPS：0.49668
- 总吞吐：63,846.43 tok/s
- 每 GPU 总吞吐：3,990.40 tok/s
- 每 GPU 输入吞吐：3,958.03 tok/s
- 每 GPU 输出吞吐：32.38 tok/s
- P50 TTFT：1.34878 秒
- P90 TTFT：4.95718 秒
- P95 TTFT：9.03514 秒
- P50 E2E：4.96224 秒
- P90 E2E：26.65591 秒
- P95 E2E：43.94675 秒
- P50 ITL：8.18 毫秒
- P90 ITL：9.21 毫秒
- P90 interactivity：108.55 tok/s/user
- GPU cache hit rate：96.632%
- 理论 cache hit rate：96.760%
- GPU KV cache 使用率：37%
- GPU KV cache capacity：42,717,184 token
- CPU HiCache 使用率：不适用

该点通过现有 AgentX 门禁。profiling phase 完成 599、取消 0、错误 0；TTFT/ITL
覆盖率为 99.9%/100%，均高于 95% 门槛。

### C16 瓶颈判断与下一步

结论：**C16 仍未形成稳定的 P/D compute 饱和；Decode batching 开始降低交互性，
KV transfer 明显变慢但尚未持续排队，是当前次要限制。**

证据：

- C8→C16 总吞吐从 32,851.28 增至 63,846.43 tok/s，增幅 94.4%，接近并发翻倍，
  尚未出现明显吞吐拐点。
- 合并 rank 后 Prefill scheduler queue P90 不超过 50 毫秒，Decode P90 不超过
  1 毫秒；Prefill 有少量十秒级尾部样本，但并非持续排队。
- Prefill 8 rank 平均 running request 合计约 0.62；Decode 8 rank 合计约 7.82，
  每个 Decode rank 的最大 running request 为 2，远低于全局 max-running 16。
- C8→C16 的 P90 ITL 从 7.64 增至 9.21 毫秒，P90 interactivity 从 130.90 降至
  108.55 tok/s/user，说明 Decode batching 已产生约 17.1% 的交互性损失，但
  Decode queue 尚未持续堆积。
- Prefill 侧记录 600 次 KV transfer，count 加权平均 latency 为 235.88 毫秒，
  相比 C8 的 96.09 毫秒增加 145%；平均速度从 90.82 降至 69.54 GB/s。
  Decode transfer queue 的 pooled P90 仍为 0、最大 2，所以 transfer 退化会增加
  TTFT，但尚未成为持续吞吐瓶颈。
- GPU cache hit 96.632%，降低 mem fraction 后仍有 42.72M token KV 容量，本点
  峰值使用 37%，没有容量压力。
- 与 P4:D8 C16 相比，总吞吐 63,846.43 vs 63,483.52 tok/s，仅高 0.57%；
  每 GPU 吞吐 3,990.40 vs 5,290.29 tok/s，低 24.6%。P8 暂未兑现额外
  Prefill GPU 的性能收益。

P/D 比例与优化建议：

- C16 以效率为目标仍应选择 P4:D8；P8:D8 仅把 P50/P90 TTFT 从
  1.480/5.105 秒小幅降至 1.349/4.957 秒。
- 当前没有增加 Prefill 或 Decode 副本的证据。继续 C32 及以上，重点观察总吞吐增益、
  Decode ITL/queue 和 Prefill queue 哪一项先出现拐点。
- KV transfer 已有退化但 queue 未堆积，暂不改变网络开关；若更高点 transfer queue
  开始非零并同步推高 TTFT，再独立 A/B NIC/lane 配置。

## AgentX C32

- 配置：`CONC=32`
- 本节数据源：`agentx/c32/bench/`
- Prefill/Decode max running：32 / 32
- Prefill/Decode CUDA graph max batch：32 / 32
- Prefill `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：0 / 0.70
- Decode `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：1 / 0.85
- launch 耗时：827.742 秒
- warmup：354/354 完成，耗时 842.56 秒
- 实际统计窗口：1,216.222 秒
- profiling 完成请求：1,614，另有 2 条窗口末在途请求被取消
- 计分成功请求：1,474
- 计分错误：140，均为 `InvalidInferenceResultError`
- 计分错误率：140/1,614 = 8.674%，低于 10% 门槛
- 平均 QPS：1.21235
- 总吞吐：150,211.00 tok/s
- 每 GPU 总吞吐：9,388.19 tok/s
- 每 GPU 输入吞吐：9,318.69 tok/s
- 每 GPU 输出吞吐：69.50 tok/s
- P50 TTFT：2.06044 秒
- P90 TTFT：5.99664 秒
- P95 TTFT：9.19702 秒
- P50 E2E：5.90590 秒
- P90 E2E：24.75090 秒
- P95 E2E：44.03000 秒
- P50 ITL：9.90 毫秒
- P90 ITL：11.83 毫秒
- P90 interactivity：84.55 tok/s/user
- GPU cache hit rate：92.548%
- 理论 cache hit rate：96.881%
- GPU KV cache 使用率：56%
- GPU KV cache capacity：42,717,184 token
- CPU HiCache 使用率：不适用

该点通过现有 AgentX 门禁：计分错误率 8.674% 低于 10%，TTFT/ITL 覆盖率均为
100%。两条取消请求发生在 1200 秒发送窗口和 drain 结束处，不计入成功吞吐。

### C32 瓶颈判断与下一步

结论：**主限制已转为 Prefill queue 与 KV transfer 的共同压力；Decode batching
继续降低交互性，但 Decode compute queue 仍低。P8 相比 P4 显著改善了 TTFT 和
transfer，但多出的 4 GPU 只换来小幅吞吐增益。**

证据：

- 合并 8 个 Prefill DP rank 后 scheduler queue pooled P90 位于 2–3 秒 bucket；
  Decode pooled P90 不超过 1 毫秒。Prefill 是 compute 排队主侧。
- Prefill 8 rank 平均 running request 合计约 3.10；Decode 8 rank 合计约 12.86。
  Decode 每 rank 最大 running request 为 4，远低于全局 max-running 32。
- Decode transfer queue 持续非零：8 rank 平均合计约 2.21 个请求，各 rank P90
  为 1–2、最大 2–3；因此 KV transfer 已成为共同瓶颈，而不只是尾部延迟。
- Prefill 侧记录 1,476 次 KV transfer，count 加权平均 latency 为 440.47 毫秒，
  平均速度为 37.42 GB/s；相较 C16 的 235.88 毫秒/69.54 GB/s 明显退化。
- C16→C32 总吞吐从 63,846.43 增至 150,211.00 tok/s，增幅 135.3%；负载仍能
  换取吞吐，但 P90 TTFT 从 4.96 增至 6.00 秒，Prefill/transfer 压力开始可见。
- P90 ITL 从 9.21 增至 11.83 毫秒，interactivity 从 108.55 降至
  84.55 tok/s/user。Decode batching 成本继续上升，但 queue 证据不支持把 Decode
  判为当前主瓶颈。
- 与 P4:D8 C32 相比，P8:D8 总吞吐提高 6.0%（150,211.00 vs 141,649.61），
  P90 TTFT 改善 48.8%（6.00 vs 11.71 秒），KV transfer latency/速度也从
  843.63 毫秒/25.40 GB/s 改善到 440.47 毫秒/37.42 GB/s。代价是每 GPU 吞吐
  低 20.5%（9,388.19 vs 11,804.13 tok/s）。

P/D 比例与优化建议：

- 受 TTFT SLO 约束时，P8:D8 已证明优于 P4:D8；受 GPU 效率约束时，P4:D8 更优。
- 当前不应增加 Decode 比例，因为 Decode compute queue 基本为空。若 C48 以上
  Prefill queue 继续快速增长，优先考虑增加 Prefill 副本；它也能分摊 P→D transfer
  lane 压力。
- KV transfer 已持续排队，但本轮固定配置不混入网络开关。完成 sweep 后可单独 A/B
  rank-to-NIC mapping 或 lane 调度，并回退无收益变化。

## AgentX C48

- 配置：`CONC=48`，P/D max running 与 CUDA graph max batch 均为 48
- 本节失败诊断源：`agentx/c48/bench/`；无正式 aggregate
- Prefill `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：0 / 0.70
- launch 耗时：872.255 秒
- warmup 目标：531
- 失败位置：返回 328、发送 361、在途 33、AgentX 记录错误 0
- 未进入 1200 秒 profiling，因此无有效 QPS、吞吐或延迟结果

现象与根因：51 条 mandatory primer 已全部越过，证明 P8 相比 P4 解除了 primer
阶段的 C48 容量限制；但 cache-pressure warmup 中 Prefill DP5 在 token usage 0.21、
queue 4、pending token 1,003,351 时执行动态 extend，出现
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、0 MB 可用显存并 fatal abort。此后 Prefill
`/health` 超时，进程表缺少 DP5；Decode 仍健康。DP5 退出后 AgentX 因在途请求不返回
停在 328/531。

OOM 前约 2 分钟，Prefill 还在 `ionic_0/1/3` 观察到 Mooncake
`transport retry counter exceeded`。这些错误可能加重资源滞留，但 OOM 现场明确位于
Prefill 动态 forward，且失效进程为 Prefill DP5，因此直接容量根因不是 Decode。

处理：终止无效 benchmark，使用匹配 C48 参数 stop，确认 HBM 归零并保留
`agentx/c48/{bench,stop-failed}`。scratch=0 + mem=0.70 已通过 C16/C32，但仍不足以
覆盖 C48 cache-pressure 的 per-rank 长上下文峰值；按约定不降低更多 mem fraction、
不改变 chunk 或增加第三个调试开关。随后以完全相同的 scratch/mem 配置 fresh launch
C64，确认更高并发边界；若 C64 同类失败，则更高点不重复已知必败的长时间运行。

瓶颈结论：**C48 的硬阻塞是 Prefill 单 rank 动态 forward 峰值显存，KV transfer
有并发错误但不是已证实的直接 fatal 根因。** 该点不能用部分 warmup 伪造性能。

后续优化方向：需要增加 Prefill 副本来分摊每 rank 的并发长上下文，而不是增加
Decode。若扩充 Prefill 后 OOM 消失，再单独检查 Mooncake retry 和 C32 已出现的
transfer queue；本轮固定 1P8:1D8 配置不继续叠加开关。

## AgentX C64

- 配置：`CONC=64`，P/D max running 与 CUDA graph max batch 均为 64
- 本节部分窗口诊断源：`agentx/c64/bench/`；不作为正式性能值
- Prefill `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：0 / 0.70
- Decode `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：1 / 0.85
- launch 耗时：849.945 秒
- warmup：707/707 完成，错误 0，耗时 1,356.42 秒
- profiling 发送/完成：1,099 / 1,052
- profiling 成功/错误：946 / 106
- profiling 错误率：106/1,052 = 10.076%，超过 10% 门禁
- 有效统计窗口：588.521 秒，未达到目标 1,200 秒
- 部分窗口平均 QPS：1.60922
- 部分窗口总吞吐：169,198.62 tok/s
- 部分窗口每 GPU 总吞吐：10,574.91 tok/s
- 部分窗口输入/输出吞吐：167,953.69 / 1,244.93 tok/s
- 部分窗口 P50/P90/P95 TTFT：3.12598 / 25.14364 / 42.20959 秒
- 部分窗口 P50/P90/P95 E2E：8.50484 / 42.87740 / 61.97069 秒
- 部分窗口 P50/P90/P95 ITL：9.48 / 14.41 / 17.10 毫秒
- 部分窗口 P90 interactivity：69.38 tok/s/user
- GPU cache hit rate / 理论值：88.862% / 95.689%
- 窗口末 GPU KV cache 使用率：99%
- GPU KV cache capacity：42,717,184 token

该点不能作为正式性能值：AIPerf 因所有 lane credit 提前结束，只得到 588.5 秒有效
窗口；同时 106 个 profiling 请求只返回 usage/metadata 或空内容，错误率刚好超过
10% 门禁。汇总 JSON 的 `records_error_dropped=139` 与门禁使用的
`106/1052`、`num_requests_successful=946` 不一致，属于产物记账问题；本报告以门禁
分母和成功数为准，并保留该差异。

### C64 瓶颈判断与下一步

结论：**部分窗口已经同时触及 Prefill/KV 容量和传输压力，且结果层错误成为硬门禁；
Decode 不是首先出现的容量故障侧。**

- C64 完整通过 707 条 warmup，说明 C48 的单 rank OOM 具有请求落 rank 和动态峰值
  相关的非单调性，不能仅凭并发数推断每一点必然失败。
- 窗口末 GPU KV 使用率达到 99%，P90 TTFT 升至 25.14 秒；实时采样中 transfer
  waiting queue 曾达到 10，Prefill queue 也持续非零，KV 容量与 P→D transfer
  已共同施压。
- `server_metrics_export.csv` 中两端各 rank 的
  `sglang:num_transfer_failed_reqs` 导出 total 合计为 198，与 106 个空响应接近
  “同一次 KV handoff 由 Prefill/Decode 两侧分别计数”的两倍关系。该证据不是
  request-id 级一一对应，但将空响应的首要嫌疑进一步收敛到 SGLang PD/Mooncake
  transfer 失败及其错误返回路径。
- P90 ITL 为 14.41 毫秒，较 C32 的 11.83 毫秒变差，但没有 Decode 进程退出或
  Decode OOM 证据；优先级仍是扩充 Prefill、降低单 rank 长上下文压力，再评估
  transfer lane。
- 部分吞吐仅可用于诊断，不能加入有效点 Pareto。完成本轮 sweep 后应以完全相同配置
  fresh launch 复测 C32/C64 的空响应是否可复现，不通过过滤隐藏错误。

## AgentX C96

- 配置：`CONC=96`，P/D max running 与 CUDA graph max batch 均为 96
- 本节失败诊断源：`agentx/c96/bench/`；无正式 aggregate
- Prefill `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：0 / 0.70
- warmup 目标：1,061
- 失败位置：返回 71、发送 101、在途 30、AgentX 记录错误 0
- 未进入 cache-pressure warmup 和 1200 秒 profiling，无有效性能值

现象：mandatory primer 推进到 71/101 后连续约 5 分钟不再返回。Prefill 容器仍为
running，但进程表缺少 DP0 scheduler；Decode 8 个 scheduler 均存活且空闲。

Debug：Prefill 日志明确记录 DP0 动态执行时
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem : 0 MB` 和
`Fatal Python error: Aborted`。现场 GPU1--GPU7 均为 100% 利用率，单卡已用显存约
228--307 GB；GPU0 因 DP0 退出只剩约 10 GB。由此排除 AgentX 客户端停发和 Decode
故障，确认直接根因仍为 Prefill 单 rank 动态 forward 峰值显存。

处理：主动终止无效 benchmark，以完全匹配 C96 参数执行 stop，确认 HBM idle 后才
fresh launch C128；不改变 scratch、静态显存比例、chunk 或网络开关。

瓶颈与优化：**C96 是 Prefill 显存容量失败，不是 Decode compute 瓶颈。** 后续若要
提升该区间，应增加 Prefill 副本来分摊超长 primer；降低 Decode 比例或继续增加
单 worker max-running 不能解决单 rank OOM。

## AgentX C128

- 配置：`CONC=128`，P/D max running 与 CUDA graph max batch 均为 128
- 本节部分窗口诊断源：`agentx/c128/bench/`；不作为正式性能值
- Prefill `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：0 / 0.70
- Decode `HSA_NO_SCRATCH_RECLAIM` / 静态显存比例：1 / 0.85
- launch 耗时：902.095 秒
- warmup：1,422/1,422 完成，错误 0
- profiling 完成：128，其中成功 106、空响应错误 22
- profiling 错误率：22/128 = 17.188%，超过 10% 门禁
- 有效统计窗口：83.797 秒，未达到目标 1,200 秒
- 以下均为部分窗口诊断值，不是正式性能值：
  QPS 1.28049，总吞吐 53,431.47 tok/s，每 GPU 3,339.47 tok/s
- 部分窗口 P50/P90/P95 TTFT：11.10275 / 18.23579 / 22.73766 秒
- 部分窗口 P50/P90/P95 E2E：17.04228 / 25.05216 / 32.66554 秒
- 部分窗口 P50/P90/P95 ITL：8.72 / 9.43 / 9.59 毫秒
- 部分窗口 GPU cache hit / 理论值：62.348% / 80.773%

现象与 Debug：warmup 曾停在 1,284/1,422、19 条在途约 9 分钟。Prefill/Decode
各 8 个 scheduler 均存活、健康检查通过、日志无 OOM，但此前连续出现多路 Mooncake
`transport retry counter exceeded`；等待后 warmup 自行恢复并完整通过。profiling
随后仅形成 83.8 秒有效窗口便耗尽 lane credit，22 条请求只返回 usage/metadata 或
空内容并触发错误门禁。汇总 JSON 的 `records_error_dropped=114` 与 profiling
门禁的 22 条错误不一致，和 C64 一样存在聚合记账口径问题。内部
`sglang:num_transfer_failed_reqs` 导出 total 合计为 43，几乎等于 22 个空响应的
两倍，进一步支持同一个失败 handoff 在 P/D 两侧各计一次的解释。

瓶颈结论：**warmup 的长停顿直接指向 KV transfer 可用性；正式阻塞是结果层空响应
错误，而不是 Decode compute。** 部分窗口负载和 cache hit 均未进入稳态，不能据其
较低 ITL 或吞吐推断真实 C128 性能。后续应先修复/复现空响应和 transfer retry，再
讨论 P:D；若只针对资源侧扩容，优先从 1P8:1D8 改为 2P8:1D8 分摊 Prefill 与传输
lane，暂无增加 Decode 的证据。

## AgentX C160

- 配置：`CONC=160`，P/D max running 与 CUDA graph max batch 均为 160
- 本节失败诊断源：`agentx/c160/bench/`；无可信完整 aggregate
- Prefill/Decode 显存配置保持 0/0.70 与 1/0.85
- launch 耗时：903.000 秒
- warmup：1,776/1,776 完成，错误 0，耗时 4,937.76 秒
- profiling 在线失败点：18/171 空响应，错误率 10.5%
- profiling 约 108 秒即由 AIPerf `failed_request_threshold` 主动取消
- 未生成可信的 1,200 秒 aggregate，无正式 QPS、吞吐和延迟值

失败前 88 秒实时诊断值为：平均约 1.3 RPS，输入/输出吞吐约
55,048/685 tok/s，P50/P95 TTFT 13.879/34.300 秒，P50/P95 ITL 8/10 毫秒；
Prefill realtime queue 约 2 running，transfer waiting queue 约 10。该短窗口仅用于
定位，不能与完整 C8/C16/C32 横向比较。

瓶颈结论：**空响应错误率是硬门禁，Prefill queue 与 KV transfer 是资源侧共同
压力；Decode ITL 在失败前仍低，不能据此要求增加 Decode。** 优化顺序应为：
先复现并修复空响应，再用 2P8:1D8 分摊 Prefill/transfer；只有在 Prefill 和 transfer
压力下降后 Decode queue/ITL 开始恶化，才增加 Decode 比例。

## AgentX C192

- 配置：`CONC=192`，P/D max running 与 CUDA graph max batch均为 192
- 本节失败诊断源：`agentx/c192/bench/`；无正式 aggregate
- Prefill/Decode 显存配置保持 0/0.70 与 1/0.85
- launch 耗时：914.084 秒
- warmup 目标：2,128
- 失败位置：返回 146、发送 208、在途 62、AgentX 记录错误 0
- 未进入 cache-pressure warmup 和 profiling，无有效性能值

现象与 Debug：mandatory primer 在 146/208 后连续约 5 分钟无返回。Prefill 日志
记录 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem : 0 MB` 和
`Fatal Python error: Aborted`；进程表缺少 Prefill DP5，而 Decode DP0--DP7 全部
存活。由此确认是 Prefill DP5 动态 forward 峰值显存耗尽，不是客户端或 Decode
故障。

处理：主动终止无效 benchmark，以完全匹配 C192 参数停止客户端、router、Prefill、
Decode 和 etcd；不引入新开关，失败现场保留于 `agentx/c192/{bench,stop-failed}`。

瓶颈结论：**C192 是 Prefill 单 rank 显存容量失败。** 当前 1P8:1D8 无法承载该
并发和长上下文分布；应优先增加 Prefill 副本/节点并分摊请求，而不是扩大单 worker
max-running 或增加 Decode。

## 跨并发结论

- 可形成完整 1,200 秒结果的点只有 C8、C16、C32；C32 虽低于门禁，但有 8.674%
  空响应，只能标为“有效 goodput、服务质量退化”。
- C48、C96、C192 在不同 warmup 阶段发生 Prefill 单 rank OOM；其非单调出现说明
  是否命中取决于超长 trace 到 DP rank 的分配和动态 shape 峰值，不能把总 GPU
  显存当成单调容量。
- C64、C128、C160 均因空响应达到或超过 10% 而无效；其中 C64/C128 还提前耗尽
  lane credit，未形成 1,200 秒窗口。
- 以可用性和结果质量为先，当前推荐 C16；以有效 goodput 上限观察，C32 最高但必须
  复测空响应。C48 以上不具备可上线的有效结果。
- 资源优化优先级为 Prefill 副本/动态显存余量 → KV transfer 稳定性 → Decode。
  当前没有增加 Decode 比例的证据；高并发下一步拓扑应优先验证 2P8:1D8。

## 问题、调试和解决方案

### 1. C8 warmup 的 Prefill DP7 显存资源耗尽

现象：P8:D8 C8 fresh launch 成功。warmup 的 7 条 primer 全部返回，cache-pressure
阶段推进到返回 77/87、发送 78、在途 1、错误 0 后不再前进。Prefill `/health`
随后超时，Decode 容器仍存活。

Debug：Prefill 日志先在 18:04:33 记录 `ionic_3` RDMA
`transport retry counter exceeded`；对应 transfer 在约 30 秒后失败，Decode 同时
记录“可能是 Prefill 已死”的对端错误。18:05:55，Prefill DP7 在仍有 175,635
pending token 时记录 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`、
`Available Free mem : 0 MB` 和 fatal abort，栈位于
`forward_absorb_rocm_prepare` 的 Prefill extend。现场进程表只有 DP0--DP6，
DP7 已退出；GPU7 仅剩 3% VRAM，其他 Prefill GPU 为 87%--95%。Decode 八个
scheduler 进程全部存活。因此最终停滞由 Prefill DP7 OOM 直接造成；RDMA 错误先于
OOM，可能增加资源滞留，但现有证据不足以把二者判为同一根因。

处理：终止无效 benchmark，只清理本轮 P8:D8 容器并保留
`agentx/c8/bench/runner.log` 与服务日志。下一次 fresh launch 只把现有
`PREFILL_MEM_FRACTION` 从 0.85 降到 0.70，其他参数不变，以同时验证额外动态显存
是否能覆盖长请求，以及 fresh Mooncake session 是否消除 RDMA 错误。若仍失败，
回退该开关，不叠加其他变化。

mem=0.70 A/B 结果：warmup 推进到返回 73/87、发送 76、在途 3 后，DP7 再次以
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、0 MB 可用显存退出。此时没有再次观察到
RDMA transport retry，且其他 rank 仅使用约 71%--88% VRAM，说明静态 KV 比例不是
充分根因；该开关已回退，失败产物保存在 `agentx/c8/bench-mem070/`。

进一步 Debug：两次 OOM 都发生在动态 shape 的 Prefill extend forward；现有
`engine.sh` 固定向容器传递 `HSA_NO_SCRATCH_RECLAIM=1`。仓库现有
`rocm-llm-bench/third_party/InferenceX/benchmarks/multi_node/amd_utils/env.sh`
则在 OOR 处理处明确设置 `HSA_NO_SCRATCH_RECLAIM=0`。因此下一次只 A/B 该开关为
0，让 ROCm 回收动态 scratch；`PREFILL_MEM_FRACTION` 回到 0.85。首次命令检查时
发现全局覆盖会同时改变 Decode，因此在服务尚未健康、benchmark 尚未运行前主动终止
并清理。随后把覆盖隔离为仅 Prefill=0、Decode 仍为基线 1，确保 A/B 只改变故障侧。
若失败则回退，不再叠加变化。

scratch reclaim A/B 结果：完整 warmup 87/87 和 profiling 337/337 均成功，
错误 0；此前两次失败的 73/87、77/87 位置均已越过，Prefill 未再 OOM。
因此确认仅在 Prefill 开启 scratch reclaim 可以解决动态 extend forward 的 OOR；
后续 P8:D8 点保持 Prefill=0、Decode=1，不叠加其他开关。

### 2. Scratch A/B 启动期间节点被外部任务占用

现象：完成失败栈清理并发起 Prefill-only scratch reclaim fresh launch 后，
137/138 分别出现外部 `glm52-c14q-137-control` 与
`glm52-c14q-138-control`，占用 GPU0--3；本轮 P8 需要两节点全部八卡。

Debug：`launch.sh` 的 HBM idle gate 在创建本轮 etcd、worker 或 router 前检测到
最高 85% VRAM，持续输出 `GPUs busy ... waiting`。Docker 名称与本轮
`glm52-pd-p8dpa-d8dpa-v518-*` 前缀不同，确认不是清理残留。

处理：不停止、不修改外部容器；保留启动命令等待节点释放。等待期间继续更新报告，
不改实验参数。

### 3. C16 在 scratch reclaim 开启后仍发生 Prefill DP4 OOM

现象：C16 fresh launch 成功，P/D 和 router 均健康，耗时 818.116 秒。AgentX
warmup 的 17 条 primer 返回 14 条后停在返回 14/177、发送 17、在途 3、错误 0。
Prefill `/health` 超时，Decode `/health` 正常；两个容器均仍显示 running。

Debug：直接检查 Prefill 日志发现 DP4 处理一条初始 pending token 约 528,037 的
primer；分块推进约 471,701 token、剩余 56,336 token 时，HSA 报
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem : 0 MB`，随后
`Fatal Python error: Aborted`。容器进程表只剩 DP0--DP3、DP5--DP7，确认 DP4
scheduler 已退出。日志中未出现 Mooncake transport retry，因此 warmup 停滞的直接
原因仍是 Prefill 动态 extend 的显存资源耗尽，而不是 Decode 或 KV transfer。

处理：终止无效 benchmark，使用匹配参数执行 stop，确认两节点 HBM 归零；失败产物
保留在 `agentx/c16/bench/` 和 `agentx/c16/stop-failed/`。scratch reclaim 已在 C8
完整解决同类 OOM，但单独使用不足以覆盖 C16 更长的 primer。下一次保持已经验证有效
的 Prefill scratch=0，只把 Prefill 静态 KV 比例从 0.85 降至 0.70，测试“回收动态
scratch + 预留更多动态 HBM”的交互；Decode 和其他参数不变。mem=0.70 曾在
scratch=1 时失败，因此本次用于判断它是否只有在 scratch 可回收时才有效。若仍失败，
立即回退 mem=0.70，不叠加第三个变化。

交互 A/B 中间结果：17/17 mandatory primer 已全部返回，成功越过基线 DP4 OOM
位置；完整 warmup 177/177、错误 0，耗时 682.13 秒。已进入 1200 秒 profiling，
前 2.5 分钟完成 48/48、错误 0。最终结论仍以完整计分窗口为准。

### 4. C32 空内容响应比例升高

现象：C32 共出现 150 条 `InvalidInferenceResultError`，其中 warmup 10 条、
profiling 140 条。错误消息均为服务只返回 usage/metadata、null/空 data 或
`[DONE]`，没有实际 completion 内容；profiling 错误率 8.674%，低于既定 10%
门槛，因此结果有效但必须保留披露。

Debug：逐条检查 `profile_export.jsonl` 确认错误类型一致；错误请求有合法 request
起止时间和 token usage，不是 HTTP 连接失败。整个窗口 P/D/router 保持健康，
AIPerf 观察到的 HTTP 传输级错误为 0，TTFT/ITL 覆盖率均为 100%，排除服务进程退出
或客户端 HTTP 流中断；这不能排除引擎内部 KV transfer 失败。后续复核
`server_metrics_export.csv` 发现 C32 两端各 rank 的
`sglang:num_transfer_failed_reqs` 导出 total 合计为 275；C32 全阶段空响应为 150，
与 C64 的 198/106、C128 的 43/22 一样，均接近 P/D 两侧各计一次的两倍关系。
这是聚合相关性而非 request-id 级证明，但强烈指向 SGLang PD/Mooncake transfer
失败后仍以 HTTP 200 结束、只输出 usage/metadata/`[DONE]` 的错误处理路径。

本轮 `launch.sh` 默认使用 Rust router；其 HTTP PD streaming 路径透传 Decode
response bytes，不解析或删除 completion 内容，因此 Infera router 主动剥离内容的
可能性较低。AIPerf 将没有 `content`、`reasoning_content` 或 tool-call delta 的
完整 SSE 判为 `InvalidInferenceResultError` 符合预期。仍需用同一 request-id 关联
原始 Decode SSE、bootstrap room、P/D rank 和 transfer 结果，才能最终区分
SGLang/Mooncake 本体故障与 Infera 的 rank/handoff 路由触发因素。

处理：按用户要求不新增脚本、不为低于门槛的错误修改引擎或 benchmark，也不通过过滤
隐藏错误；保留 140 条错误进入门禁分母并继续固定配置 sweep。若后续点超过 10%，由
现有 AIPerf live/final gate 直接判失败，再基于同一套产物定位，不预先引入开关。

### 5. 空响应根因专项调查

目标：区分静态 RDMA 配置、真实链路/QP 故障、Infera P/D rank 路由和 SGLang
failed-session 放大机制，并修复到 C32/C64 可形成无空响应的完整有效窗口。所有诊断
继续使用现有 `preflight.sh`、`launch.sh`、`agentx_bench.sh` 和 `stop.sh`；每个
A/B 均 fresh launch，不复用前一点 prefix cache。

#### 5.1 P8 全 rail 静态 WRITE 预检

历史 P4:D8 阶段的非对称预检只要求 4 条公共 GPU 路径。本次以 P8:D8 参数重新运行
完整预检，实际覆盖 137→138 的 8 条 GPU WRITE 路径，耗时 512.855 秒，全部通过。
两端 `ionic_0`--`ionic_7` 均为 ACTIVE，GID index 1 均存在可路由 RoCE GID。

结论：镜像缺少 Mooncake、GID index 固定错误、GPU memory registration 全局失败、
某条同 rank rail 静态不通均可排除。该预检是逐 GPU 的短传输，不能覆盖持续负载下
的 64 种 P/D rank 组合或 session 生命周期，因此仍需真实 AgentX 复现。

产物：
`debug-empty-response/preflight-p8-all8/validation.json`，结果为
`PASS: 1 P->D edges, 8 GPU WRITE paths per edge`。

#### 5.2 镜像内 SGLang failed-session 行为

直接检查运行镜像中的 `MooncakeKVManager.transfer_worker`，确认发送
`ret != 0` 后 `session_failures[session]` 立即加一，阈值为 1，随后把整个
`mooncake_session_id` 放入 `failed_sessions`。同一 session 后续请求在真正调用
Mooncake 前便走 early failure。镜像支持
`SGLANG_ENABLE_FAILED_SESSION_PROBE`，但当前容器未设置且镜像默认值为 false；
probe interval 默认 30 秒。

这解释了“一个瞬时 RDMA 失败为何变成持续空响应”：首次真实 send 失败会污染整个
session，直到 probe、重新注册或重启。它是故障放大器，尚不是首次 RDMA 失败的原因。

#### 5.3 fresh C32 基线的第一次真实失败

数据与日志源：`debug-empty-response/baseline-c32/`。该 600 秒运行只用于根因诊断，
不作为最终性能点。

以原 P8:D8、KV-aware、C32 参数 fresh launch，并运行 600 秒诊断窗口。在 warmup
cache-pressure 阶段抓到本轮第一次底层故障：

- 12:36:17，Prefill DP7/TP7 的 Mooncake WRITE 返回
  `transport retry counter exceeded`；失败 slice 为 65,536 bytes，
  `local_nic=ionic_7`，`peer_nic=10.245.157.237:15249@ionic_7`，
  `max_retry_cnt=9`。
- 12:36:52，Prefill DP7 将 session `10.245.157.237:15249` 标记为 failed；
  request bootstrap room 为 `7575550042809148471`，其模 8 为 7，与 Prefill
  DP7 一致。
- 同时 Decode 日志确认相同 bootstrap room 实际落在 Decode DP0，并报
  `Failed to get kvcache from prefill instance`。因此这是明确的
  **Prefill DP7 → Decode DP0 跨 rank handoff**，不是同 rank 请求。
- 故障前后 137 的 `ionic_7` 硬件计数增量为
  `req_tx_retry_excd_err +2`、`tx_rdma_ack_timeout +5`、
  `tx_rdma_retx_pkts +173,663`；138 的 remote-access、out-of-buffer 和
  invalid-packet 计数仍为 0。该组合符合发送端收不到 ACK 而重传耗尽，不符合错误
  rkey/越界写导致的 remote-access error。

Mooncake 的 destination-device affinity 正确选择了同名 `ionic_7→ionic_7`，
但目标 KV 属于 Decode GPU0；同名 rail 连通与目标 GPU 本地性不能同时满足。当前
KV-aware policy 独立选择 Prefill/Decode rank，因而会产生这类跨 rank 组合。

完整 600 秒结果为 1,326 条记录，其中 354 条 warmup、51 条
`InvalidInferenceResultError`；profiling 为 42/972 = 4.321%。服务 metric 汇总到
Prefill 39 次、Decode 40 次 transfer failed。直接以 Decode 日志中的
`bootstrap_room % 8` 恢复 Prefill rank，可得到 48 条保全在容器日志内的失败：

- P0→D7：2 次；
- P2→D3：8 次；
- P4→D3：7 次；
- P7→D0：31 次；
- 同 rank 失败：0 次。

同期 router 完整日志中共有 1,328 对 P/D pick，153 对同 rank、1,175 对跨 rank。
失败只出现在跨 rank；尤其 P7→D0 一旦 session 被拉黑后失败 31/32，展示了
“第一次 QP retry exhausted → session blacklist → 此后几乎必败”的放大过程。
Prefill 硬件计数在整段负载上的增量也不局限于单 rail：8 条 rail 中 7 条出现
`req_tx_retry_excd_err` 增长，说明真实长序列并发会在多个跨 rank 组合上触发，
不是 `ionic_7` 永久坏链。

下一步单变量 A/B：关闭 KV-aware，保留模型、Mooncake、MTP、显存、batch 和 C32
不变。Rust round-robin 对 P/D 两个等长 rank pool 分别计数，fresh launch 后会形成
DP0→DP0、…、DP7→DP7 的同 rank handoff。若 transfer retry 和空响应同时归零，
即可确认首因是跨 rank Mooncake rail/GPU affinity，而非模拟接受或 AIPerf。

#### 5.4 same-rank round-robin 单变量 A/B

数据与日志源：`debug-empty-response/round-robin-c32/`。该 600 秒运行是可用性 A/B，
不作为最终部署性能值。

除 `ENABLE_KV_AWARE=0` 外参数与 5.3 相同。Router 日志验证 P/D 两个独立 RR 序列
均按 0,1,…,7 循环，按各自 pick 序号形成 P0→D0、…、P7→D7；不存在跨 rank
handoff。

结果：

- 完整 warmup 354/354 成功，错误 0；由于关闭 cache-aware 路由，warmup 耗时约
  55 分钟，这也提供了显著长于问题基线的持续压力窗口。
- 600 秒 profiling 共 346/346 成功，
  `InvalidInferenceResultError=0`，空响应率 0%。
- Prefill 日志中 `transport retry counter exceeded=0`、
  `Session ... failed=0`、`Prefill transfer failed=0`。
- 8 条 HCA 的 `req_tx_retry_excd_err` 在整次 warmup + profiling 前后完全不变；
  `tx_rdma_ack_timeout` 也完全不变。
- profiling 超时收口时，AIPerf 主动 abort 尚在飞行的请求；Decode 会把这些
  `Aborted by AbortReq` 也计入 `sglang:num_transfer_failed_reqs`。因此该 metric
  单独看并不等价于 RDMA 错误。它们集中发生在 14:22:00--14:22:03，客户端错误仍为
  0，也没有 Prefill send/session failure。后续判定真实故障必须联合日志中的异常
  原因和客户端错误，不能把正常 benchmark cancellation 算成链路故障。

代价也符合预期：GPU prefix cache hit 从 KV-aware 基线的 93.2% 降到 46.1%，总
throughput 从 145,916 降到 42,357 token/s。故“永久改成 round-robin”虽然稳定，
却丢失约 71% 吞吐，不能作为性能搜索的最终修复。

该 A/B 同时改变的只有 rank 配对方式和其必然带来的 cache locality；它让首次真实
QP failure、session blacklist 和空响应三者同时归零。结合 5.3 的逐请求 rank
证据，根因判定为：**Infera KV-aware 分别选择 Prefill/Decode DP rank，而当前
对称 P8:D8 多 rail Mooncake 部署要求 P/D rank 共置；跨 rank 会让同名 rail 与目标
GPU affinity 冲突，并在长序列并发下触发发送端 ACK timeout/retry exhausted。**

#### 5.5 修复设计

不能直接关闭 KV-aware。根因修复与恢复增强分开处理：

1. Rust router 增加 opt-in `PD DP-rank affinity`。仍由 KV-aware 从全部 Prefill
   rank 中选择缓存命中最优者；选定 P rank 后，只在相同 effective DP rank 的
   Decode target 中执行 Decode policy。若目标池不存在该 rank，返回显式 503，
   不静默退化成不安全的跨 rank。
2. benchmark 的对称 DPA 配置默认启用该 affinity。SGLang 已有的
   `SGLANG_ENABLE_FAILED_SESSION_PROBE=1`（30 秒）是独立的恢复增强：它只负责在
   其他瞬时故障后解除永久 session blacklist，不能消除首次故障，也不能与 affinity
   同时开启来证明 affinity 的因果效果。最终 affinity A/B 固定 probe=0；probe
   另行验证。

实现覆盖 rank-multiplexed worker、独立 DP worker 和单 rank worker 的 effective
rank；普通 mixed 路由及未启用 affinity 的其他 PD 拓扑保持原行为。新增 functional
test 会先故意把 Prefill RR counter 推进到 P1、让 Decode 仍停在 D0，再验证 affinity
把实际双 leg 请求约束为 P1→D1。

#### 5.6 严格单变量验证

严格 control 数据源：`debug-empty-response/paired-control-c32/`；严格 treatment
数据源：`debug-empty-response/paired-treatment-c32/`。两者用于验证 affinity 因果，
不与 1,200 秒正式性能点混算。

为避免把显存、probe 或镜像变化混入结论，最终 C32 对照固定为同一新镜像、
Prefill/Decode mem=0.70/0.85、KV-aware=1、probe=0、相同随机种子和 600 秒诊断窗口；
唯一变量是 `PD_DP_RANK_AFFINITY`。

此前 affinity=1、probe=1 的首次尝试不作为最终 A/B：前 280 对 pick 虽然全部同 rank，
且未出现真实 transfer failure，但 Prefill DP7 在 warmup 266/354 后发生
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`，未进入 profiling。该现场只说明修复路径没有
在 OOR 前产生空响应，不能证明完整窗口通过。随后曾准备以 Prefill mem=0.60 调查
OOR，但在服务健康和 benchmark 开始前即发现它会改变 KV cache 容量及 transfer
负载，立即取消并清理；该轮没有请求，也不进入任何对比。

严格 control（affinity=0）结果：

- warmup 354/354、错误 0，耗时 837.67 秒；
- profiling 仅完成 84 条即出现 9 条空响应，9/84=10.714%，触发 10% 在线门禁；
- aggregate 共保留 438 条记录，其中 354 条 warmup、75 条 profiling success、
  33 条 `InvalidInferenceResultError` error-dropped；门禁的实时完整分母仍以
  9/84 为准，部分 aggregate 不能作为性能值；
- router 记录 445 对 P/D pick，其中 377 对跨 rank、68 对同 rank；
- Prefill 记录 9 次底层 `transport retry counter exceeded`、3 个 session 被拉黑、
  29 次 `Prefill transfer failed`；Decode 保全 29 次对应失败，rank 组合为
  P2→D4 1 次、P5→D3 6 次、P6→D2 11 次、P6→D3 11 次，同 rank 失败仍为 0。

该 control 使用已包含新代码的镜像但关闭 affinity，证明新镜像未在默认路径中意外
消除原故障。严格 treatment 保持以上所有参数不变，只开启 affinity；最终结论以其
完整 warmup、profiling、P/D 配对、底层日志和 HCA counter 为准。

严格 treatment（affinity=1）结果：

- warmup 完成 354 条，其中 353 成功、1 条空内容；600 秒 profiling 为
  581/581 成功，空响应率 0%，完整统计时长 629.225 秒；
- router 的启动配置明确记录 `pd_dp_rank_affinity: true`；全程 949 对 P/D pick
  全部同 rank，跨 rank 为 0；
- Prefill/Decode 日志中的 `transport retry counter exceeded`、session failed、
  Prefill/Decode transfer failed、OOR 和 fatal error 均为 0；
- Prefill 8 条 HCA 的 `req_tx_retry_excd_err` 和 `tx_rdma_ack_timeout` 前后完全
  不变。`tx_rdma_retx_pkts` 仅 `ionic_3 +402`、`ionic_6 +1,749`，没有升级为
  ACK timeout、retry exhausted 或应用层失败；Decode 的 error/RNR counter 也无增长；
- profiling 诊断吞吐为 81,057.35 token/s，GPU cache hit 94.221%。control 只形成
  56.1 秒部分窗口且被错误门禁终止，因此不拿两者作性能 A/B；本实验只判断可用性。

唯一 residual 是 warmup 的固定超长 primer
`conversation_id=117ebe75819d050f308a0a81647893abd02d`、turn 22：
输入 237,161 token，约 249.855 秒后收到 HTTP 200 但无实际 content。它在同一 treatment
的 Prefill DP4→Decode DP4 handoff 中发生，期间没有 transfer/session/HCA 故障；
同一 trace/turn 在
此前 baseline、round-robin、首次 fixed、strict control 以及原 C32--C192 运行中均
正常返回 1 token。因而该 1/354=0.282% warmup-only、不可稳定复现的输出校验错误不支持
“rank-affinity 仍有 transfer 漏洞”的解释，也不影响 profiling 性能统计；报告仍保留，
不把它过滤或计作 0。原问题所对应的 profiling 空响应和 KV transfer 故障已经同时归零。

### 6. Prefill OOR 根因深化与实验顺序

#### 6.1 已证实范围

C8、C16、C48、C96、C192 的失败栈均位于 Prefill 动态 extend forward，并伴随
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem : 0 MB` 和单个 DP
scheduler fatal abort；Decode 未退出。仅把静态 KV 从 0.85 降到 0.70 不能解决 C8，
Prefill-only `HSA_NO_SCRATCH_RECLAIM=0` 能解决 C8，但与 mem=0.70 组合后仍在
C48/C96/C192 失败。由此已排除“只需再重复验证 mem fraction”这一低效路径。

运行镜像为 `hsa-rocr 1.18.0.70200-43`、`rocm-core 7.2.0.70200-43`，设备为
gfx950 MI355X。主机 KFD topology 显示每卡 `num_xcc=8`；`rocminfo` 显示每卡最多
128 条 queue。ROCr 7.2.0 的官方文档和同版本源码给出：

- `HSA_NO_SCRATCH_RECLAIM=0` 并不表示所有 scratch 用后立即释放；低于阈值的
  scratch 仍会永久绑定 queue，只有超过阈值的分配使用 scratch-use-once；
- gfx950 的 CP firmware 达到最低版本时支持 asynchronous scratch reclaim；
- `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` 默认是每 XCC 3 GiB，8 XCC 即每 queue
  24 GiB；显式值是整卡 byte 数，最大为每 XCC 4 GiB；
- ROCr 源码明确说明 single-use scratch 有显著 allocation latency，因此不能未经
  性能 A/B 就把阈值降到极小值；另一方面，保留型 scratch 可能在多个 queue 间累加，
  与 PyTorch/AITER 动态 GEMM 的大 shape 共同耗尽 SGLang 预留的动态 HBM。

来源：
[ROCr 1.18 环境变量](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-7.2.0/api-reference/environment_variables.html)；
[ROCr 7.2 `amd_gpu_agent.cpp`](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp)；
[ROCr 7.2 `amd_aql_queue.cpp`](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp)。

目前最强假设是：现有 reclaim 已启用，但默认 24 GiB/queue 阈值仍允许多个大 scratch
长期驻留；遇到下一次动态 shape 扩容时，ROCr 连一 wave/CU 所需的最小 scratch 都无法
分配，才向上抛出 OOR。它符合“mem=0.70 尚有约 30% 动态余量仍被吃光”和失败 rank
不固定的现象，但在受控实验完成前仍标为假设。

#### 6.2 避免重复基线的验证顺序

不重跑旧镜像或 affinity=0 的 C48。先直接执行最终配置的 C48：保持
scratch=0、mem=0.70、async threshold 未设置，只加入已经独立验证的 rank affinity。
这一轮同时承担两个目的：

1. 若在旧 C48 失败位置附近再次 OOR，它就是 async-threshold A/B 的同镜像 control；
2. 若完整通过 warmup 和 1,200 秒 profiling，它直接成为 C48 的正式重测，不再另跑
   一次；随后在更易触发的 C96 建立 OOR control。

只有观测到同类 OOR 后，才把 Prefill 的
`HSA_SCRATCH_SINGLE_LIMIT_ASYNC` 单独设为 8,589,934,592 bytes（每 XCC 1 GiB
等效），Decode 与其他参数全部不变。验收顺序为：越过相同 trace/阶段且无 OOR →
完整标准 warmup/profiling → 与无阈值的相邻有效点比较吞吐、TTFT、HBM 峰值。若稳定
但性能下降，再测试 16 GiB；不通过则依次评估更低静态 KV、减小 chunk、增加 Prefill
副本，并分别量化性能代价，不把多个变化叠在一轮。

#### 6.3 C48 最终配置 control 再现

失败诊断源：`debug-oor/c48-affinity-control/`。

该轮不是重复旧 baseline：它首次同时使用已验证的 rank affinity 和最终
scratch=0、mem=0.70 配置，且原计划若通过便直接作为正式 C48 重测。实际在 mandatory
primer 返回 48/51、发送 51、在途 3、客户端错误 0 时，Prefill DP1 发生同类
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`、`Available Free mem : 0 MB` 和 fatal
abort；进程表只缺 DP1，Decode 仍存活。未进入 cache-pressure 或 profiling，因此无
性能结果。

5 秒 HBM 采样共 150 个样本：八卡在失败前都曾达到 98%--99%，DP1 随 scheduler
退出从 80% 降至 19%、再降至 3%；其他 rank 仍为 77%--91%。live 故障窗口未观察到
Mooncake transport retry。由此确认 rank affinity 虽解决 transfer/空响应，却没有
顺带解决 Prefill OOR；async threshold 的 control 已成立。

下一轮只给 Prefill 设置
`HSA_SCRATCH_SINGLE_LIMIT_ASYNC=8589934592`，其余参数与本 control 完全相同。
8 GiB 是 ROCr 默认整卡 24 GiB 的三分之一、等效每 XCC 1 GiB；它保留正常小 scratch
的 queue reuse，同时让更大的动态 scratch 走 single-use，先验证是否能降低 HBM 峰值
并越过相同 mandatory primer。

#### 6.4 8 GiB async scratch threshold 结果

失败诊断源：`debug-oor/c48-scratch-limit-8g/`。

严格 treatment 已通过容器环境核对：只有 Prefill 新增
`HSA_SCRATCH_SINGLE_LIMIT_ASYNC=8589934592`，仍为 scratch=0；Decode 未设置该变量且
仍为 scratch=1。结果同样停在 mandatory primer 返回 48/51、在途 3，Prefill DP2
OOR 退出，未进入 profiling。

完整日志显示 DP2 每次只处理一个 4,096-token local chunk，失败前
`#running-req=0`、`#queue-req=0`、token usage=0.29、pending token=16,813；最近的大
动态 GEMM shape 包括 M=14,788 和 M=12,288。fatal 主线程位于
`dp_gather_via_all_gatherv → prepare_mlp → eager_runner._execute_extend`，仍是动态
extend forward。Prefill/Decode 均无 transport retry 或 transfer failed。HBM 194
个样本中 DP2 在失败前达到 99%，退出后降至 3%；其余卡峰值 96%--99%。

因此“默认 24 GiB/queue 的 retained scratch 是充分根因”被该 A/B 否定：降至 8 GiB
没有降低最坏 HBM 峰值，也没有越过 control 阶段。它可能仍是显存组成之一，但单独调
该阈值不是有效修复；不继续盲目重复 16 GiB，因为它比已失败的 8 GiB 只会保留更多
scratch。下一优先级转为动态 PyTorch allocation fragmentation。

镜像为 PyTorch `2.9.1+rocm7.2.0.git7e1940d4`。最初一个只检查 allocator backend
名称的短 smoke 没有 warning，不能证明 expandable segments 真正生效；在启动 C48
treatment 后、服务尚未健康时补做了 allocator snapshot。用 256 MiB 实际 GPU
allocation 测得 `is_expandable=False`。该 build 对
`PYTORCH_CUDA_ALLOC_CONF`/`PYTORCH_HIP_ALLOC_CONF` 明确告警
`expandable_segments not supported on this platform`；generic
`PYTORCH_ALLOC_CONF` 则未改变 allocation，仍为普通 segment。

因此本镜像不具备该能力；主线 PyTorch/ROCm 后续源码支持不能外推到当前 vendor
build。立即终止尚未完成加载的服务并清理，未发送 benchmark 请求，避免执行一次
实际上与 control 相同的 C48。临时加入的配置透传也已撤销，不把 no-op 选项伪装成
修复。下一步转向直接量化动态内存组成，并评估最小性能代价的 KV reserve/chunk
边界，不再尝试 allocator alias。

#### 6.5 pressure-aware caching allocator GC

HSA 在 PyTorch 之外申请 scratch；因此即使 PyTorch 缓存中存在已经空闲的 block，
HSA 的申请也不会触发 PyTorch 自己的 allocation-failure
`release_cached_blocks` 路径。`garbage_collection_threshold` 是 native caching
allocator 已支持的独立机制：总占用超过阈值后，在后续新 allocation 时优先释放旧且
空闲的 cached block，避免每 batch 无条件 `empty_cache()` 的同步和重新分配开销。

用非法值 `garbage_collection_threshold:2.0` 的独立进程已在当前镜像准确失败于
`AllocatorConfig.cpp::parseGarbageCollectionThreshold`，证明 generic
`PYTORCH_ALLOC_CONF` 在此 build 中确实被解析；合法 0.8 配置完成实际 GPU
allocation。下一 C48 treatment 只给 Prefill 设置
`PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8`，threshold 仍未设置，
mem=0.70、scratch=0、rank affinity 等均与 6.3 control 相同。0.8 高于服务初始
约 71%--73% HBM，仅在动态压力上升后启动回收；若能通过，再用完整性能窗口判断回收
代价，而不是预设它“零影响”。

启动后补查当前 PyTorch 源码和 SGLang 调用路径，发现还有两个隐藏门槛，故以上 generic
treatment 实际是 no-op：

1. allocator 只有调用 `torch.cuda.set_per_process_memory_fraction` 后才将
   `set_fraction` 置位，GC 分支受该值保护；当前 SGLang 源码没有这项调用；
2. 当前 vendor build 的 generic 配置会校验参数，却没有把 GC threshold 接到 HIP
   allocator；同样参数通过 legacy `PYTORCH_HIP_ALLOC_CONF` 才实际生效。

独立 1+2 GiB cached block、再申请 4 GiB 的定量 probe 证实三者缺一不可：
generic+显式 fraction 发生 1 次 allocator retry 后才释放 2 个 segment；HIP
alias+显式 fraction 为 0 retry、预先释放 2 个 segment；HIP alias 但不设置
fraction 则保留全部 7 GiB、释放 0 个 segment。

该 no-op C48 仍意外越过了此前两次失败的 primer，完成 531/531 warmup、错误 0，并
进入 profiling。这证明 OOR 有时序敏感性，也意味着不能把这次通过归因给 GC；若
profiling 完整通过，它可作为有效性能样本，但不作为修复证明。

该 no-op run 最终完成完整窗口：profiling 1,174/1,174 成功、错误 0，总吞吐
120,055.21 token/s、每 GPU 7,503.45 token/s、GPU cache hit 95.745%，TTFT
p50/p95 为 19.050/85.789 秒。1,342 个 2 秒 HBM 样本中八张卡都至少两次达到 99%，
但 scheduler 始终保持 8/8、无 OOR。export 另外保留了 4 条 warmup-only
`InvalidInferenceResultError`（输入 100,722--590,919 tokens），都发生在
cache-pressure 的 `max_tokens=1` 请求；全程无 transport/session/transfer failure，
profiling 无空响应，故与原跨 rank transfer 故障区分记录。

该 no-op 对照的数据源为 `debug-oor/c48-allocator-gc080/`；它用于机制和性能代价
对照，不作为最终修复结果。

为测试真实 GC，新增 opt-in 启动 hook：每个 scheduler 在选定自身 GPU 后调用一次
`set_per_process_memory_fraction`；同时给该 role 使用实际有效的 HIP allocator
alias。独立容器 probe 已验证 hook 日志出现，且同一 allocation 序列变为
0 retry/2 segment proactive free。正式 treatment 将用 fraction=1.0（不降低
PyTorch 可用容量）和 threshold=0.8；默认两项都为空，未启用配置行为不变。

真实 GC C48 已完成。容器环境明确为
`PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`、
`INFERA_PYTORCH_MEMORY_FRACTION=1.0`、scratch=0，八个 scheduler 都各自打印一次
fraction hook 生效日志。结果：

- warmup 531/531 完成，耗时 1,024.96 秒；profiling 1,104/1,104 成功、错误 0；
- 总吞吐 116,556.08 token/s、每 GPU 7,284.75 token/s、GPU cache hit
  95.533%，TTFT p50/p95 为 23.402/87.435 秒；
- 1,280 个 2 秒 HBM 样本的全局峰值只有 93%，没有任何 94%--99% 样本；profiling
  结束后的八卡 HBM 为 83%/81%/81%/80%/80%/83%/78%/80%；
- 对比 no-op 的八卡均到 99%、结束后最高仍 98%，真实 GC 明确释放了 inactive
  allocator segment，并给外部 HSA scratch 留出约 6 个百分点的最坏余量；
- Prefill 无 OOR/fatal，P/D 无 transport retry/session blacklist。Decode 只有 1 条
  benchmark 超时取消请求对应的 `Aborted by AbortReq`，不是 RDMA failure；
- export 中 2 条 `max_tokens=1` warmup-only 空内容仍单列；profiling 空响应为 0。

本轮用于 active-GC 机制验收的 C48 数据源为
`debug-oor/c48-allocator-gc080-active/bench/agentx_conc48.json`；server metrics、
逐请求记录和运行日志均在同一轮目录。该服务实际为全局
`max-running/graph-max=8/8`，并使用 digest `34909eb3...`，因此不是严格
C48 intended-commit 性能点。

与相邻 no-op 完整窗口相比，总吞吐低 2.91%，cache hit 低 0.212 个百分点，ITL p50
从 8.71 增至 8.82 ms；两轮实际轨迹输入均值相差 3.4%，所以 2.91% 不是纯 allocator
开销的无偏估计，但足以将代价界定为低个位数而非 8 GiB scratch-use-once 可能带来的
未知高延迟。C48 已同时满足完整窗口和显存机制验收；还需在更强 C96 fresh-launch
压力下验证后，才把它升级为最终 OOR 修复。

在 no-op C48 之后复用同一 stack 做了一次非性能型 C96 压力验证，以避免额外 baseline
launch。开始新请求前，八卡无计算但 HBM 已分别停在
89%/79%/98%/78%/98%/79%/95%/93%，直接证明动态运行后有大量显存未返还给 HSA。
C96 在 `max-running=8` 下持续约 58 分钟，推进到 warmup 995/1,061，错误 0、rank
始终 8/8、无 OOR；随后主动终止。它证明较低 max-running 能降低 OOR 概率，但该点
warmup 长尾严重且不是 fresh launch，不能作为 C96 性能结果；继续等待的边际价值低于
进入真实 GC treatment。

#### 6.6 原始 C96 强度的最终验证

第一次 fresh active-GC C96 沿用了当前配置默认的
`max-running=8`、`cuda-graph-max-bs=8`。它最终完成 1,061/1,061 warmup 和完整
1,200 秒 profiling，813/813 profiling 成功、错误 0，总吞吐 73,797.25 token/s，
HBM 峰值 93%；但 warmup 耗时 5,514.23 秒，且原失败点使用的是 96/96。因此该轮只
证明 GC 在受限并发下稳定，不能冒充原 C96 的严格 treatment。aggregate 保留的 7 条
`InvalidInferenceResultError` 全部属于 warmup，profiling 仍为 0。

上述 max-running=8 混杂样本位于 `debug-oor/c96-allocator-gc080-active/`，明确不
作为正式 C96 性能值。

随后执行必要的严格 treatment：fresh launch 恢复原 sweep 的 Prefill/Decode
`max-running=96` 和 `cuda-graph-max-bs=96`，保持 scratch=0/1、mem=0.70/0.85、
chunk、模型、MTP、KV-aware、rank affinity 和节点不变，只使用已经通过 C48 机制验证的
Prefill active GC。容器快照确认 Prefill 同时存在
`PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`、
`INFERA_PYTORCH_MEMORY_FRACTION=1.0` 和 96/96 两个引擎参数，八个 scheduler 均打印
一次 fraction hook 生效日志。

结果：

- historical C96 在 mandatory primer 返回 71/101 时 DP0 OOR；本轮完整通过
  1,061/1,061 warmup，耗时 2,430.59 秒，8/8 Prefill rank 全程存活；
- 完成完整 1,200 秒 profiling：1,753 条成功、错误 0；窗口结束取消 66 条在途请求，
  最终门禁为 0/1,753；
- 总吞吐 159,170.55 token/s、每 GPU 9,948.16 token/s、平均 QPS 1.428；
  TTFT p50/p95 为 7.869/151.870 秒，ITL p50/p95 为 12.36/19.00 ms；
- 3,509 个有效 2 秒 HBM 样本中，各卡峰值依次为
  94%/87%/89%/88%/88%/89%/90%/93%，全局峰值 94%，没有任何 >=95% 或 99% 样本；
- Prefill 日志中 OOR、fatal、transport retry、session blacklist 和 transfer
  failure 均为 0；Decode 的 46 条 `KVTransferError` 全部集中在 profiling 超时收口，
  原因均为 `Aborted by AbortReq`，不是 RDMA 故障；
- aggregate 另保留 2 条 warmup-only `InvalidInferenceResultError`，输入分别为
  211,858 和 458,862 tokens；它们不进入 profiling 门禁，且没有对应 transfer/session
  故障。

本轮用于原强度 active-GC 机制验收的 C96 数据源为
`debug-oor/c96-max96-allocator-gc080-active/bench/agentx_conc96.json`；相关
server metrics、逐请求记录、三端日志和 HBM 采样均在同一轮目录。其
`max-running/graph-max=96/96` 正确，但使用 digest `34909eb3...`，只证明该新栈
下 active-GC 能越过 OOR，不进入 intended-commit 性能曲线。

这补齐了此前 `max-running=8` 的混杂因素：active GC 不仅降低 HBM 峰值，也在原始
C96 引擎强度下越过历史 OOR 位置并形成完整性能窗口。结合“rank-affinity-only C48
仍 OOR”“C48 no-op 峰值 99% 而 active GC 为 93%”和独立 allocator probe，当前证据
支持将根因收敛为：**PyTorch inactive cached segment 与 HSA/AITER 动态 scratch/临时
张量共同竞争 HBM；外部 HSA 分配无法主动触发 PyTorch failure-path 回收，pressure-aware
GC 通过提前返还 inactive segment 消除该资源死角。** OOR 具有时序随机性，单次通过
不能证明任意轨迹绝不失败，但 C48 机制 A/B 加原强度 C96 完整通过已满足本阶段修复
验收，不再重复跑相同 baseline。

非冗余重测顺序更新为：先用最终配置重测原空响应门禁失败且已有最长部分窗口的 C64；
C64 通过后，不重复同机制的 C128/C160，直接以 C192 检验最高 OOR 边界。P4 C48/C64
已被 P8 取代，不重跑。

### 7. `402df1e-2c71811` Decode `PERMISSION_FAULTS` 镜像回归调查

本节是与当前 P8D8 扫描解耦的隔离复现。当前扫描继续使用已通过真实 MTP 长稳验证的
`c29bd17-b02ab81`；复现只使用独立代码副本和 136/139，不重启扫描服务、不占用
137/138 GPU。

#### 7.1 已证实的回归边界

较新镜像 `infera/engine-sglang:glm52-v518-402df1e-2c71811` 在两次 fresh-node
TP4/EP4、C8、真实 EAGLE MTP GSM8K 中稳定复现同类故障：

- Decode 137：运行约 10 分 31 秒、完成约 723/1,319 条后，四个 TP scheduler
  同时出现 GPU memory access fault；
- Decode 从 137 换到 139 后：运行约 10 分 48 秒、完成约 731/1,319 条后再次四 rank
  同时故障；
- 两次均先通过 smoke 和 250K long-context，说明短 smoke、单请求长上下文不能代替
  多请求真实 MTP soak；
- 回退到 `c29bd17-b02ab81` 后，在相同 136/139 节点完成 GSM8K 1,319/1,319，
  随后的 1,200 秒 AgentX 为 395/395、错误 0；另一次 137/138 真实 MTP correctness
  也通过。

节点 136 上直接检查到本次对应的镜像 ID 分别为：

- failing：`sha256:9537d5a55daee5ec1d9fac1ea942e3d727a838634e802b84bf62e897cd7ae81a`；
- passing：`sha256:8bef14c08ba4764698b5b9c82f41e1a3e5c0f7bc3c4a22d161cd00dbf1feb456`。

两者前 39 个 RootFS layer 相同，之后开始分叉。后缀对应的实际源码也已在容器内
`git rev-parse` 复核：

- SGLang `c29bd17` 到 `402df1e` 相隔 58 个 revision，净 diff 为 35 个文件、
  +6,333/-105 行，加入 gfx950 四 kernel fused DSA indexer、FlyDSL sparse MLA、
  MTP/Top-p 和 HiCache 优化；
- AITER `b02ab81` 到 `2c71811` 相隔 8 个 revision，净 diff 为 23 个文件、
  +3,251/-492 行，加入 FlyDSL sparse MLA、`fp8_mqa_logits` 改动和 GLM-5.2
  BF16 skinny GEMM retune；
- Infera 顶层也不是逐字节相同：旧镜像构建 RDMA-core，新镜像改为安装 ABI-4
  libionic 并多出 PD ROCm rejection patch layer。因此一次整镜像回退只能证明
  **软件镜像栈回归边界**，不能把责任直接归给 SGLang 或 AITER 的某一个 commit。

#### 7.2 fault 位与路径含义

两节点 dmesg 的共同特征为 `gfxhub0`、VMID 3、非零 PASID、UTCL2 client
`TCP (0x8)`、`WALKER_ERROR=0`、`MAPPING_ERROR=0`，fault VA 均为 2 MiB 对齐。
137 同时出现 `PERMISSION_FAULTS=0x3/RW=0` 和 `0x5/RW=1`；139 四卡均为
`0x3/RW=0`。

这里的 `TCP` 是 GPU shader 的 Texture Cache Pipe，不是网络 TCP。`0x3` 表示无效
PTE/不可读的读访问，`0x5` 表示无效 PTE/不可写的写访问；这些位说明 GPU kernel
访问的 VA 当时无有效权限，但不能单独区分 OOB、use-after-free、CUDA/HIP graph
保存了过期地址、错误 metadata、缺失 peer mapping 或驱动页表缺陷。没有 RAS/CPER/ECC
证据，不能把它判成 HBM 硬件故障；也没有同时出现 AMD-Vi/IOMMU fault，不能仅凭
GFXHUB fault 判成 IOMMU。

四个 fatal Python stack 都停在 EAGLE target verify 的 CUDA-graph load/replay 路径，
最上层可见位置为 `_resolve_dsa_variant`。该函数在新旧 SGLang SHA 间逐字不变，而且
HIP kernel 异步执行，所以它只能证明故障发生时所处阶段，不能当作 faulting kernel。
GPU coredump handler 缺失，现场没有 native kernel PC。

两点主动路径信息进一步缩小但没有闭合范围：

- server-info 明确是 `dsa_prefill_backend=tilelang`、
  `dsa_decode_backend=tilelang`，所以新增的 FlyDSL sparse MLA attention backend
  不是直接选择项；
- `enable_dsa_fused_indexer=null` 在新 SGLang 中是 tri-state 默认开启，四 rank
  日志均明确出现 `gfx950 fused DSA indexer enabled`，并在每 rank graph/pool
  初始化前预留 581 MiB workspace。该路径在旧 SGLang 不存在，是当前首要嫌疑；
- `SGLANG_USE_AITER=1` 会让 BF16 linear 进入 AITER `tgemm.mm`，所以新 AITER
  的 skinny/split-K GEMM 路由也可能实际执行，不能因为 DSA backend 为 TileLang
  就排除 AITER；
- fault 前没有 Mooncake retry、session blacklist、transfer failure、OOM 或
  HSA out-of-resources。直接 issuer 更像 Decode compute kernel；但错误 PD metadata
  作为上游输入尚未做单变量排除。

隔离 A/B/C/E 后更严格的结论是：**同一 failing digest 上，关闭 Decode gfx950
fused DSA indexer 完整通过，而 fused 路径在默认 graph、强制同步和关闭 Decode
graph 三种模式下均稳定复现；故障对 fused-indexer 路径具有高置信度归属，CUDA
graph capture/replay 不是必要触发条件。** 仍未证明具体 device kernel/指令，也未在
OOB、index/page-table/result metadata 越界与 buffer/workspace 生命周期之间闭合根因。

#### 7.3 隔离 A/B 与同步 issuer 定位结果

独立副本根目录为 `/home/liyingli/bench_agentx_perm_fault_debug/`，完整整理文档为
`/home/liyingli/bench_agentx_perm_fault_debug/RCA.zh-CN.md`。四轮均使用真实
acceptance 的 GSM8K、TP4/EP4、C8 和同一 failing image digest：

1. **A / fused off：PASS。** 仅为 Decode 增加
   `--no-enable-dsa-fused-indexer`，18m30s 完成 1,319/1,319；越过历史故障时间和
   请求进度。采用产物：
   `/home/liyingli/bench_agentx_perm_fault_debug/results/20260914T064500Z-a-nofused/`。
2. **B / fused default positive control：FAIL，已复现。** 相对 A 仅恢复默认 fused
   路径；约 12m45s、最大完成 873/1,319 时 Decode 四卡同步出现同类 fault，Prefill
   无同类事件。采用产物：
   `/home/liyingli/bench_agentx_perm_fault_debug/results/20260914T071600Z-b-fused-default/`。
3. **E / Decode-only serialize=3：FAIL，再次复现。** 相对 B 仅在 Decode 设置
   `AMD_SERIALIZE_KERNEL=3` 和 `AMD_SERIALIZE_COPY=3`；约 13m57s、最大完成
   880/1,319 时四卡同步故障。同步后的当前线程栈前移到
   `transform_index_page_table_decode_fast` 的 `transform_index.py:148` Triton
   launch 调用点。采用产物：
   `/home/liyingli/bench_agentx_perm_fault_debug/results/20260914T074500Z-e-serialize3/`。
4. **C / 仅关闭 Decode CUDA graph：FAIL，再次复现。** 相对 B 只增加
   `--disable-decode-cuda-graph`，live 配置差异 gate 确认 Decode graph backend 从
   `full` 变为 `disabled`，运行期持续记录 `cuda graph: False`；51m44s、最大完成
   856/1,319 时四卡仍同步出现 `PERMISSION_FAULTS=0x3`。采用产物：
   `/home/liyingli/bench_agentx_perm_fault_debug/results/20260914T085501Z-c-no-decode-graph/`。

E 将主机侧同步可见 issuer 收敛到 fused-indexer Decode 的 page-table transform
launch，但 `transform_index.py:148` 不是已证实的 device faulting 源码行。C 排除了
graph replay/lifecycle 作为必要条件；其失败进度 856 与 B/E 的 873/880 接近、墙钟
时间因 eager execution 增至 51m44s，更支持触发与累计请求工作量相关，但该项仍是
推断。关键证据分别保存在每轮的 `logs/decode-0.log`、`benchmark/runner.log`、
`evidence/end/request-progress.json` 和 `evidence/end/*dmesg*`；E 的源码快照及校验值
另见 `evidence/source/`。136/139 的隔离容器已停止，GPU 0--3 已归零。

后续若需闭合到具体 kernel/内存生命周期，应在该最小复现上补齐 native kernel PC 和
fault VA 到 allocation 的映射，或对 page-table/index/result buffer 与 CUDA graph
生命周期做更细粒度单变量实验。调试开关不得作为生产修复。

#### 7.4 防复发要求

- 暂停推广 `9537d5a5...` 及其 tag；继续以完整 digest 而非 tag 作为已知
  good/bad 身份；
- 构建脚本现已显式传递完整 `SGLANG_SHA`、`AITER_SHA` build-arg，并在构建后及
  每次 launch 前将所有节点 Image ID、容器内 `git rev-parse` 和期望 SHA 逐项比较，
  任一不符即失败；
- 镜像 provenance 还必须记录 Infera commit、patch-set hash、Mooncake SHA、
  libionic/ROCm/PyTorch/driver 版本和最终 image digest。当前 tag 不包含 Infera
  版本，同一 tag 已出现不同 digest，不能视为不可变制品；
- 新镜像 promotion gate 至少包含多 rank 真实 MTP C8 长稳、完整 correctness 和
  1,200 秒压力窗口，并采集从 run 起点开始的 dmesg delta；smoke、250K 单请求和
  模拟 acceptance 均不能单独放行；
- 运行清单继续保存 server-info，但要额外把 fused indexer、DSA backend、graph、
  all-reduce 和 AITER 路由的最终解析值作为门禁字段，避免默认值升级后静默启用新
  kernel。

本阶段 RCA 状态为“fused-indexer 路径级根因已高置信度确定，CUDA graph 已排除为
必要条件，同步 issuer 已收敛到 page-table transform launch；具体 device
kernel/指令和内存生命周期根因未闭合”。
后续实验继续与 P8D8 扫描隔离，不占用 137/138。

## 代码改动

- `build_image.sh`：向 ROCm optimization base 显式传入完整 `SGLANG_SHA` 和
  `AITER_SHA`，构建及分发后在每个节点读取容器内真实 commit，不匹配立即失败。
- `lib/common.sh`：每次 launch 前强制所有拓扑节点 Image ID 完全一致，并逐节点
  校验 SGLang/AITER 完整 commit；同名 tag 漂移现在会在创建服务容器前失败。
- `config.p8d8.sh`：新增本轮 P8D8 固定拓扑 profile，避免漏传覆盖后退回
  `config.sh` 的 P4D8 默认值；四个并发容量值默认 32，逐点仍显式覆盖；认证运行
  warmup 默认固定为每 lane 10，warmup1 只允许显式用于快速筛查。
- `tools/agentx_env.py`：除既有 `max-running` 外，新增 live container
  `cuda-graph-max-bs` 与配置值的一致性门禁，防止 benchmark 在错误 graph 上运行。
- `engine.sh`：将原先固定的 `HSA_NO_SCRATCH_RECLAIM=1` 改为默认仍为 1、允许
  Prefill/Decode 分角色命令行覆盖；同时分角色透传 opt-in
  `PYTORCH_HIP_ALLOC_CONF` 和 allocator fraction，校验 fraction 范围，并只在启用时
  挂载启动 hook。默认行为不变。
- `config.sh` / `config.sh.example`：增加 Prefill/Decode allocator 配置项并说明当前
  vendor ROCm build 必须同时使用 HIP alias 和显式 fraction 才会进入 GC 路径。
- `hooks/sitecustomize.py`：在 scheduler 首次选择设备时调用一次
  `torch.cuda.set_per_process_memory_fraction`；按 device 去重，未配置环境变量时
  完全不加载 PyTorch、不改变服务行为。
- Rust router、`launch.sh` 和 `lib/common.sh`：增加 opt-in PD DP-rank affinity，
  对称 DPA 配置默认启用；仍保留未启用时的原路由路径。
- `tools/collect_agentx.py`：若失败 run 已写出部分 aggregate，通过同目录
  `runner.log` 的失败标记跳过该点，防止 C64/C128 被误画为有效 Pareto 点。
- `tests/test_simplified.py`：补充 image build-arg/commit gate、P8D8 固定 profile、
  live graph-max drift、“失败部分 aggregate 不收集”和 allocator hook
  role-scoped/opt-in 回归用例；当前 21 项全部通过。

## 产物位置

- 报告：`REPORT.zh-CN.md`
- 正式采用和明确排除的数据路径：见报告开头“数据采用口径与路径索引”
- `agentx/c<C>/`：原始 sweep，不代表修复后的最终配置
- `results.csv` / `pareto.png`：仅包含原始 sweep C8/C16/C32 的历史汇总；在
  active-GC 顺序复测完成并重建前，不作为最终配置汇总
- 指标图和 workload 分布：各结果目录中的 `bench/*.png`

## 最终验证

- `python3 -m unittest discover -s tests -v`：16/16 通过。
- `bash -n`：配置、launch/stop、AgentX、分析和 eval shell 脚本全部通过。
- IDE lints：allocator 配置、hook、`engine.sh`、`tools/collect_agentx.py` 和
  `tests/test_simplified.py` 无诊断。
- 统一分析：成功生成 3 个有效点的 `results.csv` 与 `pareto.png`；C64/C128
  失败 aggregate 被明确跳过。
- 资源清理：137/138 均无本轮容器、无 KFD GPU 进程。
