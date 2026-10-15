# 工作状态（按时间追加，UTC）

## 2026-10-12

- 17:40 137→138 同步 `infera-sglang:v0521-yihou-1010-mc4340`（ID `e8e2838f…`），17:44 完成。
- 17:53 在 137 上构建 `infera-sglang:v0521-llying-1012-mc4340-destpin`
  （`sha256:afcc2f8abff7aee8a188e574168120826f35f26fa3d3e0db3881c1e8d91e820a`）：mc4340 镜像 +
  Mooncake faae8dd4 重编，补丁 #4340 port、dest-local-rail、ib-timeout、dest-pinned-hca 依次打上，
  构建内断言全部通过（`engine.so` sha256 前缀 `f003b2b62fcfa57c`）。日志 `image/build-20261012T175344Z.log`。
  17:58 推到 138、18:02 推到 135，三台 ID 一致。
- 节点状态（17:50 前后）：135 GPU1 被别的租户的 vLLM 引擎占用（约 97 GB），`ionic_7` 无 netdev、GID 为 0；
  137、138 GPU 空闲；136 被 xiaobo 的 GLM-5.3 kernel 迭代循环间歇占用，不用。
- harness：以 v0.5.21 pack-up 的 bench-harness 为底（`scripts/bench-harness/`），改动见 README。
- 18:05 误操作：做 dryrun 时漏了 `DRY_RUN=1`，launch.sh 实际启动了 `ROUND=dry` 的 etcd、135 prefill-0、
  137 prefill-1。18:07 终止了 launch.sh 驱动（prefill-2、decode、router 未启动）。两个 prefill 留着完成
  初始化，用来验证 TP4 prefill 在真机上的启动（HCA 数 7/8 正确、PID 偏移生效、HiCache 153 GB/rank）并预热
  JIT 缓存；之后停掉，正式轮次用新的 ROUND。
- 18:17 dry prefill-1（137）就绪，冷启动约 12 分钟；18:22 dry prefill-0（135）就绪，约 17 分钟（135 有 2.4 TB 页缓存，
  锁页 ~750 GB 时要先回收，较慢）。两者 census 无致命项，HCA 数 7/8，HiCache 153.03 GB/rank，prefill KV 2,270,656 tokens/rank。
- 18:25 用 137 上 yihou 的 v0.5.21 mc4340（同 aiter/triton）缓存给 prefill-2、decode-0 做 AITER/triton 种子，
  并把 prefill-1 新编的模块合并进 prefill-2；复制 yihou 的 AgentX HF 数据集缓存（3.5 GB）到 137 客户端缓存。只复制，不动源。
- 18:28 停掉 dry 栈（`stop.sh ROUND=dry`）。停后 HiCache 栈的显存回收较慢，等所用 GPU < 1 GiB 后再正式起栈（ROUND=r1）。
- 18:26 正式轮 r1 开始（`run_3p1d.sh all`，在 137 上 setsid nohup）。问题与处理：
  1. 预检循环里 ssh 吃掉了 `while read` 的 stdin，只检查了 135（137/138 已人工确认空闲、镜像 ID 推送时已核对）。已修（`on()` 加 `</dev/null`）。
  2. launch.sh 的同节点启动门 `wait_healthy.py` 内部 ssh 同样读走了启动循环剩余的行，decode-0 那一行被吞，没有启动
     （yihou 的同节点部署只有两行、门总在最后一行，所以没暴露）。已修（门调用加 `</dev/null`）。18:32 按 launch.sh 相同参数手动在 138 启动 decode-0。
  3. 在登录节点 mv 替换了正在 137 上运行的 launch.sh / run_3p1d.sh，NFS 上旧句柄失效（"Stale file handle"），18:33 launch.sh 在
     三个 prefill 健康后、启动 router 前退出，流水线随之失败退出。教训：运行期间不改实验目录里在用的脚本（mv 替换也不行）。
  4. 手动启动的 decode-0 在 18:33 报 `LLVM ERROR: IO failure on output stream: No space left on device`：容器 TMPDIR 是 8g tmpfs，
     TP8 的 8 个 rank 同时编译 triton/LLVM 撑满。改为 `ENGINE_TMPFS_SIZE=32g`（engine.sh 新增开关）。
- 18:40 恢复（`operations/recover_r1_20261012.sh`）：保留三个已健康的 prefill，退役失败的 decode（改名 `-tmpfsfail`），以真实接受率重启
  decode-0（tmpfs 32g）并启动 router，随后从门禁阶段继续（`run_3p1d.sh after-up`，CENSUS_SINCE=恢复开始时刻）。
- 18:47 decode-0 就绪（TP8 mem 0.85：2,992,512 KV tokens/rank，每 rank max-running 32），router 见 4 个 worker，起栈 census 无致命项。
- 18:52 **门禁失败：48/48 答错**（真实接受率；输出多为乱码或重复文档末尾片段）。排查（产物在 `runtime/runs/r1/diag/`）：
  - A 短文档（单 chunk ~2.7k tokens / 两 chunk ~10k）：16 个对 6 个 → 与多 chunk 无关。
  - B router 改为 rank 亲和（只走同 rank 传输）：16 个对 4 个 → 与跨 rank / Mooncake 跨 rank 修复无关。
  - 极短 prompt（18 tokens：数数、首都、复述）全部正确。
  - C <2048 tokens（decode 走稠密注意力，多页 KV）：16 个对 7 个。
  - 逐请求归因 + logprobs：主要错法是"数字对、缺开头的 Z"，且**首 token（由 prefill 计算）分布不确定**（如 `Z` logprob −0.9，
    `304`、`</think>` 紧随），而首 token 正确时 decode 后续数字置信度 ~−0.01 → prefill 前向质量下降，不是传输问题。
  - 与 yihou 已跑通的 v0.5.21 prefill 相比，影响计算的唯一差异是 `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`（取自 yaocheng 的 v0.5.19 配置；
    yihou v0.5.21 为 0，`bench/glm5p2_pd/issue.md` §2.3 也写明它需单独做 correctness 验证、否则保持默认关闭）。
- 决定：单变量改 `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0`，停 r1，整栈重起为 r2 再跑门禁。
- 19:00 r1 已停（`stop.sh ROUND=r1`）。确认 prefill CUDA graph 为 disabled（graph bs 256 vs 64 不影响计算）。
  配置改为 `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0`（可用环境变量覆盖）。
- 19:03 `operations/start_r2.sh` 在 137 后台运行：等所用 GPU 显存 < 1 GiB 后执行 `run_3p1d.sh all`（ROUND=r2，C144/160/176）。
- 19:07 r2 起栈（修复后的 launch.sh 全自动拉起 3P+1D+router，19:15 就绪，census 无致命项；预检三台都检查到）。
- 19:16 r2 原门禁（4 万/11.5 万 tokens 文档）34/48 完全正确（r1 为 0/48）→ grouped topk=1 是 r1 失败的原因。剩余 14 个错误
  **全部是同一文档中另一条记录的真实代码**（相邻或编号相近的记录：如问 172 答 1711），即长文档检索错位，不是 KV 损坏（损坏会产生
  文档中不存在的代码或乱码）。另测简单文档（~1.7k/2.7k tokens，跨 rank）47/48（r1 同类约 40%）。
- 门禁改为两层（`kv_gate.py` 新增 other_record / fabricated 分类）：简单集 ≥95% 完全正确；困难集 0 个编造/乱码答案且 ≥60%。
  原 95% 门槛是在没有基线时定的，超出了模型本身在超长文档上的检索能力。
- 19:18 r2 从 after-up 继续（门禁 → 切换模拟接受 3.61 → 冒烟 → C144/160/176）。简单集 48/48。
- 19:19 r2 两层门禁通过：简单集 48/48；困难集 38 正确 + 10 检索错位、0 编造（79.2%）；census 干净。
- 19:24 decode 切到模拟接受 3.61、router 重启，router 见 4 worker，agentx_env 线上配置校验 PASS。
  但三个点都在启动时失败：agentx_bench.sh 以本用户 mkdir `NODE_RW_ROOT/cache/137/agentx/aiperf`，该目录是先前用容器（root）
  做 HF 缓存种子时建的。用容器 chown 回本用户；失败产物移到 `runtime/runs/r2/failed-mkdir-1924/`。bench 循环加保护：某点失败且无结果则停止。
- 19:30 r2 C144 开始（`run_3p1d.sh bench`，C144→C160→C176）。
- 19:49 C144 预热完成（1,598 请求 0 错误，约 17 分钟），进入 3600 s 测量。
- 20:15 C144 中段观察：服务端前缀命中 ~86.6%；decode 各 DP rank KV 占用不均（DP3 99% 且 33 个请求等预分配，DP5 94%，
  其余 48–80%）。与 P8D4 分析一致：decode 不上报 KV 事件，router 选 D rank 实际按请求数轮转，不看 token 量。按 token 均衡的 R2
  只在 b4 的 router 二进制里，不属于本轮"router 与 yaocheng 一致"的范围。预期更高并发收益有限，峰值可能在 C144 或更低。
- 19:49 C144 预热完成（1,598 请求 0 错误，约 17 分钟），进入 3600 s 测量。
- 20:15 C144 中段观察：服务端前缀命中 ~86.6%；decode 各 DP rank KV 占用不均（DP3 99% 且 33 个请求等预分配，DP5 94%，
  其余 48–80%）。与 P8D4 分析一致：decode 不上报 KV 事件，router 选 D rank 实际按请求数轮转，不看 token 量。按 token 均衡的 R2
  只在 b4 的 router 二进制里，不属于本轮"router 与 yaocheng 一致"的范围。预期更高并发收益有限，峰值可能在 C144 或更低。
- 19:49 C144 预热完成（1,598 请求 0 错误，约 17 分钟），进入 3600 s 测量。
- 20:15 C144 中段观察：服务端前缀命中 ~86.6%；decode 各 DP rank KV 占用不均（DP3 99% 且 33 个请求等预分配，DP5 94%，
  其余 48–80%）。与 P8D4 分析一致：decode 不上报 KV 事件，router 选 D rank 实际按请求数轮转，不看 token 量。按 token 均衡的 R2
  只在 b4 的 router 二进制里，不属于本轮"router 与 yaocheng 一致"的范围。预期更高并发收益有限，峰值可能在 C144 或更低。
- 20:57 **C144 完成**：18,001 tok/s/GPU（20 卡，总 360,023），P50/P90 interactivity 52.9/34.8，TTFT p50/p90 10.2/55.1 s，
  11,393 请求 / 4 error-dropped，服务端 GPU 缓存命中 74.8%，census 无致命项。结果 `results/r2/c144/`。C160 开始。
- 20:57 **C144 完成**：18,001 tok/s/GPU（20 卡，总 360,023），P50/P90 interactivity 52.9/34.8，TTFT p50/p90 10.2/55.1 s，
  11,393 请求 / 4 error-dropped，服务端 GPU 缓存命中 74.8%，census 无致命项。结果 `results/r2/c144/`。C160 开始。
- 22:33 **C160 完成**：17,754 tok/s/GPU（总 355,080，比 C144 −1.4%），P50/P90 interactivity 75.4/39.5，TTFT p50/p90 19.6/85.2 s，
  11,079 请求 / 3 error-dropped。census 报 12 条 decode `KVTransferError`，全部在 22:23:03–04，即测量窗口结束时 AgentX 取消 63 个在途
  请求（grace period timeout，errors=0）所致；prefill 侧无 RDMA 层错误。C176 开始。
- 22:52 吞吐随并发下降（C144 > C160），峰值在 C144 附近或更低。`operations/followup_r2.sh`（137 后台）：C176 结束后在同一栈上跑
  C128；若 C128 > C144 再跑 C112，否则停止（峰值被 128 与 160 夹住）。
- 23:00 decode 负载分析（`scripts/decode_balance.py`，结果在 `results/r2/c*/decode_balance.txt`）：C144→C160 每 rank 实际 decode 请求
  7.0→5.0，等 KV 传输 5.6→8.8，等预分配 1.6→3.0，KV>90% 时间 19.7%→28.3%。即高并发下 decode KV 被"已预分配、等 prefill 传输"的请求
  占住，有效 batch 变小（所以 C160 ITL 更低但吞吐下降），与 yaocheng 2P1D 高并发形态一致。
  v0.5.21 新参数 `--disaggregation-decode-allocation-policy {early,prefill_complete}`（默认 early；prefill_complete = prefill 完成后才在
  decode 分配，两角色须一致）正对这个瓶颈，作为基线之后的调优候选（r3）。
- 22:57 C176 预热完成（1,953 请求 0 错误，约 24 分钟），进入测量。23:18:41 出现 1 次测量期内传输失败：prefill-1 DP3 → decode DP6
  `KVTransferError: Failed to send aux data`（跨 rank），前后无 RDMA 层错误（无 `Process failed for slice` / `retry counter exceeded`），
  单个请求、偶发，继续观察。
- 22:57 C176 预热完成（1,953 请求 0 错误，约 24 分钟），进入测量。23:18:41 出现 1 次测量期内传输失败：prefill-1 DP3 → decode DP6
  `KVTransferError: Failed to send aux data`（跨 rank），前后无 RDMA 层错误（无 `Process failed for slice` / `retry counter exceeded`），
  单个请求、偶发，继续观察。
- 00:06 **C176 完成**：16,298 tok/s/GPU（总 325,962，比 C144 −9.5%），P50/P90 interactivity 96.9/49.4，TTFT p50/p90 38.5/127.2 s，
  10,009 请求 / 11 error-dropped，GPU 缓存命中 62.1%。census 13 行：12 行为窗口结束取消（23:57:52，60 个 cancelled），1 行为 23:18 的 aux 失败。
  followup 自动开始 C128。
- 00:57 准备调优轮 r3：配置加"按轮次增量"机制（`config/round.<ROUND>.sh`），`round.r3.sh` = `DECODE_ALLOC_POLICY=prefill_complete`（两角色）。
  `operations/start_r3.sh`（137 后台）：等 followup 结束 → 停 r2 → 等显存回收 → `run_3p1d.sh all` ROUND=r3，C144、C160（含两层门禁）。
- 01:29 **C128 完成**：19,758 tok/s/GPU（总 395,168，比 C144 +9.8%），P50/P90 interactivity 71.3/40.5，TTFT p50/p90 7.8/30.4 s，
  12,376 请求 / 5 error-dropped，GPU 缓存命中 86.6%（C144 74.8%）。decode：每 rank 在跑 6.06、KV 0.50、>90% 时间 6.2%、等预分配 0.21。
  census 只有窗口结束取消。峰值在 C128 或更低，followup 自动开始 C112。
- 01:32 `start_r3.sh` 在等待阶段被终止（未执行任何动作），由 `operations/next_r2_r3.sh` 取代：C112 结束后若 C112 > C128 先跑 C96；
  然后取 r2 最优并发 B，停 r2，跑 r3（prefill_complete）于 B 与 B+16。
- 02:35 更正：r2 困难集（`gate.hard.json`）的 10 个"other_record"中有 2 个其实是**正确数字但缺开头的 Z**（问 2372 答 `6127`、
  问 4623 答 `7655`，41k 与 123k 文档各 1 个），当时分类脚本把它们并进了 other_record。`kv_gate.py` 已新增 `missing_prefix` 类。
  r2 正确性实际情况：简单集 95/96（另 1 个为其他记录）；41k 文档 65/80 正确，123k 文档 7/16 正确；非正确答案全部是文档内真实代码
  （其他记录或缺 Z 的正确数字），0 个编造/乱码。缺 Z 与 r1 的首 token 症状同形，但只在长文档、2/48；是否模型本身极限，尚无参照实验证明。
- decode 侧有效并发（AIPerf Effective Decode Concurrency / Little 定律 / 服务端 8 rank 在跑之和）：C128 50.5/50.6/48.5，
  C144 59.0/59.1/56.1，C160 43.2/43.3/40.2，C176 30.8/30.9/30.0；首 token 前在途（Effective Prefill Concurrency）44.7/68.0/104.6/147.2。
- 02:44 应用户要求暂停：停掉等待中的 next_r2_r3.sh（未执行任何动作），不再自动跑 C96/r3；C112 跑完后栈保持运行。
- 02:50 **C112 完成**：20,545 tok/s/GPU（总 410,901），完成 12,319、输出 174.3 tok/s/GPU（与 C128 的 12,376 / 173.4 几乎相同；总吞吐 +4% 主要因平均输入 120.0k vs 114.9k），P50/P90 71.6/41.6，TTFT p50/p90 6.0/16.1 s，GPU 缓存命中 91.1%，census 无致命项。栈保持运行。复盘写入 `REVIEW.zh-CN.md`。
- 04:20 发现 r2 C144 是 decode 重启后的第一个点，decode 每步耗时在对齐负载下比其余四点慢 13–17%（新栈第一次跑偏慢），C144 结果偏低；详见 REVIEW §8。

## 2026-10-13 调优阶段（计划见 `PLAN-TUNING.zh-CN.md` v2）

- 04:44 用户确认：去掉阶段 1 基线复测与 R3；2a 中 prefill mem 0.90 与 HiCache 扩容合测（137 两实例 200 GB/rank，135 实例可更大）；
  按判定标准自动推进，同时人工持续监测（不只靠脚本门禁），用户随时接入。
- 04:50 新工具：`scripts/tune_round.sh`（一轮：内存快照 → 停 router → flush 3 个 prefill → 带本轮开关起 router 并按启动日志核对
  → AgentX → census / flush 校验 / 均衡与引擎节奏分析 / router 指标 → 发布到 `results/<variant>/c<C>/`）、`scripts/tune_queue.sh`
  （按 `operations/tune_queue.tsv` 顺序执行，可在运行中追加）、`scripts/tune_watch.py`（人工巡检一屏视图）、`scripts/flush_check.py`、
  `scripts/engine_pace.py`（decode 对齐负载每步耗时 + prefill 满 chunk 耗时，用于区分栈漂移与路由效果）。
  `restart_router.sh` 支持 `ROUTER_ENV`；`agentx_env.py` 透传 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`（调优轮恒为 true，
  router 仅在开会话亲和时读取该 header，故各轮客户端一致）。
- 04:55 flush 语义核对：v0.5.21 `unified_radix_cache.reset()` 同时 `cache_controller.reset()`、`mem_pool_host.clear()`、发 all-cleared
  KV 事件；实测同一 5.2k prompt 第二次命中 5,184，flush 后为 0，三个 prefill 均 "Cache flushed successfully"。decode 不 flush（无前缀缓存）。
- 04:56 阶段 1 开始（`tune_queue.sh` 在 137 后台，pid 4057511）：t1-base → t1-r1 → t1-r1aff，均 C144。t1-base flush 与 router 核对通过。
- 06:20 **t1-base（C144，基线，已 flush，decode 预热）完成**：19,503 tok/s/GPU（r2 C144 为 18,001，+8.3%），输出 168.0 tok/s/GPU
  （+6.9%），完成 11,953（+4.9%），平均输入 117.4k（r2 113.7k）。P50/P90 interactivity 80.6/37.9（r2 52.9/34.8），ITL p50 12.4 ms
  （r2 18.9），TTFT p50/p90 12.8/52.1 s，GPU 缓存命中 81.8%（r2 74.8%）。decode 对齐负载每步耗时与预热后的 r2 C128/C112 一致，
  确认 r2 C144 的偏低来自 decode 刚重启。decode 有效并发 47.7（r2 C144 为 58.9：decode 慢时请求在 decode 停留更久，Little 定律下
  在途数反而大）。census 只有窗口结束（06:14）的取消。与 r2 C128（19,758，未 flush、热缓存起步）相比低 1.3%，峰值区间 C112–C144 平坦。
- 06:21 t1-r1（+R1 `INFERA_PD_PREFILL_GUARD_RELEASE=completion`）开始，router 核对 guard=true。
- 07:46 **t1-r1（+R1）完成**：19,308 tok/s/GPU（t1-base −1.0%），但平均输入短 2.6%（114.3k vs 117.4k）；输出 170.8 tok/s/GPU（+1.7%）、
  完成 12,149（+1.6%）。TTFT p50/p90/mean −5.7%/−7.0%/−7.7%，ITL p50 12.4 → 13.8 ms（P50 interactivity 80.6 → 72.3），P90 37.9 → 39.7。
  decode 有效并发 47.7 → 50.3，首 token 前在途 73.2 → 68.7。引擎节奏与 t1-base 一致（无漂移）。census 只有窗口结束的取消。
  判定：吞吐差 < 2%，按标准为噪声，不计收益；机制方向（TTFT 降、更多请求进入 decode）与 R1 一致且无副作用，保留 R1 作为后续基线。
- 07:46 t1-r1aff（+R1+P 会话亲和）开始，router 核对 guard=true、affinity=Prefill。
- 09:12 **t1-r1aff（+R1+P 会话亲和）完成**：**21,639 tok/s/GPU**（t1-r1 +12.1%，t1-base +11.0%），输出 185.0 tok/s/GPU（+8.3%），
  完成 13,192（+8.6%），平均输入 118.0k（+3.2%）。TTFT p50/p90/mean 11.2/24.4/14.8 s（p90 −50%），ITL p50/p90 14.0/30.2 ms，
  P50/P90 interactivity 71.4/33.1（P90 从 39.7 降到 33.1）。机制：prefill 缓存份额 94.0%（87–88%）、新算 26.0k tok/s（~45k，−42%）、
  prefill 排队 ~1.4（3–5）；decode 每 rank 在跑 7.04（5.87）、KV>90% 6.0%（16.9%）、等预分配 0.51、等传输 2.49（5.82），rank 间
  均衡（0.46–0.53）；decode 有效并发 59.4，首 token 前在途 53.5。router 会话命中 12,400 / 新绑定 2,414。census 只有窗口结束取消。
  判定：≥3% 且机制一致，**接受**。瓶颈从 prefill 转到 decode（prefill 算力有余，ITL p90 上升）。
- 09:13 阶段 1b：t1b-r2（R1+P 亲和+R2 `INFERA_R2_DECODE_DEMAND=on`）开始，router 核对 r2=On。
- 10:37 **t1b-r2（R1+P 亲和+R2）完成**：**22,340 tok/s/GPU**（t1-r1aff +3.2%），完成 13,628（+3.3%），平均输入相同（118.0k），
  输出 189.4（+2.4%）。TTFT p50/p90 10.1/19.2 s（−10%/−21%），ITL p50/p90 15.8/32.3 ms，P50/P90 interactivity 63.1/31.0。
  decode：KV>90% 时间 1.1%（6.0%）、等预分配 0、等传输 1.88、有效并发 65.2、每 rank 在跑 7.74（7.17–8.67，KV 拉平但 batch 更不均）。
  census 只有窗口结束取消。判定：≥3% 且机制一致，接受。最优配置 = R1 + P 会话亲和 + R2。
- 10:38 并发推进：t1c-best C160（最优配置）开始。
- **10:42:52 事故：共享 `/home`（10 TB NFS 卷，多人共用）写满**（`write /dev/stdout: disk quota exceeded`）。三台节点上把容器日志写到 NFS 的
  `docker logs -f` 进程、137 上的队列脚本与本轮驱动都因写失败退出；容器（引擎、router、AgentX 客户端）不受影响，t1c-best C160 客户端
  继续正常运行（预热 0 错误，10:56 进入测量）。docker 日志为 json-file 且不轮转，数据无损。（排查时一度误判为节点会话清理，已更正。）
  处理：
  1. 删除 NFS 上与 137 本地盘副本大小一致的 `server_metrics_export.json`（每轮约 3.5 GB，共 9 个，释放 32 GB；
     原位留 `.MOVED` 说明副本位置 `crsuse2-m2m-137:/mnt/m2m_nobackup/liyingli_3p1d/runs/r2/...`）。之后 `/home` 可用约 180 GB。
  2. `agentx_bench.sh` 拷回时排除该文件（`agentx_conc*.json` 已含其汇总）。
  3. `tune_round.sh` 开跑前要求 `/home` ≥ 20 GB，否则等待；收尾部分拆为 `tune_post.sh`。
  4. `tune_watch.py` 显示 `/home` 余量和最旧的 server 日志距今多久没写。
  5. `log_follow.sh` 从断点（最后一个时间戳 +1 ns）续接日志，11:00 恢复，缺口已补齐。
  6. 11:01 `tune_finish.sh` 在 137 后台等 C160 客户端结束 → 取回完整日志与结果 → `tune_post.sh` 发布；随后 queue.done=5 并重启队列。
- 12:04 **t1c-best C160 完成**（tune_finish.sh 收尾）：21,161 tok/s/GPU（C144 −5.3%），完成 12,812（−6.0%），输出 177.6（−6.2%）。
  GPU 命中 74.1%（C144 81.6%）、host 命中 17.6%（11.7%），TTFT p50/p90 19.7/34.4 s（10.1/19.2），首 token 前在途 76.2（44.8），
  decode 有效并发 58.1（65.2）。引擎节奏同 C144。census 窗口内 0 致命项；flush 校验 12/12。约束回到 prefill GPU KV 容量 → 支持 2a。
- 12:05 t1c-best C128 开始（确定峰值左侧）。2a 的 prefill mem 由 0.90 下调为 0.88（137 prefill 高水位在 C160 时剩 17.0 GiB）。
- 13:28 **t1c-best C128 完成**：21,554 tok/s/GPU（C144 −3.5%），完成 13,337（−2.1%），输出 186.0（−1.8%），平均输入 116.2k（−1.5%）；
  TTFT p50/p90 6.3/13.3 s，GPU 命中 87.2%，decode KV>90% 0.1%。census 0 致命项。**最优配置下峰值在 C144（22,340）**，C128–C144 为平台。
- 13:28 阶段 2a：`restart_prefills.sh`（mem 0.88，HiCache 135=400 / 137=200 GB/rank）开始；router 与 3 个 prefill 已退役（改名 *-2a）。
- 14:02 **2a prefill 重启完成**（13:28–14:02；follower 等待因 pipefail 每实例空等 5 min，已修；显存回收约 10 min；135 的 400 GB/rank
  host 池加 92 GB indexer 区注册约 12 min）。核对：GPU KV 2,433,088 tokens/rank（原 2,270,656，+7.2%），host 池 prefill-0 400 GB、
  prefill-1/2 200 GB，DSA indexer host 区 91.7 / 45.8 GB；router 4 workers，冒烟通过；census 无致命项。
- 14:02 t2a-best C144 第一次启动失败：`agentx_env.py` 要求所有 prefill 实例设置完全一致，而 2a 有意让 HiCache 大小按实例不同。
  已改为一致性比较时排除 hicache_size / host_register_chunk_gb（各实例仍按自己的期望值校验），在活栈上验证 PASS。
  失败目录改名 `t2a-best-c144.failed-envcheck`，14:04 重新开始 t2a-best C144。
- 15:31 **t2a-best C144（2a：prefill mem 0.88 + HiCache 400/200/200）完成**：22,238 tok/s/GPU（t1b-r2 −0.5%），完成 13,520（−0.8%），
  输出 189.0（−0.2%），平均输入 118.4k（+0.3%）——噪声范围内无变化。机制：prefill 缓存份额 95.9%（94.7%）、新算 18.5k（23.6k，−22%）、
  排队 0.44；GPU 命中 84.8%（81.6%）、host 10.2%；decode 每 rank 在跑 7.97、ITL 不变、KV>90% 0.2%。C144 下 prefill 本已有余量，约束在
  decode，扩容不改变吞吐。prefill 卡显存最低空闲 135=8.8 / 137=11.2 GiB，无 OOM。census 0 致命项。
- 15:32 t2a-best C160 开始（2a 的关键对比点：之前 C160 受 prefill GPU KV 限制，21,161）。
- 16:58 **t2a-best C160 完成**：22,279 tok/s/GPU（2a 前 C160 21,161，+5.3%；与 2a C144 22,238 持平），完成 13,355，输出 185.2，
  P50/P90 interactivity 66.9/29.4，TTFT p50/p90 18.2/28.0 s，GPU 命中 77.7%，prefill 缓存份额 95.7%、排队 0.50，decode 有效并发 64.3、
  KV>90% 0.7%。census 0 致命项。2a 把平台拓宽到 C144–C160，峰值约 22.3k 不变。
- 16:59 – 次日 01:38 **空档**：我这边的会话在等待 C160 结果的命令被中断后结束，未再继续；队列在 C160 后没有后续步骤，栈保持运行、
  空闲约 8.7 小时（引擎健康，`/home` 余 742 GB）。
- 调研发现（16:50）：v0.5.21 的 `--disaggregation-decode-allocation-policy prefill_complete` 硬性要求 prefill HiCache 为
  L2 `write_back`（`_validate_prefill_complete`），我们用的是 `write_through`，按原样启动会直接报错。2b 实际是两项改动
  （write_back + prefill_complete），且峰值处 decode KV 不紧张（KV>90% 0.2%–0.7%、等预分配 0），预期收益小。
- 02:30 **瓶颈重新定位（更正此前"decode 每步耗时"的判断）**：C176 运行中用引擎 Prometheus 计时直方图做 150 s 差分（约 570 请求）：
  prefill 侧从到达到 KV 传完平均 2.84 s（排队 0.28、bootstrap 0.58、传输 0.47 s，每请求 6.4 GB），decode 调度器分配 KV / 排队 ≈ 0，
  但 decode 记录的 TTFT 平均 31.7 s。decode 容器主进程 `sglang.launch_server`（HTTP + TokenizerManager，单 Python 线程）CPU 99.9%，
  8 个 scheduler 约 190%，detokenizer 2%；prefill 的 TokenizerManager 仅 8–13%（不流式输出）。decode KV 使用率约 0.5、ITL p50 15.9 ms，
  GPU 侧未到上限。结论：吞吐平台（C144–C176 约 22.3k）更可能来自 decode 的单进程 TokenizerManager 饱和（承担全部流式输出）。
  v0.5.21 支持 `--tokenizer-worker-num N`（HTTP 模式；仅 gRPC / elastic EP 不支持），decode 当前为 1、stream_interval 1。
- 02:35 用户决定：C176 后只重启 decode 加 `--tokenizer-worker-num 4`，预热后测 C144/C160/C176，再据数据分析瓶颈后决策（2P1D 推后）。
- 03:12 **t2a-best C176 完成**：21,218 tok/s/GPU（C160 −4.8%），完成 12,393，输出 169.8，TTFT p50/p90 31.5/41.8 s，GPU 命中 67.1%、
  host 26.4%，首 token 前在途 104.8，decode 有效并发 53.7、每 rank 在跑 6.41。census 0 致命项。平台右边界在 C160。
  后台 sampler（`scripts/sampler.py`，每 60 s 记录引擎延迟直方图与各容器进程 CPU，02:47 起）在 C176 窗口后 16 分钟（3,022 请求）的拆分：
  prefill 到达→完成 5.87 s（排队 1.86、bootstrap 0.98、传输 0.70 s，7.55 GB/请求）；decode 分配 KV / 排队 0，TTFT 34.4 s，ITL 14.3 ms；
  decode 主进程（HTTP/TokenizerManager）CPU 101%（饱和），scheduler 约 189%，detokenizer 2%；prefill 主进程 33–35%。
  data_parallel_controller 在所有引擎都是 100%（空闲时也是，属于忙等轮询，不是瓶颈指标）。
- 03:14 `tune_round.sh` 换为按拓扑计算 prefill 数的版本；`config/round.r2.sh` 增加 `EXTRA_ARGS_decode_0="--tokenizer-worker-num 4"`、
  `DECODE_LOG=decode-0.after-tok4.log`；队列：restart_decode（TAG=tok4）→ 30 分钟预热 warm-tok4 → t4-tok4 C144 / C160 / C176。
- 03:16 decode 重启开始。
- 03:20 decode（`--tokenizer-worker-num 4`，4 个 multi-tokenizer worker 注册成功）重启完成，03:20:55 开始 30 分钟预热 warm-tok4。
- 03:37 / 03:53 **warm-tok4 与 t4-tok4 C144 均失败**（AgentX 失败请求超过 10% 阈值中止：C144 54/144 = 37.5% 无内容响应，TTFT 232 s）。
  原因：重启后 decode DP4 收不到 KV —— decode 日志 596 条 "Decode transfer failed"，其中 595 条在 DP4（"Failed to get kvcache from
  prefill instance"），prefill 侧报 "Failed to send kv chunk to 10.245.157.237:35223"，到该 decode 会话反复 failed / 经探测恢复；
  从重启后第一分钟（03:21）开始。网卡链路正常。R2 按 KV 需求选 rank：失败请求立即释放 KV，DP4 看起来最空，吸入更多请求，
  失败比例被放大。tune_post 因有部分结果文件仍记为 "done"——已修：AgentX rc≠0 时记 FAIL 并停队列。
- 04:02 停掉队列与正在进行的 t4-tok4 C160（中止，记 abort）。新增 `scripts/transfer_smoke.py`（96 个并发约 8k token 请求，检查非空响应
  与 decode 传输失败）：在故障 decode 上 15/96 返回 HTTP 500（能检出）。
- 04:09 同配置（tok4）再次重启 decode（TAG=tok4b）：冒烟 96/96 非空、decode / prefill 传输失败均为 0 → DP4 故障为那次重启的偶发问题，
  非 tokenizer worker 所致。04:11 重新开始：warm-tok4b（30 分钟）→ t4b-tok4 C144 / C160 / C176。以后每次重启引擎后先跑传输冒烟再测量。
- 04:25–04:43 warm-tok4b 测量阶段的初步读数（sampler，10 分钟约 3,100 请求）：**decode 侧 TTFT 2.6 s（此前 29–34 s）**，prefill 到达→完成 3.1 s，
  请求速率约 5.2 req/s（此前约 3.7），decode 每 rank 在跑约 15、KV 使用率约 0.78，服务端 ITL 25.9 ms（batch 变大，ITL 上升）。
  但 04:24 起 decode 持续出现传输失败（101 次，DP0 94 / DP7 7），主要来自 prefill-1 → decode DP0 的会话（prefill-2 为 0），失败率约 3%。
  判断：只重启 decode、prefill 长期运行时，prefill 侧保留的旧传输连接状态导致部分 prefill→decode 通道失效（两次 decode 重启都出现，
  而只重启 prefill 的 2a 无此问题）。04:44 停掉预热（decode 已承载约 20 分钟真实负载），只重启 3 个 prefill（decode 保持预热）。
- 04:44 / 04:53 restart_prefills（TAG=tok4p）被执行了两次（工具层面重复执行）：第一次已把旧 prefill / router 退役并启动新 prefill-0/1，
  第二次又把它们停掉并在改名时冲突退出。2a 期间日志完整保存在 `prefill-N.before-tok4p.log`。04:54 改为在 137 后台单次执行（TAG=tok4q）。
- 05:14 prefill 重启完成（tok4q；池大小核对通过：GPU KV 2,433,088、host 400/200/200 GB）。05:15 传输冒烟（192 并发 × 8k token）：
  192/192 非空、decode / prefill 传输失败 0、8 个 decode rank 都有 batch。05:16 开始 t4c-tok4 C144 → C160 → C176（decode 不再单独预热，
  用 C144 的 decode 每步耗时核对是否已在稳态）。
- 05:30 **prefill-2 挂起**（t4c-tok4 C144 预热阶段）：05:30:04 后 prefill-2 不再出 batch，引擎指标显示其 queue 15 / inflight 10 / running 9，
  137 card4–7（prefill-2）GPU 占用 100%；py-spy（`runtime/runs/r2/tune/hang-prefill-2-0538/py-spy.txt`）显示 4 个 scheduler 主线程都卡在
  `process_batch_result_disagg_prefill` → `torch.cuda stream.synchronize()`，即 GPU 上的计算不返回（集合通信死锁或 kernel 卡死），日志无报错。
  挂起前最后的日志是 aiter 对未调优 GEMM 形状（M=652）的提示。客户端卡在预热 1,441/1,598、38 在途。05:38 中止 t4c-tok4 C144（abort）。
  `restart_prefills.sh` 增加 `INSTANCES=` 只重启指定实例；05:39 只重启 prefill-2（TAG=hang1）。
- 05:39–06:08 prefill-2 重启第一次一直空等：挂起后 card4–7 残留约 2 GiB 驱动层显存（`rocm-smi --showpids` 无进程占用），超过脚本 2048 MiB 阈值。
  阈值改为 4096 MiB 后 06:08 重启（TAG=hang1b），06:12 就绪；06:13 传输冒烟 192/192、0 失败。06:14 开始 t4d-tok4 C144 → C160 → C176。
- 07:05 新增 `scripts/hang_check.py`（prefill 有排队 / 在途请求但超过 180 s 无 "Prefill batch" 即判定挂起），纳入每次巡检；
  判定挂起时：保存 py-spy → 中止该轮 → 只重启该实例（`restart_prefills.sh INSTANCES=`）→ 传输冒烟 → 重测该点。若在后续高并发点再次出现，
  先停下与用户讨论（可能与 tok4 后到达 prefill 的请求速率提高约 35–40% 有关）。
- 07:36 **t4d-tok4 C144 完成（decode `--tokenizer-worker-num 4`）**：**25,269 tok/s/GPU**（2a C144 22,238，+13.6%；此前最优 22,340，+13.1%），
  输出 208.9（+10.5%），完成 15,177（+12.3%），TTFT p50/p90/均值 2.1/5.4/3.2 s（11.4/20.4/12.4），ITL p50/p90 21.1/25.3 ms，
  **P50 / P90 interactivity 47.4 / 39.6**（63.1 / 31.0）——P50 略低于 50。decode 有效并发 86.3（66.1）、每 rank 在跑 10.3、KV 0.62、
  KV>90% 2.8%；首 token 前在途 13.2（45.9）。sampler 全窗口（15,004 请求）：prefill 到达→完成 3.11 s，decode TTFT 2.92 s（差距消失），
  服务端 ITL 20.8 ms；decode 4 个 tokenizer worker 合计 106%（最忙 29%），scheduler 约 180%。census 0 致命项、无传输失败、无挂起。
  decode 对齐负载每步耗时 @8/@10/@12 = 41.5/55.4/67.5 ms（预热参照 40.0/52.x/63.x，偏慢 4–6%，KV 使用率更高、上下文更长时每步更重）。
  结论：TokenizerManager 饱和确实是此前吞吐平台的原因；修复后约束转到 decode GPU（ITL），C144 已越过 P50 interactivity 50。
- 08:00 口径核对：AgentX interactivity 用 aiperf Full-Response ITL = 每请求（首个有内容 chunk → 请求结束）/（输出 token − 1），取请求分布的
  P50 / P90 后取倒数。P8D4 最优（23,051，C80，8+4 卡，`llying/b4-align-crsuse2-20261008/results/b4-crsuse2-137-135-d4-ponly-g64-m090-r2-20261010T0636Z`）：
  ITL p50/p90 21.1/24.3 ms → P50/P90 interactivity 47.4/41.1，TTFT p50/p90 4.0/16.1 s，decode 有效并发约 47.5（输出 2,296 tok/s × TPOT 20.7 ms），
  约 11.9/rank；R2 前一版 21,081 为 ITL p50 19.9 ms、P50 50.3。tok4 C144：每 rank 约 10.8、ITL p50 21.1 ms、P50 47.4 —— 与 P8D4 最优同一负载 / 同一 ITL，
  每个 decode rank 在 ITL 约 20 ms 时约承载 10–11 个请求。prefill 服务不报 TTFT（不流式输出），只报 e2e / queue / bootstrap / transfer；decode 服务的
  `time_to_first_token` 从其收到请求起算，约等于客户端 TTFT。
- 09:08 **t4d-tok4 C160 完成**：**28,479 tok/s/GPU**（C144 +12.7%），输出 223.5（+7.0%），完成 16,293（+7.4%），平均输入 125.9k（+5.0%），
  **P50 / P90 interactivity 49.4 / 44.7**（47.4 / 39.6），ITL p50/p90 20.2/22.4 ms，TTFT p50/p90 2.5/7.1 s，GPU 命中 80.5%、host 13.9%，
  decode 有效并发 87.5、每 rank 在跑 10.42、KV 0.68、KV>90% 11.7%。sampler：prefill 到达→完成 4.03 s，decode TTFT 3.73 s，ITL 19.7 ms。
  按判定标准（≥ C144 + 2%）继续 C176（队列自动开始）。说明：变体 t1c-best / t2a-best / t4d-tok4 的 router 开关相同（R1 + P 亲和 + R2），
  R1 自 t1-r1 起每轮都开着。
- 09:20 C144 vs C160（tok4）复核：同一最忙 rank 负载下 decode 每步耗时 C144 比 C160 / 预热参照（t2a C144）慢 3–8%（@8 41.5 vs 38.6 / 39.7 ms，
  @10 55.4 vs 53.0 / 53.1，@12 67.5 vs 66.0 / 63.7）；其余条件相同（decode 有效并发 86.3 / 87.5，每 rank 在跑 10.32 / 10.42，各 rank 平均 / 最忙
  在跑比 0.70 / 0.71，输出长度均值 999 / 996）。→ C144 是重启后首轮、decode 尚未完全进入稳态，其 ITL（21.1 ms）、interactivity（47.4）
  与吞吐都偏保守；C176 后在已预热栈上重测 C144。prefill 忙碌时间占比（按日志估算）C144 35–36%、C160 40–41%（单 rank 32–49%）。
  Mooncake "Process failed for slice … transport retry counter exceeded"（64 KB RDMA WRITE 分片在 RC 层重传耗尽，Mooncake 重提交，
  retry_cnt 0/9）：C160 期间 2 条，请求级传输失败 0；过去两天零星出现，作为观察项。
- 09:45 t4d-tok4 C176 中段（09:27:40 进入测量）：decode 各 rank KV 平均 0.90–0.92、峰值 1.00，每 rank 在跑约 14、等预分配 0.3–0.5 ——
  decode KV 在 C176 打满。09:32:49 prefill-0 一次分片失败后，09:33:24–26 decode DP7 出现一串请求级传输失败（约 12 条），仅此一次；无挂起。
- 09:45 用户：不重测 C144（不影响 C160 为当前最高点的判断）；等 C176 与报告整理完再一并提交（只提交 llying/ 下本实验与 config-compare 两个目录）。
- 10:38 **t4d-tok4 C176 完成**：**30,562 tok/s/GPU**（C160 +7.3%），输出 232.5（+4.0%），完成 17,044（+4.6%），平均输入 129.2k（+2.6%），
  P50 / P90 interactivity 48.9 / 44.7，ITL p50 20.5 ms，TTFT p50/p90 3.1/11.1 s，GPU 命中 75.7%、host 18.6%；decode 有效并发 92.1、每 rank 在跑 11.0、
  KV 0.76、KV>90% 37.7%、等预分配 0.52；sampler：prefill 到达→完成 5.6 s（排队 1.23、bootstrap 0.97），decode 分配 KV 等待 0.97 s、TTFT 5.25 s、
  ITL 19.9 ms。census：窗口内 09:32–09:33 一次分片失败后的成串传输失败（约 10 条），其余为窗口结束取消。吞吐仍在上升，decode KV 开始成为约束。
  队列已无后续步骤，栈空闲等待决策。
- 10:44 t4d-tok4 C192 开始（C176 仍在上升）。10:52 提交并 push：`5a9f4af8`（全部实验内容）、`3926d81f`（复现：`config/best.sh`、
  `scripts/reproduce_best.sh`、`REPRODUCE.zh-CN.md`；`hang_check.py` 改为从本轮开始时刻起算，修正轮次刚开始时的误报）。
- 10:55 用户离开 3–4 小时，要求以推高吞吐为目标自主推进。队列追加：C192 后从活栈拿掉 prefill-0（2P1D，16 卡，decode 不动）→
  t5-2p1d C176 → C160。依据：prefill 计算忙碌约 35–45%，吞吐受 decode 约束；总吞吐降幅 < 20% 即每卡吞吐提高。
- 11:31 **prefill-2 第二次挂起**（t4d-tok4 C192 测量中）：同一签名 —— 4 个 scheduler 都在 `process_batch_result_disagg_prefill` →
  `stream.synchronize()`，137 card4–7 GPU 100%，挂起前为 aiter 未调优 GEMM 形状提示（M=404 / 3080 / 193），最后一个 batch 为 6 个请求、
  87 万缓存 token（`hang-prefill-2-1140/py-spy.txt`）。连锁：decode 为其在途请求预分配的 KV 被占住（KV 0.93–0.96、等传输 7–14/rank），
  decode 产出降到约 2,500 tok/s。两次挂起都在 prefill-2（137 card4–7），prefill-1 / prefill-0 从未挂起 → 怀疑该组 GPU 或只在该实例触发的路径。
  11:40 中止 C192（abort），改为拿掉 prefill-2 构成 2P1D（prefill-0 + prefill-1，decode 不动）；decode 残留状态清零，传输冒烟 192/192。
  11:46 开始 t5-2p1d C176 → C160（每卡按 16 卡计）。
- 11:31 **prefill-2 第二次挂起**（t4d-tok4 C192 测量中）：同一签名 —— 4 个 scheduler 都在 `process_batch_result_disagg_prefill` →
  `stream.synchronize()`，137 card4–7 GPU 100%，挂起前为 aiter 未调优 GEMM 形状提示（M=404 / 3080 / 193），最后一个 batch 为 6 个请求、
  87 万缓存 token（`hang-prefill-2-1140/py-spy.txt`）。连锁：decode 为其在途请求预分配的 KV 被占住（KV 0.93–0.96、等传输 7–14/rank），
  decode 产出降到约 2,500 tok/s。两次挂起都在 prefill-2（137 card4–7），prefill-1 / prefill-0 从未挂起 → 怀疑该组 GPU 或只在该实例触发的路径。
  11:40 中止 C192（abort），改为拿掉 prefill-2 构成 2P1D（prefill-0 + prefill-1，decode 不动）；decode 残留状态清零，传输冒烟 192/192。
  11:46 开始 t5-2p1d C176 → C160（每卡按 16 卡计）。
- 12:29 t5b-2p1d C176 重跑（新 router）。前 10 分钟两 prefill 均衡；12:40 起在预热回放阶段又失衡：prefill 的 bootstrap 队列积压
  （prefill-0 76、prefill-1 21，平均 bootstrap 20 s），decode 调度器预分配 / 传输队列为 0、tokenizer worker 合计 44%（非瓶颈），
  卡点在 prefill↔decode bootstrap 握手；13:05 积压已清零，进入测量。C192 部分数据（挂起前 26 分钟，sampler）：5.03 req/s、输出约 4,690 tok/s、
  decode TTFT 10.2 s，对比 C176 同期 5.19 req/s、约 4,840 tok/s、5.0 s —— C192 未超过 C176，3P1D 当前配置约在 C176 到顶（decode KV 约束）。
- 13:10 用户决定：2P1D C176 测完作 16 卡参考 → 查 prefill-2 挂起根因（GPU 组对调）→ 3P1D + decode mem 0.92 或 1P(TP8)1D。
  合并为一次重启：`restart_all_engines.sh`（先 decode 后 prefill，冒烟）+ `config/stage-mem092.sh`（decode mem 0.92；prefill-1 ↔ prefill-2
  GPU 组对调）→ 30 分钟预热 → t6-mem092 C176 / C192。若再挂起：跟着 card4–7（现 prefill-1）→ 疑 GPU；仍在 prefill-2（现 card0–3）→ 疑实例路径。
  需有 root 权限的人协助查看 137 card4–7 在 05:30、11:31 前后的 dmesg / RAS（普通用户读不到）。
- 14:43 **t5b-2p1d C176 完成（2P1D，16 卡：prefill-0 + prefill-1，decode tok4）**：28,900 tok/s/GPU（3P1D C176 30,562，−5.4%；总吞吐 −24%），
  完成 13,575（−20%），输出 230.7 tok/s/GPU（232.5），平均输入 122.6k（129.2k），P50 / P90 interactivity 54.8 / 49.2，TTFT p50/p90 8.7/46.1 s。
  瓶颈转到 prefill：GPU 命中 50.3%（75.7%）、host 40.1%，prefill 排队 10.2 s、bootstrap 4.2 s、到达→完成 20.4 s；两实例仍失衡（排队 prefill-1 8.1 /
  prefill-0 2.7）。decode：有效并发 63.3（92.1），每 rank 在跑 7.6，KV 0.82、KV>90% 61.8%、等传输 6.95 / 等预分配 1.94 per rank（KV 被等 prefill /
  传输的请求占住）。结论：2P(TP4)1D 在 C176 不提高每卡吞吐；1P(TP8)1D（每 rank GPU KV 约 +29%，无实例间失衡）未测。
- 14:47–14:50 `restart_all_engines.sh`：decode 以 mem 0.92 重启成功（KV 3,363,648 tokens/rank，+12.4%），但 restart_decode 的 router 校验按新拓扑
  要求 4 个 worker、prefill-2 尚未拉起 → 失败，队列停止（脚本缺陷）。14:50–16:08 栈空闲；期间巡检等待卡住约 3 小时未发现。
- 16:09 续做 prefill 重启：第一次因 `pgrep -f` 自匹配误判"已在运行"未启动；随后一次因 router 改名 `router-mem092` 冲突退出（已停 router）；
  16:10 以 TAG=mem092p 重启三个 prefill（prefill-1 → card4–7，prefill-2 → card0–3）。
- 16:59 prefill-1 / prefill-2 重启完成（TAG=mem092q；prefill-1 在 card4–7、prefill-2 在 card0–3）。此前 16:36 prefill-1 DP3 绑定 KV 事件端口 28028
  时瞬时 "Address already in use" 启动失败（端口随后空闲，偶发冲突），重启后正常。17:00 传输冒烟命令被中断（decode 81 条 "Aborted by AbortReq"
  为客户端取消，非传输故障）。**17:00 – 次日 02:10 空档约 9 小时**：我这边会话在冒烟命令被中断后结束，未继续，栈空闲。
- 02:12 重跑传输冒烟（后台执行）：192/192、0 失败。02:15 启动队列：warm-mem092（30 分钟）→ t6-mem092 C176 → C192。
- 02:14 warm-mem092 开始；**prefill-2 第三次挂起**（约 02:29，此时 prefill-2 已换到 card0–3、prefill-1 在 card4–7）：sglang scheduler 看门狗
  03:29:01 报 "Scheduler watchdog timeout (3600 s)" 后进程被 kill（容器 exit 137，OOMKilled=false）。**挂起跟着 prefill-2 实例走、不跟 card4–7 走 →
  基本排除 GPU 硬件，指向该实例特有的东西**（与 prefill-1 同配置，差别：同节点第二个实例、PID_OFFSET 60000 vs 30000、端口、启动顺序）。
  另：NFS 上 server 日志从 14 日 21:59 起停止写入（docker logs follower 退出，原因未查），hang_check / 巡检基于日志的部分自那时起失效。
  03:40 中止 warm-mem092（abort），队列停止。栈状态：decode（mem 0.92，tok4）、prefill-0、prefill-1（card4–7）在运行，prefill-2 已退出。
- 03:55–04:40 核对与 InferenceX 官方 AgentX 的一致性（InferenceX main `3babaeed` / `agentx-harness 1.0.9` vs 我们的 `918524ff` / aiperf `754356e`）：
  客户端场景约束、参数、数据集、吞吐口径、golden AL 都一致；aiperf 差异只影响截止后子 agent 轮次的收尾。tokenizer 提升依赖"sglang HTTP 服务自己流式输出"的
  架构（官方 AMD sglang-router recipe 同样设了 `tokenizer-worker-num 8` + `stream-interval 20`；NVIDIA 用 Dynamo 前端 + `skip-tokenizer-init`，无此瓶颈）。
  主要差异是官方要求每个并发点重新部署，而我们在复用的栈上测。详见 `RESULT.zh-CN.md` §5。同时把讨论中的专题分析整理进 `RESULT.zh-CN.md` §4，
  `REVIEW.zh-CN.md` §6 加容量模型更正。
- 04:55 用户：继续 debug 并修复 prefill-2 挂起。05:00 `log_follow.sh` 重启日志 follower（全部在 14 日 21:59 后退出；`/home` 余 136 GB）。
  排查（详见 `RESULT.zh-CN.md` §4.12）：IPC socket 撞名排除（只有 scheduler 绑定 `@hsa<pid>`，prefill-1 运行 12 小时 PID 才到 34141）；
  两个容器配置只差 GPU 组、PID_OFFSET、端口、缓存目录；第三次挂起的看门狗栈：主线程 `copy_done.synchronize()`、transfer_worker 等 event，
  DP3/DP1/DP0 KV 池 available 只有 1280/3200/1920 token。prefill-2 日志（去掉 HTTP 访问行）存到 `runtime/runs/r2/tune/hang-prefill-2-0329/`。
  主要假设：prefill `HSA_NO_SCRATCH_RECLAIM=0` + 显存紧张 → ROCr scratch 分配进入 retry 等待，与集合通信互锁（ROCr 7.2 源码 `AcquireQueueMainScratch`）。
  rocgdb 可以挂到运行中的 scheduler（约 5 s），能列出 HSA 队列读写指针和活跃 dispatch；libhsa-runtime 无符号，原生栈只有地址。
  新增 `scripts/hang_capture.sh`、`scripts/hang_watch.py`（按引擎计数器判断挂起，不依赖日志）、`scripts/start_hang_watch.sh`；在空闲的 prefill-1 上试抓一次，
  约 1 分钟（`runtime/runs/r2/tune/capture-test-prefill-1-0513/`）。
- 05:14 `restart_prefills.sh ROUND=r2 OVERLAY=config/stage-mem092-nsr.sh TAG=nsr INSTANCES=prefill-2`：只有 prefill-2 加 `HSA_NO_SCRATCH_RECLAIM=1`，
  prefill-0/1 保持 0 作对照。队列第 34 步改名为 warm-mem092b（旧结果目录已存在）。
- 05:18 prefill-2 健康（KV 2,433,088 token/rank、host 200 GB；launch_server 进程环境 `HSA_NO_SCRATCH_RECLAIM=1`，prefill-1 为 0；scheduler 的
  `/proc/<pid>/environ` 被 setproctitle 覆盖，不可读）。05:19 传输冒烟 192/192、0 失败。05:19 hang 监视器在 137 启动。05:21 队列启动 warm-mem092b。
  `start_queue.sh` / `start_hang_watch.sh` 的远端命令原来把整个 `a && b` 链放进后台，子 shell 占着 ssh 的 stdout，ssh 不返回（之前"启动队列"的
  后台任务一直挂到队列结束就是这个原因）；改为只把重定向后的命令放后台。
- 05:25 日志 follower 又断了：日志末尾 `write /dev/stdout: disk quota exceeded`。`/home` 是全体共用的 10 TB CephFS 子卷，用量 99% 上下浮动，
  写满的瞬间所有写入报 EDQUOT（21:59 那次同因）。新增 `scripts/bench-harness/follow_log.sh`：容器运行期间循环 `docker logs -f`，写失败退出后
  30 s 从文件最后一个时间戳续接（容器 ID 变化即退出，避免重启后两个 follower 写同一文件）；`engine.sh` 和 `log_follow.sh` 改用它，
  `log_follow.sh` 跳过已有 follower 的实例（可重复执行）。05:30 四个实例都换上新 follower。另：登录节点会话最长 24 小时，到点被强制关闭
  （`~/SHARED-HOST-LIMITS/`，10-13 16:51 那次监控空档的原因），长任务都要放在计算节点上。
- 05:50 发现 `follow_log.sh` 第一版把 `docker logs` 的 stderr 丢进了 /dev/null，而 scheduler 日志正是容器的 stderr：05:30 之后四个日志只有 HTTP
  访问行（prefill-0/1、decode 从 22:00 补写的那段也一样）。改为 `2>&1`，停掉 follower，把文件截回新 follower 开始写之前的位置
  （prefill-0/1、decode 截到 05:03 的大小，prefill-2 截到 05:30:46.435916178 那一行之后），重启 follower 从截断点补写；05:52 四个日志都已追到当前，
  batch 行齐全。warm-mem092b 已平稳度过冷启动段（05:36 进入 30 分钟测量），hang 监视器无停滞。
- 06:11 t6-mem092 C176 开始（06:29 进入 60 分钟测量）。
- 06:39 加固（用户同意）：① `follow_log.sh` 不再抄 `"GET /metrics` 访问行（约占一半行数，没有脚本读它），先停 follower 再换文件再重启
  （bash 增量读脚本，不能原地改运行中的脚本）；② `hang_capture.sh` 先写节点本地 `NODE_RW_ROOT/hang-captures/`，再带重试拷回 `/home`；
  ③ `hang_watch.py` 每 10 分钟调用一次 `log_follow.sh`，补起丢失的 follower；06:40 重启监视器。待做：`agentx_bench.sh` 的结果拷回加重试
  （此刻 C176 正在执行该脚本，等队列空闲再改）。
- 07:35 t6-mem092 C176 = 30,291（30,562 的 −0.9%，无收益；KV>90% 37.7% → 15.3%，每 rank 在跑 +4.6%，输出持平）：C176 的约束是 decode 每步计算。
  07:37 C192 由队列自动开始（07:59 进入测量）；它是按"KV 受限"旧假设预先排的，C176 出结果后没有及时复核（用户指出），用户决定跑完。
- 08:40 用户：先 PD 配比 / router / cache，单步计算放后面；write_back 先在 2P1D 上做（PLAN-TUNING 末节）。ROCm 上 write_back 在 kernel + page_first 下
  用 `transfer_kv_all_layer_mla_lf_pf`（与 write_through 同一 kernel，只是回写时机不同）。新增 `config/stage-2p1d-wb.sh`、`scripts/hicache_check.py` /
  `hicache_check.sh`（16 个约 4 万 token 的探针 → 约 2900 万 token 填充把探针挤到主机 → 重发探针，首 token 必须一致，且主机回读量 ≥ 探针 token 的一半）。
  队列追加第 37–41 步：drop prefill-1 → restart prefill-0/2（write_back）→ hicache_check → warm-2p1d-wb → t7-2p1d-wb C176。
- **07:57 共享 `/home` 写满（EDQUOT），编排进程全部退出**：队列驱动 `tune_queue.sh`、轮次脚本（`tune_round.sh` / `agentx_bench.sh` / `tune_post.sh`）写日志或状态失败后退出，
  `hang_watch.py` 在 07:56:45 后退出（Python 写失败抛异常）；sampler 早在 14 日 22:00 那次就退出了，一直没被发现。引擎和 AgentX 客户端写节点本地盘，不受影响：
  C192 客户端 09:07 正常跑完（exit 0）。日志 follower 按设计在 08:52 自动续上。09:15 用户问 C192 时才发现。
- 09:15 C192（t6-mem092）= **31,829 tok/s/GPU**（目前最高），P50 / P90 交互 47.2 / 43.3；decode KV>90% 45.9%、等预分配 0.47、等传输 3.38/rank。
  我此前判断"C192 不会提吞吐"是错的（见 RESULT §3.1 第 10 条的解读：比预热好的 C176 +4.1%，比今天 C176 +5.1% 里混有预热因素；未做重复测量）。
- 09:26–09:50 讨论（用户）：下一步改为 3P1D + prefill_complete + write_back，在 C192 上预热 60 分钟后测 C192（PLAN-TUNING 末节"09:40 修订"）；
  t7（2P1D + write_back）推后；decode mem 加到多少放在配置定下来之后。
- 09:55 `/home` 回到 76 GB 空闲，手工补救 C192：从 137 本地拷回客户端摘要，按 `tune_post.sh` 的步骤重建 census、flush check、prefill / decode 均衡、
  每步节奏，抓 router 指标（router 未重启），写 `results/t6-mem092/c192/RESCUED.txt`；`queue.done` 设为 36，队列里 t7 五行注释为 `#DEFERRED`。
  待做：编排脚本对 EDQUOT 的加固（写失败不退出、拷回失败等待重试、重启前检查 `/home` 余量），重启 sampler 和 hang 监视器。
- 10:02–10:50 讨论（用户）：(1) 先把 decode TP8 的 batch 推到最大（它也决定后面计算优化的 batch 区间），prefill_complete 是扩 decode 在跑容量最有效的手段
  （C192 时在跑请求占 KV 77%、每个 22.0 万 token，等传输占约 19%）；(2) 3P1D 扩容不再增长后再减 prefill 卡，减卡后按各形态自己的瓶颈选手段，不照搬
  prefill_complete。执行顺序写入 PLAN-TUNING 末节"10:50 执行顺序"。
- 10:56–11:01 防 `/home` 写满加固：`scripts/wait_home_space.sh`（队列每步开始前、轮次开始与收尾前等 /home ≥ 20 / 5 GB）；`tune_round.sh` 经
  `tee --output-error=warn` 写 agentx.log（agentx_bench 在 set -e 下写满即退出的根因）；`agentx_bench.sh` 的 runner.log 同样处理并显式取客户端退出码，
  结果拷回失败每 60 s 重试、最多 2 小时；`tune_queue.sh` 进度文件写失败重试；`hang_watch.py` 写失败改写节点本地 `NODE_RW_ROOT/hang_watch.fallback.log`；
  `sampler.py` 写失败把样本留在内存下轮补写；新增 `scripts/start_sampler.sh`（幂等，自身 stderr 写节点本地）。新增 `scripts/gate_throughput.sh`
  （结果低于阈值即让队列停下）。11:00 sampler、hang 监视器重启。
- 11:01 队列启动 t8（第 37–42 步）：`restart_all_engines.sh OVERLAY=config/stage-pcwb.sh TAG=pcwb`（3P1D + prefill_complete + prefill write_back，
  decode 也设 prefill_complete）→ hicache_check → warm-pcwb（C192，60 分钟）→ t8-pcwb C192 → 门禁（≥ 31,192）→ t8-pcwb C208。
- 11:02 第 37 步失败：新 decode 启动时报 `prefill_complete does not yet support attention tensor parallelism greater than one`。原因是 v0.5.21 的
  `_validate_prefill_complete` 用 `dp_size` 算 attention DP 宽度，而 v0.5.21 自己把 `--dp-size 8 --enable-dp-attention` 解析为 `dp_size=1`、
  `attn_dp_size=8`（server_info 可证），于是把完整 DP attention 误判为"attention TP > 1"；该校验的本意（`tp_size == attn_dp_size`）恰恰支持完整 DP attention。
  修法：运行时补丁 `patches/runtime-mounts/pd_disaggregation_hook.llying-attndp.py`（优先用 `attn_dp_size`），经 `config/mounts.pcwb.tsv` 只在
  t8 overlay 里挂到 prefill 和 decode（`PREFILL_MOUNTS_ADD` / `DECODE_MOUNTS_ADD`）。栈此时：decode 已退出、router 已退役、三个 prefill 仍是旧配置。
- 11:07 重跑（TAG=pcwb2，旧 TAG 的容器名已被占用）又因 `restart_all_engines.sh` 要向已退役的 router 查在线 prefill 数而退出（JSON 解析空串）；
  改为查不到 router 时用拓扑里的 prefill 数。11:09 第三次启动，decode 通过校验（启动参数 `prefill_complete`，补丁已挂入），继续重启 prefill。
- 11:43 三个 prefill 以 stage-pcwb 起来（KV 2,433,088 token/rank，host 400/200/200 GB），11:45 传输冒烟 192/192、0 失败。
- **11:51 `hicache_check` 失败，队列按设计停下**（`runtime/runs/r2/launch/hicache_check.pcwb.json`）：16 个探针中 5 个首 token 与第一次不同
  （winter→signal、river→vel、signal→window、needle→wal、c→button），logprob 最大差 3.28；第 2 步淘汰 1.87M token、回写 1.87M（write_back 在淘汰时回写，行为正确），
  第 3 步主机回读 0、GPU 命中 55.6 万 / 55.7 万，即**主机路径根本没被走到**，探针仍在 GPU 上被命中。两处检查设计问题：① 填充量按 2P1D（8 rank、19.4M token）设计，
  3P1D 有 12 个 prefill rank、约 29M token，2,900 万填充挤不出探针；② 探针问"随机词里出现最多的词"，各词几乎等概率，答案接近平局，数值差异就能翻转 top-1。
  但 3.28 的 logprob 差远大于一般数值噪声，不能直接当误报：需要用答案确定的探针（长文中的唯一键值检索）重做，并在旧配置（write_through + early）上做同样的对照，
  判断这是前缀缓存命中本身的问题（旧配置下也存在），还是 prefill_complete / write_back 引入的。11:51 之后栈空转（会话已断开，无人处理）。
- **15:40:15 我们在 137 上的 router、etcd、prefill-1、prefill-2 在同一秒收到 docker kill 并退出**（docker events 可见；同机 `dev_xb`、`crusoe-vector` 正常运行，
  不是守护进程重启；不是我们的脚本）。15:51 查看时 **135 ssh 连接超时**，prefill-0 状态未知；decode（138）仍在运行。未重启任何东西，等用户确认。
