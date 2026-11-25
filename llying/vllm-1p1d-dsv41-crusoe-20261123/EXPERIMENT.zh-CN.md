# DeepSeek-V4.1-Flash 1P1D：crusoe 复现与提升吞吐实验记录

日期：2026-11-23 起（UTC）　负责人：liyingli

AUS 上的基线与并发扫描见 `../vllm-1p1d-dsv41-20261120/`；与 B300 Dynamo-vLLM 828k 配置的对比与优化方向见
`../vllm-1p1d-dsv41-20261120/vs-b300-dynamo-828k/ANALYSIS.zh-CN.md`。本目录记录两件事：

1. 在 crusoe 上复现 AUS 的 1P1D 结果（同一镜像、同一配方）；
2. 在 P2D2（4 卡）不变的前提下推高 AgentX 吞吐的各轮实验。

## 1. 环境

| 项 | 值 |
|---|---|
| 集群 | crusoe spur，分区 `amd-spur-v2`，账户 / QoS `amd-hyperloom` / `amd-hyperloom-qos`（同一账户关联同时运行的作业数有上限，与本用户的其他作业共用） |
| 节点 | `crsuse2-m2m-v3-*`：MI355X × 8 的虚拟机，amdgpu-dkms 6.16.13，8 × Pensando ionic 400G，2.42 TiB 内存，本地 NVMe |
| 申请方式 | 半个节点：`--gres gpu:4 -c 118 --mem 1200G`（不独占）；P 用分到的两张低号卡，D 用两张高号卡 |
| 镜像 | `infera-vllm-dsv41:054985cc1e03-nightly-9c1d45c2`：kit `054985cc1` 在节点上构建，与 AUS 用的 `c33de6336` 内容相同；tar 在 `/shared_nfs/liyingli/images/` |
| kit | 分支 `llying/example/vllm_1p1d_dsv41_flash`，`cluster/cluster.crusoe.sh`（第 2 节） |
| 模型 | `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4.1-Flash`（revision `314113f8`），每次作业先拷到本地 NVMe（510 GB，约 190 s） |
| 压测 | InferenceX `bd334b4f`（AUS c768 跑分前最后一个 main 提交），agentx-harness 1.0.14，`cc-traces-weka-062126`，正式 3600 s、每 lane 10 个预热请求，decode golden 3.51 |

## 2. 与 AUS 的差异（都在 `cluster/cluster.crusoe.sh` 与 kit 提交 `3731453b8` 里）

| 项 | AUS | crusoe | 原因 |
|---|---|---|---|
| KV 显存 | GEM | GEM | 6.16.13 对 dma-buf 注册的显存计两次，且本集群没有 peer memory |
| KV 走的网卡 | 每张 GPU 用自己 PCIe 交换机上的网卡 | 两条 leg 的所有 GPU 都用 D 第一张卡对应的网卡（如 `ionic_2`），本机 RDMA 回环 | 每张 ionic 是一条独立路由的 rail，`ionic_i` 只能到 `ionic_i`；任意 P rank 都可能发给任意 D rank，而镜像里的 Mooncake（PyPI 版）不支持按目的端选本地网卡 |
| 宿主机 libionic | 不挂 | 不挂 | 宿主机库要 GLIBC 2.38，镜像是 2.35；镜像自带的驱动可用（容器内 9 个 RDMA 口都在） |
| UMBP 服务端可见的 GPU | 全部 8 张 | 只有 P 的两张（`UMBP_GPUS`） | P rank 按自己的 HIP 下标报告 KV 所在设备；P 不是 0 号起的卡时，服务端会映射到错误的物理 GPU，并在别人的卡上建上下文 |
| 端口 | 默认 | etcd 22379、router 28700、P 31300、D 31301 | 半节点会和其他作业共机，30000 与 sglang 默认端口相同 |
| GSM8K | 运行时从 HF 下载 | 本地 parquet 转成 JSONL（同一份 test 集） | 不依赖外网 |

## 3. 复现结果

| | crusoe c640 | AUS c640（重跑） | AUS c768 |
|---|---|---|---|
| run ID | `pd1p1d-c640-crusoe-1123T1307` | `pd1p1d-c640-1123T0714` | `pd1p1d-c768-hyper-1121T1745` |
| 节点 / 作业 | v3-105 / 306372（4 卡，节点上无其他作业） | n02-25（整机） | n02-25（整机） |
| 正确性 | GSM8K 0.980，needle 16k/64k/128k 32/32，kv-aware 8/8 | GSM8K 0.980，32/32，8/8 | GSM8K 0.975，32/32，8/8 |
| 总吞吐 tok/s/GPU | **463,531** | 471,698 | 459,183 |
| 输出吞吐 tok/s/GPU | 3,488 | 3,549 | 3,475 |
| 正式计分请求 / 错误 | 53,730 / 0 | 54,356 / 0 | 53,690 / 0 |
| TTFT p50 / p90（s） | 4.71 / 13.28 | 5.20 / 15.88 | 16.96 / 31.75 |
| ITL p50（ms） / 交互性 p50 | 30.7 / 32.6 | 29.1 / 34.4 | 30.5 / 32.8 |
| P 每 rank 排队 / 运行 | 35 / 3.6 | 43 / 3.8 | 116 / 3.9 |
| P GPU 命中 / UMBP / 重算 | 73.3% / 22.9% / 3.9% | 73.2% / 22.9% / 3.9% | 59.1% / 36.7% / 4.2% |
| P 每 rank 现算 tok/s | 35,975 | — | 38,737 |
| D 每 rank 运行（均值） / KV 使用率 | 211 / 0.81 | 202 / 0.78 | 210 / 0.81 |
| D 每步（迭代计数口径） | 每步 ~188 条 / 93 ms，与 AUS 拟合式 23.3 + 0.371 × 条数 相差 ±0.4% | — | ~189 条 / 93.5 ms |
| Mooncake 传输 均值 / p90 | 25 / 58 ms（单网卡回环） | — | 16 / 40 ms |

结论：crusoe c640 与 AUS c640 相差 −1.7%，在单轮波动内；P 侧各项与 AUS c640 几乎相同，D 侧与 AUS 拟合式吻合。
复现成立，本集群可以用作后续优化实验的平台。

## 4. 过程中的问题

| 时间（UTC） | 问题 | 原因 | 处理 |
|---|---|---|---|
| 10:00 | 作业 306107 准备阶段失败 | HF 缓存里的 GSM8K parquet 是指向 `blobs/` 的符号链接，只挂快照目录时容器内找不到 | 用 `realpath` 挂实际文件 |
| 13:01 | v3-157 上 docker 不可用（`docker=000`） | 该节点上本账号没有 `docker` / `video` / `render` 组 | 节点体检显式检查 `docker` 组；v3-157 加入排除名单 |
| 16:55 | c704（作业 306623，v3-180）直接以 golden 起栈时 decode 两个 rank 的 KV 为 30.3M / 31.4M token | 只在"直接以 golden 起栈"时出现，此前所有"先 block 再重启到 golden"的 run（这里与 AUS）都是 29,919,641 | 基线与各轮统一用 `GATE=1`（先 block 起栈过门禁再重启到 golden），D 的 KV 大小一致 |
| 17:37 | c704（作业 306623，v3-180）预热 30 分钟后失败：0 个正式请求，AIPerf 报 648 个 `ServerDisconnected` | **预热第一波没在宽限期内跑完，不是推理服务故障**。AIPerf 的预热先给每条 lane 发一个冷的长 prompt（平均约 12.8 万 token，长尾到几十万），要等这一波几乎全部返回才继续回放：c640 时 685 个第一波请求用了约 26 分钟（`sent` 一直停在 685），c704 时 755 个到 1800 s（`AGENTIC_WARMUP_GRACE_PERIOD`）还有 18 个没返回，AIPerf 拆掉预热并判失败。v3-180 上同时跑着 10 个不属于任何 spur 作业的容器（其他用户的 kernel agent，起始负载约 70），prefill 被拖慢 | ① 节点体检（`scripts/node_check.sh`）拒绝"没有其他作业却有外来容器"或启动前负载已高的节点，自动排除并重交（最多 4 次），v3-180 已排除；② c768 起预热宽限期用 3600 s（`WARMUP_GRACE`，InferenceX 自己的配方用 1800 或 3600，计分窗口不变）。**更正（11-24 07:40）**：这里的归因不对，见 4.2 节：17:37:26 AIPerf 进入预热第二阶段，1 s 后出现 9 个 `Connection reset by peer`，其中一个根请求失败导致中止，与宽限期无关 |
| 19:30 | R1 第一次（作业 306919，v3-032）起服务正常（KV 见 5.2 节），smoke 第一个请求时 decode 两个 rank 同时卡在 `execute_dummy_batch`，300 s 后 fatal | **decode DP2 死锁第 3 次**，这次两个 rank 都在空批次：请求 19:30:31 到达，prefill 19:30:33 算完；decode 在等 KV 期间两个 rank 以约 150 次/s 陪跑空批次（截至最后一次落盘 19:30:48.5，两边都是 2,290 次 FULL 6/1，完全对齐），约 19:30:52 起都不再返回；P→D 的 RDMA 握手 19:30:58 才完成（基线在请求后 1 s 内完成）。数据：`raw/pd1p1d-c768-r1-capacity-1123T1904/` | 三次死锁的共同点：decode 有请求正在等 P 的 KV、两个 rank 在频繁空转。同一 arm 重跑（R1b），流水线改为：起服务后立即启动 `hang_watch`（不再等 golden smoke），smoke 期间 decode 死掉则保留现场、重新部署一次；逐步记录改为每 1 s 落盘 |
| 20:57 | R2（MegaMoE，作业 307071，v3-135）AgentX 预热第一个请求时 **prefill** 卡住，21:02 fatal | **DP2 死锁换了一条 leg、换了 MoE 通信方式也出现**：DP1 执行 7,526 token 的新请求（eager），在 `sample_tokens` 超时；DP0 的空批次（eager，6 token）在 CPU 侧已完成，下一个空批次在 `execute_dummy_batch` 超时。与 decode 的三次一样，发生在空闲后第一个请求、一边真实一边空批次时，因此与 AgRs / MoRI 无关 | `pipeline5.sh`：两条 leg 都启动 `hang_watch2.sh`（kit 版本推广到 prefill），任一 leg 的引擎死掉都保留现场并重新部署一次 |
| 20:32 | R1b（作业 307146，v3-021，同机另有一个 4 卡作业）起服务 50 分钟未完成，作业被外部取消（`CANCELLED 0:9`，`Reason=None`） | **加载权重极慢**：P、D 都在同一段 shard（约第 14–20 个）停住，D 停 10 分钟、P 停 24 分钟，之后每个 shard 约 1 s；P 又在第 47 个（101.5 GB 的 Engram 表文件）停住。停住期间节点 NVMe 读为 0，P worker 在 `copy_` 里、CPU 约 200%。加载耗时与时段相关：13:00–17:00 两次 P 加载 55–91 s，17:47 以后三次 446 s、581 s、>2,000 s，与 arm、流水线版本、是否有同机作业都对不上，像是节点主机侧（页缓存 / 锁页内存）的间歇问题。取消发生在我用 `srun --overlap` 进作业查看进程后 30–60 s；spur 会因作业步异常结束整个作业，此后不再 `srun` 进运行中的实验作业 | 改用 `pipeline4.sh`：在 `pipeline3.sh` 基础上，AgentX 期间 decode 死锁时保留现场（`ax-try1/`）、以 golden 重新部署并重跑一次 AgentX（剩余时间够 3.5 h 才重跑） |
| 19:00 | c704 基线重跑（作业 306918，v3-086，同机另有一个 4 卡作业）预热中 decode 引擎 fatal：`engine core reported fatal failure` → `LLM became permanently unhealthy` | **decode DP2 死锁复现**（AUS 4 次中 1 次，见 `../vllm-1p1d-dsv41-20261120/DECODE_HANG.zh-CN.md`）。发生在预热第一波刚结束、第二波大量请求开始进入 decode 时；逐步记录显示卡住前一步：rank1 收进 4 个 KV 刚到的新请求（24 token，FULL graph 24/4），rank0 用同尺寸的空批次陪跑；随后两边都跑 1 个请求的空批次，下一步（rank0 一个 332,907 token 上下文的新请求，rank1 空批次）两个 worker 都没有返回，300 s 后超时 | 原因未定位。kit 新增只采集的 `engine/hang_watch.sh`（crusoe wrapper 在每个 AgentX 点自动启动），下次出现时抓两个 decode worker 的 py-spy 栈，确认各自停在哪个集合通信里 |

### 4.1 DP2 lockstep 死锁：现场证据（R5 第一次，01:07）

到目前为止：AUS 1 次（decode），crusoe 5 次（base704 decode、R1 第一次 decode、R2 MegaMoE prefill、R5 decode、R7 prefill）。R5 这次两个
watcher 都抓到了现场（`raw/pd1p1d-c640-r4-a4w4p-1124T0023/ax-try1/hang-dumps*/`）：

- **CPU 侧都已发完这一步**：两个 decode worker 的主线程都回到 `worker_busy_loop` 等下一个 RPC；两个 EngineCore 都在
  `_wait_for_output` 等 `sample_tokens` 的结果；两个 worker 的 `WorkerAsyncOutputCopy` 线程都卡在 `AsyncOutput.get_output` →
  `hipEventSynchronize`，即在等 GPU 完成这一步。
- **两张 decode GPU 都 100% 忙**（功耗 314 / 338 W，空闲约 250 W），P 的两张 0%：两边的 kernel 都在自旋，符合 GPU 上集合通信
  对不上、互相等待。
- **最后几步**（逐步记录每 1 s 落盘）：rank 1 最后一个真实步是 1 个刚拿到 KV 的新请求（上下文 73,635 token，FULL 6/1），
  rank 0 同时在跑空批次（FULL 6/1）；与 AUS 那次"一边新请求、一边空批次"相同。两边 CPU 侧的跨 DP 协商（批次描述、是否 replay、
  是否跳过草稿）在代码上都是对齐的（`dp_utils.py`、`model_state.py` 的 `_agree_across_dp`），FULL graph 步不走 SWA replay。
- 与 MoE 通信方式无关（AgRs 与 MoRI 都出现过）、与 decode graph 模式无关（FULL_AND_PIECEWISE 与 FULL_DECODE_ONLY 都出现过）。

**prefill 侧的另一种现场（R7 第一次，06:45，AgRs + a4w4，非 MegaMoE；`raw/pd1p1d-c704-r4-a4w4p-1124T0608/ax-try1/hang-dumps-prefill/`）：**

- 触发与 R2 完全相同：空闲 137 s 后 AgentX 的第一个请求（同一个 7,526 token 的 prompt），一个 rank eager 跑这个新请求，
  另一个 rank eager 跑 6 token 的空批次，两边 `execute_model` 都在 CPU 侧返回了。
- 之后：空批次那一侧的主线程已进入**下一步**的跨 DP 协商，卡在 gloo `cpu_all_reduce_`；真实请求那一侧的主线程空闲等 RPC，
  它的 EngineCore 在等上一步的输出。两边的输出拷贝线程、真实请求一侧的两个 UMBP 写入线程（`_MoriWorkerHandle._store_sync`）
  都在 `hipEventSynchronize` 等这一步的 GPU event。
- **两张 P 的 GPU 都 0%**（约 252 W，与空闲相同）：这一步在两张卡上都没做完，但没有 kernel 在跑，即 stream 在等一个永远不会
  到的依赖（跨 stream / 跨进程的 event），而不是 decode 那种两边 kernel 自旋。

**R8 第一次（08:31，decode，c576）是第二种表现出现在 decode 上**：两个 rank 空闲 161 s 后同时做了一个空步（请求在等 KV），
随后一个 worker 在 `execute_dummy_batch` → 跨 DP 协商的 gloo all-reduce 里等，另一个 worker 空闲等 RPC、它的 EngineCore 在等
上一步输出、输出拷贝线程在等 GPU event；两张 D 的 GPU 都 0%。`raw/pd1p1d-c576-r4-a4w4p-1124T0734/ax-try1/`。

因此至少有两种表现：GPU 100%（集合通信互等，R5 decode）与 GPU 空闲（stream 依赖不满足，R7 prefill、R8 decode）。GPU 空闲那种
三次都发生在空闲 130–160 s 之后的第一步。

**一个混杂因素**：今天出现死锁的 6 次都用 `pipeline3.sh` 及以后，它们比 c640 基线（`pipeline.sh`）多了两样诊断：
`--profiler-config torch`（每个进程启动时初始化 torch profiler；栈里 HIP 调用都经过 `rocprofiler::hip::hip_api_impl`）与
`vpd_steplog`（每 500 步调一次 `torch.cuda.mem_get_info`）。不带这两样的 2 次 crusoe 运行都没死锁，样本太少，不能下结论；
R7b（c704）用 `PROFILE_AT=0 STEPLOG=0` 跑，作为对照。共同点是"一边真实批次、一边空批次"，
prefill 的两次都是空闲后的第一个请求。

下一步：带 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL`（或 RCCL 的 collective trace）复现，看两个 rank 最后各自进了哪一个集合通信；
或在死锁时用 `rocgdb` 看 wave 停在哪个 kernel。二分顺序：关异步调度、关 DSpark、decode 改 PIECEWISE。

### 4.2 预热阶段切换时连接被重置（AIPerf 判"根预热请求失败"）

AgentX 的预热分两段：先给每条 lane 发一个冷 prompt，等它们几乎全部返回后进入第二段（cache pressure：所有 lane 同时开始
回放）。对比各轮的 `aiperf.log`：

| run | 进入第二段 | 之后 1 s 内 `Connection reset by peer` | 结果 |
|---|---|---|---|
| c704（17:37，v3-180） | 17:37:26.6 | 9 | 根请求失败，中止 |
| R4 c768 | 00:22:19.3 | 1 | 根请求失败，中止 |
| R7 第二次 c704 | 07:33:46.4（第一波 1,500 s 就基本完成） | 111 | 根请求失败，中止 |
| c640 基线、R1 c768、R5 c640、R6 c768 | — | 0 | 正常 |

即三次"预热失败"都是阶段切换时数百个请求同时发出、1 s 内有连接被对端重置，而不是 1,800 s 的超时（R4 那次我也误判为
第一波超时，见上）。router 日志没有任何报错；Rust router 不主动关闭空闲客户端连接（`infera/server/args.py`），上游连接池
空闲 4 s 即丢弃（低于引擎的 5 s keep-alive），容器 `nofile` 已是 1048576。尚未定位重置来自哪一层。`pipeline7.sh` 在 AgentX 期间
每 10 s 记录宿主机 `TcpExt` 计数器（ListenOverflows / ListenDrops / TCPAbortOnData 等）与 router 端口的已建立连接数，
下一次出现时可以直接看出是 listen 队列溢出还是连接被中止。

**补充（11-24 16:30，R8 第二次的现场）**：进入第二段（09:10:14.1）后 2.3 s 内 AIPerf 发出约 1,000 个请求（`sent` 623 → 1,609，
含子 agent 分支）；router 按约 120 个/s 串行选路（每个请求一次 prefill pick、一次 decode pick），日志里没有任何 warn / error，
一直到 AIPerf 中止。75 个重置在 AIPerf 日志里都打在 09:10:16.438–.442，但同一 worker 前一刻刚报过事件循环超时（70 ms），
"同一毫秒"可能只是事件循环卡住后集中处理，不能据此认定服务端同时发了 RST。router 与 AgentX 容器都是 `--network host`，
没有 NAT / docker-proxy。`hang_repro2.py` 每个 cycle 用 480 个同时发出的大请求复现，并在突发前、发出后 2 s、结束时读
TcpExt 计数器与 router 端口的 listen 队列，失败的请求逐个记录错误类型（`surge-errors.log`）。

**新线索（11-24 18:45）：被重置的是复用连接池里空闲连接的那批请求。** R8 第二次的 `profile_export.jsonl` 里：
- 75 个重置**全部**是切换后最先发出的请求（09:10:14.145–.196，前 50 ms），2.3 s 后一起失败；之后发出的近千个都正常。
- AIPerf 的 HTTP 默认值（`aiperf/common/environment.py`、`transports/http_defaults.py`）：连接池保留空闲连接 300 s
  （`KEEPALIVE_TIMEOUT`），socket 设 `TCP_KEEPIDLE=60`、`TCP_KEEPINTVL=30`、`TCP_KEEPCNT=1`、`TCP_USER_TIMEOUT=30000` ms、
  收发缓冲各 10 MB。第一波最后 300 s 里大约返回了 75–80 个请求，即切换时池里大约有这么多条空闲 keep-alive 连接；
  最先发出的请求正好复用它们，后面的请求只能新建连接。
- `hang_repro2.py` 每个请求都新建连接（urllib），所以 480 个并发也复现不出来。
- 另外，引起这次中止的第一波根请求之一（输入 81.3 万 token，跑了 23 分钟）在切换前一刻（09:10:14.088）返回了
  "没有任何内容"的响应（`InvalidInferenceResultError`）。
- 测试：`scripts/keepalive_reuse.py`，按 AIPerf 的 socket 选项建 N 条 keep-alive 连接，各发一个流式请求，闲置
  30–290 s 后在同一批连接上同时发请求，统计重置。很轻（短 prompt、`max_tokens=1`），可以从登录节点直接连运行中作业的
  router，不需要进作业。

**四次失败的共同模式（11-24 19:45，各轮 `profile_export.jsonl`）：**

| run | 重置数 | 被重置的请求 | 同轮第一波"无内容"的长请求（`InvalidInferenceResultError`） |
|---|---|---|---|
| c704 基线（11-23 17:37） | 9 | 切换后最先发出的 9 个（同一 40 ms 内），1 s 后一起失败 | 2 个（3.7 万、55.6 万 token） |
| R4 c768 | 1 | 切换后第 1 个 | 5 个（38.5 万–62.5 万 token） |
| R7 第二次 c704 | 111 | 切换后最先发出的 111 个，连续，1 s 后一起失败 | 1 个（44.4 万 token） |
| R8 第二次 c576 | 75 | 切换后最先发出的 75 个，连续，2.3 s 后一起失败 | 1 个（81.3 万 token） |

被重置的总是切换后最先发出的、连续的一批，之后的请求全部正常；而四次失败的轮次里，第一波都有超长请求返回了空内容。
从登录节点对 C2 的 router 做 keep-alive 复用测试（闲置最长 290 s），以及 4 次 480 并发突发，都没有复现（4.6 节）。
kit 给 AIPerf 设了 `AIPERF_HTTP_TCP_USER_TIMEOUT=900000`，其余是 AIPerf 默认值（连接池保留空闲连接 300 s，keepalive
60 s / 30 s / 1 次）。后续正式跑分都用 `pipeline8.sh`，再遇到时用 TCP 计数器和停顿报告直接定位。

**根因（11-24 20:15，R9 的 `tcp-counters.txt`）：router 的 listen backlog 只有 128，SYN cookie 校验失败的连接被内核重置。**
R9（4.7 节）在 20:06:38.702 切换，最先发出的 58 个请求（.750–.783）在 1.05 s 后被重置。切换前后这一个 10 s 采样间隔里，
宿主机 TcpExt 的增量：`ListenOverflows` +1,427、`ListenDrops` +1,427、`TCPReqQFullDoCookies` +232、`SyncookiesSent` +142、
`SyncookiesRecv` +288、**`SyncookiesFailed` +58（与重置数完全相等）**、`TCPSynRetrans` +767；之前每个间隔都是 0。
- 机制：约 1,000 个新连接在 1 s 内到达，accept 队列（128）满了，握手的最后一个 ACK 被丢弃（ListenOverflows）。客户端
  以为连接已建立并开始发请求；SYN 队列也满了，内核改用 SYN cookie。之后这些连接的包到达时，服务端已经没有它们的状态，
  只能按 cookie 校验，校验失败就回 RST（SyncookiesFailed）。最先发起连接的那批正好落在这个窗口里，所以被重置的
  总是"切换后最先发出的一批"。之前的 480 并发突发只溢出了 7–9 次，没到 cookie 失败的程度。
- router（`rust/router/src/main.rs`）用 `tokio::net::TcpListener::bind`，运行时 `ss` 显示的 backlog 是 128；宿主机
  `net.core.somaxconn=4096`、`tcp_syncookies=1`、`tcp_abort_on_overflow=0`。
- 修复（不改镜像、不需要管理员）：`instr/listen_backlog.c`，一个 LD_PRELOAD 库，把 `listen()` 的 backlog 提到 4096
  （只依赖 glibc ≤ 2.34 的符号，镜像是 2.35）；kit 的 `start_router` 新增 `ROUTER_PRELOAD`（kit 提交 `de90d8127`）；
  `pipeline9.sh` 默认启用，并在 summary 里报出 router 实际的 backlog。根本的修法是 router 自己用更大的 backlog
  （`socket2` / `TcpSocket::listen(4096)`），需要改 Infera router 代码。
- **验证（R9b，11-24 20:15，v3-085）**：起栈后 `ss` 显示 router backlog 4096；预热第一波加 cache-pressure 阶段 7,085 个请求
  0 错误，20:58:05 进入计分；整个预热期间 `ListenOverflows` / `SyncookiesSent` / `SyncookiesFailed` 都是 0（R9 同一时刻
  分别是 +1,427 / +142 / +58）。

### 4.3 整机 GPU 停顿（R7b，10:06）：DP2"死锁"可能是同一现象拉长到 300 s 以上

R7b 计分中 P 与 D 的两个 watcher 同时触发（`raw/pd1p1d-c704-r4-a4w4p-1124T0910/hang-dumps-*`）：**四张 GPU 都是 0%**，两个进程里
所有等 GPU 的线程（输出拷贝、UMBP 传输）都在 `hipEventSynchronize` 里忙等，2–3 分钟后一起恢复，没有引擎死掉。两个独立的引擎
同时停，说明不是 vLLM 的 DP 协商问题，而是驱动 / KFD 层面 GPU 在一段时间内不完成（或不上报完成）工作。R7、R8 那几次"GPU
空闲、event 不完成"的死锁与之形态相同，只是超过 300 s 被 vLLM 判死。

假设：我们注册给 GPU 的主机内存有几百 GB（UMBP 449 GB、两条 leg 的 Engram 表各约 189 GiB），走的是不钉住的 userptr；内存
规整（登录节点 `vm.compaction_proactiveness=20`，计算节点待确认）或页迁移触发 MMU 通知时，KFD 要撤下进程的全部队列、重新校验
这些内存后再恢复，规模这么大时可能要几分钟。之前"同一段 shard 上加载权重停住十几分钟、NVMe 空闲"也可能是同一类问题。
`pipeline_hang.sh` 在复现时每 10 s 记录每张 GPU 的忙碌度与内核的 compaction / migration / NUMA 计数器，用来对齐停顿时刻；
若确认，需要管理员在这些节点上关闭主动规整（`vm.compaction_proactiveness=0`）或改用 hugetlb 承载 UMBP。
**更正（11-24 17:10，见 4.4 节）**：计算节点上主动规整本来就是关的；停顿对应的是直接规整与 khugepaged 引起的页迁移。

### 4.4 死锁复现 hangB0（11-24 13:51，v3-107）：停顿每个 cycle 都有，且每次都伴随页迁移

作业 309197，`pipeline_hang.sh`，最佳 arm（`r4-a4w4p`）+ `STEPLOG=1 PROFILE_AT=1`；循环为空闲 150 s → 1 个 7,500 词的请求 →
48 个 1k–64k 词的突发请求。起栈 1,517 s（平时约 460 s），跑了 4 个 cycle 后于 14:43 被取消（`Reason=UserCancel`）：用户的
GPU 利用率 watchdog 会取消连续 45 分钟利用率低于 25% 的作业，而这个循环约 85% 的时间在空闲，加上 25 分钟起栈，正好触发。
数据：`raw/hang-r4-a4w4p-1124T1351/`（`host-telemetry.txt`、`slow-steps.txt`、`hang-dumps-decode/`）。

**1. 计算节点的内核参数**（遥测首行）：`kernel.numa_balancing=0`、`vm.compaction_proactiveness=0`、`vm.zone_reclaim_mode=0`、
THP `enabled=madvise defrag=madvise`。即主动规整与 NUMA 平衡本来就关着，4.3 节"请管理员关主动规整"不成立。
但**直接规整**（分配高阶页时当场规整，`compact_stall`）很多：突发开始后 10 s 内 240 次、迁移 17.5 万页；空闲时 khugepaged
还在合并大页（`thp_collapse_alloc`）。

**2. 停顿不是偶发，每个有负载的 cycle 都有。** 逐步记录里，上一步还在跑真实请求、下一次 `execute_model` 却隔了 5 s 以上的
（`scripts/slow_steps.py`；有请求在跑或排队时引擎是连续出步的，这种间隔就是上一步的 GPU 工作没完成）：

| leg | 停顿窗口（UTC） | 时长 | 窗口内页迁移 | 说明 |
|---|---|---|---|---|
| P（两 rank 同时） | 14:23:16–14:24:04、14:24:04–14:24:52、14:25:11–14:25:59 | 48.3、47.7、48.4 s | 9.4 万、5.0 万、4.6 万页 | 16,384 token 的满步，正常约 0.35 s；同时段 8 张卡忙碌度都是 0 |
| P | 14:30:10–14:30:35、14:34:17–14:34:44 | 24.1、26.1 s | 2.4 万、1.8 万页 | 空闲 150 s 后的第一步 |
| D rank0 | 14:24:04–14:24:33、14:24:57–14:25:47、14:26:04–14:27:39、14:35:19–14:35:44 | 28.6、50.0、95.0、24.8 s | 5.0 万、6.5 万、1.8 万、1.1 万页 | 1 个请求的 FULL graph 步，正常约 4 ms |
| D 两 rank | 14:30:34–14:31:46 | 71.7 s | 17.9 万页 | |

合计 D rank0 停了 270 s、P 每个 rank 约 195 s，而 4 个 cycle 一共约 20 分钟。**唯一没有停顿的 cycle 4（14:38:15–14:38:52，
GPU 一直忙）期间页迁移和直接规整都是 0。** 这些停顿只有超过 60 s 的那一次（D，14:26–14:27）触发了 watcher：
rank0 的输出拷贝线程在 `hipEventSynchronize` 里忙等，rank1 在 `execute_dummy_batch` 的 gloo all-reduce 里等 rank0，
8 张卡都是 0%，约 95 s 后自己恢复。现场与 R8 的"死锁"完全相同，只是没超过 vLLM 的 300 s。
P 日志里 14:23:03–14:23:12 有几次 JIT 编译，但之后的停顿期间没有，排除 JIT。

**3. 判断**：DP2"死锁"、R7b 的整机停顿、这里每个 cycle 的几十秒停顿是同一件事，只是长短不同；超过 300 s 时 vLLM 判死。
停顿与页迁移（直接规整、khugepaged 合并）一一对应。工作假设：迁移碰到注册给 GPU 的、没有钉住的主机内存（userptr，
Engram 表每条 leg 约 189 GiB）时，MMU 通知让 KFD 撤下该进程的全部 GPU 队列，并在重新校验整段 userptr 后才恢复，规模这么大要
几十秒。这也会拖累正式跑分（R7b 那 3 分钟就让 c704 作废）。

**4. 下一步（已提交，`pipeline_hang2.sh` + `hang_repro2.py`）**：
- 复现循环改为空闲 150 s → 唤醒请求 → 突发 480 个带共享前缀的大请求（同时用来复现 4.2 节的连接重置）→ 300 s 持续负载
  （48 个并发、1k–64k 词）。GPU 大部分时间在忙，不会再被 watchdog 取消；起栈后先做 JIT 预热。
- 遥测改为每 2 s 一次（`scripts/host_telemetry.sh`），并加入 KFD 每个进程的队列撤下时间
  `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms`（不需要 root）：停顿时它若涨几十秒，就直接证明是 KFD 撤队列。
- A/B：hangC0（基准，同 hangB0 的诊断设置）与 hangC1（`THP_OFF=1`：容器内所有 Python 进程
  `prctl(PR_SET_THP_DISABLE)`，`instr/thp_off.pth`，不需要 root；这些进程不再有 THP 缺页分配，khugepaged 也不再合并它们的内存）。
  比较停顿次数与总时长、`compact_stall`、`evicted_ms`。
- **根因线索（vLLM 源码）**：我们的配置是 `EngramConfig(cpu_offload=True, dp_shared_memory=True)`，每条 leg 两张 Engram 表
  各 94.42 GiB 放在 `/dev/shm` 文件里，DP 两个 rank 各自 `cudaHostRegister`（`common/engram.py` `share_shm_table`）。
  这些是普通的可迁移页，KFD 按 userptr 跟踪，所以直接规整一旦迁移其中的页，就会触发 MMU 通知并撤队列。vLLM 有现成的
  opt-in `VLLM_ROCM_ENGRAM_HIP_VMM=1`：表改放在驱动分配的 HIP VMM 主机内存里（`amd/ops/engram_host_table.py`，dmabuf FD
  在 DP rank 间共享，GPU 用 2 MiB 页映射），不会被迁移、也不是 userptr。新 arm `arms/r4-hipvmm.sh`（两条 leg 都打开，
  数值不变）做 hangC2，替换排队中的 hangC1（`THP_OFF` 留作后备）。

### 4.5 hangC0（11-24 16:58，v3-196）：停顿就是 KFD 撤队列，由页迁移触发

作业 309439，`pipeline_hang2.sh`（带负载的复现），最佳 arm，诊断设置同 hangB0；同机还有别人的一个 4 卡作业。起栈 2,168 s，
跑了 6 个 cycle（约 55 分钟）后，为了把作业槽让给 hangC2，我提前取消了它。数据：`raw/hang2-r4-a4w4p-1124T1658/`
（`slow-steps.txt`、`host-telemetry.txt.gz`、`repro-1.txt`、`hang-dumps-prefill/`）。

**1. 每次停顿都对应一次等长的 KFD 撤队列。** `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms` 在队列恢复时把这次
撤下的时长一次性加上去（这一轮的遥测把 8 张卡的值加在一起，下表已除以 8；之后的版本取各卡最大值）：

| 停顿（逐步记录） | 时长 | 同一窗口里结束的撤队列 | 触发前的页迁移 |
|---|---|---|---|
| D 两个 rank，17:31:02–17:31:39（起栈时的 dummy run） | 37.2 s | 36.4 s，2 个进程 | 3,758 页 |
| P 两个 rank，17:38:04–17:39:36（cycle 1 突发） | 91.8 s | 5 个进程，分别 45、48、52、86、91.5 s | 17:38:05–07 的 4 s 内 14.7 万页 |
| P，18:12:28–18:15:23（cycle 5 空闲 150 s 后的唤醒请求，用了 176 s） | 175.2 s | 5 个进程，分别 85、90、167、176、198 s | 18:12:28 起 4.8 万页 |

停顿恰好在最后一个进程恢复的那一秒结束（17:39:37、18:15:24）；停顿期间 8 张卡都是 0%。同机别人的作业的 GPU 进程
也一样，在这期间反复被撤 2–15 s，累计约 225 s，所以这是节点上普遍存在的现象，不只发生在我们的部署上。

**2. 结论**：内存规整迁移了注册给 GPU 的主机内存（userptr）里的页，MMU 通知让 KFD 撤下每个注册了这段内存的进程的
全部 GPU 队列；恢复前要重新校验整段 userptr（Engram 表每条 leg 189 GiB，shmem 不用 THP，即约 4,950 万个 4 KiB 页；
UMBP 主机池 418 GiB，服务端启动时注册它就用了 59.9 s），所以要 45–200 s；迁移若持续，恢复还要重来。超过 300 s，
vLLM 就判为"DP2 死锁"。4.1 节的两种现场都能解释：GPU 0%（队列被撤）；R5 那次 GPU 100%，可以理解为一个 rank 被撤，
另一个 rank 在集合通信里自旋等它。

**3. 连接重置没有复现**：6 个 cycle 每次同时发 480 个带 1.6 万词共享前缀的请求，全部成功；没有 listen 队列溢出
（ListenOverflows / ListenDrops 不变），TCPAbortOnData 每个 cycle 0–6 次。router 端口的 listen backlog 是 128
（`ss` 的 Send-Q），低于 somaxconn 4096，但这一轮没有溢出。所以单纯"几百个并发大请求"不足以触发，AgentX 那几次另有条件
（约 1,000 个请求、aiohttp 客户端、请求体更大），待查。

**4. 下一步**：hangC2（`VLLM_ROCM_ENGRAM_HIP_VMM=1`）。D 不用 UMBP，所以 D 的停顿只能来自 Engram 表：如果 hangC2 里
D 不再停、P 还会停，剩下的就是 UMBP 主机池。UMBP 可以用 hugetlb 承载（UMBP 有 `hugepages` 选项，但需要管理员预留
大页），或者由管理员设 `vm.compact_unevictable_allowed=0`，再由我们 mlock 这两块内存（容器里 memlock 已不限）。

### 4.6 hangC2（11-24 18:40，v3-178）：Engram 表改放 HIP VMM 主机内存后，GPU 停顿消失

作业 309447，`arms/r4-hipvmm.sh`（R4 加两条 leg 都设 `VLLM_ROCM_ENGRAM_HIP_VMM=1`），复现与诊断设置同 hangC0；同机同样有
`chaojhou` 的作业。四个 worker 的日志都确认 "94.42 GiB shared across DP replicas via HIP VMM"。跑了起栈加 4 个 cycle
（约 45 分钟）后，我取消它，把槽给正式跑分。数据：`raw/hang2-r4-hipvmm-1124T1840/`。

| | hangB0（`/dev/shm`） | hangC0（`/dev/shm`） | **hangC2（HIP VMM）** |
|---|---|---|---|
| 起栈耗时 | 1,517 s | 2,168 s | **617 s** |
| 超过 5 s 的停顿 | 4 个 cycle 里 11 次，24–95 s | 起栈加 6 个 cycle 共 3 次，37 / 92 / 175 s | **起栈加 4 个 cycle 共 0 次** |
| 同期页迁移 | — | 约 3,450 万页（累计计数） | 20 分钟 1,620 万页、直接规整 6.4 万次（不比 C0 少） |
| 我们 worker 的撤队列 | — | 45–198 s | **0.01–1.7 s**；只剩几次 2–5 s 的间隔，都对应 1.5–4 s 的短暂撤队列 |

- 迁移照常发生，撤队列也照样发生，但 worker 的撤队列只要毫秒到 1–2 s 就能恢复：Engram 表不再是 userptr，恢复时只需
  重新校验很小的范围。这直接确认了 4.4–4.5 节的机制，也说明修复在我们这边就能做，不需要改内核参数。
- 起栈从 25–36 分钟降到 10 分钟，Engram 表放置从约 100 s 降到 47 s。之前"加载权重时卡住十几分钟、NVMe 空闲"
  （第 4 节 R1b），很可能也是同一个撤队列问题。
- **UMBP 主机池仍然受影响**：按启动顺序应是 UMBP 服务端的那个进程（注册了 418 GiB 主机池），有一次被撤了 209 s
  （18:54:43–18:58:12）。P 侧的 UMBP 客户端检测到服务端无响应（"batch_exists got no reply within 1s"、"transfers are refused
  while N timed-out transfer(s) are still blocked in MORI"）就把 UMBP 标成不可用、改为重算，所以 P 的 GPU 不停，但这段时间
  UMBP 缓存失效（`umbp-warnings.txt`，54 条）。正式跑分里这样的情况：R5 c640 0 次，R6 c768 4–5 次，R7b c704 92–99 次。
  UMBP 贡献约 23% 的命中，这是吞吐损失，也解释了各轮之间的一部分波动。修复需要管理员：UMBP 有 hugetlb 选项
  （kit 里写死了 `UMBP_DRAM_USE_HUGEPAGES=0`），要在节点上预留约 214k 个 2 MiB 大页（449 GB）；或者设
  `vm.compact_unevictable_allowed=0`，我们再 mlock 这块内存。
- 连接重置：C2 的 4 次 480 并发突发全部成功；有两次 `ListenOverflows` +7/+9，说明 router 128 的 listen backlog 确实会溢出，
  但客户端重传后都成功了。从登录节点对 C2 的 router 跑 `keepalive_reuse.py`（80 条连接，闲置 30/90/150/240/290 s 后复用）
  全部成功，所以"空闲连接复用会被重置"至少在跨节点路径上不成立。下一次正式跑分用 `pipeline8.sh`（`pipeline7.sh` 加主机遥测
  和停顿报告），遇到重置时直接看 TCP 计数器。

下一步：R9 = `r4-hipvmm` c640 正式跑分（`run_opt6.sbatch` → `pipeline8.sh`，`GATE=1 PROFILE_AT=3300 STEPLOG=1`，与 R5 只差
HIP VMM 这一项）。验证真实负载下的稳定性和吞吐。

### 4.7 R9（11-24 19:27，v3-178）：HIP VMM 过了精度门禁；AgentX 在预热切换时被连接重置中止

作业 309628，`r4-hipvmm` c640，`pipeline8.sh`。数据：`raw/pd1p1d-c640-r4-hipvmm-1124T1927/`。
- 起栈 374 s。精度门禁通过：gsm8k 0.960（R4 0.965，n=200，差 1 题），needle 16k / 64k / 128k 全部 32/32，kv-aware 8/8。
- AgentX 第一波 1,410 s 基本完成（685 个里返回 682 个），第一波和切换期间**没有任何超过 5 s 的停顿，UMBP 失效 0 次**。
  20:06:38 切换时 58 个连接被重置，AIPerf 中止（根因见 4.2 节末尾：router 的 listen backlog 128）。
- 下一轮 R9b：同配置加 router backlog 修复（`run_opt7.sbatch` → `pipeline9.sh`，`GATE=0`）。R10（c704，`pipeline8.sh`，
  无 backlog 修复）为了让槽给 R9b 被我取消，c704 改用 `pipeline9.sh` 重排为 R10b。

### 4.9 UMBP 可用性是各轮吞吐差异的一阶因素（11-25 02:50）

按"遥测开始后第一个新出现的 KFD 进程就是 UMBP 服务端"（up.sh 先启动它）统计计分窗口内它被撤队列的总时长；P 侧的
"UMBP 无响应"警告与它的每次撤队列在时间上一一对应（如 R9b 21:08–21:14 的警告对应一次 200.6 s 的撤队列，R12 01:36–01:44
连续的警告对应一次 545 s 的撤队列）：

| run | 计分窗口内 UMBP 不可用 | 最长一次 | UMBP 传输失败 | 吞吐 tok/s/GPU |
|---|---|---|---|---|
| R10b c704 | 0 s（0%） | 48 s（计分前） | 724 / 5,530 万（≈0） | 494,398 |
| R9b c640 | 381 s（10.6%） | 201 s | 153 万 / 5,200 万（2.9%） | 465,373 |
| R12 c704（32k 步） | 1,035 s（28.8%） | 545 s | 264 万 / 4,120 万（6.4%） | 367,981 |

之前的轮次也有这个问题，只是没被单独看见：P 侧 UMBP 警告在 13 个旧 run 里出现过 6 个（c704 基线 7 次、c704 crusoe 2 次、
R1 c768 14 次、R6 c768 5 次、R8 c576 16 次、R7b 92 次，R7b 传输失败 65%）；作为参照的 crusoe 基线 c640 和 R5 c640
恰好是 0。Engram 修好之前，同一次页迁移也会撤掉引擎进程的队列，P 整体停住甚至被判死，UMBP 失效被更大的停顿盖住；
`pipeline8.sh` 之前也没有统计这类警告。结论：在 UMBP 主机池移出 userptr 之前，单轮吞吐的比较要同时看 UMBP 不可用时长。

### 4.10 R11 第一次 AgentX 时 decode 死锁（GPU 100%，与 KFD 无关）；历史上的同类问题（11-25 03:05）

- R11（作业 309932，`r4-hipvmm` c768，v3-006）第一次 AgentX 在预热中（02:41 起）decode 停住，02:46 `sample_tokens` RPC 300 s
  超时、引擎被判死，流水线保留现场并重新部署。watcher 现场（`logs/pd1p1d-c768-r4-hipvmm-1125T0156/ax-try1/hang-dumps-decode/`）：
  **D 的两张卡 100% 忙**（功耗 318 / 336 W，低于正常计算），P 的两张 84–85%；rank0 主线程空闲等 RPC、输出拷贝线程在
  `hipEventSynchronize` 等 GPU；rank1 第一次抓栈失败（EPERM）。02:39–02:47 **没有任何 KFD 撤队列，也没有成规模的页迁移**。
  这是 R5 那种"GPU 自旋"型死锁，HIP VMM 修不了它；4.5 节里"R5 可能是一个 rank 被撤、另一个在集合通信里自旋"的推测，
  被这次（没有撤队列）否定。之后重新部署的 R11 在第二次 AgentX 期间被我取消（用户选择修复前暂停正式跑分）。
- 历史（提交 `d3d58d16d`、`f21267357`、`40dcf1f9a`，`vllm-p4d4-crusoe-20261117`）：11-18 的 m12 卡死时内核报
  `evict_process_worker`（KFD 驱逐），当时怀疑新加的 1.9 TB"匿名 mmap + `hipHostRegister`"锁页内存，改为 `hipHostAlloc`；
  m13（仍是第一版）期间同节点驱逐计数又从 64 涨到 128，结论改为"大概率是 decode 显存压力"，decode 显存目标降到 0.93；
  之后 f1 / f2（`hipHostAlloc` + 0.93）都没有驱逐告警。两个改动同时做，没有分开验证；从今天的机制看，"大块可迁移内存注册给
  GPU"与当时的现象同样吻合。对 UMBP 的直接启示是：池子不要用"可迁移内存 + `hipHostRegister`"，而用驱动分配的锁页内存
  （`hipHostMalloc` / HIP VMM）。`umbp_standalone_server` 有环境变量 `UMBP_DRAM_HOST_REGISTER`（语义待确认，可能是关掉对池子的
  注册、改走 CPU 拷贝线程），可以先用复现验证。

### 4.11 UMBP 根本修复的验证：DRAM 池改用 `hipHostMalloc` 锁页内存（11-25 03:40 起）

- 验证用的小库 `instr/umbp_pinned_pool.c`（只预加载到 UMBP 服务端，kit 新增 `UMBP_PRELOAD_EXTRA` / `UMBP_ENV_EXTRA`，
  提交 `8b65e8825`）：只接管那一次"可读写、私有匿名、不指定地址、≥ 64 GiB"的 `mmap`，改为
  `hipHostMalloc(Portable | Mapped | NonCoherent)`；对这段内存的 `hipHostRegister` / `hipHostUnregister` / `madvise` 变成空操作，
  `munmap` 改为 `hipHostFree`；`hipHostMalloc` 失败时退回原路径并打日志。GPU 的 gather kernel 仍通过 `hipHostGetDevicePointer`
  的别名访问，`hipMemcpy` 仍走锁页路径。登录节点上用假 HIP 库测过三条路径（接管、失败回退、低于门槛不动）。
- 复现：`run_hang3.sbatch` → `pipeline_hang3.sh` + `hang_repro3.py`。在 `hang_repro2.py` 基础上，负载阶段 30% 的请求原样重放
  5 个 cycle 之前的请求（那时已挤出 P 的 GPU 缓存，只能从 UMBP 读），每个 cycle 从 P 的 `/metrics` 统计 UMBP 读写字节、
  成功 / 失败对象数、UMBP 命中占比，以及重放请求和新请求的延迟；结束时 `scripts/umbp_avail.py` 统计 UMBP 服务端的撤队列。
- A/B：hangD0（作业 310470，`UMBP_PINNED=1`）与 hangD1（作业 310471，`UMBP_PINNED=0`，原注册池），都是 `r4-hipvmm`、
  router backlog 修复、`BUDGET_S=9000`、`PROFILE_AT=0`。判据：D0 的 UMBP 服务端不再有长撤队列、P 侧 0 次 UMBP 警告；
  UMBP 读写吞吐、重放请求延迟不比 D1 差。通过后再做 MORI 补丁（DRAM 层新增驱动锁页分配方式）并重编 `umbp_standalone_server`。
- **D0 起栈（03:38，v3-006）**：UMBP 服务端的 428,199 MiB 池由 `hipHostMalloc(0x80000003)` 分配，用时 84.2 s；对它的
  `hipHostRegister` 被跳过，UMBP 报告 "2/2 devices gatherable"（GPU gather 路径可用）；P 的两个 worker 正常注册 GPU IPC，
  整体起栈 414 s。即 418 GiB 的驱动锁页分配在这个集群上可行。
- MORI 补丁（`instr/mori-v1.2.3.post1-umbp-pinned-dram.patch`，+87/−8 行）：新增 `HostBufferBacking::kHipPinned`
  （`hipHostMalloc` 失败时退回匿名页），已经是锁页内存的区域不再 `hipHostRegister`、只记录设备别名，`UMBP_DRAM_PINNED=1`
  沿 DRAM ownership 配置传到 `HostPageMemorySource`。`build_umbp.sbatch` 在镜像里编译 `umbp_standalone_server`（作业 310476，排队中）。
- **D0 前 3 个 cycle（04:10）：锁页池去掉了长时间撤队列，但 UMBP 服务端仍有短时间撤队列。** 起栈到 cycle 3 共 7 次，
  11.4 / 10.8 / 54.3 / 10.9 / 21.6 / 21.7 / 10.8 s（C2 注册池同期为 100–393 s、累计 1,419 s）；cycle 2 有 39,353 次 UMBP 写入失败。
  这 7 次**都没有页迁移、也没有别的进程同时被撤**，时长近似 10.8 s 的整数倍，所以不是 userptr 机制，是另一个触发源。
  P 卡剩余显存只有 3.1–4.2 GiB（逐步记录的 `mem_get_info`；D 卡 7.4 GiB），UMBP 服务端与 P worker 共用 P 卡，
  推测是显存压力下驱动把 UMBP 服务端的显存对象挤出、撤其队列（与 11-18 那次 decode 显存压力引起驱逐同类），尚未证实。
  遥测从 D1 起加了每张卡的显存占用、amdgpu / KFD 内核日志和每进程显存表（`host-telemetry.txt.dmesg` / `.procs`）。
- **D0 cycle 6–7（04:42）：重放请求走 UMBP 读路径，0 失败。** 每个 cycle UMBP 读 12.2 / 14.3 GiB（70,509 / 82,452 个对象），
  失败 0；P 命中构成 GPU 1.8% / 3.2%、UMBP 21.7% / 24.1%（重放的前缀确实已挤出 GPU 缓存）；重放请求延迟 p50 28.3 / 26.8 s，
  新请求 29.5 / 25.4 s（主要是 P 排队，不足以比较 UMBP 读性能，读吞吐等 D1 对比）。cycle 3 起约 45 分钟没有撤队列。
- 另一个待查的可能：锁页池是 TTM 分配的系统内存，计入 TTM 全局页数上限（默认约为整机内存的一半）；418 GiB 池加两条 leg 的
  Engram HIP VMM 表约 378 GiB，再加同机作业，可能逼近上限而触发换出。原注册池是用户页，不计入。遥测加了每张卡 GTT 占用和
  TTM / amdgpu 的上限参数。
- **D0 cycle 8–14（05:45）：重放仍全部成功（每 cycle UMBP 读 10–15 GiB，0 失败），但从 cycle 9 起每个 cycle 唤醒后 2–5 s
  （突发阶段）UMBP 服务端都被撤一次队列，11–33 s**；cycle 3–8 完全没有。cycle 9 正是池子写到 UMBP 高水位（0.9 × 418 GiB，
  每 cycle 写约 45 GiB）、之后每次写入都要先淘汰的时候。UMBP 的淘汰（`PageBackend::MaybeEvictToLowWatermark`）只在分配器位图里
  释放页，但持有 `mutex_` 排他锁一次从 90% 降到 70%，期间查询和读写都被阻塞，能解释"UMBP 无响应"，解释不了 KFD 撤队列计数。
  候选：P 卡显存压力（剩余 3–4 GiB；唤醒后突发让 P 的非 torch 分配短时冲高，驱动把同卡上 UMBP 服务端的显存对象挤出并撤其队列）、
  `DeviceGather` 描述符缓冲按需重新分配（池子满后片段变多）。正式跑分的稳态（R10b 计分窗口）里没有这类短撤队列。
  排了同配置、带新遥测（每卡显存 / GTT、内核日志、每进程显存）的 hangD0b（作业 310756）来区分。
- **D0 结束（06:03，17 个 cycle、139 分钟）**：0 次引擎死亡、0 次连接重置；重放共从 UMBP 读 152 GiB、877,611 个对象，
  **0 失败**；UMBP 服务端撤队列 20 次、累计 389 s、最长 54 s（C2 注册池约 45 分钟里 8 次、累计 1,419 s、最长 393 s）；
  P 侧 UMBP 警告 40 次、拒绝 6 次（都来自那些短撤队列）；一次 6.2 s 的 P 停顿（05:52:39，期间无迁移、无撤队列）。
  数据：`raw/hang3-r4-hipvmm-p1-1125T0336/`。对照组 hangD1（注册池，同一节点 v3-006）06:04 开始。
- **D1（注册池，v3-006，06:04 起）到 cycle 7（07:09）**：0 次撤队列，但这段时间节点很安静（前 30 分钟只迁移约 1.2 万页，
  D0 同期 9.4 万页），注册池还没受到页迁移的考验。重放阶段的 UMBP 读与 D0 对比（cycle 6 两组负载完全相同）：
  D0 / D1 读 12.2 / 11.9 GiB、重放延迟 p50 28.3 / 30.6 s；cycle 7 读 14.3 / 17.4 GiB、完成请求 587 / 640、新请求延迟 p50
  25.4 / 23.5 s（不同时段、同机负载不同）；两组都 0 读失败。即锁页池（non-coherent `hipHostMalloc`）没有看到明显的读性能损失。
  关键对比点是 D1 的池子写到高水位之后（cycle 9 起）：若 D1 也在每次唤醒时出现短撤队列，就与锁页池无关。
- **D1 到 cycle 14（08:06）：池子写满后仍 0 次撤队列，短撤队列与锁页池有关。** D1 累计写入约 498 GiB（超过 0.9 × 418 GiB 高水位，
  cycle 9 起每次写入都要先淘汰），cycle 9–14 唤醒突发 6 次、UMBP 服务端（pid 1050919，遥测里有它的 `evicted_ms`，值为 0）
  0 次撤队列、P 侧 0 次 UMBP 警告；同期 D0（锁页池，同一节点 v3-006）cycle 9–14 每次唤醒都撤一次（11–33 s）。
  两组的短撤队列期间都没有页迁移（D1 全程仅迁移 1.2 万页），所以 D1 的 0 次不是因为节点更安静。
  因此"UMBP 淘汰持锁"不是原因（D1 同样在淘汰）；剩下的候选是锁页池本身：TTM 全局页数上限 / GTT 记账（锁页池计入，
  注册池的用户页不计入）或 P 卡显存压力下与锁页池相关的驱动对象被挤出。D0b（作业 310756，锁页池 + 每卡 GTT / 显存 /
  内核日志遥测）用来区分。MORI 编译作业 310476 已经用户同意改到 `amd-agentx` 账户排队。
- **D1 结束（08:30，17 个 cycle、139 分钟）**：0 次引擎死亡、0 次连接重置、0 次超过 5 s 的停顿；重放共从 UMBP 读 166 GiB、
  954,065 个对象，0 失败；UMBP 服务端全程 0 次撤队列、P 侧 0 次 UMBP 警告（D0 为 20 次、累计 389 s）。每卡 GTT 占用峰值
  377 / 188 / 188 / 0 / 0 / 0 / 0 / 188 GiB（作为 D0b 的对照）。数据：`raw/hang3-r4-hipvmm-p0-1125T0604/`。
  注意：D1 全程只有约 1.2 万页迁移，原注册池的长撤队列（C2）在这台节点这段时间没有被触发，不能据此认为注册池没问题。
- **MORI 补丁编译完成（作业 310476，`amd-agentx`，08:58，v3-006）**：在镜像里 26 s 编完，rc=0，产物
  `$W/logs/build-umbp-310476/umbp_standalone_server`（5.75 MB，sha256 见 `raw/build-umbp-310476/`）。补丁代码（`HostMemAllocator` 的
  `hipHostMalloc` 路径、`UMBP_DRAM_PINNED`）静态编在二进制里，依赖的 `libmori_*.so` 与镜像原二进制相同，可用 bind mount 替换。
  但 D1 显示锁页池本身带来池子写满后的短撤队列，要不要用它替换预加载小库，等 D0b 的结论。
- **更正（11-25 09:40，D0b）：短撤队列也是页迁移触发的，`hipHostMalloc` 池在 KFD 里仍是 userptr。** 之前"短撤队列期间没有页迁移"
  的判断用的是撤队列**结束**时刻（`evicted_ms` 在恢复时才累加）附近的窗口，看错了地方。按开始时刻（结束减时长）取 ±5 s：
  D0b 的 15 次撤队列**每一次**都有内存规整（`compact_stall` ≥ 1）和页迁移（64–67,518 页）；同一作业随机 10 s 窗口里只有 6–7%
  有，偶然的概率可以忽略。旁证：418 GiB 池子没有出现在任何一张卡的 GTT 占用里（D0b 与 D1 每卡 GTT 都是 377/188/188/188 GiB，
  其中是 Engram 的 HIP VMM 表），即 ROCm 把 `hipHostMalloc` 的系统内存也当作 userptr 注册给 KFD。`numa_balancing=0`，排除。
  D0b 的 UMBP 服务端显存（146 MB）在撤队列前后不变，显存压力的推测也不成立。内核日志收不到（作业内 `dmesg` 无权限）。
  因此 MORI 的 `UMBP_DRAM_PINNED` 补丁（同样是 `hipHostMalloc`）不采用。
- **GLM-5.2（`dev/pd_opt/vllm_glm52_agentx` 的 `llying/`）的同类证据**：
  - m12（11-18，v3-006，CPU 层 1.9 TB"匿名 mmap + `hipHostRegister`"）decode 变慢到卡死，内核日志 `evict_process_worker` 4→64 次，
    之后改为 `hipHostAlloc`（`d3d58d16d`）；后来又归因于 decode 显存余量（U29），始终没分开。
  - 改用 `hipHostMalloc` 之后的 J3b：`amdgpu_amdkfd_restore_userptr_worker hogged CPU ... 65536 times`
    （`upstream-vllm-20261117/results/main-300563/dmesg-after.txt:147`），健康检查只数 `evict_process_worker`，所以记成 0。
    独立证明 `hipHostMalloc` 内存仍走 userptr 撤队列 / 恢复。
  - s2b / s2c / a4 的 prefill 静默卡住（CPU 层 `hipMemcpyBatchAsync`、`sample_tokens` RPC 超时）和 a24 的 63–77 s 传输停顿，
    与 userptr 撤队列吻合（推断，GLM 没有 `evicted_ms` 记录）。GLM 没有我们 R11 那种 GPU 自旋型 decode 死锁的记录。
  - GLM 整机独占、可用内存约 800 GB、没有 Engram，规整压力和注册内存总量都小于我们，撤队列短而被吸收。
- **下一步：UMBP 池改用 HIP VMM 主机内存**（与 Engram 修复同一方式：`hipMemCreate` 主机 NUMA 位置 + dmabuf 导出，GPU 用
  `hipMemMap`，CPU mmap dmabuf FD；这些是驱动自有页，不在 MMU 通知路径上）。先用 `move_pages` 主动迁移自己的页做单元验证
  （`hipHostMalloc` / `hipHostRegister` / HIP VMM 三种分配各自是否被撤队列），再改预加载小库上复现。
- **D0b 结束（10:58，17 个 cycle、140 分钟，锁页池 + 新遥测，v3-006）**：0 次引擎死亡 / 连接重置；UMBP 读 138 GiB、0 失败，
  但写入失败 160,820 次；UMBP 服务端撤队列 **67 次、累计 972 s、最长 57 s**（D0 为 20 次 / 389 s，D1 注册池同节点 0 次）。
  67 次里 50 次（75%）开始时刻 ±5 s 内有 ≥ 64 页迁移和内存规整（随机 10 s 窗口 9.0%，n=878）；其余 17 次在 2 s 采样的计数器上
  看不到迁移（采样对不齐，或计数器不记录的触发，如 khugepaged 合并；未定）。结论不变：锁页池仍是 userptr，
  撤队列与节点当时的规整强度有关（D1 那段时间节点很安静）。数据：`raw/hang3-r4-hipvmm-p1-1125T0831/`。
- **KFD 撤队列单元验证（作业 311256，v3-006，11:00）：HIP VMM 主机内存不会被撤队列，`hipHostMalloc` 会。**
  `scripts/kfd_migrate_probe.py`（`probe_kfd.sbatch`，镜像内、一张卡）每种方式分配 64 GiB，CPU 写满、GPU 读一次，然后
  对整段 `mprotect` 只读再改回（一次 MMU 失效、不搬页），读本进程 KFD `evicted_ms`（8 张卡的统计相加）与之后 GPU 读的耗时：

  | 分配方式 | 分配用时 | `mprotect` 后 evicted_ms | 之后 GPU 读 | 对照（等待 3 s） |
  |---|---|---|---|---|
  | 匿名 mmap + `hipHostRegister`（UMBP 原样） | 25.8 s | **+102,281** | **11.0 s** | 0 / 0.01 s |
  | `hipHostMalloc`（锁页池小库 / MORI 补丁） | 15.7 s | **+26,369** | **2.0 s** | 0 / 0.01 s |
  | HIP VMM 主机内存（Engram 的方式） | 10.0 s | 0 | 0.01 s | 0 / 0.01 s |
  | UMBP 原样 + 预加载 `instr/umbp_vmm_pool.so` | 6.9 s（池 2.4 s） | 0 | 0.01 s | 0 / 0.01 s |

  即 `hipHostMalloc` 的系统内存同样挂在 MMU 通知上；HIP VMM 主机内存（TTM 页、CPU 经 dmabuf 映射）不受影响；新小库在真实
  HIP 上工作（池 32 × 2 GiB、`hipHostRegister` 跳过、设备别名可读、释放正常）。`move_pages` 在默认 seccomp + `CAP_SYS_NICE`
  下返回 EPERM，那几行无效（"move-back moved=4096" 是状态数组初值，不是真的搬了）。节点：2 个 NUMA 节点、THP `madvise`、
  `compact_unevictable_allowed=1`、`numa_balancing=0`、amdgpu 6.16.13。数据：`raw/kfdprobe-311256/`。
  假 HIP 测试那一步因镜像的 `LD_LIBRARY_PATH` 先于 RUNPATH 加载了真库而没跑成，已修（作业 311331 重跑，含失败回退路径）。
- **hangE（作业 311332，`run_hang4.sbatch` → `pipeline_hang4.sh`，`UMBP_POOL=vmm`）**：与 D0 / D0b / D1 相同的负载
  （`r4-hipvmm`、BUDGET_S=9000），判据：UMBP 服务端 0 次撤队列、0 次 UMBP 读写失败，重放读性能不差于 D1。
  第一次（311334，11:03）被起栈时的检查拦下：UMBP 的池长 449,000,001,536 字节不是 2 MiB 粒度的整数倍，小库要求整块，
  0.1 s 内退回了原注册池（"falling back to anonymous mmap"），即这次跑的是对照组，已取消。小库改为按粒度向上取整、最后一块
  变小（假 HIP 测试改用非整倍长度），测试作业 311358 通过后（`afterok`）hangE 以 311359 重跑，用该作业编出并测过的 `.so`。
- **hangE 起栈（311359，v3-006，11:10–11:18）：UMBP 池在 HIP VMM 主机内存里。** 428,199 MiB、210 块、2 张 P 卡都映射，
  分配 30.6 s（注册池 51–84 s），`hipHostRegister` 跳过，UMBP 报告 2/2 卡可 gather；起栈 424 s。每卡 GTT 占用 P 卡从
  377 / 188 涨到 795 / 607 GiB（+418 GiB），即池子成了驱动的 TTM 页（注册池和 `hipHostMalloc` 池都不计入 GTT）。
  **新现象：起栈时 4 个引擎 worker 被撤队列，各 80–88 s**（11:12:55–11:17:33，恰在 Engram 的 HIP VMM 表创建、GTT 上涨期间；
  D1 起栈时只有约 2 s）。推测：池子 418 GiB 加 Engram 表后 TTM 用量约 800 GiB，创建新 TTM 对象时 TTM 换出 / 移动了已有对象，
  KFD 随之撤其所属进程的队列。只在起栈时出现则无害（起栈时长没有变长）；若服务期间也出现就是新问题，正在逐次监视。
- **hangE 服务期间（11:21 起）引擎 worker 也开始被撤队列**：4 个 worker 各 2–3 次、每次 1–3 s，开始时刻都有页迁移；
  D0 / D0b / D1 服务期间引擎进程 **0 次**。同时内存规整明显加剧：服务开始后 210 s 内 431 次 `compact_stall`
  （D0b 同阶段 900 s 内 287 次，D1 0 次），约 6.6 倍。推测：小库把 418 GiB 全放在 GPU 所在的 NUMA 节点 0（Engram 表也在那里），
  该节点可移动内存大减，规整加剧并碰到了引擎自己的 userptr 内存（如 torch 锁页缓冲，同样走 `hipHostMalloc`）。
  遥测没有分节点内存，尚未直接证实。小库新增 `UMBP_VMM_NUMA=interleave`（第 i 块放第 i mod n 个有内存的节点），
  hangE 结束后由 hangE2（作业 311405，`UMBP_ENV_EXTRA=UMBP_VMM_NUMA=interleave`）验证；hangE 继续跑完，
  主要判据（UMBP 服务端撤队列）不受影响。
- **hangE cycle 11（12:44）decode 死锁，与 R11 同型（GPU 自旋）**：12:44:04–12:44:12 三个引擎 worker 被撤队列 0.5–1.2 s
  （上面那种由页迁移触发的短撤队列）；两个 decode rank 的逐步记录最后一步都在 12:44:04.55（记录器每 1 s 落盘，之后最多丢 1 s）；
  12:44:12 起两张 decode 卡（cards 1、2）100% 忙 5 分钟、功耗 307 / 340 W（R11 为 318 / 336 W），py-spy 所有线程空闲，
  12:49:04 `sample_tokens` RPC 300 s 超时、引擎被判死；流水线保留现场（`hang1/`）、12:58 重新部署后继续。
  **这是第一次看到 GPU 自旋型 decode 死锁与 KFD 撤队列在时间上重合**（R11 当时没有撤队列，所以撤队列不是唯一诱因，但可能是诱因之一：
  一个 rank 的队列被撤再恢复时，另一个 rank 在跨卡同步里自旋）。在此之前 cycle 1–10 的 UMBP：读 13 GiB / cycle、0 失败，
  UMBP 服务端撤队列累计 0 s。含义：引擎进程上的短撤队列（NUMA 节点 0 压力）不只是几秒停顿，可能引发死锁，hangE2 的 interleave
  更关键；同时引擎进程自己的 userptr 内存（torch 锁页缓冲等）也应移出 userptr。
- **hangE 结论（13:22 提前结束，作业 311359；数据 `raw/hang4-r4-hipvmm-vmm-1125T1110/`）**：
  1. **UMBP 撤队列根治**：cycle 1–10 累计写入约 439 GiB（cycle 9 起超过 0.9 高水位、开始淘汰），UMBP 服务端撤队列累计 **0 s**
     （11 次亚秒级，推测来自服务端自己的小块 `hipHostMalloc` 缓冲），P 侧 0 次 UMBP 警告，UMBP 读 70 GiB、0 失败、写 0 失败；
     同阶段 D0b（锁页池）每次唤醒都撤 7–57 s、写失败 16 万次。
  2. **副作用**：池子全在 NUMA 节点 0 → 规整约 6.6 倍 → 引擎 worker 每次唤醒被撤 1–3 s。
  3. **decode 自旋型死锁一次**，起点与一次引擎撤队列重合（见上）。
  重新部署后的第二段池子从空开始，到 BUDGET 用完前写不满，只能再给 1–2 个死锁样本，不如尽早跑 hangE2，所以提前取消。
- **hangE2（作业 311405，13:22 起，`UMBP_VMM_NUMA=interleave`）**：判据：UMBP 服务端仍 0 s；规整强度回到 D0b / D1 水平；
  引擎 worker 服务期间不再被撤队列；无 decode 死锁。
- **更正与补充（13:30）**：
  - 上面"UMBP 撤队列根治"只指 UMBP 服务端（418 GiB 池）；userptr + 规整 → 撤队列的机制本身仍在，引擎进程里还有较小的
    userptr 内存（如 torch 锁页缓冲，`hipHostMalloc` 即 userptr）。按全部带遥测的 run 统计服务期间（起栈 15 分钟后）的撤队列：
    hang2（Engram 修复前）引擎 82 次 / 63,000+ s；hangC2 引擎 57 次 / 1,180 s；**R9b 引擎 136 次 / 328 s、最长 8.7 s**（当时被
    UMBP 失效掩盖）；R10b 0；D0 / D0b 引擎 0、UMBP 18–60 次；D1 全 0；hangE 引擎 78 次 / 98 s、最长 4.4 s，UMBP 0 s。
    即这是一直存在的系统性问题，多少取决于剩余 userptr 量和节点当时的规整强度（同机作业、碎片、我们的内存布局）。
  - **R11 死锁前后（02:30–02:48，2 s 采样、毫秒计数器）所有进程撤队列增量都是 0**；02:49–02:56 的 3 次在死锁和重新部署之后。
    所以 GPU 自旋型 decode 死锁不以撤队列为必要条件；hangE 那次的重合可能是诱因之一，也可能是巧合。
  - 下一步：给引擎进程加只记录的预加载小库，统计 `hipHostMalloc` / `hipHostRegister` 的来源与大小，大块改 HIP VMM；
    试 `THP_OFF=1`（引擎进程关 THP，减少我们自己触发的直接规整）；decode 死锁用 `pipeline10.sh` 的逐步落盘 + 开始记录抓现场。
- **hangE2 起栈（311405，13:22–13:28）**：池子按块交替放在 2 个 NUMA 节点（16.4 s），起栈 343 s；起栈期间引擎 worker
  撤队列各 10–12 s（hangE 为 80–88 s）。
- **改为合并验证 hangF（作业 311609，`run_hang5.sbatch` → `pipeline_hang5.sh`，13:45 起）**：hangE2 只跑了 1 个 cycle 即取消，
  把降规整的两项和两项只记录的诊断放进同一轮，并加快节奏：
  - 改动：UMBP 池 HIP VMM + NUMA 交替（`UMBP_VMM_NUMA=interleave`）、引擎进程 `THP_OFF=1`（`prctl` 关 THP，减少我们自己触发的直接规整）。
  - 诊断（不改行为）：引擎进程预加载 `hostalloc_log.so`（GEM 分配器之后）记录 userptr 内存来源与大小；decode 逐步记录每步落盘 + 记录每步开始。
  - 节奏：IDLE_S 150→60、LOAD_S 300→150、BUDGET_S 5400（每小时唤醒次数约翻倍；池子不必写满，hangE 已验证 UMBP）。
  - 归因：两项改动的作用用遥测区分（`thp_fault_alloc` / `compact_stall` 的变化对应 THP_OFF，起栈期撤队列对应 NUMA）。
  - 判据：引擎撤队列 < 计分窗口 1%、单次 < 5 s；UMBP 服务端 0 s；无 decode 死锁。
  - 第一次（311609）因 `hostalloc_log.c` 的宏写错编译失败，测试作业失败、`afterok` 挡住了 hangF；修后 311610 通过，hangF 以 311611 运行
    （`hang5-r4-hipvmm-vmm-1125T1344`，13:44 起，起栈 354 s，池子交替放在 2 个 NUMA 节点）。
  - **hostalloc 结果（起栈后）**：4 个引擎 worker（P 两个、D 两个）各持有约 **4.4 GB `hipHostMalloc`（即 userptr）**：
    一块 **4 GiB** 的 pinned `torch.zeros`（13:46:16 模型构建期间，"Loading model from scratch" 之后、Engram 表之前；torch 锁页分配器
    按 2 的幂取整，实际请求 2–4 GiB），一块 128 MiB（预热期间），其余约 3,360 次小分配（含 RCCL）。4.4 GB 的重新校验约 1–3 s，
    与引擎撤队列的时长吻合。4 GiB 那块的 Python 调用点下一轮由日志小库抓（≥ 64 MiB 时在释放内部锁后打印 Python 栈）。
  - 所有 run 的引擎日志都有 `[queue-intercept] ... falling back to system-memory ring`（镜像自带的队列拦截层把 GPU 命令队列的
    环形缓冲放在系统内存），从最早的 hang2 到 R10b 都有，不是新变量；之后单独评估它对性能的影响。
  - `THP_OFF` 检查只覆盖 `cli.main serve` 进程（3 个里 2 个已关），worker 进程未直接检查；看 `thp_fault_alloc` 的变化判断。
- **hangF 结束（15:14，20 个 cycle、服务 84 分钟）：撤队列消失。** 数据：`raw/hang5-r4-hipvmm-vmm-1125T1344/`。

  | | D0b（锁页池） | hangE（VMM 池、全在节点 0） | **hangF（VMM 池交替 NUMA + THP_OFF）** |
  |---|---|---|---|
  | 服务期间撤队列（≥ 0.5 s） | 60 次 / 889 s，最长 57 s | 54 次 / 46 s，最长 2.2 s | **0** |
  | 起栈期间撤队列 | 7 次 / 83 s | 56 次 / 358 s | 21 次 / 33 s |
  | 服务期间 compact_stall / thp_fault_alloc / 迁移页 | 639 / 3,675 / 49.7 万 | 221 / 2,413 / 4.9 万 | **0 / 0 / 0** |
  | UMBP 读失败 / 写失败 | 0 / 160,820 | 0 / 0 | **0 / 0** |
  | decode 死锁 | 0 | 1（11 个 cycle） | **0（20 个 cycle）** |

  `thp_fault_alloc` 是整机计数器，引擎关 THP 后从约 2,400 降到 0，所以驱动规整的大页分配主要来自我们自己的引擎进程，
  不是同机作业（同机的 chaojhou 作业两次都在）。NUMA 交替把起栈撤队列从 358 s 降到 33 s。这也说明独占节点并不能自动消除问题。
  UMBP 每 cycle 读 3–5.6 GiB（周期缩短、池子未满，读量比 hangE 小），0 失败。
- **恢复正式跑分：R13（作业 311795，`run_opt9.sbatch` → `pipeline11.sh`）**：`r4-hipvmm` c704、GATE=0、PROFILE_AT=3300，
  与 R10b（494,398）对比；`pipeline11.sh` = `pipeline10.sh` + `UMBP_POOL=vmm`（NUMA 交替）+ `THP_OFF=1` + hostalloc 日志 +
  分析里每进程撤队列与规整计数（`scripts/kfd_evict_summary.py`）。前置的小库测试作业 311794 重新编出两个 `.so`（`afterok`）。
  R13 15:23 起栈（354 s），池子 HIP VMM 交替 NUMA，smoke 全过，15:24 开始 AgentX c704。
- **引擎 4 GiB userptr 的来源（R13 hostalloc 的 Python 栈）**：`GPUModelRunner.__init__` → `RequestState`（`v1/worker/gpu/states.py:37`）
  的 `all_token_ids = StagedWriteTensor((max_num_reqs, max_model_len), int32, uva_instead_of_gpu=True)` → `UvaBuffer`
  （`buffer_utils.py:43`，`torch.zeros(..., pin_memory=True)`）。`max_model_len` 1,048,576 使它达 GiB 级；vLLM 有意放在锁页主机
  内存（UVA）以省显存，GPU kernel 每步直接读写。其页一旦被迁移，所属 worker 就被撤队列。可选修法：降 `--max-model-len`（按比例缩小）；
  或给 `UvaBuffer` 换成 HIP VMM 主机内存（它本来就分 CPU / GPU 两个视图，与 Engram 表的做法相同）。THP_OFF 后已无规整触发，
  暂列为后续项。
- 修复落地后恢复正式跑分用 `run_opt8.sbatch` → `pipeline10.sh`（`pipeline9.sh` 加 `UMBP_PINNED=1`、decode 逐步记录每步落盘和
  开始记录、结束时 `umbp_avail.py`）。

### 4.8 R9b（11-24 20:15，v3-085）：两个修复都在，完整跑完；比 R5 低 4.4%，差距来自 UMBP 失效

作业 309692，`r4-hipvmm` c640，`pipeline9.sh`（backlog 修复），`GATE=0 PROFILE_AT=3300 STEPLOG=1`，节点上没有别的作业。
数据：`raw/pd1p1d-c640-r4-hipvmm-1124T2015/`（`analysis/compare.txt` 对比 R5）。

| | R9b（HIP VMM + backlog） | R5（`r4-a4w4p`） | crusoe 基线 |
|---|---|---|---|
| 总吞吐 tok/s/GPU | **465,373** | 486,737 | 463,531 |
| 输出 tok/s/GPU | 3,541 | 3,653 | 3,488 |
| TTFT p50 / p90 | 5.81 / 22.60 s | 3.40 / 11.31 s | 4.71 / 13.28 s |
| 客户端 ITL p50 / 交互性 p50 | 26.6 ms / 37.7 | 29.2 ms / 34.2 | 30.7 ms / 32.6 |
| P 完成数 / 每请求 prefill 时间 / 排队 | 53,509 / 0.807 s / 9.6 s | 55,981 / 0.761 s / 3.7 s | — |
| P 每 rank 现算 tok/s、每步 token | 41.9k、13.3k | 39.6k、11.9k | 36.0k |
| P 命中：GPU / UMBP / 未命中 | 82.4% / 13.2% / 4.4% | 82.3% / 13.8% / 3.9% | 73.3% / 22.9% / 3.9% |
| UMBP 传输失败 / 提交 | 153 万 / 5,200 万（2.9%） | 0 / 5,450 万 | 0 / 5,180 万 |
| 超过 5 s 的停顿 | 2 次，都是计分 3,300 s 处抓 profile（P 19 s、D 5.4 s，期间无迁移、无撤队列） | — | — |

- **稳定**：预热切换 0 错误（4.2 节末尾），整轮没有一次 KFD 撤队列导致的停顿。
- **P 没有变慢，反而更忙**：每 rank 现算 41.9k tok/s，R5 是 39.6k。每个请求要现算的 token 多了约 11%
  （2 × 41.9k × 3600 / 53,509 ≈ 5,640，R5 ≈ 5,090），和未命中率的增量（3.9% → 4.4%，+13%）对得上；多出来的未命中
  就是丢掉的 UMBP 命中（13.8% → 13.2%）。原因是 UMBP 服务端在计分期间又被撤了几次队列（21:08–21:11 等，
  P 侧报了 30 次"UMBP 无响应"、20 次拒绝传输），这期间的 UMBP 命中都变成了重算。
- **D 变快**：decode-only 步 dt = 21.0 ms + 0.341 ms × 条数（R5 19.4 + 0.361），客户端 ITL 低 9–12%，很可能是 Engram 表改用
  2 MiB 页映射后，查表的 TLB 开销降了。吞吐由 P 决定，所以 D 变快只体现在交互性上。
- 结论：HIP VMM 本身不影响 P 的速度，剩下的差距要靠把 UMBP 主机池也移出 userptr 来消除（管理员选项见 4.6 节）。

**UMBP 主机池的分配方式与根本修法（11-25 03:20 按 MORI 源码更正；之前按二进制字符串推断为 memfd 共享内存，不对）**：
镜像里的 MORI 是 `v1.2.3.post1`（vLLM `docker/Dockerfile.rocm_base` 的 `MORI_BRANCH`，源码 `github.com/ROCm/mori`）。
- DRAM 池：`HostPageMemorySource::Allocate`（`src/umbp/distributed/peer/backend/page_backend.cpp:253`）用 `kAnonymous`，即
  `mmap(MAP_ANONYMOUS | MAP_PRIVATE)` 私有匿名内存，`MADV_POPULATE_WRITE` 预先触页（`local/host_mem_allocator.cpp` `AllocAnonymous`）；
  memfd 共享内存只用于客户端把自己的主机缓冲区注册给服务端（`standalone_process_client.cpp:680`），与 DRAM 池无关。
- 注册：`HostTierRegistration::Register`（`local/tiers/host_registration.cpp`）对整块池子调用
  `hipHostRegister(Mapped | Portable)`，GPU 的 gather kernel 用 `hipHostGetDevicePointer` 得到的别名直接访问，`hipMemcpy` 走锁页路径。
  `UMBP_DRAM_HOST_REGISTER=0` 只是跳过这一步：拷贝退回可分页 `hipMemcpy`（驱动中转），gather kernel 关闭，会损失性能。
- **根因**：可迁移的普通内存被注册成 GPU 可访问，驱动只能按 userptr 跟踪；规整迁移其中任何一页，就撤掉 UMBP 服务端的全部
  GPU 队列并重新校验整块 418 GiB。
- **根本修法**：DRAM 池改用驱动分配的锁页内存（`hipHostMalloc(Mapped | Portable)`，或 HIP VMM 主机内存），不再
  "mmap + `hipHostRegister`"；这类内存不在内核的可迁移页面里，规整不会碰它，也没有 userptr 校验。这与 vLLM 对 Engram 的修法
  （4.6 节）、11-18 改用 `hipHostAlloc`（4.10 节）是同一思路。需要给 MORI 打补丁、在镜像环境里重新编译 `umbp_standalone_server`。
- 不改代码的替代方案：管理员设 `vm.compact_unevictable_allowed=0` 并由我们 mlock 池子；或预留 hugetlb 大页（`UMBP_HUGEPAGES=1`）。

## 5. 提升吞吐的各轮实验

方法（与 review 时讨论的一致）：

- 在已饱和的并发上做 A/B，用分项指标归因，总吞吐作最终确认：λ（P 每秒完成请求数）、P 每 rank 现算 tok/s 与命中构成、
  D 每步耗时与每步条数。有收益的改动再重新扫并发，比较峰值对峰值。
- 每轮都在作业里记录逐步数据（`instr/vpd_steplog.py`，每个 engine 的每一步：耗时、token、请求数、graph 模式），并在计分窗口
  3300 s 处对 P（5 s）、D（3 s）各抓一次 torch profile；分析在节点上完成（`scripts/steplog_summary.py`、`trace_cat.py`）。
- 改了数值路径的轮次（graph 模式、EP / MegaMoE、top-k 后端）过完整的精度门禁。

### 5.1 prefill 每步耗时（基线，冷 prefill）

c704 基线预热第一波全是冷 prompt，P 的逐步记录给出纯 prefill 的单步成本（`raw/pd1p1d-c704-base-1123T1746/steplog-p/`）：

| 每 rank | 满步（16,384 token） | 小步（约 110 token） |
|---|---|---|
| 步数 | 5,783 / 5,807 | 92 / 103 |
| 单步耗时 p50 | 381 ms | 17–24 ms |
| 新算吞吐 | 42.0k / 41.6k tok/s | — |
| 上下文（ctx_max p50） | 11.2 万 / 10.7 万 token | — |

即每步约 20 ms 固定成本（与每步把本 rank 约 150 GiB 权重读一遍同量级）加每 token 约 22 µs。按 B300 对比报告 3.2 节的估算，
每 token 的 MoE AllGather + ReduceScatter 在两卡之间那一条 xGMI 上约 11–14 µs，若不重叠，约占一半；R1 的 profile 用来核实。

### 5.2 各轮

| 轮次 | arm | 改动 | 状态 |
|---|---|---|---|
| 基线 | `arms/base.sh` | AUS 配方原样，c704 | 两次未完成（第 4 节：v3-180 预热超时、v3-086 decode 死锁） |
| R1 | `arms/r1-capacity.sh` | P/D GMU 0.95；P 去掉 FULL decode graph；D FULL_DECODE_ONLY 覆盖到 288 条；D prefill-schedule-interval 2；c768 | KV 容量：P 29.9M → 38.1M token/rank（+27%），D 29.9M → 40.7M（+36%）；graph 显存 P 12.7 → 9.1–10.8 GiB、D 12.7 → 4.6–4.9 GiB。第一次（作业 306919）smoke 时 decode 死锁；第二次（307146）加载权重 50 分钟后被取消；第三次（307403，v3-135，`pipeline4.sh`）起服务 416 s，正确性通过（gsm8k 0.965，needle 全对）。**c768 吞吐 439,432 tok/s/GPU**（AUS 基线 c768 459,183），λ 14.28（AUS 15.02），TTFT p50 24.5 s；decode 明显变快（ITL 均值 24.7 ms，AUS 29.7 ms；每步固定成本 23.3 → 20.4 ms），但吞吐由 P 决定，未提升。分析见 5.3 节 |
| R3 | `arms/r1-capacity.sh` | 同 R1，c640（与 crusoe 基线 c640 463,531 同并发对比），`pipeline5.sh`，`GATE=0`（配置与 R1 相同，已过门禁） | 作业 307621 起服务中被我取消，换成下面更可能提升吞吐的 R4；c640 对比改作 R4 的备选 |
| R4 | `arms/r4-a4w4p.sh`（起不来则 R1 于 c640） | R1 + **P 的 MoE 用 FP4 激活**（`VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1`，vLLM 中对 deepseek_v41 验证过的可选开关；D 仍是 a8w4）；c768，过门禁 | 依据：MoE GEMM 占 P GPU 时间 26%，MI355X 上 FP4 MFMA 吞吐是 FP8 的两倍。作业 307626（v3-120）：P 改用 `flydsl_mxmoe_g1_a4w4_*` kernel；正确性与 R1 相同（gsm8k 0.965，needle 16k/64k/128k 全对，kv-aware 8/8）；同样的 needle@128k 负载下 **P 满步 p50 324 ms，R1 361 ms（−10%）**。AgentX c768 未出计分数据：预热开始 2 分钟后 P 的两个 worker 都卡在一次 FlyDSL JIT（`gemm1_a4w4_port_fp4_…_bm64`，停在 `mgpuModuleLoad` → HSA ioctl，py-spy 现场在 `raw/…/hang-dumps-prefill/`）约 3 分钟；冷的第一波因此拖到 1,824 s，第二阶段开始时还在路上的 3 个根请求之一被对端重置连接，AIPerf 判"根预热请求失败"中止。R1 的第一波是 1,772 s，同样贴着约 1,800 s 的边 |
| R5 | `arms/r4-a4w4p.sh` | 同 R4，c640（与 crusoe 基线 c640 同并发；第一波约 1,500 s，余量充足），`GATE=0`（R4 已过门禁） | 作业 307636（v3-021）。第一次 AgentX 时 decode DP2 死锁（4.1 节），流水线保留现场、重新部署后重跑。**c640 吞吐 486,737 tok/s/GPU，crusoe 基线 c640 463,531（+5.0%）**；输出 3,653 tok/s/GPU（+4.7%）；TTFT p50 / p90 3.40 / 11.31 s（基线 4.71 / 13.28 s）；单用户 34.2 tok/s（基线 32.6）；前缀命中 0.980（理论 0.967）。归因：P 单请求处理时间 0.851 → 0.761 s（−11%），P 完成数 +3.8%，P 排队 4.6 → 3.7 s；P 不再饱和（平均每步 11.9k token，排队约 30 个/rank），c640 已是并发受限，需看 c768 |
| R6 | `arms/r4-a4w4p.sh` | 同 R4，c768，`GATE=0`；`pipeline6.sh` 在 AgentX 前加一步 JIT 预热（`scripts/jit_warm.py`：36 个 1k–64k 词的随机 prompt、`max_tokens=1`，让各种 token 数的 P 步先跑一遍） | 作业 307706（v3-021）：JIT 预热 39 s（36/36 成功），冷启动第一波 1,680 s（R1 为 1,772 s，R4 因 JIT 停顿到 1,824 s）。**c768 吞吐 477,212 tok/s/GPU**：R1 同并发 439,432（+8.6%），AUS 基线 c768 459,183（+3.9%）；P 完成 55,777 个（AUS c768 54,056），P 单请求处理 0.847 s（AUS 0.925 s）；P 仍饱和（排队约 110 个/rank），TTFT p50 13.9 s。低于 c640 的 486,737，这套配置的峰值仍在 c640 附近 |
| R7 / R8 | `arms/r4-a4w4p.sh` | 同 R4，c704 / c576（找准峰值；`pipeline6.sh`，`GATE=0`） | R7（308427，v3-176，起服务 1,955 s）：第一次 AgentX 时 prefill DP2 死锁（4.1 节），重新部署后第二次在预热阶段切换时连接被重置（4.2 节），未出数据。R8（308616，v3-086）运行中 |
| R7b | `arms/r4-a4w4p.sh` | c704 重跑，`pipeline7.sh`（加 TCP 计数器记录），`PROFILE_AT=0 STEPLOG=0` | 作业 308712（v3-086）：预热切换（09:46:39）0 次重置，计分 09:57–10:57。**10:06:50 左右 P、D 四张 GPU 同时停了约 3 分钟**（两条 leg 都报两次 60 s 停顿，未 fatal；现场见 4.3 节），c704 下 P 本来饱和，积压在计分窗口内没消化（P 排队 p90 257 s），吞吐 256,786 tok/s/GPU，作废 |
| R8 第二次 | `arms/r4-a4w4p.sh` | c576 | 预热切换时 75 个连接被重置（4.2 节），未出数据 |
| R9 / R9b | `arms/r4-hipvmm.sh` | R4 + Engram 表放 HIP VMM 主机内存（`VLLM_ROCM_ENGRAM_HIP_VMM=1`）；R9b 起加 router backlog 修复（`pipeline9.sh`），c640 | R9：精度门禁通过，预热切换时连接被重置。**R9b：465,373 tok/s/GPU**，稳定；比 R5 低 4.4%，差距来自 UMBP 失效（4.8 节） |
| R10b | `arms/r4-hipvmm.sh` | 同 R9b，c704 | 作业 309694（v3-085，同机有 `chaojhou` 的作业）：预热切换 0 错误（7,795 个请求）。**494,398 tok/s/GPU，当前最好**（R5 c640 +1.6%，crusoe 基线 +6.7%，AUS c640 +4.8%）；TTFT p50 / p90 9.13 / 23.26 s；P 完成 56,662 个，每 rank 现算 41.9k tok/s，每步平均 14.8k token（约 90% 饱和）；超过 5 s 的停顿只有计分 3,300 s 处抓 profile 的两次；UMBP 失效 4 次、拒绝 0 次 |
| R12 | `arms/r12-p32k.sh` | R9b + P 每步 32,768 token（默认 16,384）：每步约 20 ms 固定成本摊薄，预计 P 每 token −2.6%；c704，过精度门禁 | 作业 309933（v3-006，同机另有一个整半节点的作业）：精度门禁通过（gsm8k 0.975，needle 全对，kv-aware 8/8）。367,981 tok/s/GPU，TTFT p50 / p90 14.2 / 66.8 s。每 token 计算确实更便宜（P 每 rank 现算 43.1k tok/s，R10b 41.9k）；P 的 KV 容量 38.5M → 26.5M token/rank（−31%），GPU 命中 78.4% → 66.3%，更多命中落到 UMBP（17.4% → 27.9%）。**更正（11-25 02:50）：这一轮的下降不能归因于 KV 变小**——计分窗口内 UMBP 服务端有 1,035 s（28.8%）被撤队列、不可用（最长一次 545 s），UMBP 传输失败 6.4%（R10b 为 0），未命中 4.1% → 5.7%。KV 变小让 P 更依赖 UMBP，UMBP 失效时损失更大，两者分不开。**结论待定，需要在 UMBP 修好后重测** |
| R11 | `arms/r4-hipvmm.sh` | 同 R9b，c768 | 排队中（作业 309932，暂时 hold，让 R12 先跑） |
| R2 | `arms/r2-megamoe.sh`（起不来则 `arms/r2-lazy.sh`） | R1 + 两条 leg EP2 + `aiter_mega_moe`（MoRI EPv1；镜像内 MoRI 1.2.3 不支持 EPv2） | 作业 307071，v3-135：起服务 570 s，KV P 38.9M / D 41.0M token/rank；正确性通过（gsm8k 0.985，needle 全对）。**prefill 更慢**：同样的 needle@128k 负载下，16,384 token 的满步 p50 **487 ms**，R1 为 361 ms（基线 c704 预热 381 ms），慢 35%，而 P 正是瓶颈，故放弃 P 侧 MegaMoE。AgentX 第一个请求时 prefill DP2 死锁（DP1 跑 7,526 token 的新请求、DP0 跑空批次），未出计分数据 |

### 5.3 瓶颈：P 的每 token 计算，其中三分之一是单条 xGMI 上的 MoE 通信

数据：`raw/pd1p1d-c768-r1-capacity-1123T2103/`（计分窗口 3600 s 的逐步记录、窗口 3300 s 处 P 5 s / D 3 s 的 torch profile、
与 AUS c768 的指标对比 `analysis/compare.txt`），`raw/rccl-bench-307619/`（通信微基准）。

**1. P 一直满载，λ 由 P 的现算速度决定。** 计分窗口内两个 P rank 各跑约 9,100 个真实步，98% 是 16,384 token 的满步（99.5% eager），
步长 p50 389 ms，每 rank 现算 40.9k tok/s；P 排队均值约 182 个/rank，TTFT 的 93% 是 P 排队。λ = 51,402 / 3600 =
14.28 req/s，即每个请求约 5.7k 个需要现算的 token（2 × 40.9k ÷ 14.28）。

**2. 加缓存不会再提高吞吐。** 计分窗口中段（22:20）的服务端统计：P GPU 前缀命中 67.2%（AUS c768 59.1%，R1 加了 27% 的 P KV），
含 UMBP 的总命中 95.8%，trace 的理论上限 96.0%。没命中的 token 几乎都是 trace 本身的新 token，P 的现算量已经到底；
更多 GPU 缓存只是把 UMBP 命中换成 GPU 命中，不减少计算。

**3. P 的 GPU 时间构成**（profile 窗口 5.45 s，GPU 100% 忙，各类之和等于忙碌时间，即通信与计算没有重叠）：

| 类别 | rank 0 | rank 1 | 说明 |
|---|---|---|---|
| 通信（RCCL，`ncclDevKernel_Generic_1`） | 1,748 ms（32%） | 2,079 ms（38%） | MoE 的 AllGather + ReduceScatter：每个 MoE 层 4 次（hidden、topk 权重、topk id 的 all-gather，加 reduce-scatter），大的两次各约 3.3 ms；rank 1 多出的 330 ms 是在等 rank 0 |
| MoE GEMM | 1,391 ms（26%） | 1,403 ms（26%） | FlyDSL a8w4（`mfma_moe1_…afp8_wfp4…fp8q_sort`、`gemm2_a4w4…`） |
| 其他 | 940 ms（17%） | 919 ms（17%） | mHC（`mega_mhc_kernel`）、KV 写入（`fusedDeepseekV4FullCacheKernel`）、indexer top-k（`radix_topk…`）、Engram 查表等 |
| dense GEMM | 411 ms（8%） | 415 ms（8%） | |
| indexer logits | 349 ms（6%） | 140 ms（3%） | 两 rank 的长上下文请求分布不均，rank 0 多 |
| sparse MLA | 291 ms（5%） | 297 ms（5%） | |
| MoE 其他 / quant | 290 ms（5%） | 168 ms（3%） | |

**4. 通信已经顶到链路带宽，调 RCCL 没有用。** 两张卡之间只有一条 xGMI。按 P 一步的真实消息尺寸（每 rank 16,384 × 5,120 bf16 =
168 MB）测 all-gather / reduce-scatter（`scripts/rccl_bench.py`，镜像内 RCCL）：默认 3.12 / 3.17 ms，即每方向 53.9 / 52.9 GB/s；
`NCCL_MIN_NCHANNELS` 16 / 32 / 64、`NCCL_PROTO=Simple` 都在 ±3% 内（64 通道反而略慢）；同尺寸的点对点拷贝 2.77 ms（60.5 GB/s）。
引擎里每次约 3.3 ms，与微基准一致。

**5. 现成开关里能动通信的都试过或不可用：**

- EP2 + `aiter_mega_moe`（R2）：P 满步 487 ms，比 TP-MoE 的 361 ms 慢 35%。
- DBO（`--enable-dbo`，两个 microbatch 交替通信与计算）：vLLM 对带 Engram 的 DeepSeek V4.1 直接拒绝（`config/vllm.py`：
  "DeepSeek V4.1 Engram does not support DBO or microbatching"）。
- fp8 dispatch：P 用的 aiter MXFP4 MoE 声明 `expects_unquantized_inputs = True`（量化融合在 MoE kernel 里），AgRs 的
  all-gather 因此始终是 bf16；没有开关。
- `sparse_indexer_topk_backend`：只作用于 decode 的 top-k。

**6. D 有余量。** D 每 rank 平均 166 条、KV 用 47%，decode-only 步 dt = 20.4 ms + 0.366 ms × 条数，FULL graph 覆盖 99% 的步。
D 的 GPU 时间：MoE GEMM 37%、其他 26%、通信 21%。

### 5.4 下一步：要再往上，需要改代码

按"预计收益 / 工作量"排序；收益按 P 每步 389 ms、通信 125 ms 估算，λ 与 P 每 token 成本成反比：

| 方向 | 做法 | 预计收益 | 风险 / 工作量 |
|---|---|---|---|
| A. MoE 通信与计算重叠 | 把每层的 all-gather 按 token 切块，和 MoE GEMM 流水（AsyncTP 式）；或在 all-gather 期间算 shared expert | 通信藏住一半即 P −16%，λ 约 +19% | 中：改 vLLM 的 AgRs prepare/finalize 与 aiter MoE 调用 |
| B. fp8 dispatch | all-gather 前把 hidden 量化成 fp8（MXFP4 MoE 本来就用 fp8 激活），需要 aiter MoE 接受预量化输入 | all-gather 字节减半，通信 −25%，P −8% | 中：kernel 接口；要过精度门禁 |
| C. P 改 TP2 + QuickReduce | P 一个 TP2 引擎，MoE / attention 的 all-reduce 用 `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP8`/`INT4` 压缩 | 通信量减到 1/2–1/4；顺带消除 P 的 DP2 死锁 | 高：P TP2 → D DP2 的异构 KV 传输、MLA KV 在 TP 下复制（P 缓存减半，但见 5.3 第 2 点影响小）、UMBP |
| D. 用 D 的余量做一部分 prefill | router 对"新 token 少"的后续轮次直接发给 D（D 开前缀缓存，复用上一轮留下的 KV），P 只做长的新 prompt | 视分流比例；D 现在约一半负载 | 高：router 新策略 + D 侧 kv_both |
| E. P 两 rank 的负载均衡 | kv-aware 打分里加入 rank 的待算 token 数 | rank 间等待约 6% | 低到中：router 策略，需权衡命中率 |

## 6. 复现

```bash
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123      # 工作目录：scripts/、arms/、instr/、logs/
# 一次 run = 一个作业：节点体检 -> 拷模型 / 取镜像 -> pipeline5.sh（起栈、门禁、AgentX、分析、打包；任一 leg 死锁时重新部署一次）
cd $W && sbatch -J lyl-dsv41-r1c768 -x "$(paste -sd, exclude_nodes.txt)" \
    --export=ALL,ARM=arms/r1-capacity.sh,CONC=768,GATE=1,PROFILE_AT=3300 run_opt3.sbatch
bash $W/scripts/monitor.sh        # 登录节点：新步骤、作业结束、长时间无进展
sbatch -x "$(paste -sd, $W/exclude_nodes.txt)" $W/bench_rccl.sbatch     # 两卡 RCCL 微基准（5.3 节第 4 点）
```

本目录 `scripts/`、`arms/`、`instr/` 是工作目录的副本（`run_opt*.sbatch`、`bench_rccl.sbatch` 在 `scripts/` 下）。各 run 用的版本：
c640 `run_once.sbatch` + `pipeline.sh`；c704 `run_once2.sbatch` + `pipeline2.sh`；base704、R1 前两次、R2 `run_opt.sbatch` +
`pipeline3.sh`；R1 第三次 `run_opt2.sbatch` + `pipeline4.sh`；R3 起 `run_opt3.sbatch` + `pipeline5.sh`。`raw/<run>/` 由 `scripts/pack_raw.py` 生成，格式与 AUS 目录相同。完整原始数据在
`$W/logs/<run>/`。
