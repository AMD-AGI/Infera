# 单节点 P4D4：Infera + vLLM v0.34.0 + Mooncake 复现 GLM-5.2 AgentX（2026-11-17，crusoe v3-135，进行中）

对标：sglang 在本集群单节点 P4D4 C104 53,351 tok/s/GPU（`dev/pd_opt/glm_5.2_agentx` 分支 `llying/pd-repro-crusoe-20261116/`）。
方案见 `../vllm-pd-plan-20261117/PLAN.zh-CN.md` 5.3 节；脚本见 `../vllm-pd-tools-20261117/`。

## 0. 当前结论

- **目前最好的有效计分：s4 = 26,541 tok/s/GPU（C104，测量 1800 s，2026-11-17 22:53–23:23）**，为 sglang 本集群 53,351 的 49.7%。
  | | vLLM s4 | sglang P4D4 C104 |
  |---|---|---|
  | tok/s/GPU | 26,541 | 53,351 |
  | 输出 tok/s/GPU / req/s | 245.5 / 2.272 | — / 3.469 |
  | 完成 / 丢弃 | 4,146 / 3（1800 s） | 12,582 / 1（3600 s） |
  | 交互 p50 / p90 | 62.6 / 49.7 | 69.9 / 60.8 |
  | ITL p50 / p90（ms） | 16.0 / 20.1 | — |
  | TTFT p50 / p90 / 均值（s） | 22.5 / 79.2 / 33.8 | 4.20 / 9.23 / 5.65 |
  | prefill 命中（采样器） | 87.5% | 96% |
  差距几乎全在 prefill：TTFT 高 5 倍多，decode 的 ITL / 交互度接近 sglang，decode KV 多数时间不到一半。详见各轮。
  （s4 的 AIPerf 回放本身通过了覆盖率检查；InferenceX 聚合因缺 `KV_OFFLOAD_BACKEND_METADATA` 失败，事后用同一份产物重跑聚合得到上面的数，
  `../vllm-pd-tools-20261117/scripts/agentx_post.sh`。）

- **PD 链路跑通，门禁干净（18:12）**：Infera rust router（与 sglang recipe 相同的参数）→ vLLM prefill TP1×PP4（GPU 0-3）→
  Mooncake RDMA 同节点回环（GEM KV）→ vLLM decode DP4（GPU 4-7，MoE 平铺 TP4）。smoke 6/6、GSM8K 196/200（乱码 0、开头坏 0）、
  16k 捞针 32/32 + 长前缀 31/32、真实接受长度 3.46。
- **GEM KV 在 vLLM 上可用**：每卡 150–163 GB 的 KV 由 `infera.engine.vllm.gem_mooncake`（`get_mem_pool_context()` 返回 GEM 内存池）分配，
  Mooncake 以 dma-buf 分 1 GiB 块注册，注册后没有 sglang 未修时那种"假 OOM"。
- **s1（prefill 只有设备缓存、FCFS）在预热中就颠簸，19:05 中止**：预热 36.5 min 只返回 498 / 1,155 个请求（约 8 个 / min，
  sglang 稳态约 208 个 / min），prefill 前缀命中 5.2%（命中 410 万 / 查询 7,841 万 token，平均每个请求现算约 16 万 token），
  prefill 一直满载（约 3.3 万 tok/s 现算，100 个请求因 "capacity" 排队），KV 池用量 2–4%。机制见第 2 节。
- **s2**：prefill 加 1 TB CPU KV 层（OffloadingConnector，对应 sglang 每 rank 250 GB 的 HiCache）+ 缓存感知准入
  （`--cache-aware-admission-window 128 --cache-aware-admission-threshold 0`，前缀已在缓存里的请求先进，接近 sglang 的 SPF），19:06 起。

- **s2（GEM KV + CPU 层 + 缓存感知准入）**：prefill 在第一次 offload 写出（66k 冷请求的第一个完整块）时 GPU 访存故障（1b 节）。
- **s2b（prefill 不用 GEM、显存目标 0.6 + CPU 层 + 缓存感知准入；decode GEM）**：smoke 6/6、offload 正常；预热 17 min 后 **PD 死锁**
  （1c 节），20:17 中止。
- **s2c（同 s2b，但 prefill 回到 FCFS；decode 显存目标 0.95）**：预热约 25 min 后自行走出过载（prefill 排队从 ~130 降到个位数）；
  测量 20:57 起，**21:32 prefill worker 静默卡住，21:36 引擎因 `RPC call to sample_tokens timed out` 退出**，此后请求全部 500，
  AIPerf 以"测量期指标覆盖率 < 95%（TTFT 77.9%）"拒收结果。卡住前的稳态（20:57–21:32，采样器）：

  | | vLLM s2c | sglang P4D4 C104 |
  |---|---|---|
  | prefill 输入（含已缓存） | 约 20.7 万 tok/s | 约 42.4 万 tok/s |
  | prefill 现算 | 约 2.3 万 tok/s | 1.66 万 tok/s |
  | 命中（本地 / CPU 层） | 38.9% / 50.0%，合计 89% | 96% |
  | prefill 有排队的时间 | 62%（排队 p50 7、p90 109） | 66% |
  | AIPerf 请求吞吐（含卡住后） | 1.32 req/s | 3.47 req/s |
  | 每卡网卡流量 | 4 TB / h（每轮都传完整 KV） | — |

  prefill 现算不比 sglang 慢，差距来自命中率：受 prefill 限时，输入吞吐 ≈ 现算 /（1 − 命中），89% → 96% 约翻倍。
  卡住的原因未定：卡住前 worker 没有任何报错；最后一级 PP worker 停在 gloo `irecv` 等上一级。最可疑的是某次 offload 的 `hipMemcpyBatchAsync`
  （vLLM 注释记录它在 ROCm 7.15 上描述符过多会出错，本镜像是 ROCm SDK 10.0），未证实。
- **s3（CPU 层 2 TB）起不来**：KV 初始化（锁定主机内存）6 min 后一个 PP worker 被 SIGKILL（退出码 -9）。（更正）ROCm 上 `CPUOffloadingSpec`
  不用 `/dev/shm` 共享区域（`_uses_shared_region()` 在 ROCm 上为 False），CPU 层是每个 rank 自己的锁页主机内存；2 TB 锁页加上其余进程
  接近 2.75 TB 物理内存，是主机 OOM。1.25 TB 时（s5）主机还剩约 1.2 TB 可用，实际上限在 1.3–1.5 TB 之间。
- **s4（1 TB CPU 层 + Python router 串行下发 + 缓存感知准入 128 / 0 + offload 批量拷贝描述符上限 512 + decode 单个 API server，
  测量 1800 s）**：22:17 起；另挂看门狗（prefill 停 180 s 就用 py-spy 抓全部栈）。预热比 s2c 快（1,200 s 时返回 779 对 453，测量提前约 90 s 开始）。
  测量前 11 min：prefill 输入 18.7 万 tok/s、现算 2.3 万 tok/s、命中 87.5%（本地 60.3% + CPU 层 27.2%）；decode KV p50 0.20 / p90 0.48、
  在跑合计 p50 26（最忙 rank 8）——**瓶颈完全在 prefill**，decode 大量空闲。差距的结构性原因见 U9（写穿 vs 写回）。
- **s5（最终，3600 s）**：s4 + CPU 层 1.25 TB（失败则退回 1 TB）+ decode 测量中段采 3 s trace。

## 1g. s5 / s7 中止的根因：Infera Python 路由的文件描述符用完（已修）

- s7（1.25 TB，1800 s）在预热第 18 min 同样中止。这次路由日志在：`OSError: [Errno 24] Too many open files`（2,048 次，`socket.accept()`
  失败），随后 01:54:22 四个请求同时得到 502 `prefill leg failed`（路由向 prefill 建连失败），其中有根请求，AIPerf 中止。s5 的 502 同因。
- 路由容器没设 `--ulimit nofile`，容器默认软上限 1024（引擎容器是 65536）；Python 串行路由关了 keep-alive，每段请求新建连接。
  在途请求最多约 130、每请求约 3 个连接，到不了 1024，说明还有描述符泄漏（约每请求不到 1 个；s4 跑 55 min 没撞上，原因未明）。
- 修复：`pd1_up.sh` / `pd2_drive.sh` 的路由容器加 `--ulimit nofile=1048576:1048576`。s7b 重跑 1.25 TB，并每 30 s 记录路由进程的描述符数与类型
  （`router-fds.log`），用来定位泄漏。证据 `results/s7-abort/`。
- 非致命的空内容失败（每轮 3–6 条，固定在少数会话）仍待查。

## 1f. s6c（s4 + ARC 淘汰，1800 s）= 23,864 tok/s/GPU，比 LRU 差，保留 LRU

| | s4（LRU） | s6c（ARC） |
|---|---|---|
| tok/s/GPU | 26,541 | 23,864 |
| req/s | 2.272 | 2.073 |
| 完成 / 丢弃 | 4,146 / 3 | 3,790 / 5 |
| 交互 p50 / p90 | 62.6 / 49.7 | 65.5 / 53.7 |
| TTFT p50 / p90 / 均值（s） | 22.5 / 79.2 / 33.8 | 15.3 / 111.8 / 40.8 |

- 测量期每 5 min 的现算比例 0.10–0.18（s4 0.08–0.15，`results/s*/hit_timeline.txt`）。单次运行，未量化跑间波动；测量第 25 min 抓了 13 s decode trace。
- **decode trace**（`results/s6c/trace/`，4 个 DP rank，约 68 步、每步约 43 ms）：MoE GEMM 31%、**MoE 分发 / 合并通信 26%**
  （每层 2 次 RCCL `ncclDevKernel_Generic`，平均 71.5 µs）、其他 18%、sparse MLA 6%、indexer 5%。各 rank 通信 411–758 ms 不等，
  差额是等最慢的 rank：dp3 的 indexer 是 dp0 的 2.4 倍（上下文更长），它是关键路径。vLLM 有给 DP 组的 aiter 自定义 AG/RS 路径，
  这里没走上，原因待查（队列项 `decode-comm`：默认 / EP / EP + MoRI 对比）。

## 1d. s5（1.25 TB CPU 层，3600 s）：预热第 18 min 一条根请求 502，AIPerf 中止整轮

- 23:56:04 会话 `0a279af1` 第 27 轮（根请求，agent_depth 0）得到 HTTP 502 `{"error":"prefill leg failed"}`：Infera Python 串行路由
  POST prefill 时出现 httpx 网络层异常（`_dispatch_serial` 的 `prefill_unreachable` 分支）。AgentX 预热阶段根请求失败即中止整轮，没有计分。
- 另有一类**非致命**失败：`InvalidInferenceResultError`（流里没有任何内容），s4 预热 3 条、s5 预热 4 条，固定落在少数会话
  （`11310dce` 第 264 / 266 轮在两轮都失败），输入长度 7.7 万–68 万不等（成功请求最长 75 万），与长度无关，原因待查（候选：生成内容被
  reasoning 解析剥空）。s4 能计分是因为这类失败不是根请求失败。
- 中止前最后一次采样（23:56:26）：prefill 运行 0、因 capacity 排队 52、GPU KV 99.1%；decode 4 个 rank 全部空闲。
- **证据不全**：引擎日志被下一轮部署覆盖（`logs/engines/*.log` 每轮重写），路由日志从未保存。之后各轮把三个容器的日志实时写进各自的 run
  目录（队列项 `queue/s6c7.sh`），502 与空内容再出现时可以直接对上。

## 1e. s6（vLLM FS 二级层）：起栈失败，vLLM 的 bug

- `TieringOffloadingSpec` 下 PP1 / PP2 等待 `/dev/shm/vllm_offload_*.mmap` 达到 1,249,999,527,936 字节超时（30 s），而 PP0 已建好"1250.00 GB"的文件；
  等待条件是 `>=`，说明各 PP stage 算出的区域大小不一致（各 stage 层数 20/20/20/18）。ROCm 上 `CPUOffloadingSpec` 不用共享区域（每个 rank 自己的锁页内存），
  所以 s4 / s5 不受影响。原始日志 `results/s6-fs-fail/prefill-fail.log.gz`。列入上游候选。
- 随后改为两轮 1800 s 的单变量对照：s6c（ARC 淘汰）、s7（1.25 TB）。

## 1b. GEM KV + OffloadingConnector：第一次写出就 GPU 访存故障（未解决）

- 现象：prefill 某个 PP worker 在第一次 offload 写出时 `HSA_STATUS_ERROR_MEMORY_FAULT`（之前的请求都不足一个块，不触发 offload）。
- 隔离（`../vllm-pd-tools-20261117/scripts/swap_fault_test.py` 与聚合复现）：
  - offload 的拷贝 `ops.swap_blocks_batch`（ROCm 上 `hipMemcpyBatchAsync`）在普通 / GEM 显存 × 锁页 / `hipHostRegister` mmap 的所有组合上正常，
    包括 8192 个描述符铺在 64 GB 缓冲区上；
  - 聚合 PP4 + offload（无 GEM、无 Mooncake）：100 GB 正常；1 TB 第一个请求卡 > 5 min（锁定主机内存，另见下）；
  - 聚合 PP4 + MultiConnector[GEM Mooncake + offload 100 GB]：**复现故障**。所以是 GEM 与 offload 的组合，与 PD、容量无关。
- 尚未排除：150 GB 级单个 GEM BO（测试只到 64 GB）、写出时的"源端访问顺序任意"属性（`is_src_access_order_any`）、
  `/dev/shm` 背后的共享 offload 区。绕法：prefill 不用 GEM（用较低显存目标容纳 dma-buf 重复记账），decode 用 GEM。

## 1c. PD 停顿：prefill 侧重排 + decode 到达即预留（推断，证据不全）

- 现象（s2b，20:16）：prefill 运行中 0、排队 76、KV 99.7%，prompt 吞吐 0；AIPerf 的返回数停在 198 超过 1 min。
  **更正（20:42）**：当时读到的 decode "4 个 rank 运行中 0、KV 84–97%" 不可靠——decode 开了 4 个 API server（vLLM 按 DP4 自动设置），
  每次抓 `/metrics` 只落到其中一个进程，计数与 gauge 都是局部的（s2c 中 decode 收到 296 个请求，抓到的成功数每个 rank 只有 1）。
  所以 s2b 能确定的只有 prefill 侧停顿；是循环等待还是 decode KV 满造成的反压，需要 decode 用单个 API server 重测。
- 机制：vLLM Mooncake 是并发下发，decode 收到请求就为整段 prompt 预留 KV 等 prefill 推送；prefill 算完的请求占着 KV 等 decode 来拉
  （租约 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`）。prefill 开了缓存感知准入后与 decode 的到达顺序不一致：decode 预留的是 prefill 还没算的请求，
  prefill 占着的是 decode 还没预留的请求，两边都满时循环等待。两侧都 FCFS 时顺序一致，不会死锁（s1 没有死锁）。
- 含义：在 vLLM PD 下，prefill 侧任何重排（缓存感知准入、优先级、SPF）都需要配合 decode 侧"按 prefill 完成顺序预留"或"prefill 完成后再下发 decode"
  （串行拉取，如 NIXL）才安全。这是上游候选（Mooncake connector / router 的下发方式），也是 sglang 的 SPF 在 vLLM 上不能直接照搬的原因：
  sglang 的 prefill 要等 decode 预分配完成才开始算。
- 另：1 TB CPU 层的锁定内存使起栈慢（472 s）或第一个请求慢（77 s 至 > 5 min）。

## 1a. s1 的颠簸（为什么 prefill 只有设备缓存不行）

- AgentX 的会话很长：s1 里 prompt 平均约 16 万 token，104 条会话的工作集约 1,500 万 token 以上，prefill PP4 设备 KV 池 1,315 万 token。
- 预热开头 115 条会话各发一个冷启动的长请求（约 1,700 万 token）。FCFS 下每条会话的下一轮要在 prefill 排队约 100 个请求、约 450 s，
  这期间新算了约 1,500 万 token，它的前缀已被 LRU 换出；命中率塌到 5%，计算量更大、排队更长，自我强化。
- chat template 不是原因：用 `/tokenize` 比较带推理内容和工具调用的两轮 agent 对话，下一轮完全包含上一轮的 token（公共前缀 100%）。
- sglang 用 HiCache（容量）+ SPF（后续轮未命中 token 少，插到冷请求前面，趁缓存还在算完）避开了这个问题。vLLM 有 CPU 层
  （OffloadingConnector）和缓存感知准入窗口（只看 GPU 前缀缓存、二分"已缓存 / 未缓存"，不是按未命中 token 排序）；
  按未命中 token 排序、且把 connector（CPU 层）命中算进去的 SPF 是上游候选。

## 1. 部署

| 项 | prefill | decode |
|---|---|---|
| 并行 | TP1 × PP4，`VLLM_PP_LAYER_PARTITION=20,20,20,18` | `-dp 4 -tp 1`（MoE 平铺 TP4），内置 DP 均衡关，router 用 `X-Data-Parallel-Rank` 选 rank |
| 批 / 缓存 | `--max-num-batched-tokens 8192`（chunk 8192）、前缀缓存开、调度 fcfs（vLLM 没有 SPF） | `--max-num-seqs 64`、`--stream-interval 30`、前缀缓存关（与 sglang 的 decode 相同） |
| KV | FP8 e4m3；GEM 内存池；MultiConnector[Mooncake producer] | FP8 e4m3；GEM 内存池；Mooncake consumer |
| MTP | `{"method":"mtp","num_speculative_tokens":3}`（P 也要带，生成草稿层 KV） | 同左；计分时加 `rejection_sample_method: synthetic, synthetic_acceptance_length: 2.99` |
| KV 池 | 每卡 KV 缓冲 150–163 GB（权重 107–119 GiB） | 每 rank 2,764,032 token（sglang TP4 decode 关 PTPC 时 3,043,072，少 9%） |
| 起栈 | 151 s | 682 s（每 rank 读全部 shard，约 2 s / shard） |

Mooncake：镜像 `infera-vllm:v0340-0c5e5e89`（v0.3.14 + #4540 + #4541），`MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18
MC_MAX_MR_SIZE=1073741824 MC_GID_INDEX=1`，每个容器一个拓扑文件（容器内 `cuda:i` → 该 GPU 的 ionic，无备选，`results/gate-p4d4-r1/config/`）。
router：`--router-backend rust --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill
--pd-prefill-guard-release completion --decode-input-demand on`。

## 2. 待查

- **KV 流量分布**：门禁期间 8 张 ionic 各约 5.4 GB/min、ACK 超时 0、重传 0；按"写入走目的端首选网卡"的设置，预期只在 ionic_4-7 上。
  单节点回环不受影响，两节点 P8D8 前要用 `MC_LOG_RDMA_SLICE_AFFINITY=1` 确认实际选卡。
- decode KV 池比 sglang 少 9%：显存目标（vLLM 默认 0.92，sglang 0.95）与 DP 下 dense 层每 rank 一份。
- prefill 的 aiter BF16 GEMM 多个形状无调优表（日志 "not found tuned config"），sglang recipe 带了两份 GLM-5.2 调优表。
