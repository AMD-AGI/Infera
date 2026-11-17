# 单节点 P4D4：Infera + vLLM v0.34.0 + Mooncake 复现 GLM-5.2 AgentX（2026-11-17，crusoe v3-135，进行中）

对标：sglang 在本集群单节点 P4D4 C104 53,351 tok/s/GPU（`dev/pd_opt/glm_5.2_agentx` 分支 `llying/pd-repro-crusoe-20261116/`）。
方案见 `../vllm-pd-plan-20261117/PLAN.zh-CN.md` 5.3 节；脚本见 `../vllm-pd-tools-20261117/`。

## 0. 当前结论

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
- **s3（CPU 层 2 TB）起不来**：KV 初始化（锁定主机内存）6 min 后一个 PP worker 被 SIGKILL（退出码 -9）。vLLM 的 CPU 层固定建在
  `/dev/shm/vllm_offload_<engine_id>.mmap`（`v1/kv_offload/cpu/shared_offload_region.py`），本节点 `/dev/shm` 是 1.4 TB 的 tmpfs，
  且 2 TB 锁定内存已接近 2.75 TB 物理内存。本节点 CPU 层实际上限约 1.3 TB，比 1 TB 多得有限，不再加容量。
- **s4（1 TB CPU 层 + Python router 串行下发 + 缓存感知准入 128 / 0 + offload 批量拷贝描述符上限 512 + decode 单个 API server，
  测量 1800 s）**：22:17 起；另挂看门狗（prefill 停 180 s 就用 py-spy 抓全部栈）。

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
