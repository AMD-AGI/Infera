# B 线：GLM-5.2 P4D4 prefill 侧 GEM + offload 故障、vLLM 多级 offload 在 PP 下起不来（2026-11-18，v3-258，进行中）

接手 `../vllm-p4d4-crusoe-20261117/HANDOFF-trackB.zh-CN.md`。节点 crusoe v3-258（作业 294784），镜像 `infera-vllm:v0340-0c5e5e89`
（vLLM 0.34.0+rocm100、torch 2.13.0+rocm10.0.0、ROCm SDK 10.0）。工作目录 `/home/liyingli/bench_agentx/vllm-trackB-20261118/`，
命令经 `scripts/qb.sh`（包一层 `q2.sh`，先 `cd` 到工作目录并设 `R`）投递到 `queue-n2/`。

## 0. 结论

- **B1（U2）根因已定位并修复**：不是 vLLM offload 的地址 / 大小问题，也与 150 GB 级 BO、`srcAccessOrder`、拷贝所在的流无关，
  而是 **HIP 的 `hipMemcpyBatchAsync` 把 GEM 导入内存的拷贝放到了 device 0 上执行**：
  - GEM 分配器用 `hipImportExternalMemory` + `hipExternalMemoryGetMappedBuffer` 把 BO 映射进进程，HIP 为它建的内存对象从不设置
    `getUserData().deviceId`（默认 0；`hipMalloc`、`hipIpcOpenMemHandle`、`hipHostRegister`、VMM 导入都会设）。
  - `hipMemcpyBatchAsync` 按每个描述符设备侧内存对象的 `deviceId` 分组，`deviceId` 与流的设备不同时，把这一组放到那个设备的
    null stream 上执行。PP rank k > 0（device k）的 offload 写出 / 读回于是在 device 0 上跑，而 BO 只映射在 device k 的页表里，
    device 0 访问即 `HSA_STATUS_ERROR_MEMORY_FAULT`。
  - 所以只有 GEM + offload 组合出问题；单卡独立测试（只用 device 0）测不出来；PP0 不出问题；不用 offload 时 GEM 的 kernel 访问、
    Mooncake RDMA 都不走这条路。
- **修复（Infera GEM 分配器，`../vllm-image-v0340-20261117/gem/hip_gem_alloc.c`）**：BO 改用 HIP VMM 导入
  （`hipMemImportFromShareableHandle` 导入 dma-buf + `hipMemAddressReserve` / `hipMemMap` / `hipMemSetAccess`），HIP 记录的设备就是
  导入时的设备。旧路径保留为 `HIP_GEM_IMPORT=external`。批量拷贝带宽与普通显存相同（约 51 GB/s），KV 池大小不变。
  上游另需在 ROCm clr 修 `hipExternalMemoryGetMappedBuffer`（设置 `deviceId`）。
- 验证（各一次）：独立测试 GEM 在 device 0–3、D2H / H2D 全部正常（修前 device 1–3 全部故障）；聚合 PP4 复现配置 smoke 5/5、
  0 次故障（修前第 4 项 500、1 次故障）；**P4D4（prefill GEM、显存目标 0.88、1.25 TB CPU 层）门禁通过，计分 33,203 tok/s/GPU，
  比 s7b 高 10.3%**（1.5 节）。
- **B2（U11）修复已验证**：引擎核心统一各 worker 每块 KV 字节数的最大值，offload 配置按它分槽。聚合 PP4 上不修时复现
  "加入方等文件长到期望值"超时，修后起栈、从 FS 层读回的 3 个重复请求答案全对；P4D4 上 4 个 PP worker 加入同一块 1,250 GB 共享区
  （2 节）。
- **FS（NVMe）二级层不能用于 InferenceX 提交**：AgentX 规则只允许 CPU DRAM 级的 KV offload，NVMe offload 明确不在范围内
  （暂定 v3.5 / v4）；matrix schema 里 `kv-offloading` 只能是 `none` / `dram`；已发布 recipe 没有用磁盘层的（3 节）。
  正在跑的 FS 层 3600 s 只作诊断，之后接一轮只用 DRAM 的 3600 s 对照。
- **vLLM 写穿 vs sglang 写回**：sglang recipe 用 `--hicache-write-policy write_back`（1 TB），vLLM OffloadingConnector 只有写穿。
  估算不重复的有效容量：vLLM 约 2,500 万 token（1.25 TB），sglang 约 3,200 万（GPU 1,200 万 + 主机 2,000 万）。但 1800 s 测量期
  现算比例已接近 trace 理论下限（3 节），"CPU 层不够"在 1800 s 内不成立，3600 s 待对照。

## 1. B1：GEM KV + OffloadingConnector 第一次 offload 写出 GPU 访存故障（U2）

### 1.1 复现（`scripts/repro_gem_offload.sh`，聚合 PP4，GPU 4–7，MultiConnector[GEM Mooncake + 100 GB offload]）

- r0（原 GEM 库）：起栈 160 s，smoke 前 3 项通过，第 4 项（66k 冷请求，第一次跨满一个块、触发 offload 写出）失败，引擎卡住。日志
  （`results/b1/repro-r0-baseline/engine-excerpt.log`）：PP1（pid 1172）先为 agent `0x26fa9970` 新建队列，紧接着
  `Memory access fault by GPU node-6 (Agent handle: 0x26fa9970) on address 0x75fe40030000`。
- KFD node-6 = `0003:00:01.0` = 物理 GPU 4 = 容器内 device 0（`HIP_VISIBLE_DEVICES=4,5,6,7`）。即 **PP1 进程里的拷贝跑在了
  PP0 的卡上**。

### 1.2 隔离（`scripts/swap_dev_test.py`，一个用例一个进程，GPU 0–3，与 vLLM 引擎无关）

256 个 64 KiB 描述符，`ops.swap_blocks_batch`（= `hipMemcpyBatchAsync`）为主，另测单次 `hipMemcpyAsync` 与 torch `copy_`：

| 缓冲 | 缓冲所在设备 / 拷贝流设备 | 方法 | 方向 | 原 GEM 库 | VMM 导入 |
|---|---|---|---|---|---|
| GEM | 0 / 0 | batch | D2H | 正常 | 正常 |
| GEM | 1 / 1、3 / 3 | batch | D2H | **故障（node-2 = device 0）** | 正常 |
| GEM | 1 / 1 | batch | H2D | **故障（device 0）** | 正常 |
| GEM | 1 / 0、2 / 线程默认设备 0 | batch | D2H | **故障（device 0）** | 正常 |
| GEM | 1 / 1 | 单次 `hipMemcpyAsync`、torch `copy_` | D2H / H2D | 正常 | 正常 |
| 普通 `hipMalloc` | 1 / 0、1 / 1、线程默认设备 | batch | D2H | 正常 | — |

（`results/b1/swapdev1.txt`、`swapdev2.txt`、`swapdev3-vmm.txt`；故障总在 device 0，与缓冲和流在哪个设备无关。）

不依赖 GEM 与 vLLM 的最小复现（`scripts/batch_ext_repro.c`：`hipMalloc` → `hsa_amd_portable_export_dmabuf` → `hipImportExternalMemory`
导回同一设备 → `hipMemcpyBatchAsync` D2H）：device 1、3 上导入的内存故障，ROCr 报 `GPU index: 0, kernel: __amd_rocclr_copyBufferBatch`；
同一设备的 `hipMalloc` 指针、device 0 上导入的内存正常（`results/b1/batch_ext_repro.txt`）。

### 1.3 根因（ROCm clr 源码，`ROCm/rocm-systems` develop 的 `projects/clr`，与镜像 ROCm 10.0 的行为一致）

- `hipamd/src/hip_memory.cpp` `hipExternalMemoryGetMappedBuffer`：建 `amd::Buffer` 视图、`getDeviceMemory(当前设备)`、
  `MemObjMap::AddMemObj`，不设 `getUserData().deviceId`（`amd::Memory::UserData::deviceId` 默认 0，`rocclr/platform/memory.hpp`）。
- 同文件的批量拷贝（`hipMemcpyBatchAsync` 的实现）：`hipCopyBuffer` 类（D2H 时 dst 是锁页主机内存）按
  `src_memory_type == Device && dst_memory_type == Host ? srcMemories[i]->getUserData().deviceId : dstMemories[i]->getUserData().deviceId`
  分组；`EnqueueBatchCommands` 对 `device_id != stream.DeviceId()` 的组用 `hip::getNullStream(*g_devices[device_id]->asContext())`。
- 单次 `hipMemcpyAsync` 走 `GetDeviceById()`（按内存对象的上下文，导入时是当前设备），所以正常。
- 这也解释了 `hip_gem_alloc.c` 里早已记录的"HIP 把导入的外部内存报成 device 0"（`hipPointerGetAttributes` 读的也是 `deviceId`）。

### 1.4 修复：GEM BO 改用 HIP VMM 导入（`results/b1/hip_gem_alloc-vmm-import.diff`）

- `hip_gem_alloc()`：`GEM_CREATE` → `PRIME_HANDLE_TO_FD` 之后，`hipMemImportFromShareableHandle(dma-buf fd, PosixFileDescriptor)`
  （clr 设 `deviceId = 当前设备`）→ `hipMemAddressReserve`（2 MiB 对齐）→ `hipMemMap` → `hipMemSetAccess(本设备读写)`；
  `hip_gem_free()` 对应 `hipMemUnmap` / `hipMemAddressFree` / `hipMemRelease`。其余（Mooncake 的 dma-buf 导出钩子、
  `ibv_reg_mr` 拒绝、一页 dma-buf 锚点、自检）不变。`HIP_GEM_IMPORT=external` 回到旧导入。
- KFD 记账：VMM 导入同样是 KFD 的 dma-buf 导入（不计入 `kfd.vram_used`）；聚合 PP4 显存目标 0.92 时可用 KV 仍是 166.64 GiB、
  13,148,608 token，与旧库相同。
- 性能（`scripts/swap_perf_test.py`，5,120 个随机分布的 42 KiB 描述符）：VMM-GEM 批量 D2H / H2D 51.0 / 51.7 GB/s，普通显存 50.8 GB/s；
  对比：逐个 `hipMemcpyAsync` 只有 3.3–4.3 GB/s、提交耗时 10–45 ms，所以"在钩子里把 GEM 描述符拆成单次拷贝"不可取
  （`results/b1/swapperf1.txt`、`swapperf2-vmm.txt`）。
- 构建：`scripts/build_gem.sh`（与 Dockerfile 同一条 gcc 命令）产出 `libhip_gem_alloc.so`，bind mount 到容器的
  `/opt/infera-gem/libhip_gem_alloc.so`，不重建镜像。
- 聚合 PP4 复现（r1，`results/b1/repro-r1-vmm/`）：smoke 5/5、0 次故障；4 个 PP worker 共写出 2.38 GB（约 46 GB/s）。
  这轮 smoke 没有读回（重复请求命中 GPU 前缀缓存），读回方向由独立测试与 P4D4 覆盖。

### 1.5 P4D4（prefill GEM + 显存目标 0.88 + 1.25 TB CPU 层，其余同 s7b）

`scripts/b1_p4d4.sh r1 gate score`：先真实接受率部署 + `gate.sh`，再重新部署计分（AgentX C104 1800 s）。decode 不变（GEM、0.95、
原 GEM 库）；prefill 挂载新库，`--gpu-memory-utilization 0.88`。

- 部署：prefill 起栈 251 s（含 1.25 TB 锁页），decode 703 s。prefill GPU KV **12,082,688 token**（每卡可用 155.12 GiB），
  s7b（不用 GEM、0.6）约 460 万，sglang 约 1,200 万。
- **门禁通过（04:55，一次）**：smoke 6/6；GSM8K 197/200（乱码 0、开头坏 0、截断 2）；16k 捞针 32/32、长前缀 GSM8K 31/32；
  真实接受长度 3.458（每位置 0.925 / 0.821 / 0.712）；prefill 0 次访存故障，门禁期间写出 CPU 层约 24 GB。
  对照主 session 的 P4D4 门禁（`gate-p4d4-r1-gemkv-real`，prefill 无 offload）：196/200、32/32、31/32、3.462。
  （`results/b1/gate-b1-r1/`）
- **计分 b1-r1（05:08–06:04，C104 1800 s，一次）= 33,203 tok/s/GPU，比 s7b（30,091）高 10.3%**，prefill 0 次访存故障
  （`results/b1/score-b1-r1/`；s7b 在 v3-135 上跑，节点不同，跑间波动未量化）：

  | | s7b（prefill 无 GEM、0.6） | **b1-r1（prefill GEM、0.88）** | sglang |
  |---|---|---|---|
  | tok/s/GPU | 30,091 | **33,203** | 53,351 |
  | 输出 tok/s/GPU / req/s | 287.6 / 2.511 | 315.8 / 2.765 | — / 3.469 |
  | 完成 / 丢弃 | 4,593 / 5 | 5,053 / 7 | 12,582 / 1（3600 s） |
  | 交互 p50 / p90 | 59.4 / 42.3 | 56.0 / 42.7 | 69.9 / 60.8 |
  | ITL p50 / p90（ms） | 16.8 / 23.7 | 17.9 / 23.4 | — |
  | TTFT p50 / p90 / 均值（s） | 28.50 / 41.82 / 26.33 | 18.96 / 37.59 / 20.57 | 4.20 / 9.23 / 5.65 |
  | prefill 本地前缀命中（计数器） | 14.9% | 48.1% | — |

  GPU 前缀缓存从约 460 万扩到 1,208 万 token，本地命中从 14.9% 升到 48.1%，TTFT p50 降 33%；交互度略降（decode 更忙）。

### 1.6 对之前结论的更正

- 1b / U2 的候选嫌疑"150 GB 级单个 GEM BO、`srcAccessOrder=Any`、`/dev/shm` 共享区、拷贝所在的流与 GEM 映射的上下文"都不是原因：
  device 0 上 64 KiB × 256 的 GEM 批量拷贝正常，非 0 设备上同样的拷贝必然故障，且改用 VMM 导入后同一测试全部正常。
- U1 / U2 的状态可改为"根因已定（HIP 批量拷贝按未设置的 `deviceId` 选设备），Infera 侧已绕开（VMM 导入）"；上游候选新增 ROCm clr：
  `hipExternalMemoryGetMappedBuffer` 设置 `deviceId`（与 `hipIpcOpenMemHandle` 一致），或 `hipMemcpyBatchAsync` 按流的设备 /
  内存对象的上下文选设备。

## 2. B2：`TieringOffloadingSpec` 在 PP > 1 时起不来（U11）

- 机制（vLLM 0.34.0 源码）：`build_offloading_config` 的 `worker_kv_bytes_per_block = kv_cache_tensors[0].size // num_blocks` 用的是
  **本 worker** 的 `KVCacheConfig`；PP 下每个 worker 只含本 stage 的层（PP0 / 1 / 2 / 3 分别有 27 / 25 / 25 / 24 个 KV 张量），调度器用的是
  PP0 配置的深拷贝（`generate_scheduler_kv_cache_config`）。`CPUOffloadingSpec` 据此算 `kv_bytes_per_chunk = 每块字节 × world_size ×
  blocks_per_chunk`、`num_chunks`、每 worker 槽宽，各方都不一样：
  - `TieringOffloadingSpec`（各平台都用 `/dev/shm` 共享区）：区域总大小不同，加入方等创建方的文件长到自己的期望值，30 s 超时（s6）；
    即使大小碰巧相同，行内槽位偏移也不同。
  - ROCm 上的 `CPUOffloadingSpec`（每 rank 私有锁页张量）：调度器按 PP0 的 `num_chunks` 分配 CPU 块号，各 worker 的张量按自己的
    `num_chunks` 分配；若 PP0 不是最大的 stage，调度器会分出超过某些 worker 张量末尾的块号（待用日志数据确认是否发生，见下）。
- 修法（`results/b2/vllm-offload-pp.diff`）：`KVCacheConfig` 新增 `max_worker_bytes_per_block`，在
  `get_kv_cache_configs` 里与 `num_blocks`（取最小）、`needs_kv_cache_zeroing`（取或）一样跨 worker 统一（取最大），调度器配置随深拷贝继承；
  `build_offloading_config` 用它（不小于本 worker 的值）计算槽宽，与本 worker 不同时打一行日志。改 3 个文件：
  `v1/kv_cache_interface.py`、`v1/core/kv_cache_planning.py`、`distributed/kv_transfer/kv_connector/v1/offloading/config.py`，
  bind mount 进容器（`overlay/vllm/**`）。
- 实测每块 KV 字节数（64 token / 块，FP8 + indexer + MTP）：PP0 **796,416**、PP1 / PP2 779,520、PP3 742,656。PP0 最大，所以 ROCm 上
  现用的 `CPUOffloadingSpec`（调度器按 PP0 算 chunk 数）不会越界，只是 PP1–3 约 2–7% 的锁页内存分了没用。
- **聚合 PP4 验证**（`scripts/b2_agg_tier.sh` + `scripts/tier_probe.py`；GPU 800 块 × 64 token、CPU 层 2 GB、FS 层在本地 NVMe；
  6 条各约 2.4 万 token 的带暗号长 prompt，再重复前 3 条；各一次，`results/b2/agg/`）：
  - 不修（bug）：PP0 建了 2.00 GB 的区域，PP1 / PP2 打开（它们期望的大小 ≤ PP0，但槽位偏移已错），PP3 等文件长到 1,998,323,712 字节
    30 s 超时，引擎退出。与 s6 的现象一致。
  - 修后（fix）：起栈 101 s，9/9 正确；FS 层命中 1,461 个 chunk、读回 4.66 GB，写入 9.31 GB（2,922 个文件），外部前缀命中 93,504 token。
  - 回归（`CPUOffloadingSpec` + 修复）：9/9 正确。
- **P4D4 + FS 层（诊断，`scripts/b2_p4d4.sh r1`，3600 s，进行中）**：配置 = b1-r1 + `TieringOffloadingSpec`（1.25 TB `/dev/shm` 共享区
  + FS 层 32 读 / 16 写线程）。4 个 PP worker 加入同一块 1,250.00 GB 区域，无超时。
  - **新问题：PP0 / PP2 的区域锁页失败**（`host_register failed for rank=0 at 549.74 of 1250.00 GB; the offload region stays pageable`，
    rank 2 在 755.90 GB）。`pin_mmap_region` 只在"每行省下 ≥ 3 MiB"时才只注册本 worker 的槽（这里行 3,186,688 字节、槽 796,416，
    只省 2.4 MB），于是 4 个 worker 各注册整块 1.25 TB（合计约 5 TB），超出 KFD 的锁页额度。可分页的两个 stage 拷贝变慢，
    **这轮吞吐不能用于比较**，命中率随时间的变化仍可比。上游候选：多 worker 私有槽布局时总是只注册本 worker 的槽（或按总注册量判断）。
  - FS 层写入很快：测量开始 6 min 时已写 1.64 TB（每块都写穿到 FS）。

## 3. CPU 层容量：写穿 vs 写回；InferenceX 对 offload 层级的规则

### 3.1 vLLM 写穿、sglang 写回

- vLLM 0.34 OffloadingConnector：每步把新算完的整块排进下一步写出（写穿），只有 `store_threshold`（被提供 N 次才写）这种选择性写穿，
  没有"GPU 换出时才写"的模式。
- sglang P4D4 recipe（`../dp-graph-variant-fix-20261103/kit/engine/worker.sh`）：prefill 同样 PP4 × TP1、`--mem-fraction-static 0.88`，
  `--enable-hierarchical-cache --hicache-size 250 --hicache-write-policy write_back --hicache-io-backend kernel --hicache-mem-layout page_first`
  （每 rank 250 GB，共 1 TB）。
- 估算（按 vLLM 实测每块字节数，chunk 行 3,186,688 字节 / 64 token；sglang 的每 token 字节数假设相同、未实测）：

  | | 主机内存 | 主机层 token | GPU token | 不重复的有效容量 |
  |---|---|---|---|---|
  | vLLM 写穿 | 1.25 TB | 约 2,510 万 | 1,208 万（基本都在 CPU 层有副本） | ≈ 2,500 万 |
  | sglang 写回 | 1.0 TB | 约 2,000 万 | 约 1,200 万 | ≈ 3,200 万 |

  vLLM 用写穿追平约需 1.6 TB 锁页，高于本机 1.3–1.5 TB 的 OOM 边界（U10）。b1-r1 把 GPU 缓存扩大 2.6 倍，有效容量几乎不变，
  只是命中从 CPU 层挪到本地 GPU。
- 但 1800 s 内容量不是瓶颈：测量期每 5 min 现算比例 s7b 0.036–0.086、b1-r1 0.038–0.078、sglang ≈ 0.04；AIPerf 报的 trace 理论命中
  95.66%（现算 ≈ 0.043）。（`results/q2/b1-r1-hit_timeline.txt`，s7b 见 `../vllm-p4d4-crusoe-20261117/results/s7b/hit_timeline.txt`）
  当前与 sglang 的差距（33k 对 53k）主要不在命中：TTFT p50 19 s 对 4.2 s，而现算比例相近（主 session 1h：单 API server）。
- 合规地扩大有效容量的方向：OffloadingConnector 写回 / 排他存放（U9，同样 1.25 TB 有效容量约 3,700 万 token）；锁页量逼近 OOM 边界。

### 3.2 InferenceX（提交 918524ff，2026-08-27）的规则与已发布 recipe（`results/q2/`）

- `MODELS.md`「KV cache offloading policy」：AgentX 初期**只允许 CPU DRAM KV offload**（vLLM Connector、LMCache、SGLang HiCache、
  Mooncake CPU DRAM、Dynamo KVBM、CPU DRAM P2P pooling 等），是否启用由提交方决定；"Other offloading tiers, including NVMe KV cache
  offloading, are outside the initial scope"，暂定 v3.5 或 v4。DRAM 上限：每台基准 DRAM × 所用 GPU 比例，MI355X 每台 ≤ 3 TB
  （P4D4 用满 8 卡 → ≤ 3 TB）。
- `utils/matrix_logic/validation.py`：单机 / 多机 agentic 条目 `kv_offloading: Literal["none", "dram"]`，`dram` 时必须给
  `kv-offload-backend`；`process_agentic_result.py` 再校验一次。master config 必须用 `cluster:<name>` runner；`configs/runners.yaml`
  声明每台可用 DRAM，生成器按 `dram-utilization` 算 `total-cpu-dram-gb`，recipe（如 `server_sglang.sh`）按它设 HiCache 大小。
- 已发布 recipe（`configs/amd-master.yaml`、`benchmarks/`）：offload 条目全是 `dram`，后端 hicache / vllm-native / vllm-simple /
  lmcache（0.4.5、0.5.3）/ mooncake；dsv4 sglang-disagg 的"L3"是 Mooncake store（RDMA 汇聚的 DRAM segment），不是磁盘。
  **没有用 NVMe / SSD / 文件层的 recipe**（无 LMCache `local_disk`、HiCache file 后端、vLLM tiering fs）。
- 结论：FS 二级层不能计入 AgentX 成绩；B2 的 FS 层结果只作诊断。U11 修复本身仍是有效的上游 PR（PP > 1 下 `TieringOffloadingSpec`
  即使不配二级层也会因共享区不一致起不来）。

## 4. 上游 PR / issue 素材

| 目标 | 内容 | 素材 |
|---|---|---|
| ROCm clr（`ROCm/rocm-systems`，`projects/clr/hipamd/src/hip_memory.cpp`） | `hipExternalMemoryGetMappedBuffer` 建的内存对象不设 `getUserData().deviceId`（默认 0），`hipMemcpyBatchAsync` 按它分组、把非 0 设备上导入内存的拷贝放到 device 0 的 null stream，device 0 没有该映射，GPU 访存故障；`hipPointerGetAttributes` 也报 device 0。建议：`hipExternalMemoryGetMappedBuffer`（以及 `hipImportExternalMemory` 的 `ExternalBuffer`）设 `deviceId = hip::getCurrentDevice()->deviceId()`，与 `hipIpcOpenMemHandle` 相同 | 最小复现 `scripts/batch_ext_repro.c`（不依赖 GEM：`hipMalloc` → dma-buf → `hipImportExternalMemory` 导回同一设备 → `hipMemcpyBatchAsync`）；`results/b1/swapdev*.txt` |
| Infera（GEM 分配器，`llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c`，随镜像构建） | BO 改用 HIP VMM 导入（`hipMemImportFromShareableHandle` + `hipMemAddressReserve` / `hipMemMap` / `hipMemSetAccess`），旧路径 `HIP_GEM_IMPORT=external` | `results/b1/hip_gem_alloc-vmm-import.diff` |
| vLLM（U11） | `KVCacheConfig.max_worker_bytes_per_block` 在 `get_kv_cache_configs` 里跨 worker 取最大，`build_offloading_config` 按它分槽：修 `TieringOffloadingSpec` 在 PP > 1 时区域大小 / 槽位不一致，也让 `CPUOffloadingSpec` 的调度器与各 worker 的 chunk 数一致 | `results/b2/vllm-offload-pp.diff`；`results/b2/agg/` |
| vLLM（ROCm 上 `TieringOffloadingSpec` 的锁页） | `pin_mmap_region` 在多 worker 私有槽布局下每个 worker 注册整块共享区（PP4 × 1.25 TB ≈ 5 TB），超出 KFD 锁页额度后该 worker 的 CPU 层退成可分页内存；建议私有槽布局时只注册本 worker 的槽，或按总注册量决定 | 2 节 P4D4 + FS 日志 |

## 5. 文件

- `scripts/`：`qb.sh`（投递）、`repro_gem_offload.sh`（聚合 PP4 复现，可挂载替换库 / vLLM 文件）、`swap_dev_test.py` +
  `swap_dev_cases.sh`（按设备隔离）、`swap_perf_test.py`（拷贝开销）、`build_gem.sh`、`b1_p4d4.sh`（B1 P4D4 门禁 + 计分）、
  `monitor_b.sh`（登录节点监控）、`batch_ext_repro.c`（ROCm 最小复现）、`p4d4_run.sh`（`b1_p4d4.sh` 加 `RUN_PREFIX`）、
  `b2_agg_tier.sh` + `tier_probe.py`（B2 聚合读回验证）、`b2_p4d4.sh`（P4D4 + FS 层，每轮新目录、跑完删除）。
  其余脚本同 `../vllm-pd-tools-20261117/scripts/`（只改了 `R=`；`agentx.sh` 加了 `PREP_ONLY=1`）。
- `results/b1/`：隔离测试输出、两轮复现的配置 / smoke / 日志摘录、GEM 分配器 diff、门禁与计分摘要。
- `results/b2/`：vLLM diff、聚合验证（bug / fix / cpuspec）。`results/q2/`：InferenceX 规则 / schema / recipe 摘录、b1-r1 命中时间线。
