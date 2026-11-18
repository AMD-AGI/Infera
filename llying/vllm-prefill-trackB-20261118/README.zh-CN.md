# B 线：GLM-5.2 P4D4 prefill 侧 GEM + offload 故障、vLLM 多级 offload 在 PP 下起不来（2026-11-18，v3-258 / v3-137）

接手 `../vllm-p4d4-crusoe-20261117/HANDOFF-trackB.zh-CN.md`。节点 crusoe v3-258（作业 294784），镜像 `infera-vllm:v0340-0c5e5e89`
（vLLM 0.34.0+rocm100、torch 2.13.0+rocm10.0.0、ROCm SDK 10.0）。工作目录 `/home/liyingli/bench_agentx/vllm-trackB-20261118/`，
命令经 `scripts/qb.sh`（包一层 `q2.sh`，先 `cd` 到工作目录并设 `R`）投递到 `queue-n2/`。

## 0. 结论

- **B1（U2）已解决**：根因在 HIP，`hipMemcpyBatchAsync` 按设备侧内存对象的 `deviceId` 选执行设备，而
  `hipImportExternalMemory` / `hipExternalMemoryGetMappedBuffer` 导入的 GEM BO 从不设置 `deviceId`（默认 0）。PP rank k > 0 的 offload
  拷贝于是在 device 0 上执行，device 0 没有该 BO 的映射 → `HSA_STATUS_ERROR_MEMORY_FAULT`。修复在 Infera GEM 分配器：BO 改用 HIP VMM 导入
  （`hipMemImportFromShareableHandle` + `hipMemAddressReserve` / `hipMemMap` / `hipMemSetAccess`），批量拷贝 51 GB/s、KV 池不变（1 节）。
- **P4D4 最好成绩（一次）：b1-r1 = 33,203 tok/s/GPU（C104，1800 s）**，比 s7b（30,091）高 10.3%：prefill 用 GEM、显存目标 0.88
  （GPU KV 1,208 万 token，s7b 约 460 万），其余同 s7b；门禁通过（smoke 6/6、GSM8K 197/200、捞针 32/32、接受长度 3.458），0 次访存故障。
- **B2（U11）已修并验证**：`KVCacheConfig.max_worker_bytes_per_block` 在 `get_kv_cache_configs` 里跨 worker 取最大，offload 按它分槽；
  聚合 PP4 复现原超时、修后读回正确，P4D4 上 4 个 PP worker 加入同一块 1,250 GB 共享区（2 节，diff 在 `results/b2/`）。
- **NVMe FS 二级层不能用于 InferenceX 提交**：AgentX 只允许 CPU DRAM 级 KV offload，NVMe 明确不在范围内；FS 层 3600 s 只作诊断
  （34,350 tok/s/GPU，两个 stage 的 CPU 层退成可分页内存，吞吐不可比）（3.2 节）。
- **写穿 vs 写回**：sglang recipe 用 HiCache `write_back`，vLLM OffloadingConnector 只有写穿。实测 1.25 TB 写穿在 1800 s 和 3600 s 运行的
  前 36 min 内现算比例约 5%，接近容量不受限（FS）时，容量还不是瓶颈；3600 s 后半段见 3.5 节同节点对照。vLLM 0.34.0 的
  `SimpleCPUOffloadConnector` lazy（写回）在 P4D4 上**更差**：28,696 tok/s/GPU，现算比例约 14%（写穿约 5%），GPU 未命中的 token 只有
  12% 在 CPU 层（一次 CPU 命中分配的 GPU 块超出 lazy 扫描窗口，#66708 / #66766 所述）（3.3–3.5 节）。ATOM 没有写回模式；
  OffloadingConnector 的写回在上游 RFC #61396 二期 #65681，未合并。
- **未解决**：DRAM-only 3600 s 对照三次都在预热第 15–23 min 因主机侧进程崩溃中止（prefill API server、decode + prefill、路由；
  两次直接崩在 HF tokenizers 里，一次的崩溃地址是被 ASCII 文本覆盖的指针），无 MCE / OOM，倾向于 tokenizers 0.23.3 并发下的堆破坏，
  影响所有 P4D4 计分（3.5 节）。因此 3600 s 写穿后半段的命中没有数据。b1-r1（33,203）的重复轮因主机 OOM 无效：
  13:03 拆除后约 **1.27 TB 锁页内存泄漏**（无任何进程持有），v3-258 重启前不宜再跑大 CPU 层（3.5 节）。

### 0.1 给主 session 合并的更新（CANDIDATES / 1b / 1e 等）

- U2：根因已定（HIP 批量拷贝按未设置的 `deviceId` 选设备），Infera 侧已修（GEM 分配器 VMM 导入，`hip_gem_alloc.c` 已提交，
  镜像下次构建即生效）；1b 的嫌疑项（150 GB BO、`srcAccessOrder`、`/dev/shm`、流）全部排除。U1 可解除"等 U2"。
- 新上游候选：ROCm clr `hipExternalMemoryGetMappedBuffer` 设置 `deviceId`（最小复现 `scripts/batch_ext_repro.c`）。
- U11：修复已验证（3 文件 diff）；另发现 ROCm 上 `TieringOffloadingSpec` 每个 worker 锁页整块共享区，PP4 × 1.25 TB 超出 KFD 额度后退成可分页（新候选）。
  U10 补充：现用 `CPUOffloadingSpec` 中调度器按 PP0 算 chunk 数，PP0 恰是最大的 stage，所以不会越界，只是 PP1–3 有 2–7% 锁页未用。
- U9：写回在 vLLM 0.34.0 只有 `SimpleCPUOffloadConnector` lazy 可用，P4D4 实测更差；OffloadingConnector 的写回见 RFC #61396 / #65681。
- 新候选：`SimpleCPUOffloadConnector` 锁页分块要求 `num_cpu_blocks` 使按层主机张量页对齐，否则起栈失败（绕法 `num_cpu_blocks=389120`）。
- 新风险：P4D4 主机侧偶发崩溃（今天 7 轮 P4D4 里 3 轮，均为 b1 配置的 3600 s 对照；崩在 prefill / decode / 路由的 tokenizers 或
  libtorch），见 3.5 节。
- 节点状态：v3-258 在 13:03 一轮拆除后泄漏约 1.27 TB 锁页内存（`Shmem` 不下降、无进程持有），之后一轮主机 OOM；需要重启或上报管理员。
  U10 补充：锁页 CPU 层的进程异常结束后，内存可能不归还，下一轮部署前应检查 `/proc/meminfo` 的 `Shmem`。

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
- **P4D4 + FS 层（诊断，`scripts/b2_p4d4.sh r1`，3600 s，07:29–09:07，一次）= 34,350 tok/s/GPU**（req/s 2.526，完成 / 丢弃 9,164 / 28，
  TTFT p50 / p90 19.55 / 35.71 s，0 次访存故障；FS 层共写 3.52 TB、110 万个文件）。配置 = b1-r1 + `TieringOffloadingSpec`
  （1.25 TB `/dev/shm` 共享区 + FS 层 32 读 / 16 写线程）。4 个 PP worker 加入同一块 1,250.00 GB 区域，无超时。FS 层不合规，只作诊断。
  FS 文件由容器以 root 写入，宿主机 `rm` 失败；已改为在 root 容器里删（`b2_p4d4.sh`），本轮残留 3.3 TB 已删除。
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

### 3.3 写回（lazy / 换出时写）的现状调研（2026-11-18 09:20–09:45，`gh` + 源码）

| 实现 | 写入时机 | 状态 / 版本 | 对我们（vLLM 0.34.0、ROCm、PP4、MultiConnector[Mooncake + offload]、GLM-5.2） |
|---|---|---|---|
| vLLM `OffloadingConnector`（现用） | 写穿（eager），可选 `store_threshold` | 0.34.0 | 有效容量 ≈ CPU 层 |
| vLLM RFC #61396 + #64286（一期，store trigger 接口，+3,098/−1,709）+ #65681（二期，`"store_mode": "lazy"`，`ON_OVERWRITE`：块交给新主人那一步、前向前同步写，+1,854/−93） | 写回 | 一期 2026-11-12 合并（不在 0.34.0），二期未合并（多次 needs-rebase，作者注明为"实现建议"）；三期 `ON_FREE_QUEUE`（semi-lazy）未开始 | 要移植一期 + 二期，约 5,000 行改动；二期限制：`TieringOffloadingSpec` 只接受 eager、拒绝 self-describing KV events、V2 runner 下有限制。PR 数据：Qwen3-32B CPU 层吃紧时 CPU 命中 +52 pp、吞吐 +64% |
| vLLM #43946（OffloadingConnector 换出监听 + 同一步写，`KV_OFFLOAD_LAZY_STORE=1`） | 写回 | 作者 2026-09-19 自行关闭（未说明，之后由 RFC #61396 接手） | 只支持单 KV group；TPU 480B 上 +56% 输出吞吐 |
| vLLM `SimpleCPUOffloadConnector` `"lazy_offload": true` | 写回（每步从 GPU 空闲队列头扫，把将被换出、CPU 里没有的已缓存块异步写出，保持 `target_free` 个块"空闲或已备份"） | **0.34.0 已有**，含 #65237（扫描从队头开始、写完的块回到换出端，11-05 合并）；未合并的改进 #66708（换出前一定先写、优先淘汰已加载回 GPU 的 CPU 副本，+122/−32）、#66766（扫描覆盖 CPU 加载，+96/−2）、#44791（最短前缀阈值） | 支持 ROCm（`hipMemcpyBatchAsync`，8,192 描述符分块）、PP（按 world_size 计完成数、按 rank 平分容量）、MultiConnector（转发 `bind_gpu_block_pool`）；主机内存每 rank 私有匿名 mmap + 分块注册，无 tiering 那种整区重复注册。InferenceX 已发布 recipe 在用：minimax-m3 B200 / B300（`lazy_offload: true`）、kimi-k3 MI355X vLLM（`SIMPLE_LAZY_OFFLOAD` 开关），后端名 `vllm-simple`。风险：#42371（持续负载下 Running 0 / Waiting N 卡住，未关闭）、#66708 指出的漏写与重复副本；GLM-5.2 的 MLA + indexer + MTP 在 ROCm 上未见验证 |
| ATOM（`ROCm/ATOM` 9b2e67b，`atom/kv_transfer/offload/`） | 写穿：每算完一个 `LMCACHE_CHUNK_SIZE`（256 token）就存，请求结束时补存尾部；LMCache CPU 层 LRU / SLRU 淘汰；`OFFLOAD_MIN_SAVE_TOKENS` 门槛 | — | **没有写回模式**；MP 模式拒绝 PP；vLLM 插件 connector 只是 LMCache 字节编解码适配，不是 vLLM 的 OffloadingConnector |

**自己做的路线（按代价从低到高）**

1. **先换用 `SimpleCPUOffloadConnector` 的 lazy 模式**（只改配置）：`P_KV_EXTRA` 里 OffloadingConnector 换成
   `{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1250000000000,"lazy_offload":true}}`，
   InferenceX 元数据 `KV_OFFLOAD_BACKEND=vllm-simple`。先在聚合 PP4 上用 `tier_probe.py` 验读回正确性（GPU 缓存压小），再上 P4D4 3600 s 与 DRAM-only 对照比。
2. 若出现漏写 / 重复副本（CPU 命中低于预期），把 #66708 + #66766（共约 220 行，只动 Simple 的 manager）做成 overlay。
3. 长期（OffloadingConnector 上游方向）：不另起设计，向 RFC #61396 / #65681 提供 ROCm + PP + MLA / DSA + MTP + PD 的验证与修复。
4. 只有前三条都不行才自己写：参照 #43946 / #65681 二期——
   - 调度器：在 `BlockPool` 把带 hash 的已缓存块交给新主人（`_maybe_evict_cached_block`）时记下 `(block_id, chunk key)`；
     同一步 `build_connector_meta` 为其中 CPU 层没有的 chunk 发 store job（无 request 上下文，`req_context=None`），
     并把块的 chunk 记录（overwrite map）在计算完成 / 从 CPU 加载完成时登记，前缀命中不算覆盖；
   - worker：在前向之前（`start_load_kv`）把这批 D2H 拷贝放到计算流上或让计算流等待拷贝流，保证先读后写
     （每步约 128 块 × 0.8 MB / stage ≈ 100 MB，约 2 ms）；
   - 多 KV group（GLM-5.2 的 MLA + indexer + MTP draft）要按 chunk 对齐各 group 的块，或限定 group 数；PP 无需特殊处理（调度器单一、各 stage 拷自己的层）；
   - 可选排他存放：块从 CPU 加载回 GPU 后降低其 CPU 副本的保留优先级（#66708 的做法）。
   估计 300–500 行 + 单测，正确性风险集中在"块被重新分配但旧内容已被覆盖"的时序上。

### 3.4 SimpleCPUOffloadConnector（lazy）在 PP4 上的聚合验证（`results/b3/agg/`，各一次）

同一读回探测（GPU 800 块 × 64 token，6 条约 3.1 万 token 的 prompt + 重复前 3 条），只换 offload：

| 配置 | 读回正确 | 外部（CPU / FS）命中 token | 备注 |
|---|---|---|---|
| OffloadingConnector eager，CPU 2 GB（`cpuspec`） | 9/9 | 0 | CPU 层装不下 |
| OffloadingConnector eager + FS 层（`fix`） | 9/9 | 93,504（FS 命中 1,461 chunk） | 重复请求从 FS 读回 |
| Simple eager，CPU 6 GB | 9/9 | 0 | 每 rank 1,883 块 |
| **Simple lazy，CPU 6 GB** | 9/9 | **16,704**（两次相同） | DEBUG 日志：32 个写出事件全部完成，每次约 140 块（合计约 4,500 块 > 容量，CPU 侧 LRU 淘汰） |
| Simple lazy + #66708（overlay） | 9/9 | 320 | — |

- **更正（10:00）**：09:45 曾据"OffloadingConnector 全部命中"判断 Simple lazy 漏写、并中止了 Simple lazy 的 P4D4；那一轮的命中来自 FS 层，
  不可比。同为只有 CPU 层时，eager（两种 connector）都是 0，lazy 反而有命中，符合写回的预期。这个探测只证明读回正确，比不出容量。
- 0.34.0 的 Simple 在 ROCm + PP4 + GLM-5.2（MLA + indexer + MTP）上能起栈、写出能完成、读回正确，0 次访存故障。
- #66708 在这个小探测里没有改进（320 对 16,704，一次），不采用；P4D4 用 0.34.0 原版 lazy（与 InferenceX recipe 一致）。

### 3.5 P4D4 3600 s：DRAM-only 对照、Simple lazy、主机侧崩溃

- `b1-r2-3600`（b1-r1 配置，3600 s，09:08 起）预热第 17 min（09:39）中止：prefill 引擎子进程 `exited (code=-11)`，日志只有
  `Segfault encountered ... tokenizers::tokenizer::added_vocabulary::AddedVocabulary::refresh_added_tokens`（Rust 帧，无 Python 栈）。
  崩溃前 30 s AIPerf 返回数停在 366、在途 125；之后预热中 85 个请求出错，AgentX 因根请求失败中止。prefill 0 次访存故障。
- `refresh_added_tokens` 在改 fast tokenizer 的 added-token 相关状态时调用（如切换 `encode_special_tokens` / `split_special_tokens`），
  与其他线程上同一个 tokenizer 的 encode 并发时会出现这类崩溃；vLLM / tokenizers 上游未搜到对应 issue。之前 s7b、b1-r1、b2-r1 等轮未出现，
  属偶发（一次）。会影响所有 P4D4 计分，**请主 session 留意**（多 API server 时每个进程各有一个 tokenizer）。
- **第二次对照（`b1-r3-3600`，10:00 起）也失败**：预热 1,147 / 1,155 时（10:36）decode 引擎 `exited (code=-11)`，prefill 引擎
  `exited (code=-4)`（SIGILL），根请求失败，AgentX 中止；prefill 0 次访存故障。节点内核日志（用带 `CAP_SYSLOG` 的容器只读 `dmesg`）：
  - `09:37:07 traps: python3[2099165] general protection fault ... in tokenizers.abi3.so`（第一次对照）
  - `10:36:12 traps: python3[2196557] trap invalid opcode ip:739c3a2cf000 ... in libtorch_cpu.so`（第二次；ip 恰在页边界）
  - 无 MCE / EDAC / OOM 记录；decode 的 SIGSEGV 没有对应的 trap 行。
  两次都在 DRAM-only（`CPUOffloadingSpec` + GEM）3600 s 运行的第 15–23 min，崩在不同进程、不同库；同配置 b1-r1（1800 s，运行 56 min）、
  FS 那轮（`TieringOffloadingSpec`）都没出现。当时推测是偶发的主机内存被破坏（如 DMA 写错页）；第三次崩溃后更倾向于 tokenizers
  的堆破坏（见下）。
- **Simple 的锁页对齐问题（上游候选）**：P4D4 上 Simple 自动算出 `num_cpu_blocks=392,382`，按层切分的主机张量多数起点不对齐页，
  `host_registration_ranges` 在一个锁页分块里找不到"块边界且页边界"的切点，起栈失败（`No block and page boundary fits the registration
  chunk limit`）。绕法：显式 `"num_cpu_blocks": 389120`（95 × 4,096），起栈 231 s，每 rank 锁页 282.5–288.6 GB。
- Simple lazy 的 P4D4 3600 s（`b3-r4-simple-lazy`，v3-258，10:47 起，不单独跑门禁，计分自带 smoke）：预热 23.5 min（b1-r1 20.5 min），
  测量 11:23 起。**中期（11:35）明显不如写穿**：每 5 min 现算比例 0.075–0.21（b1-r1 写穿 0.038–0.078），本地 GPU 命中 0.75–0.89、
  CPU 层只有 0.01–0.06。`/metrics`：GPU 前缀命中 2.48 亿 / 3.14 亿 token；未命中 GPU 的 6,620 万 token 里只有 799 万（12%）在 CPU 层找到，
  读回 124,840 块。即被 GPU 换出的块大多没有 CPU 副本。推断原因（#66708 / #66766 所述）：lazy 每步只为约 2 × chunk（256 块 / group）
  的换出端预备份，而一次 CPU 命中会一次性分配上千个 GPU 块，超出扫描窗口的块被直接换出。之前的小探测几乎没有 CPU 读回，
  所以 3.4 节"#66708 无改进"不适用于 P4D4。
- **b3-r4 最终（一次）= 28,696 tok/s/GPU**（req/s 2.142，完成 / 丢弃 7,767 / 6，交互 p50 / p90 59.7 / 46.3，TTFT p50 / p90 / 均值
  27.25 / 86.56 / 38.60 s，prefill 本地命中 73.4%，0 次访存故障），低于写穿（b1-r1 1800 s 33,203；b2-r1 写穿 + FS 3600 s 34,350）。
- 第二节点 v3-137（burst）**12:14 被抢占取消**（`PREEMPTED`，运行 1 h 17 min），其上的写穿对照 `b1-r4-3600-n4` 只跑了 36 min 测量期，
  无计分，排在其后的 Simple lazy + #66708 未跑。
- 测量期每 5 min 现算比例（`results/q2/*hit_timeline.txt`）：

  | 配置 | 测量期 | 现算比例 | CPU / FS 层命中 |
  |---|---|---|---|
  | 写穿 DRAM-only（v3-137，被抢占前 36 min） | 11:38–12:14 | 0.015–0.081（多数约 0.05） | 0.32–0.41 |
  | Simple lazy（v3-258，60 min） | 11:23–12:23 | 0.114–0.208（多数约 0.14） | 0.04–0.27（随时间上升） |
  | 写穿 + FS（容量近似无限，60 min） | 08:04–09:04 | 0.027–0.078 | 0.27–0.57 |

- **第三次对照（`b1-r5-3600`，v3-258 同节点，12:27 起）也在预热末尾（13:02）中止**：AIPerf 13:02:28 收到路由的
  `ServerDisconnectedError`，根请求失败；三个服务容器日志里都没有退出记录。内核日志：`13:02:22 python3[2468822]: segfault at 746e65747469
  ... in tokenizers.abi3.so`。访问地址 `0x746e65747469` 按小端是 ASCII `"ittent"`，即一个指针被文本覆盖：同一进程内的堆破坏
  （数据竞争 / 释放后使用），不像硬件或 DMA 写错页。路由（Python 后端，用 tokenizer 算前缀哈希）是最可能的崩溃进程。
- **三次崩溃汇总**（都在 v3-258，镜像 tokenizers 0.23.3 / transformers 5.22.0）：09:37 prefill API server（tokenizers GP fault）；
  10:36 decode SIGSEGV + prefill SIGILL（`libtorch_cpu.so`，页边界 ip）；13:02 路由（tokenizers segfault，指针被 ASCII 覆盖）。
  分别落在三个不同进程，共同点是 HF tokenizers；**更倾向于是 tokenizers 在并发分词下的软件问题**（之前"疑似 DMA 写错主机内存"的推测降级）。
  同期完成的运行：b1-r1（1800 s）、b2-r1（FS，3600 s）、b3-r4（Simple lazy，3600 s）；v3-137 上的对照在被抢占前 36 min 未崩。
  建议主 session：给 P4D4 的各进程加 faulthandler / core dump，或试 `TOKENIZERS_PARALLELISM=false`、换 tokenizers 版本。

- **b1-r1 的重复轮（`b1-r6-1800`，13:05 起）无效**：测量期 72 / 713 个请求失败（10.1%，超过 10% 门槛），计分 13,404 不可用。
  原因是**主机 OOM**：`13:45:08 Out of memory: Killed process 2500533 (VLLM::Worker_PP) ... shmem-rss:368435556kB`。
  拆除后（13:48–13:53，无容器、无 KFD 进程、无僵尸 / D 状态进程、`/dev/shm` 为空）`Shmem` 一直是 1,271,876,212 kB：
  **约 1.27 TB 锁页内存泄漏**，最可能来自 13:02 崩溃那轮（`b1-r5`）prefill 的 CPU 层，13:03 拆除后没有释放；新一轮再锁 1.25 TB 即超出
  2.75 TB 物理内存。早上几轮崩溃后的下一轮都能正常锁页，所以不是每次崩溃都会泄漏。**v3-258 在重启前不宜再跑大 CPU 层**（作业 15:00 结束后节点带着这块泄漏回池）。

  结论：至少在 3600 s 运行的前 36 min，1.25 TB 写穿的命中已接近容量不受限时（FS），容量不是瓶颈；0.34.0 的 Simple lazy 现算约为写穿的 3 倍。
  写回的收益只来自容量，1800 s 内写穿已接近 trace 理论上限，所以不再申请可被抢占的节点去跑 1800 s 的 #66708；是否值得做写回取决于
  v3-258 同节点对照（`b1-r5-3600`）后半小时的命中是否下降。
- 新节点：amd-agentx-qos 每人 2 个作业已满（294784 + 主 session 排队的 296460），0day QOS 不允许本用户，hyperloom / primus-cicd
  额度满；已用 amd-burst-qos 提交 `hold_b.sbatch`（作业 296512，低优先级、可被取消），**10:56 在 v3-137 起来、12:14 被抢占**，独立工作目录
  `/home/liyingli/bench_agentx/vllm-trackB-n4-20261118/`，预排任务自动完成镜像构建、模型拷贝（38 s）、AgentX 准备，
  DRAM-only 对照（`b1-r4-3600-n4`）11:13 起。

## 4. 上游 PR / issue 素材

| 目标 | 内容 | 素材 |
|---|---|---|
| ROCm clr（`ROCm/rocm-systems`，`projects/clr/hipamd/src/hip_memory.cpp`） | `hipExternalMemoryGetMappedBuffer` 建的内存对象不设 `getUserData().deviceId`（默认 0），`hipMemcpyBatchAsync` 按它分组、把非 0 设备上导入内存的拷贝放到 device 0 的 null stream，device 0 没有该映射，GPU 访存故障；`hipPointerGetAttributes` 也报 device 0。建议：`hipExternalMemoryGetMappedBuffer`（以及 `hipImportExternalMemory` 的 `ExternalBuffer`）设 `deviceId = hip::getCurrentDevice()->deviceId()`，与 `hipIpcOpenMemHandle` 相同 | 最小复现 `scripts/batch_ext_repro.c`（不依赖 GEM：`hipMalloc` → dma-buf → `hipImportExternalMemory` 导回同一设备 → `hipMemcpyBatchAsync`）；`results/b1/swapdev*.txt` |
| Infera（GEM 分配器，`llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c`，随镜像构建） | BO 改用 HIP VMM 导入（`hipMemImportFromShareableHandle` + `hipMemAddressReserve` / `hipMemMap` / `hipMemSetAccess`），旧路径 `HIP_GEM_IMPORT=external` | `results/b1/hip_gem_alloc-vmm-import.diff` |
| vLLM（U11） | `KVCacheConfig.max_worker_bytes_per_block` 在 `get_kv_cache_configs` 里跨 worker 取最大，`build_offloading_config` 按它分槽：修 `TieringOffloadingSpec` 在 PP > 1 时区域大小 / 槽位不一致，也让 `CPUOffloadingSpec` 的调度器与各 worker 的 chunk 数一致 | `results/b2/vllm-offload-pp.diff`；`results/b2/agg/` |
| vLLM（`SimpleCPUOffloadConnector` 锁页分块） | 自动算出的 `num_cpu_blocks` 使按层切分的主机张量不对齐页时，`host_registration_ranges` 找不到切点、起栈失败；建议把 `num_cpu_blocks` 向下取整到使各层起点页对齐的倍数，或允许切点落在页边界而非块边界 | 3.5 节 |
| vLLM（ROCm 上 `TieringOffloadingSpec` 的锁页） | `pin_mmap_region` 在多 worker 私有槽布局下每个 worker 注册整块共享区（PP4 × 1.25 TB ≈ 5 TB），超出 KFD 锁页额度后该 worker 的 CPU 层退成可分页内存；建议私有槽布局时只注册本 worker 的槽，或按总注册量决定 | 2 节 P4D4 + FS 日志 |

## 5. 文件

- `scripts/`：`qb.sh`（投递）、`repro_gem_offload.sh`（聚合 PP4 复现，可挂载替换库 / vLLM 文件）、`swap_dev_test.py` +
  `swap_dev_cases.sh`（按设备隔离）、`swap_perf_test.py`（拷贝开销）、`build_gem.sh`、`b1_p4d4.sh`（B1 P4D4 门禁 + 计分）、
  `monitor_b.sh`（登录节点监控）、`batch_ext_repro.c`（ROCm 最小复现）、`p4d4_run.sh`（`b1_p4d4.sh` 加 `RUN_PREFIX`）、
  `b2_agg_tier.sh` + `tier_probe.py`（B2 聚合读回验证）、`b2_p4d4.sh`（P4D4 + FS 层，每轮新目录、跑完删除）。
  其余脚本同 `../vllm-pd-tools-20261117/scripts/`（只改了 `R=`；`agentx.sh` 加了 `PREP_ONLY=1`）。
- `results/b1/`：隔离测试输出、两轮复现的配置 / smoke / 日志摘录、GEM 分配器 diff、门禁与计分摘要。
- `results/b2/`：vLLM diff、聚合验证（bug / fix / cpuspec）。`results/q2/`：InferenceX 规则 / schema / recipe 摘录、b1-r1 命中时间线。
