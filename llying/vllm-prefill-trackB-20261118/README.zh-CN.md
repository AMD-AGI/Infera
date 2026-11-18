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
  0 次故障（修前第 4 项 500、1 次故障）。P4D4（prefill GEM、显存目标 0.88、1.25 TB CPU 层）门禁与计分进行中（见 1.5）。
- B2（U11）：修法已写好（引擎核心统一各 worker 每块 KV 字节数的最大值，offload 配置按它分槽），待在 PP4 上验证（见 2 节）。

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

进行中：`scripts/b1_p4d4.sh r1 gate score`（先真实接受率部署 + `gate.sh`，再重新部署计分 AgentX C104 1800 s）。

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
- 修法（`results/b2-vllm-offload-pp.diff` 待验证后放入）：`KVCacheConfig` 新增 `max_worker_bytes_per_block`，在
  `get_kv_cache_configs` 里与 `num_blocks`（取最小）、`needs_kv_cache_zeroing`（取或）一样跨 worker 统一（取最大），调度器配置随深拷贝继承；
  `build_offloading_config` 用它（不小于本 worker 的值）计算槽宽，与本 worker 不同时打一行日志。改 3 个文件：
  `v1/kv_cache_interface.py`、`v1/core/kv_cache_planning.py`、`distributed/kv_transfer/kv_connector/v1/offloading/config.py`。

## 3. 文件

- `scripts/`：`qb.sh`（投递）、`repro_gem_offload.sh`（聚合 PP4 复现，可挂载替换库 / vLLM 文件）、`swap_dev_test.py` +
  `swap_dev_cases.sh`（按设备隔离）、`swap_perf_test.py`（拷贝开销）、`build_gem.sh`、`b1_p4d4.sh`（B1 P4D4 门禁 + 计分）、
  `monitor_b.sh`（登录节点监控）。其余脚本同 `../vllm-pd-tools-20261117/scripts/`（只改了 `R=`；`agentx.sh` 加了 `PREP_ONLY=1`）。
- `results/b1/`：隔离测试输出、两轮复现的配置 / smoke / 日志摘录、GEM 分配器 diff。
