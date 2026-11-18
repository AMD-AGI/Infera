# B 线交接给主 session（2026-11-18 15:20）

B 线（prefill GEM + offload 故障、vLLM 多级 offload 在 PP 下起不来）在 v3-258（作业 294784，已于 15:00 到期）和 v3-137（burst 作业 296512，
12:14 被抢占）上完成。细节与过程在同目录 `README.zh-CN.md`；本文只写主 session 需要的：能直接用的东西怎么开、要合并的条目、未解决的崩溃、
节点状态、数据在哪。所有成绩都是单次运行。

## 1. 已验证可用的东西与开法

### 1.1 GEM VMM 导入库（U2 的修复）：prefill 可以用 GEM + CPU offload

- **问题与修法**：ROCm clr 的 `hipExternalMemoryGetMappedBuffer` 不设内存对象的 `deviceId`（默认 0），`hipMemcpyBatchAsync` 按它分组执行，
  PP rank k > 0 的 offload 拷贝跑到 device 0 上而故障。GEM 分配器改用 HIP VMM 导入（`hipMemImportFromShareableHandle` +
  `hipMemAddressReserve` / `hipMemMap` / `hipMemSetAccess`），HIP 记录的就是导入时的设备。
- **文件**
  - 源码：`llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c`（本分支已提交，md5 `36521d5d…`；diff 在 `results/b1/hip_gem_alloc-vmm-import.diff`）。
    下次按该 Dockerfile 重建镜像即自带新库，不用再挂载。
  - 构建好的库（与镜像 `infera-vllm:v0340-0c5e5e89` 配套）：`/home/liyingli/bench_agentx/vllm-trackB-20261118/gem/libhip_gem_alloc.so`
    （md5 `b948027d…`），**与主线的 `/home/liyingli/bench_agentx/vllm-pd-20261117/bin/libhip_gem_alloc-vmm.so` 是同一个文件**。
    重新构建：`scripts/build_gem.sh`（在镜像里用 Dockerfile 同一条 gcc 命令）。
- **开法（不重建镜像）**：prefill 容器把新库覆盖到镜像里的路径，其余 GEM 设置不变。
  ```bash
  P_GEM=1 \
  P_MOUNT_EXTRA="/home/liyingli/bench_agentx/vllm-pd-20261117/bin/libhip_gem_alloc-vmm.so:/opt/infera-gem/libhip_gem_alloc.so:ro" \
  P_ARGS='--gpu-memory-utilization 0.88 --block-size 64 ...' \
  bash scripts/pd1_up.sh RUN
  ```
  `P_GEM=1` 时 `pd1_up.sh` 会加 `INFERA_AMD_GEM_ALLOCATOR` / `LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so` 和
  `"kv_connector_module_path":"infera.engine.vllm.gem_mooncake"`；库要求的 `MC_DISABLE_HIP=1 MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0`
  已在 `mc_env` 里。库的开关：`HIP_GEM_IMPORT=vmm`（默认）/ `external`（旧导入，与 offload 同用会故障）。
  decode 仍用镜像里的旧库即可（decode 不做 offload，不受 U2 影响；VMM 库未在 decode 上验证）。
- **验证过的完整配置（b1-r1）**：s7b 配置（Python 串行路由、缓存感知准入 128 / 0、1.25 TB CPU 层 LRU、offload 描述符上限 512、decode 单 API server）
  + prefill GEM（新库）、`--gpu-memory-utilization 0.88`。脚本 `scripts/b1_p4d4.sh`；实际命令行 `results/b1/score-b1-r1/engine-config.env`。

  | 验证 | 结果 |
  |---|---|
  | 独立测试（`scripts/swap_dev_test.py`，device 0–3，D2H / H2D） | 修前 device 1–3 全部故障，修后全部正常；批量拷贝 51 GB/s（与普通显存相同） |
  | 聚合 PP4 复现（GEM Mooncake + 100 GB offload） | 修前 smoke 第 4 项 500、1 次故障；修后 5/5、0 次 |
  | P4D4 门禁 | smoke 6/6、GSM8K 197/200（乱码 0）、16k 捞针 32/32、接受长度 3.458 |
  | P4D4 计分 b1-r1（C104 1800 s） | **33,203 tok/s/GPU**（s7b 30,091，+10.3%），GPU KV 1,208 万 token，本地命中 48% |

- **注意**：同一配置的 3600 s 运行三次因主机侧崩溃中止（第 3 节）；主线 s12 / s12b / s12d（prefill GEM 0.88）也崩过。根据现有数据，
  **GEM（或 GEM 带来的更高负载）与崩溃的关系不能排除**，见 3.3。

### 1.2 U11 修复：PP > 1 时 offload 的共享区大小 / 槽位一致

- **修法**：`KVCacheConfig` 新增 `max_worker_bytes_per_block`，在 `get_kv_cache_configs` 里跨 worker 取最大（与 `num_blocks` 取最小同处），
  调度器配置随深拷贝继承；`build_offloading_config` 用它（不小于本 worker 的值）计算每 worker 槽宽。
- **文件**（整文件覆盖，基于镜像里的 vLLM 0.34.0，换 vLLM 版本要重做；diff 在 `results/b2/vllm-offload-pp.diff`）：
  `overlay/vllm/v1/kv_cache_interface.py`、`overlay/vllm/v1/core/kv_cache_planning.py`、
  `overlay/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py`（本目录下；工作目录副本在
  `/home/liyingli/bench_agentx/vllm-trackB-20261118/overlay/vllm/`）。
- **开法**：三个文件 bind mount 到 prefill 容器的 `/usr/local/lib/python3.12/dist-packages/vllm/` 下对应位置；**没有开关**，挂上即生效。
  ```bash
  O=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-prefill-trackB-20261118/overlay/vllm
  V=/usr/local/lib/python3.12/dist-packages/vllm
  U11=""; for f in v1/kv_cache_interface.py v1/core/kv_cache_planning.py \
      distributed/kv_transfer/kv_connector/v1/offloading/config.py; do U11+=" $O/$f:$V/$f:ro"; done
  P_MOUNT_EXTRA="<GEM 库挂载> $U11" bash scripts/pd1_up.sh RUN
  ```
  生效时每个 PP worker 打一行 `KV offload: 779520 KV bytes per block on this worker, 796416 on the largest worker of the engine`。
- **影响**：
  - `TieringOffloadingSpec` 在 PP > 1 时必需（不修时加入方等共享区文件 30 s 超时，即 s6 的失败）。
  - 现用的 `CPUOffloadingSpec`：调度器的 chunk 数不变（PP0 本来就是最大的 stage，每块 796,416 字节；PP1 / PP2 779,520，PP3 742,656），
    PP1–3 少锁约 2.7% 主机内存（1.25 TB 时约 34 GB），不改变命中。
- **验证**：聚合 PP4（`scripts/b2_agg_tier.sh` + `scripts/tier_probe.py`）不修时复现超时，修后从 FS 层读回的重复请求 9/9 正确；
  P4D4 上 4 个 PP worker 加入同一块 1,250 GB 共享区（b2-r1）；`CPUOffloadingSpec` + 修复的回归 9/9。
- **FS（NVMe）二级层**：InferenceX AgentX 只允许 CPU DRAM 级 KV offload，**FS 层不能用于提交**，只作诊断。开法见 `scripts/b2_p4d4.sh`
  （每轮新目录；FS 文件由容器以 root 写，必须在 root 容器里删，脚本已处理）。b2-r1 诊断 34,350 tok/s/GPU（3600 s），
  但 PP0 / PP2 的共享区锁页失败、退成可分页内存（见 2.2 的 U19），吞吐不可比。

### 1.3 验证过但不推荐

- **`SimpleCPUOffloadConnector` + `lazy_offload`（vLLM 0.34.0 唯一可用的写回式 offload）**：
  `{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1250000000000,"lazy_offload":true,"num_cpu_blocks":389120}}`，
  InferenceX 后端名 `vllm-simple`。必须显式给 `num_cpu_blocks`（4,096 的倍数，见 U20），否则起栈失败。
  P4D4 3600 s（b3-r4）= **28,696 tok/s/GPU**，现算比例约 14%（写穿约 5%）：GPU 未命中的 token 只有 12% 在 CPU 层找到。
- vLLM #66708（Simple lazy 的改进，未合并）做成了 overlay：`overlay-simple/vllm/v1/simple_kv_offload/manager.py`，只在聚合小探测上跑过，未上 P4D4。

## 2. 要合并进 CANDIDATES 的条目

当前 CANDIDATES 到 U16，A 线交接用了 U17 / U18，以下新条目从 U19 起编号。

### 2.1 现有条目的更新

| # | 改动 |
|---|---|
| U1 | 状态：U2 已解决，可推进（Infera `gem_mooncake.py` + 新 GEM 库，P4D4 prefill / decode 都用过 GEM） |
| U2 | 状态改为"**已解决**（Infera GEM 分配器 VMM 导入；P4D4 门禁 + 1800 s 计分通过，一次）"；ROCm clr 部分加证据：不依赖 GEM 的最小复现 `../vllm-prefill-trackB-20261118/scripts/batch_ext_repro.c`（`hipMalloc` → dma-buf → `hipImportExternalMemory` 导回同一设备 → `hipMemcpyBatchAsync`，device 1 / 3 故障于 `GPU index: 0, kernel __amd_rocclr_copyBufferBatch`，结果 `results/b1/batch_ext_repro.txt`）；1b 列的嫌疑（150 GB BO、`srcAccessOrder`、`/dev/shm`、流）全部排除 |
| U9 | 证据：sglang recipe 是 `--hicache-write-policy write_back --hicache-size 250`（×4 = 1 TB）；估算有效容量 vLLM 写穿约 2,500 万 token、sglang 写回约 3,200 万。但 1.25 TB 写穿在 1800 s 与 3600 s 运行的前 36 min 现算约 5%，接近容量不受限（FS）时，**容量暂不是瓶颈**；3600 s 后半段无数据（对照三次崩溃）。vLLM 现有写回 = `SimpleCPUOffloadConnector` lazy，P4D4 上更差（28,696）；OffloadingConnector 的写回 = RFC #61396 二期 #65681（未合并，`store_mode: lazy`）；Simple lazy 的补漏 #66708 / #66766 未合并；ATOM 无写回。改法：跟进上游 #65681 / #66708，提供 ROCm + PP + DSA + MTP + PD 的验证。状态"调研完成，待上游" |
| U10 | 补充：`CPUOffloadingSpec` 的调度器按 PP0 的配置算 chunk 数，PP0 恰是最大的 stage，所以不越界；锁页 CPU 层的进程异常结束后，内存可能不归还（U23） |
| U11 | 状态"**已修**（3 文件，聚合 PP4 + P4D4 验证），待提 PR"；改法定为引擎核心统一 `max_worker_bytes_per_block`；证据 `../vllm-prefill-trackB-20261118/` 2 节、`results/b2/` |

### 2.2 新条目

| # | 缺口 | 证据 | 建议改法 | 仓库 | 状态 |
|---|---|---|---|---|---|
| U19 | ROCm 上 `TieringOffloadingSpec` 每个 worker 锁页**整块**共享区：`pin_mmap_region` 只在"每行省 ≥ 3 MiB"时才只注册本 worker 的槽（PP4 时行 3.19 MB、槽 0.80 MB），PP4 × 1.25 TB ≈ 5 TB 注册，超出 KFD 额度后 PP0 / PP2 退成可分页内存（`host_register failed ... stays pageable`） | `../vllm-prefill-trackB-20261118/` 2 节（b2-r1 日志） | 多 worker 私有槽布局时总是只注册本 worker 的槽，或按总注册量决定 | vLLM | 待提 |
| U20 | `SimpleCPUOffloadConnector` 锁页分块要求按层切分的主机张量起点页对齐：自动算出的 `num_cpu_blocks=392,382` 时 `host_registration_ranges` 找不到"块边界且页边界"的切点，起栈失败（`No block and page boundary fits the registration chunk limit`） | 同上 3.5 节 | `num_cpu_blocks` 向下取整到使各层起点页对齐的倍数，或允许切点只对齐页 | vLLM | 待提 |
| U21 | **P4D4 主机侧偶发崩溃**（tokenizers 堆破坏嫌疑）：今天 7 轮 P4D4 计分里 3 轮在预热第 15–23 min 中止，分别崩在 prefill API server、decode + prefill、路由侧进程；两次崩在 `tokenizers.abi3.so`，一次的崩溃地址是被 ASCII 文本覆盖的指针 | 本文第 3 节；`results/b1/control-failures/` | 见 3.4 的排查顺序 | Infera / vLLM / HF tokenizers | 定位中 |
| U22 | 写回式 offload 在 P4D4 上的覆盖缺口：Simple lazy 每步只为约 2 × chunk 的换出端预备份，一次 CPU 命中分配的上千个 GPU 块超出窗口，被直接换出（CPU 层只找到 12% 的 GPU 未命中） | 同上 3.5 节（b3-r4 `/metrics`） | 合入 #66708（加载时按分配量扫描）后在 P4D4 复测 | vLLM | 待复测 |
| U23 | **锁页内存泄漏**：13:02 崩溃那轮在 13:03 拆除后，v3-258 上约 1.27 TB `Shmem` 不归还（无容器、无 KFD 进程、无僵尸 / D 状态进程、`/dev/shm` 为空，5 min 内不变），下一轮锁 1.25 TB 时主机 OOM | 同上 3.5 节；`results/b1/control-failures/b1-r6-1800-score.txt` | 部署前检查 `/proc/meminfo` 的 `Shmem`；上报节点管理员；查 KFD / amdgpu 在进程异常退出后的 GTT BO 释放 | ROCm 驱动 / 运维 | 待查 |

## 3. tokenizers / 主机侧崩溃

### 3.1 证据（都在 v3-258；镜像 tokenizers 0.23.3、transformers 5.22.0）

| 时间 | 运行 | 现象（服务日志） | 内核日志（`docker run --rm --cap-add SYSLOG ... dmesg -T`） |
|---|---|---|---|
| 09:37 | b1-r2-3600（b1 配置，3600 s） | prefill 引擎子进程 `exited (code=-11)`；日志 `Segfault encountered ... tokenizers::tokenizer::added_vocabulary::AddedVocabulary::refresh_added_tokens`（无 Python 栈） | `traps: python3[2099165] general protection fault ... in tokenizers.abi3.so` |
| 10:36 | b1-r3-3600 | decode `exited (code=-11)`，prefill `exited (code=-4)` | `traps: python3[2196557] trap invalid opcode ip:739c3a2cf000 ... in libtorch_cpu.so`（ip 恰在页边界，像跳到被破坏的函数指针）；decode 无对应行 |
| 13:02 | b1-r5-3600 | 三个服务容器日志都无退出记录；AIPerf 13:02:28 收到路由（`:28300`）的 `ServerDisconnectedError` | `python3[2468822]: segfault at 746e65747469 ... error 6 in tokenizers.abi3.so`：地址按小端是 ASCII `"ittent"`，即指针被文本覆盖 |

- 三次都在 b1 配置的 3600 s 运行里；同期完成的有 b1-r1（同配置 1800 s，运行 56 min）、b2-r1（Tiering + FS，3600 s）、b3-r4（Simple lazy，3600 s）；
  v3-137 上的 b1 配置在被抢占前跑了 36 min 测量期未崩。均无 MCE / EDAC / OOM 记录（OOM 只在 13:45 的 b1-r6，原因是 U23 的泄漏）。
- 13:02 那次的进程**未能确认身份**（容器日志都没有退出记录）；路由（Python 后端，kv-aware 策略用 tokenizer 算前缀哈希，U15）最可能，
  AIPerf 本身也用 tokenizers 计数。

### 3.2 机制线索（源码，未复现）

- tokenizers 0.23.3 `tokenizers/src/tokenizer/added_vocabulary.rs`：`refresh_added_tokens()` 只在 `add_tokens` / `add_special_tokens`
  和更换 normalizer 时调用（重建 added-token 的匹配结构）。崩在这里说明运行时有代码在改某个 tokenizer 的 added tokens / normalizer，
  而同一个 tokenizer 正被别处使用。
- transformers 5.22.0 `tokenization_utils_tokenizers.py`：每次 encode 都调用 `set_truncation_and_padding(...)`，
  且 `split_special_tokens` 与当前值不同时写 `self._tokenizer.encode_special_tokens`，即**每次调用都改共享的 Rust tokenizer 状态**；
  不同调用方（API server 的 chat 模板、推理 / 工具解析器、路由）设置不同时，多线程下会互相改写。
- "指针被 ASCII 覆盖"像同一进程里字符串缓冲区越界或释放后复用（prompt 文本覆盖了堆上的指针），不像 GPU DMA 写错页（KV 数据是二进制 FP8）。
  所以 B 线之前"疑似 DMA 写错主机内存"的推测已降级。

### 3.3 与 GEM 的关系

- 支持"与 GEM 无关"：13:02 的崩溃最可能在路由（不用 GPU）；崩溃点在 tokenizers / libtorch 的 CPU 代码，且有一次是文本覆盖指针。
- 不能排除 GEM：B 线所有 P4D4 都开了 prefill GEM；主线 s12 / s12b / s12d（prefill GEM 0.88）都崩了，s13（不用 prefill GEM）没崩（一次）。
  一个可能的统一解释是 **GEM 0.88 让 prefill 更快，API server / 路由的并发分词更多，竞争窗口更大**；但这只是推断。

### 3.4 建议的排查方法（按代价从低到高）

1. **每轮都能归因**：所有容器（prefill、decode、路由、AgentX）加 `-e PYTHONFAULTHANDLER=1`（段错误 / 非法指令时打印所有线程的 Python 栈）；
   每 30 s 把 `ps -eo pid,ppid,etimes,cmd` 和各容器的 `docker top` 写进 run 目录，用来对上 dmesg 里的 PID；每轮结束后抓
   `docker run --rm --cap-add SYSLOG --entrypoint bash <镜像> -c 'dmesg -T | tail -200'`。core dump 依赖宿主机 `core_pattern`（这台节点是 apport 管道，
   GPU coredump 日志也提示 apport 不存在），普通用户改不了，先靠 faulthandler。
2. **不依赖 GPU 的复现**：`scripts/tokenizer_stress.py`（本目录；**未运行**）在计算节点上用 GLM-5.2 的 tokenizer 开 16 个线程对约 11 万 token 的文本
   并发 encode / `apply_chat_template`，奇偶线程用不同的 `split_special_tokens` 与截断设置，跑 30 min。崩了就是可以报给
   huggingface/tokenizers 的复现；不崩再加 `add_special_tokens` 之类的运行时修改，或换成真实的 AgentX 对话。
3. **减少并发分词的调用方**：路由改用 Rust 后端（主线 s13 已用，也解决 U15）；查 vLLM 0.34 API server 是否在线程池里分词 / 渲染 chat 模板
   （以及推理、工具解析器是否持有同一个 tokenizer），必要时每线程一个 tokenizer 副本；试 `TOKENIZERS_PARALLELISM=false`（只关 Rust 内部并行，
   作用有限但成本为零）。
4. **区分 GEM 因素**：同一配置（Rust 路由、上面的归因开关）分别跑 prefill GEM 0.88 和不用 GEM（显存目标 0.6）各 2–3 轮 3600 s，数崩溃次数；
   若只在 GEM 下崩，再用 `HIP_GEM_IMPORT=external` + 不开 offload 区分"VMM 导入"与"GEM 本身"。
5. **换 tokenizers 版本**：派生镜像里 `pip install tokenizers==<其他版本>`（transformers 5.22 的版本约束内），对比崩溃频率。

## 4. 节点状态（15:20）

| 节点 | 状态 | 说明 |
|---|---|---|
| v3-258（作业 294784，15:00:43 TIMEOUT 结束） | 已被其他用户的作业 296250（yueliu14，约 15:00 起）占用，`mix` | **13:53 时约 1.27 TB `Shmem` 泄漏**（U23），未见重启；对方大概率遇到主机内存不足，**建议告知管理员 / 该用户**。B 线没有动 `queue-n2/STOP`，runner 随作业结束；`queue-n2/` 里 B 线的项都已结束 |
| v3-137（burst 作业 296512，12:14 被抢占） | `drain`（原因不可见） | 抢占时批处理脚本的清理已删掉 `vpd-*` 容器 |

- 两台节点本地盘上残留（可重用或删除）：`/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4`、`work/cache/`（InferenceX、AgentX 依赖与 trace）、
  `build-0c5e5e89/`；`kvfs-trackB/` 已清空。镜像 `infera-vllm:v0340-0c5e5e89` 在两台的本地 docker 里。
- 可复用：`/home/liyingli/bench_agentx/vllm-trackB-20261118/hold_b.sbatch`（amd-burst-qos，可被抢占；runner 执行 `queue-b/`，用 `qb4.sh` 投递）。
  本用户的 amd-agentx-qos 每人 2 个作业，`amd-agentx-0day-qos` 不允许本用户。

## 5. 原始数据在哪

- **B 线工作目录** `/home/liyingli/bench_agentx/vllm-trackB-20261118/`：`runs/<run>/`、`logs/summary.out`（每轮一行成绩）、`results/`（隔离测试、门禁、
  InferenceX 摘录、命中时间线）、`gem/`、`overlay/`、`overlay-simple/`、`src/vllm/`（镜像里 vLLM 0.34.0 的 Python 源码）、`src-up/`（ROCm/ATOM、
  vLLM main 浅克隆）、`upstream-prs/`（#66708 / #66766 diff）、`queue-b/`（第二节点的队列项与日志）。
- **第二节点工作目录** `/home/liyingli/bench_agentx/vllm-trackB-n4-20261118/`：`runs/b1-r4-3600-n4-score/`（被抢占，36 min 测量期的 `samples.jsonl`）。
- **队列项日志**：`/home/liyingli/bench_agentx/vllm-pd-20261117/queue-n2/1118-*.{run,log,rc}`（B 线投递的项，名字如 `Lb1r1`、`Lb3score2`）。
- 每个 P4D4 run 目录里：`engine-config.env`（实际命令行）、`samples.jsonl`（10 s 一次的引擎指标）、`agentx-c104/`（AIPerf 产物：`runner.log`、
  `aiperf_artifacts/logs/aiperf.log`、成功时的 `profile_export_aiperf.json` 与 `agentx_conc104.json`）、`vpd-{prefill,decode,router}.stream.log.gz`
  （实时落盘的容器日志）、`smoke.out`、`m0.json` / `m1.json` / `metrics-delta.json`、`hang/`、`prefill-faults.txt`。命中时间线：
  `python3 scripts/hit_timeline.py runs/<run>/samples.jsonl http://<节点 IP>:27201 300`（在计算节点上跑）。

| run | 内容 | 结果 |
|---|---|---|
| `repro-r0-baseline` / `repro-r1-vmm` | 聚合 PP4，GEM Mooncake + offload，修前 / 修后 | 故障 1 次 / 0 次 |
| `b1-r1-gate` / `b1-r1-score` | P4D4，prefill GEM 0.88 + 1.25 TB 写穿 | 门禁通过 / 33,203（1800 s） |
| `b2-agg-{bug,fix,cpuspec}` | 聚合 PP4，U11 修前 / 修后（Tiering + FS）/ CPUOffloadingSpec + 修复 | 超时 / 9/9 / 9/9 |
| `b2-r1-score`（FS 目录用量在 `b2-r1/`） | P4D4 + Tiering + FS（诊断） | 34,350（3600 s） |
| `b2-agg-simple-*` | 聚合 PP4，Simple eager / lazy / lazy + #66708 / lazy DEBUG | 均 9/9 |
| `b3-r4-simple-lazy-score` | P4D4 + Simple lazy | 28,696（3600 s） |
| `b1-r2-3600-score` / `b1-r3-3600-score` / `b1-r5-3600-score` | b1 配置 3600 s 对照 | 三次崩溃中止（第 3 节） |
| `b1-r6-1800-score` | b1-r1 的重复 | 主机 OOM（U23），无效 |
| `b3-r1-*`、`b3-r2-*`、`b3-r3-*` | 中止的 Simple 尝试（误判后停掉 / 锁页对齐失败） | 无结果 |

- **仓库里的摘要**（本目录）：`results/b1/`（隔离测试、复现、门禁、计分摘要、`control-failures/` 含三次崩溃与 OOM 的摘录和 dmesg）、
  `results/b2/`（U11 diff、聚合验证）、`results/b3/`（Simple 聚合验证、b3-r4 摘要、#66708 diff）、`results/q2/`（InferenceX 规则 / schema / recipe
  摘录、各轮命中时间线）、`results/q3/`（InferenceX 里 Simple / ATOM recipe 的用法）、`scripts/`（全部脚本）。
