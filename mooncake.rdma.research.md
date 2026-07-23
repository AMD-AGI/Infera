# Mooncake RDMA / HIP dma-buf 深度调研

> 归属:sglang image upgrade 项目 · Phase 2(enable USE_HIP_DMABUF / 查 KFD 耗尽)前置调研
> 日期:2026-07-23
> 方法:mooncake 官方 source code + git log + PR/issue body(远程 API 调研 + 本地
> `/home/yihou/dev/git.16-19/Mooncake` git 逐条核对)。**所有结论均有 commit/PR/行号
> 支撑,无猜测。**
> 本地 Mooncake checkout:`main` HEAD `e97eee2c`(2026-07-23,`v0.3.12-pre1-184`)。

---

## 0. 调研目标(用户提的 4 问)

1. 最新 mooncake(main)里 dma-buf / USE_HIP 的代码路径是怎样的?
2. USE_HIP 下 multi-node 的 MR 注册问题,上游是否已修复?
3. `ibv_reg_dmabuf_mr` 在哪里,它和 `ibv_reg_mr` 的**使用条件 / 编译条件**是否变了?
4. `USE_HIP_DMABUF` 这个编译开关的**原始真实意图**是什么?为什么"既然要支持
   multi-node DMA、默认却不打开",而**当前不打开的情况下我们也成功跑通并吃到了
   跨机 GPU(NodeA)↔GPU(NodeB) 的带宽**?

---

## 1. 三条正交的 HIP 轴(先厘清概念——用户 Q4 的前提有个反转)

mooncake 里"HIP 相关开关"其实是**三条互相独立**的轴,不能混为一谈:

| 轴 | 是什么 | 作用域 | 默认 | 我们镜像现状 |
|---|---|---|---|---|
| **USE_HIP** | 编译 ROCm/HIP 后端 | 编译期 | — | ON |
| **HIP transport**(`"hip"` 协议,`hipIpcOpenMemHandle`/XGMI) | **单机** GPU-IPC 快路 | 运行期 | #2682 起无条件安装 | 用 `MC_ENABLE_HIP_TRANSPORT` **关掉** |
| **USE_HIP_DMABUF** | MR **注册方法**(`ibv_reg_dmabuf_mr` vs `ibv_reg_mr`) | 编译期 | `option(... ON)` | 因 base 传播 bug **被编译掉** |

**关键纠正**:
- **HIP transport 从不是跨机路径**,它是单机 IPC。跨机 KV 永远走 **RDMA transport**。
  所以关掉 HIP transport 不影响跨机能力。
- **USE_HIP_DMABUF 默认是 ON**(`src/CMakeLists.txt:106` `option(USE_HIP_DMABUF ... ON)`),
  不是"默认不打开"。用户感知的"没打开"其实是**旧 base 上被传播 bug 编译掉了**。
- 跨机带宽我们**确实吃到了**,但不是靠 dmabuf,而是靠**另一条等价 GPUDirect 路径**:
  bare `ibv_reg_mr` + amdgpu peermem(见 §5)。

---

## 2. 关键时间线(git log 实证)——我们的 base 卡在"闯祸 commit,修复之前"

| commit | 日期 | PR | 作用 | 相对我们 base |
|---|---|---|---|---|
| `01d1eb2a7e` | 2026-07-01 | **#2682** | 装 HIP transport,**只为单机**;`hip` 优先级硬编码 > rdma → **闯祸** | = **我们的 base** |
| `45b84d36cf` | 2026-07-03 | **#2725** | 修跨机:①`selectDevice` 协议盲→SIGSEGV;②`MC_DISABLE_HIP` 逃生阀;③**修 CMake 传播 bug** | base 之后 |
| `af29ca7e02` | 2026-07-06 | **#2753** | 自动修复:`isHipReachableTarget()` per-target 同主机门,跨机自动回落 rdma,**免 env** | base 之后 |
| `7755a146`  | 2026-07-16 | **#2523** | **共享单 dma_buf fd** across NICs,消 ×N BAR1 | base 之后 |

> base 的 mooncake 版本 = `v0.3.12-pre1+23`(即 `01d1eb2a`)。三个修复(#2725/#2753/#2523)
> **全部晚于 base**。`git merge-base --is-ancestor` 已验证。

sglang v0.5.15.post1 base 打包的正是 `01d1eb2a`(见 CLAUDE.md 记录 + 本地 git 确认存在)。

---

## 3. Q1 — main 里 dma-buf / USE_HIP 的代码路径

main 相比 base 做了 **export/register 拆分重构(#2523)**:

```
registerLocalMemory
  → exportDmabuf(addr, &exp)                       # 每块 allocation 只导出一次 dma_buf fd
  → 各 NIC: registerMemoryRegionInternal(..., exp) # 各 NIC import 同一个 fd
  → closeDmabufExport(exp)
```

- **`exportDmabuf()`** `rdma_context.cpp:386`(main):按后端分支决定导出。
  HIP 分支 `#elif defined(USE_HIP_DMABUF)`(main line 465)用 `hipPointerGetAttributes`:
  - host → `kHostReg`(bare ibv_reg_mr)
  - managed → 回退 `kHostReg`(页会迁移,fd 会 stale;有 WARNING)
  - device 且 `isKernelDmabufSupported()` → `hsa_amd_portable_export_dmabuf` 拿 fd → `kDmabufReg`
  - device 但内核不支持 → 回退 `kHostReg`
- **`registerMemoryRegionInternal()`** `rdma_context.cpp:561`(main):
  `exp.method==kDmabufReg` → `ibv_reg_dmabuf_mr`(line 578);否则 `ibv_reg_mr`(581/585)。
- 重构意图(注释 line 573-577):**"all NICs share one dma_buf object (and one BAR1 window)"**
  —— 8 NIC 共享 1 fd / 1 BAR1 窗口(不是每 NIC 各导一份)。**这条与 Phase 2 资源耗尽直接相关**。

**base 时代形态对比**(`git show 01d1eb2a7e:.../rdma_context.cpp:480-530`):export/register
**尚未拆分**,每块 allocation 各自 `hsa_amd_portable_export_dmabuf` + `ibv_reg_dmabuf_mr` +
`close(fd)`,即**每 NIC ×N 导入**。#2523 才改成共享单 fd。

---

## 4. Q2 — multi-node reg 问题上游是否修复:**修了,两连击**

### #2725(commit message 逐字为证,`git show 45b84d36cf`)
> "After multi-protocol (rdma+hip) segment registration, a single GPU address carries
> two buffer descriptors: a hip twin (IPC handles, no lkey/rkey) and an rdma twin
> (valid lkey). The RDMA data path did not disambiguate them, which crashes cross-node
> GPU KV transfer on AMD."

三条独立 fix:
1. `selectDevice()` 协议盲 → 选到 hip twin(空 lkey)→ OOB → SIGSEGV。修:`ENABLE_MULTI_PROTOCOL`
   下跳过 protocol 非 "rdma" 的 buffer。
2. `selectTransport()` 硬编码 hip 优先级(4)> rdma(2) → 跨机选到 hip → `hipIpcOpenMemHandle`
   失败。修:加 `MC_DISABLE_HIP` 逃生阀。
3. **传播 bug**(见 §5)。

**验证环境(重磅)**:
> "Validated on **two independent AMD RoCE fabrics (bnxt_re, ionic)**: cross-node 1P1D
> disaggregation KV transfer correct, concurrency sweep 100% success, zero SIGSEGV."

**ionic = 我们的 NIC**。上游已在 ionic 上把跨机 1P1D + 并发跑通。

### #2753(自动化,`multi_transport_locality.h` + `multi_transport.cpp`)
`isHipReachableTarget(target, local)` = 比较 target 与 local 的 host 部分(处理 IPv4/
hostname/IPv6/大小写)。`selectTransport` 里 `if (buffer.protocol=="hip" && !hip_reachable)
continue;`(main `multi_transport.cpp:499`)—— **跨机自动跳过 hip buffer 回落 rdma,无需任何 env**。
#2682 的 hardcode `if(p=="hip") return 4` 被 #2725 改成 `return std::getenv("MC_DISABLE_HIP")?0:4`。

### #2682 自认单机(PR body)
> "Support rdma+hip multi-protocol segments for **single-node** disaggregation."
> "Scope is intentionally bounded to rdma+hip." 测试仅 "Single-node 1P1D on MI355X."
→ **压根没考虑跨机**,那个无条件 hip 优先就是跨机 PD 崩溃根因。

**我们的 `MC_ENABLE_HIP_TRANSPORT` 补丁 = #2753 的粗版替代**(我们直接不装 HIP transport;
上游用 per-target 门,单机快路仍保留)。**rebase 到 ≥#2753 的 base 后,本补丁可撤。**

---

## 5. Q3 — `ibv_reg_dmabuf_mr` 位置 + 编译条件变化(=传播 bug 修复)

- **位置**:`rdma_context.cpp`,`registerMemoryRegionInternal` 内(main:578;base:426/506)。
- **编译条件**:`#if defined(USE_MLU)||USE_MACA||USE_CUDA||USE_HIP_DMABUF`(main line 571)。
- **变化核心(传播 bug)**:

| | base `01d1eb2a` | main(#2725 起) |
|---|---|---|
| define 目标 | `transfer_engine PRIVATE USE_HIP_DMABUF` | **`rdma_transport PRIVATE USE_HIP_DMABUF`** + hsa link |
| 后果 | `rdma_context.cpp` 编在 `rdma_transport` OBJECT 库,**拿不到宏** → dmabuf 分支 `#else` 掉 → 静默 bare `ibv_reg_mr` | 宏到达 → dmabuf 分支真正编入 |

`git show 01d1eb2a7e:.../rdma_transport/CMakeLists.txt` 证明 base 的 rdma_transport
**只有 MLX5DV,无 USE_HIP_DMABUF** → base 上 dmabuf 100% 被编译掉(实锤)。

#2725 diff 注释与我们 memory 记录**逐字一致**:
> "A PRIVATE define on transfer_engine never reaches that object compilation, so the
> dmabuf path is silently compiled out and GPU MRs fall back to plain ibv_reg_mr
> (EINVAL on device memory)."

**结论**:传播 bug 上游 07-03(#2725)已修;我们 base 只是早两天,卡在修复前。

- **运行时条件**(未变):`isKernelDmabufSupported()`(main line 71)查
  `CONFIG_PCI_P2PDMA` + `CONFIG_DMABUF_MOVE_NOTIFY`;`MOONCAKE_DISABLE_HIP_DMABUF` 强关;
  managed 内存回退 bare。chi2879/chi2865 两个内核配置均具备。

---

## 6. Q4 — USE_HIP_DMABUF 原始意图 + 为何不走它也吃到跨机带宽

### 原始意图(issue #751 + PR #2225,2026-05-31,agent 查 PR body)
bare `ibv_reg_mr` **无法注册 AMD GPU 显存**做 RDMA —— 除非有 peer-memory 内核模块。
#751 报 MI300X SGLang PD `Failed to register memory ... Invalid argument [22]`。维护者:
> "lack of nv_peermem driver support (NVIDIA only)... ibv_reg_dmabuf_mr is based on a
> general DMA buffer, you should be able to disable CUDA and substitute... ROCm APIs."

PR #2225 实现:device/managed → `hipMemGetAddressRange` → `hsa_amd_portable_export_dmabuf`
→ `ibv_reg_dmabuf_mr()`;host 仍 bare。**USE_HIP_DMABUF 就是为跨机 GPUDirect 注册而生的
AMD 方案**,走上游内核标准 dma-buf,免 nvidia-peermem 那种 out-of-tree 模块。

### 为何编译掉 dmabuf 仍能跨机 GPU↔GPU?——存在两条等价 GPUDirect 路
1. **bare `ibv_reg_mr` + amdgpu peermem**(需 vendor `ib_peer_mem`/amdgpu peermem 模块)
   —— **我们现在走的**
2. **`ibv_reg_dmabuf_mr` + dma-buf**(纯上游内核,免 vendor 模块)—— USE_HIP_DMABUF 那条

两条都是 zero-copy GPUDirect,给的是**同一个跨机 GPU↔GPU DMA 能力**。#751 年代 bare 在 AMD
上直接失败(旧 ionic 25.08 = FAIL_14),才要 dmabuf;但**新 ionic 26.03 上 bare 对显存 OK**。
于是链条:base 传播 bug 编掉 dmabuf → 运行时静默落 bare → 新 ionic 上 bare 恰好能用 →
**跨机带宽照吃**。不是巧合,是"两条路里另一条正好通"。

一句话:**跨机能力来自 RDMA transport(始终在)+ bare `ibv_reg_mr`;HIP transport 关不关、
dmabuf 编不编,都不影响这条跨机路。**

---

## 7. KFD 耗尽 / HIP-209 专项(Phase 2 的真问题)

- **mooncake 上游 issue/PR 全文搜 `KFD`/`hipModuleLoad`/`209`/`no kernel image` → 查无此条**
  (agent 用 GitHub search API 核实)。KFD 唯一命中是 #3061 的 `--device=/dev/kfd` docker flag。
- 两条相邻但**不同**的资源问题:
  - **#2523**:N 张 NIC 各导一份 dma-buf → **BAR1**(非 KFD)×N 耗尽。修法 = 共享单 fd
    (= main 现在的 export/register 拆分)。**8 NIC ×8→×1,是资源耗尽最可能的直接解药。**
  - **#2752**:单机下为"无跨机消费者的 device buffer"导 dma-buf,把 HSA runtime 搞坏 →
    下次 `hsa_queue_create` 崩(scratch `group_segment_size=(uint32_t)-1`)。被称
    "AMD ROCr-runtime robustness bug"。是 **HSA scratch,非 KFD/209**,且**单机**场景。
- vLLM `build_mooncake_rocm.sh` 关 dmabuf 躲的 KFD-209,在 mooncake 上游**无对应记录**。
  很可能是 **#2523 之前"每 NIC ×N 导入"形态**下的特定 gpu-util/KV-pool 规模现象。

---

## 8. Phase 2 净结论 + 下一步(compact 后从这里接着干)

### 结论
1. **不用再手写任何 mooncake 补丁**。传播 bug(#2725)、跨机 hip 误选(#2753)、资源 ×N
   (#2523)三个修复都在上游 main、且都晚于我们 base。我们的 `MC_ENABLE_HIP_TRANSPORT`
   补丁 = #2753 粗版替代,rebase 后可撤。
2. **KFD-209 上游查无此条**,其最可能诱因(×N dma-buf 导入)恰被 #2523 消除。
3. 正确下一步不是"在旧 base 上修 KFD",而是**用含 #2523 的新形态复压验证**。

### 下一步(开发/调试/验证)
- **目标**:在含 #2523 共享-fd + 默认 USE_HIP_DMABUF 的 mooncake 上,ionic 上跑 DSv4 PD,
  TRACE 确认走 `ibv_reg_dmabuf_mr`(非 bare),高 gpu-util/并发压测,看 **KFD-209 是否随
  ×N→×1 消失**。上游已在 ionic 证明跨机正确性,我们主要验资源侧。
- **两种取源**:
  - (A) 本地 `/home/yihou/dev/git.16-19/Mooncake`(main HEAD,已含全部修复)直接搭复压环境;
  - (B) 把镜像 mooncake bump 到含 #2523 的 pin,重建 infera image 后按 PD 复现 kit 压测。
- **复现 kit**:`examples/deepseek_v4/engine/pd_{mooncake,mori}/sglang/`。
- **节点**:chi2879(prefill 10.2.122.10)+ chi2865(decode 10.2.122.52),MI355X gfx950,
  jump 149.28.124.225。模型 `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed`。
- **验证手段**:mooncake `MC_LOG_LEVEL=TRACE` 看每个 KV-pool buffer 的注册路径;
  KFD/HIP-209 观测在 decode warmup / 首推理;VRAM 用 amdgpu sysfs `mem_info_vram_used`
  (非 `hipMemGetInfo` 假象,非 rocm-smi 绝对值)。
- **RDMA 调试守则**:每轮重置(kill procs+containers,确认 GPU/DMA-buf/mem 释放,fresh
  docker run,shell+MVP 连通性再开跑);小实验可不重启 container 但须释放 mem/pin/reg。
- **ionic ABI 注意**:pd-final/sglang image 内 libionic 是 54.0-149(ABI 1),26.03 内核
  驱动只认 ABI 4 → 0 RDMA devices;须注入 host 的 54.0-187 .so。(顺带 infera 侧真问题:
  该把镜像 libionic bump 到 187 对齐 26.03,vllm image 已是 187。)

---

## 9. 证据索引(可复核)

- 本地 Mooncake:`/home/yihou/dev/git.16-19/Mooncake` @ `main` `e97eee2c`。
  - `git show 45b84d36cf`(#2725,传播+跨机修复,commit msg 三条 fix + ionic 验证)
  - `git show 01d1eb2a7e:.../rdma_transport/CMakeLists.txt`(base 无 USE_HIP_DMABUF,实锤)
  - `git show 01d1eb2a7e:.../rdma_context.cpp` sed 480-530(base 每 NIC ×N 导入形态)
  - `git log -S closeDmabufExport`(#2523 = 7755a146 引入共享 fd,07-16)
  - `git merge-base --is-ancestor 7755a146 01d1eb2a7e` → false(#2523 在 base 之后)
- main 现源:`rdma_context.cpp`(exportDmabuf 386 / registerInternal 561 /
  isKernelDmabufSupported 71)、`src/CMakeLists.txt:104-127`、`multi_transport.cpp:469-520`、
  `include/multi_transport_locality.h`、`transfer_engine_impl.cpp:402`。
- 远程 PR/issue(agent 核 body):#751、#2225、#2682、#2725、#2753、#2523、#2752、#3061。
- 相关 memory:`project_mooncake_hip_ipc_env_rename.md`、`project_sglang_dmabuf_propagation.md`、
  `project_ionic_dmabuf_2x_shadow.md`、`project_sglang_image_upgrade.md`。
