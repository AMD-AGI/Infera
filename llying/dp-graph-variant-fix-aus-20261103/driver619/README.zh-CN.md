# amdgpu 驱动把 RDMA 固定的显存算两次：识别方法、原理和两种修法

适用于 AMD GPU（ROCm / KFD）上需要把大块显存注册给 RDMA 网卡的应用，例如 SGLang + Mooncake 的 PD 分离推理。
本文可以独立阅读，用来判断别的集群上的问题是不是同一个、能不能用同样的办法解决。
详细排查过程和全部实测见上一级目录的 `DRIVER619-20261113.zh-CN.md`。

本文提到的文件，除特别注明的以外，都在本目录 `driver619/` 下。

---

## 1. 问题是什么

**现象**：应用把大块显存注册给 RDMA 网卡（`ibv_reg_dmabuf_mr` 或 `ibv_reg_mr`）之后，这个进程任何新的显存申请都失败，可实际显存还空着很多。常见表现：

- PyTorch 段错误，调用栈是 `c10::hip::HIPStream::stream()`。SGLang 里在 `scheduler.py` 的 `run_event_loop`，执行
  `self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream` 时崩，紧跟在 Mooncake 注册 KV 之后。
  PyTorch 读 `.cuda_stream` 时才真正创建 HIP stream，要给硬件队列申请显存，申请失败拿到空 stream，于是段错误。
- 新一些的 ROCm（如 ROCm 10）不段错误，在同一行报 `hipErrorOutOfMemory`。
- 第一次加载 kernel 时报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`（HIP error 209）。
- 随后常见 `No HIP GPUs are available`。

**触发条件**（三个同时满足）：
1. 驱动是受影响版本（第 2.1 节的表）；
2. 注册给网卡的是 KFD 分配的显存，也就是普通 `hipMalloc` 或 PyTorch 分配的显存；
3. 注册量超过"分配完之后剩下的显存"。

只注册几百 MB 的应用不会出事，比如 RCCL 的通信缓冲；PD 推理每卡注册 100–200 GB 的 KV 缓存，一定会出事。

**根因**：驱动内核里管 GPU 计算显存的部分（KFD）在每次分配显存前做限额检查，把"已分配、又被网卡固定（pin）住"的显存算了两次：
一次在 `kfd.vram_used`（分配时记），一次在 `vram_pin_size`（固定时记）。所以它认为显存早就用光了。

---

## 2. 怎么判断是不是同一个问题

按顺序做，前一步能确认就不用做后一步。

### 2.1 查驱动版本

```bash
cat /sys/module/amdgpu/version          # DKMS 驱动会有这个文件，例如 6.19.14.31400100
uname -r                                # 不是 DKMS 时，看内核版本
```

| 驱动 | dma-buf 注册（`ibv_reg_dmabuf_mr`） | peer memory 注册（`ibv_reg_mr`，需要 amdgpu peer memory 客户端） |
|---|---|---|
| amdgpu-dkms 6.16.13（amdgpu 30.30、30.30.4，已核对） | **多扣** | 不多扣（见第 3 节，抵消项反而让限额失效） |
| amdgpu-dkms 6.18.4、6.19.0、6.19.4、6.19.14、7.1.3、7.1.9（amdgpu 31.10、31.20、31.30、31.40、31.40.1、31.50、31.60，全部核对过源码） | **多扣** | **多扣** |
| 发行版自带的 amdgpu（Linux 主线 v7.4-rc1 之前） | **多扣**（修复前的主线限额检查没有抵消项，见上游补丁删掉的那几行） | 主线没有这个客户端 |
| Linux 主线 v7.4-rc1 及以后，或打了第 4 节补丁的驱动 | 正常 | 正常 |

更早的 DKMS 版本没有核对。版本号和 repo.radeon.com 目录的对应关系：例如 `6.19.14.31400100` 对应 `amdgpu/31.40.1/`，`6.16.13` 对应 `amdgpu/30.30.x/`。

如果节点上有 DKMS 源码，也可以直接看限额检查那几行：

```bash
grep -n -A4 "uint64_t vram_available =" /usr/src/amdgpu-*/amd/amdgpu/amdgpu_amdkfd_gpuvm.c | head -8
grep -c vram_pin_size_xcp /usr/src/amdgpu-*/amd/amdgpu/amdgpu.h     # 非 0：已含上游修复
```

`amdgpu_amdkfd_reserve_mem_limit` 里第一处 `vram_available =` 后面如果只有 `- atomic64_read(&adev->vram_pin_size);`，没有
`+ atomic64_read(&adev->kfd.vram_pinned)`，就是受影响的版本（第 3 节有对照）。

**注意**：在 6.16.13 这类老驱动上，用 dma-buf 注册同样会出事。如果另一个集群是老驱动、应用走 dma-buf，就属于同一个问题，只是触发路径不同，
修法也更简单（见第 6 节）。

### 2.2 看驱动报告的可用显存

在应用注册前后各读一次 `torch.cuda.mem_get_info()`（即 `hipMemGetInfo`）：
- 受影响时用 dma-buf 注册：可用显存正好减少注册量，常见是直接变成 0。实际上这次注册并没有新分配任何显存。
- 用 peer memory 注册：可用显存会变成远大于显存总量的数，比如 22,487 GiB。

这两种异常本身就是这个问题的特征。

### 2.3 跑最小复现（几分钟，确定性的）

`bench/kfd_pin_repro.py`：单进程单 GPU，只依赖 HIP、HSA 和 libibverbs，不需要 torch、SGLang、Mooncake。
在能访问 `/dev/kfd`、`/dev/dri`、`/dev/infiniband` 的容器里运行（`--ulimit memlock=-1`，用 ionic 网卡时还要挂宿主机的 libionic）：

```bash
# 40 GiB，dma-buf 注册
python3 kfd_pin_repro.py --gpu 0 --alloc kfd --reg dmabuf  --gib 40  --nics <网卡名>
# 40 GiB，peer memory 注册（需要 amdgpu peer memory 客户端）
python3 kfd_pin_repro.py --gpu 0 --alloc kfd --reg peermem --gib 40  --nics <网卡名>
# 注册超过一半显存，复现崩溃
python3 kfd_pin_repro.py --gpu 0 --alloc kfd --reg peermem --gib 150 --nics <网卡名>
```

每一步输出一行 JSON。看 `before` 和 `registered` 两行的 `headroom_gib`，它表示驱动还肯让这个进程再分配多少显存。
11-13 在 MI355X（288 GB）上的结果：

| 命令 | 受影响（6.19.14） | 不受影响（6.16.13 + peer memory） |
|---|---|---|
| kfd / dmabuf / 40 | 287.25 → 207.25（扣 2 × 40） | 287.25 → 207.25（dma-buf 在 6.16 上也多扣） |
| kfd / peermem / 40 | 287.25 → 207.25（扣 2 × 40） | 287.25 → 247.5（扣 1 × 40） |
| kfd / peermem / 150 | 剩 0，`after` 一步建 stream 时段错误（exit 139） | 剩 137.5，`after` 一步全部 ok |

`headroom` 减少了 2 倍注册量，就是同一个问题。

### 2.4 容易混淆的情况

- **显存真的不够**：`/sys/bus/pci/devices/<bdf>/mem_info_vram_used` 是实际占用。实际占用不高、驱动却拒绝分配，才是这个问题。
- **Mooncake 的 KFD pin anchor**（`MC_HIP_KFD_PIN_FIX`，上游 PR #4657）：日志里有 `KFD pin probe: ... lowered KFD free VRAM by 64 MiB` 说明探测到了这个问题。
  但在 6.18.4 及以后的驱动上，这个补偿只修正报告值，修不了限额检查，照样崩（第 3 节）。
- **别的起栈段错误**：如果不注册显存（比如不分离 PD 的服务）也崩，就不是这个问题。

---

## 3. 原理

驱动源码在 `amd/amdgpu/amdgpu_amdkfd_gpuvm.c`（DKMS 包里路径相同；上游是 `drivers/gpu/drm/amd/amdgpu/`）。

**限额检查**，即 `amdgpu_amdkfd_reserve_mem_limit()`，每次 KFD 分配显存都要过：

```c
/* amdgpu-dkms 6.16.13 */
uint64_t vram_available = vram_size - reserved_for_pt - reserved_for_ras -
        atomic64_read(&adev->vram_pin_size) +
        atomic64_read(&adev->kfd.vram_pinned);
/* amdgpu-dkms 6.18.4 .. 7.1.9，以及主线 v7.4-rc1 之前 */
uint64_t vram_available = vram_size - reserved_for_pt - reserved_for_ras -
        atomic64_read(&adev->vram_pin_size);
if (adev->kfd.vram_used[xcp_id] + vram_needed > vram_available)
        ret = -ENOMEM;
```

**报告可用显存**，即 `amdgpu_amdkfd_get_available_memory()`，`hipMemGetInfo` 读到的就是它。所有版本都是
`总量 − vram_used − vram_pin_size + kfd.vram_pinned − 预留`。

三个计数怎么变：
- `kfd.vram_used`：KFD 分配显存时加上，BO（buffer object，驱动管理显存的单位）销毁时减去。
- `vram_pin_size`：任何 BO 第一次被固定时加上整个 BO 的大小（`amdgpu_bo_pin`）。
- `kfd.vram_pinned`：本意是"KFD 自己固定的 BO"，用来抵消重复。但只有 `amdgpu_amdkfd_gpuvm_pin_bo` 会加它，而且每调一次都加整个 BO 的大小。

两种注册方式：
- **dma-buf**：网卡驱动调 `amdgpu_dma_buf_pin` → `amdgpu_bo_pin`。`vram_pin_size` 增加，`vram_pinned` 不变，所以**任何版本都多扣**。
- **peer memory**：`ibv_reg_mr` 进入 DKMS 的 `kfd_peerdirect.c`，`amd_get_pages` 每注册一次（每张网卡 × 每个 MR 分块）都调
  `amdgpu_amdkfd_gpuvm_pin_bo`，`vram_pinned` 每次加一整个 BO。
  - 6.16 的限额检查带 `+ vram_pinned`，加多了反而让限额失效，所以不出事；
  - 6.18.4 起限额检查去掉了这一项，照样多扣。报告值则被抵消得远超显存总量。实测 `1 张网卡 × 40 块 × 40 GiB + 207 = 1,807 GiB`，与上表吻合。

所以条件是：`已分配（含 KV）+ 已固定（KV）+ 新申请 > 显存 − 预留` 时，新申请就失败。

---

## 4. 修法 A（根本修法）：驱动打 AMD 上游补丁

### 4.1 上游补丁

[torvalds/linux `9131cc85d0de` "drm/amdkfd: correct vram_available calculation"](https://github.com/torvalds/linux/commit/9131cc85d0dedd8ba4cae60dbe5985d081551f30)
- 作者 Zhen Han（AMD），2026-09-20 提交，2026-10-20 合入，cherry-pick 自 `894cbb31c50e`，标了 `Cc: stable`；
- Linux 主线**从 v7.4-rc1 开始包含**，v7.3 不含；stable 分支里截至 11-13 没有搜到单独的回移提交（GitHub 搜索结果，不保证完整）；
- AMD 在 repo.radeon.com 发布的 DKMS，截至 11-13 最新的 7.1.9（31.60）**都不含**。

补丁做了三件事：
1. 限额检查改为复用报告可用显存的同一个函数 `__get_available_memory()`，带上 `+ kfd.vram_pinned` 抵消；
2. `kfd.vram_pinned` 改在 `amdgpu_bo_pin` / `amdgpu_bo_unpin` 里、BO **第一次固定和最后一次释放**时更新（新函数
   `amdgpu_amdkfd_account_vram_pin` / `_unpin`），并且只对 KFD 分配的 BO 记（`if (!bo->kfd_bo) return;`）。
   这样 dma-buf 固定也会被抵消，peer memory 也不会再每次都多加；
3. 按显存分区（XCP）分别记账（`vram_pin_size_xcp[]`、`vram_pin_size_common`、`kfd.vram_pinned[]`）。

### 4.2 文件

- `upstream-9131cc85d0de.patch`：上游原补丁（5 个文件，+127 / −43）。
- `backport-9131cc85d0de-amdgpu-dkms-6.19.14.31400100.patch`：回移到 amdgpu-dkms 6.19.14.31400100 的源码（`-p1`，相对于 DKMS 源码根目录）。
  上游 11 处改动里 10 处直接打上，第 7 处只是缩进不同（DKMS 源码这里用空格），已手工处理。
  **编译验证还在作业里排队，结果出来后更新到这里。**
- 别的 DKMS 版本：先 `patch -p1 --dry-run` 试一下，把上游补丁路径前缀 `drivers/gpu/drm/amd/` 改成 `amd/`；失败的块按同样的意思手工改。

### 4.3 管理员怎么用（参考步骤，我们没有权限，没在集群上做过）

```bash
V=6.19.14-2377056.22.04                     # /usr/src/amdgpu-<版本> 的版本部分
NV=${V}kfdfix
sudo cp -a /usr/src/amdgpu-$V /usr/src/amdgpu-$NV
sudo sed -i "s/^PACKAGE_VERSION=.*/PACKAGE_VERSION=\"$NV\"/" /usr/src/amdgpu-$NV/dkms.conf
(cd /usr/src/amdgpu-$NV && sudo patch -p1 < backport-9131cc85d0de-amdgpu-dkms-6.19.14.31400100.patch)
sudo dkms add -m amdgpu -v $NV && sudo dkms build -m amdgpu -v $NV && sudo dkms install -m amdgpu -v $NV --force
# 然后重启节点（或在没有任何 GPU 进程时卸载并重新加载 amdgpu）
```

装好以后用第 2.3 节的 `kfd_pin_repro.py` 验证：`kfd / dmabuf / 40` 应只扣 1 × 40，`kfd / peermem / 150` 应能正常建 stream。

---

## 5. 修法 B（不改驱动）：KV 缓存改用 GEM 分配

### 5.1 解决什么、怎么解决的

驱动修好之前，让 KV 缓存只被算一次。限额检查的条件是 `kfd.vram_used + 新申请 ≤ 显存 − 预留 − vram_pin_size`。
KV 必须固定给网卡，`vram_pin_size` 这一项减不下来；能做的只有让 KV 不进 `kfd.vram_used`，也就是不经过 KFD 分配。

做法：
1. 通过 DRM render node（`/dev/dri/renderD*`）的 `DRM_IOCTL_AMDGPU_GEM_CREATE` 在显存里分配 GEM BO，导出 dma-buf fd；
2. 用 `hipImportExternalMemory`（`hipExternalMemoryHandleTypeOpaqueFd`）导入进程。KFD 导入 dma-buf 时（`import_obj_create`）不做限额记账；
3. 分配后立即用一页大小的 `ibv_reg_dmabuf_mr` 把整个 BO 固定住，让驱动报告的可用显存和限额检查从这一刻起就把它算进去，一次；
4. 网卡注册（Mooncake）只走 dma-buf。

按源码，这样做在老驱动、受影响驱动、修好的驱动上都只算一次：上游补丁的 `account_vram_pin` 对非 KFD 分配的 BO 直接返回。实测（MI355X）：

- 单进程：GEM 分配并固定 40 / 150 / 200 GiB 后，剩余可分配分别为 247.25 / 137.25 / 87.25 GiB（只扣一次），6.19.14 和 6.16.13 一样；
  之后建 stream、再分配都正常。
- torch + Mooncake：6.19.14 上注册 150、198 GiB 的 GEM KV 后一切正常，RDMA 写入跨 MR 边界的数据校验通过。
  同一对卡、同一张网卡上，RDMA 写入速度与普通显存完全相同；GPU 读写带宽也相同。
- 端到端：GLM-5.2 1P1D PP8 生产配置，prefill 在 6.19.14 节点上、decode 在 6.16.13 节点上，以及两者对调，都通过了门禁
  （冒烟 6/6，200 题探针 194、197，乱码 0，MTP 接受长度 3.69、3.51，参考值为 197 和 3.60）。计分结果待补。

### 5.2 组成

| 文件 | 作用 |
|---|---|
| `gemalloc/hip_gem_alloc.c` | 编译成 `libhip_gem_alloc.so`。①给 `torch.cuda.memory.CUDAPluggableAllocator` 用的 `hip_gem_alloc` / `hip_gem_free`，按上面四步分配和释放；②作为 `LD_PRELOAD` 垫片，只对自己分配的地址接管 `hipMemGetAddressRange`（HIP 不记录导入显存的范围）、`hsa_amd_portable_export_dmabuf`（ROCr 碰到导入显存会段错误）、`hipIpcGetMemHandle`（导入显存没有 IPC 句柄，Mooncake 的 HIP 传输会因此让整个注册失败；这里返回一个打不开的句柄），并**拒绝**这些地址上的 `ibv_reg_mr` / `_iova` / `_iova2`；其他地址原样转给原函数 |
| `sglang/mooncake_utils-amd-gem.diff`（改后的整文件 `sglang/mooncake_utils.py`） | SGLang `srt/disaggregation/mooncake/utils.py`：`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM` 时，KV pool 用这个分配器建 `torch.cuda.MemPool`。`AMD_GEM` 故意不加进 `SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES`，所以 Mooncake 传输代码看到的仍是"未开启自定义池"，传输路径与默认相同（开了自定义池，传输会改成按层提交，那是为 NVLink 设计的）。基于镜像 `infera-sglang:v0522-fdbafb60` 里的 SGLang |
| `config/gemkv.env` | 我们 kit 的配置示例：给 prefill、decode 两个角色加环境变量和挂载 |
| `bench/kfd_pin_repro.py`、`bench/mc_gem_test.py` | 验证用：单进程记账测试；torch + Mooncake 的注册、建 stream、RDMA 回环写入与校验、带宽对比 |

### 5.3 怎么用

1. **编译**（在应用镜像里，需要 ROCm 头文件和 libibverbs-dev）：
   ```bash
   gcc -O2 -Wall -shared -fPIC -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include hip_gem_alloc.c \
       -o libhip_gem_alloc.so -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -libverbs -lpthread -ldl
   ```
2. **SGLang**：把 `mooncake_utils-amd-gem.diff` 打到 SGLang 的 `python/sglang/srt/disaggregation/mooncake/utils.py` 上，或者把改好的文件挂载覆盖。
   其他 SGLang 版本只要还有 `init_mooncake_custom_mem_pool`，手工加同样的分支即可。
3. **prefill 和 decode 的容器环境变量都要设**：
   ```bash
   LD_PRELOAD=/path/libhip_gem_alloc.so
   SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM
   SGLANG_AMD_GEM_ALLOCATOR=/path/libhip_gem_alloc.so
   MOONCAKE_DISABLE_HIP_DMABUF=0     # 必须走 dma-buf
   MC_HIP_KFD_PIN_FIX=0              # Mooncake 的 pin anchor 是一次 peer memory 注册，必须关
   MC_DISABLE_HIP=1                  # GEM 显存没有 HIP IPC 句柄，不能走 Mooncake 的节点内 HIP 传输
   # 可选：HIP_GEM_PIN_NIC=<网卡名>  一页固定用哪张网卡，默认第一张能用的
   ```
   容器要能访问 `/dev/dri`（render node）和 `/dev/infiniband`，`memlock` 不限；PD 推理的容器本来就满足。
4. **确认生效**：SGLang 日志里有 `KV memory pool on device cuda: amdgpu GEM BOs`。`[hip_gem_alloc] GEM memory ... has no HIP IPC handle` 是预期的提示。
   不能出现 `[hip_gem_alloc] fatal`：分配器发现环境变量不对，或者没有预加载，会直接中止进程，这是有意的。
   注册之后的 `mem_get_info` 应该是正常值（不是 0，也不会大于显存总量）。

### 5.4 限制和风险

- **绝对不能让这些显存走 peer memory 注册**：`kfd_peerdirect.c` 的 `amd_get_pages` 无条件读 `bo->kfd_bo->domain`，GEM BO 这里是空指针，会触发内核空指针错误，
  在共享节点上影响别人。垫片拒绝了这些地址上的 `ibv_reg_mr*`，分配器也会检查环境变量；但用别的传输库时（UCX、NIXL 等），要先确认它对设备显存只走 dma-buf。
- 需要内核和网卡驱动支持 dma-buf MR（`CONFIG_PCI_P2PDMA`、`CONFIG_DMABUF_MOVE_NOTIFY`，网卡实现 `ibv_reg_dmabuf_mr`）。只在 ionic（AMD Pensando）网卡上测过。
- 同节点的 KV 传输只能走 RDMA 回环，不能走 Mooncake 的 HIP IPC。
- 只有 SGLang KV pool 走 GEM；其余注册的小 buffer（元数据等）仍是 KFD 分配，多扣的量可以忽略。
- 这是验证用的形态。`LD_PRELOAD` 属于运行时替换函数，InferenceX 不允许。正式版需要 Mooncake 原生支持 GEM 显存
  （登记表、按 BO 的 fd 注册、跳过 IPC 句柄、禁止 peer memory），再加一个 SGLang 的显存池类型。
- 已测环境：MI355X 288 GB；ionic 网卡；镜像 `infera-sglang:v0522-fdbafb60`（ROCm 7.2.0，torch 2.9.1+rocm7.2.0，镜像内的 Mooncake 不带 pin anchor）；
  驱动 6.19.14.31400100 和 6.16.13。

---

## 6. 用到另一个集群：怎么选

| 另一个集群的情况 | 建议 |
|---|---|
| 能联系管理员 | 修法 A。驱动修好后什么都不用改，所有应用都受益 |
| 老驱动（6.16 这一代）、应用走 dma-buf 注册 | 最简单：改用 peer memory 注册。需要节点上有 amdgpu peer memory 客户端，Mooncake 用 `MOONCAKE_DISABLE_HIP_DMABUF=1`。这一代驱动的 peer memory 路径不多扣 |
| 受影响驱动（6.18.4 及以后），SGLang + Mooncake，暂时等不到驱动修复 | 修法 B，按第 5.3 节配置 |
| 受影响驱动，别的推理框架或传输库 | 修法 B 的分配器核心（GEM 分配、导入、一页固定）可以复用，但要按那个传输库实际调用的 HIP / HSA / verbs 接口调整垫片，并先确认它不会对设备显存回退到 `ibv_reg_mr` |
| 不确定是不是同一个问题 | 先跑第 2.3 节的 `kfd_pin_repro.py`，几分钟就有结论 |
