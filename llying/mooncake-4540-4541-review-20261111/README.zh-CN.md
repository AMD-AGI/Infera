# Mooncake #4540 / #4541 review 与推进分析（2026-11-11）

- [kvcache-ai/Mooncake#4540](https://github.com/kvcache-ai/Mooncake/pull/4540)：`[TransferEngine] Add opt-in destination-local GPU write rails`（+263/−6，9 个文件）
- [kvcache-ai/Mooncake#4541](https://github.com/kvcache-ai/Mooncake/pull/4541)：`[TransferEngine] Make the RDMA ACK timeout configurable`（+55/−2，5 个文件）
- 两个都是 llying-001 在 10-09 开的 draft，基于 main `d645d6cd`。至今**没有任何 review 或评论**。
- 对照的最新 main：`d6a738dc`（11-11），比两个 PR 的基线多 **370 个提交**。
- 本报告只做分析，**没有改代码，也没有在 PR 上留言**。

---

## 结论

| | #4541 `MC_IB_TIMEOUT` | #4540 `MC_ENABLE_DEST_LOCAL_RAIL` |
|---|---|---|
| 和最新 main 的文本冲突 | 没有 | 1 处：`tests/CMakeLists.txt` 末尾双方都追加了内容，保留两边即可 |
| main 更新后功能是否还成立 | 成立 | 产品代码成立；**测试在 HIP 编译下会失败**（PR 一开始就有的问题） |
| 需要改的 | 合并 main；决定通知 QP 要不要一并纳入；整理描述 | 合并 main 并解决冲突；修测试；补新的硬件对照；整理描述 |
| CI | 10-09 跑过完整 CI，3 项失败，都与本 PR 无关 | **当前 head 上完整 CI 从来没跑过**（只有 `triage`） |
| 我们的部署还需要它吗 | crsuse2 需要（1.07 s 是实测出来的合适值）；作为通用开关对上游也有价值 | 10-29 已核实 AUS 和 SemiAnalysis CI 集群**不需要**；只有 crsuse2 这类环境需要 |
| 建议 | **改完就转 ready**：改动小、默认值不变、TENT 已有同类参数，合入可能性高 | **先别转 ready**：先决定有没有目标部署还需要它；需要的话改完并在 crsuse2 上做新的 A/B 再转；不需要就保持 draft 并留一条说明，或关闭 |

---

## 一、两个 PR 是什么

### #4541：让 RDMA 的 ACK 超时可配置

- **背景**：RDMA RC 连接发出一个包后，在"ACK 超时"内收不到对端确认就重传；重传次数用完，这个 QP 就报错。超时时长 = 4.096 µs × 2^n，n 由 QP 进入 RTS 状态时的 `ibv_qp_attr.timeout` 决定。classic 传输里 n 写死为 14（约 67 ms）。
- **改动**：新增环境变量 `MC_IB_TIMEOUT`（整数 1–31），默认仍是 14。用已有的严格整数解析函数，非法值打 warning 并保持默认；写进配置 dump；在 `RdmaEndPoint::doSetupConnection()` 里 QP 进入 RTS 时使用（重连也走这条路径）。
- **为什么需要**：crsuse2 上 decode 端 ionic 网卡在反压下会在网卡内部丢帧，67 ms 太短，会把短暂停顿误判成断链。实测 n=18（约 1.07 s）最合适；n=24（约 69 s）恢复太慢会造成传输失败（Infera `llying/cross-rank-rca-20260921/REPORT.zh-CN.md`）。

### #4540：按目的 GPU 选"本地写入"的网卡

- **背景**：PD 分离时，prefill 的 GPU i 要把 KV 写到 decode 的 GPU j。默认策略按**发送端** GPU i 选本端网卡；开启已有的 `MC_ENABLE_DEST_DEVICE_AFFINITY` 后，远端会选和本端**同名**的网卡。当所有 rank 共用整组网卡、PCIe 拓扑又被隐藏时，结果可能是接收端网卡跨 NUMA 写 GPU j，或者多张网卡同时写同一张 GPU。在 crsuse2 上，这两种情况都会触发网卡内部丢帧。
- **改动**（默认关闭）：
  1. `Topology::resolvePinnedDevices()`：给每块 GPU 固定一个首选 HCA。多块 GPU 共用同一组首选 HCA 时，按 GPU 序号轮流分配；结果是确定的，两端算出来一样。
  2. `RdmaTransport::destinationLocalHca()`：对 WRITE 请求，查出目的 buffer 所在的 GPU 和它被固定的 HCA，把这个 HCA 的**名字**当作本端网卡的初始提示。
  3. `submitTransferTask()`：请求级（以及请求级失败时的 slice 级）初始选卡，在没有显式 `nic_hint` 时使用上面的提示；重试仍按原策略，可以换网卡。
  4. 新开关 `MC_ENABLE_DEST_LOCAL_RAIL=1`，必须同时开 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`，否则忽略并打 warning。
- **效果**：写往 GPU j 的流量两端都走 `ionic_j`。crsuse2 的 RCA 里，这个策略（配合 #4541 的 1.07 s）让 C80 的 51 次传输失败降到 0，吞吐 +5.6%，64 对 GPU 的验收矩阵全部通过；而默认策略 60 秒内出现 22,813 次 ACK timeout。

---

## 二、和最新 main 的冲突

用 `git merge-tree` 检查（基线 `d645d6cd`，main `d6a738dc`）：

| PR | 双方都改过的文件 | 冲突 |
|---|---|---|
| #4541 | `config.h`、`config.cpp`、`rdma_endpoint.cpp`、`tests/config_test.cpp`、文档 `index.md` | **无** |
| #4540 | `config.h`、`config.cpp`、`topology.h`、`topology.cpp`、`rdma_transport.h`、`rdma_transport.cpp`、`tests/CMakeLists.txt`、文档 `index.md` | **1 处**：`tests/CMakeLists.txt` 文件末尾（PR 追加新测试目标，main 也在末尾加了内容），两边都保留即可 |

---

## 三、main 更新后功能是否还成立（逐项核对）

### #4541

| 检查项 | 结论 |
|---|---|
| 数据 QP 进入 RTS 的位置还在不在 | 在：`doSetupConnection()` 里仍是 `attr.timeout = kTimeout;`（main 第 1690 行），PR 改的就是这里 |
| 删掉 `kTimeout` 常量会不会让别处编译失败 | 不会：main 上 `kTimeout` 只有这一处使用 |
| 严格解析函数是否还在 | 在：main 的 `MC_EFA_CQ_THREADS` 就是用同样的 `TryParseInteger<int>(value, {.trim_ascii_whitespace = true, .allow_leading_plus = true})` |
| main 上是否已有同类功能 | 没有：classic 传输里仍然没有 `MC_IB_TIMEOUT` 或其他 ACK 超时开关 |
| **通知 QP** | **不覆盖**：`RdmaEndPoint::connectNotification()`（通知消息专用 QP，在 `doSetupConnection()` 里一起建立）写死 `timeout = 14`、`retry_cnt = 3`。这在 PR 基线时就已存在，不是 main 新引入的 |
| TENT | TENT 已有同类参数 `endpoint.send_timeout`（默认同样是 14）。#4541 等于给 classic 补齐，可以在描述里点明这种对应关系 |

### #4540

| 检查项 | 结论 |
|---|---|
| `submitTransferTask()` 里两处设置提示的位置 | 都还在（main 第 1373 行、第 1439 行附近），结构没变 |
| 远端按"同名网卡"选卡（`MC_ENABLE_DEST_DEVICE_AFFINITY`） | `worker_pool.cpp` 的 `selectPeerDevice()` 在基线时就有，逻辑未变 |
| 依赖的函数和成员 | `selectDevice` 两个重载、`resolveBufferLocation`、`preferred_hca`、`resolved_matrix_`、`getHcaIndex`、`hca_list_`、`disableDevice` 都在，签名未变 |
| `resolve()` 的变化 | main 改成先把存储类型排序再编号，让两端编号一致。PR 最后只用 HCA 的名字作为提示，而且自己的分组用有序的 `std::map`，结果确定，不受影响 |
| 插入 `resolvePinnedDevices()` 的位置 | `resolve()` 末尾、`disableDevice()` 最后的 `return 0` 之前，main 上结构没变，仍然合适 |
| `GPU_PREFIX` 现在按厂商定义 | HIP 是 `hip:`，CUDA 一类是 `cuda:`，新增 Enflame 是 `gcu:`。产品代码用的是这个宏，拓扑发现和内存位置识别也都用它（main 第 422 行还有注释："Hardcoding "cuda:" silently disables HCA peer affinity under USE_HIP"），所以产品代码在 AMD 上没问题 |
| **测试里写死 `"cuda:"`** | **有问题**：`rdma_destination_local_test.cpp` 里有 28 处写死 `"cuda:"`，在 HIP（以及 Enflame）编译下这些"GPU"不会被识别，测试会失败。PR 基线时 HIP 前缀就已经是 `hip:`，所以问题从一开始就有 |
| 本地选卡缓存 `BatchRdmaDeviceCache` 会不会跳过提示 | 不会：它只在本端 segment **只有一张 HCA** 时才缓存，多网卡时每次都重新选。但它的注释写着"调用方要保持 affinity hint 不变"，而 PR 让提示随目的 GPU 变化，以后如果缓存放宽到多网卡，就会悄悄失效（见问题 4540-P2b） |
| 拥塞控制和网卡暂停（#4197、#5064、`rdma_rail_pause_seconds`） | 不冲突：worker 发送前仍会检查选中的网卡是否被暂停，被暂停就换设备（`worker_pool.cpp` 第 689 行），提示不会绕过它 |
| "带名字的 hint 先找本地位置、再找全局网卡"的语义 | 基线时就是这样，没有变化（见问题 4540-P3b 的文档建议） |
| main 上是否已有同类功能 | 没有。CI 集群日志里的 "Auto-discovered Mooncake matched rails" 不在 Mooncake 或 SGLang 代码里，是部署层（启动脚本）按 GPU 分配同号网卡的做法 |

---

## 四、问题清单（按优先级）

### #4541

| 编号 | 优先级 | 问题 | 原理 | 怎么改 |
|---|---|---|---|---|
| 4541-P1a | P1 | 基线落后 370 个提交，CI 是 10-09 的结果 | 维护者会以新 main 上的 CI 为准 | 合并 main（无冲突），推送触发 CI |
| 4541-P1b | P1 | 10-09 的 CI 有 3 项失败 | `build-wheel-cu13`：runner 收到关机信号，退出码 137；`test-wheel-ubuntu`：PG 测试 "queued sends corrupted"（PR 不改默认值，几乎可以肯定无关）；`CI Gate`：汇总项 | 合并后重跑；如果 PG 测试还失败，到 main 上找同样的失败作为证据 |
| 4541-P2 | P2 | 通知 QP 不受 `MC_IB_TIMEOUT` 控制 | `connectNotification()` 写死 `timeout = 14`、`retry_cnt = 3`。通知 QP 和数据 QP 走同一张网卡，数据 QP 需要更长超时的场景里，重试只有 3 次的通知 QP 反而更容易失败 | 二选一：通知 QP 也改用 `cfg.ib_timeout`（一行改动，推荐，语义统一），或在文档写明"只影响数据 QP" |
| 4541-P3a | P3 | 描述可以点明 TENT 的对应参数 | TENT 的 `endpoint.send_timeout` 就是同一个东西 | 描述加一句"classic 与 TENT 的 `endpoint.send_timeout` 对应" |
| 4541-P3b | P3 | 描述和 checklist 还是 draft 口吻 | "human submitter review pending; Codex reviewed"、self-review 未勾选，维护者会犹豫 | 做完逐行自审后勾选，改掉"待人工 review"的措辞，保留 AI 使用披露 |
| 4541-P3c | P3 | 没有在硬件上确认参数确实生效 | 单测只覆盖解析 | 在任意 RDMA 机器（甚至 soft-RoCE）上设 `MC_IB_TIMEOUT=18` 跑一次传输：配置 dump 里显示 `ib_timeout = 18`，并用 `ibv_query_qp` 或驱动计数确认 QP 的 timeout 属性 |

### #4540

| 编号 | 优先级 | 问题 | 原理 | 怎么改 |
|---|---|---|---|---|
| 4540-P1a | P1 | 和 main 有冲突，基线落后 370 个提交 | `tests/CMakeLists.txt` 末尾冲突 | 合并 main，冲突处两边都保留 |
| 4540-P1b | P1 | **测试在 HIP/Enflame 编译下失败** | 28 处写死 `"cuda:"`，而产品代码用 `GPU_PREFIX`（HIP 下是 `hip:`）。这个功能本来就是为 MI355X 做的，测试却只在 CUDA 编译下成立 | 测试里的位置统一用 `GPU_PREFIX + "0"` 这类写法构造；拓扑 JSON 字符串也按 `GPU_PREFIX` 拼 |
| 4540-P1c | P1 | 当前 head 上完整 CI 从来没跑过 | 只跑了 `triage`；fork 的 PR 需要维护者触发 | 合并 main 推送后，请维护者触发 CI（`run-ci` 标签已有），重点看 ROCm 构建 |
| 4540-P2a | P2 | **缺少基于 main 的新证据，而且我们自己的目标集群已不需要** | 描述里也承认历史结果"不是对这个实现的新验证，也不能证明单靠固定网卡就有吞吐收益"。10-29 核实：AUS 和 SemiAnalysis 集群靠部署层"每个 rank 只给同号网卡"加上游 `MC_ENABLE_DEST_DEVICE_AFFINITY` 就不出问题；只有 crsuse2 这类"整组网卡共享 + 反压丢帧"的环境需要 | 先决定是否推进（见第五节）。推进的话：在 crsuse2 上用基于 main 的构建做开关 A/B（传输失败数、ACK timeout 数、吞吐），并把"哪类拓扑需要它"写进描述 |
| 4540-P2b | P2 | 本地选卡缓存的前提被打破 | `BatchRdmaDeviceCache` 注释要求 hint 固定；PR 让 hint 随目的 GPU 变化。现在因为只缓存单网卡场景所以无害，但缓存一旦放宽就会悄悄失效 | 有目的端提示时绕过缓存（或把提示纳入缓存键），并加一行注释说明 |
| 4540-P3a | P3 | 开启后每个请求多一次远端 buffer 查找和两次字符串构造 | `destinationLocalHca()` 每个请求调用一次 `selectDevice` 和两次 `resolveBufferLocation`（构造 `std::string`）。main 这期间做了不少热路径优化（#4793、#4287、#5039），维护者可能会问 | 默认关闭时开销为 0，可以先在描述里说明；进一步可以按（目的 segment、buffer）缓存固定的 HCA 名字 |
| 4540-P3b | P3 | 文档没写清楚取舍 | 带名字的 hint 会先在源 GPU 的本地网卡里找，找不到再到全局网卡里找。所以开启后，发送端可能用一张不在源 GPU 本地的网卡（发送端跨 NUMA 读），换取接收端本地写；前提是两端网卡同名代表同一条 rail | 文档写明这个取舍，以及"两端网卡命名要一致"的前提 |
| 4540-P3c | P3 | 描述和 checklist 还是 draft 口吻 | 同 4541-P3b | 同上 |

---

## 五、是否应该从 draft 打开，怎么推进

### #4541：建议改完就转 ready

理由：改动小（+55/−2），默认值不变，有明确的使用场景和实测依据（crsuse2 的 n=18），TENT 已有同类参数，和 main 无冲突。

步骤：
1. 合并 main（无冲突）。
2. 通知 QP 也改用 `cfg.ib_timeout`（推荐），并在文档里写明"数据 QP 和通知 QP 都受影响"；同步补一个单测或断言。
3. 硬件上做一次最小确认（4541-P3c）。
4. 逐行自审后勾选 self-review，改掉描述里"待人工 review / draft"的措辞，点明与 TENT `endpoint.send_timeout` 的对应关系。
5. 推送，确认 CI 通过（PG 测试如仍失败，附上 main 上同样失败的链接）。
6. 转 ready，并在 PR 里 at Transfer Engine 的维护者请求 review（公开操作，执行前先确认）。

### #4540：建议先别转 ready，先做一个决定

决定的问题：**我们（或上游用户）还有没有部署需要它？**
- 我们自己的目标集群（AUS、SemiAnalysis CI 集群）已经核实不需要。
- crsuse2 需要，但前提是那边的部署还在用"每个 rank 共用整组网卡"的方式。

**如果需要（继续推进）：**
1. 合并 main，解决 `tests/CMakeLists.txt` 冲突。
2. 修测试的 `GPU_PREFIX` 问题（4540-P1b）；处理缓存前提（4540-P2b）。
3. 在 crsuse2 上用基于 main 的构建做开关 A/B：同一节点对、C80、开启与关闭各跑一次，记录传输失败、ACK timeout 和吞吐（RCA 报告的实验合同可以直接复用）。
4. 描述改写：写清楚"适用于哪类拓扑"（共享整组网卡、PCIe 拓扑被隐藏），给出新数据，说明与部署层"每 rank 同号网卡"做法的关系；补 4540-P3b 的文档。
5. 推送并请维护者触发 CI，通过后转 ready。

**如果不需要：** 保持 draft，并留一条简短说明（适用场景、历史数据、为什么暂缓），避免维护者花时间 review 一个我们自己都不急着要的功能；或者直接关闭，以后有需要再基于新 main 重开。

---

## 附：证据来源

- 代码核对：本地 Mooncake clone（`~/code/bench_agentx/Mooncake`，浅克隆），`git merge-tree` 和 `git show origin/main:<文件>`。
  - #4541 相关：`rdma_endpoint.cpp` 第 499 行起 `connectNotification()`、第 1690 行 `doSetupConnection()`；TENT `tent/include/tent/transport/rdma/params.h` 第 86 行 `send_timeout = 14`。
  - #4540 相关：`rdma_transport.cpp` 第 1339 行 `BatchRdmaDeviceCache`、第 1373 / 1439 行两处提示、`pickTopologyDevice()`；`include/transport/rdma_transport/rdma_batch_cache.h`；`worker_pool.cpp` 的 `selectPeerDevice()` 和第 689 行网卡暂停处理；`topology.cpp` 的 `resolve()` / `disableDevice()`。
- CI：#4541 run `37919583429`（`build-wheel-cu13` 退出码 137；`test-wheel-ubuntu` 的 PG 测试失败）；#4540 只有 `triage`。
- 历史实验：Infera 仓库 `llying/cross-rank-rca-20260921/REPORT.zh-CN.md`（根因、51→0 失败、+5.6%、ACK 超时取值）；`llying/inferencex-publish-plan-20261027/README.zh-CN.md` 第 5.10 节（10-29 核实 CI 集群和 AUS 不需要这两个补丁）。
