# Cross-Rank Mooncake 传输根因与根治报告

状态（2026-10-08）：根因已闭合（3.6 节），修复已在 137→136 上验证（3.7、3.8 节），
22 个硬门禁过 21 个；138→136、136→138 复核结果相同。

- 根因：Decode 端网卡的 DMA 写被反压时会在网卡内部丢帧，PFC 拦不住。Mooncake 的
  HCA 选择让 cross-rank 传输持续制造两种反压：接收网卡跨 socket 写显存，以及多张
  网卡同时写同一张 GPU。
- 修复：Mooncake 用户态 3 个补丁，让写往 Decode GPU j 的请求两端都走 `ionic_j`，
  RC ACK 超时设为 1.07 s，不需要 root。最终镜像 v3：`88286215b79d`。
- 结果：C80/1200 s（`PD_DP_RANK_AFFINITY=0`）下，基底镜像有 51 个 transfer
  failure；v3 下 transfer failure、CQE12、retry exhausted、session failed、客户端
  错误均为 0，吞吐 +5.6%。64 pair 验收矩阵全部通过，计数器全为 0。
- 未过的门禁：C80 的 1200 秒采样中仍有 ACK timeout，v3 共 20 次，集中在 4 个时间窗。
  这些超时没有丢包，也没有报错；把超时拉长到 4.3 s（v4）后次数几乎不变（18 次），
  原因待查（3.7 节末）。
- 所有 `mc-rca-*` / `glm52-pd-crossrank-rca-*` 容器已清理。
- 2026-10-08 在 138→136 上复核：v3 同样 0 失败；把超时拉长到 69 s 后出现 transfer
  failure，说明每次 ACK timeout 都是真实的 QP 停顿，1.07 s 是正确选择。停顿发生在接收
  网卡内部；交换角色（136→138）后 138 作为接收端出现同样的现象，不是 136 单机的问题，
  修复同样 0 失败（3.7 节末）。

## 1. 固定实验合同

2026-09-28 起节点对改为 Prefill（矩阵 source）`crsuse2-m2m-137`、
Decode（矩阵 destination）`crsuse2-m2m-136`，见
`operations/resume-20260928T1225Z.md`。137 与 136 拓扑相同、8 条 rail 健康。
137 从未参与故障复现，因此修复前必须先在 137→136 重建 C80 故障基线，
before/after 只在同一节点对上比较。以下为 138→136 阶段的原始合同，
除节点外其余条目不变。

- Prefill：`crsuse2-m2m-138`（现为 `crsuse2-m2m-137`）
- Decode：`crsuse2-m2m-136`
- 基底镜像：
  `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`
- P8D8 / C80 / warmup per lane 1 / 1200 秒
- HiCache 两端关闭
- shared 8-HCA list
- `PD_DP_RANK_AFFINITY=0`

最终修复镜像允许产生新 digest，但必须以上述镜像为唯一基底，并保持其余实验合同
不变。

## 2. 已冻结的软件事实

- SGLang：`0.5.19.dev20260917+ga9fb1c3238`
- Torch：`2.9.1+rocm7.2.0.git7e1940d4`
- AITER：`0.1.21.dev48+g4ad998328.d20260917`
- Mooncake `engine.so` SHA256：
  `f22b1364501c748d8a6946f0bf3731057d585229fc20efdcad453205747cd56d`
- Mooncake build-id：`a15395b0cb3f1a1ada26f5fb397280e2264ff6ec`
- `engine.so` 同时链接 `ibv_reg_dmabuf_mr` 和
  `hsa_amd_portable_export_dmabuf`，包含 upstream cross-host locality routing。
- host `libionic.so`：
  `f67c9b4897d5f17e857701ad9c1990765cbe44d67830a47afd44224572072cc3`

原始 provenance 位于 `provenance/`。节点替换前的 137/136 清场现场保留在
`provenance/pre-clear/` 和 `operations/clear-20260921T1235Z/`，不作为最终
138/136 运行合同。

## 3. 实验结论

### 3.1 C80 复现

运行：
`runs/repro-01-138-136-20260921T1254Z/`。

- 运行窗口：2026-09-21 12:54:20 至 13:52:13 UTC；
- `live-config.json` 证明 Router affinity 关闭、两端 HiCache 关闭、两端均使用
  `ionic_0,...,ionic_7`；
- benchmark 正常退出，实际 profiling 统计窗口 1229.9466 秒；
- profiling 完成 2608 个请求，其中 2552 个有效、56 个
  `InvalidInferenceResultError`，失败率 2.147%；
- 56 个客户端错误都在 5 秒内一一对应到 Decode transfer failure，最大时间差
  2351.393 ms；不存在未关联的客户端错误；
- 另外 4 个 Decode-only failure 发生于 13:49:28–13:49:31 UTC，即 profiling
  超时后的 16 个 in-flight request 取消阶段，不计入上述 56 个传输失败。

Mooncake 与 HCA 直接证据：

- `CQE error 12`：36；
- `transport retry counter exceeded`：36；
- Prefill transfer failure：56；
- Decode transfer failure：60（56 个业务失败加 4 个 teardown cancellation）；
- Mooncake session failed：13；
- 18 个首发 send failure，随后放大为 38 个 failed-session cascade；
- 36 个 CQE 覆盖全部 8 个本地 HCA 和全部 8 个 Decode rank；
- 36 个 CQE 的 `local_nic` 与 `peer_nic` 名称全部相同；
- Prefill 节点 138 的 `req_tx_retry_excd_err` 增量为 172，8 个 HCA 均有增量；
- Prefill 节点 138 的 `tx_rdma_ack_timeout` 增量为 1547，8 个 HCA 均有增量；
- 上述两个 hard-fault counter 在 Decode 节点 136 无增量。

Rank/NUMA 关联是本次复现最强的定位证据：

- Router 共记录 2788 个 P/D 选择：1350 个 cross-NUMA、1438 个 same-NUMA；
- 可同时从 Prefill/Decode bootstrap room 闭合的 56 个业务失败全部为 cross-rank
  且 cross-NUMA；
- cross-NUMA 失败为 56/1350，即 4.148%；
- same-NUMA 失败为 0/1438；
- 失败覆盖 15 个 rank pair：P0→D7(2)、P1→D4(3)、P1→D6(7)、
  P1→D7(6)、P2→D6(2)、P3→D4(1)、P3→D5(5)、P3→D6(3)、
  P4→D1(3)、P5→D2(9)、P5→D3(3)、P6→D1(4)、P6→D2(2)、
  P7→D1(3)、P7→D3(3)。

权威机器可读结果位于：

- `analysis/summary.json`；
- `analysis/transfer-failures.json`；
- `analysis/client-server-matches.json`；
- `analysis/cqe-events.json`；
- `analysis/failure-rank-pairs.csv`；
- `analysis/hca-counter-deltas.csv`；
- `analysis/hca-timeseries-fault-events.csv`。

### 3.2 64 对单变量矩阵

有效结果：
`matrix/full-batch-20260922T0220Z/`。

- 几何：1 GiB buffer、32×32 MiB chunk、每 pair 持续 3 秒；
- source-local：64/64 PASS；
- destination-local：64/64 PASS；
- auto（shared 8-HCA，destination affinity 开启）：64/64 PASS；
- 共 192/192 byte-verify PASS；
- 全部 pair 的 CQE12、retry exhausted、
  `req_tx_retry_excd_err Δ`、`tx_rdma_ack_timeout Δ` 均为 0；
- `matrix-certification.json` 对三种 policy 均给出 PASS；
- 中途一次 P1→D6 SSH DNS 解析失败不含任何 RDMA fault，已原位重跑并
  PASS，最终 summary 不含该基础设施伪失败。

顺序吞吐为：

- source-local：same-NUMA 平均 39.17 GB/s，cross-NUMA 平均
  36.79 GB/s；
- destination-local：same-NUMA 平均 34.50 GB/s，cross-NUMA 平均
  28.45 GB/s；
- auto：same-NUMA 平均 47.22 GB/s，cross-NUMA 平均 45.03 GB/s。

因此，固定 GPU pair 或固定单 rail 的隔离传输不是确定性触发条件；生产故障还需要
多 rank/QP/rail 并发或上层 session 生命周期条件。

### 3.3 8-pair 并发矩阵

有效 harness 使用与生产一致的形状：每节点一个容器，容器内 8 个 GPU worker；
每个 wave 同时运行 8 个互不重叠的 source/destination GPU pair。8 个
permutation offset 覆盖全部 64 pair，三种 policy 共 24 waves。

工具校正过程：

- 最初的“一 pair 一容器”并发原型触发
  `hipIpcGetMemHandle invalid argument` / `register_failed`；
- 这些失败的 HCA hard counter 为 0，发生在 RDMA 提交前，是多容器 HIP IPC
  测试伪故障，不纳入根因证据；
- 改为每节点一个共享容器后，30 秒 8-pair smoke 为 8/8 PASS，
  吞吐 38.23–41.12 GB/s，所有 RDMA 门禁为 0。

正式部分运行：
`matrix/concurrent-soak-shared-20260922T0625Z/`。

- 几何：每 pair 1 GiB buffer、32×32 MiB chunk、持续 90 秒；
- 因共享节点被更高优先级实验回收，运行在 10/24 waves 后主动停止；
- 已完成 source-local offsets 0–7：8/8 waves、64/64 pair PASS；
- 已完成 destination-local offsets 0–1：2/2 waves、16/16 pair PASS；
- 累计 80/80 pair observation PASS；
- 10 个 wave 的 CQE12、retry exhausted 和两个 HCA hard counter
  增量全部为 0；
- auto 尚未开始，destination-local 尚余 offsets 2–7。

并发吞吐出现新的性能信号：source-local 下 same-NUMA 平均
36.97 GB/s，而 cross-NUMA 平均 23.81 GB/s、最低 12.00 GB/s。
source-local 在发送端使用 source GPU 本地 HCA，并在接收端使用同名 HCA；
因此 cross-NUMA pair 的接收 HCA 对 destination GPU 是非本地的。这个下降支持
“Decode 端跨 NUMA DMA/互连在并发压力下退化”的方向，但尚未出现 transport
fault，也缺少完整 destination-local/auto 对照，不能据此宣布根因。

以下目录只保留作工具校正 provenance，不得与有效 RDMA 结果混合：

- `matrix/concurrent-soak-20260922T0551Z/`；
- `matrix/concurrent-gpu-isolated-smoke-20260922T0620Z/`；
- `matrix/concurrent-ipc-host-smoke-20260922T0621Z/`。

### 3.4 接收端跨 NUMA 写入机理（137→136，host memory）

结果：`matrix/hostmem-numa-ab-137-136-20260928T1242Z/`，脚本
`scripts/run_hostmem_numa_ab.py`。不用 GPU；每条 rail `ionic_k` 一条
`ib_write_bw` 流写到 136 同名 HCA，8 QP/HCA、1 MiB 消息、20 秒，CPU 与
发送端内存始终与 HCA 同 NUMA，唯一变量是接收端（或发送端）buffer 的 NUMA。
计数器口径见 `scripts/hca_mechanism.py`：`nic_rx_loss` = 接收端 MAC 收到的
单播帧 − RDMA 引擎接受的包；本组实验中 136 MAC 收到的帧数与 137 发出的帧数
逐一相等，所以该差值就是丢在接收网卡内部的包。

| case | rail | 接收 buffer | 发送 buffer | 每 rail Gb/s | 136 每 rail nic_rx_loss | 136 PFC pause_tx | 137 ack_timeout |
|---|---|---|---|---|---|---|---|
| a1/a2 | 8 | 本地 | 本地 | 385.8–389.7 | ≤5k（噪声） | 0 | 0 |
| b1 | 8 | 跨 NUMA | 本地 | 197–221 | 13.2M–22.7M | 6.60M | 21063 |
| c1 | 仅 ionic_0 | 跨 NUMA | 本地 | 389.7 | 192 | 0 | 0 |
| d1 | NUMA0 的 4 条 | 跨 NUMA | 本地 | 208–220 | 18.3M–21.8M | 3.24M | 10128 |
| e1 | 8 | 本地 | 跨 NUMA | 121–312 | 0–64k，oos 0 | 0 | 15 |
| e2 | NUMA0 的 4 条 | 本地 | 跨 NUMA | 245–377 | 0 | 0 | 0 |

b1 的单条 rail（ionic_0）明细：137 发出 229,090,159 帧、136 MAC 全部收到，
RDMA 引擎只接受 206,417,184（丢 22,672,975，9.9%）；136 发出 1.62M 个 seq NAK、
976,810 个 priority-3 PFC pause（累计暂停 1.16 秒），但仍然丢包；137 重传
100,971,720 包（占发送 44%），其中 76.7M 被接收端判为 duplicate；
`rx_rdma_ecn_pkts`、CNP 均为 0。

结论：

- 单条 rail 跨 NUMA 写入能跑满线速且干净（c1），所以不是“跨 NUMA 不可用”；
- 多条 rail 同时把数据写进接收端另一 socket 的内存时，接收网卡内部丢包，
  PFC 已发出但没有阻止丢包，ECN/DCQCN 没有参与；go-back-N 重传和
  ACK timeout 随之暴增，带宽减半；
- 发送端跨 NUMA 读（e1/e2）只让发送端变慢，接收端没有序列错误；
  e1 仍有 15 次 ACK timeout，需在 GPU 显存实验中复核。

2026-10-06 补充排除项（`matrix/hostmem-tclass-ab-137-136-20261006T1047Z/`、
`matrix/hostmem-pcie-ro-ab-137-136-20261006T1051Z/`，与 b1 同形）：

- GRH traffic class 默认、0、104（DSCP 26）、106（DSCP 26 + ECT0）四轮
  结果相同：每轮 136 丢包 1.49–1.67 亿、PFC pause 5.9–10.8M、137 ACK timeout
  1.86–2.11 万；ECT0 下 `rx_rdma_ecn_pkts` 仍为 0。所以不是 RoCE 流量落入
  非 lossless 优先级这类可由软件改 TC 解决的问题，交换机也没有做 ECN 标记；
- `ib_write_bw --disable_pcie_relaxed` 下接收端跨 NUMA 丢包相同（1.61 亿），
  PCIe relaxed ordering 请求与否不影响；
- 发送端跨 NUMA（关闭 relaxed ordering）8 rail 共 1594 Gb/s，两端计数器全干净。

### 3.5 GPU 显存机理确认（137→136，Mooncake）

结果：`matrix/gpu-numa-mechanism-137-136-20261006T1054Z/`。与 3.3 节相同的
共享容器 8-pair 并发 harness，每 pair 1 GiB GPU buffer、32×32 MiB、60 秒，
每 wave 前后对两节点全部 HCA 做完整计数器快照（`mechanism.json`）。
offset 4 的 8 个 pair 全部跨 NUMA，offset 0 全部同 NUMA。

| policy | offset | 每 pair GB/s | 136 丢包 | 136 OOS | 136 PFC pause_tx | 137 重传包 | 137 ACK timeout |
|---|---|---|---|---|---|---|---|
| source-local | 4 | 20.8–26.0 | 41,983 | 63,371 | 156,941,398 | 412,935 | 0 |
| source-local | 0 | 38.1–40.8 | 0 | 0 | 0 | 0 | 0 |
| destination-local | 4 | 6.8–20.1 | 0 | 0 | 0 | 0 | 0 |
| destination-local | 0 | 38.2–40.9 | 0 | 0 | 0 | 0 | 0 |
| auto（生产策略） | 4 | 17.7–20.8 | 289,834,904 | 26,967,960 | 104,698,756 | 1,132,500,317 | 22,813 |
| auto（生产策略） | 0 | 31.6–38.3 | 2 | 0 | 4,928 | 0 | 0 |

- 接收端跨 NUMA 写 GPU 显存时，136 网卡持续发 PFC pause，仍有丢包和
  go-back-N 重传；同 NUMA 完全干净。机理与 host memory 一致；
- 生产策略 auto（shared 8-HCA + destination affinity）在全跨 NUMA 时最重：
  每个 source GPU 随机使用本 NUMA 的 4 条 rail，每个接收 HCA 同时承接 2 个
  source 写往另一 socket 的流量。60 秒内 137 产生 22,813 次
  `tx_rdma_ack_timeout`（生产故障中的 hard counter），136 网卡内部丢包 2.9 亿；
  数据仍 byte-verify 通过，wave 因 hard counter 门禁失败。这是故障机理在
  生产策略下的按需复现；
- destination-local 把跨 socket 段移到发送端的 DMA 读：两端计数器全为 0，
  代价是发送端跨 socket 读 GPU 显存较慢（8 pair 全跨 NUMA 时每 pair 6.8–20.1
  GB/s）。

这些 VM 的 PCI 拓扑是扁平的（GPU 与 HCA 都直挂在 `pci0002:00` /
`pci0003:00` 根下），Mooncake 按 PCI 距离选 HCA 时同 NUMA 的 4 个 HCA 距离
相同，因此 `hip:i` 的 preferred HCA 是本 NUMA 的 4 个、每次随机选一个；
destination affinity 再让 D 端用同名 HCA。138→136 复现的 36 个 CQE 中
35 个是“本端 HCA 与 Decode rank 不同 NUMA”，与此一致；唯一例外
（`ionic_2`→D1）发生在同一接收 HCA 同时承接跨 NUMA 写入的时段，接收网卡
丢包作用于该 HCA 上的所有 QP。

### 3.6 根因

结论：Decode 端 ionic 网卡的 PCIe DMA 写一旦被反压，就会在网卡内部丢掉已经
收下的 RoCE 帧，PFC 没有阻止。Mooncake 在这台 VM 上的 HCA 选择让 cross-rank
传输持续制造两种反压：接收网卡跨 socket 写显存，以及多张网卡同时写同一张
GPU。丢包足够密时 RC 重试耗尽，Mooncake 发送失败，SGLang 再把失败放大成
session 级联。链条如下：

1. VM 的 PCI 拓扑扁平（GPU 与 HCA 都直挂在每个 socket 的根下），Mooncake
   按 PCI 距离选 HCA 时同 NUMA 的 4 个 HCA 打平：`hip:i` 的 preferred 是本
   NUMA 的 4 个 HCA，每个请求随机选一个。裸机上同一 PCIe switch 下的那张网卡
   会单独胜出；
2. 本端 HCA 只按 Prefill buffer 选，destination affinity 让 Decode 端用同名
   HCA（rail 隔离，`ionic_k` 只通 `ionic_k`），于是出现两种触发条件：
   - 条件 A，跨 socket 写：i、j 不同 NUMA 时，Decode 端接收 HCA 与 GPU j
     不在同一 socket；
   - 条件 B，多网卡汇入同一 GPU：一个 Decode GPU 同时承接多个 Prefill rank
     的 KV 时，请求分散在多张网卡上（基底最多 8 张），线速之和超过该 GPU 的
     PCIe x16 链路（实测约 51 GB/s）；
3. 两种条件下接收网卡的 DMA 写都被反压，网卡已从 MAC 收下的帧在 RDMA 引擎前
   被丢弃；PFC pause 已发出，但没有阻止丢包；
4. go-back-N 重传、sequence NAK 与 ACK timeout 暴增；同一 QP 连续重试 7 次
   （`retry_cnt=7`）仍无进展时报 `transport retry counter exceeded`（CQE12），
   Mooncake 返回发送失败；
5. SGLang `MooncakeKVManager` 一次发送失败就把该 Decode rank 的 session 加入
   `failed_sessions`，此后发往该 rank 的请求直接以 `session ... is not alive`
   失败，直到 Decode 重新注册或 30 秒 probe 成功（3.1 节 18 个首发失败放大成
   38 个级联失败）。

原来的 same-rank 方案（`PD_DP_RANK_AFFINITY=1` 加 per-GPU `ionic_i`）不出错，
是因为 P_i→D_i 只走 `ionic_i`：每张 Decode GPU 只被同 socket 的一张网卡写入，
两个条件都不出现。

证据：

- 生产相关性：138→136 复现 56 个失败全部 cross-NUMA（56/1350，same-NUMA
  0/1438）；137→136 基线 51 个失败中 49 个 cross-NUMA（49/1359）、2 个
  same-NUMA（2/1342）；CQE 的本端与对端 HCA 名称全部相同；
- 生产中的条件 B：137→136 基线的 2 个 same-NUMA 失败都是 P0→D3（1 个首发、
  1 个级联），同一时段 D3 有一次 `ionic_0→ionic_0` 的 CQE12，即同 socket
  网卡写 GPU 3 也被重试耗尽，条件 A 解释不了；
- 条件 A 单变量：host memory（3.4 节）与 GPU 显存（3.5 节）都只因接收 buffer
  跨 NUMA 而丢包；
- 条件 B 单变量：fan-in 矩阵（3.7 节）中 4 个同 NUMA source 写同一 GPU，经 4 张
  网卡时丢包 2.8 亿帧，只经该 GPU 自己的网卡时为 0；
- 生产策略按需复现：auto 全跨 NUMA 60 秒 22,813 次 ACK timeout；
- 同一 C80 合同下的 base、v1（只消除 A）、v3（消除 A 和 B）对比见 3.7、3.8 节。

已排除：

- 单一坏 GPU、HCA 或 Decode rank：36 个 CQE 覆盖 8 HCA 和 8 D rank；
- worker/container 先死亡：首发事件是 CQE12 与 retry exhausted，随后才有
  session failed 和请求级级联；
- Router affinity 漂移、HiCache、CPU staging：live assertion 证明 affinity 关、
  HiCache 关，传输为 GPU RDMA；
- 确定性坏 pair/rail：顺序 192-pair 全部通过；
- 交换机或线路丢包：接收端 MAC 收到发送端发出的每一帧，丢包在接收网卡内部；
- RoCE traffic class / DSCP / ECN 配置：TC 默认/0/104/106 结果相同，ECN 标记为 0；
- PCIe relaxed ordering：关闭后丢包相同；
- 发送端跨 NUMA 读：不丢包（只变慢）；单 rail 单流跨 NUMA 写：线速且干净，
  必须多 rail 并发才触发；
- 总带宽：v1 的 C80 profiling 期间 136 入向平均约 16 GB/s 仍丢包，同 NUMA
  1:1 置换 270 GB/s 却干净，决定因素是每张 GPU 的入向网卡数。

需要 root 的平台部分（网卡在 PFC 下仍内部丢包、跨 socket P2P DMA 带宽与
读延迟）不影响本修复的有效性，交 IT 的建议见第 4 节。

### 3.7 修复：Mooncake 目的端固定 rail

修复位于 Mooncake 用户态的 HCA 选择和 QP 参数，不需要 root。最终版本 v3：

- 补丁基于镜像实际使用的上游 Mooncake `faae8dd4`，按顺序应用：
  1. `fix/mooncake-dest-local-rail.diff`（3 个文件 +36/−2）：新增
     `MC_ENABLE_DEST_LOCAL_RAIL`（要求同时开启
     `MC_ENABLE_DEST_DEVICE_AFFINITY`，否则告警并忽略），WRITE 请求以“对端为
     目标 buffer 选的 HCA”作为本端 HCA 的 hint；
  2. `fix/mooncake-ib-timeout.diff`（3 个文件 +22/−2）：新增 `MC_IB_TIMEOUT`，
     替换硬编码的 RC ACK 超时指数 14（67 ms）；
  3. `fix/mooncake-dest-pinned-hca.diff`（3 个文件 +33/−3）：多张 GPU 打平在
     同一组 preferred HCA 上时，Topology 按 GPU 序号给每张 GPU 分一张；目的端
     是 GPU 时总用这一张；
- 镜像：`fix/Dockerfile` 以 `fd7220a5` 为唯一基底，经仓库的
  `deploy/docker/scripts/build_mooncake_sglang.sh` 重编 `engine.so`（脚本新增
  `MOONCAKE_PATCHES`，不带补丁时行为不变），镜像 ENV 默认
  `MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=18`。`fix/build_image.sh <peer>`
  在本节点构建并 `docker save | docker load` 到对端，两端 image ID 必须一致；
- v3 镜像：`infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18`，
  `sha256:88286215b79d43306dce88dcdf299a13d939c6b708691dc192ca7c7c33c518a5`；
  运行配置 `config/config.rca-fix.p8d8.sh` 只替换镜像，其余合同不变。v4
  （`…-ibto20`，`sha256:6c87a4f8cef1…`）只改了 ENV `MC_IB_TIMEOUT=20`，用于检验超时
  推测，不是最终版本。

行为：每个写往 Decode GPU j 的请求两端都走 `ionic_j`（GPU 0–3 对
`ionic_0–3`、GPU 4–7 对 `ionic_4–7`，与硬件映射表 `rdma_map.yihou.md` 的
GPU_n↔ionic_n 一致）。

- 接收网卡只写同 socket 的、且只写自己那一张 GPU，条件 A、B 都消除；每张
  Decode GPU 的入向上限是一张网卡的线速（实测约 41 GB/s），低于其 PCIe 链路；
- 跨 socket 段移到发送端的 DMA 读；多个 Prefill rank 同时写同一张 GPU 时在
  发送网卡排队，排队不丢包；
- 发送网卡跨 socket 读显存在饱和下会停顿超过 67 ms（v1 A/B），生产负载下
  偶尔超过 1.07 s（v3 C80），产生不丢包的伪 ACK timeout。`MC_IB_TIMEOUT=20`
  （4.3 s）吸收这一停顿；真实丢包由 sequence NAK 立即触发重传，不受超时影响；
- READ 请求、host memory buffer（aux 数据仍在本 NUMA 的 HCA 中随机选）、对端
  无 hint、本端没有同名 HCA 时退回原逻辑；关闭开关即原行为。

为什么不能只改配置：

- 本端 HCA 只按本端 buffer 的位置选；`MC_ENABLE_HCA_PEER_AFFINITY` /
  `MC_NIC_PEER_AFFINITY` 只按本端 HCA 过滤对端候选。要让接收网卡只写本地显存，
  本端 rail 必须随目标 GPU 变化，只有知道目标 buffer 的请求级选择能做到；
- `MC_CUSTOM_TOPO_JSON` 能表达 GPU↔HCA 固定，但它是每节点静态文件、需要挂进
  容器、按进程可见的 GPU 序号索引，且不能代替按目标选 rail；per-GPU RDMA 列表
  （合同禁止）在 rail 隔离下让 i≠j 不可达；
- RC ACK 超时在 Mooncake 中是硬编码常量，没有环境变量。

迭代：
- v1 只有补丁 1（在目的端 NUMA 的 4 张网卡里随机选），消除了条件 A；v1 的 C80
  暴露出条件 B。
- v2 加补丁 2（超时 18）。
- v3 再加补丁 3，同时消除了条件 A 和 B。
- v4 只把超时默认值改为 20（Mooncake 层命中构建缓存，`engine.so` 与 v3 相同），
  用来检验"剩余 ACK timeout 是略超 1.07 s 的停顿"这个推测。结果推测不成立，
  所以最终版本回到 v3。

#### v1 并发饱和 A/B（条件 A）

结果：`matrix/fix-destrail-on-137-136-20261006T1113Z/`（offset 0–7，覆盖
全部 64 pair）与 `matrix/fix-destrail-off-137-136-20261006T1113Z/`。
同一修复镜像、生产策略 auto，唯一变量是 `MC_ENABLE_DEST_LOCAL_RAIL`；
8 pair 并发、60 秒持续满速写，压力远高于生产（138→136 C80 运行中 Prefill
节点 KV 发送速率 10 秒均值平均 62 Gb/s、峰值 260 Gb/s，这里是 1070–2200 Gb/s）。

| 开关 | offset（跨 NUMA pair 数） | 8 pair 合计 GB/s | 136 丢包 | 136 OOS | 136 PFC pause_tx | 137 重传包 | 137 ACK timeout |
|---|---|---|---|---|---|---|---|
| 0 | 4（8） | 150.2 | 287,269,617 | 26,677,352 | 104,646,606 | 1,123,716,526 | 22,578 |
| 0 | 0（0） | 276.7 | 0 | 0 | 0 | 0 | 0 |
| 1 | 4（8） | 133.8 | 8 | 0 | 0 | 1,416,699 | 7,161 |
| 1 | 0（0） | 270.5 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 / 7（2） | 234.4 / 247.3 | 2 / 0 | 0 | 184 / 0 | 48,343 / 11,118 | 248 / 54 |
| 1 | 2 / 6（4） | 142.1 / 134.2 | 0 | 0 | 0 / 352 | 1,792,764 / 2,130,355 | 14,585 / 13,372 |
| 1 | 3 / 5（6） | 142.2 / 140.1 | 0 / 2 | 0 | 0 | 2,375,900 / 1,789,768 | 14,703 / 11,349 |

- 开关 0 与基底镜像 auto offset 4（3.5 节）逐项一致：修复镜像关掉开关就是原
  行为，差异只来自开关；
- 开关 1 下 64 pair 全部 byte-verify，接收端丢包、OOS、PFC 消失（≤8 帧
  为非 RDMA 帧）；
- 跨 NUMA wave 仍有 ACK timeout，但不是丢包：137 发出的数据帧全部被 136
  RDMA 引擎接受，136 发回的 ACK 全部到达 137，137 没有收到 seq NAK，
  且每个 wave 137 的重传包数与 136 的 `dup_request` 完全相等。即发送端在
  67 ms（`kTimeout=14`）内没等到 ACK 就重传了已送达的包：发送网卡跨 socket
  读 GPU 显存在饱和下停顿超过 ACK 超时。全同 NUMA wave 为 0，单 rail 单流的
  destination-local（3.5 节）也为 0；本 socket 读与跨 socket 读混在同一发送
  HCA 上的 offset 2/3/6 比全跨的 offset 4 更多；
- 深度扫描（offset 2，`matrix/fix-depth-sweep-137-136-20261006T1126Z/`）：
  `MC_MAX_WR=64` 15,580、`MC_MAX_WR=16` 3,714、`MC_NUM_QP_PER_EP=1` 2,741
  （跨 NUMA pair 降到 8.8–10 GB/s），只能缓解、不能消除，未纳入修复；
- 吞吐：≥4 个 pair 跨 NUMA 时合计 134–142 GB/s，开关 0 的 offset 4 为
  150 GB/s；跨 socket DMA 的总带宽是两种方向共同的上限。

#### v1 的 C80 生产负载结果

运行：`runs/fix-01-137-136-20261006T1141Z/`（11:41–12:29 UTC，v1 镜像
`7448464559fb`，live config 证明两端镜像、`MC_ENABLE_DEST_LOCAL_RAIL=1`、
affinity 关、shared 8-HCA）。

- 传输：Prefill/Decode transfer failure、CQE12、retry exhausted、session
  failed、`session not alive` 全为 0；两端 `num_transfer_failed_reqs`、
  `num_bootstrap_failed_reqs`、`failed_session_recoveries` 均为 0；
- Router 配对保持全局分布：cross-rank/cross-NUMA 1389、cross-rank/same-NUMA
  1028、same-rank 349；
- 客户端：profiling 2592 个请求 0 错误；warmup 有 2 个
  `InvalidInferenceResultError`，是 `max_tokens=1` 的 warmup 请求（ISL
  435,986 与 645,954）返回了不含正文的单 token，repro-01 中同一请求返回 1 个
  token；±5 秒内没有任何服务端传输失败。它们不是传输故障，但 certify 的
  客户端门禁仍按失败计；
- HCA：137 `req_tx_retry_excd_err` 0，`tx_rdma_ack_timeout` 1245（8 个 HCA
  都有）；136 网卡内部丢包 736 万帧（137 发出 49.2 亿帧的 0.15%）、OOS
  123 万、PFC pause_tx 1118 万，8 个接收 HCA 都有。

结论：接收网卡此时只写本 socket 显存，生产负载下仍丢包，说明还有第二个
触发条件。它与总带宽无关：profiling 期间 136 平均入向只有约 16 GB/s（每分钟
约 1 TB），远低于
矩阵中同 NUMA 270 GB/s 仍干净的水平。差别在于生产中多个 Prefill rank 会同时
写同一个 Decode GPU，而 v1 让每个 Decode GPU 经本 NUMA 的 4 张网卡接收，多张
网卡的线速汇入同一 GPU 的 PCIe x16 链路；矩阵 wave 都是 1:1 置换，每个 Decode
GPU 最多只收一个 source 的流量（受 source GPU 自身链路限速）。fan-in 矩阵验证
见下一小节。

#### fan-in 矩阵（条件 B）

结果：`matrix/fanin4-v2-137-136-20261006/`（v2 镜像）与
`matrix/fanin4-v3-137-136-20261006/`（v3 镜像）。`run_concurrent_matrix.py
--fan-in 4`：每 4 个连续 source 同时写一个 destination GPU（offset 0 为
P0–3→D0、P4–7→D4，全同 NUMA；offset 4 为 P0–3→D4、P4–7→D0，全跨 NUMA），
60 秒持续满速写。从这一轮起 worker 与 SGLang rank 一样能看到全部 8 张 GPU，
buffer 的 Mooncake location 即物理 GPU 号（pair worker schema 3）。

| 镜像与策略 | offset | 每 pair GB/s | 每 D GPU GB/s | 136 丢包 | 136 OOS | 136 PFC pause_tx | 137 重传包 | 137 ACK timeout |
|---|---|---|---|---|---|---|---|---|
| v2 auto（目的 NUMA 内 4 张网卡） | 0 | 12.3–13.5 | 51 | 280,401,207 | 20,375,043 | 60,769,560 | 1,326,269,151 | 34,509 |
| v2 auto | 4 | 13.3–14.0 | 55 | 100,465,714 | 7,974,739 | 70,304,420 | 394,492,405 | 16,774 |
| v2 destination-local（只用 `ionic_j`） | 0 | 9.3–11.2 | 41 | 0 | 0 | 0 | 0 | 0 |
| v2 destination-local | 4 | 8.0–8.5 | 33 | 0 | 0 | 0 | 0 | 0 |
| v3 auto（固定 rail） | 0 | 9.2–11.3 | 41 | 0 | 0 | 0 | 0 | 0 |
| v3 auto | 4 | 7.9–8.7 | 33 | 0 | 0 | 0 | 0 | 0 |

- 4 张网卡写同一张 GPU 时，入向被该 GPU 的 PCIe 链路限在 51–55 GB/s，网卡
  大量丢包；offset 0 的写入全部在本 socket，说明条件 B 单独即可触发；
- 这两个 wave 在 1.07 s 超时下仍有上万次 ACK timeout，是真实丢包后的尾部
  超时，不是伪超时；
- 每张 GPU 只经自己的网卡接收时完全干净，入向是一张网卡的线速（本 socket
  读约 41 GB/s，跨 socket 读约 33 GB/s）；offset 4 的发送端跨 socket 读在
  1.07 s 超时下没有 ACK timeout；
- v3 的生产策略 auto（shared 8-HCA）与只用 `ionic_j` 的结果一致，证明固定
  rail 的代码路径按预期工作。

#### v3 的 C80 与 64 pair 验收矩阵

运行：`runs/fix-02-137-136-20261006T1337Z/`（13:37–14:24 UTC，v3 镜像
`88286215b79d`，live config 证明两端镜像、`MC_ENABLE_DEST_LOCAL_RAIL=1`、
`MC_IB_TIMEOUT=18`、affinity 关、shared 8-HCA）；验收矩阵
`matrix/cert-v3-137-136-20261006/`（8 个置换 wave，每 wave 60 秒，覆盖 64 pair）。
`certify_results.py` 的 22 个门禁过 21 个（`analysis/certification.json`）：

- 矩阵：64 pair 全部 byte-verify，CQE12、retry exhausted、
  `req_tx_retry_excd_err`、`tx_rdma_ack_timeout` 全为 0，丢包、OOS、PFC 也为 0；
- AgentX：warmup 164 个、profiling 2608 个请求全部成功；transfer failure、
  CQE12、retry exhausted、session failed、`session not alive` 均为 0；8 个
  Decode rank 各 347–348 个请求；
- 未过：137 `tx_rdma_ack_timeout` 20。

Decode 日志里有 1 行 `Decode transfer failed ... Aborted by AbortReq`（14:21:19，
DP4）。aiperf 在 14:21:17 profiling 宽限期结束时取消了 7 个在途请求，Decode 的
`num_aborted_requests_total` 正好是 7（基线 8、v1 10，也都等于各自的取消数；
基线 51 个真实传输失败不计入这个指标）。这是客户端取消恰好落在 KV 传输阶段，
不是传输失败。`analyze_reproduction.py` 现在把这种行单独计为
`transfer_client_aborted`，certify 要求它不超过客户端取消数。

HCA 计数（10 秒时间序列）：16 张网卡里只有 `ionic_3` rail 和 137 `ionic_7`
有事件，其余全为 0（≤4 帧非 RDMA 帧）。

| 时间窗 (UTC) | 137 HCA | ACK timeout | 137 重传包 | 136 重复包 | 136 seq NAK |
|---|---|---|---|---|---|
| 14:08:24–34 | `ionic_3` | 8 | 1,575 | 1,575 | 0 |
| 14:16:54–17:04 | `ionic_3` | 5 | 3,009 | 1,435 | 2 |
| 14:18:44–54 | `ionic_7` | 2 | 0 | 0 | 0 |
| 14:19:34–44 | `ionic_3` | 5 | 2,794 | 1,290 | 2 |

- 136 全程没有发 PFC pause，接收 buffer 没有压力，与条件 A、B 下丢包总伴随
  大量 pause 的特征不同；
- 第一窗的重传全是重复包，与 v1 A/B 中的伪超时相同，只是 ACK 等待超过了
  1.07 s；`ionic_7` 的 2 次超时没有重传，即超时时发送端连原包都还没发出，
  停顿在发送端；
- 后两窗多出的约 3,000 个非重复重传，与 136 `ionic_3` 全程 3,023 帧
  `nic_rx_loss` 吻合。推断是伪重传与迟到的原包交错，接收端发 NAK 后按
  go-back-N 丢弃乱序包，不是网卡因 DMA 反压丢帧；
- 稳态饱和不触发：验收矩阵中 rail 3 以 38 GB/s 跑 60 秒为 0；fan-in 8
  （`matrix/fanin8-v3-137-136-20261006/`，8 个 source 经一张网卡写同一张 GPU，
  本 socket 读与跨 socket 读混在同一发送网卡上，每张网卡约 36 GB/s）在 D3、
  D0、D4、D7 上也为 0。

结论：v3 下接收端不再因反压丢包，只剩生产负载中偶发的 ACK timeout，没有一次报错。
retry exhausted 为 0，说明没有一个 QP 连续超时 7 次（约 7.5 s）。当时推测这些超时
是发送网卡 DMA 读偶尔停顿超过 1.07 s，下一小节用 v4 检验了这个推测，结果不成立。

#### v4：检验"剩余 ACK timeout 是略超 1.07 s 的停顿"

v4 镜像 `6c87a4f8cef1` 与 v3 的唯一区别是 `MC_IB_TIMEOUT=20`（4.3 s）。如果剩余超时
来自略超 1.07 s 的停顿，超时拉长 4 倍后，次数应该大幅下降。

运行：
- C80：`runs/fix-03-137-136-20261006T1453Z/`（14:53–15:38 UTC）；
- 验收矩阵：`matrix/cert-v4-137-136-20261006/`；
- 计数器轮询对照：`matrix/monitor-ab-v4-137-136-20261006/`。

结果：
- certify 22 个门禁过 21 个，未过的同样只有 `tx_rdma_ack_timeout`（18）；
- 客户端、transfer failure、CQE12、retry exhausted、session failed 均为 0；
- 8 个 Decode rank 各 341–342 个请求；
- 验收矩阵 64 pair 全部通过，计数器全为 0。

| 时间 (UTC) | 137 HCA | ACK timeout |
|---|---|---|
| 15:23:36 | `ionic_3` | 2 |
| 15:31:06 | `ionic_5` | 10 |
| 15:32:46 | `ionic_0` / `ionic_1` | 4 / 2 |

- 两个节点各跟踪 36 个故障类计数（`hw_counters` 与 `ethtool -S`）。全程只有三个
  计数有变化：137 的 ACK timeout 18 次、重传 3,938 包，136 的 `resp_rx_dup_request`
  3,935。两端都没有丢包、seq NAK、OOS 或 PFC pause。也就是说，原包都送达了接收端，
  发送端在 4.3 s 内没等到 ACK 而重传。
- 次数与超时长度几乎无关：1.07 s 下 20 次，4.3 s 下 18 次。所在网卡也不固定：v3
  的 20 次有 18 次在 `ionic_3`，v4 分散在 4 张网卡上。所以它们不是略超 1.07 s 的
  停顿，推测不成立；真实原因还没有定位。
- 计数器轮询对照：同一 300 秒饱和 wave，开 1 秒轮询和不开轮询都是 0 次超时，排除
  轮询的干扰。
- 影响：20 分钟内多传约 16 MB，没有报错。7 次重试耗尽需要同一 QP 连续超时，从未
  发生。

所以最终默认值保持 18（1.07 s，NCCL 的默认值）。4.3 s 不能减少这些超时，反而会让
对端真正失联时，要约 30 秒（7 × 4.3 s）才报错，而 1.07 s 下约 7.5 秒。

#### 138→136 复核与 69 s 超时（2026-10-08）

在 09-21 复现所用的 138→136 上，同一合同各跑一轮，计数器 1 秒采样；明细见
`operations/ack-timeout-138-136-20261008.md`。

- **A 轮 v3（`runs/fix-04-138-136-20261008T0339Z/`）：**修复在第二个节点对上同样成立。
  profiling 2608 个请求，客户端错误、transfer failure、CQE12、retry exhausted、session
  failed 均为 0。另外，这一轮开始传输时，8 条 rail 同时在网络里丢了约 12 万帧（MAC 层
  就少了），全部靠 1.07 s 超时恢复，没有失败。
- **B 轮 v3 + `MC_IB_TIMEOUT=24`（69 s，`runs/fix-05-138-136-20261008T0425Z/`）：**
  profiling 只完成 1387 个请求，有 3 个 transfer failure 和 1 次 session failed，aiperf
  因 TTFT 只覆盖 63.7% 的窗口判为失败。一次停顿先触发了 Mooncake 自己的 30 秒同步传输
  超时。
- **结论：**剩余的 ACK timeout 不是计时器误触发。每一次都对应一个真的停住、只能靠超时
  重传恢复的 QP，停顿时长等于超时时长，所以超时必须短。v3 的 1.07 s 是正确选择。
- **停顿发生在接收网卡：**纯重复包事件中原包都已进入 136 的 RDMA 引擎，但 136 没有及时
  回 ACK；约 1.5 千包的事件中包到达了 136 的 MAC，却在网卡内部被丢弃，且没有任何错误
  计数。
- **交换角色 136→138（`runs/fix-06-136-138-20261008T0608Z/`）：**v3 同样 0 失败
  （2579 个请求，门禁 21/22，266,477 tok/s）。138 作为接收端出现同样两种现象：两次纯重复包
  超时（`ionic_4`、`ionic_0`，共 14 次），以及 `ionic_3` 上 4 次共 4,496 包在 138 网卡内部
  被丢（MAC 层无缺口），后者有 NAK，按序号重传恢复，没有超时。所以这是 ionic 网卡作为
  接收端的行为，不是 136 这块网卡的问题；136 作为发送端没有异常。用户态计数只能定位到
  这里，再往下需要网卡内部计数或固件日志（第 4 节第 1 条）。

Session 拉黑级联（3.1 节 13 次 session failed、38 次 `session not alive`）
发生在 CQE12 首发失败之后，是对首发失败的放大；首发失败为 0 时没有触发源。
它不作为本次修复对象（README 禁止以应用层重试作为修复），验收以
`session_failed`/recovery 为 0 证明修复后不再出现。

### 3.8 最终验收

最终版本：v3 镜像 `infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18`
（`sha256:88286215b79d…`），镜像 ENV 为 `MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=18`。

`scripts/certify_results.py` 把下列门禁固化为机器检查，并核对 AgentX run 的固定
配置、1200 秒时长，以及矩阵与 AgentX 使用同一镜像。矩阵只取 `auto` policy 作为
64 pair 验收集。结果写在 `runs/<run>/analysis/certification.json`。

| 硬门禁 | v3（`fix-02` + `cert-v3`） | v4（`fix-03` + `cert-v4`） |
|---|---|---|
| 64 个 `P_i→D_j` 全部 byte-verify | 通过 | 通过 |
| AgentX 客户端失败 0 | 通过 | 通过 |
| Prefill/Decode transfer failure 0 | 通过 | 通过 |
| CQE12 / retry exhausted 0 | 通过 | 通过 |
| `req_tx_retry_excd_err` 无增量 | 通过 | 通过 |
| `tx_rdma_ack_timeout` 无增量 | 矩阵通过；C80 **未过（20）** | 矩阵通过；C80 **未过（18）** |
| session failed / recovery 0 | 通过 | 通过 |
| affinity off 下 Decode rank 负载全局分布 | 通过（每 rank 347–348） | 通过（每 rank 341–342） |

同一合同（137→136，P8D8，C80，warmup per lane 1，1200 秒，HiCache 关，shared
8-HCA，`PD_DP_RANK_AFFINITY=0`）下的结果。每个版本各跑一次：

| 运行 | 镜像 | profiling 请求 | 客户端错误 | transfer failure | 吞吐 tok/s | TTFT 均值 / p50 s | ITL 均值 / p95 ms | 137 ACK timeout |
|---|---|---|---|---|---|---|---|---|
| `repro-02` | 基底 `fd7220a5` | 2478 | 54 | 51 | 256,452 | 10.62 / 4.88 | 12.02 / 21.16 | — |
| `fix-01` | v1 | 2592 | 2（warmup 空响应） | 0 | 268,402 | 9.77 / 5.00 | 12.19 / 20.75 | 1245 |
| `fix-02` | **v3（最终）** | 2608 | 0 | 0 | 270,759 | 10.09 / 4.79 | 12.58 / 20.21 | 20 |
| `fix-03` | v4 | 2556 | 0 | 0 | 264,555 | 9.74 / 4.52 | 12.68 / 20.45 | 18 |

- **功能：** 修复后 transfer failure 从 51 个降到 0。v3 的吞吐比基底高 5.6%，TTFT
  均值低 5%。
- **代价：** ITL 均值高约 5%（12.02 → 12.58 ms），p95 反而更低。每张 Decode GPU 只经
  一张网卡接收，入向上限在本 socket 读时约 41 GB/s，跨 socket 读时约 33 GB/s（3.7 节
  fan-in 矩阵）。C80 下的实际 KV 速率远低于这个上限：138→136 运行中，整个 Prefill
  节点的发送速率平均只有 62 Gb/s。
- **未闭合：** C80 中偶发的 ACK timeout（v3 20 次集中在 4 个时间窗，v4 18 次集中在 3 个；见
  3.7 节"v4"小节）。它们不丢包、不报错，次数与超时长度无关。10-08 的复核表明，它们是接收
  网卡内部的真实停顿（136、138 作为接收端都有），而且必须靠短超时恢复（3.7 节末）。在定位到具体原因之前，
  `tx_rdma_ack_timeout` 门禁按未通过记录。

## 4. 交 IT 的建议（需要 root，不阻塞本修复）

本修复让 RoCE 不再依赖“网卡跨 socket 写”这条会丢包的路径，但以下平台问题仍在，
会影响任何把网卡和远端 socket 显存配对使用的负载：

1. 网卡在 PFC 下仍内部丢包。RoCE 无损类（priority 3）里，接收网卡因 PCIe
   写回压而丢掉已收下的帧，同时又在发 PFC pause，说明 pause 阈值/headroom
   不足以覆盖 DMA 写的停顿。请 IT/AMD 在 136/137（及同型号节点）检查 ionic
   固件版本与已知问题，以及 priority 3 的 xoff/xon 阈值、headroom 和 RX
   buffer 分配（例如 `nicctl show qos`、`nicctl show port`，具体命令以厂商
   文档为准），调到 PFC 能在丢包前生效。验证方法无需 root：重跑
   `scripts/run_hostmem_numa_ab.py` 的 b1 形状（8 rail，接收 buffer 跨 NUMA），
   `nic_rx_loss` 应从现在的约 10% 变为 0。修复后 C80 仍有少量接收网卡内部的停顿：
   原包已进入 RDMA 引擎却迟迟不回 ACK，或包到达 MAC 后被丢，都没有错误计数，136、138
   作为接收端都会出现（3.7 节末）。请厂商用网卡内部计数或固件日志一并定位；
2. 跨 socket P2P DMA 的带宽与读延迟。网卡与另一 socket 的 GPU 之间，8 rail
   合计只有约 134–190 GB/s；网卡读远端显存在饱和下会停顿超过 67 ms（v1 A/B）。
   请检查宿主机 BIOS 的 xGMI 链路宽度/速率与 Data Fabric 设置，以及直通设备的
   IOMMU/ATS 模式；
3. 向 VM 暴露真实 PCIe 拓扑（或提供 GPU↔NIC 亲和表）。目前 guest 中 GPU 与
   HCA 都直挂在根下，Mooncake/RCCL 只能按 NUMA 区分，无法优先选同一 PCIe
   switch 下的 GPU-NIC 对。本修复按 GPU 序号配对，与硬件映射表一致，但无法
   在 guest 内验证物理上是否同一 switch；
4. 交换机没有对 RoCE 做 ECN 标记（ECT0 流量 `rx_rdma_ecn_pkts` 仍为 0），
   DCQCN 不起作用。若开启 priority 3 的 ECN 标记，发送端会在接收端 buffer
   溢出前降速。

## 5. 运维记录

- 2026-09-22 暂停：`operations/pause-20260922T0649Z.md`；
- 2026-09-28 换节点对为 137→136：`operations/resume-20260928T1225Z.md`；
- 2026-10-06 停止 136/137 上他人的容器：`operations/stop-others-136-137-20261006T1037Z.md`；
- 2026-10-06 恢复、全部任务与容器清理：`operations/resume-20261006T1045Z.md`；
- 2026-10-08 收尾（最终版本定为 v3、补跑认证、修复时间戳解析、目录移到 `llying/`）：
  `operations/wrapup-20261008.md`；
- 2026-10-08 剩余 ACK timeout 调查（138→136、69 s 超时、交换角色 136→138）：
  `operations/ack-timeout-138-136-20261008.md`。
