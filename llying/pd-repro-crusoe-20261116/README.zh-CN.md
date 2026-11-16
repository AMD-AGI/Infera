# crusoe（spur）上复现 GLM-5.2 AgentX PD 最高吞吐配置：P8D8 C208 与单节点 P4D4 C104（2026-11-16）

复现对象是 `../dp-graph-variant-fix-aus-20261103/` 在 AUS 上的两个结果：1P(PP8) + 1D(TP8/DP8) C208 56,593 tok/s/GPU（RESULT 7.5），
单节点 4+4 C104 53,209（RESULT 7.1，`MC_DISABLE_HIP=1`，3600 s）。踩坑与检查清单另见 `skill/crusoe-glm52-pd/SKILL.md`。

## 0. 结论

| 配置 | 本集群 | AUS | 差 |
|---|---|---|---|
| P8D8 C208（16 卡，v3-096 PP8 prefill → v3-110 TP8/DP8 decode） | **54,468** | 56,593 | −3.75% |
| P4D4 C104（单节点 v3-036，prefill GPU 0-3 → decode GPU 4-7） | **53,351** | 53,209 | +0.3% |

- 两者都在 AUS 记录的节点间波动内。P8D8 的差在 prefill：本集群 prefill 的每批负载、单段耗时、批次间隔、现算吞吐（19,196 tok/s）与 AUS 11-09
  另一次 PP8 计分（19,276 tok/s）几乎重合，AUS 自己 11-06（20,453）与 11-09 之间就差 6%；decode 步长与 AUS 完全相同（第 2 节）。
- 要在本集群跑出这两个数，必须做三处改动（第 3 节）：KV 用 GEM 分配（驱动对 dma-buf 注册的显存计两次，本集群又没有 peer memory）；
  **不往容器里注入宿主机 libionic**（否则 Mooncake 找不到网卡、悄悄走 TCP）；**修正 GEM 显存的设备号**（否则 dest-local-rail 把所有 KV 压到 ionic_0）。
  另外要处理坏节点和 spur 的作业语义（第 3.4 节）。
- 前两次 P8D8 计分和第一次 P4D4 门禁 / 计分是在这些问题下跑的，不算结果（第 2.3 节）。

## 1. 环境

| 项 | 值 |
|---|---|
| 集群 | crusoe spur，分区 `amd-spur-v2`；节点 `crsuse2-m2m-v3-*` 是 Cloud Hypervisor（KVM）虚拟机：8 × MI355X、8 × Pensando DSC VF（400G）直通，2.75 TiB 内存，本地 NVMe `/mnt/m2m_nobackup` |
| 驱动 / 内核 | amdgpu-dkms 6.16.13（限额检查保留 `+ kfd.vram_pinned`），内核 6.8.12-322.crusoe，ionic 26.08.17.001，宿主机 Ubuntu 24.04 |
| 作业 | P8D8：291252（`amd-primus` / `amd-primus-qos`，v3-096 + v3-110）；P4D4：291495（同 QoS，v3-036）。`amd-agentx-4-qos` 组上限 10 节点、每人 2 节点，当天满额 |
| 镜像 | `infera-sglang:v0522-fdbafb60`（`/shared_nfs/liyingli/images/infera-sglang-v0522-fdbafb60.tar`），11 个基线文件 md5 一致 |
| 运行包 | 验证包 `../dp-graph-variant-fix-20261103/` + AUS 的 kit 改动（`../dp-graph-variant-fix-aus-20261103/opt/kit-router-hooks.diff`）+ opt 配置 + `ptpc-fix/glm4_moe.py`（镜像原文件 + `opt/glm4_moe.pr45800.diff`，偏移 5 行）+ 本目录 `patches/` |
| 配置 | P8D8 `arm-C-1p1d-pp8-gemkv.env`；P4D4 `arm-C-prod-ptpcfix-nohip-gemkv.env` + `DURATION=3600`；站点 `config/site.local.env.crusoe`（含 `KIT_NO_HOST_LIBIONIC=1`），GEM `config/gemkv.env.crusoe` |
| 模型 | `GLM-5.2-MXFP4`，每次作业先拷到本地 NVMe（`scripts/stage_model.sh`，438 GB 35–70 s） |

工作目录（日志、队列、引擎日志的副本）在 `/home/liyingli/bench_agentx/pd-repro-20261116/`；引擎日志原件在各 prefill 节点的
`/mnt/m2m_nobackup/liyingli_pdr/work/runs/`。

## 2. 结果

### 2.1 P8D8 C208（`results/score-p8d8-gemfix-c208.txt`，门禁 smoke 6/6、探测 195/200、16k 32/32 + 31/32、接受长度 3.62）

| | 本集群 | AUS 11-06 |
|---|---|---|
| tok/s/GPU | 54,468 | 56,593 |
| 输出 tok/s/GPU / req/s | 410.0 / 4.851 | 428.9 / 5.036 |
| 完成 / 丢弃 | 17,594 / 52 | 18,265 / 47 |
| 交互 p50 / p90 | 88.7 / 73.8 | 86.0 / 70.9 |
| ITL p50 / p90（ms） | 11.3 / 13.6 | 11.6 / 14.1 |
| TTFT p50 / p90 / 均值（s） | 8.10 / 16.37 / 12.06 | 7.50 / 14.47 / 10.82 |
| 缓存命中 | 97.05% | 96.98% |
| prefill 满块中位 / 现算 / 排队均值 | 345 ms / 19,196 tok/s / 60.7 | 324 ms / 20,453 / 52.6 |
| decode 步长 @12 / @17；等预分配 / 等传输 | 33 / 40 ms；4.3 / 94.4 | 33 / 40 ms；4.0 / 85.2 |

prefill 与 AUS 的对照（`scripts/pp_stage_time.py`，口径同 `BOTTLENECK-PP8-20261109` 4b 节；`results/pp8-prefill-stages.txt`）：

| C208 测量整小时 | 本集群 | AUS 11-09（n04-33 → n10-29，prefill 同配置） |
|---|---|---|
| 每批新 token / 请求 / 已缓存 token | 7,752 / 2.90 / 25.6 万 | 7,749 / 2.85 / 25.0 万 |
| PP0 批次间隔中位（p10 / p90） | 332（160 / 647）ms | 326（165 / 651）ms |
| 单段耗时（相邻段中位） | 268–374 ms | 261–369 ms |
| prefill 现算 | 19,196 tok/s | 19,276 tok/s |

结论：差距来自 prefill 现算的节点间波动（REVIEW-20261112 记录同配置不同节点对 prefill 每块 323 对 362 ms），不是本集群特有的慢；
单节点 P4D4 的 prefill 满块（320 / 329 ms）也与 AUS（331 ms）一致。修复后 KV 流量分到 8 条 rail，各 15–22 TB，没有传输失败；
重传与 ECN 仍偏多（crsuse2 接收端反压丢帧，`../cross-rank-rca-20260921/`），等传输比 AUS 长约 10%。

### 2.2 P4D4 C104（`results/score-p4d4-gemfix-c104.txt`，门禁 smoke 6/6、探测 198/200、16k 32/32 + 32/32、接受长度 3.66）

| | 本集群 | AUS 11-05 |
|---|---|---|
| tok/s/GPU | 53,351 | 53,209 |
| req/s / 完成 / 丢弃 | 3.469 / 12,582 / 1 | 3.483 / 12,625 / 2 |
| 交互 p50 / p90 | 69.9 / 60.8 | 70.3 / 61.0 |
| TTFT p50 / p90 / 均值（s） | 4.20 / 9.23 / 5.65 | 4.25 / 9.64 / 5.99 |
| prefill 满块 / 现算 | 320 ms / 16,631 | 331 ms / 16,682 |
| decode 步长 @16；等预分配 / 等传输 | 41 ms；2.4 / 16.6 | 40 ms；2.9 / 17.5 |

同节点回环流量在 decode GPU 对应的 ionic_4-7 上，各 20-23 TB，重传可忽略（`results/ionic-counter-deltas.txt`）。

### 2.3 不算结果的运行（问题见第 3 节）

| 运行 | 结果 | 原因 |
|---|---|---|
| P4D4 门禁 06:08（TCP） | 16k 长上下文 64 个请求错 29 个（HTTP 500） | Mooncake 走 TCP，30 s 传输超时 |
| P4D4 计分 06:36（TCP） | 预热中停掉 | 同上 |
| P8D8 计分 07:46（TCP） | 预热 20 min 时一个根请求传输超时，AIPerf 中止 | 同上 |
| P8D8 计分 08:36（TCP，`MC_TRANSFER_TIMEOUT=300`） | 发现走 TCP 后停掉 | 同上 |
| P8D8 计分 09:10（RDMA，旧 GEM 库） | 测量 47 min 时 prefill PP0 `Prefill out of memory ... Available full tokens: 0`，prefill 崩溃（`results/score-p8d8-rdma-oldshim-prefill-oom.txt`） | 所有 KV 压在 ionic_0：122 TB、重传 46.7 亿包、ACK 超时 167 万，等传输的请求锁满 prefill KV |
| P4D4 计分 09:29（RDMA，旧 GEM 库） | 48,903（−8.1%），TTFT 均值 +2.8 s（`results/score-p4d4-rdma-oldshim-c104.txt`） | 同节点回环全在 ionic_0 |

## 3. 问题、根因与修法

### 3.1 RDMA 注册的显存计两次 → KV 用 GEM 分配

- 驱动 6.16.13：dma-buf 注册（`amdgpu_dma_buf_pin`）只加 `vram_pin_size`、不加 `kfd.vram_pinned`，KFD 分配的显存被固定后**计两次**；
  peer memory 注册在 6.16 上不多扣（`../dp-graph-variant-fix-aus-20261103/driver619/README.zh-CN.md` 第 2-3 节）。
- **本集群没有 peer memory**：amdgpu 带 peer memory 客户端代码、MOFED 导出了 `ib_register_peer_memory_client`，但没有
  `/sys/kernel/mm/memory_peers`，对显存 `ibv_reg_mr` 8 张网卡全部 EFAULT。所以只能走 dma-buf，而这条路在 6.16 上就多扣。
  这就是本集群以前"mem fraction 只能开很小"的原因。与 AUS（6.19 上两条路都多扣）是同一处记账缺陷、不同触发路径。
- 宿主机最小复现（`scripts/driver619/repro_host.sh`，docker 不可用时直接用宿主机 ROCm 7.2；`results/driver-repro-*.md`）：

  | 用例（v3-058） | 剩余可分配：注册前 → 后 |
  |---|---|
  | KFD + dma-buf 40 GiB | 287.25 → 207.25 GiB（扣 2 × 40） |
  | KFD + dma-buf 150 GiB | → 0，新建 stream 段错误（exit 139） |
  | KFD + peer memory | `ibv_reg_mr` EFAULT（无 peer memory） |
  | GEM + dma-buf 40 / 150 GiB（等 10 s 再测） | 247.0 / 137.0 GiB（只扣一次） |

  （GEM 用例不等待时"注册后余量 0.5 GiB"是测量假象：上一轮余量搜索释放的显存尚未反映到 `mem_info_vram_used`，搜索上限被压低。）
- 修法：沿用 AUS 的方案 B（`config/gemkv.env.crusoe`：GEM 分配器 `LD_PRELOAD`、`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM`、
  `MOONCAKE_DISABLE_HIP_DMABUF=0`、`MC_HIP_KFD_PIN_FIX=0`、`MC_DISABLE_HIP=1`）。decode KV 409.5 万 token / rank、prefill 156-163 GB / 段，与 AUS 相同。

### 3.2 容器里没有 ionic 网卡，Mooncake 悄悄走 TCP → 不注入宿主机 libionic

- 镜像入口 `infera-inject-host-ionic` 把挂进来的宿主机 libionic 复制到镜像的 provider 上。本集群宿主机是 Ubuntu 24.04，它的 libionic 需要
  **GLIBC_2.38**，镜像（22.04）只有 2.35：`dlopen` 失败，libibverbs 只报 `couldn't load driver 'libionic-rdmav34.so'`，`ibv_devices` 里没有 ionic。
  Mooncake 日志 `Found 0 HCAs` → `TcpTransport`，所有 KV 走 TCP（`results/ibverbs-glibc-v3-219.txt`）。
- **镜像自带的 libionic（54.0-187）与宿主机内核模块兼容**，不注入时 8 张 ionic 都在。
- 修法：kit 加开关 `KIT_NO_HOST_LIBIONIC=1`（`patches/kit-no-host-libionic.diff`，随 `KIT_VARS` 带到 decode 节点），本集群站点配置默认打开。
- 走 TCP 时的现象：同节点"回环"3.5-9 GB/s 且与切片 / QP / 线程参数无关、网卡 RDMA 计数器不动、长上下文请求 30 s 超时、跨节点高并发时
  传输超时（`results/mooncake-loopback-TCP-v3-219.txt`）。改 RDMA 后同节点 GEM 回环 13-29 GB/s（`results/mooncake-loopback-RDMA-v3-219.txt`），
  原生 `ib_write_bw` 主机内存回环单 QP 33 GB/s、4 QP 49 GB/s（`results/ib_write_bw-loopback-v3-219.txt`）。
- 确认方法：引擎日志里每个 rank 应有 `Found 8 HCAs` 和 `installTransport, type=rdma`，且没有 `couldn't load driver 'libionic-rdmav34.so'`。

### 3.3 GEM 显存被报成 GPU 0，dest-local-rail 把所有 KV 压到 ionic_0 → 预加载库修正设备号

- `hipPointerGetAttributes` 对 `hipImportExternalMemory` 导入的显存一律报 `device=0`（GPU 3 / GPU 5 上分配的 GEM 都报 0，普通显存正确；
  `results/gem-pointer-device.txt`）。分配器导入前已 `hipSetDevice`，是 HIP 对外部显存的返回问题。
- Mooncake `getMemoryLocation` 用这个字段给注册的缓冲区标位置 `hip:<device>`，Infera 的 `rdma_dest_local_rail.diff` 按位置选目的端网卡。
  结果每个 decode rank 的 KV 都标成 `hip:0`，全部写往 GPU 0 的网卡 ionic_0；GPU 4-7 在另一个 socket，正是 crsuse2 RCA 的"接收网卡跨 socket 写显存
  → 网卡内部丢帧"。修复前 P8D8 计分 95 min：ionic_0 发 122 TB、重传 46.7 亿包、ACK 超时 167 万，其余网卡几乎为 0；修复后 8 张网卡各 15-22 TB，
  ACK 超时合计约 15 万（`results/ionic-counter-deltas.txt`）。
- 修法：`patches/hip_gem_alloc-device.diff`——分配时记下 BO 所在设备，拦截 `hipPointerGetAttributes`，对 GEM 地址返回真实设备。
  验证要从全局符号表调用（ctypes `CDLL(None)`），直接从 `libamdhip64.so` 句柄取函数会绕过预加载。
- 影响面：AUS 上 PP8 + GEM 计分 TTFT +60% 未查明（DRIVER619 6.4 节），很可能是同一原因（AUS 网卡不丢帧，只拖慢没崩）。
  `../dp-graph-variant-fix-aus-20261103/driver619/delivery/` 的 Mooncake 原生 GEM 分配器要用登记的设备号定位置，不能信 `hipPointerGetAttributes`。

### 3.4 节点与 spur 作业语义

| 现象 | 原因 / 处理 |
|---|---|
| `docker load` 停在 17 MiB（v3-058）/ 510 MiB（v3-220），`docker images` / `info` / `version` 挂住 | containerd 卡死：`/_ping`、容器列表正常，经 containerd 的接口 8-20 s 超时；无内存 / IO 限额与压力，版本与正常节点相同（Docker 29.9.2、containerd 2.4.2），两台服务都是 11-09 重启后运行；v3-220 加载前健康检查是通过的。无 root 修不了，已排除，需 IT 重启 containerd（`results/diag-v3-058-vm-nic.txt`）。加载加 `timeout 1200` |
| 两节点作业开始几秒就结束（291051、291150、291162） | v3-159 没挂 /home（`stage_model.sh: No such file or directory`）；spur 的作业步骤默认在提交目录启动，**任何步骤非零退出（含起不来）都会结束整个作业**。处理：所有步骤 `srun -D /tmp`；sshshim 让步骤恒退出 0、真实退出码经共享目录文件传回（`patches/sshshim-step-exit-status.diff`）；作业开头逐节点查 /home 与 docker `images/json`，坏节点写入 `config/exclude_nodes.20261116.txt` 并自动重交（`scripts/hold2b.sbatch`、`hold1p.sbatch`，上限 6 次） |
| 作业内节点间 ssh 被拒；每节点只允许一个交互会话 | 作业内用 sshshim（`srun --overlap`）；批处理脚本常驻一个队列执行器，登录节点用 `q.sh` 投递命令，不另开会话 |
| 模型冷读 NFS：主权重 330-540 s、NextN 325 s，起栈约 20 min | 每次作业先拷模型到本地 NVMe（35-70 s，NFS 顺序读 6-12 GB/s），起栈 5 min |

### 3.5 查过但不需要的

- `HSA_ENABLE_SDMA=0`：yihou 在 v3 节点上遇到过权重加载卡在 SDMA 拷贝，本次 SDMA 开着加载正常，未设。
- rail：每张网卡一条三层路由的 rail（`fc01:{1..8}00::/24`），网卡名到 rail 的对应在 v3-058 与旧 136/137/138 上相同（ionic_0→8、1→7、2→5、3→6、
  4→4、5→3、6→1、7→2），同名即同 rail，镜像里的 dest-local-rail + `MC_IB_TIMEOUT=18` 适用。VM 的 PCIe 拓扑被压平（每个 NUMA 一条虚拟根总线），
  GPU 与网卡的物理就近关系不可见。
- `MC_TRANSFER_TIMEOUT`：走 RDMA 且设备号修正后没有传输超时，不需要加长。
- Mooncake #5005（`MC_HIP_STREAM_PRIORITY=high`）：只作用于 HIP 传输，本配置同节点也走 RDMA 回环，不相关。
- 本集群上 Mooncake 自带的 HIP 传输对普通显存取 IPC 句柄会随机失败（`hipIpcGetMemHandle invalid argument`）：GEM 方案下 HIP 路由本来就关着
  （`MC_DISABLE_HIP=1`，GEM 预加载库返回不可用句柄），不影响；若改走 HIP 传输需先解决。

## 4. 复现

```bash
R=/home/liyingli/bench_agentx/pd-repro-20261116          # 工作目录：pkg/（运行包）、scripts/、queue*/、logs/
# 运行包：验证包 + kit-router-hooks.diff + opt/config/*.env + 本目录 patches/（kit、sshshim、hip_gem_alloc.c）
#   + ptpc-fix/glm4_moe.py（首次运行由 run_item_v2.sh 从镜像生成）+ config/site.local.env.crusoe -> pkg/config/site.local.env
sbatch -x "$(paste -sd, $R/exclude_nodes.txt)" $R/hold2b.sbatch   # 两节点，队列 queue3/
$R/q3.sh p8 'export MODEL=/mnt/m2m_nobackup/liyingli_pdr/models/GLM-5.2-MXFP4; bash scripts/stage_model.sh; \
  ssh $NODE1 "bash $R/scripts/stage_model.sh"; bash scripts/chain_v2.sh p8d8 arm-C-1p1d-pp8-gemkv.env 208'
sbatch -x "$(paste -sd, $R/exclude_nodes.txt)" $R/hold1p.sbatch   # 单节点，队列 queue4/
$R/q4.sh p4 'export MODEL=...; bash scripts/stage_model.sh; MODE=1node bash scripts/chain_v2.sh p4d4 \
  arm-C-prod-ptpcfix-nohip-gemkv.env 104 $R/config-crusoe/dur3600.env'
```

`chain_v2.sh` = 门禁（smoke、200 题、16k × 64、真实接受长度；与 AUS 条目 36 相同的干净判据）→ 干净才 3600 s 计分（`pd_perf_chain.sh` +
`score_summary.sh`）。计分前后用 `scripts/ionic_counters.sh` 记网卡计数器，确认流量分布。

## 5. 文件

| 路径 | 内容 |
|---|---|
| `skill/crusoe-glm52-pd/SKILL.md` | 在本集群跑单节点 / 跨节点 PD 的坑与检查清单 |
| `patches/kit-no-host-libionic.diff` | kit：`KIT_NO_HOST_LIBIONIC=1` 不挂宿主机 libionic |
| `patches/hip_gem_alloc-device.diff`、`patches/hip_gem_alloc.c` | GEM 预加载库：`hipPointerGetAttributes` 返回真实设备（基于 `../dp-graph-variant-fix-aus-20261103/driver619/gemalloc/`） |
| `patches/sshshim-step-exit-status.diff` | sshshim：步骤从 `/tmp` 启动、恒退出 0、真实退出码经文件传回 |
| `config/` | 站点配置、GEM 配置、附加配置（`dur3600`、`sdma0`、`mctimeout300`）、排除节点 |
| `scripts/` | 持有作业（自检 + 自动换节点）、`q.sh`、条目脚本 `run_item_v2.sh` / `chain_v2.sh`、模型拷贝、节点信息、网卡计数器、prefill 分段统计、VM / 网卡诊断 |
| `scripts/driver619/` | 驱动记账复现（宿主机版）、Mooncake 同节点读写测试（扫描 / 并发 / 每网卡 / 参数 / 大小）、GEM 设备号测试、测试用 IPC 句柄替身 |
| `results/` | 计分、驱动复现、ibverbs 对照、回环测量、网卡计数器增量、prefill 分段、作业日志、`opt.out`（全部条目的摘要行） |

## 6. 建议

1. Infera 镜像入口：宿主机 libionic 在容器里 `dlopen` 不了就不要替换（现在替换后只剩一行 libibverbs 告警，网卡全丢）。本集群所有用这个镜像跑 PD 的作业都受影响。
2. Mooncake 原生 GEM 分配器（driver619 delivery）：GEM 缓冲区的位置按登记的设备号给出。
3. 报 IT：v3-058、v3-220 的 containerd 卡死；v3-159（可能还有 v3-169 / v3-171）没挂 /home。根本修法仍是驱动打上游补丁 `9131cc85`（driver619 README 第 4 节）。
