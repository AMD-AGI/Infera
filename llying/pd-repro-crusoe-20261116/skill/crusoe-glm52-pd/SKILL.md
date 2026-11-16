---
name: crusoe-glm52-pd
description: >-
  Pitfalls and checklist for running GLM-5.2 SGLang + Mooncake prefill/decode (PD) disaggregation on the crusoe spur
  cluster (crsuse2-m2m-v3-* MI355X VMs, ionic RDMA), one-node P4D4 and two-node P8D8. Use when deploying or debugging
  PD on crusoe/spur, when Mooncake logs "Found 0 HCAs" or KV transfers time out, when mem-fraction must be lowered
  because RDMA-registered VRAM is counted twice, when KV traffic piles onto one ionic NIC, or when a spur job ends
  seconds after it starts.
---

# GLM-5.2 PD on crusoe spur：坑与检查清单

来源：2026-11-16 在本集群复现 AUS 的 P8D8 C208（54,468 对 56,593）与单节点 P4D4 C104（53,351 对 53,209）。
证据、数字和补丁见 `../../README.zh-CN.md`（`llying/pd-repro-crusoe-20261116/`）。
下文的 `scripts/`、`patches/`、`config/` 都指 `llying/pd-repro-crusoe-20261116/` 下的目录。

## 开跑前清单

```
- [ ] 作业：所有 srun 步骤带 -D /tmp；作业内节点间命令走 sshshim（步骤恒退出 0）
- [ ] 节点体检：能看到 /home 工作目录；docker images/json 15 s 内返回
- [ ] 引擎容器不挂宿主机 libionic（KIT_NO_HOST_LIBIONIC=1）
- [ ] KV 用 GEM 分配，GEM 预加载库是带设备号修正的版本
- [ ] 模型拷到本地 NVMe 再起栈
- [ ] 起栈后：每个 rank "Found 8 HCAs" + "installTransport, type=rdma"
- [ ] 计分前后记 ionic 计数器，确认流量分到 decode GPU 对应的网卡上
```

## 1. 申请节点与 spur 作业语义

- 分区只有 `amd-spur-v2`。`amd-agentx-4-qos`：组上限 10 节点、**每人 2 节点**、最长 48 h，常满。`amd-burst-qos` 会被抢占取消，不要用来计分。
  其他账户的 QoS（如 `amd-primus-qos`）是否可用要先问用户。查自己能用哪些：`sacctmgr show user | awk '$1=="<user>"'`。
- spur 的 `sbatch` 没有 `--test-only`，`scontrol update` 改不了排除列表（要改就重交）。
- **任何作业步骤非零退出（包括因目录不存在起不来）都会结束整个作业**。作业步骤默认在提交目录启动，节点没挂 /home 就当场失败。
  - 所有 `srun` 加 `-D /tmp`，`sbatch` 加 `#SBATCH -D /tmp`。
  - 远程命令包一层：步骤内 `( cmd ); echo $? > <共享目录>/rc; exit 0`，本地读回退出码（本目录 `patches/sshshim-step-exit-status.diff`）。
- 作业内节点之间 `ssh` 被拒；每节点只允许一个交互会话（`srun --jobid --overlap` 走伪终端，输出带 CR，不继承环境）。
  做法：批处理脚本里常驻一个队列执行器，登录节点往共享目录投递命令（`scripts/hold2b.sbatch` + `scripts/q.sh`）；到另一节点用 sshshim。
- 作业的 PID 命名空间是独立的，看不到 dockerd；登录节点有内存和 24 h 会话限制，重任务都放到节点上。

## 2. 每次作业开始的节点体检

坏节点在本集群很常见（2026-11-16：10 台里 2 台 containerd 卡死、1-3 台没挂 /home）。作业开头逐节点检查，不合格就写入排除名单并重交：

```bash
test -r <工作目录>/q.sh && echo home-ok
curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json   # 要 200
cat /sys/module/amdgpu/version; rocm-smi --showmemuse    # 6.16.13；显存 0%
```

- **containerd 卡死**：`/_ping`、容器列表正常，`version` / `info` / `images/json` / `load` 挂住。可能在 `docker load` 开始后才卡（加载前检查通过也不保险），
  所以加载要 `timeout 1200`（正常约 6 min）。没有 root 修不了，换节点并报 IT。
- 起栈前等所有 GPU 显存 < 2%（拆栈后驱动释放要一会儿）。

## 3. 镜像与 RDMA：不要注入宿主机 libionic

- Infera 镜像入口会把挂到 `/host-libionic/libionic.so` 的宿主机库复制进容器。v3 宿主机（Ubuntu 24.04）的 libionic 要 **GLIBC_2.38**，
  镜像只有 2.35，加载失败后容器里没有 ionic 设备，**Mooncake 打印 `Found 0 HCAs` 然后用 TcpTransport，不报错**。
- 镜像自带的 libionic 与宿主机内核模块兼容：引擎容器不挂宿主机 libionic（kit 开关 `KIT_NO_HOST_LIBIONIC=1`）。自己写的测试容器也一样。
- 走 TCP 的样子：同节点传输 3.5-9 GB/s、与 Mooncake 参数无关；ionic `hw_counters` 不动；16k 以上请求 30 s 传输超时（HTTP 500）；跨节点 C208 预热时超时。
- 检查：

```bash
docker logs <engine> 2>&1 | grep -aoE "Found [0-9]+ HCAs|installTransport, type=[a-z]+|libionic-rdmav34" | sort | uniq -c
docker run --rm --device /dev/infiniband <image> ibv_devices      # 不挂宿主机库时应列出 ionic_0..7
```

## 4. 显存计两次：KV 用 GEM 分配

- amdgpu 6.16.13 上 dma-buf 注册的 KFD 显存被计两次（`kfd.vram_used` + `vram_pin_size`）；peer memory 注册不多扣，但**本集群没有 peer memory**
  （显存 `ibv_reg_mr` EFAULT）。所以 Mooncake 只能用 dma-buf（`MOONCAKE_DISABLE_HIP_DMABUF=0`），KV 一大就"假 OOM"，新建 stream 段错误。
- 确认：`scripts/driver619/repro_host.sh`（宿主机 ROCm 即可）——KFD + dma-buf 40 GiB 后余量少 80 GiB 就是这个问题。
- 修法：GEM KV（`config/gemkv.env.crusoe`）。mem fraction 保持 prefill 0.88 / decode 0.95。必需的环境：

```bash
LD_PRELOAD=<libhip_gem_alloc.so>  SGLANG_AMD_GEM_ALLOCATOR=<同一个 .so>  SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM
MOONCAKE_DISABLE_HIP_DMABUF=0  MC_HIP_KFD_PIN_FIX=0  MC_DISABLE_HIP=1      # 再挂改过的 mooncake/utils.py
```

- GEM 显存绝不能走 `ibv_reg_mr`（驱动 `amd_get_pages` 访问空 `kfd_bo` 会内核空指针）；预加载库已拒绝，别的传输库要先确认只走 dma-buf。
- 根本修法是驱动打上游补丁 `9131cc85`，需要管理员。

## 5. KV 传输与网卡分布

- 每张 ionic 一条三层路由的 rail，同名网卡跨节点同 rail（ionic_0→rail 8、1→7、2→5、3→6、4→4、5→3、6→1、7→2）。不同 rail 之间不通。
  接收端网卡在反压下会内部丢帧：镜像里的 `MC_ENABLE_DEST_LOCAL_RAIL=1` + `MC_ENABLE_DEST_DEVICE_AFFINITY=1` + `MC_IB_TIMEOUT=18` 必须开（kit 默认开）。
- **GEM 显存设备号**：HIP 对导入的外部显存一律报 `device=0`。Mooncake 据此把 KV 标成 `hip:0`，dest-local-rail 把**所有 KV 写往 ionic_0**
  （跨 socket 写 → 重传风暴 → 传输变慢 → prefill KV 被等传输的请求锁满 → `Prefill out of memory ... Available full tokens: 0` 崩溃）。
  预加载库必须拦截 `hipPointerGetAttributes` 返回真实设备（`patches/hip_gem_alloc-device.diff`）。验证用 ctypes `CDLL(None)` 调用，
  直接从 `libamdhip64.so` 句柄取函数会绕过预加载。
- 每次计分前后记 `scripts/ionic_counters.sh`，看 `tx_rdma_ucast_bytes` 是否分到各网卡、`tx_rdma_ack_timeout` / `tx_rdma_retx_pkts` 是否集中在一张。
  正常：P8D8 8 张各 15-22 TB / 小时量级；P4D4 在 ionic_4-7 上。
- `MC_TRANSFER_TIMEOUT` 不用加长；出现 30 s `Sync batch data transfer timeout` 先查上面三条（TCP、设备号、网卡分布）。

## 6. 单节点 P4D4

- `MC_DISABLE_HIP=1`：同机 KV 走同一张网卡的 RDMA 回环（两张不同网卡之间不通）。RDMA 正常时 GEM 回环 13-29 GB/s。
- 两个引擎同在一台机器、都 `--network host`：Infera readiness 端口 30090 只有一个能绑上，另一个报 `address already in use` 后不开这个端口继续运行，不影响服务和健康检查。
- Mooncake 自带 HIP 传输对普通显存取 IPC 句柄在本集群会随机失败；GEM 方案下 HIP 路由本来就关着，无影响。要改走 HIP 传输先解决这个。
- 计分要 `DURATION=3600`（`arm-C-prod-ptpcfix-nohip.env` 默认 1500 s，口径不同）。

## 7. 跨节点 P8D8

- prefill 节点 8 张卡一个 PP8，HiCache 8 × 250 GB（主机内存 2.75 TiB 够）；decode TP8/DP8。
- 计分链的守护逻辑看到引擎 `Fatal Python error` 会停掉 AgentX；没有 JSON 时先查 prefill / decode 日志里第一个异常。
- AIPerf 预热中任一根请求失败就中止整次运行（`warmup_failure`）。

## 8. 模型与起栈

- 从 `/shared_nfs` 冷读：主权重 330-540 s、NextN 325 s（加载器是零散的页读取，NFS 顺序读本身有 6-12 GB/s）。
  每次作业先拷到 `/mnt/m2m_nobackup/.../GLM-5.2-MXFP4`（`scripts/stage_model.sh`，35-70 s），`MODEL` 指向本地副本（两节点同路径），起栈约 5 min。
- `/shared_nfs` 约 99% 满，别往上写大文件；日志写 /home 或本地盘。
- `HSA_ENABLE_SDMA=0` 不需要（2026-11-16 SDMA 开着加载正常）；若权重加载停在 "loading shards 100%"、线程卡在 `memcpy_and_sync`，再加。

## 9. 判读与对齐

- 结果与 AUS 对比时看分项：decode 步长（按最忙 rank 请求数）、prefill 满块与现算吞吐、等预分配 / 等传输、TTFT。
- PP8 吞吐受 prefill 限，prefill 现算在 AUS 不同节点对之间就差 6%（20,453 对 19,276 tok/s）；用 `scripts/pp_stage_time.py`
  按 BOTTLENECK-PP8 4b 的口径（每批新 token / 已缓存 token、单段耗时、PP0 批次间隔）与 AUS 记录对照，不要只比满块中位数。
- 只在本集群重跑只能估单次波动，解释不了与 AUS 的差。
