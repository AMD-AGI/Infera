# PD 分离特有坑 / 踩坑记录(what / why / how / context)

> 本文件只记 **PD 分离特有** 的坑。基础 spur 潜规则(JobHoldMaxRequeue、burst-qos、
> 绝对路径挂载、docker exec -d、冷启动判活等)见上一级 `../../spur_repro/GOTCHAS.md`,不重复。

---

## ★坑 #1(最坑):跨节点 mooncake RDMA `Failed to register memory [12]`(ENOMEM)

**What**: 跨节点 PD 用 `--disaggregation-transfer-backend mooncake`(RDMA)时,server
起来后每个 rank 在 mooncake TransferEngine init 阶段刷:
```
E rdma_context.cpp:243] Failed to register memory 0x...: Cannot allocate memory [12]
```
权重能加载完,但两端**卡在 KV 池 init,永远不 ready**(GPU ~10% util、VRAM 冻结)。

**Why**: mooncake 要把大块 GPU KV 缓冲注册成 RDMA memory region(MR),向 ionic 网卡
pin 内存。`[12]`=ENOMEM。**注意不是 memlock 限制** —— 容器内 `ulimit -l` 已 unlimited
(`--cap-add=IPC_LOCK --ulimit memlock=-1`)。小 MR 的 `ib_write_bw` rail test 明明能
跑到 213.7 Gb/s,所以是 mooncake 的**大 GPU-buffer 注册**这一步失败,疑似需要 dmabuf
路径 / MR-cache / slice 参数,或某个 host 级 RDMA pinned-page 上限。

**How(绕过,已验证)**: 换 `--disaggregation-transfer-backend mooncake_tcp`
→ **0 注册报错**,TP8 两端干净加载。KV 改走 ens3 TCP。功能可用(DSv4 是 MLA、KV 极小,
RDMA 非吞吐瓶颈,legacy 已证换后端 <2%),但不是 RDMA 快路。
**How(彻底解,未做)**: 调 mooncake 环境变量 —— `.so` 里有 `MC_SLICE_SIZE`、
`MC_MAX_WR/CQE/SGE`、`MC_FRAGMENT_RATIO` 等旋钮;或查 mooncake 对 ROCm dmabuf GPU MR
的支持要求。留作后续。

**Context**: 节点内 MVP 用 `mooncake_tcp` 走 loopback 就完全避开了这个坑(零 RDMA);
所以先在单机把 PD wiring 全跑通,跨节点才暴露此问题。证据:`logs/xnode_*.log`(mooncake,
含 ENOMEM)对照 `logs/xtcp_*.log`(mooncake_tcp,干净)。

---

## ★坑 #2:两个通信层别混 —— TP 用 RCCL,PD KV 用 mooncake(不是一回事)

**What**: 容易把"PD 传输后端"和"TP 集合通信"混为一谈,以为选了 mooncake_tcp 就"没用 RCCL"。

**Why / 澄清**:
- **TP 组内**(prefill 的 4 rank、decode 的 4 rank 各自 all-reduce/gather)一直走
  **RCCL**(日志 `sglang is using nccl==2.27.7`,ROCm 上这个 .so 就是 RCCL),经 XGMI
  卡间直连。**与 transfer-backend 无关**。
- **PD 的 KV handoff**(prefill→decode)走 **mooncake**(一个独立 transfer engine,
  不是集合通信库)。本轮让它走 `mooncake_tcp`(loopback TCP);换 `mooncake` 是 RDMA,
  但仍不是 RCCL。

**How**: 分析/汇报时永远讲清"哪一层":TP=RCCL/XGMI,PD-KV=mooncake(_tcp)。

---

## ★坑 #3:节点内 PD 为啥走 tcp + 够不够快

**What**: 节点内 1P1D 选了 `mooncake_tcp` 而非 RDMA,会被问"够快吗、为啥不用 RDMA"。

**Why**: 是 **MVP 求正确避坑** 的选择,不是榨吞吐:mooncake RDMA 要注册 ionic MR
(见坑#1),节点内不想碰这套;`mooncake_tcp` 走 127.0.0.1 零 RDMA 配置就把 PD wiring
跑通。**够快吗**:功能够、非最优 —— loopback TCP 是 GPU→host 内存→内核 TCP→host→GPU,
有拷贝开销,不是 GPU 直连。但没成为瓶颈,因为 **DSv4 是 MLA、KV 被压得极小**
(~43KB/token,8k 请求 ~352MB,且 handoff 每请求只传一次)→ 这条慢路对整体吞吐影响很小。

**How(真要压性能)**: 节点内应上 mooncake RDMA loopback,或更理想 GPU P2P/XGMI 直传,
省掉 host 往返。**但其实节点内做 PD 分离对吞吐本身没意义** —— 那轮目的就是上跨节点前
先在单机验证 wiring。

---

## ★坑 #4:login-node NSS 抽风 —— 阻塞 sbatch 和 scancel

**What**: `whoami` → `cannot find name for user ID 50112975: Connection refused`。
`sbatch` 报 `no account for user 'unknown'`;`scancel` 报 `user unknown cannot cancel
job owned by yihou`。

**Why**: login 节点的名字解析后端(NSS/LDAP/sssd)拒连,spurd 无法把 UID→username 映射。
用户是 NSS-only(不在 /etc/passwd),`getent passwd <uid>` 也空。身份在 **spurd 服务端**
校验,客户端 LD_PRELOAD/nss_wrapper 改不了(且本机没装 nss_wrapper)。

**How**:
- **已占的 job / spur exec 不受影响**(用 job 自身身份,非 submit-time 名字查询)——
  实验照跑。
- 新 submit / cancel 只能**等 NSS 恢复**(本次实测故障持续 >5h),或换一个 login 节点。
- **兜底**:job 会在 walltime(本例 8h)到点**自动过期释放**。本次两个 job 就是靠
  walltime 自动清的(watcher 等不到 NSS,但 GPU 早已 docker rm 释放,只剩账面过期)。
- **Lesson**:`sbatch`/`scancel` 报 `user 'unknown'` 时先 `whoami` 一眼确认是 NSS 故障,
  别改 flag 瞎试。

**Context**: 本轮抢第 2 节点和收尾 scancel 都撞上它。GPU 实际释放靠 `docker rm -f` +
`pkill launch_server`,与 slurm 账面解耦。

---

## ★坑 #5:跨节点第 2 节点缺镜像 —— 不能 ssh save|load,用 retry-loop 拉

**What**: 抢到的第 2 节点常没有 mori-0615 镜像(每节点 docker 本地,不共享)。

**Why**: legacy 用 `ssh <有镜像节点> 'docker save' | ssh <目标> 'docker load'`,但 **spur
compute 节点 ssh 被封**(见 ../GOTCHAS.md)。且节点 egress 到 docker.io cloudfront
偶发 `i/o timeout`(单次 pull 会中途失败)。

**How**: `scripts/do_pull.sh` —— 循环重试 `docker pull` 直到成功(共享 NFS 上的
"Already exists" 层会复用,只补缺的)。`setsid ... < /dev/null &` detached 起,靠
`docker images` 出现 tag 判完成(别靠 pgrep,setsid 的客户端在别的 session 看不到)。

**Context**: 本轮 260 节点用 do_pull.sh 拉齐 mori-0615(sha 与 292 一致 976831ec7f19)。
