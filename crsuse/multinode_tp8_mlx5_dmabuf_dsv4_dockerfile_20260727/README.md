# SGLang + Mooncake **dma-buf** Dockerfile —— 制作、验证、传输方式对比

**时间**: 2026-07-26 ~ 2026-07-27 (UTC)
**执行**: yihou @ AMD `crsuse2-m2m`(amd-spur)
**一句话**: 交付一个**正式的 Dockerfile**(`deploy/docker/Dockerfile.sglang.dmabuf`),把 base sglang
镜像的 Mooncake 重编成**启用 `ibv_reg_dmabuf_mr`**(GPUDirect,无 peermem 时唯一可行);并端到端验证
了 2 节点 mlx5+dmabuf PD 跑通,同时系统对比了单机 TP4 P↔D 的三种 KV 传输方式。

> ⚠️ **与 `crsuse/` 内其它交付区分**:
> - 本目录 `multinode_tp8_mlx5_dmabuf_dsv4_dockerfile_20260727/` = **Dockerfile 方案**(可复现构建)+ 传输对比。
> - `multinode_tp8_mlx5_dmabuf_dsv4_manualcommit_20260724/`(前一轮)= 手动 `docker commit` 镜像的方案(非 Dockerfile)。
> - `../spur_repro/` = 更早的单机 mix bring-up。
> 三者独立,勿混。

---

## 任务(用户口头 spec)

> 根据前面对 dmabuf 开关的分析,写一份 Dockerfile 放在 `deploy/docker/Dockerfile.sglang.dmabuf`,
> **只用于启动 `ibv_reg_dmabuf_mr` 运行的情况**,Mooncake 依然用镜像内源码重编的方式;apply 已有的
> `patches/mooncake_cpp` 那两个 patch(若非 no-op 且不影响现有功能)。然后**严格按此 Dockerfile
> build 一个镜像,2 节点 mlx5+dmabuf PD 分离测通**。再追加:单机 TP4 P↔D 分别试 mlx5 / ionic /
> 是否需要开 `MC_ENABLE_HIP_TRANSPORT`。

## 成果一句话

- ✅ `Dockerfile.sglang.dmabuf` 经**真实 `docker build`** 验证可用;编出 `dsv4-sgl-dmabuf:v1`,
  `DMABUF_COMPILED_IN=yes`。
- ✅ **2 节点 mlx5 + dmabuf PD** 跑通:smoke Paris/4/Jupiter,KV 不翻倍(237/288),全程 mlx5 RDMA、零 ionic。
- 📊 **单机 TP4 P↔D 三方式对比**(见下表):mlx5 RDMA loopback 是唯一稳定可行的。

---

## 结论表(单机 TP4 P↔D,同节点同脚本,只换传输)

| 传输方式 | 结果 | 关键点 |
|----------|------|--------|
| **hip transport** (`MC_ENABLE_HIP_TRANSPORT=1`) | ❌ 不通 | hip 装上、数据面零网卡,但 `hipIpcOpenMemHandle` 打不开对方**独立进程**的 GPU handle → 全部 `address not found` |
| **mlx5 RDMA loopback**(不开 hip) | ✅ 稳定 | smoke 全对,rdma on mlx5_0,KV 不翻倍。**推荐** |
| **ionic RDMA loopback**(不开 hip) | ⚠️ 不稳 | 第 1 个请求 OK,之后 mooncake session 死。ionic 无 ODP 的已知问题 |

## dma-buf 在不同 NIC 上的行为(核心洞察,贯穿全部实验)

| | mlx5(有 ODP) | ionic(无 ODP) |
|--|------|------|
| `ibv_reg_dmabuf_mr` | dynamic attach → **不 pin、不翻倍** | 强制 **pin 整块** → 翻倍/耗 KFD |
| 2 节点 TP8 PD | ✅ 跑通(KV 237/288) | ❌ 崩(node322 在 KV 注册步 SIGSEGV,已独立复现) |
| 单机 TP4 PD | ✅ 稳 | ⚠️ 1 次后 session 死 |

## 交付目录结构

```
sglang_dmabuf_dockerfile_20260727/
├── README.md          ← 本文件(索引 + 总结 + 结论表)
├── ENVIRONMENT.md     ← 硬件/软件/镜像sha/git/RDMA/模型/密钥
├── REPRODUCE.md       ← 从 docker build 到 2节点PD + TP4三方式 的严格复现
├── GOTCHAS.md         ← ★dmabuf 开关 what/why/how + 全部踩坑/弯路/错误分析
├── scripts/
│   ├── Dockerfile.sglang.dmabuf.copy   ← 交付快照(真身在 deploy/docker/)
│   ├── build_mooncake_dmabuf.sh.copy   ← dmabuf 重编脚本快照(真身在 deploy/docker/scripts/)
│   ├── start_ctr_mlx5.sh               ← 起容器
│   ├── pd_server_mlx5.sh               ← 2节点 PD leg(强制 mlx5)
│   ├── pd_server_tp4_rdma.sh           ← 单机 TP4 leg(参数化 NIC: mlx5/ionic, 走 RDMA)
│   ├── pd_server_tp4_hip.sh            ← 单机 TP4 leg(MC_ENABLE_HIP_TRANSPORT=1)
│   ├── repro_ionic_oom.sh              ← ionic dmabuf pin/崩 复现器
│   └── probe_and_repro.sh              ← 抢节点+复现 watcher(每10min tick)
├── results/
│   ├── tp4_VERDICT.txt                 ← 单机 TP4 三方式对比的完整判定
│   └── v1_pd_smoke_result.json         ← 2节点 PD smoke: "Paris"
└── evidence/
    ├── docker_build_test.log           ← docker build 全过程(DMABUF_COMPILED_IN=yes)
    ├── v1_{prefill,decode,router}.log  ← 2节点 mlx5 PD server 日志
    ├── tp4mlx5_{prefill,decode}.log    ← 单机 TP4 mlx5(成功)
    ├── tp4ionic_{prefill,decode}.log   ← 单机 TP4 ionic(session 死)
    ├── tp4hip_{prefill,decode}.log     ← 单机 TP4 hip(address not found)
    ├── ionic_2node_crash_node322.log   ← 2节点 ionic 崩溃复现(mooncake conn.py SIGSEGV)
    └── ionic_repro_watch.log           ← watcher tick 记录
```

## 阅读顺序

1. **`GOTCHAS.md`** —— dmabuf 开关到底怎么回事 + 所有坑/弯路(最有价值)。
2. `ENVIRONMENT.md` —— 镜像 sha / git / RDMA / 模型路径。
3. `REPRODUCE.md` —— 从 `docker build` 照着复现。

## 关键环境速记(详见 ENVIRONMENT.md)

- base 镜像: `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (sha256:**40e940a0…**)
- 编出镜像: `dsv4-sgl-dmabuf:v1`(由 Dockerfile.sglang.dmabuf build)
- bundled Mooncake: `/sgl-workspace/Mooncake` @ **01d1eb2a**(upstream #2682)
- git: 分支 `yihou.dev.sglang.mooncake.experiment` @ `f76be2f`
- GPU 8× MI355X(gfx950),ROCm 7.2.0(容器);RDMA: 8× ionic(无ODP)+ 1× mlx5(有ODP)
- **不需要任何密钥**(公开 base 镜像 + 共享 NFS 模型 + 集群自带身份)

## 未提交/依赖

- 模型 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(806G,不入 git)
- 编出的镜像 tar(27G,**不入交付**;用 Dockerfile 重建,见 REPRODUCE)
- 原始 workspace `temp/pd_mlx5_1p1d/`(gitignored,**保留未删**)
