# 环境快照 — DSv4 + SGLang PD 分离 on spur

采集时间: 2026-07-23 (UTC)。节点 crsuse2-m2m-292(P)、crsuse2-m2m-260(D,跨节点用)。

> 硬件/软件基础与上一级单机 R4 交付一致(同集群、同镜像、同模型),此处只补 **PD 特有**
> 的差异(第二节点、RDMA fabric)。基础项详见 `../../spur_repro/ENVIRONMENT.md`。

## 硬件

| 项 | 值 |
|----|----|
| 集群 | AMD `crsuse2-m2m`(Spur 0.5.1 调度器,单分区 `amd-spur`) |
| 节点(本实验) | crsuse2-m2m-292(P/节点内),crsuse2-m2m-260(D,跨节点)。节点可替换 |
| GPU | 每节点 8× **AMD Instinct MI355X**(`gfx950`, CDNA4) |
| 单卡显存 | ~288 GB;TP4 时权重 210GB/卡、TP8 时 159GB/卡 |
| CPU | AMD EPYC(236 逻辑核) |
| 内存 | ~2.75 TB |
| amdgpu 驱动 | 6.14.14;host ROCm 7.2.0 |

## RDMA fabric(RoCE, ionic)—— PD 跨节点相关

spur **与 legacy chi 集群同款 ionic RoCE**(实测,见 `logs/` + 原始 `RECON.md`):

- `ibv_devices`:**ionic_0..ionic_7**(Pensando RoCE)+ **mlx5_0**(Mellanox)。
- `rdma link`:所有 ionic_N/1 state ACTIVE、physical LINK_UP。
  netdevs = enP2p0s9-12 / enP3p0s9-12(同 index 即同轨)。
- **GID index**:idx0 = `fe80::…` link-local(会崩,勿用);idx1 = `fc01:…` 全局
  RoCEv2 → **`MC_GID_INDEX=1`**(与 legacy 一致)。ionic 是 **IPv6-only GID**(无 IPv4)。
- **单轨实测吞吐**:`ib_write_bw -d ionic_0 -x 1 --report_gbits` 292↔260 =
  **213.7 Gb/s**(GID1,RoCEv2)→ fabric 健康,跨节点 KV 传输有充足带宽。
- **容器免 ionic 注入**:mori-0615 镜像已自带可用 ionic provider,容器内
  `ibv_devinfo` 直接见 9 个 PORT_ACTIVE。**不需要 legacy 的 `inject_ionic.sh`**
  (这是 spur 相对 legacy stock 镜像的简化)。

**双平面拓扑**(跨节点 PD 用):
- **控制/bootstrap 面** = `ens3`(292=10.245.146.70,260=10.245.144.92,同 /20,
  ping 0.47ms)。承载 sglang 进程通信 + PD TCP bootstrap。
- **数据面** = ionic RoCE(IPv6 GID,KV RDMA 传输)。

单机(节点内)PD **不涉及 RDMA/ionic** —— `mooncake_tcp` 走 loopback。

## 软件

| 项 | 值 |
|----|----|
| Docker | 每节点自带 dockerd(root dir `/mnt/m2m_nobackup/docker`) |
| 镜像 | `rocm/sgl-dev:sglang-0.5.13.post1-rocm720-mi35x-mori-0615` |
| 镜像 sha256 | `sha256:976831ec7f1976bb0ff4d469600e38546549e60a4dec7e5148e853694976e387` |
| SGLang | `0.0.0.dev14036+g19c78552d.d20260615`(镜像自带) |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4`;镜像内 ROCm 7.2.0;Python 3.10 |
| **PD 能力** | `--disaggregation-mode {null,prefill,decode}`;transfer-backend `{mooncake, nixl, ascend, fake, mori, mooncake_tcp}` |
| transfer engine | `import mooncake` OK、`import mori` OK(`nixl` 缺);本轮用 **mooncake_tcp** |
| router | `sglang_router 0.3.2`(mini-lb PD 路由,同 legacy) |
| 集合通信(TP) | **RCCL**(日志 `sglang is using nccl==2.27.7`) |

**镜像选择理由**:与单机 R4 成功实验完全同款镜像(dsv4 backend + fused-compress env
版本敏感)。跨节点第 2 节点(260)镜像用 `do_pull.sh` retry-loop 从 docker.io 拉
(节点 egress 到 cloudfront 偶发 timeout,循环重试即可;不能 ssh 做 save|load)。

## 模型(依赖的绝对路径 — 未提交到 git)

- **路径**: `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(所有节点挂载)
- **类型**: 原生 `deepseek_v4`,FP8 attn + MXFP4 MoE("fp4");806 GB,64 shards。

## 密钥 / 认证

- **本实验不需要任何密钥/token**:公开 docker.io 镜像 + 共享 NFS 模型 + spur 自带身份。
