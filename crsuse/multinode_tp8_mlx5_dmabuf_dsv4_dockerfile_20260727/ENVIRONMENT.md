# 环境快照 — SGLang dma-buf Dockerfile

采集时间: 2026-07-27 (UTC),节点 crsuse2-m2m-{269,059,322,112,215}(多轮)

## 硬件

| 项 | 值 |
|----|----|
| 集群 | AMD `crsuse2-m2m`(Spur 调度器,单分区 `amd-spur`) |
| GPU | 8× **AMD Instinct MI355X**(gfx950, CDNA4),单卡 VRAM ~288 GiB |
| CPU | 236 逻辑核 |
| 内存 | ~2751 GB |
| 节点内核 | 6.8.0-107-generic(Ubuntu 24.04.4) |
| amdgpu 驱动 | 6.14.14 |
| ROCm(host) | 7.0.1 |

## RDMA fabric

节点上 9 个 RDMA 设备:8× AMD Pensando ionic + 1× Mellanox mlx5。

| 项 | mlx5_0 | ionic_0..7 |
|----|------|------|
| 厂商 | Mellanox ConnectX VF | AMD Pensando DSC VF |
| netdev | **ens3**(有 IP) | enP*(ip=none) |
| 驱动 | `mlx5_core` 24.10-3.2.5 | `ionic` 25.08.4.004 |
| 固件 | 28.43.3608 | 1.117.1-a-63 |
| 链路 | 200 Gb/s (4X HDR) | 400 Gb/s (4X NDR) |
| **ODP** | ✅ `ODP_SUPPORT`+`ODP_SUPPORT_IMPLICIT` | ❌ 无 |
| RoCEv2 GID | idx **3**(IPv4-mapped, routable) | idx **1**(`fc01:...` global) |

- ionic 需 `RDMAV_FORK_SAFE=1`(否则 `rdma_context setup failed: fork compatibility`)。

## 软件

| 项 | 值 |
|----|----|
| **base 镜像** | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| base sha256 | `sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| **编出镜像** | `dsv4-sgl-dmabuf:v1`(由 `Dockerfile.sglang.dmabuf` build) |
| 镜像内 ROCm | 7.2.0 |
| 镜像内 OS | Ubuntu 22.04(base) |
| bundled Mooncake | `/sgl-workspace/Mooncake` @ **01d1eb2a**(upstream #2682) |
| sglang | v0.5.15.post1 |
| Python | 3.10(`/opt/venv`) |

## Repo / 代码(★本轮核心产物)

- git 分支: `yihou.dev.sglang.mooncake.experiment`
- commit SHA: `f76be2f75e17f7456ec9ed44e06dbeb6e7f9cce7`
- **新增文件(真身在 repo,本交付含快照)**:
  - `deploy/docker/Dockerfile.sglang.dmabuf` —— dmabuf 变体镜像
  - `deploy/docker/scripts/build_mooncake_dmabuf.sh` —— 镜像内 Mooncake dmabuf 重编脚本
- **复用的仓内 patch**(Dockerfile 会 apply):
  - `deploy/docker/patches/mooncake_cpp/transfer_engine_impl.diff`(HIP-transport gate)
  - `deploy/docker/patches/mooncake_cpp/rdma_auto_chunk_mr_2017.diff`(超 max_mr_size 分块)
  - `deploy/docker/patches/mooncake_cpp/apply_mooncake_cpp_patches.sh`(applier)
- 实测:两个 patch 都能 `git apply --check` **clean** 到 base 的 Mooncake 01d1eb2a
  (虽 pin 在 747003c,但这两个文件上下文没漂移)。

## 模型(依赖的绝对路径 — 未提交 git)

- **路径**: `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`
- **大小**: 806 GB,64 shards;native `deepseek_v4`,FP8 attn + MXFP4 MoE
- **可见性**: 共享 NFS,每个计算节点都挂载
- **显存占用**:TP8 每卡 159 GiB 权重;**TP4 每卡 210 GiB**(TP4 需 `--mem-fraction-static ≥ 0.76`)

## 文件系统 / 挂载

| 挂载点 | 说明 |
|--------|------|
| `/shared_nfs` | 共享 NFS,模型在此 |
| `/home/yihou` | 用户 NFS home;脚本/日志/镜像 tar 中转 |

## 密钥 / 认证

- **本实验不需要任何密钥/token**:base 镜像公开 docker.io,模型在共享 NFS,spur 集群自带身份。
- 节点间传镜像走 `docker save → NFS tar → docker load`(**spur 禁普通用户 ssh 计算节点**)。

## 运行态服务配置

| 场景 | mem-fraction | TP | context | max-running-req | KV pool |
|------|------|------|---------|------|------|
| 2 节点 PD(TP8) | P 0.85 / D 0.90 | 8 | 9472 | 128 | ~5.8M/6.9M tok |
| 单机 TP4 P↔D | 0.85 | 4 | 9472 | 64 | 1.94M tok |

- 2 节点 decode 稳态 VRAM 237/288 GiB(= 159 权重 + 78 KV,**未翻倍**)。
- 单机 TP4 mlx5 稳态 243/288(210 权重 + 33 KV)。
