# 环境快照 — DSv4 1P1D mlx5 + dmabuf

采集时间: 2026-07-24 (UTC),节点 crsuse2-m2m-294 / -059

## 硬件

| 项 | 值 |
|----|----|
| 集群 | AMD `crsuse2-m2m`(Spur 0.5.1 调度器,单分区 `amd-spur`) |
| Prefill 节点 P | crsuse2-m2m-294,data-plane IP(mlx5/ens3)= **10.245.156.178** |
| Decode 节点 D | crsuse2-m2m-059,data-plane IP(mlx5/ens3)= **10.245.150.73** |
| GPU | 8× **AMD Instinct MI355X**(gfx950, CDNA4),单卡 VRAM ~288 GiB |
| CPU | 236 逻辑核 |
| 内存 | ~2751 GB |
| 节点内核 | 6.8.0-107-generic(Ubuntu 24.04.4) |
| amdgpu 驱动 | 6.14.14 |
| ROCm(host) | 7.0.1 |

## RDMA fabric

节点上有 **9 个 RDMA 设备**:8× AMD Pensando ionic(400G RoCE,一 GPU 一张)+ 1× Mellanox mlx5。
**本实验强制只用 mlx5**(它是唯一有 IP 的网卡,且支持 ODP)。

| 项 | mlx5_0(本实验用) | ionic_0..7(未用) |
|----|------|------|
| 厂商/型号 | Mellanox ConnectX VF | AMD Pensando DSC VF |
| netdev | **ens3**(有 IP) | enP*(ip=none) |
| 驱动 | `mlx5_core` **24.10-3.2.5** | `ionic` 25.08.4.004 |
| 固件 | **28.43.3608** | 1.117.1-a-63 |
| 链路 | **200 Gb/sec (4X HDR)** | 400 Gb/sec (4X NDR) |
| link_layer | Ethernet(RoCEv2) | Ethernet(RoCEv2) |
| **ODP** | ✅ `ODP_SUPPORT` + `ODP_SUPPORT_IMPLICIT` | ❌ 无 |
| RoCEv2 GID index | **3**(IPv4-mapped, routable);idx1=fe80 link-local 勿用 | 3 |

- 跨节点 mlx5 连通实测(`ib_write_bw -d mlx5_0 -x 3`,P→D):**80 Gb/s peak**(200G 链路)。
- **为何用 GID idx3**:idx1 是 `fe80` link-local(跨节点会崩),idx3 是 `::ffff:10.245.x` 的
  IPv4-mapped RoCEv2,可路由。两节点都在 `10.245.x/20`,同段可达。

## 软件

| 项 | 值 |
|----|----|
| Docker | 每节点自带 dockerd(root dir `/mnt/m2m_nobackup/docker`) |
| **base 镜像** | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| base 镜像 sha256 | `sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| **编出的镜像** | `dsv4-sgl-dmabuf:mlx5` |
| 编出镜像 sha256 | `sha256:8746f670edac0166fad3ce071301af9110e82e7cf22827a805030e31cc5a7a5b` |
| 镜像内 ROCm | 7.2.0 |
| 镜像内 OS | Ubuntu 22.04(base)——注意 host 是 24.04,glibc 不同 |
| bundled Mooncake | 源码 `/sgl-workspace/Mooncake` @ **01d1eb2a**(upstream #2682) |
| Python | 3.10(`/opt/venv`) |

**镜像来源说明**:编出的镜像**不入 git**(27GB)。它 = base 镜像 + `scripts/build_mc_dmabuf.sh`
在容器内重编 Mooncake(启用 `USE_HIP_DMABUF` + `MC_ENABLE_HIP_TRANSPORT` gate)后 `docker commit`。
用 REPRODUCE 步骤可从 base 完整重建。

## Repo / 代码

- git 分支: `yihou.dev.sglang.mooncake`
- commit SHA: `f0938ba1c20ecf53481381e901683973e5b70807`
- 参考(对拍源): legacy `/home/yihou/dev/git/legacy.infera/infera/pd_1p1d_dpa_8k1k_20260714_235121`
- 相关仓内文件(理解 dmabuf 决策):
  - `deploy/docker/Dockerfile.sglang`(HIP-transport gate)
  - `deploy/docker/scripts/patch_mooncake_sglang.sh`
  - `deploy/docker/scripts/build_mooncake_rocm.sh`
  - `deploy/docker/patches/mooncake_cpp/apply_mooncake_cpp_patches.sh`(B.1 dmabuf 被 drop 的原因)

## 模型(依赖的绝对路径 — 未提交 git)

- **路径**: `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`
- **类型**: 原生 `deepseek_v4`,FP8 attn + MXFP4 MoE
- **大小**: 806 GB,64 个 safetensors shard
- **可见性**: 在共享 NFS,**每个计算节点都挂载**,直接绝对路径挂进容器,免拷贝
- **KV dtype**: fp8_e4m3(server 自动)

## 文件系统 / 挂载

| 挂载点 | 说明 |
|--------|------|
| `/shared_nfs` | 共享 NFS,模型在此;所有节点一致 |
| `/home/yihou` | 用户 NFS home;跨节点共享;脚本/日志在此;镜像 tar 中转也走这 |

## 密钥 / 认证

- **本实验不需要任何密钥/token**:base 镜像从公开 docker.io 拉,模型在共享 NFS,
  spur 提交用集群自带身份。
- 节点间传镜像用 `docker save → NFS tar → docker load`(**spur 禁止普通用户 ssh 计算节点**,
  所以不能用 `docker save | ssh | docker load` 流式,改走 NFS 中转)。

## 运行态服务配置(bench 时)

| 参数 | prefill | decode |
|------|---------|--------|
| mem-fraction-static | 0.85 | 0.90 |
| context-length | 9472 | 9472 |
| max-running-requests | 128 | 128 |
| page-size | 256 | 256 |
| chunked-prefill-size | 8192 | 8192 |
| 实际 KV 容量 max_total_num_tokens | 5,804,800 | 6,887,168 |
| KV 分配后空闲 available_gpu_mem | 63.46 GB | 50.70 GB |
| TP | 8 | 8 |

- decode 稳态 VRAM ~237/288 GiB/卡 = 159(权重)+ ~78(KV+buffer)。**未翻倍**
  (若 pin 则需 159+2×78=315 > 288 → OOM;实际没 OOM,证明 dmabuf dynamic attach 生效)。
- router: `sglang_router --pd-disaggregation`,port **8200**(8100 被节点上他人进程占了)。
