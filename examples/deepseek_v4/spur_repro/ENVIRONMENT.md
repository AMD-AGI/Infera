# 环境快照 — DSv4 + SGLang 单机 on spur

采集时间: 2026-07-23 (UTC), 节点 crsuse2-m2m-292

## 硬件

| 项 | 值 |
|----|----|
| 集群 | AMD `crsuse2-m2m`(Spur 0.5.1 调度器,单分区 `amd-spur`) |
| 节点 | crsuse2-m2m-292(本次实验;节点是可替换的,见坑#1) |
| GPU | 8× **AMD Instinct MI355X**(`gfx950`, CDNA4),Card Model 0x75a3 |
| 单卡显存 | ~288 GB(VRAM Total 309,220,868,096 B);权重占 159GB/卡,余 ~117GB 给 KV |
| CPU | 236 逻辑核 |
| 内存 | ~2751 GB |
| 节点内核 | 6.8.0-107-generic |
| amdgpu 驱动 | 6.14.14 |
| ROCm(节点) | 7.2.0 |

RDMA/IB:本实验是**单机**,不涉及 RDMA/InfiniBand/ionic。PD 分离(跨机 KV 传输)才需要,
见 infera 仓库 `examples/deepseek_v4/engine/pd_*`。

## 软件

| 项 | 值 |
|----|----|
| Docker | client=server **29.6.1**(每节点自带 dockerd,root dir `/mnt/m2m_nobackup/docker`) |
| 镜像 | `rocm/sgl-dev:sglang-0.5.13.post1-rocm720-mi35x-mori-0615` |
| 镜像 sha256 | `sha256:976831ec7f1976bb0ff4d469600e38546549e60a4dec7e5148e853694976e387` |
| SGLang | `0.0.0.dev14036+g19c78552d.d20260615`(镜像自带,与 legacy manifest 一致) |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| ROCm(镜像内) | 7.2.0 (rocm720) |
| Python | 3.10(`/opt/venv`) |

**镜像选择理由**:与 legacy 成功实验(2026-07-07 chi2865)**完全同款镜像**。DSv4 的
`--attention-backend dsv4` 与 fused-compress env 都是版本敏感的,用同镜像才能干净对拍。
镜像可从 docker.io 直接 `docker pull`(节点已验证可拉,约 100GB+)。

备选镜像(节点本地已有,未使用):`primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`;
infera 基础镜像 `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`(可拉)。

## 模型(依赖的绝对路径 — 未提交到 git)

- **路径**: `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`
- **类型**: 原生 `deepseek_v4`,`expert_dtype=fp4` + `quant_method=fp8`(= InferenceX "fp4":
  FP8 attention + MXFP4 MoE);`config.json` 确认 `model_type=deepseek_v4`。
- **大小**: 806 GB,64 个 safetensors shard。
- **可见性**: 在 `/shared_nfs`(NFS filer 172.27.255.2),**每个计算节点都挂载**,免拷贝。
  (注:legacy 用的是 `-Pro-fixed` 变体路径 `/mnt/vast/...`,本集群用上面的 spur 路径。)

## 文件系统 / 挂载

| 挂载点 | 说明 |
|--------|------|
| `/shared_nfs` | 共享 NFS(100T),模型在此;所有节点一致,直接绝对路径挂进容器 |
| `/home/yihou` | 用户 NFS home(10T);跨节点共享;脚本/日志在此 |

## 密钥 / 认证

- **本实验不需要任何密钥/token**:镜像从公开 docker.io 拉取,模型已在共享 NFS,
  spur 提交用集群自带身份(无需额外凭证)。
- 若要拉私有镜像(如 `harbor.crusoe.primus-safe.amd.com/...`)才需 docker login,本次未用到。
