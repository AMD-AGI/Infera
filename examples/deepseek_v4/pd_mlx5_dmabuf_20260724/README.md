# DSv4 1P1D 跨节点 PD — 强制单张 mlx5 + dmabuf(KV 零 pin 不翻倍)

**时间**: 2026-07-24 (UTC)
**执行**: yihou @ AMD `crsuse2-m2m`(amd-spur)
**结论**: ✅ 成功。DeepSeek-V4-Pro 用 SGLang + Mooncake 在**两个 8×MI355X 节点**上跑通
1P1D PD 分离,**KV cache 传输全部强制走单张 mlx5 网卡**,并把 Mooncake 重编成
**启用 `ibv_reg_dmabuf_mr`**——利用 mlx5 的 ODP 做 dynamic attach,**KV 池注册零 pin、
不翻倍**。smoke 正确(Paris/4/Jupiter),bench_serving 扫到 conc=128 峰值 **26.6k tok/s**。

---

## 任务(用户下发 spec,口头)

> 参照 legacy 的 `pd_1p1d_dpa_8k1k_20260714_235121`,**强制指定用 mlx5 网卡支持 8 卡通讯**,
> 跑通跨节点 1P1D 的 dsv4 sglang mooncake。镜像用 `deploy/docker/Dockerfile.sglang` 现编
> (阅读 patch,理解运行时使用 `ibv_reg_dmabuf_mr` 的正确开关)。模型
> `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`。
>
> 背景:1. inference 中大模型卡间通讯带宽极低,回退可接受。2. KV cache 一般几十~上百 GB,
> **注册后翻倍会直接导致整卡溢出不可用**,问题更严重。

后续用户又追加要求:**改用 base 镜像直接改并编译,打开 dmabuf 开关,用原生 sglang 跑
(不走 infera)**;并做 bench_serving 扫 conc=1/16/32/64/128/256。

## 成果一句话

Mooncake 的 dmabuf 路径(`ibv_reg_dmabuf_mr`)在 base 镜像里**默认被编译掉了**(见下)。
本实验把它**编译回来**并**强制走有 ODP 的 mlx5**,于是 KV 池按 dmabuf **dynamic attach**
注册——**不 pin、不翻倍**。decode 稳态 237/288 GiB(=159 权重 + 78 KV,而非 159+2×78 溢出)。

## 为什么强制 mlx5(而不是默认的 8×ionic)

| | ionic(默认数据面) | **mlx5(本实验强制)** |
|--|------|------|
| ODP | ❌ 无 → dmabuf 会 **pin 整个 KV 池** → 翻倍 → KFD 耗尽(HIP-209) | ✅ 有 `ODP_SUPPORT` → dmabuf **dynamic attach** → 零 pin |
| 带宽 | 8×400G RoCE | 单张 200G(回退,用户接受) |
| 数量 | 8 张(一 GPU 一张) | 1 张(全 8 GPU 共用) |

用户明确"带宽回退可接受、KV 翻倍不可接受",所以选 mlx5——**拿带宽换显存安全**。

## 交付目录结构

```
pd_mlx5_dmabuf_20260724/
├── README.md          ← 本文件(索引 + 总结)
├── ENVIRONMENT.md     ← 硬件/软件/镜像sha/git版本/RDMA配置/模型路径/密钥
├── REPRODUCE.md       ← 从零到 smoke + bench 的严格复现步骤
├── GOTCHAS.md         ← ★dmabuf 开关分析 + spur/PD 踩坑(本次最有价值产出)
├── scripts/
│   ├── build_mc_dmabuf.sh   ← ★把 Mooncake 重编成启用 dmabuf(核心 fix)
│   ├── start_ctr_mlx5.sh    ← 起 PD 容器
│   ├── pd_server_mlx5.sh    ← 原生 sglang PD leg(prefill|decode),强制 mlx5
│   └── bench_sweep_mlx5.sh  ← bench_serving 扫并发
├── results/
│   ├── smoke_result.json    ← smoke: capital of France → "Paris"
│   ├── BENCH_SUMMARY.txt     ← conc 1..256 汇总表
│   └── bench/mlx5_c{1,16,32,64,128,256}.jsonl  ← bench 原始
└── evidence/
    ├── prefill.log / decode.log  ← server 全量日志(含 RDMA device mlx5_0 GID3)
    ├── router.log                ← PD router
    ├── build.log / build2.log    ← dmabuf 重编(build2 = 加 CMake 传播后成功)
    └── sweep.log                 ← bench 扫描全量输出
```

## bench_serving 结果(KV over 单张 mlx5)

ISL=4096 OSL=512(<ctx 9472),`--random-range-ratio 1.0`,经 router。

| conc | tot tok/s | out tok/s | req/s | TTFT_med | TPOT_med |
|-----:|----------:|----------:|------:|---------:|---------:|
| 1 | 679 | 75 | 0.15 | 287ms | 12.63ms |
| 16 | 7,529 | 837 | 1.63 | 557ms | 17.23ms |
| 32 | 12,574 | 1,397 | 2.73 | 532ms | 20.16ms |
| 64 | 19,015 | 2,113 | 4.13 | 557ms | 25.64ms |
| 128 | **26,666** | 2,963 | 5.79 | 4,175ms | 29.02ms |
| 256 | 26,588 | 2,954 | 5.77 | 23,184ms | 29.03ms |

吞吐线性爬到 conc=128(server `max_running_requests=128` 上限);conc=256 超过软上限 →
排队,TTFT 从 4.2s 涨到 23s,吞吐持平。**单张 mlx5 的 KV 传输未成为瓶颈**(ITL 曲线平滑)。

## 阅读顺序

1. **`GOTCHAS.md`** — dmabuf 开关到底怎么回事(base 镜像为何编译掉、怎么修回来)+ spur/PD 坑。
2. `ENVIRONMENT.md` — 确认环境(镜像 sha、git、RDMA、模型路径)。
3. `REPRODUCE.md` — 照着复现。

## 关键环境速记(详见 ENVIRONMENT.md)

- 节点: prefill=crsuse2-m2m-294 (10.245.156.178) ↔ decode=crsuse2-m2m-059 (10.245.150.73)
- GPU: 8× MI355X (gfx950), ROCm 7.0.1(host)/7.2.0(容器), amdgpu 6.14.14
- RDMA(强制用): **mlx5_0**(ens3, fw 28.43.3608, driver 24.10-3.2.5, 200 Gb/s HDR, RoCEv2 GID idx3)
- base 镜像: `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (sha256:40e940a0…)
- 编出的镜像: `dsv4-sgl-dmabuf:mlx5` (sha256:8746f670…) — 节点本地,未入 git(见 REPRODUCE 重建)
- git: 分支 `yihou.dev.sglang.mooncake` @ `f0938ba`
- **不需要密钥**(公开 base 镜像 + 共享 NFS 模型 + 集群自带身份)

## 未提交/依赖(见 ENVIRONMENT.md)

- 模型 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(806G,不入 git)
- 编出的镜像 tar `dsv4-sgl-dmabuf.tar`(27GB,**不入交付**;用 `scripts/build_mc_dmabuf.sh`
  从 base 重建即可,见 REPRODUCE)。原始 workspace `temp/pd_mlx5_1p1d/`(gitignored,保留)。
