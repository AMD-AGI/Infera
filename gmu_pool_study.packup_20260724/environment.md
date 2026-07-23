# Environment — DSv4-Pro KV pool × gmu study

Ran: 2026-07-24, single node (chi2879). Study is single-node launch-read +
one single-node mix run; no cross-node RDMA needed to reproduce.

## Hardware

| 项 | 值 |
|----|----|
| 节点 | chi2879（Slurm k8s 分区, 本人 hold, 伪装名 `training`） |
| GPU | 8× AMD Instinct MI355X (gfx950)，**VRAM total = 309,220,868,096 B = 287.98 GiB/卡** |
| CPU | 128 核 (AMD EPYC) |
| RAM | ~3 TB (3023 GiB) |
| kernel | 6.8.0-124-generic |
| host ROCm | 7.2.0 |
| ionic 内核模块 | 26.03.3.001 |

> 起测前 8 卡 VRAM used ≈ 298 MB/卡（仅系统 exporter），干净。

## Software

| 组件 | 值 |
|------|----|
| docker 镜像 | `infera/engine-sglang:pd-mcgate` |
| 镜像 digest | `sha256:87fa3ca9c50cc41145aeaed9f867a0dff42fe48df88a32b2e44900ef832f8f1d` |
| **sglang** | **0.5.15.post1**（editable install `/sgl-workspace/sglang`, git commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`） |
| sgl_kernel | 0.4.4 |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |
| kv-cache-dtype | fp8_e4m3（默认） |

## Model

| 项 | 值 |
|----|----|
| 模型 | DeepSeek-V4-Pro |
| 路径 | `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro`（真权重, symlink→`/mnt/vast/john/huggingface/DeepSeek-V4-Pro`，容器须 `-v /mnt/vast:/mnt/vast`） |
| 校验 | shard `model-00001-of-00064.safetensors` = 1,853,358,176 B；`tokenizer.json` = 6,367,146 B（非 LFS stub） |
| 结构 | 61 层，MLA + DSA indexer，fp8 e4m3 attn + MXFP4 MoE |
| 关键 dim | q_lora_rank=1536，index_head_dim=128，index_n_heads=64，index_topk=1024，qk_rope_head_dim=64；MLA cell 组成 = qk_nope(448 FP8)+qk_rope(64)×2(BF16)+scales = **584 B/token/层** |

## 并行配置（本研究固定项，只扫 gmu）

`--tp-size 8 --dp 8 --enable-dp-attention --ep-size 8`，`--page-size 256`，
`--context-length 9472`，`--attention-backend dsv4`，`--disable-radix-cache`，
`--swa-full-tokens-ratio 0.15`，`--cuda-graph-max-bs 512`，`--chunked-prefill-size 163840`。

## 关键 env（决定走 unified KV 路径 —— 影响物理排布）

`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` + ROCm → `is_unified_kv_triton()=True`
→ KV buffer 走 `DeepSeekV4UnifiedKVPool`（每层一块 bf16 `[swa_pages+compress_pages, 512]`）。
完整 R4 perf env set 见 `scripts/kv_gmu_sweep.sh` 头部（verbatim）。

## Repo state（分析产物所在）

- 本 packup 所在 repo：`infera.rdma`，分支 `yihou.dev.rdma`。
- 引用的 sglang 源码：见 `src_refs/`（4 个文件，从容器 `/sgl-workspace/sglang` 原样拷出，tag v0.5.15.post1）。
