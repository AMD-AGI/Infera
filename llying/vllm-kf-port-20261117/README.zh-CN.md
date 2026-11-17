# KF / sglang DSA decode kernel 移植到 vLLM（GLM-5.2，gfx950）

目标：把 sglang 上合入慢的四个 gfx950 DSA decode PR 和 KernelOptForGLM52 的 sparse MLA 候选接到 vLLM v0.34.0 的 ROCm DSA 路径，
先用微基准在 vLLM 的真实 decode 形状上和 vLLM 现有 kernel 比，再接进引擎做端到端 A/B，最后走 vLLM 上游。

## 1. 对照表

| 来源 | 内容 | vLLM v0.34.0 对应位置（现有实现） | 接入方式 |
|---|---|---|---|
| [sglang #47278](https://github.com/sgl-project/sglang/pull/47278) | 64 头 FP8 sparse MLA 的 split-K decode 方案（sglang 自带 Triton kernel） | `v1/attention/backends/mla/rocm_aiter_mla_sparse.py`：默认 aiter asm `mla_decode_fwd`（持久化 work 元数据）；`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` 时 aiter Triton `sparse_mla_fwd` | 新增第三种实现。数据一致：vLLM 是逐张量 FP8 `[slots, 576]` + 全局 slot id，sglang 是 `[pages, 1, 576]`（页大小 1）+ 稠密 `[rows, 1, topk]`；`q_scale·k_scale` 折进 `sm_scale`，输出乘 `k_scale` |
| KernelOptForGLM52 `campaign-sparse/best/kf_sparse_mla.py` | 同上 kernel 的 campaign 优化版（单算子 1.08–1.38 倍） | 同上 | 同上 |
| [sglang #47897](https://github.com/sgl-project/sglang/pull/47897) | gfx950 paged MQA logits（从 aiter gluon preshuffle kernel 派生、重调） | `v1/attention/ops/rocm_aiter_mla_sparse.py::rocm_fp8_paged_mqa_logits` → aiter `deepgemm_fp8_paged_mqa_logits`（Preshuffle、64 token 页、ChunkK 256） | KV 布局相同（每页 `64×128` FP8 后接 `64×4` f32 scale）。sglang 每行一个 query：vLLM 侧把 `[B, next_n]` 展开成逐行长度与逐行 block table（每步一次，各 indexer 层共用）。kernel 支持 1–128 行，超过回落 aiter |
| [sglang #47872](https://github.com/sgl-project/sglang/pull/47872) | 单遍 Triton top-k（长上下文 decode 行） | 同文件 decode 段：aiter `top_k_per_row_decode` 或 `torch.ops._C.top_k_per_row_decode` | 独立 Triton 文件，替换 decode top-k |
| [sglang #47276](https://github.com/sgl-project/sglang/pull/47276) | MTP verify 行的融合 indexer logits + top-k | 依赖 sglang 的 JIT HIP `dsa_gfx950` 基础设施，vLLM 没有对应物 | 工作量最大，放最后 |

## 2. 微基准（`scripts/`）

- `bench_sparse_mla.py`：64 头、top-k 2048、FP8 q 与 KV、行数 16–208（DP4 每 rank 的请求数 × MTP 4 行）。
  候选：vLLM 默认 asm（按 vLLM 元数据构建方式：页 1、kv_granularity 16、fast_mode、max_split 16）、aiter Triton、sglang #47278、KF campaign。
  全部对 float32 参考算 SNR；轮换 8 个 KV 池，避免 256 MB MALL 让 gather 虚快。
- `bench_paged_logits.py`：32 头 × 128、64 token 预洗牌页、next_n 4、上下文 40K–200K。候选：aiter（vLLM 调法）、#47897 原参数、#47897 改 4 行 LPT 分组。
  逐行可见列对 aiter 结果比较。
- `run_bench.sh`：节点上一张卡一个候选并行跑；`shim/` 是 sglang kernel 文件需要的几个 sglang 函数的最小替身（架构、CU 数、FP8 格式）。
- 第一次运行排在 P4D4 s5 与 s6 之间的 GPU 空档（队列项 `Lp4d4s6b`），结果写到 `results/`。

## 3. 状态

- 2026-11-17 23:55：对照表、两个微基准写好，等 GPU 空档。
