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

## 3. 第一次微基准（2026-11-17 23:57，v3-135，MI355X，一卡一个候选，`results/sparse-mla-1/`）

### 3.1 sparse MLA decode（µs / 次，括号内是对 float32 参考的 SNR dB）

| 行数 | vLLM 默认 aiter asm | aiter Triton | sglang #47278 | KF campaign |
|---|---|---|---|---|
| 16 | **59.2**（33.4） | 77.1（33.2） | 92.4（33.4） | 86.5（30.1） |
| 32 | **60.1**（33.3） | 76.5（33.2） | 86.5（33.3） | 83.5（29.8） |
| 64 | **57.0**（33.2） | 77.5（33.1） | 87.8（33.2） | 88.7（29.3） |
| 104 | **56.8**（33.0） | 77.7（33.0） | 87.9（33.1） | 85.1（28.2） |
| 128 | **59.2**（33.2） | 78.2（33.1） | 88.6（33.2） | 87.5（28.3） |
| 208 | **84.1**（33.0） | 86.6（33.1） | 161.9（33.1） | 164.8（33.1） |

**结论：vLLM 默认的 aiter asm 持久化 kernel 已经比 #47278 和 KF 版本快 1.4–2 倍**。这两项是在 sglang 自带的 Triton kernel 基础上优化的，
对 vLLM 没有收益，不移植；`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1` 也更慢，保持默认。KF 版本在 16–128 行时 SNR 低 3–5 dB，另需注意。

### 3.2 indexer paged MQA logits（µs / 次，上下文 40K–200K 随机，next_n 4）

| 请求数（行数） | vLLM 默认 aiter | #47897 原参数（6 行 LPT 分组） | #47897 + 4 行分组 / NG2 32 | 倍数 |
|---|---|---|---|---|
| 4（16） | 28.7 | 20.4 | **15.3** | 1.88 |
| 8（32） | 83.5 | 48.4 | **40.5** | 2.06 |
| 16（64） | 140.1 | 84.9 | **71.2** | 1.97 |
| 26（104） | 251.2 | 132.4 | **110.3** | 2.28 |
| 32（128） | 272.0 | 167.9 | **139.3** | 1.95 |

两种 #47897 与 aiter 在可见列上的最大绝对差 7.6e-6（logits 量级 50–70），等价。逐行长度与展开 block table 每步构建一次约 28 µs，各 indexer 层共用。
**结论：#47897 是第一个要移植的项**；MTP3 下用 4 行分组（团队的 K5 设置）。按 s4 的 decode 形状（每 rank 约 26 个请求、104 行），每个 indexer 层省约 140 µs。

## 4. 状态与下一步

- 下一步：把 #47897 的 kernel 接到 `rocm_fp8_paged_mqa_logits`（开关控制，超过 128 行回落 aiter），做 GSM8K 门禁与端到端 A/B；
  然后测 #47872（top-k）与 #47276；以 vLLM 上游 PR 的形式提交。
