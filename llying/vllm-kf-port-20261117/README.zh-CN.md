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

## 4. #47897 接入 vLLM（覆盖层）

- `overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py`：#47897 的 kernel 文件（sglang@8a9f215c），NG2 改为 32，LPT 分组行数改为调用参数。
- `overlay/rocm_aiter_mla_sparse.py.diff`：`rocm_fp8_paged_mqa_logits` 在 `VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`、gfx950、next_n 2–8、
  预洗牌 64 token 页、行数 ≤ 128 时走 gluon kernel；逐行长度优先用调用方给的 `[B, next_n]`，否则由每请求长度推出；block table 用
  `repeat_interleave` 展开（不需要主机同步，可进 CUDA graph）；一个请求的 MTP 行作为一个 LPT 组。其余情况保持 aiter。
- 运行时按文件 bind mount 覆盖镜像里的两个文件（`scripts/run_overlay_test.sh`），不重建镜像；上游 PR 即这份 diff 加 kernel 文件。
- 函数级测试（2026-11-18 00:25，`results/overlay-test-1/`）：1–32 个请求、一维 / 二维长度都走 gluon，与 aiter 最大差 7.6e-6；40 个请求（160 行）
  回落 aiter；CUDA graph 捕获后换内容重放结果一致。**PASS**。

## 4.0 更正：vLLM 在 ROCm 上把 MTP decode 拉平

- `v1/attention/backends/mla/indexer.py::_supports_native_decode(4)` 在 ROCm 上为 False，`_use_flattening` 为真：decode 时每个 MTP 行是一行
  （next_n = 1），带逐行长度与逐行 block table，和 sglang 调 #47897 的方式一样。
- 因此 3.2 的 aiter 基线（`[B, next_n=4]`）不是 vLLM 实际的调用；第一版覆盖层要求 next_n ≥ 2，在引擎里**从没生效**
  （第一次聚合 A/B 实为原版对原版：GSM8K 197 / 195，可当 A/A 参考）。
- 修正后的覆盖层接受拉平行（LPT 组 = 1 + num_speculative_tokens，可用 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS` 覆盖；组须为 ≥ 2 的偶数，
  否则回落 aiter），函数级测试加拉平形式（一维 / `[rows, 1]` 长度）全部通过（`results/ab2/overlay-test.txt`）。
- 对 vLLM 实际的拉平 aiter 调用重测（`results/ab2/paged_logits.txt`）：

| 请求数（行数） | aiter 拉平（vLLM 实际） | #47897 4 行分组 | 倍数 |
|---|---|---|---|
| 4（16） | 39.0 | 15.3 | 2.54 |
| 8（32） | 96.1 | 39.9 | 2.41 |
| 16（64） | 163.4 | 69.3 | 2.36 |
| 26（104） | 247.9 | 110.8 | 2.24 |
| 32（128） | 324.9 | 139.1 | 2.34 |

## 4.0b 第二次聚合 A/B（2026-11-18 02:20–03:00）：gluon 路径仍未生效，结果无效

- 门禁（覆盖层打开）：GSM8K 198/200、长上下文 32/32 + 32/32，正确。
- c104 bench（随机 64K 输入、1024 输出）：开 TPOT p50 217.4 ms / 关 206.9 ms，ITL p50 53.2 / 53.1 ms。**但 aggK 日志里没有
  "Using the gfx950 gluon paged-MQA logits kernel"**，新 kernel 仍没被调用。最可能的原因：worker 运行时 `get_current_vllm_config()`
  拿不到真实配置，`_gluon_lpt_group_rows()` 得 0，条件不成立。下一次用 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4` 显式给出，
  并在启动后先确认日志里出现上面那一行再测。
- 这个 bench 被 64K prefill 主导（TPOT 200 ms 量级），不适合量 decode kernel；应改成一波请求先全部 prefill 完、长输出（例如 4096）的形状，
  或直接用引擎内 trace 比较 indexer 时间。

## 4.1 在 decode 每步里的份量（P4D4 s6c 的 decode trace）

- indexer 占 decode 每步约 5%（logits 93 µs + top-k 47 µs，每步各约 20 次），sparse MLA 约 6%；MoE GEMM 31%、MoE 通信 26% 才是大头。
- 但各 DP rank 的上下文长短不同，indexer 最重的 rank（dp3，是 dp0 的 2.4 倍）是每步的关键路径，其余 rank 在通信里等它；
  #47897 把 logits 减半，关键路径每步约省 2.4 ms（约 5–6%），比按平均 rank 估算的大。
- 因此 decode 侧的顺序调整为：MoE 通信（vLLM 的 aiter 自定义 AG/RS 为何没走上 / EP + MoRI）与 #47897 并行推进，#47872 次之。

## 5. 状态与下一步

- 下一步：GPU 空出后做聚合模式 GSM8K 门禁（DP4 + MTP3 + FP8 KV，开关打开）与 decode ITL 开 / 关对比，再进 PD 端到端；
  然后测 #47872（top-k）与 #47276；以 vLLM 上游 PR 的形式提交。
