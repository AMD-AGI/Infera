# vLLM v0.34.0（ROCm 10.0）上 GLM-5.2-MXFP4 单节点聚合正确性（2026-11-17，crusoe v3-135）

方案第一阶段（`../vllm-pd-plan-20261117/PLAN.zh-CN.md` 5.2 节）：Infera 8 月记录过 GLM-5.2 在 vLLM/ROCm 上 decode 乱码、FP8 KV 乱码，
所以先在不拆分的单节点上验证 decode 目标配置。

## 0. 结论

- **v0.34.0 原版镜像上 GLM-5.2-MXFP4 是正确的，包括 decode 目标配置（DP4 + MoE TP 平铺、FP8 KV、前缀缓存、MTP K3）**：
  门禁与 sglang 同口径、同标准，全部干净，正确率与 sglang PD 门禁（198/200）和原版 sglang 参照（0.973）同一水平。
- **真实 MTP 接受长度 3.50**（K3，上限 4；逐位置 0.93 / 0.83 / 0.73），sglang 关 PTPC 时 3.61、带 #45800 的 PTPC 3.60–3.62。
- Infera 8 月的"vLLM 上 GLM-5.2 乱码"在 v0.34.0 上不复现（那时是 v0.25 时期，v0.34.0 的 ROCm DSA 实现已重写为
  `vllm/models/deepseek_v32/amd/`）。

## 1. 环境

| 项 | 值 |
|---|---|
| 节点 | crusoe spur `crsuse2-m2m-v3-135`（作业 294199），8 × MI355X，amdgpu 6.16.13 |
| 镜像 | `vllm/vllm-openai-rocm:v0.34.0`（`sha256:3cb773d2…`，ROCm SDK 10.0、torch 2.13、aiter v0.1.28.post1），原样运行 |
| 模型 | `GLM-5.2-MXFP4`，拷到本地 NVMe（438 GB，82 s） |
| 环境变量 | `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| 脚本 | `../vllm-pd-tools-20261117/scripts/vllm_up.sh`、`gate.sh`（smoke、GSM8K 200、16k × 64、vLLM 计数器算接受长度） |

公共参数：`--max-num-seqs 64 --max-num-batched-tokens 8192 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --enable-prefix-caching`。

## 2. 结果（`results/summary.out`）

| 组 | 配置 | smoke | GSM8K 200（正确 / 开头坏 / 乱码 / 撞上限） | 16k 捞针 / 长前缀 GSM8K | 真实接受长度 |
|---|---|---|---|---|---|
| aggB | GPU 4-7，TP4，bf16 KV，无 MTP | 5/5 | 195 / 0 / 0 / 3 | 32/32 / 31/32 | — |
| **aggA** | GPU 0-3，**DP4 + TP1（MoE 平铺 TP4）、FP8 KV（fp8_e4m3）、MTP K3** | 5/5 | **197** / 0 / 0 / 3 | 32/32 / 31/32 | **3.50** |

## 3. 观察

- 起栈约 11 min：每个 shard 约 2 s（282 个，约 9 min）。DP4 每个 rank 都读全部 shard；超过 vLLM 默认的引擎就绪超时 600 s，
  第一次 aggA 因此失败，`VLLM_ENGINE_READY_TIMEOUT_S=3600` 后正常。加载提速以后再看（`--load-format`、预取）。
- 这个模型在 ROCm 上不走 torch.compile（日志 "does not support it"），cudagraph 从 `FULL_AND_PIECEWISE` 退为 `FULL_DECODE_ONLY`：
  decode（含 MTP verify）走完整 cudagraph，prefill 走 eager。MTP 层的 BF16 专家走 aiter unquantized MoE，共享专家融合只对主干开。
- DP4 内置负载均衡不认 `x-session-id`：smoke 的长前缀第二次请求被派到另一个 rank，没有命中前缀缓存（6.9 s，TP4 下 0.24 s）。
  PD 部署由 router 用 `X-Data-Parallel-Rank` 选 rank，不受影响；这也说明 D 侧前缀缓存要配合 router 的 rank 粘性才有用（方案 5.5）。
- 两个容器同时初始化同一个命名卷会失败（"file exists"）；镜像没有 `render` 组，`--group-add render` 失败。都已在 `vllm_up.sh` 里处理。
