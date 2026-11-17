# GLM-5.2 AgentX PD：在 Infera + vLLM v0.34.0 上复现并超越 sglang recipe 的方案（2026-11-17，待评审）

本文只做静态分析和计划，没有跑任何实验。对标的 sglang 结果见 `dev/pd_opt/glm_5.2_agentx` 分支的
`llying/pd-repro-crusoe-20261116/`（crusoe 复现）和 `llying/dp-graph-variant-fix-aus-20261103/`（AUS 原始 recipe、RESULT、瓶颈分析）。
vLLM 源码引用均指 tag `v0.34.0`（`62d4bc3c`），本地浅克隆在 `/home/liyingli/bench_agentx/baseline/src-snapshots/vllm-v0.34.0`。

## 0. 结论

- 基础镜像：`vllm/vllm-openai-rocm:v0.34.0`（11-16 发布，digest `sha256:3cb773d23f06…`）。镜像内是 ROCm SDK 10.0
  （以 wheel 形式安装在 site-packages）、torch 2.13、aiter v0.1.28.post1、mori v1.2.3、triton 3.8、Python 3.12、Ubuntu 22.04，
  自带 PyPI 版 Mooncake ROCm（`mooncake-transfer-engine-rocm>=0.3.13`，当前最新 0.3.14）和 AINIC 1.117.5（libionic 54.0-187）。
  ROCm 10.3 预览版已在本集群同款驱动（amdgpu 6.16.13、ionic 26.08.17）上跑通过 SGLang PD，ROCm 10 用户态与驱动的兼容风险较低。
- vLLM v0.34.0 对 GLM-5.2 的支持比 Infera 记录时（8 月，v0.25 时期）完整得多：ROCm 上有专门的 DSA 实现、aiter 稀疏 MLA、FP8 KV、
  MTP、PP、DP attention + MoE TP 平铺、Mooncake PD（P 侧 PP 对 D 侧非 PP）、CPU KV offload、模拟接受长度。逐项见第 3 节。
- **第一道门禁是正确性**：Infera 文档记录过 GLM-5.2 在 vLLM/ROCm 上 decode 先正常后乱码、FP8 KV 一上来就乱码，gfx942 上 PD + MTP3
  长上下文输出损坏；vLLM 自带的 GLM-5.2 MI355X MXFP4 recipe 也是关前缀缓存、不开 MTP。v0.34.0 是否已修好，必须先在单节点上验证。
- 缺口（要么用 Infera 侧绕开，要么走 vLLM 上游）：没有 SPF 调度；MooncakeConnector 没有自定义 KV 内存池（crusoe 上 GEM KV 需要）；
  Mooncake 不逐块发送 KV；KF kernel 没有对应物。
- 可能超过 sglang 的方向：D 侧前缀缓存 + D rank 会话粘性（vLLM 的 Mooncake consumer 只拉本地未命中的块）、decode MoE 改 EP、
  短增量请求在 D 上 prefill（按场景 A/B 后再定）。

## 1. 目标与口径

| 配置 | sglang crusoe | sglang AUS | vLLM 目标 |
|---|---|---|---|
| P8D8：1P(TP1×PP8) + 1D(TP8/DP8)，C208，16 卡 | 54,468 | 56,593 | ≥ 54,468 |
| P4D4：单节点，prefill GPU 0-3，decode GPU 4-7，C104 | 53,351 | 53,209 | ≥ 53,351 |

- 只换引擎，其余不变：同一集群（crusoe spur `amd-spur-v2`）、同一模型（`GLM-5.2-MXFP4`，每次作业拷到本地 NVMe）、同一压测
  （InferenceX AgentX、AIPerf 回放、3600 s、`WARMUP_PER_LANE=10`、K3 下模拟接受长度 2.99）、同一 router（Infera rust 后端）。
- sglang 的两个数里含 PTPC FP8 草稿 MoE（AUS 上关掉它 −1.6%，52,830 对 53,666）和 KF kernel overlay。
- **11-17 评审决定**：PTPC 先不应用，以后再决定是否尝试（vLLM 有通用在线 FP8 / PTPC，`model_executor/layers/quantization/online/fp8.py`，
  但没有"只量化 Quark checkpoint 里 MTP 层 BF16 专家"的开关，所以不是"vLLM 本身具备"的情形）。KF 有对应的 sglang PR
  [#47276](https://github.com/sgl-project/sglang/pull/47276)（gfx950 融合 indexer logits + top-k，MTP verify 行）、
  [#47278](https://github.com/sgl-project/sglang/pull/47278)（gfx950 64 头 FP8 sparse MLA 的 split-K decode）、
  [#47897](https://github.com/sgl-project/sglang/pull/47897)（gfx950 paged MQA logits，decode / MTP verify 行）、
  [#47872](https://github.com/sgl-project/sglang/pull/47872)（单遍 Triton top-k，长上下文 decode 行），正因为它们在 sglang 合入慢才转到 vLLM，
  **这些特性都要在 vLLM 上支持**：先用 vLLM 原版 kernel 测基线，再逐项接入做性能对比（5.5 节）。KF MoE overlay 没有 sglang PR，同样列入。
- 正确性门禁沿用 `chain_v2.sh`：smoke 6/6；GSM8K 200 题乱码 0、开头坏 ≤ 1；16k × 64 捞针与长前缀；真实接受长度 ≥ 3（K3）。
  门禁干净才计分。发布前再补 1,319 题、lm-eval、100k 长上下文。

## 2. sglang recipe 拆解

实际命令行来自 `dp-graph-variant-fix-20261103/kit/engine/worker.sh`、`up.sh`，叠加 `config/recipe-e10.env` → `arm-C-prod.env` →
`arm-C-prod-ptpcfix.env` →（P8D8）`arm-C-1p1d-pp8.env` 或（P4D4）`arm-C-prod-ptpcfix-nohip.env`，再叠 `gemkv.env`。

| 部分 | sglang 设置 | 作用 / 实测 |
|---|---|---|
| prefill 并行 | TP1 × PP8（PP4），层切分 10×7+8（20,20,20,18），`--pp-async-batch-depth 1`，`SGLANG_PP_COMM_OVERLAP=1` | 单一 radix 树，免两实例重复缓存 |
| prefill 批 | chunk 8192，`--max-prefill-tokens 8192`，mem 0.88，`--schedule-policy shortest-prefill-first` | SPF 比 FCFS +15.4%（2P1D，10-20） |
| prefill 缓存 | HiCache 250 GB/rank，write_back，kernel IO，page_first；evict-headroom 512K、backup-quota 16（我们的补丁） | 命中率约 97% |
| decode 并行 | TP8 + DP8 attention（P4D4 为 TP4/DP4），mem 0.95，cuda graph bs ≤ 256，`--stream-interval 30` | |
| decode kernel | KF overlay（sparse MLA、indexer、top-k、MoE）、调优 GEMM 表、custom all-reduce、`SGLANG_USE_ROCM700A=0`（graph 内 aiter reduce-scatter） | E11 每步约省 3 ms |
| 投机解码 | EAGLE 3 步 / topk 1 / 4 个草稿 token，两侧都带（prefill 要生成草稿层 KV），`index_share_for_mtp_iteration=true` | 真实接受长度 3.6 |
| 正确性修法 | graph variant 跨 rank 一致（sglang#48302）、pp-chunk-wait、`deepseek_v2.py` PP 中间级、`glm4_moe.py`（#45800） | sglang 专有 |
| 传输 | Mooncake faae8dd4 + dest-local-rail，`MC_IB_TIMEOUT=18`，`MC_MAX_MR_SIZE=1G`；crusoe 上 GEM KV、`MC_DISABLE_HIP=1`、不注入宿主机 libionic | |
| router | `--router-backend rust --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on`；prefill 发 KV 事件，decode 不发 | |

瓶颈（AUS RESULT 7.5、BOTTLENECK-PP8）：P8D8 受 prefill 限（单个 PP8 prefill 92% 时间有排队，现算约 19–20K tok/s，decode 有余量）；
P4D4 在 C104 时 prefill 与 decode 都到顶。AgentX 每卡吞吐 ∝ 并发 /（卡数 × lane 周期），lane 周期 = 工具 / 思考间隔（约 50%，服务端改不了）
+ TTFT + decode。

## 3. vLLM v0.34.0 静态分析

### 3.1 模型与 kernel

| 项 | 结论 | 代码 |
|---|---|---|
| GLM-5.2 | `GlmMoeDsaForCausalLM` 在 ROCm 上用 `vllm/models/deepseek_v32/amd/`（与 DeepSeek-V3.2 同一实现） | `model_executor/models/registry.py:125`，`models/deepseek_v32/__init__.py` |
| 稀疏 MLA | `ROCMAiterMLASparseBackend`：KV dtype 支持 bf16 / fp8 / fp8_e4m3；MTP verify 行按 decode 处理；cudagraph `UNIFORM_BATCH` | `v1/attention/backends/mla/rocm_aiter_mla_sparse.py:352-357,441,457` |
| indexer / top-k 共享 | 按 checkpoint 的 `index_topk_freq=4` 跳过层，与 sglang 一致；MTP 步间共享 top-k | `model_executor/models/utils.py:1019`，`models/deepseek_v32/amd/mtp.py:127` |
| 融合 | fused norm+rope、fused q、aiter 一阶段 all-reduce + RMSNorm（decode 尺寸）、aiter 共享专家融合 | `models/deepseek_v32/amd/rocm.py`，`models/common/ops/fused_allreduce_rms_norm.py` |
| MXFP4 MoE | Quark `w_mxfp4_a_mxfp4` 按激活格式筛选后选 `AITER_MXFP4_MXFP4`（CK，gfx950），可用 `--moe-backend aiter_mxfp4_mxfp4` 指定 | `model_executor/layers/fused_moe/oracle/mxfp4.py:171,408,427-431` |
| MTP | `glm_moe_dsa` 自动映射 `DeepseekV32MTPModel` | `config/speculative.py:831-860` |
| 模拟接受长度 | `rejection_sample_method: "synthetic"` + `synthetic_acceptance_length` | `config/speculative.py:613-631,1666-1680` |
| vLLM 自带 recipe | MI355X MXFP4：TP8、`--no-enable-prefix-caching`、不开 MTP、`VLLM_ROCM_USE_AITER_FP8BMM/FP4BMM=0` | `recipes/models/zai-org/GLM-5.2.yaml:86-100,294-318` |

### 3.2 并行

- decode：`-dp 8 -tp 1` 不开 EP 时，MoE 平铺成 TP8（`FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp`，all-gather / reduce-scatter），
  与 sglang 的"DP attention + MoE TP8"等价（`model_executor/layers/fused_moe/config.py:1143-1151,1235-1260`）。
  开 `--enable-expert-parallel` 则走 EP，可选 MoRI all-to-all（`distributed/device_communicators/all2all.py:933,1088`）。
- 每个 DP rank 可被 `X-data-parallel-rank` 请求头定向（`entrypoints/generate/base/serving.py:193-198`），Infera router 已在用。
- prefill：TP1 × PP8，`VLLM_PP_LAYER_PARTITION` 切层；静态上未见 PP 与 MTP 互斥的检查（与 PP 互斥的只有 adaptive verification 等，
  `config/vllm.py:3306`），要实测。
  PP + 异步调度下 decode 有"同一请求间隔 pp_size 步"的约束（`v1/core/sched/scheduler.py:832`），prefill 分块之间没有，
  同一请求的多个 chunk 应能在各级间流水，需实测确认。

### 3.3 PD 传输

- MooncakeConnector：P 推送，P/D 并发下发（与 sglang 的 bootstrap 相同形态）；P 的 PP 与 D 不同时按层名对齐
  （`distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:602-622,1775-1790`）。
- **D 侧本地前缀命中只拉剩余块**：`count = 远端 prompt token 数 − 本地 num_computed_tokens`，只为未缓存的块发起传输
  （同文件 `:1188-1196,1219-1227`）。sglang 的 D 每轮都收完整 KV。
- 不做逐层 / 逐块发送（同文件 `:956`）：长请求的 KV 在整个 prefill 结束后才发。sglang 的 P 每算完一块就发。
- NIXL：生产端支持 PP，消费端不支持（`nixl/base_worker.py:943-961`）；MoRIIO 未见 PP 支持。P 用 PP，所以走 Mooncake。
- 自定义 KV 内存池：connector 基类有 `get_mem_pool_context()`，`gpu_worker.py:925` 分配 KV 时调用，`MultiConnector` 透传；
  MooncakeStoreConnector 已用它接 NVLINK / BAREX（`mooncake/store/worker.py:1501-1531`），**MooncakeConnector 没有实现**。
- ROCm 上带 RDMA 注册的 connector 时，可扩展 KV（VMM）自动关闭，走普通分配（`config/vllm.py:3230-3238`）。

### 3.4 prefill 缓存层（HiCache 的对应物）

- `OffloadingConnector`（CPU 层，`cpu_bytes_to_use` 为全部 worker 合计，`lru` / `arc`，默认只 offload prompt 块），算完的块即写入
  CPU（write-through），命中时异步拉回 GPU。与 MooncakeConnector 用 `MultiConnector` 组合。
- 已考虑 DSA（MLA + indexer 的 `UniformTypeKVCacheSpecs` 组，`offloading/config.py:236`）；"canonical 布局"只支持纯 TP，默认布局没有此限制
  （`docs/features/kv_offloading_usage.md:151-153`）。PP8 + DSA + Mooncake 三者组合未见上游测试。
- 写入带宽估算：每 token KV 约 47 KB（78 层 × 576 B 的 FP8 latent + 约 20 层 indexer × 132 B），按 prefill 现算 2 万 tok/s 计约 1 GB/s，
  write-through 的开销可忽略。

### 3.5 调度

- 只有 `fcfs` / `priority`（`config/scheduler.py:20`），外加"缓存感知准入窗口"（已缓存前缀的请求优先准入，`config/scheduler.py:225-245`）。
  **没有 SPF**。

### 3.6 Infera 侧现状（新分支基线 main `a6a0ba275`）

- `deploy/docker/Dockerfile.vllm`：基础镜像固定 v0.25.1（`:12`）；默认重编 aiter v0.1.16.post1（`:34-35`，叠到 v0.34.0 上等于降级）；
  patch 循环失败只打印 skipped（`:71`）；自编 Mooncake faae8dd4 + `transfer_engine_impl.diff`（`:86`）。
- router：Python（`infera/router/disagg_protocols/vllm_mooncake.py`）与 Rust（`rust/router/src/protocol.rs`）都实现了 vllm-mooncake 并发推送，
  注释写明按 vLLM 0.21 对齐；Python router 没有 vllm-nixl。Rust router 能解 vLLM 的 KV 事件（`rust/router/src/kv_event.rs`）。
- 引擎包装：`infera/engine/vllm/args.py` 能识别 `MultiConnector` 里的 Mooncake 子 connector；KV 事件只预留 1 个端口
  （`infera/engine/vllm/__main__.py:70`），vLLM 内置 DP 下各 rank 在"基础端口 + rank"上发布，D 侧开事件时要确认 8 个 rank 都能收到。
- 已有记录：GLM-5.2 在 vLLM/ROCm 上 decode 乱码、FP8 KV 乱码（`manual/recipes/glm5.2.md:152`、`manual/examples/k8s_glm5.2_sglang.md:14-20,85-87`，
  8 月）；gfx942 PD + MTP3 长上下文损坏（`manual/wip/gfx942-e2e.md:74,244`）。

## 4. 风险（按影响排序）

1. **正确性**（阻塞）：见 3.6 最后一条。v0.34.0 已重写 ROCm DSA 路径，但"FP8 KV + 前缀缓存 + MTP + DP + PD"组合在 MI355X 上没有公开验证。
2. **Mooncake 版本**：镜像自带 PyPI 0.3.14，不是 Infera 的 faae8dd4 + 补丁。要逐项实测 dma-buf 注册、HIP 传输开关（`MC_DISABLE_HIP`）、
   dest-local-rail、GEM 显存设备号（我们的 Mooncake#5584）、`MC_MAX_MR_SIZE` 对 ionic 的 1 GiB 限制。缺哪项就针对 ROCm 10 + Python 3.12
   自编 Mooncake（faae8dd4 或上游 main，加我们的补丁）。
3. **crusoe 驱动 dma-buf 重复记账**：KV 必须用 GEM 分配，否则只能把显存比例压得很低（ROCm 10.3 预览里 prefill 0.65、decode 0.60）。
   做法见 5.1；GEM 预加载库要针对 ROCm 10 SDK 重新编译。
4. **Infera 适配**：Dockerfile、patch 逐个核对（v0.25.1 的 7 个 vLLM patch 在 v0.34.0 上多半锚点已变）、vllm-mooncake 协议字段、KV 事件端口。
5. **prefill 侧组合**：PP8 + OffloadingConnector + MooncakeConnector 未经验证；Mooncake 不逐块发送；没有 SPF。
6. **decode kernel 性能**：没有 KF overlay，走 aiter 原版。sglang TP8 下 MoE 本来也没用上 KF（AUS RESULT 4.6：gemm1 回退 aiter 原版），
   KF MoE 只覆盖 TP4 形状的部分 batch 范围；sparse MLA / indexer 的差距要靠 trace 量化。

## 5. 执行阶段

每个阶段一个独立目录（第 8 节），先写预测再跑，结果和偏差写进该目录的 README。

### 5.1 阶段 0：镜像与静态检查（1 节点，不占 GPU 计分）

- 新增 `Dockerfile.vllm` 的 v0.34.0 变体（或参数化基础镜像）：`BUILD_AITER=0`；逐个 patch 在 v0.34.0 源码上判定"仍需要 / 已上游 / 锚点变了"，
  更新 `patch.upstream.status.md`；patch 失败改为构建失败；Mooncake 先用镜像自带版本，按风险 2 的清单探测，不满足再自编；
  hipFile、Rust router、`pip install .[vllm]` 在 Python 3.12 上验证。
- crusoe 专用：GEM 预加载库针对 ROCm 10 重编；Infera 侧加一个 MooncakeConnector 子类，覆盖 `get_mem_pool_context()`，
  返回基于 GEM 分配器的 `torch.cuda.MemPool`（不改 vLLM），配 `LD_PRELOAD`、`MC_DISABLE_HIP=1`；容器入口不注入宿主机 libionic。
- 产出：镜像 tar（`/shared_nfs/liyingli/images/`）、构建日志、patch 判定表、Mooncake 探测结果（同节点 RDMA 回环带宽、8 张网卡是否都在、计数器分布）。

### 5.2 阶段 1：单节点聚合正确性（1 节点）

- 逐项打开，每步跑 smoke + GSM8K 200 + 16k 长上下文：TP8 bf16 KV → FP8 KV → 前缀缓存 → MTP K3（真实接受长度）→ DP8（MoE TP 平铺）→ TP4/DP4。
- 出现乱码就停下二分，定位到 kernel / 元数据 / MTP 路径；这里查到的问题是第一批 vLLM 上游 PR。
- 通过判据：与 sglang 门禁同标准，真实接受长度 ≥ 3。

### 5.3 阶段 2：单节点 P4D4（1 节点）

- P：TP1 × PP4，`VLLM_PP_LAYER_PARTITION=20,20,20,18`，chunk 8192，前缀缓存，`MultiConnector`（Mooncake producer + Offloading CPU 层），
  `--speculative-config` 与 D 相同（P 要产出 MTP 层 KV）。D：`-dp 4 -tp 1`，FP8 KV，MTP K3 + 模拟接受长度 2.99（计分时）。
  router：与 sglang 相同的 rust 参数。传输走同节点 RDMA 回环（`MC_DISABLE_HIP=1`）。
- 先门禁，再 C104 3600 s 计分，对标 53,351；同时记录 prefill 满块耗时、现算吞吐、命中率、D 步长 @16、传输耗时，与 sglang 逐项对照。
- 诊断脚本改为读 vLLM 指标（`vllm:num_requests_*`、KV 使用率、`vllm:mooncake_xfer_time_seconds`、投机解码接受数）。

### 5.4 阶段 3：两节点 P8D8（2 节点）

- P：TP1 × PP8（`10,10,10,10,10,10,10,8`），D：`-dp 8 -tp 1`，C208，对标 54,468。计分前后记网卡计数器，确认 KV 分在 8 条 rail 上。

### 5.5 阶段 4：调优与超越（A/B，每项先写预测）

| 方向 | 做法 | 预期 / 判据 |
|---|---|---|
| D 侧前缀缓存 + D rank 粘性 | D 开前缀缓存与 KV 事件，router 按 D 侧重叠度选 rank（`--kv-decode-overlap-weight`） | 传输量与等传输明显下降；P4D4 上 D 的 KV 池常满，命中率是未知数 |
| 短增量在 D 上 prefill | 未命中 token 低于阈值的请求不走 P，直接在 D 上算（需改 router） | 按场景 A/B，你决定是否采用 |
| decode MoE | TP8 平铺 对 EP8（MoRI） | sglang TP8 MoE GEMM 带宽利用率仅 52% |
| prefill 调度 | 缓存感知准入窗口；router 给 `priority` 近似 SPF；上游 SPF | 对标 sglang SPF 的 +15% |
| prefill 参数 | chunk 8192 / 16384、层切分、`max-num-batched-tokens`、异步调度 | |
| kernel | 采 decode trace，按 MoE GEMM / sparse MLA / indexer / 通信 / dense GEMM 拆分，与 sglang 的 TP4 / TP8 拆解对比 | 差距大的项进 aiter / vLLM 上游 |
| KF 特性移植 | 按 sglang #47278（sparse MLA split-K）、#47897（paged MQA logits）、#47872（单遍 top-k）、#47276（融合 indexer + top-k）、KF MoE 的顺序，逐项接到 vLLM 的 ROCm DSA 路径（`models/deepseek_v32/amd/`、`v1/attention/backends/mla/rocm_aiter_mla_sparse.py`、`sparse_indexer.py`），每项单独 A/B | 每项走 vLLM 上游 PR |
| MTP K | K3 / K5 按负载选 | 只影响 Pareto 低并发点 |

## 6. 上游计划

| 候选 | 仓库 | 触发条件 |
|---|---|---|
| 阶段 1 查到的 GLM-5.2 / DSA / MTP / PD 正确性问题 | vLLM | 定位并有最小复现 |
| MooncakeConnector 支持 `custom_mem_pool`（AMD GEM） | vLLM（依赖 Mooncake 的 HIP GEM 分配器，#5584 / #5587） | crusoe 上 GEM KV 跑通且收益可量化 |
| SPF（或按未命中 token 排序的准入）调度策略 | vLLM | A/B 证明有收益 |
| Mooncake 生产端逐块发送 KV（PP prefill） | vLLM | trace 显示 prefill 结束后的整体发送拉长 TTFT |
| PP + Offloading + DSA 组合的问题 | vLLM | 阶段 2 / 3 遇到 |
| MoE TP8 形状、sparse MLA、indexer kernel | aiter（vLLM 跟随版本） | 阶段 4 trace 证明差距 |
| router 协议对齐 v0.34、KV 事件端口、镜像升级 | Infera | 阶段 0–2 |

vLLM 流程：先开 issue / RFC 说明问题与数据，PR 带 DCO sign-off、单测与 AMD CI 结果，性能数据引用本分支的目录。

## 7. 资源与顺序

- crusoe spur：`amd-agentx-4-qos` 每人 2 节点。阶段 0–2 用 1 节点，阶段 3 用 2 节点；沿用 `pd-repro-crusoe-20261116/scripts/` 的
  持有作业（自检、坏节点排除、自动重交）、`q.sh` 队列、模型拷到本地 NVMe、网卡计数器脚本。
- 顺序：阶段 0 → 1 → 2 → 3，阶段 4 从阶段 2 通过后开始穿插。阶段 1 是最大的不确定项：如果 v0.34.0 仍有乱码，先投入定位与修复。

## 8. 目录约定（本分支 `llying/`）

每个主题一个目录 `llying/<主题>-<YYYYMMDD>/`，内含 `README.zh-CN.md`（结论、环境、结果、问题与根因、复现命令）、`config/`、`scripts/`、
`patches/`、`results/`，需要时加 `skill/`。大于 500 KB 的日志不入库（pre-commit 限制），放 `/shared_nfs/liyingli/` 并在 README 写路径。
计划中的主题：

- `vllm-pd-plan-20261117/`：本文
- `vllm-image-v0340-<日期>/`：阶段 0
- `vllm-agg-correctness-<日期>/`：阶段 1
- `vllm-p4d4-crusoe-<日期>/`、`vllm-p8d8-crusoe-<日期>/`：阶段 2、3
- `vllm-opt-<方向>-<日期>/`：阶段 4 每个方向一个
- `upstream-vllm-<日期>/`：上游 PR 草稿与跟踪
