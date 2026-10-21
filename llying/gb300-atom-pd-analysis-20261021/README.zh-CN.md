# GB300 Dynamo+SGLang 与 ATOM MI355X PD 的配置分析，及对我们 infera + sglang PD 基线的启示（10-21）

对象：InferenceX 10-20 更新的 GLM-5.2 AgentX **GB300 NVL72 Dynamo+SGLang** PD 点；ATOM 在 MI355X 上的 PD 点（InferenceX
[PR #4045](https://github.com/SemiAnalysisAI/InferenceX/pull/4045)，未合入、未上榜，用其非正式运行 39028510364 的结果）；对照是我们
10-20 至 10-21 在 crsuse2 138 + 140 上的 sweep（`../2p1d-v0522-20261020/`，infera + sglang v0.5.22，2P(TP4)1D(TP8) / 1P(TP4)1D(TP8)）。
所有数字可用 `scripts/compare_pd.py` 复现；图见 `../2p1d-v0522-20261020/figures/pareto-p90-20261021.png`（已加入 1P1D C88 和 ATOM PD）。

## 结论

1. **prefill 每卡能力三家基本持平，差距全在 decode。** 饱和时每张 prefill 卡每秒 0.65–0.77 个请求（GB300 C264 0.77、ATOM C96 0.73、
   我们 1P1D C88 0.69，三者 TTFT p90 都在 9–11 s）。同一 interactivity（p90 ≈ 42）下，ATOM 每张 decode 卡每秒服务 0.73 个请求，我们只有
   0.34；GB300 在 p90 ≈ 90 时仍有 0.98–1.16。
2. **每卡吞吐 ≈ prefill 卡占比 × 每 prefill 卡请求率 × 每请求 token 数**（见第 3 节，四个代表点误差 ≤ 2%）。prefill 能力相同，
   谁的 decode 用更少的卡撑住目标交互，谁的 prefill 占比就更高、每卡吞吐就更高：p90 ≈ 42 时 ATOM 每张 prefill 卡配 1 张 decode 卡，
   我们配 2 张；GB300 在 p90 ≈ 90 时配 0.67 张。
3. **对 ATOM（同硬件）：我们 decode 的单步效率并不差，差在 DP rank 不均衡。** 按"每接受 1 个 token 的步长"比，同等单卡负载下两者相当
   （每卡约 7 个在跑：我们 13.9 ms、ATOM 15.4 ms；约 15 个：22.4 / 20.4 ms）。但我们 DP attention 的每一步等最忙的 rank，最忙 rank 是均值的
   1.6 倍（中位）到 2.4 倍（p90）；ATOM 的 decode 是 TP4 + DCP4，一个 batch，没有这个问题。
4. **对 GB300：同等单卡负载下每接受 token 快约 2 倍**（每卡约 13 个在跑：11 ms 对我们 21.6 ms），来自硬件（FP4 算力约 1.5 倍、NVLink 上的
   EP all-to-all）和 NVFP4 kernel；再加上 decode 按请求数均衡、NVL72 让 decode 跨 tray 做 EP8 / EP16、KV 走 NVLink。机柜级 scale-up
   这部分在 8 卡节点上无法复制。
5. **可借鉴、按投入产出排序**：decode 按在跑请求数均衡（GB300 原样做法）→ prefill 带 MTP 参数（AUS 在做）→ decode 缩到 4 卡（1P1D 4+4，
   当前镜像可测）→ TP4 + DCP4（需新 sglang）→ PP4 prefill、非专家层 FP8、恢复快速 all-reduce。目标见第 7 节：阶段 1（当前镜像）p90 ≈ 45
   处 ≥ 33k、≈ 55 处 ≥ 28k；阶段 2 追平 ATOM PD（p90 ≈ 43 处 ≥ 40k）；阶段 3 追平 B300 单机线（8 卡节点同级）。

## 1. 数据与可比性

- **接受长度相同**：InferenceX 规定 agentic 推测解码一律用 golden AL 模拟接受（`evidence/gb300-dynamo-sglang/glm5.2_mtp-golden-al.yaml`，
  由 srt-slurm 的 `synthetic_acceptance.py` 注入）。GLM-5.2 thinking_on：K=3 → 2.99、K=4 → 3.33、K=5 → 3.61。GB300 全部点 K=5（3.61），
  与我们相同；ATOM 1P1D 用 K=3（2.99），1P2D K=4，1P3D K=5。可以自选 K，但不能换接受长度。
- **数据集与客户端相同**：`semianalysis_cc_traces_weka_062126`，AIPerf，3,600 s 计分；GB300 客户端参数 `enable_thinking=true`、`temperature=1`、
  `top_p=0.95`。各点的平均输入不同（111k–170k，取决于并发与采样），第 3 节的公式按每请求 token 数归一。
- 来源：GB300 recipe `evidence/gb300-dynamo-sglang/disagg-variants.yaml` 与配置项 `nvidia-master-gb300-disagg.yaml`（InferenceX main
  `6046414d`）；结果来自 `../2p1d-v0522-20261020/evidence/inferencex-api-glm52-20261021.json`（10-21 拉取，GB300 PD 点日期 10-20，
  镜像 `lmsysorg/sglang:nightly-dev-20261016-a93f16f0`，dynamo 1.6.0.dev20261016）。ATOM recipe `evidence/atom-pd-pr4045/disagg-variants.yaml`、
  PR 描述 `pr4045.json`、9 个点的汇总 `run39028510364/*.json`（镜像 `rocm/atom-dev:nightly_202610131249`）。
- ATOM 的点是 PR 的非正式运行，PR 自述 ATOM 自己的 CI 在另一个集群上 C112 能到 46.1k（同配置，差距未解释），即集群间约 ±15%。

## 2. GB300 NVL72 Dynamo+SGLang：策略与配置

**按并发换形态**（每个 tray 4 张 B300，prefill 一个 worker 占一个 tray）：

| 点 | 形态 | GPU | tok/s/GPU | intvty p90 | ITL p50 | TTFT p50 / p90 |
|---|---|---|---|---|---|---|
| C264 | 3P(TP1×PP4) + 1D(TP8/EP8/DPA) | 12+8 | **63,647** | 89.4 | 10.3 ms | 2.07 / 9.43 s |
| C216 | 同上 | 12+8 | 55,488 | 94.7 | 9.9 ms | 1.68 / 5.49 s |
| C256 | 4P + 1D(TP16/EP16/DPA) | 16+16 | 44,453 | 114.3 | 8.0 ms | 1.38 / 4.07 s |
| C56 | 1P + 1D(TP8/EP8/DPA) | 4+8 | 28,407 | 131.8 | 7.2 ms | 1.10 / 3.16 s |
| C40 | 1P + 3D(TP4) | 4+12 | 16,924 | 184.8 | 4.4 ms | 0.95 / 2.25 s |
| C30 | 1P + 5D(TP4) | 4+20 | 8,942 | 246.3 | 3.1 ms | 0.68 / 1.83 s |

高吞吐端用多个 prefill 配一个大 EP decode，高交互端用一个 prefill 配多个小 TP4 decode。

**prefill**（每 worker）：
- 并行与调度：TP1 × PP4，`SGLANG_PP_LAYER_PARTITION=20,20,20,18`（注释：默认的 19,19,20,20 把 LM head 和 MTP 草稿层也压在最后一级）。chunk 8,192–16,384，`max-running-requests 16`，关闭 CUDA graph，`mem-fraction-static 0.75–0.88`。
- 缓存：HiCache `write_back`、`hicache-io-backend direct`，100–185 GB。
- 精度、内核与 MTP：NVFP4（`modelopt_fp4`，cutedsl GEMM / MoE）。**prefill 也带 EAGLE 5/1/6 参数**，草稿层的 KV 随 PD 传给 decode，与 AUS 在 `d1e7ff1b` 找到的冷态根因是同一件事。

**decode**：
- 并行与 MoE：TP8 / EP8 / DP attention（2 个 tray）或 TP16 / EP16（4 个 tray）。`moe-a2a-backend flashinfer`（NVLink 上的 all-to-all），MoE kernel `flashinfer_trtllm`，草稿层 MoE 不做 a2a。
- 批大小上限：`cuda-graph-max-bs-decode 32`、`max-running-requests` = 32 × DP（每 rank 最多 32 个，步长有上界）。
- 显存与缓存：`mem-fraction-static 0.9–0.94`，`disaggregation-decode-extra-slots 0`。EP 点关 radix；TP4 多实例点开 decode radix（`disaggregation-decode-enable-radix-cache`）。
- 其他：`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`（准入时少预留新 token），`stream-interval 30`、`skip-tokenizer-init`（减 CPU 开销）。

**router / 传输**：
- router：dynamo KV router，`router-replica-sync`。prefill 按会话亲和（nginx 按 `X-Dynamo-Session-ID`，TTL 3,600 s）。
- **decode 按请求数均衡**：`router-decode-active-request-weight 8192`，注释原话是 "the active-request term dominates KV overlap, so decode picks balance request counts"。
- KV 传输：NIXL，环境里设了 `MC_FORCE_MNNVL=1`、`NCCL_MNNVL_ENABLE=1`，即走多节点 NVLink。

**为什么这么好（量化）**：
1. **decode 单步快**：每卡约 13 个在跑时每步约 40 ms（ITL 10.3 ms × 3.61），我们同等最忙负载约 78 ms（C88 步长表 @13）→ 每接受 token 11 ms 对 21.6 ms。
   来源：B300 NVFP4 稠密算力约 15 PFLOPS（MI355X MXFP4 约 10），HBM 两者都约 288 GB / 8 TB/s；MoE 用 EP + NVLink all-to-all（跨 tray 仍在
   NVLink 域内）；NVFP4 的 trtllm / cutedsl kernel。
2. **decode 不被最忙 rank 拖累**：按请求数均衡 DP rank（我们按在途 token 均衡，见第 5 节）。
3. **大 decode batch 下交互仍高**：每张 decode 卡输出 1,166 tok/s（我们 C176 531），所以 decode 卡可以少，C264 时 prefill 占 60% 的卡。
4. **prefill 并不比我们强**：C264 时每张 prefill 卡 0.77 req/s、TTFT p90 9.4 s，与我们 1P1D C88（0.69、9.45 s）同级。

## 3. 统一的视角：每卡吞吐由 prefill 占比决定

AgentX 每个请求要先过 prefill，所以 **每卡吞吐 ≈ prefill 卡占比 × 每 prefill 卡请求率 × 每请求 token 数（输入 + 输出）**：

| 点 | prefill 占比 | 每 prefill 卡 req/s | 每请求 token | 估算 tok/s/GPU | 实测 | intvty p90 |
|---|---|---|---|---|---|---|
| GB300 3P1D C264 | 12/20 = 0.60 | 0.77 | 140k | 64.7k | 63.6k | 89.4 |
| ATOM 1P1D C96 | 4/8 = 0.50 | 0.73 | 117k | 42.7k | 42.9k | 42.1 |
| 我们 2P1D C176 | 8/16 = 0.50 | 0.55 | 130k | 35.8k | 35.6k | 35.7 |
| 我们 1P1D C88 | 4/12 = 0.33 | 0.69 | 120k | 27.3k | 27.5k | 42.8 |

prefill 每卡上限三家相同（约 0.7 req/s，再往上 TTFT 爆，ATOM C112 / C128 的 TTFT p50 15–16 s 就是例子），所以**提高每卡吞吐的唯一办法是
让 decode 用更少的卡撑住目标交互**：p90 ≈ 42 时 ATOM 是 1:1，我们是 1:2（2P1D C176 虽是 1:1，但 prefill 只跑到 0.55、交互只有 35.7）。
按我们自己的 prefill 能力，1P1D 若 decode 只用 4 卡，上限约 0.5 × 0.69 × 120k ≈ 41k/GPU。

## 4. decode 负载对比（Little 定律：在跑 = 每秒请求 × 平均输出 × 平均 ITL；我们的估算与日志实测一致，C176 104 / 105、C88 54 / 54）

| 点 | decode 卡 | 每卡在跑 | 每步 = ITL × AL | 每接受 token (ms) | 交互 p90 |
|---|---|---|---|---|---|
| GB300 3P1D C264（DEP8，K=5） | 8 | 12.9 | 40 ms | 11.1 | 89.4 |
| GB300 1P1D C56（DEP8） | 8 | 2.7 | 32 ms | 8.9 | 131.8 |
| GB300 1P3D C40（3 × TP4） | 12 | 0.9 | 22 ms | 6.1 | 184.8 |
| ATOM 1P1D C96（TP4+DCP4，K=3） | 4 | 14.7 | 61 ms | 20.4 | 42.1 |
| ATOM 1P1D C56 | 4 | 7.1 | 46 ms | 15.4 | 58.4 |
| ATOM 1P3D C24（3 × TP4，K=5） | 12 | 0.5 | 20 ms | 5.5 | 146.8 |
| 我们 2P1D C176（TP8/DPA，K=5） | 8 | 12.9（最忙 rank 中位 18） | 88 ms | 24.4 | 35.7 |
| 我们 1P1D C88 | 8 | 6.7（最忙中位 11 / p90 16） | 71 ms | 19.7 | 42.8 |
| 我们 1P1D C32 | 8 | 1.5（最忙中位 2 / p90 4） | 40 ms | 11.1 | 73.2 |

我们按最忙 rank 在跑数分组的步长（C88）：@3 34、@7 50、@11 74、@13 78、@15 81 ms → 每接受 token 9.4 / 13.9 / 20.5 / 21.6 / 22.4 ms。由此：
- **同等单卡负载、假设完全均衡时，我们与 ATOM 每接受 token 相当**（7 个 13.9 对 15.4，15 个 22.4 对 20.4）：我们每步更长，但 K=5 每步多接受 21%。
- **实际差 2 倍来自 DP rank 不均衡**：C88 的每卡均值只有 6.7，但每步由最忙的 11–16 决定。
- **低负载下的步长下限差约 1.7–2 倍**：我们 @1–3 约 30–34 ms，ATOM / GB300 的 TP4 decode 16–22 ms。这决定高交互端（C32 73 对 ATOM 1P3D 147）。
  其中一部分是 MTP 冷态（AUS a5-pmtp：@6 48.7 → 36.5 ms、@8 50.4 → 40.0 ms），其余待 profile（TP8 + DP attention 的同步、
  `--disable-custom-all-reduce` 后的 RCCL、5 次草稿前向、triton DSA kernel）。

## 5. ATOM MI355X PD（PR #4045）给我们的启示

**配置**：
- 部署：1P1D 是**单节点 8 卡**，P 在 GPU 0–3、D 在 GPU 4–7，同机 Mooncake RDMA 传 KV。
- prefill：PP4 × TP1（层 20/20/20/18），`max-num-batched-tokens 8192`，eager，**也带 MTP**（与 decode 同 K）。
- decode：**TP4 + DCP4**（latent KV 沿序列切到 4 卡，不复制），`gpu-memory-utilization 0.94`，graph 捕获到 256。K 按并发选：C56–C128 K=3，1P2D K=4，1P3D（3 × TP4，无 DCP）K=5。
- 全局：
  - 精度：FP8 KV、block 16；**非专家层在线 FP8**（`online_quant_config: ptpc_fp8`，专家层保持 MXFP4，MTP 层不量化）。
  - 编译与内核：torch compile level 3 + 完整 CUDA graph；aiter 非 triton MLA（`ATOM_USE_TRITON_MLA=0`）。
  - 通信：`AITER_QUICK_REDUCE_QUANTIZATION=INT4`（TP all-reduce 用 INT4 QuickReduce）。
- 缓存与 router：Mooncake Store 作 prefix cache 下层，1P1D 两个 NUMA 各一个 768 GiB 共享池，走 hugepage。router AToMesh：1P1D round_robin，多 D 时 cache_aware。

**结果（每卡按全部 GPU 计）**：

| 点 | GPU | tok/s/GPU | intvty p90 | TTFT p50 / p90 | 对照（我们） |
|---|---|---|---|---|---|
| 1P1D C96 | 8 | **42,884** | 42.1 | 3.73 / 11.35 s | 1P1D C88：27,453 @ 42.8（12 卡） |
| 1P1D C72 | 8 | 38,487 | 48.8 | 1.93 / 4.00 s | 1P1D C64：23,241 @ 51.4 |
| 1P1D C56 | 8 | 31,067 | 58.4 | 1.45 / 2.85 s | — |
| 1P2D C56 | 12 | 24,030 | 80.0 | 1.34 / 2.71 s | 1P1D C32：14,153 @ 73.2 |
| 1P3D C24 | 16 | 10,073 | 146.8 | 1.06 / 2.19 s | — |

**启示**：
1. **同一硬件上每卡可以高 1.56 倍**（C96 对我们 C88，交互相同），两边总吞吐几乎一样（34.3 万 / 32.9 万 tok/s），差的就是 decode 用 4 卡还是 8 卡。
   这是 MI355X 上一个已被证明的软件目标。
2. **差距的主因是 decode 的 DP 不均衡，不是算子**（第 4 节）。TP4 + DCP4 一个 batch 天然均衡；我们要么均衡 DP rank，要么换没有 DP 拆分的 decode 布局。
3. **decode 4 卡在 KV 上可行**：
   - 我们 decode 每 rank 403 万 token（FP8，179 GB）。TP4/DPA 时权重翻倍（约 118 GB/卡），每 rank 约 300 万，4 卡约 1,190 万。
   - 实际占用（decode 日志 `#token`，8 rank 合计）：1P1D C64 均值 665 万、最大 1,011 万（可行）；C88 均值 1,054 万、p90 1,328 万（不够）。
   - 所以当前镜像下 4 卡 decode 适合 C64–C72，更高并发要 DCP（KV 不复制）或 FP8 非专家层腾显存。
4. **prefill 不是差距**：ATOM 饱和点 0.73 req/s/卡与我们相同。它的 prefill 缓存命中 92.7%（理论 96.4%），低于我们的 95.8%（理论 96.55%）。
   Mooncake Store 这一项对我们收益很小，我们的 HiCache 已接近理论命中。
5. **prefill 带 MTP**：GB300、ATOM 都这样做，与 AUS 的修法一致。
6. **K 可以按并发选**：ATOM 高并发用 K=3。但按第 4 节，我们 K=5 时每接受 token 并不吃亏，K=3 只在大 batch 且计算受限时可能有利，优先级低。
7. sglang 侧的对应物：
   - [PR #4064](https://github.com/SemiAnalysisAI/InferenceX/pull/4064)（MI355X SGLang TP4/EP4 + DCP4 单机点）依赖 sgl-project/sglang#43735 和 ROCm triton 稀疏 MLA 的 DCP 移植 #44322，结果还是 TBD。它也说明 DSA DCP 不能和 HiCache 同用，decode 不用 HiCache，所以无碍。是否已在 v0.5.22 中需确认。
   - [PR #4308](https://github.com/SemiAnalysisAI/InferenceX/pull/4308) 是 ATOM 单机方案换 1018 镜像后的调参：TP4 + DCP4 那一路接每 rank 256 GiB 的 LMCache 内存层，低并发用 mono decode。

## 6. 我们今天的瓶颈（摘要，详见 `../2p1d-v0522-20261020/README.zh-CN.md`）

| 点 | tok/s/GPU @ p90 | 主要瓶颈 |
|---|---|---|
| 2P1D C176 | 35,637 @ 35.7 | decode 最忙 rank 中位 18 / p90 24（均值 13），每步 80–96 ms；decode KV 是约束（73% 时间有 rank > 0.9），R2 必须开 |
| 1P1D C88 | 27,453 @ 42.8 | prefill 饱和（实例忙 81%，rank 均值 57%，排队均值 4.6）；decode 最忙中位 11 / p90 16（均值 6.7） |
| 1P1D C64 | 23,241 @ 51.4 | decode 最忙中位 7 / p90 10（均值 4.2）；prefill 实例忙 78%，rank 33% |
| 1P1D C32 | 14,153 @ 73.2 | decode 步长下限（最忙 2–4 时 35–45 ms） |

共同点：
- decode 交互由最忙 rank 决定。R2 均衡的是在途 token（KV），不是在跑请求数。
- 步长有冷态成分（MTP 草稿 KV 未传）。
- `--disable-custom-all-reduce`（正确性必需）使集合通信走 RCCL。

## 7. 可借鉴的 feature 与预期效果

预期均为按本次实测步长表和第 3 节公式的估算，需逐项 A/B 验证。

| 优先级 | feature（来源） | 我们现状 | 预期效果 | 投入 / 风险 |
|---|---|---|---|---|
| P0 | **decode 按在跑请求数均衡 DP rank**（GB300 `router-decode-active-request-weight`） | R2 按在途 token 均衡，最忙 / 均值 p90 1.8–2.4 | 最忙 p90 压到均值 × 1.3：1P1D C88 每步 82 → ~64 ms，**p90 42.8 → ~55**（吞吐受 prefill 限，~28k）；2P1D C176 p90 35.7 → ~44；1P1D C64 51.4 → ~65–75 | router 小改：在 KV 不超阈值（如 0.85）的 rank 中选在跑最少的，都接近阈值时退回 R2。免改代码的先验：C64 / C88 关 R2 |
| P0 | **prefill 带 EAGLE 参数**（GB300、ATOM；AUS `d1e7ff1b`） | 冷态：@6 46、@8 54 ms | AUS 实测热态 @6 −12、@8 −10、@12 −4 ms；C176 吞吐 +2%。与上一项叠加后 1P1D C88 p90 ~60–65、C64 ~75+ | AUS 在做；prefill 多一层草稿（AUS −0.8%） |
| P0 | **decode 缩到 4 卡：1P1D 4+4，decode TP4/DPA**（ATOM 1P1D 的卡数配比） | 1P1D 用 4+8 | C64：总吞吐不变则 279k / 8 ≈ 35k/GPU；decode 每 rank 8.5 个，最忙约 13–15 → p90 ~38–42（有均衡则 ~49）。**预计 30–35k @ 38–49**，可同时支配 1P1D C88 与 2P1D C176 附近的线 | 当前镜像只改配置；KV 上限约 C64–C72（第 5 节第 3 条）。P 在 138 GPU 4–7、D 在 140 GPU 4–7（rail 对齐） |
| P1 | **decode TP4 + DCP4**（ATOM；sglang#43735 + #44322） | DP attention | 无 rank 不均衡、KV 不复制，4 卡可到 C96：**追平 ATOM PD，~40k @ ~43、~31k @ ~58** | 需新 sglang（ROCm DSA DCP），验证正确性（GSM8K） |
| P1 | **prefill PP4 × TP1**（GB300、ATOM） | TP4 / DP4 attention；C88 实例忙 81% 而 rank 均值 57%，会话亲和把长会话钉在热 rank | 去掉 DP 同步与热 rank，prefill 容量 +15–30% → 1P1D 可到 C96–C104 而 TTFT p90 < 10 s | 需确认 sglang ROCm 的 PP + PD + HiCache + DSA |
| P1 | **恢复快速 all-reduce**（ATOM INT4 QuickReduce；之前的窄修复：只关 custom reduce-scatter / all-gather） | 全走 RCCL | mixed 上关 custom AR 曾使 p90 −15%，恢复后每步约 −10–15% | 正确性优先，需先验证 |
| P2 | **非专家层 FP8**（ATOM `ptpc_fp8`） | 注意力、dense MLP、gate、lm_head 为 BF16（checkpoint `exclude`），DP attention 每 rank 复制约 30 GB | 每步少读 ~15 GB（约 −2 ms）；4 卡 decode 每 rank 多 ~0.34M token（+11%） | 需 sglang 支持或离线重量化 checkpoint；GSM8K 重验 |
| P2 | decode 准入与 CPU 开销（GB300 `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`、`stream-interval 30`、`skip-tokenizer-init`） | 默认 | C176（KV 受限）多收几个请求；CPU 开销小幅下降 | 低成本；`skip-tokenizer-init` 需 router 传 token id，先确认兼容 |
| P2 | decode EP + all-to-all（GB300 DEP8） | MoE TP8（`--ep-size 1`） | 去掉 DP 下的全量 gather / reduce-scatter，GEMM 更大 | 需 ROCm 上的 a2a 后端（如 MORI）在 sglang 可用 |
| P3 | MTP K 按并发选（ATOM K=3） | K=5 | 每步草稿 5 → 3、验证 6 → 4 token，但每步接受 −17%；只有大 batch 且计算受限才划算 | 低优先级，只在 C176 上值得一试 |
| — | Mooncake Store 大共享池（ATOM） | prefill 缓存份额 95.8%（理论 96.55%） | < 1% 重算可省 | 不做 |
| — | NVL72（72 卡 NVLink 域、KV 走 NVLink、Grace C2C 900 GB/s） | 8 卡节点 + 8 × 400G | — | 硬件，不可复制 |

## 8. 目标

**前沿目标**（横轴 interactivity p90 → tok/s/GPU）：

| 阶段 | 依赖 | p90 ≈ 40–45 | p90 ≈ 55–60 | p90 ≈ 70–80 | 对照 |
|---|---|---|---|---|---|
| 今天（基线） | — | 35.6k @ 35.7、27.5k @ 42.8 | 23.2k @ 51.4 | 14.2k @ 73.2 | MI355X 上 C176 吞吐最高；p90 > 50 被 InferenceX 单机 TP4/EP4 压住 |
| 阶段 1（约 2 周） | 当前镜像：decode 均衡、prefill MTP（AUS）、1P1D 4+4 | **≥ 33k** | **≥ 28k** | **≥ 23k** | 接近 B300 单机线（44.9 → 37.4k、61.1 → 29.7k）的 85%–95% |
| 阶段 2（1–2 月） | 新 sglang：TP4 + DCP4、PP4 prefill、快速 AR、FP8 非专家层 | **≥ 40k** | **≥ 31k** | **≥ 24k** | 追平 ATOM PD（42.9k @ 42.1、31.1k @ 58.4、24.0k @ 80.0） |
| 阶段 3（季度） | kernel / 通信：步长下限、EP a2a | ≥ 42k | ≥ 30k @ 60 | ≥ 23k @ 85 | 追平 B300 单机线（每 8 卡节点同级） |

**软件工程指标**（用 `../2p1d-v0522-20261020/scripts/analyze_pd_logs.py` 每点检查）：

| 指标 | 今天 | 阶段 1 | 阶段 2–3 | 参照 |
|---|---|---|---|---|
| decode 最忙 rank / 均值（p90） | 1.8–2.4 | ≤ 1.3 | 1.0（无 DP 拆分）或 ≤ 1.2 | GB300 按请求数均衡；ATOM 单 batch |
| decode 每步 @1–3（下限） | 30–34 ms | ≤ 28 ms | ≤ 20 ms | ATOM / GB300 TP4 16–22 ms |
| decode 每步 @7 | 50 ms | ≤ 40 ms | ≤ 35 ms | AUS 热态 @8 40.0 ms |
| p90 ≈ 42 时每 decode 卡输出 tok/s | 343 | ≥ 500 | ≥ 700 | ATOM 722；GB300 @ p90 ≈ 90 1,166 |
| prefill 饱和点（req/s/prefill 卡） | 0.69 | 0.69 | ≥ 0.85 | 三家现在都约 0.7 |

**硬件侧**：
1. **对标口径**：MI355X 与 B300 的 HBM 相同（288 GB / 约 8 TB/s），FP4 稠密算力约 2/3。8 卡节点上的合理目标是**每节点追平 B300 单机线**。
   GB300 NVL72 的 PD 点依赖 72 卡 NVLink 域（decode EP8 / EP16 跨 tray、KV 走 NVLink、Grace C2C 给 HiCache 900 GB/s）。p90 ≈ 90 处 55–64k
   这一档需要机柜级 scale-up，不是当前集群能做到的，应作为硬件路线图项而不是本项目目标。
2. **当前集群的阻塞项**：
   - 138 GPU 2 卡死，要管理员 `rocm-smi --gpureset -d 2`。不恢复就做不了 16 卡 2P1D，4+4 只能用 GPU 4–7。
   - prefill 内存访问错误（0xfffffffff000，约每 17 个 prefill 实例小时一次）需要找 AMD 一起定位。
   - ionic 对 1 GiB 对齐的 2 GiB MR 注册返回 EINVAL（现靠 `MC_MAX_MR_SIZE=1 GiB` 绕过），应推动驱动修复。
   - 下线后显存回收 10–65 分钟拖慢 sweep。
3. **网络**：8 × 400G 目前没有证据成为瓶颈（decode 侧"等传输"计数包含 prefill 排队和计算，不能单独说明传输慢）；1P1D 4+4 跨节点时只用 4 条
   rail，建议做 4+4 时顺带测 Mooncake 单次传输时延。

## 9. 建议的下一步（待确认后执行）

1. **1P1D 4+4 C64**（当前镜像、只改配置、约 1.5 小时）：验证"decode 4 卡"这一条，预期 30–35k @ 38–49。
2. 1P1D C88 / C64 关 R2 各一次：免改代码地先验证按请求数均衡的方向（R2 关时 router 近似按派发个数轮转）。
3. 等 AUS 的 prefill MTP 修复合入后重测前沿点。
4. 调研 sglang 的 TP4 + DCP4（#43735 / #44322）与 PP4 prefill 在 ROCm 上的可用性，决定阶段 2 是否用 v0.5.22 之后的镜像。
