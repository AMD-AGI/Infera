# GLM-5.2 prefill 计算流与优化候选（MI355X，TP1 × PP4，chunk 8192）

**结论先行**（初稿 10-27 17:40，数字来自 10-23 旧 trace 按中位数的稳态，新 trace 出来后替换）：
1. 稳态下**稀疏 MLA 一项占 prefill 计算约 44%**（每层 5.2 ms，约 0.45 PFLOPS、约 1.9 TB/s 的 gather），而且走的不是调过的路径
   （gfx950 FP8 优化只认每卡 8 / 16 头，我们 TP1 每卡 64 头）→ 第一优先：配置 / 路由（microbench 在排队），再看 kernel 改写。
2. **停顿**：少数 batch 在每个 PP 级上停 0.5–1.2 s（等 HiCache 回载的事件 + 反常慢的索引 H2D），让 PP0 的平均每步比中位数多约 26%。
   → 第二优先：新 trace 定位，再试 HiCache I/O 参数。
3. MoE（17%）已有 8192 token 的 FlyDSL 调优项、trace 与调优表一致；BF16 GEMM（15%）效率 1.0–1.7 PFLOPS，调优上限约 1–2%；
   indexer（logits + top-k，约 10%）和 q / v 吸收 bmm（5%，访存受限）是中等候选。

## 模型结构（`config.json`）

- 78 层，hidden 6144；前 3 层 dense MLP（intermediate 12288），其余 75 层 MoE：256 路由专家 top-8 + 1 个共享专家（追加为第 257 个专家，
  一次 `fused_moe_` 算 9 个），moe_intermediate 2048，专家权重 MXFP4（`per_1x32`）；attention 投影、dense MLP、router 是 BF16。
- MLA：64 头，q_lora 2048，kv_lora 512，rope 64，qk_head 256（nope 192 + rope 64），v_head 256；prefill 用吸收形式
  （q 吸收到 512 维潜空间，和 576 维的 KV 潜向量算 attention，再吸收回 v）。KV cache FP8（`fp8_e4m3`），每 token 576 B。
- DSA：indexer 32 头 × 128 维，top-k 2048；**只有 21 层算 indexer**（`indexer_types`：第 0、1、2 层，之后每 4 层：6、10、…），
  其余层复用前一个 indexer 层的 top-k。PP 切分 20 / 20 / 20 / 18 → PP0 有 7 个 indexer 层（0、1、2、6、10、14、18），PP1–PP3 各 5 个。
- MTP 草稿层（NextN，在 PP3）：prefill 侧 MoE 是 BF16、未调优（`2stage default`）；decode 侧用 FP8（`SGLANG_GLM_NEXTN_MOE_PTPC=1`，只在 decode）。

## 每层算子（prefill 一个 batch ≈ 8192 token，PP0 旧 trace 中位耗时）

| # | 算子 | 形状 | 实现（kernel） | 中位耗时 | 效率 |
|---|---|---|---|---|---|
| 1 | input norm（残差加 + RMSNorm） | 8192×6144 | aiter `add_rmsnorm_quant_kernel` | 0.06 ms | |
| 2 | q_a + kv_a 合并投影 | [8192×6144]·[6144×2624] BF16 | hipBLASLt（未调优，`torch_gemm`） | 0.25 ms | 1.06 PFLOPS |
| 3 | q_a / kv_a norm | | aiter `fused_qk_rmsnorm` | 0.02 ms | |
| 4 | q_b 投影 | [8192×2048]·[2048×16384] BF16 | aiter triton `gemm_a16w16`（256×128×64） | 0.56 ms | 0.98 PFLOPS |
| 5 | **indexer**（21 层） | wq_b 4096×2048、wk 128×6144、weights_proj 32×6144 | hipBLASLt / flydsl；k LayerNorm（CK）、rope、Hadamard、FP8 量化、写 / 取 index-K cache | 每层约 0.3 ms | |
| 5a | indexer logits | 每请求 [q_tokens × kv_len]，32 头 × 128 FP8 | aiter gluon `fp8_mqa_logits` | 0.57 ms / 请求 | |
| 5b | top-k 2048 | 每请求每 query 行 | sglang `topk_packed_kernel`（每行一个 block） | 0.24 ms / 请求 | |
| 6 | q 吸收 bmm | [64 头] [8192×192]·[192×512] | aiter triton a8w8 batched（token-group 预量化） | 0.32 ms | 0.33 PFLOPS，约 2.3 TB/s，访存受限 |
| 7 | rope + 拼接 + 写 KV cache | q [8192,64,512+64]，k [8192,1,576] | aiter triton `fused_qk_rope_cat_and_cache_mla` | 0.18 ms | |
| 8 | **稀疏 MLA** | 8192 query × 64 头 × top-k 2048 × 576 维（FP8） | sglang triton `_sparse_mla_fwd_split_dim_kernel` | **5.2 ms** | 0.45 PFLOPS；gather 9.7 GB → 1.9 TB/s |
| 9 | v 吸收 bmm | [64 头] [8192×512]·[512×256] | 同 6 | 0.32 ms | |
| 10 | o_proj | [8192×16384]·[16384×6144] BF16 | hipBLASLt（未调优） | 0.97 ms | 1.70 PFLOPS |
| 11 | post-attn norm + MXFP4 量化 | | `add_rmsnorm_quant` + `dynamic_per_group_scaled_quant` | 0.08 ms | |
| 12 | router | [8192×6144]·[6144×256] BF16 | flydsl hgemm | 0.05 ms | |
| 13 | 选专家、排序 | top-8 + 共享 | aiter `topk_reg`、opus moe sorting、mxfp4 sort | 约 0.06 ms | |
| 14 | **MoE** | 8192 token × 9 专家，6144 → 2×2048 → 6144，MXFP4 | FlyDSL `mfma_moe1_*` + `gemm2_a4w4_*` | 1.27 + 1.10 ms | 约 2.4 PFLOPS（MXFP4 峰值约 1/4） |
| 14' | dense MLP（第 0–2 层） | gate_up 24576×6144、down 6144×12288 | hipBLASLt（gate_up 未调优） | 1.66 + 0.92 ms | 1.49 / 1.34 PFLOPS |

每个 MoE 层约 10.4 ms，indexer 层另加约 3.5 ms（4 个请求 × (logits + top-k) + 投影）。PP0：20 层 × 10.4 + 7 × 3.5 ≈ 233 ms，与每步中位数
243 ms 吻合（其余是输入准备、embedding、调度拷贝）。

## 优化候选（按 PP0 稳态的可得收益排序）

| 候选 | 针对 | 估计收益（prefill 计算） | 成本 / 风险 | 精度 / 合规 | 状态 |
|---|---|---|---|---|---|
| 稀疏 MLA 换配置或路由（8 warp、每 16 / 32 头一个 program、BLOCK_K 128、i32 页偏移） | 8 | 每快 10% ≈ 4.4% | 低：只挂一个 python 文件到 prefill | 同一算法、fp32 累加，逐元素对比 | microbench 排在 trace 那一步 |
| 稀疏 MLA kernel 改写（相邻 query 的 top-k 高度重叠，可按块合并 gather；或 FlyDSL / CK 版本） | 8 | 可能 20%+ | 高 | 同上 | 看 microbench 后定 |
| HiCache 停顿 | 停顿步 | 平均每步约 26%（PP0） | 中：先定位，再试 `--hicache-io-backend direct`、写策略、索引用 pinned 内存 | 无 | 等新 trace |
| indexer logits + top-k 融合（不写出 [q × kv_len] 的 fp32 分数矩阵） | 5a + 5b | 约 10% 的一部分 | 高：新 kernel | top-k 结果须逐位一致或证明等价 | 待定 |
| q / v 吸收 bmm（访存受限，0.33 PFLOPS） | 6、9 | 约 5% 的一半 | 中 | FP8，与现状同精度 | 待定 |
| prefill BF16 GEMM 调优（5 个形状 × M 8192 / 4096） | 2、5、10、14' | 1–2% | 低：挂一份 CSV 到 prefill | 不变 | 调优排在 trace 那一步 |
| MoE 换更新的 FlyDSL kernel（KernelForge 在 Kimi-K2 形状上 1.24×） | 14 | 17% × 约 0.2 ≈ 3% | 高：要移植 | 不变 | 待定 |
| attention 投影改 FP8（q_b、o_proj、q_a+kv_a 走 ptpc FP8 GEMM） | 2、4、10 | 约 7% | 中 | **改变精度**，InferenceX 合规未确认 | 只测不采用，需用户决定 |
| prefill 草稿层 MoE 改 FP8 | PP3 草稿层 | 看 PP3 trace | 低（decode 已有开关） | 同上，合规未确认 | 待新 trace |

已排除：chunk 16384（10-21 测过，比 8192 差约 3%）、PP 层切分（10-23 测过，−1.0%）、MoE 配置（已调）。

## 复现

- trace 归类：`python3 scripts/prefill_trace_map.py TRACE.json.gz ... [--top N] [--json OUT]`（每 rank 约 25 s、6.6 GB 内存；在 138 上跑）。
  输出：每步主 stream 计算（中位 / 均值、停顿步及其 top kernel）、长 kernel 期间 GPU 上重叠的操作、按模块 / 模型帧 / 叶子帧 / kernel（含中位耗时）/
  GEMM 形状 / kernel 调用点 / 旁路 stream 调用点的时间。
- 稀疏 MLA microbench：`docker run ... -e HIP_VISIBLE_DEVICES=N -v .../scripts:/w --entrypoint python3 infera-sglang:v0522-fdbafb60
  /w/bench_sparse_mla.py [--mode overlap|uniform] [--quick]`。
- prefill GEMM 调优：`gemm-tune/tune_prefill_bf16.sh GPUS`（在 GPU 所在节点上跑）。
