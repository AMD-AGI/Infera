# GLM-5.2 kernel 优化：TP8 / TP4 decode trace 复核、AKLR 评估与优化方案（11-09）

背景：`HANDOFF-20261106.zh-CN.md`、`TP8-KERNEL-HANDOFF.zh-CN.md`、`RESULT.zh-CN.md` 4.1 / 4.6 / 5 节、`QA-20261106.zh-CN.md` 第三节、
`BOTTLENECK-PP8-20261109.zh-CN.md`。AKLR：[forge_task.md](https://github.com/AMD-AGI/AKLR/blob/main/docs/forge_task.md)、
[AKLR#52](https://github.com/AMD-AGI/AKLR/pull/52)（GLM-5.2 TP4 MoE 有 / 无知识库对照）。本次只做了分析，没有占用 GPU、没有改队列。

数据、脚本：`results/opt/kernel-20261109/`（输出）、`opt/scripts/ana/kdeep.py`、`kcat.py`（新脚本）。单位：ms/步 = verify graph 每步 GPU 时间，
各 rank 平均；TB/s = 该 kernel 每步读的权重字节 ÷ 耗时。

## 一、结论

1. **trace 复核与交接文档一致**：原脚本 `dtrace_deep.py` 在 4 组 trace 上重跑，交接文档的表逐项复现（文档 TP8 用的是 083802 那组、TP4 用 232136 那组）；
   新写的 `kdeep.py` 换了一种方法（按 CPU 注解识别 graph、按每层的 all-gather / reduce-scatter 切层、逐层逐 kernel 对位），结果也一致。
   但有 **6 处需要更正或补充**（第二节 2.3），最重要的两条：
   - **TP8 的 MoE M 是按最忙 DP rank 补齐的 `8 × graph_bs × 4`，trace 时是 640–768**（真实 token 只有 456–508），而 KF MoE 的门控是 M 97–455，
     所以 **trace 里 TP8 的 KF MoE 一次都没走**，看到的是纯 aiter 基线；TP8 需要覆盖的 M 是 384–1024。
   - MoE 之外还有两块之前没列为优化项、TP4 和 TP8 都有的空间：**MLA absorb BMM**（每层 40 µs 只读 14.7 MB，0.36 TB/s，每步 3.1–3.2 ms）和
     **稠密投影 GEMM**（o_proj / q_b / fused_qkv_a 只到 4.2 / 3.6 / 2.2 TB/s，每步 6.1–6.3 ms）。
2. **要优化的 kernel（按 TP8 收益排序，第三节）**：① TP8 MoE 全链路（约 5.5–6.5 ms/步，置信度高，TP4 已经做到）；② MLA absorb BMM（约 1.5 ms）；
   ③ 稠密投影（约 1.5 ms；改 FP8 可到 3–4 ms，但改精度）；④ sparse MLA（TP8 每行比 TP4 慢 1.5 倍，约 1.5 ms，原因未明）；⑤ indexer
   （fused2 在 K3 下大多用不上 + 21 层的小 kernel 链，约 1.5–2 ms）；⑥ 通信与 norm 融合（约 1.5 ms）。TP8 理论合计约 14 ms/步（verify 42 ms 的 1/3），
   按六成兑现约 8 ms；TP4 理论约 7 ms、兑现约 4 ms（TP4 的 MoE 已在 6.1–6.4 TB/s，空间主要在 MoE 以外）。
3. **prefill 侧没有任何 kernel 级数据**（集群上只有 decode trace）。当前最高的 PP8 配置受 prefill 限制，粗估长上下文 prefill 里 indexer
   （每个新 token 对全部约 10 万–18 万上下文打分）可能是最大头。decode kernel 变快先体现为交互度，要转成吞吐需要 prefill 先解开，所以 **prefill profile 应和
   decode kernel 并行启动**（第三节 3.3）。
4. **AKLR 能用，而且和我们的问题贴得很近，但不能直接拿来跑**（第四节）：
   - 能直接用的：知识库里有 GLM-5.2 TP8 MoE（inter 256）的 case study 和 decode MoE 技术页、GLM-5.2 稠密 GEMM 形状的 case study；库里有 GLM-5.2 的
     o_proj FlyDSL kernel（M≤128 对 aiter 1.11–1.41x）、整层 TP8 MoE Triton 插件（M 128 / 256 / 1024+ 对 SGLang 1.06 / 1.16 / 1.33–1.42x）、
     GLM 形状的 DSA sparse MLA decode 插件和 indexer 前处理插件；工具链 `aklr validate`（graph 计时 + `--stress`）、`AKLR_EXPORT`、
     GLM-5.2 的模型级 micro-bench（`profile_decode.py` / `profile_prefill.py`）、`forge_task` + KernelForge 自动调优（`--anchor start` 可以对我们自己的
     生产 kernel 计分）。
   - 不能直接用的原因：基线是原版 aiter（`dbd8bf5`）/ 原版 SGLang，不是我们的 KF + 调优表 + aiter `2bb5491`；`n_tokens` 等变量轴按精确值匹配，
     库里只有 2 的幂，**我们的 M（TP8 640–768、per-rank 72–96）几乎都不在 Point 上**，命中不了；fused_moe 契约只要求 SQNR ≥ 13 dB，比我们的门禁松；
     AKLR 的运行时 / SGLang 插件本质也是启动时替换函数，**InferenceX 大概率同样不接受**，最终仍要以 aiter / SGLang PR 交付。
   - AKLR#52 说明的是：知识库让同一个 agent 在 GLM-5.2 TP4 MoE 上用一半时间和花费达到同样的 1.165x（对 aiter，13 个 M 平均）。对我们，
     TP4 MoE 已经被 KF 做到 6 TB/s 以上，再跑意义不大；**TP8 MoE 正好是知识库有 case study、库里有半成品、我们有明确 M 范围的任务**，最适合用它。
5. **方案（第五节）**：先花 1–2 天搭好"工作负载表 + 生产基线 + 单节点复现 + prefill profile"，然后 TP8 MoE 走"手工移植 KF（与 yihou）+ AKLR forge
   （对生产基线计分）"两条线并行，再做 attention 侧（absorb BMM、sparse MLA、稠密投影），最后做融合与通信；每项都过"算子 SQNR / `--stress` →
   graph 内 kdeep 对位 → `verify_suite` → AgentX"四级门禁，交付走 aiter / SGLang 上游。

## 二、trace 复核

### 2.1 数据与方法

| 组 | 配置 | 路径（`/perf_apps/liyingli/vfix-logs/opt/trace/`） | rank × 步 |
|---|---|---|---|
| TP8-1 / TP8-2 | 2P1D C208，decode TP8 / DP8，K3 | `opt-trace2p1d-c208/traces/decode-083302`、`decode-083802` | 8 × 38 |
| TP4-1 / TP4-2 | 4+4 C104，decode TP4 / DP4，K3 | `opt-traceC-c104/traces/decode-231636`、`decode-232136` | 4 × 38 |

- 原脚本重跑：`dtrace_deep.py`（输出 `results/opt/kernel-20261109/dtrace_deep_rerun.txt`）。
- 新脚本 `kdeep.py`：graph 类型取自包住 `hipGraphLaunch` 的 CPU 注解（`step[VERIFY bs=N]` / `draft` / `draft_extend`），不依赖"每 3 次 launch 一组"的假设；
  每步各 rank 的 bs 也从注解读出；verify graph 在每个 reduce-scatter 之后切层，同 kernel 序列的层归为一个"签名"，逐位置给出平均耗时。
  `kcat.py` 在它的输出上做细分类并算带宽（未归类的 kernel ≤ 0.08 ms/步）。计算都在 n10-29（35341 持有、空闲）上跑，只用 CPU。

### 2.2 与交接文档一致的部分

| ms/步 | 文档 TP8 | 重跑 TP8-1 / TP8-2 | 文档 TP4 | 重跑 TP4-2 / TP4-1 |
|---|---|---|---|---|
| MoE gemm1 | 6.91 | 6.88 / 6.92 | 10.96 | 10.96 / 10.12 |
| MoE gemm2 | 4.63 | 4.59 / 4.63 | 5.66 | 5.66 / 5.24 |
| MoE reduction | 1.03 | 1.00 / 1.03 | 0 | 0 / 0 |
| 排序 / 量化 | 2.50 | 2.51 / 2.50 | 1.11 | 1.11 / 1.10 |
| MoE GEMM 带宽 | 4.18 TB/s | 4.21 / 4.18 | 5.81 | 5.81 / **6.29** |
| all-gather（传输 + 等待） | 2.65（2.03 + 0.63） | 2.82（1.98 + 0.84）/ 2.65（2.03 + 0.63） | 2.83 | 2.83 / 2.36 |
| reduce-scatter | 2.74（2.39 + 0.35） | 2.69 / 2.74 | 2.32 | 2.32 / 1.95 |
| sparse MLA 类 / indexer 类 | 8.05 / 3.85 | 7.92 / 3.87；8.05 / 3.85 | 6.82 / 3.36 | 6.82 / 3.36；5.72 / 3.24 |
| dense GEMM 类 / norm | 8.12 / 1.15 | 8.08 / 1.15；8.12 / 1.15 | 7.89 / 1.19 | 7.89 / 1.19；7.54 / 1.17 |

- "TP8 MoE 三个部件都没用上 KF、MoE GEMM 只有 52% 峰值带宽、空间约 5.6 ms/步、通信 TP4 / TP8 基本相同"这几条结论成立。
- 文档没引用的 TP4-1 那组里，KF 的 MoE GEMM 到了 6.29 TB/s（gemm1 6.37、gemm2 6.16），说明 TP4 的 MoE 基本到顶了。

### 2.3 更正与新发现

**1. TP8 的 MoE M 是补齐后的 640–768，trace 中 KF MoE 根本没被调用。**
- 每步各 rank 的 bs（注解）：TP8-1 合计 116–127、最大 20–22（例：`[10, 14, 12, 14, 20, 21, 10, 16]`），TP8-2 合计 114–118、最大 22–23；最大 / 平均 1.38–1.53。
- DP attention + CUDA graph 下所有 rank 用同一个 graph bs（≥ 最忙 rank），MoE 的输入是 8 个 rank all-gather 后的 `8 × graph_bs × 4` 行。
  decode 日志（`dprof-c208-1104T0758/.../glm52_2p1d-decode.log`）里 graph 捕获时 KF 的决策正好是这些值：
  `KF: M = 128 … 448`（graph bs 4–14），`orig: M = 32 64 96 512 576 … 1024`（graph bs 1–3、16–32）。
- trace 时 graph bs 20–24 → **M 640–768，全部走 aiter**，trace 里没有一步走过 KF。交接文档"即使走了 KF，TP8 的 gemm1 用的也是 aiter 原版"
  这句从 trace 看不出来，但代码证实它成立：KF 的专用 gemm1 只在 `2 × D_INTER == 1024`（inter 512）时启用（`kf_moe.py` `_g1_engages`），
  inter 256 时 KF 只改了 aiter 查表用的 M 键（`get_padded_M` → token=128 行）和前处理排序，gemm1 / gemm2 仍是 aiter 为 i256 表选的 kernel。
- 门控上限 455 的来源：KF 的 HIP 前处理排序要求 `M × topk ≤ 4096`（`kf_moe.py` `_HIP_SORT_MAXP`）。**放宽门控之前先要改这个排序**。
- 补齐的代价：C208 时 MoE 里约 30–35% 的行是补齐行。MoE 是读权重受限，补齐行几乎不多读权重，但 all-gather / reduce-scatter 的字节、
  排序和 attention 的行数都按补齐后算。DP 均衡（router 权重）降低最大 / 平均比，对所有这些都有好处。
- 另外，KF 是按 yihou 的 **TP4 + K5** 部署调的：`__init__.py` 注明 sparse MLA / indexer 只在 `T = 6 × bs` 的 verify 行上验证过，`kf_moe.py`
  写的是 `M = 24 × bs`（4 rank × 6 token）。我们是 K3（每请求 4 行），TP8 又是另一种形状，有几处落在它的调优范围外（见第 5 条）。

**2. MoE 前处理是 3.45 ms/步、7 个 kernel，不是 2.5 ms。** 原分类把 router GEMM（`hgemm_bf16_t64x64x64x6_ksd_w4x2x1`，12.6 µs × 75）算进了 dense GEMM。
TP8 每层：router 12.6 + top-k 5.4 + 追加 shared expert 4.3 + opus 排序两段 13.9 + MXFP4 激活量化 4.8 + scale 排序 5.0 = **46 µs**；
TP4（KF）：router 9.2 + top-k 5.1 + 追加 4.4 + `glm52_moe_sort_fill` 4.9 = **24 µs**（激活量化在 KF gemm1 里 inline）。

**3. "sparse MLA 类"8 ms 里有 3.2 ms 是 MLA absorb BMM。** `_fused_fp8_bmm_rope_cat_and_cache_mla_kernel`（q_nope × W_kc + RoPE + cat + 写 KV cache，
23 µs/层）和 `_batched_gemm_a8w8_…_prequant…`（× W_vc，16.6 µs/层）两个 Triton kernel 每层只读 6.3 + 8.4 MB 的 FP8 权重，等效 0.36 TB/s，
TP4 也一样（3.1 ms/步）。这块和 TP 无关，之前没有作为优化项。

**4. sparse MLA 在 TP8 下每行慢 1.5 倍，"按最大 batch 补齐"解释不了。** `_sparse_mla_split_k_kernel` 每次调用：TP8 57.5 µs（88–96 行），
TP4-1 30.9 µs（72 行）、TP4-2 41.6 µs（96 行）——每行 TP8 约 0.64 µs、TP4 约 0.43 µs。TP4 同样按最大 bs 补齐（最大 / 平均 1.22–1.74），所以补齐
不是差异来源。可能的原因（未证实）：KF 的 split-K plan 按行数选择、88 行落在不好的 plan 上；两组负载的 KV 分布不同；KV 池大小（TP8 每 rank 409 万 token，
TP4 309 万）导致的 gather 局部性差异。按 TP4 的每行开销，TP8 约可省 1.3–1.6 ms/步。

**5. indexer 的 fused2 在 K3 下大多用不上，原因是分组按 K5 设计。** fused2 的 `_LPT_GROUP_ROWS = 6` 对应 K5 每请求 6 行；K3 的行数是 `4 × graph_bs`，
只有 graph bs 是 3 的倍数（且 ≤ 24）时才满足"6 的倍数且 ≤ 96"。trace 印证：TP8-1 graph bs 22 → 88 行 → fused2 0%；TP8-2 部分步 graph bs 24 → 96 行 →
25%；TP4-1 graph bs 18 → 72 行 → 70%。每次调用 preshuffle 116 µs、fused2 82–106 µs。修法比 QA 里的"补齐到 6 的倍数"更直接：做一个按 4 行分组的 K3 版本。

**6. 通信里"等最慢 rank"的部分集中在 21 个 indexer 层之后。** TP8 普通层 all-gather 30 µs、indexer 层 52.7 µs；TP4 23.0 / 43.8 µs。
indexer 的工作量与上下文长度成正比、各 rank 不同，下一次 all-gather 就要等最慢的那个，每步约 0.45–0.5 ms。按"上下文 token"而不只是"请求数"做 DP 均衡，
或让 indexer 对长度不敏感，可以收回一部分。

另：TP8 每步 verify 41.9 + draft 1.6 + draft_extend 1.7 = 45.2 ms，步长中位 46.1 ms，graph 之外的空闲 / eager 约 1 ms（2%），与 4.1 节一致，不再投入。

### 2.4 一个普通 MoE 层（不带 indexer，57 层 / 步）的逐 kernel 对位

| 环节 | TP8 kernel（aiter 回退） | TP8 µs | TP4 µs（KF） | 读权重 MB（TP8 / TP4） | TP8 TB/s |
|---|---|---|---|---|---|
| input add + RMSNorm | `add_rmsnorm_quant` | 4.4 | 4.5 | | |
| fused_qkv_a（6144→2624） | `hgemm_bf16_t64x64x64x6_ksd` | 14.8 | 14.4 | 32.2 | 2.18 |
| q / kv a-norm | `fused_qk_rmsnorm` | 4.3 | 4.4 | | |
| q_b_proj（2048→16384） | `hgemm_bf16_t128x64x64x4` | 18.8 | 18.0 | 67.1 | 3.56 |
| W_kc absorb + RoPE + cat + 写 KV | `_fused_fp8_bmm_rope_cat_and_cache_mla`（Triton） | 23.1 | 20.5 | 6.3 | 0.27 |
| sparse MLA | `_sparse_mla_split_k` + `reduce_dq` | 57.9 + 4.5 | 30.9 + 4.5 | | |
| W_vc absorb | `_batched_gemm_a8w8_…`（Triton） | 16.6 | 16.4 | 8.4 | 0.51 |
| o_proj（16384→6144） | hipBLASLt `MT160x96x64` + `PostGSU8` | 42.5 + 5.1 | 40.0 + 4.7 | 201.3 | 4.23 |
| post-attn add + RMSNorm | `add_rmsnorm_quant` | 4.6 | 4.6 | | |
| all-gather | aiter `allgather_vec` | 30.0 | 23.0 | | |
| router GEMM（6144→256） | `hgemm_bf16_…_w4x2x1` | 12.6 | 9.2 | 3.1 | |
| top-k / 追加 shared | `topk_reg` / `_fused_append_shared_experts` | 5.4 + 4.3 | 5.1 + 4.4 | | |
| 排序 / 激活量化 / scale 排序 | opus P0_v2 + P23 / `dynamic_per_group_scaled_quant` / `mxfp4_moe_sort` | 5.5 + 8.5 + 4.8 + 5.0 | KF `glm52_moe_sort_fill` 4.9 | | |
| gemm1（gate_up + SiLU + 量化） | aiter `mfma_moe1_…_t64x64x256` / KF `gemm1_a4w4_port_…_i512_bm16` | 91.4 | 133.7 | 429.5 / 859.0 | 4.70（TP4 6.43） |
| gemm2（down） | aiter `gemm2_…_bm64_…_reduce` / KF `…_bm16_…_atomic` | 61.1 | 69.1 | 214.7 / 429.5 | 3.51（TP4 6.21） |
| MoE reduction | `moe_reduction_bf16_t9` / 无 | 13.4 | 0 | | |
| reduce-scatter | aiter `reduce_scatter_split_first_dim` | 35.1 | 24.8 | | |
| **合计** | | **473.6** | **437.0** | | |

- TP8 的 MoE 部分 212 µs 读 644 MB（3.0 TB/s），TP4 226 µs 读 1,288 MB（5.7 TB/s）：**TP8 每卡权重减半，MoE 时间几乎没减**。
- 带 indexer 的 18 个 MoE 层每层另加约 219 µs：wq_b / wk / weights_proj 三个小 GEMM 20 µs、LayerNorm / RoPE / 两次 Hadamard / 量化 / 写 index-K cache /
  乘法约 37 µs、logits 116 µs（preshuffle）、top-k 三段 46 µs，以及上面第 6 条的 all-gather 等待。
- 前 3 个 dense 层的 MLP（TP8 切分）、lm_head、logits all-gather 合计 < 0.4 ms/步，不是重点。

## 三、需要优化的 kernel

### 3.1 decode：清单与收益（理论值，按 TP8 排序）

| # | 对象 | TP8 ms/步 | TP4 ms/步 | 现状 | 目标 | TP8 可省 | TP4 可省 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| 1 | **TP8 MoE 全链路**（router → 排序 / 量化 → gemm1 → gemm2 → reduction） | 15.98 | 17.79 | GEMM 3.5–4.7 TB/s，前处理 46 µs / 层，另有单独 reduction | KF 在 TP4 的水平：GEMM ≈ 6 TB/s、atomic gemm2、前处理 ≈ 24 µs / 层 | **5.5–6.5** | ≤ 0.5 | 高：TP4 已实测做到 |
| 2 | **MLA absorb BMM**（W_kc + RoPE + cat + 写 KV；W_vc） | 3.19 | 3.09 | 40 µs / 层只读 14.7 MB | ≈ 18 µs / 层 | 1.5–1.8 | 1.5–1.7 | 中：下限需 microbench |
| 3 | **稠密投影**（o_proj / q_b / fused_qkv_a，BF16） | 6.34 | 6.14 | 4.2 / 3.6 / 2.2 TB/s | 5.5 / 4.5 / 4.0 TB/s | 1.5–1.8 | 1.4–1.7 | 中高；改 FP8 可省 3–4（改精度） |
| 4 | **sparse MLA decode** | 4.79 | 3.18 | TP8 每行 0.64 µs，TP4 0.43 µs | 先追平 TP4 的每行开销 | 1.3–1.6 | 0.3–0.8 | 中：原因未明 |
| 5 | **indexer logits**（fused2 覆盖） | 2.43 | 1.89 | K3 下 fused2 只在 graph bs 为 3 的倍数时可用 | K3 的 4 行分组版本 | 0.6–0.8 | 0.2–0.4 | 高 |
| 6 | **indexer 前处理 + top-k**（21 层 × 约 13 个小 kernel） | 2.16 | 2.13 | 多数在 4–5 µs 的启动地板上 | 融合成 2–3 个 kernel | 1.0–1.3 | 1.0–1.3 | 中 |
| 7 | **通信**（all-gather / reduce-scatter） | 5.60 | 4.91 | 传输约 4.4 + 等待约 1.2 | FP8 载荷 all-gather、RS + add + RMSNorm 融合、indexer 负载均衡 | 1.0–1.5 | 0.8–1.2 | 中低：要改模型侧结构 |
| 8 | **norm 融合**（234 次 / 步，每次约 4.4 µs） | 1.04 | 1.06 | 启动地板 | 并入 RS 或 GEMM 收尾 | 0.3–0.5 | 0.3–0.5 | 中 |
| | **合计（理论）** | **42.0** | **40.7** | | | **约 14** | **约 7** | |

- 兑现率按六成估：TP8 约 8 ms/步（verify 的 19%），TP4 约 4 ms/步（10%）。各项基本独立（不同 kernel），但 #1 与 #7 都涉及 MoE 输入的格式（量化放在 all-gather 前还是后），
  要一起设计。
- **与 RESULT 第 5 节"TP4 截距已在带宽 roofline 上"的关系**：那是步长拟合的截距（零负载外推）。kernel 级看，在实际负载下 o_proj、q_b、absorb BMM 等都离带宽上限很远，
  TP4 仍有约 4 ms/步可做，主要在 MoE 以外。
- 端到端：步长下降先体现为同负载下的交互度（TP8 约 +15–20%）。PP8 / 2P1D 受 prefill 限制，吞吐要等 prefill 解开；单节点 4+4 是 decode KV 与 prefill
  同时接近上限，decode 变快可以在同交互度下加并发。
- 草稿 graph（每步 3.3 ms）受 InferenceX Check 13 约束（草稿不能改精度），只能做等精度的 kernel 优化，收益小，不列入。

### 3.2 每一项的做法要点

1. **TP8 MoE**：KF 的手段基本可以参数化移植到 inter 256：(a) 为 (6144, 256, 257) 在 M 384–1024 上重新选 block_m 16 / 32 的配置行（KF 的 token=128 键只在 inter 512、M ≤ 512 上测过）；
   (b) HIP 前处理排序放宽 `M × 9 ≤ 4096` 的限制（M 到 1024 即 9,216 对，每线程多处理几对或多 CTA）；(c) gemm2 用 atomic 实例（inter 256 时 K 只有一个
   256 的 tile），去掉单独的 `moe_reduction`；(d) gemm1 的专用 body（BM16 / BN128 / BK256、inline 量化，现在由 `_g1_engages` 限定 inter 512）放开到
   inter 256 并重调；(e) 门控放宽到 1024，
   K5 时 M = 48 × graph_bs 另列。AKLR 的 TP8 kernel（`4fbd9d02`，把排序折进 stage 1）和整层 Triton 插件可以作为起点或对照（第四节）。
2. **MLA absorb BMM**：64 头、每头 [rows, 192] × [192, 512] 和 [rows, 512] × [512, 256]，FP8 权重 + per-token 激活量化，rows = 4 × graph_bs。
   权重总共 14.7 MB，理论上每个 BMM 几 µs；现在的时间主要是 Triton kernel 的块划分（BLOCK_M 64 对 72–96 行）和 RoPE / cat / 写 cache 的附带工作。
   AKLR 没有这两个 GLM 形状的 kernel，需要新写 schema。
3. **稠密投影**：先用 AKLR 现成的 o_proj FlyDSL kernel、qkv_a / n6144_k2048 的 Triton kernel 在我们的 M（4 × graph_bs，72–96 行）上对生产的 hipBLASLt
   调优结果测一次；q_b_proj（16384 × 2048）就是 AKLR Rosetta Stone GEMM 任务的形状（1.30x，kernel 未入库，可以向 AKLR 团队要）。FP8 路线即
   QA-20261106 交接项 1、2，空间更大但改精度、要过 eval。
4. **sparse MLA**：先做 1 小时 microbench：同一个 KF kernel 在 72 / 88 / 96 行、真实索引分布下的耗时，确认是 plan 选择还是数据分布；再和 AKLR 的
   `dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048` 插件比（它的基线是原版 SGLang，不是 KF）。
5. **indexer logits**：fused2 的 4 行分组版（交接项 3 的修法改为按 K3 设计），逐元素对比 preshuffle 结果；交给 yihou 配合他的 #47897。
6. **indexer 前处理**：AKLR 的 `dsa_indexer_prepare_…_q_lora_rank2048_n_heads32` 与 `dsa_indexer_key_prepare_model_dim6144` 插件正是这条链
   （对原版 SGLang 2.2–5.5x），直接测；top-k 三段（est / collect / select，46 µs / 层）看 yihou 的 #47872。
7. **通信**：all-gather 在补齐后的 640–768 行上搬 BF16；把 MXFP4 / FP8 量化提到 all-gather 之前（router 改在本 rank 的 token 上算，再 gather top-k）可减半以上字节；
   reduce-scatter 与下一层 input add + RMSNorm 融合省一次启动。这两项都要动 SGLang 的 DeepseekV2 层结构，适合先用 AKLR 插件形式做原型。
8. **norm**：随 7 一起做。

### 3.3 prefill（未测，建议并行做 profile）

- 现有数据只有 decode trace；PP8 计分（56,593）的瓶颈在 prefill（`BOTTLENECK-PP8` 第一节）。
- 粗估一个 8,192 token 满块（命中约 97%，上下文 10 万–18 万）每层的计算量：稠密投影约 2.5 TFLOP（BF16）、MoE 约 5.6 TFLOP（MXFP4）、sparse MLA 约 2.3 TFLOP，
  而 **indexer 每层约 7–12 TFLOP（FP8，O(新 token × 上下文)）**，还要对每行 10 万以上的分数做 top-k。按可达算力折算，一个 PP4 级（约 20 层）的计算下限
  约 100–130 ms，实测单级 250–274 ms。这只是量级估计，但说明 prefill kernel（尤其 indexer）可能比 decode kernel 更直接影响吞吐。
- 做法：AKLR 的 `profile_prefill.py --ISL 150000 --prefix-cache-hit 0.97`（单节点，逐算子）或在 PP8 运行里对一个 prefill rank 做 torch profile
  （和 `BOTTLENECK-PP8` 第四节第 1 项的定位运行合并）。

## 四、AKLR 评估

### 4.1 AKLR 提供什么

| 组成 | 内容 | 对我们的用处 |
|---|---|---|
| `aklr/ops`、`aklr/plugins/sglang` | 每个算子有契约（输入输出、误差界）、参考实现、基线、按形状存放的 kernel JSON（源码内嵌、CI 测出的每个 Point 的 speedup） | 现成 kernel 与 schema 模板 |
| `python -m aklr validate` | audit（无状态、不改只读输入）+ bench（CUDA graph 回放、每次刷 L2、32 轮单侧下界）；`driver.py --stress` 在少 CU、并发下查 kernel 是否互等 | 严格的算子级计时 / 正确性工具，可以直接测我们自己的 kernel |
| `AKLR_EXPORT=1` + `python -m aklr export` | 从一次真实运行导出算子的实际调用形状 | 生成"我们的 M"工作负载表 |
| `aklr/benchmarks/models/glm52` | GLM-5.2 的 `profile_decode.py`（支持 TP8 / DP8、EAGLE、`SGLANG_SIMULATE_ACC_LEN`、`--full-step`、`--trace`）和 `profile_prefill.py`（`--prefix-cache-hit`） | 单节点复现 decode 拆解，补上 prefill 拆解 |
| 知识库（Rosetta Stone，`harness/skills/aklr-kb`） | 123 页、1,204 条带证据的发现：GEMM、MoE、attention、gfx950 / CDNA4 ISA、16 个已交付 kernel 的 case study | 写 kernel 时直接查；forge 默认加载 |
| `forge_task` + KernelForge（Hyperloom 的 AKLR 分支，公开） | schema → 任务 → 多轮 agent 调优（每轮 20–70 min，约 $10 / h GEMM、$17 / h MoE）→ ablate / finalize / stress / distill → publish；`--schema-file` 自定义工作负载，`--anchor start` 对起始 kernel 计分 | 自动调优主力；`--anchor start` 解决基线问题 |
| `geak-campaign` | 用 GEAK 代替 KernelForge 跑同样的流程 | 备选 |

本机已有 `~/.local/bin/claude`（是否已登录未查），Hyperloom 仓库可访问。

### 4.2 与 GLM-5.2 直接相关的现有资产

| 我们的 kernel | AKLR 中的对应 | 现有结果（CI，speedup） | 在我们的 M 上 |
|---|---|---|---|
| TP8 MoE（算子级，不含 router） | `fused_moe_shuffled_16x16_model_dim6144_n_experts257_topk9_inter_dim256`：`1fee5982`、`4fbd9d02`（GEAK，排序折进 stage 1，atomic stage 2）；case study `cases/fused_moe_shuffled_16x16.md`、`moe/decode.md` | m 1 / 2：2.47 / 1.99；m 8–64：1.09–1.13；m ≥ 128：0.32–0.86，不服务 | M 640–768 无 kernel |
| TP8 MoE（整层，含 router 与 shared） | 插件 `deepseek_v2_mxfp4_moe_shuffled_16x16_model_dim6144_n_routed256_inter_dim256`（Triton，9 个文件） | m 128 1.055、256 1.16、1024 1.36、2048 1.33、3072 1.40；512 / 768 没有 Point | 无 Point，可作起点 |
| TP4 MoE | 算子级 `…inter_dim512` 只有 schema（AKLR#52 的实验任务）；整层插件 `…inter_dim512` | 插件 m 256 1.04、1024 1.16 | KF 已 6.1–6.4 TB/s，空间小 |
| o_proj（6144 × 16384） | `gemm_n6144_k16384`：FlyDSL `e8a254fd`、`8f7ffe4c` | m 1–128：1.11–1.41；256–1024：1.03–1.10 | 72–96 行不在 Point 上，需补测 |
| q_b_proj（16384 × 2048） | `gemm_n16384_k2048` 只有 schema；Rosetta Stone GEMM 任务就是此形状（1.30x，未入库） | — | 可向 AKLR 团队要 kernel |
| fused_qkv_a（2624 × 6144） | Triton（GEAK）`b88c8d5f`、FlyDSL；case `bf16_glm52_wide_n.md` | Triton m 2 / 4 / 8 / 64：1.05–1.18 | 需补 72–96 |
| router（256 × 6144）、indexer wk / weights_proj / wq_b | `gemm_n256_k6144`、`n128_k6144`、`n32_k6144`、`n4096_k2048`；case `bf16_glm52_narrow_n.md` | 只在 m ≤ 8 或 ≥ 2048 赢 | 不覆盖 |
| sparse MLA decode | 插件 `dsa_sparse_mla_decode_n_heads64_kv_dim576_topk2048`（Triton） | n_tokens 8–32：1.03–2.16（对原版 SGLang） | 行数与 KV 池大小（n_slots 65,536）都对不上 |
| indexer 前处理 | 插件 `dsa_indexer_prepare_model_dim6144_q_lora_rank2048_n_heads32`、`dsa_indexer_key_prepare_model_dim6144` | 3.2–5.5 / 2.2–2.7（对原版 SGLang） | n_tokens 1–256 有 Point，需补我们的值 |
| indexer logits | `fp8_paged_mqa_logits_page_size64_n_heads64`（DeepSeek-V4，64 个 index 头） | 1.04–1.88 | 形状不同（GLM-5.2 是 32 头），只能借鉴方法 |
| add + RMSNorm（6144） | HIP（GEAK）、Triton | 1.02–1.11 | 小 |
| MLA absorb BMM | 只有其他模型形状的 `bmm_*` schema | — | 需新写 |

完整列表：`results/opt/kernel-20261109/aklr_glm_related_speedups.txt`。

### 4.3 AKLR#52 这个 case 说明什么

- 内容：GLM-5.2 TP4 MoE（inter 512，13 个 M 从 1 到 4096），同一个 KernelForge agent（Claude Opus 5.5）各跑 15 小时，有知识库到 1.165x 用 7h24m / $87，
  没有用 14h54m / $170；8 小时时 1.176x 对 1.047x。只改文档，没有交付 kernel。
- 加速来源：agent 在第一轮就从知识库里找到"TP8 形状的 GLM-5.2 MoE"的 case study（即 4.2 表的 i256 kernel），照着写出 1 / 2 / 4 token 的两次启动路径。
- 对我们的启示：
  - 分数是 13 个 M 的平均，提升主要在小 M（m = 1 时 2.47x）；我们关心的是 2–3 个 M 区间，必须"一个负载区间一次运行"，用我们自己的 M 列表。
  - 基线是 aiter。TP4 上 KF 已经比 aiter 快不少（6 TB/s 以上），再对 aiter 跑 TP4 MoE 意义不大；**TP8 MoE 才是合适的题目**：知识库里有同形状的 case
    study，库里有两个半成品，我们知道确切的 M 区间和 KF 在 TP4 上达到的上限。
  - 这个结论（有知识库省一半时间）是两组各一次运行，且知识库内容与任务很接近；换到 absorb BMM 这类知识库没有直接 case 的题目，效果会打折。

### 4.4 局限与注意事项

1. **基线不是我们的生产路径**：AKLR CI 用原版 aiter `dbd8bf5` / 原版 SGLang；我们用 aiter `2bb5491` + KF overlay + 两张调优 GEMM 表。表里的 speedup 不能直接换算。
   做法：`forge_task prepare --starting-kernel <生产路径的封装> --anchor start`，或先用 `aklr validate` 把现成 kernel 和生产 kernel 放在同一台机器上对测。
2. **工作负载按精确值匹配**：`n_tokens` 是 `var` 轴（不是 `any_size`），dispatch 只服务 schema 里列出的值；KV 池大小（`n_slots` / `n_pages`）也是轴。
   我们的 M 是 `32 × graph_bs`（TP8 K3）、`16 × graph_bs`（TP4 K3）、per-rank `4 × graph_bs`，K5 时乘 6，要自己写 schema（或用 `AKLR_EXPORT` 导出）。
3. **精度契约比我们的门禁松**：`fused_moe@1.0` 只要求整体 SQNR ≥ 13 dB、允许 atomic（结果不确定），知识库也写明部分 kernel 用 FP8 中间结果或截断。
   我们要另加算子级 SQNR（对 aiter，建议 ≥ 40 dB，与 KF 的 45–50 dB 同级）和 `verify_suite`（200 题、16k、真实接受长度）。
4. **计时环境与 graph 内不同**：AKLR 单算子、空闲 GPU、每次刷 L2；我们的 graph 里有补齐行、真实路由偏斜（shared expert 拿到全部 token）、前后 kernel 的
   L2 残留。每个 kernel 落地后都要用 kdeep 在 graph 内复核。AKLR 测到的每次调用约 4 µs 的地板，在 graph 内对小 kernel 也成立（我们看到的是 4.2–4.6 µs）。
5. **交付路径**：AKLR 运行时和 SGLang 插件（`AKLR_SGLANG_PLUGIN=1`）是在 SGLang 启动时替换函数，和 KF overlay 是同一类做法，InferenceX 大概率同样不接受
   （需要确认）。内部实验可以用它代替 KF overlay（有版本声明、自动回退、CI 测量）；提交 InferenceX 要把 kernel 移进 aiter（FlyDSL kernel 本来就在
   `aiter/ops/flydsl` 下）或 SGLang（Triton）并提 PR。
6. **环境**：AKLR 要求 Python ≥ 3.11、FlyDSL 0.3.2（CI 实际用 0.3.4.1），CI 镜像 `lmsysorg/sglang:v0.5.22-rocm10-mi35x`；我们的镜像
   `infera-sglang:v0522-fdbafb60`（sglang e9b22a0818）能否直接装，需要先试。
7. **成本与资源**：每个 campaign 8–15 小时、约 $100–250；每个运行要一张完全空闲的卡。AUS 节点紧张、常被抢占，forge 运行最好放在能长时间持有的单节点上。

## 五、优化方案

### 5.1 原则

- **记分板**：以 2.4 节的逐 kernel 对位表为准，每个改动落地后重采 40 步 trace，用 `kdeep.py` / `kcat.py` 更新；目标写成"每次调用 µs"和"TB/s"。
- **基线 = 生产路径**：所有算子级比较都对 KF + 调优表 + 我们的 aiter，不对原版 aiter。
- **工作负载 = 我们的 M**：从 graph 捕获列表和 `AKLR_EXPORT` 得出，按负载区间分 schema（TP8 MoE 384–1024、TP4 per-rank 32–128 等）。
- **快循环在前、慢循环在后**：算子级（分钟）→ 单节点 step 级（`profile_decode.py --full-step`，十几分钟）→ graph 内 trace（2 分钟采集）→ `verify_suite`
  （约 1 小时）→ AgentX 计分（约 1.5 小时）。只有前一级通过才进入下一级。
- **并行**：每个 forge campaign 一张卡，可以在一台 8 卡节点上同时跑 4–6 个；TP8 step 级复现要整台 8 卡，和 forge 错开或放在另一台。

### 5.2 阶段 0：基础设施（1–2 天）

1. **工作负载表**：TP8 K3 MoE M = {32·b | b ∈ 捕获列表}，重点 384–1024；TP4 K3 MoE 96–448；per-rank 稠密 / attention 行数 = 4 × graph_bs（16–128）。
   用一次短运行打开 `AKLR_EXPORT=1` 核对，并记录真实路由的专家分布（给 schema 用）。
2. **生产基线封装**：把 KF MoE / KF sparse / hipBLASLt 调优路径包成 AKLR 能调用的 `kernel.py`（`--starting-kernel`），并在 AKLR 契约下测一遍，
   作为 `--anchor start` 的基准。
3. **单节点 decode 复现**：`profile_decode.py --tp-size 8 --enable-dp-attention --dp-size 8 --speculative-algorithm EAGLE …`，
   `SGLANG_SIMULATE_ACC_LEN=2.99`，CONC 按 graph bs 22 左右设，`--full-step` 对齐 verify ≈ 42 ms；之后每个 kernel 改动都先在这里看 step 时间。
4. **prefill profile**（3.3 节）：单节点 `profile_prefill.py` 或合进 PP8 定位运行。
5. **现成 kernel 快测**：在我们的 M 上 `aklr validate` / microbench：o_proj FlyDSL、qkv_a / n6144_k2048 Triton、TP8 整层 MoE 插件、sparse MLA 插件、indexer 前处理插件；
   以及 sparse MLA 的 72 / 88 / 96 行对照（2.3 第 4 条）。这一步决定哪些项可以"拿来即用"、哪些要新写。

### 5.3 阶段 1：TP8 MoE（第 1 周，收益最大）

- **A 线（手工，和 yihou 一起）**：按 3.2 第 1 条把 KF 移植到 inter 256、M 384–1024。KF 的每一步都有现成的测量方法（文件里写了 rocprofv3 / 交替 A/B 的做法），
  风险低。
- **B 线（AKLR forge，1–2 张卡并行）**：schema `fused_moe_shuffled_16x16_…_inter_dim256`，`n_tokens` = {384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024}，
  起始 kernel 用 A 线的中间版本或 aiter 路径，`--anchor start`，12 小时；知识库已有同形状 case study，预期收敛比 AKLR#52 更快。
  K5 的 M 区间（48 × graph_bs）另开一个 campaign。
- **C 线（对照）**：AKLR 的整层 Triton 插件在 256 / 1024 和补测的 704 上对生产路径测；若接近，作为 B 线的 Triton 起点。
- **门禁与目标**：算子 SQNR（对 aiter）≥ 40 dB、`--stress` PASS；graph 内 MoE 每层 212 → ≤ 130 µs；`verify_suite` 与 2P1D 门禁持平（196/200、
  接受长度 ≥ 3.4）；预期 TP8 每步 −5.5 ~ −6.5 ms。

### 5.4 阶段 2：attention 侧（第 2 周，TP4 / TP8 都受益）

- **MLA absorb BMM**：新写两个 schema（GLM 形状、FP8 权重、rows 16–128），forge（Triton 或 FlyDSL）；目标每层 40 → ≈ 18 µs。
- **sparse MLA**：按阶段 0 的对照结果，先修 plan / 门控（若是选择问题，可能很快）；否则用 AKLR 插件或 forge（知识库有 DeepSeek-V4 sparse MLA 的 case，可借鉴）。
- **稠密投影**：现成 kernel 能用就用，不能用就分别为 o_proj / q_b / qkv_a 开 campaign（per-rank M 区间 16–128）。是否走 dense FP8 由你决定（第六节）。
- **indexer logits**：fused2 的 K3 版本（交 yihou）。

### 5.5 阶段 3：融合与结构（第 3 周起）

- router + top-k + 追加 shared expert 融合（A 线的 KF 移植会顺带做掉一部分）。
- indexer 前处理链融合（先测 AKLR 插件）、top-k。
- 通信：FP8 / MXFP4 载荷 all-gather、RS + add + RMSNorm 融合、按上下文 token 的 DP 均衡；这些要改 SGLang 层结构，先用 AKLR 插件形式做原型再上游。

### 5.6 交付

- kernel 进 aiter（FlyDSL kernel + 调优表行）或 SGLang（Triton / 门控与接线），各一个 PR；InferenceX 版本不依赖 overlay / 插件。
- forge 产出的经验用 `forge-distill` 写回 AKLR 知识库（AKLR 的流程要求），也便于以后 GLM 系列复用。
- 每项结果记进 STATUS / RESULT，trace 与 kdeep 输出放 `results/opt/kernel-*/`。

## 六、需要你决定

1. ~~优先级：先 TP8 还是先 TP4~~ **已定（11-09）：TP8 MoE 优先，同时做 TP4 / TP8 共用的 absorb BMM。** 07:44 起按计划开始阶段 0，
   进展与实测见 `KERNEL-PROGRESS-20261109.zh-CN.md`（2.3 第 4 条的 sparse MLA 原因已查清：KV 池超过 2³¹ 字节）。
   **10:10 用户决定**：跑两节点 PP8 验证（作业 35372）；absorb 配置暂不进单节点 4+4 生产配置；**所有 kernel 工作由我们完成**（yihou 不再负责 GLM-5.2），
   本文件中"交给 yihou"的各项一律改由我们负责，KF overlay 由我们维护。kernel 待办见 KERNEL-PROGRESS 第十二节。
2. 是否接受改精度的 dense FP8 路线（稠密投影可省 3–4 ms，需过 eval；草稿模型不能动）。
3. 是否用 AKLR forge（每个 campaign 约 $100–250，需要 Claude Code 登录和一台能长时间持有的节点）；以及和 yihou（KF）、AKLR 团队的分工。
4. prefill profile 是单独跑一次单节点 `profile_prefill.py`，还是并进 `BOTTLENECK-PP8` 第四节第 1 项的两节点定位运行。
5. n10-29（35341）目前空闲持有，阶段 0 的第 2、3、5 项可以在上面做（单节点、不碰 AgentX 队列），是否开始。

## 七、当前安排与后续 kernel 计划（11-09 16:40 UTC 更新）

实测依据见 `KERNEL-PROGRESS-20261109.zh-CN.md` 第十二节。

### 7.1 已安排的任务

1. **两节点 PP8 作业 35398**（6 h，排除 n03-33，排队中）：同一对节点依次跑 64c（PP8 + MoE 调优表的行 + absorb，不加 KV 上限，C208 计分约 2 h）和
   64d（PP8 原臂，约 2 h），分清 62 的 −5.7% 是 KV 上限的代价还是节点差异。作业开始、两项出分时自动唤醒分析。
2. **单节点 n02-21（35370，约 17:09 到期）**：EP8 MoE kernel 扫描已完成（32 组）。现有 FlyDSL kernel 下 EP8 最好 163.3 µs/层，对 TP8 的 166.5 µs
   只快约 3 µs，**EP8 布局不做**。35370 到期后没有单节点可做 kernel 开发。

### 7.2 后续 kernel 计划（按执行顺序）

| 顺序 | 项目 | 内容 | 预期 | 工期 |
|---|---|---|---|---|
| 1 | 真实路由数据 | overlay-kopt 加环境变量控制的钩子，从真实 decode 导出若干步 topk_ids；之后 MoE 单算子一律用真实路由（调优表的行在 graph 内只兑现 1/3，原因是均匀路由） | 前提 | 约半天 |
| 2 | K2 MoE 前处理融合 | 两段排序 + 激活量化 + scale 排序 4 个小 kernel（约 22 µs/层，延迟限制）合成 1–2 个，目标约 10 µs | 约 −0.9 ms/步（TP8） | 1–2 天 |
| 3 | K1 TP8 gemm2 写回 | 方案 A：shared expert 先普通写入初始化输出，省清零和 1/9 atomic（−5~−8 µs/层）；方案 B：各路由部分和普通写入（约 78 MB，可留在 Infinity Cache）再归约（−10~−15 µs/层）。FlyDSL 是 Python DSL，可用 overlay 替换 | −0.4~−1.1 ms/步 | 2–4 天 |
| 4 | K3 sparse MLA | 看 64c / 64d：KV 上限若有吞吐代价，改 KF 的 64 位偏移路径与 86 行 split 台阶 | −1.2 ms/步（不砍 KV）+ 0.5 | 多日 |
| 5 | K5 进 TP8 | TP4 同节点 A/B 已证明真实负载下接受长度无代价（2.990 对 2.989），下一次 PP8 kopt 臂加上 | −0.4~−0.6 ms/步 | 只改配置 |
| 6 | K4 窄 dense GEMM | fused_qkv_a、q_b_proj、router：调参只有 3–4%，需要新 kernel（split-K），适合 AKLR forge | −1.0~−1.4 ms/步 | 待定 |
| 7 | K6 / K7 | indexer 前处理 + top-k 融合；通信融合 | 各约 1 ms/步 | 之后 |

- **关于吞吐**：TP4 A/B 中 decode 变快只体现在交互度（+6%），吞吐不变，因为该点受 prefill 限制；PP8 的吞吐也由 prefill 决定。Pareto 前沿上这仍是改进，
  但要提吞吐需要看 prefill，所以第六节第 4 条（prefill profile）建议尽早做。
- **测量注意**：graph 回放下 Kineto 的分 kernel 时间只有在 `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` 时可靠（单算子测试也一样），排名一律按总时间。

### 7.3 需要你决定

1. 35370 到期后是否提交新的单节点作业继续 kernel 开发（7.2 第 1–3 项都需要）。
2. K4 是否使用 AKLR forge（第六节第 3 条）。
3. K5、K8 是否进生产配置（目前与 absorb 一样只留在实验臂）。
4. kit 标准步骤停掉别人残留容器的做法是否继续。

## 附录：复现

```bash
# 在计算节点上（不要在登录节点跑 python 分析）
A=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/ana
T=/perf_apps/liyingli/vfix-logs/opt/trace
python3 $A/dtrace_deep.py --moe-gb 48.3 $T/opt-trace2p1d-c208/traces/decode-083802/*.trace.json.gz   # 原脚本
python3 $A/kdeep.py --json tp8.json $T/opt-trace2p1d-c208/traces/decode-083302/*.trace.json.gz       # 逐层对位
python3 $A/kcat.py --tp 8 tp8.json                                                                   # 细分类 + 带宽
# KF 门控决策（graph 捕获时每个形状打印一次）
rg -o "moe M=\d+ E=257 .* -> \w+" /perf_apps/liyingli/vfix-logs/runs-n06-25/dprof-c208-1104T0758/logs/glm52_2p1d-decode.log | sort -u
```
