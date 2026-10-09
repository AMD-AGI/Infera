# GLM-5.2 decode profiling 分析：TP8 / dpa=on / EP8 / C=256 的前 10 耗时条目

**主交付运行**：`iterations/p1b_ep8_c256_graphon_midwindow_yihou/`
CUDA graph **开启**（与已发布数字一致），完整 OSL，profiling 窗口取全程中点。

---

## 0. 先看这次运行是否可信（分析之前的四道闸）

| 闸口 | 门限 | 实测 | 判定 |
|---|---|---|---|
| `complete` | true | true | ✅ |
| `verify_iterations` | 2768 | 2768 | ✅ |
| `useful_output_tokens` | C×10000 = 2 560 000 | 2 560 000（精确） | ✅ |
| `realized_accept_length` | 与 packup **逐位相同** | `3.6134393063583814` | ✅ |
| 端到端 TPOT（**仅健全性检查**） | 距 26.2519 ms 内 5 % | 26.3154 ms（**+0.24 %**） | ✅ |
| trace 大小 | < 50 MB | 870 KB | ✅ |
| 拓扑 | tp8 / dp8 / ep8 / dpa / a2a=none / local batch 32 | 全部相符 | ✅ |
| `target_graph_iterations` | 2768（全程走 graph） | 2768 | ✅ |

`realized_accept_length` 与 packup **逐位相同**，是"这套 harness 仍是产出已发布数字的那一套"的最强证据——该值由参数确定性决定，任何配置漂移都会改变它。

**TPOT 26.3154 ms 不是性能数字**（`is_performance_measurement: false`）。2768 次迭代中有 10 次被 profiler 扰动。性能口径仍以 packup 的 **26.2519 ms** 为准。

---

## 1. 一次 decode 迭代的时间预算（这是理解 top-10 的前提）

| 层次 | ms/iter | 占 wall | 测量机制 |
|---|---:|---:|---|
| **wall（全程实测）** | **95.070** | 100 % | 2768 次迭代的 elapsed |
| GPU 被 DeviceTimer 括住的时间 | 92.856 | **97.7 %** | CUDA event，**全程 2768 次**，未被 profiler 扰动 |
| 其中：命名 kernel 的 self device time | 88.78 | 93.4 % | torch.profiler，窗口内 10 次迭代 |
| 括号内但不属于任何 kernel | ≈ 4.5 | ≈ 4.8 % | 前两者之差 |
| 括号之外（CPU / launch / 同步） | **2.214** | **2.3 %** | wall − DeviceTimer |

**结论：这个配置是彻底 GPU-bound 的**，CPU 侧只占 2.3 %。优化 CPU 调度路径在这里几乎没有空间；要降 TPOT 只能动 kernel。

### 阶段分解（DeviceTimer，全程 2768 次迭代，非窗口采样）

| 阶段 | 秒 | 占比 | 说明 |
|---|---:|---:|---|
| `target_verify` | 231.535 | **90.1 %** | 目标模型的 verify 前向 |
| `eagle_draft` | 15.320 | 6.0 % | EAGLE 草稿（steps=5） |
| `eagle_draft_extend` | 10.170 | 4.0 % | 草稿扩展 |

投机解码的**全部开销（draft + draft_extend）只占 10 %**，换来 3.61 的接受长度。

---

## 2. 前 10 耗时条目

**口径**：rank 0，窗口为第 1384–1394 次迭代（10 次），context ≈ 75000，
按 **self device time**（kernel 自身 GPU 时间，不含子调用）排序，
**只含 `row_kind == device_kernel` 的行**。百分比分母是该窗口 kernel 总计 **887.79 ms**。

| # | self GPU (ms) | 占比 | 累计 | 调用次数 | µs/次 | kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 216.96 | **24.4 %** | 24.4 % | 1580 | 137.3 | `main_kernel` |
| 2 | 113.74 | **12.8 %** | 37.2 % | 791 | 143.8 | `ncclDevKernel_Generic_1` |
| 3 | 75.58 | 8.5 % | 45.8 % | 750 | 100.8 | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t64x128x256_pm1_async_v33` |
| 4 | 75.45 | 8.5 % | 54.3 % | 810 | 93.2 | `aiter::cross_device_reduce_2stage<bf16,8>` |
| 5 | 58.75 | 6.6 % | 60.9 % | 220 | **267.0** | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` |
| 6 | 52.24 | 5.9 % | 66.8 % | 790 | 66.1 | `hgemm_bf16_128x192x64x3_SPK4_W2x4x1_BLDS1_TN_AS1_CP1_0` |
| 7 | 50.32 | 5.7 % | 72.4 % | 750 | 67.1 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix3_fp4opt` |
| 8 | 35.21 | 4.0 % | 76.4 % | 780 | 45.1 | `_fused_fp8_bmm_rope_cat_and_cache_mla_kernel_BLOCK_SIZE_M_64` |
| 9 | 20.56 | 2.3 % | 78.7 % | 20 | **1027.9** | `aiter::allgather_vec<bf16,8>` |
| 10 | 18.92 | 2.1 % | 80.8 % | 790 | 24.0 | `hgemm_bf16_128x128x64x5_SPK1_W2x4x1_BLDS1_TN_AS1_0` |

**前 10 覆盖 kernel 总时间的 80.8 %**；窗口内共出现 **86 个不同的 device kernel**。

### 不要把下面两行加进上表（会重复计时）

| 条目 | ms | 次数 | 性质 |
|---|---:|---:|---|
| `step[TARGET_VERIFY bs=32]` | 838.93 | 10 | `record_function` 标注，**包住**上表多数 kernel |
| `draft_extend` | 40.54 | 10 | 同上 |

torch 把 `record_function` 标注也报成 `DeviceType.CUDA` 行。若不分类直接排序，`step[TARGET_VERIFY bs=32]` 会以 838.93 ms 稳居第一，而它**与其下的 kernel 完全重叠**——这是本次实现中一个真实且已修复的陷阱。榜单只取 `device_kernel` 行，标注单列。

---

## 3. 怎么读这张表

> ⚠️ **下面这张表已被本节末尾的新表取代，保留仅为记录调查过程。**
> 它是在 `main_kernel` 身份未知时做的归并；`main_kernel` 现已查明是 **DSA 稀疏注意力**本身
> （见本节 "`main_kernel` 的身份：已确定"）。**请直接看本节末尾的最终归类表。**
> 当时的错误结论是"通信是最大类别"——实际是注意力。

**（已作废）按功能归并的初版，`main_kernel` 身份未知时**：

| 功能类别 | 条目 | 合计占比 |
|---|---|---:|
| 通信（TP8 all-reduce / all-gather） | #2 `ncclDevKernel_Generic_1` + #4 `cross_device_reduce_2stage` + #9 `allgather_vec` | 23.6 % |
| MoE 专家计算（MXFP4） | #3 `mfma_moe1` + #7 `mfma_moe2` | 14.2 % |
| 稠密 GEMM（bf16） | #6 + #10 `hgemm_bf16` | 8.0 % |
| DSA 稀疏注意力 logits | #5 `_gluon_deepgemm_fp8_paged_mqa_logits` | 6.6 % |
| MLA / RoPE / KV 写入 融合 | #8 | 4.0 % |
| `main_kernel`（身份待定） | #1 | 24.4 % |

关于 EP1 的说明**不受影响，仍然成立**：packup 里 EP1 在 9/10 个并发点上快于 EP8，
但**本次没有做 EP1 的对照 profiling，因此不能用通信占比去解释它**。

### 关于 #1 `main_kernel`：**身份未确定，不做推测**

它以 24.4 % 位居第一，每次迭代调用 158 次（1580/10），平均 137.3 µs。这个名字是符号表里的通用名，**profiling 数据本身不足以判定它属于哪个算子**。

**已用 graph-OFF trace 做了一次反查，结论是：`main_kernel` 不是一个 kernel，而是至少 4 个不同的 kernel 共用了同一个编译器生成的通用名。** graph-ON 的 trace 里 kernel 事件不带 `grid`/`block`（replay 不产生 `hipLaunchKernel` 记录），所以这次拆解**只能在 graph-OFF 里做** —— 这本身就是 §4"两种模式给的是不同层次"的一个具体例子。

按 launch 签名 + 所属 stage 标注拆解（graph OFF，窗口内 10 次迭代，rank 0）：

| grid | 所属 stage | 次/迭代 | 合计 | 平均 |
|---|---|---:|---:|---:|
| `(768, 4, 1)` | `step[TARGET_VERIFY bs=32]` | **78.0** | 202.83 ms | 260.0 µs |
| `(3072, 1, 1)` | `step[TARGET_VERIFY bs=32]` | **78.0** | 9.38 ms | 12.0 µs |
| `(768, 4, 1)` | `draft_extend` | 1.0 | 2.60 ms | 260.3 µs |
| `(128, 16, 1)` | `draft` | 4.0 | 2.39 ms | 59.8 µs |
| `(512, 1, 1)` | `draft` | 4.0 | 0.35 ms | 8.8 µs |
| `(3072, 1, 1)` | `draft_extend` | 1.0 | 0.12 ms | 11.7 µs |

**主导项是 `grid=(768,4,1)` 那一支，占 `main_kernel` 全部时间的 93.2 %（202.83 / 217.67 ms），每次迭代恰好 78.0 次。**
目标模型 `num_hidden_layers = 78`（已核实 `config.json`），草稿模型 `num_nextn_predict_layers = 1`，
而同一签名在 `draft_extend` 里恰好 1.0 次/迭代 —— **每层每次前向调用一次**，两侧计数完全自洽。

**第三重独立印证**（leader 复核时发现）：`draft` stage 下的两个签名 `(128,16,1)` 与 `(512,1,1)`
各自恰好 **4.0 次/迭代**，而 §4 中 DeviceTimer 在 graph-OFF 下测到 `decode` 类别
**n = 11072 = 4 × 2768**，即草稿模型每次迭代做 4 次 eager 前向。
两个完全独立的机制（chrome trace 的 launch 计数 / CUDA event 的类别计数）给出同一个"4"。
连同 78↔`num_hidden_layers`、1↔`num_nextn_predict_layers`，这张表的三组计数**全部与模型结构自洽**。

另外两项实测约束：
- 该 launch **不在任何 `cpu_op` 之内**（1660 次全部无 aten 父算子），即它由自定义扩展绕过 aten dispatcher 直接下发。
  **这也正是"用 graph-OFF 的 CPU 算子层反查父算子"这条思路失败的原因**——失败的方法自己解释了自己；
- 也**不在任何 `## Call CompiledFxGraph ##` 标注之内**，即不是 torch.compile/Inductor 产物。

> **口径提醒**：上表按 stage 拆分。若只按 launch 签名聚合（不分 stage），`(768,4,1)` 与 `(3072,1,1)`
> 各为 **79.0 次/迭代** = 78（`target_verify`）+ 1（`draft_extend`）。两种数法都已复核，互相一致。
> graph-ON 侧的 1580 次 `main_kernel` 中，携带 `grid` 参数的为 **0 个**——该拆解在 graph-ON 下不可得，已复核。

### `main_kernel` 的身份：**已确定**——DSA 稀疏注意力（TileLang JIT）

用 `--profile-with-stack` 跑了一次 graph-OFF（`iterations/p2_shapes_stack_graphoff_yihou/`，窗口 2 次迭代），
trace 里有 715 218 条 `python_function` 事件构成完整 Python 调用树；按时间戳定位每次 `hipLaunchKernel`
所在的最内层 Python 帧，**四个签名全部落在同一条调用链上**：

```
<built-in method forward of tilelang_cython_wrapper.CythonKernelWrapper object>
tilelang/jit/adapter/cython/adapter.py(343): lambda_forward
tilelang/jit/kernel.py(191): __call__
sglang/kernels/ops/attention/dsa/tilelang_kernel.py(1317): tilelang_sparse_fwd
sglang/srt/layers/attention/dsa_backend.py(3333): _forward_tilelang
sglang/srt/layers/attention/dsa_backend.py(2220): forward_extend   ← 目标模型
        （或 dsa_backend.py(2570): forward_decode                  ← 草稿模型）
sglang/srt/layers/attention/base_attn_backend.py(215): forward
sglang/srt/layers/radix_attention.py(150): forward
nn.Module: RadixAttention_1 （目标） / RadixAttention_0 （草稿）
```

**结论：#1 那 24.4 % 是 DSA 稀疏注意力本身**，由 TileLang JIT 编译、经 Cython wrapper 直接下发。
`(768,4,1)` 与 `(3072,1,1)` 走 `forward_extend`（目标模型，`target_verify` 是 EXTEND 模式），
`(128,16,1)` 与 `(512,1,1)` 走 `forward_decode`（草稿模型）——与前表按 stage 的拆分完全一致。

**名字的来源也随之确定**：`tilelang_kernel.py:1314` 处该 kernel 构造函数 `return main`，
TileLang 把名为 `main` 的函数编译出的入口符号即 `main_kernel`。这解释了为什么 4 个不同的 kernel 同名。

**前面两个"否定"现在都有了原因**（它们不是噪声，而是同一事实的两个侧面）：
- **不在任何 `cpu_op` 之内** —— TileLang 的 Cython wrapper 直接下发 kernel，根本不经过 aten dispatcher；
- **不在 `## Call CompiledFxGraph ##` 之内** —— 它是 TileLang 产物，不是 Inductor 产物。

**`--profile-record-shapes` 对它无效，且这一点是实测的**：p0b 已经开着该选项（默认 `True`），
trace 里 167 923 条 `cpu_op` 中有 166 883 条带 `Input Dims`，证明该功能确实在工作；
而 `main_kernel` 的 kernel 事件与其 `hipLaunchKernel` **都不带任何 shape 字段**。
`record_shapes` 是 aten dispatcher 的功能，对绕过 dispatcher 的 kernel 天然不可见。
**真正解决问题的是 `--profile-with-stack` 的 Python 调用树，不是 shapes。**

### 由此重排的功能归并

| 功能类别 | 条目 | 合计占比 |
|---|---|---:|
| **DSA 稀疏注意力** | #1 `main_kernel`（TileLang sparse fwd） + #5 `_gluon_deepgemm_..._mqa_logits` | **31.0 %** |
| 通信（TP8 all-reduce / all-gather） | #2 + #4 + #9 | 23.6 % |
| MoE 专家计算（MXFP4） | #3 + #7 | 14.2 % |
| 稠密 GEMM（bf16） | #6 + #10 | 8.0 % |
| MLA / RoPE / KV 写入 融合 | #8 | 4.0 % |

**注意力（31.0 %）而非通信（23.6 %）才是这个配置下最大的单一开销类别。**
§3 上方那张按 `main_kernel` 身份待定写的归并表应以本表为准。

### 榜单里还有哪些条目是"同名聚合"？——已全面排查（leader 复核）

`main_kernel` 是聚合，这不保证它唯一。对 graph-OFF 的 p0b trace 按 **launch 签名**逐一排查
按总时间排前 12 的 kernel 名（无需 GPU，只用已有 trace）：

| 签名数 | launches | ms | kernel 名 |
|---:|---:|---:|---|
| **4** | 1660 | 217.67 | `main_kernel` ← 聚合 |
| **3** | 840 | 93.00 | `ncclDevKernel_Generic_1` ← 聚合 |
| 1 | 750 | 75.61 | `mfma_moe1_...` |
| 1 | 890 | 66.41 | `aiter::allgather_vec` |
| 1 | 220 | 59.08 | `_gluon_deepgemm_..._mqa_logits` |
| 1 | 790 | 52.37 | `hgemm_bf16_128x192x64x3` |
| 1 | 750 | 49.54 | `mfma_moe2_...` |
| 1 | 780 | 35.39 | `_fused_fp8_bmm_rope_cat_and_cache_mla` |
| 1 | 790 | 19.00 | `hgemm_bf16_128x128x64x5` |
| **2** | 1500 | 14.44 | `aiter::dynamic_per_group_scaled_quant_kernel` ← 聚合 |
| **2** | 50 | 14.23 | `ck::kernel_moe_gemm_2lds` ← 聚合 |
| 1 | 760 | 14.14 | `aiter::bf16gemm_fp32bf16_tn_96x64_splitk` |

**结论：top-10 中另有 1 项是聚合（#2 `ncclDevKernel_Generic_1`），但它不影响 §3 的归类。**
`main_kernel` 之所以要紧，是因为它的聚合**跨越了功能类别的歧义**——不知道该归到哪一类；
其余几项的多签名都落在**同一功能类别内部**，归并结果不变。

`ncclDevKernel_Generic_1` 的拆解（graph-OFF，10 次迭代）：

| grid | 次/迭代 | 合计 | 占该条目 |
|---|---:|---:|---:|
| `(105,1,1)` | **79.0** | 91.88 ms | **98.8 %** |
| `(96,1,1)` | 4.0 | 0.90 ms | 1.0 % |
| `(8,1,1)` | 1.0 | 0.23 ms | 0.2 % |

**主导支同样是 79.0 次/迭代**，与 `main_kernel` 主导支完全相同的"每层一次"节律
（78 目标层 + 1 draft_extend 层）。这正是张量并行的含义：**每个 transformer 层之后一次 TP all-reduce。**
次要支 4.0 次/迭代对应草稿模型每迭代 4 次前向，与前面第三重印证一致。

（`aiter::dynamic_per_group_scaled_quant_kernel` 两支各 75.0 次/迭代，
`ck::kernel_moe_gemm_2lds` 两支合计 5.0 次/迭代；两者都在 top-10 之外，且各自同属一类，不再展开。）


### 关于 #5 和 #9 的"每次调用很贵"

- #5 每次 267 µs，但整个窗口只调 220 次（22 次/迭代）——**单次最贵的高频 kernel**。
- #9 `allgather_vec` 每次 1027.9 µs，窗口内仅 20 次（2 次/迭代），属于低频大块通信。

---

## 3.5 顺带量化出来的结果：CUDA graph 到底省了什么

graph-OFF 对照运行 `p0b_ep8_c256_graphoff_midwindow_yihou/` 与 P1b **除 `--disable-cuda-graph` 外逐字相同**，
且两者 `realized_accept_length` **完全相同**（`3.6134393064`）——计算量一致，只是调度方式不同。
这是一次干净的单变量对照。

| 每次迭代 | graph **ON** | graph **OFF** | 差异 |
|---|---:|---:|---:|
| wall | 95.070 ms | 135.306 ms | **+42.3 %** |
| DeviceTimer 括住的 GPU 时间 | 92.856 ms | 131.673 ms | +41.8 % |
| **命名 kernel 的 self device time** | **88.779 ms** | **86.685 ms** | **−2.4 %** |
| kernel 占 wall | 93.4 % | **64.1 %** | — |
| 非 kernel 时间 | 6.29 ms | **48.62 ms** | **+673 %** |
| TPOT | 26.3154 ms | 37.4528 ms | +42.3 % |

**kernel 时间几乎没变（−2.4 %），42.3 % 的 TPOT 代价全部来自非 kernel 时间。**
换句话说，在这个配置下 CUDA graph **不让 kernel 变快，它消除的是每次迭代约 40 ms 的下发/间隙开销**。

**那 48.62 ms 具体由什么占据，没有测过，不做解释。** 需要对 trace 时间线做 gap 分析才能回答。

不过 trace 的事件计数本身是实测的，可以给出边界（数据来自两次运行的 chrome trace，窗口内 10 次迭代，rank 0）：

| trace 事件类别（次/迭代） | graph **ON** | graph **OFF** | 倍数 |
|---|---:|---:|---:|
| `kernel`（真实 GPU kernel） | 2417 | 2432 | **×1.006** |
| `cuda_runtime`（host 侧运行时调用） | 205 | 5392 | **×26.3** |
| `cpu_op` | 753 | 16792 | **×22.3** |

**kernel 数量两模式几乎相同（+0.6 %），host 侧运行时调用相差 26 倍。** 这与 §1 的
"kernel 时间不变、非 kernel 时间 +673 %" 相互印证。若把 48.62 ms 平摊到 2432 次 kernel 上是
每次约 20 µs——**这只是一个数量级参考，不是对该时间去向的测定**；上面那句"不做解释"仍然成立。

（模型层数已核实：`config.json` 的 `num_hidden_layers = 78`、`num_nextn_predict_layers = 1`。）

---

## 4. cuda_graph 开 / 关：两种模式都能 profile，但给的是**不同层次**

| | graph **ON**（已发布数字的配置） | graph **OFF** |
|---|---|---|
| device kernel 层 | ✅ 86 个，带 self device time | ✅ 有，同一批 kernel |
| CPU 算子层（`aiter::gemm_a16w16`、`sglang::reg_all_gather_into_tensor`…） | ❌ **不发射** | ✅ 有，CPU+CUDA 双时间 |
| `step[DECODE bs=32]` 等 forward 内标注 | ❌ 不发射 | ✅ 有 |
| DeviceTimer 阶段分解 | ✅ | ✅（**类别名不同**，见下） |

**为什么**：replay 一个 hipGraph 时图内不执行 Python，`model_runner.forward` 里的 `record_function` 根本不会运行。SGLang 自己也踩过——`frozen_kv_mtp_cuda_graph_runner.py:448` 的注释原文是 "the graph bypasses `model_runner.forward`'s record_function"，并在图运行器层手动补了一个标注。

**一个容易误读的坑（DeviceTimer 类别名随模式变化）**：

| 模式 | target 侧类别 | draft 侧类别 |
|---|---|---|
| graph ON | `target_verify` 90.1 % | `eagle_draft` 6.0 %, `eagle_draft_extend` 4.0 % |
| graph OFF | `target_verify` 90.3 % | `decode` 6.8 %（**n=11072 = 4×2768**）, `extend` 2.9 % |

graph-OFF 里的 `decode` 是**草稿模型每次迭代 4 次 eager 前向**，不是目标模型——`n` 恰好是迭代数的 4 倍，
在全量 2768 次迭代上坐实了这一点。看起来"某个阶段消失了"，其实只是换了名字。
已逐一核对：两侧类别集合**交集为空**，且 `sum(by_category) == sum(by_runner)` 精确相等，无重复计数。
**阶段占比在两种模式下稳定（90.1/6.0/4.0 vs 90.3/6.8/2.9），尽管标签不同。**

### graph-OFF 的 top-10 与 graph-ON 的差异

104 个不同 kernel（graph-ON 为 86 个）。**十项中有九项与 P1b 相符，误差在数个百分点内**，排序基本一致。
两处值得注意，**仅作为观察记录，未查明原因**：

| kernel | graph ON | graph OFF |
|---|---|---|
| `aiter::allgather_vec` | 20.56 ms，**n=20** | 66.41 ms，**n=890** |
| `ncclDevKernel_Generic_1` | 113.74 ms，n=791 | 93.00 ms，n=840 |

**两种模式下集合通信的构成不同。为什么，不知道。** 这不影响 §2 的结论——那张表来自 graph-ON，
而 graph-ON 正是已发布数字的配置。

graph-OFF 另外带有 CPU 算子层（`cpu_side` 行，**不进 kernel 榜单**）：
`aiter::fused_moe_` 139.93 ms n=800、`aiter::gemm_a16w16` 94.16 ms n=4060、
`sglang::reg_all_gather_into_tensor` 70.87 ms n=890。
这一层正是反查 `main_kernel` 身份的可用线索（见 §7 留白 1）。

---

## 5. 两套独立机制互相验证

| 量 | torch.profiler（Kineto 标注） | DeviceTimer（CUDA event） | 差异 |
|---|---:|---:|---:|
| 窗口内 `target_verify` | 838.93 ms | 838.88 ms | **0.006 %** |

两条完全独立的测量路径给出同一个数，这是"两个机制都在测我们以为的东西"的最强验证。

另一个**报告但不调和**的比值：窗口内 kernel self time 合计 887.8 ms vs DeviceTimer 窗口总计 933.2 ms → **0.951**。
那 4.9 % 是落在计时括号内、却不属于任何 kernel 的时间。**没有测过它是什么，不做猜测**，也没有为了让两者对上而调整任何数字。

---

## 6. 窗口位置的影响：比我预期的小得多

选择中点窗口的理由是"attention 成本随 context 增长，短端窗口会低估其占比"。拿到两组数据后，**这个理由基本不成立**（context 70072 vs 75000，每次迭代 self CUDA）：

| kernel | ctx≈70072 | ctx≈75000 | 变化 |
|---|---:|---:|---:|
| `main_kernel` | 21.73 ms | 21.70 ms | −0.2 % |
| `ncclDevKernel_Generic_1` | 11.38 | 11.37 | −0.1 % |
| `mfma_moe1` | 7.54 | 7.56 | +0.2 % |
| `cross_device_reduce_2stage` | 7.47 | 7.55 | +1.0 % |
| **`_gluon_deepgemm_..._mqa_logits`** | 5.49 | 5.87 | **+7.0 %** |
| 其余六项 | — | — | 均在 ±0.7 % 内 |
| kernel 合计 | 88.47 | 88.78 | +0.35 % |
| 不同 kernel 数 | 86 | 86 | 相同 |

**只有 DSA paged-MQA logits 那个 kernel 明显随 context 变化，+7.0 % 恰好与 +7.0 % 的 context 增长相当。其余全部持平。** 所以 top-10 的**排序**在短窗口下也会是一样的。

中点窗口的运行仍然是更好的产物——它是完整的一次运行，带 TPOT 交叉验证——但**当初支持它的理由大部分是错的**。其余 kernel 为何对 context 不敏感，**没有测过，不做解释**。

---

## 7. 明确留白（不猜）

1. ~~**`main_kernel` 的身份未确定**~~ —— **已关闭**。经 `--profile-with-stack` 的 Python 调用树定位为
   **DSA 稀疏注意力（TileLang JIT `tilelang_sparse_fwd`）**，见 §3。
   （附带结论：`--profile-record-shapes` 对该 kernel 天然无效，因为它绕过 aten dispatcher；
   当初把 shapes 列为待做测量是错的，实际有效的是 with-stack。）
2. **kernel 合计与 DeviceTimer 相差 4.9 %** 的构成未测。
3. **通信占 23.6 %，但没有 EP1 对照 profiling**，不能用它解释 packup 里 EP1 更快的现象。
4. **`idle` 类别从未触发**，原因未查（需要 DP 分片 batch 不等的场景才能触发）。
5. 仅 rank 0 出 trace，**未验证各 rank 的 kernel 分布是否一致**。
6. `DEBUG_CLR_GRAPH_PACKET_CAPTURE` 已接入但**从未真正被使用**——默认设置下归因本就成立。该 flag 本身**未被刻画**，详见 `research/rocm_profiling_env.md` §2（结论标为 provisional）。
