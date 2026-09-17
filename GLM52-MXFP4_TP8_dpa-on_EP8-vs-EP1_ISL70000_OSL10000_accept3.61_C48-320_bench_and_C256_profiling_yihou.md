# GLM-5.2-MXFP4 decode 汇总：TP8 / dpa=on / EP8 vs EP1 并发扫描 + C=256 profiling

**启动关键参数**（除标注外，全部实验逐字相同）

| 参数 | 值 |
|---|---|
| 模型 | `/perf_apps/data/models/GLM-5.2-MXFP4`（78 层，1 nextn-predict 层） |
| 并行 | `--tp-size 8 --dp-size 8 --enable-dp-attention`（dpa=on），`--moe-a2a-backend none` |
| 专家并行 | `--ep-size 8` / `--ep-size 1`（**唯一自变量**，见表 1） |
| ISL / OSL | `--input-len 70000` / `--output-len 10000` |
| 投机解码 | EAGLE，`steps=5`、`draft-tokens=6`、`topk=1`，`--accept-length 3.61`（`match-expected` / `real-draft-token`） |
| 并发 C | 48…320（表 1）；profiling 固定 **C=256**，local batch = C/8 = 32 |
| 其他 | `--max-running-requests <C>`（**必需，非调参**）、`--mem-fraction-static 0.85`、`--kv-cache-dtype fp8_e4m3`、`--dsa-{decode,prefill}-backend flydsl`、`--dsa-topk-backend aiter`、`--warmup-steps 10`、`--random-seed 1234` |
| 硬件 / 栈 | 8× MI355X (gfx950) 单机；ROCm 7.2.0，torch 2.9.1+rocm7.2.0，triton 3.7.0；SGLang `402df1e1e45…`，AITER `2c71811b32c…` |

**数据来源（均为一手实测，非转述）**
- 吞吐扫描：`packups/glm52_tp8_dpa_ep8_vs_ep1_sweep_yihou.packup_20260914-0740/`（20 个点）
- Profiling：`packups/glm52_tp8_ep8_dpa_c256_isl70k_osl10k_decode_profiling_yihou.packup_20260915-0130/`（5 次运行）
- Profiling 能力实现：commit `23f472d2`

**全部 20 个吞吐点 + 2 个 profiling 全量运行的 `realized_accept_length` 均为
`3.6134393063583814`、`verify_iterations` 均为 `2768`**——该值由参数确定性决定，逐位一致说明所有
运行来自同一套未漂移的 harness。

---

## 表 1　并发扫描：EP8 vs EP1（专家并行开 / 关）

同机、同容器、同镜像、背靠背执行，**唯一自变量是 `--ep-size`**。
`ΔTPOT = (EP1 − EP8) / EP8`，负值表示 EP1 更快。

| C | local batch | EP8 TPOT (ms) | EP1 TPOT (ms) | ΔTPOT | EP8 tok/s | EP1 tok/s | EP8 tok/s/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 6 | 12.0832 | 10.5585 | **−12.62 %** | 3972.46 | 4546.11 | 496.6 |
| 64 | 8 | 13.1428 | 11.9746 | −8.89 % | 4869.57 | 5344.65 | 608.7 |
| 96 | 12 | 16.2379 | 14.9122 | −8.16 % | 5912.10 | 6437.67 | 739.0 |
| 128 | 16 | 17.8478 | 17.3231 | −2.94 % | 7171.77 | 7388.97 | 896.5 |
| 160 | 20 | 20.8425 | 19.7278 | −5.35 % | 7676.61 | 8110.39 | 959.6 |
| 192 | 24 | 22.7251 | 21.9950 | −3.21 % | 8448.81 | 8729.24 | 1056.1 |
| 224 | 28 | 24.2158 | 24.2623 | **+0.19 %** | 9250.17 | 9232.45 | 1156.3 |
| **256** | **32** | **26.2519** | **25.6740** | **−2.20 %** | **9751.66** | **9971.17** | **1219.0** |
| 288 | 36 | 28.9181 | 28.3732 | −1.88 % | 9959.15 | 10150.44 | 1244.9 |
| 320 | 40 | 30.7719 | 30.4108 | −1.17 % | 10399.09 | 10522.58 | 1299.9 |

**读法提醒**：harness 定义 `tok/s = C × 1000 / TPOT`，两列是**同一个测量的两种单位**，不是两个独立结果。

**观察**：EP1 在 10 个并发点中的 9 个更快，但优势随 C 单调收窄——C=48 时 −12.62 %，C=320 时仅 −1.17 %。
C=224 是唯一反号点（+0.19 %）。**每点均为单次运行、无重复**，0.19 % 远小于未测量的运行间波动，
因此该反号点记为**开放问题，不作为效应**。

---

## 表 2　Profiling：CUDA graph 开 / 关（C=256, EP8）

同机同容器，**除 `--disable-cuda-graph` 外逐字相同**，且两次 `realized_accept_length` 逐位相同
（`3.6134393063583814`）——计算量完全一致，差异只来自调度方式。

| 每次迭代 | graph **ON** | graph **OFF** | 差异 |
|---|---:|---:|---:|
| wall（全程 2768 次实测） | **95.070 ms** | **135.306 ms** | **+42.3 %** |
| DeviceTimer 括住的 GPU 时间 | 92.856 ms | 131.673 ms | +41.8 % |
| **命名 kernel 的 self device time** | **88.779 ms** | **86.685 ms** | **−2.4 %** |
| kernel 占 wall | 93.4 % | 64.1 % | — |
| 非 kernel 时间 | 6.29 ms | 48.62 ms | **+673 %** |
| 端到端 TPOT | 26.3154 ms | 37.4528 ms | +42.3 % |
| kernel launch 次数 | 2417 | 2432 | ×1.006 |
| `cuda_runtime` host 调用次数 | 205 | 5392 | **×26.3** |
| `cpu_op` 事件数 | 753 | 16792 | ×22.3 |
| 可见 device kernel 种类 | 86 | 104 | — |
| chrome trace 体积（10 次迭代） | 0.83 MB | 7.59 MB | ×9.1 |

**结论：CUDA graph 不让 kernel 变快**（kernel 时间仅差 −2.4 %），**它消除的是每次迭代约 40 ms 的
下发开销**。host 侧运行时调用相差 26 倍，从独立方向印证了这一点。

### 两种模式给的是**不同层次**，不是不同质量

| 可观测层 | graph ON | graph OFF |
|---|---|---|
| device kernel + self device time | ✅ 86 个 | ✅ 104 个（同一批底层 kernel） |
| CPU 算子层（`aiter::gemm_a16w16`、`sglang::reg_all_gather_into_tensor`…） | ❌ 不发射 | ✅ 含 CPU+CUDA 双时间 |
| `model_runner.forward` 内的 `record_function` 标注 | ❌ 不发射 | ✅ 有 |
| kernel 的 `grid`/`block` 启动参数 | ❌ **0/1580 条携带** | ✅ 全部携带 |
| DeviceTimer 阶段分解 | ✅ | ✅（类别名不同，见下） |

**原因**：replay hipGraph 时图内不执行 Python，`record_function` 不会运行；replay 也不产生
`hipLaunchKernel` 记录。SGLang 自己踩过同一个坑——`frozen_kv_mtp_cuda_graph_runner.py:448` 的注释
即 "the graph bypasses `model_runner.forward`'s record_function"。

**易误读点**：DeviceTimer 的类别名随模式变化。graph-OFF 的 `decode` 是**草稿模型每迭代 4 次 eager
前向**（`n = 11072 = 4 × 2768`），不是目标模型；目标模型两种模式下都叫 `target_verify`。
阶段占比本身稳定：90.1 / 6.0 / 4.0（ON）vs 90.3 / 6.8 / 2.9（OFF）。

---

## 表 3　C=256 前 10 耗时条目（graph ON，即已发布数字的配置）

rank 0，窗口为第 1384–1394 次迭代（10 次，context ≈ 75000），按 **self device time** 排序，
**仅含 `row_kind == device_kernel` 的行**，分母为窗口 kernel 总计 **887.79 ms**。

| # | self GPU (ms) | 占比 | 累计 | 次数 | µs/次 | kernel | 归属 |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 216.96 | **24.4 %** | 24.4 % | 1580 | 137.3 | `main_kernel` | **DSA 稀疏注意力（TileLang JIT）** |
| 2 | 113.74 | **12.8 %** | 37.2 % | 791 | 143.8 | `ncclDevKernel_Generic_1` | TP all-reduce |
| 3 | 75.58 | 8.5 % | 45.8 % | 750 | 100.8 | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t64x128x256` | MoE (MXFP4) |
| 4 | 75.45 | 8.5 % | 54.3 % | 810 | 93.2 | `aiter::cross_device_reduce_2stage<bf16,8>` | 通信 |
| 5 | 58.75 | 6.6 % | 60.9 % | 220 | **267.0** | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | DSA 注意力 logits |
| 6 | 52.24 | 5.9 % | 66.8 % | 790 | 66.1 | `hgemm_bf16_128x192x64x3_SPK4` | 稠密 GEMM |
| 7 | 50.32 | 5.7 % | 72.4 % | 750 | 67.1 | `mfma_moe2_afp4_wfp4_bf16_cshuffle` | MoE (MXFP4) |
| 8 | 35.21 | 4.0 % | 76.4 % | 780 | 45.1 | `_fused_fp8_bmm_rope_cat_and_cache_mla_BLOCK_M_64` | MLA/RoPE/KV 融合 |
| 9 | 20.56 | 2.3 % | 78.7 % | 20 | **1027.9** | `aiter::allgather_vec<bf16,8>` | 通信 |
| 10 | 18.92 | 2.1 % | 80.8 % | 790 | 24.0 | `hgemm_bf16_128x128x64x5_SPK1` | 稠密 GEMM |

**前 10 覆盖 kernel 总时间的 80.8 %**，窗口内共 86 个不同 device kernel。

> **不得并入上表**（会重复计时）：`step[TARGET_VERIFY bs=32]` 838.93 ms / 10 次、`draft_extend`
> 40.54 ms / 10 次。torch 把 `record_function` 标注也报成 `DeviceType.CUDA` 行，它们**包住**上表
> 多数 kernel。不做分类直接排序，榜首会变成这条标注。

---

## 表 4　按功能类别归并

| 类别 | 条目 | 合计占比 |
|---|---|---:|
| **DSA 稀疏注意力** | #1 + #5 | **31.0 %** |
| **通信（TP all-reduce / all-gather）** | #2 + #4 + #9 | **23.6 %** |
| MoE 专家计算（MXFP4） | #3 + #7 | 14.2 % |
| 稠密 GEMM（bf16） | #6 + #10 | 8.0 % |
| MLA / RoPE / KV 写入 融合 | #8 | 4.0 % |
| 其余 76 个 kernel | — | 19.2 % |

**注意力（31.0 %）而非通信（23.6 %），是该配置下最大的单一开销类别。**

---

## 表 5　一次 decode 迭代的时间预算（graph ON）

| 层次 | ms/iter | 占 wall | 测量机制 |
|---|---:|---:|---|
| **wall（全程 2768 次实测）** | **95.070** | 100 % | `elapsed / verify_iterations` |
| GPU 被 DeviceTimer 括住 | 92.856 | **97.7 %** | CUDA event，全程，未被 profiler 扰动 |
| 其中命名 kernel 的 self device time | 88.779 | 93.4 % | torch.profiler，窗口 10 次迭代 |
| 括号内但不属于任何 kernel | ≈ 4.5 | ≈ 4.8 % | 前两者之差（**未测其构成**） |
| 括号之外（CPU / launch / 同步） | 2.214 | **2.3 %** | wall − DeviceTimer |

**阶段分解**（DeviceTimer，全程 2768 次迭代）：`target_verify` 90.1 % / `eagle_draft` 6.0 % /
`eagle_draft_extend` 4.0 %。**投机解码的全部开销只占 10 %**，换来 3.61 的接受长度。

---

## 影响分析

### 1. 该配置是彻底 GPU-bound 的，CPU 侧优化空间几乎为零

97.7 % 的墙钟被 GPU 占满，非 GPU 部分仅 **2.214 ms/iter（2.3 %）**。
**含义**：优化 Python 调度路径、批处理逻辑、host 侧开销，理论上限也只有 2.3 %。降 TPOT 必须动 kernel。

### 2. CUDA graph 是这个规模下的必需项，而非可选优化

| | 数值 | 含义 |
|---|---:|---|
| 关闭 graph 的 TPOT 代价 | **+42.3 %** | 26.32 → 37.45 ms |
| 其中来自 kernel 变慢 | **−2.4 %**（即没有） | kernel 执行本身不受影响 |
| 其中来自非 kernel 时间 | **+673 %** | 6.29 → 48.62 ms/iter |
| 每迭代 kernel 下发次数 | **2417** | 78 层 × 每层多次下发 |

**含义**：代价与 kernel 效率无关，纯粹是 2417 次下发的 host 开销无法再被摊掉。
**任何迫使关闭 CUDA graph 的改动（动态 shape、可变 batch、host 侧回调）都要按 +42 % TPOT 计价。**

### 3. 注意力是第一优化目标，其次是通信

| 优化方向 | 当前占比 | 说明 |
|---|---:|---|
| DSA 稀疏注意力 | **31.0 %** | #1 每层每次前向一次，260 µs/次；#5 单次最贵（267 µs） |
| TP 通信 | **23.6 %** | #2 主导支 **79.0 次/迭代**，即**每个 transformer 层之后一次 all-reduce** |
| MoE | 14.2 % | MXFP4，已用 `mfma` 路径 |

**#5 `_gluon_deepgemm_fp8_paged_mqa_logits` 是唯一随 context 显著增长的 kernel**：context
70072 → 75000（+7.0 %）时它 +7.0 %，其余 kernel 全部在 ±0.7 % 内，kernel 合计仅 +0.35 %。
**含义**：在 70k–80k 这一段，只有注意力 logits 一项随上下文线性增长；进一步拉长上下文时，注意力
占比会继续上升，而其余部分基本不变。

### 4. EP8 vs EP1：通信占比高与 EP1 更快方向一致，但**不能据此归因**

| C | EP1 相对 EP8 的 TPOT 优势 |
|---:|---:|
| 48 | −12.62 % |
| 128 | −2.94 % |
| 256 | −2.20 % |
| 320 | −1.17 % |

优势随并发单调收窄。profiling 显示通信占 23.6 %，方向上与"EP8 引入额外通信"一致，
**但本次只 profiling 了 EP8，没有 EP1 对照**，因此**不能用该占比解释表 1 的结果**。
这是最明确的下一步实验。

### 5. 容量上限（来自吞吐 packup 的实测算术）

每 rank KV 池 `#tokens = 3,448,128`（EP8），每请求预留 `80,064` tokens。
C=320 需要 `40 × 80,064 = 3,202,560`，占 **92.9 %**，通过；
C=352 需要 `44 × 80,064 = 3,522,816`，**超出池容量**。
**下一个并发点会先撞上 request 预留上限，而不是显存**（`max_memory_allocated` 从 C=48 的
253.45 GB 仅增至 C=320 的 256.36 GB，卡上有 288 GB）。

---

## 结论

1. **GLM-5.2-MXFP4 在 TP8/dpa/EP8/C=256 下的 decode 是 GPU-bound 的**：97.7 % 墙钟被 GPU 占满，
   单次迭代 95.07 ms，其中 88.78 ms 在命名 kernel 内。
2. **最大单一开销是 DSA 稀疏注意力（31.0 %），不是通信（23.6 %）。** 榜首 `main_kernel` 经 Python
   调用树确认为 TileLang JIT 编译的 DSA sparse attention，每层每次前向调用一次。
3. **CUDA graph 值 +42.3 % 的 TPOT，且与 kernel 效率无关**——两种模式 kernel 时间只差 2.4 %，
   全部差异来自 2417 次/迭代的下发开销。
4. **投机解码很划算**：draft + draft_extend 合计仅占 10 % 的 GPU 时间，换来 3.61 的接受长度。
5. **EP1 在 10 个并发点的 9 个上快于 EP8，但优势随并发收窄**（−12.62 % @C=48 → −1.17 % @C=320）。
   在 C≥128 的实用区间差异已在 3 % 以内。**单次运行、无重复**，小于 3 % 的差异应视为未测量波动。
6. **两种 CUDA graph 模式都能 profile，给的是不同层次**：graph-ON 有 kernel 层但没有 CPU 算子层和
   启动参数；graph-OFF 两者都有。**归因必须用与被测配置相同的模式**——已发布数字全程走 graph，
   所以表 3 来自 graph-ON。

### 明确留白（不猜）

| # | 未决问题 | 需要的测量 |
|---|---|---|
| 1 | kernel 合计与 DeviceTimer 相差 **4.9 %** 的构成 | trace 时间线的 gap 分析 |
| 2 | graph-OFF 那 48.62 ms/iter 非 kernel 时间的具体去向 | 同上（20 µs/次下发只是数量级参考，非测定） |
| 3 | **通信占比能否解释 EP1 更快** | EP1 的对照 profiling（最高优先级） |
| 4 | C=224 的 EP8/EP1 反号（+0.19 %） | 该点重复 3 次以上，比较离散度与 0.19 % 的关系 |
| 5 | 两种模式下集合通信构成不同（`allgather_vec` n=20 vs n=890） | 未查 |
| 6 | `idle` 类别从未触发 | 需要 DP 分片 batch 不等的场景 |
| 7 | 仅 rank 0 出 trace，各 rank 分布是否一致未验证 | 多 rank trace |
| 8 | `DEBUG_CLR_GRAPH_PACKET_CAPTURE` 未被刻画 | 见 profiling packup `research/rocm_profiling_env.md` §2（标为 provisional）。**不要设置该 flag** |

---

## 复现指引

| 目标 | 入口 |
|---|---|
| EP8/EP1 并发扫描 | `packups/glm52_tp8_dpa_ep8_vs_ep1_sweep_yihou.packup_20260914-0740/REPRODUCE.md` |
| C=256 profiling | `packups/glm52_tp8_ep8_dpa_c256_isl70k_osl10k_decode_profiling_yihou.packup_20260915-0130/REPRODUCE.md` |
| profiling 能力源码 | `glm52_decode_internal_yihou_20260909_1057/bench/{profile_decode.py,profiling_yihou.py}`，commit `23f472d2` |

**两个 packup 的 `notes.md` 记录了所有会产出"看起来合理但错误"结果的陷阱**，其中三个（标注重复计时、
DeviceTimer 类别名随模式变化、通用 kernel 名掩盖多个 kernel）若未发现，都会直接污染本文的表格。
