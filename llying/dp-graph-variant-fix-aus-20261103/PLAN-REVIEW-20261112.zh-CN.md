# 按 Pareto 目标复核 11-09 的 kernel 计划（2026-11-12 08:20）

计划原文：`KERNEL-PLAN-20261109.zh-CN.md`（第三节清单、第五节方案、第七节后续顺序）；实测记录：`KERNEL-PROGRESS-20261109.zh-CN.md`。

## 0. 结论

**主体还能用，但要做 4 处修改。**原计划是按"TP8 decode 每步能省多少"排序的，而 decode 变快就是让 Pareto 曲线右移，所以方向没错。要改的是：

1. **按并发分段衡量收益。**原计划只用 C208（繁忙）的 trace。曲线左端（低并发、高交互度）由固定开销决定：小 kernel 的启动地板、MegaMoeTP 在小 batch 变慢。要补上每卡 batch 2–8 的数据和目标。
2. **prefill profile 作为并行的另一条线。**它决定曲线最右端那个点（最高吞吐）。原计划也建议并行做，但一直没做。
3. **把"交付"提成一条正式的工作线。**InferenceX 不允许 overlay / 运行时替换函数，而我们现在所有的收益——包括生产在用的 KF——都靠 overlay 生效。不进 aiter / SGLang 上游，曲线上就不算数。原计划第 5.6 节写了，但一项都没做。
4. **按现在的数据重新排序。**MoE 仍然第一；bf16 稠密 GEMM 的空间已经基本用完（不改精度的话）；还查出一个漏项——**K5 一直没进 TP8 的实验配置**，可以立即补上。

## 1. 计划各项的状态

### 1.1 decode 清单（计划第 3.1 节，"可省"是计划里的理论值）

| # | 项目（计划估计 TP8 可省） | 已做 | 未做 | 现在还剩 |
|---|---|---|---|---|
| 1 | **TP8 MoE 全链路**（5.5–6.5 ms） | 调优表补行（端到端已验证）；MegaMoeTP 接入、调参、q3（graph 内 −1 ~ −4 ms / 步；同节点 A/B 交互度慢尾 +7.7%） | KF 移植到中间维 256（A 线）；AKLR forge（B 线）；AKLR 整层插件对照（C 线）；前处理融合（K2）；gemm2 写回（K1）；用真实路由数据测试 | **3–6 ms**：MegaMoeTP 只有 3.3–3.8 TB/s，TP4 的 KF 是 5.81 TB/s |
| 2 | MLA absorb BMM（1.5–1.8 ms） | 两个 Triton kernel 的配置（W_vc −44%，W_kc −21%；TP4 graph 内 −0.95 ms，门禁通过，已进 kopt 基线） | 新 kernel（目标每层 18 µs） | 约 0.7 ms（现在每层 27 µs） |
| 3 | 稠密投影（bf16 1.5–1.8 ms；FP8 3–4 ms） | AKLR 现成 kernel（无收益）、调参（3–4%）、Triton 原子累加（更慢）、调优表补行（−0.1 ms）、GEMM 合并原型（−0.25 ms） | 新的 split-K kernel；dense FP8（需你决定） | bf16 约 0.5 ms（o_proj 已到约 5 TB/s）；**FP8 约 3 ms**（11-06 测时被激活量化和低效的 FP8 GEMM 抵消，需要先解决这两点） |
| 4 | sparse MLA（1.3–1.6 ms） | 原因查清（KV 池超过 2 GiB + 86 行的 split 台阶）；拆两段的修法（−0.28 ms，逐位一致）；KV 上限方案（−1.1 ms，容量 −9%） | 86 行的 split 台阶（84 → 88 行 +23%）；完全追平小池的 kernel | 0.3–1.1 ms + 约 0.5 ms（台阶） |
| 5 | indexer logits，fused2 的 K3 版本（0.6–0.8 ms） | 实现与验证（逐位一致，每次 −6 ~ −27 µs）；**已进 TP4 实验配置** | **没进 TP8 的任何实验配置**（计划 7.2 第 5 项"下次 PP8 臂加上"漏了） | **0.4–0.6 ms，只改配置** |
| 6 | indexer 前处理 + top-k（1.0–1.3 ms） | 今天：trace 估计"繁忙时按 48 行分段调用融合 indexer"约 −0.9 ms | microbench 确认；AKLR 的 indexer 前处理插件没测 | 约 0.9 ms（待确认） |
| 7 | 通信（1.0–1.5 ms） | MegaMoeTP 把 MoE 的 all-gather / reduce-scatter 藏进了 kernel | M1（先量化再 all-gather，与 MegaMoeTP 二选一）；RS + add + RMSNorm 融合 | 小（MoE 部分已解决） |
| 8 | norm 融合（0.3–0.5 ms） | — | 全部 | 0.3–0.5 ms（曲线左端更重要） |

### 1.2 计划第五节的其他部分

| 项目 | 状态 |
|---|---|
| 阶段 0：工作负载表 | 部分完成：今天从源码确认了 decode 的 graph batch 列表和每卡 M（20 个值）；真实路由数据没导出 |
| 阶段 0：把生产路径包装给 AKLR | 未做 |
| 阶段 0：单节点 decode 复现（`profile_decode.py`） | 未做；改用固定并发合成负载的 graph 内 trace，效果相当 |
| 阶段 0：prefill profile | **未做**（11-09 起反复建议） |
| 阶段 0：现成 kernel 快测 | 部分完成：AKLR o_proj / qkv_a（无收益）、sparse MLA 行数对照；AKLR 整层 MoE 插件、indexer 插件没测 |
| 四级门禁（算子 → graph 内 → verify_suite → AgentX） | 每项都只做到前一两级；端到端积压 5 个实验配置 |
| 交付（aiter / SGLang PR） | **未做** |
| 计划外完成的 | MegaMoeTP 的计时、插桩与调参；EP8 布局评估（不做）；lm_head、MTP 层 MoE 的排除 |

## 2. 按 Pareto 目标重新排序（TP8 每步，繁忙时一步 37–43 ms）

| 排名 | 项目 | 估计 | 作用在曲线的哪段 | 成本 | 状态 |
|---|---|---|---|---|---|
| 1 | **TP8 MoE GEMM 效率**（到 ≥ 5 TB/s） | −3 ~ −6 ms | 中高并发 | 数天 | 需要你确认作为主线 |
| 2 | **K5 进 TP8 配置** | −0.4 ~ −0.6 ms | batch ≥ 13 | **几分钟（只改配置）** | 可以立即补上 |
| 3 | sparse MLA：KV 上限 / 拆两段 + 86 行台阶 | −0.3 ~ −1.6 ms | 全段 | 低到中 | 实验配置已就绪；台阶未做 |
| 4 | 繁忙时 indexer 分段调用融合路径 | 约 −0.9 ms | batch ≥ 13 | 1–2 天 | 待 microbench |
| 5 | **曲线左端**：MegaMoeTP 小 batch 回退 + norm / 小 kernel 融合 | 小 batch 时 −1 ~ −1.5 ms（占小 batch 一步 28 ms 的 4–5%） | 低并发、高交互度 | 1–2 天 | 未开始 |
| 6 | absorb BMM 新 kernel | −0.7 ms | 全段 | 数天 | 未开始 |
| 7 | 今天的 q3 + GEMM 合并 + 调优表 | −0.65 ms | 中高并发 | 已完成，待端到端 | 实验配置已就绪 |
| 8 | dense FP8（需改精度） | 约 −3 ms（要先解决激活量化和 FP8 GEMM 效率） | 全段 | 数天 + 精度评测 | 需你决定 |
| 并行线 | prefill profile → prefill kernel | 曲线最右端 | — | profile 约半天 | 未做 |
| 并行线 | 上游交付（MegaMoeTP 接入 SGLang、KF 的 sparse / indexer / MoE 上游、我们的修改） | 决定成绩能否在 InferenceX 上算数 | — | 多日 | 未做 |

把排名 1–7 都做完，繁忙时 TP8 每步可省约 6–11 ms（15–25%），相当于同并发下交互度提高同样的比例；曲线左端约 4–5%。

## 3. 建议的执行顺序

1. **立即（只改配置）：**给 TP8 的实验配置补上 K5（`DSA_LPT_GROUP_ROWS=4 DSA_LPT_NG2=32`），并入今天的组合。
2. **主线（数天）：**TP8 MoE。先量清楚 MegaMoeTP 里 GEMM1、GEMM2 各受什么限制，同时导出真实路由数据用于测试；再决定是改 MegaMoeTP 内部，还是移植 KF 到中间维 256。
3. **穿插（各 1–2 天）：**
   - 曲线左端：MegaMoeTP 的小 batch 回退门槛、norm 融合；
   - 繁忙时 indexer 分段的 microbench；
   - sparse 86 行台阶。
4. **每完成一批就测一次步长曲线：**固定并发合成负载，每卡 batch 2 / 4 / 8 / 12 / 16 / 20 / 24，看曲线各段的变化。
5. **并行线：**
   - 安排一次 prefill profile；
   - 列出上游交付清单：哪些改动、进哪个仓库、谁来评审，从已经稳定的项开始（K5、absorb 配置、调优表行、MegaMoeTP 的 SGLang 接入）。

## 4. 需要你决定

1. 是否以 TP8 MoE 为主线（第 3 节第 2 条）。
2. 上游交付由谁来做、什么时候开始。这决定我们的成绩能否在 InferenceX 上算数。
3. dense FP8 是否值得投入（要改精度、过精度评测）。
4. 有没有 GB300 曲线上要追的具体点（交互度、吞吐），用来把目标换算成每个并发下需要的步长。
