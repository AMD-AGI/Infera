# sglang#48302 review 意见清单（按优先级）

- PR：[sgl-project/sglang#48302](https://github.com/sgl-project/sglang/pull/48302)，`[Fix] [AMD] dsa: One decode graph variant on every attention-DP rank, one verify graph`，作者 llying-001
- 审阅对象：PR head `7f393397`（11-07 合并过 main），对照最新 main `affb3eb5`（11-11）
- 现状：没有任何人工 review（只有机器人留言）；11-09 已 ping HaiShaw、hnyls2002、ch-wan，还没有回应。所以本清单的意见全部来自我的 review。
- **只给意见，没有改代码。**
- 原理和执行过程的详细讲解见同目录 `EXPLAIN.zh-CN.md`。

## 总览

| 编号 | 优先级 | 一句话 | 处理方式 |
|---|---|---|---|
| R1 | P1 | 正确性完全依赖"上界不小于真实最大长度"，而这只在 v0.5.22 build 上审计过，main 已前进 300 个提交 | 在最新 main 上重做审计和乱码探测；可选加一个默认关闭的 debug 断言 |
| R2 | P1 | 分支落后 main 300 个提交，最近一次 CI 还是 11-07 的 head，有 10 项失败 | 合并 main、重跑 CI，确认失败项与本 PR 无关 |
| R3 | P2 | "所有 rank replay 同一张图"只覆盖 decode/idle：DSV4.1 的 DSpark verify 和 NVIDIA megakernel 的 variant 不在范围内 | 描述写明范围；更好是把 DSpark verify 也纳入，或加一次性 warning |
| R4 | P2 | 标题写 `[AMD] dsa`，但 NVIDIA 上 DSV4.1 decode 的选图行为也变了 | 描述写明并说明为什么安全，请 NVIDIA 侧确认 |
| R5 | P2 | 描述里"没有 DP attention 时规则不变"不完全准确 | 改一句描述 |
| R6 | P2 | 没有预先回答"为什么不在 DP attention 下直接只用 sparse/fallback" | 描述补一段设计取舍 |
| R7 | P3 | 有一个测试用 `inspect.getsource` 加正则检查源码文本 | 改成行为测试 |
| R8 | P3 | MLP sync 行的列号写死为 13 | 现有测试已兜底；合并 main 时核对即可，可选补注释 |
| R9 | P3 | reviewer 跟进 | R2 完成后再 ping 一次，点名需要的 codeowner |

---

## P1

### R1　正确性依赖"上界 ≥ 真实最大长度"，但只在旧 build 上验证过

- **问题**：PR 让每个 rank 上报"本 rank 这一步最长序列的上界"，取各 rank 最大值后统一选图。整个修法是否正确，取决于这个上界**永远不小于**任何 rank 的真实最大长度。PR 描述里写明：端到端验证（包括 21,504 次 replay 的审计）是在 v0.5.22-based build 上做的，在 main 上只跑了单测。
- **原理**：
  - 三种 variant 都是"用更大的上界选图，结果仍然正确；上界偏小，结果可能悄悄出错"：
    - DSA：dense 图跳过 indexer 的 logits 计算，只有当所有序列都 ≤ `index_topk`（2048）时才正确（`dsa_indexer.py` 注释："topk_transform selects every valid page slot when kv_len <= index_topk"）。上界偏小导致选了 dense，而某条序列 > 2048，这条序列就只看到一部分 KV，不报错，只是输出变差。
    - aiter 的 `SeqLenBucketGraphVariants`：按长度选分区数，`aiter_backend.py` 注释原话："too few drop KV silently"。
    - DSV4.1：上限更大的 variant 是更通用的那个，选大了只影响性能。
  - 上界有两个来源，都依赖调度器的语义：
    - 有 host mirror 时用 `seq_lens_cpu.max()`。这要求调度时的 `seq_lens_cpu` 和 replay 时设备上的长度完全一致。
    - 没有 mirror 时（spec 的 acceptance 还在路上）用 `max(req.seqlen) + 2 × step_width`。这要求"最多落后一步"这个假设成立。
  - main 在这 300 个提交里改过不少调度器和 spec 的代码，例如 #49623（每个请求记录持有它的列表）、#49309（mamba 的 verify 状态提交）、#47627（DSv4.1 verify 元数据提前准备）、#49253（暂停时回收 KV）。这些改动是否影响上面两个假设，目前没有验证。
- **怎么改**：
  - 在最新 main 上重做一次审计：审计版 runner 每次 replay 读一次设备上的真实最大长度，与上界比较，统计"上界 < 真实值"的次数（应为 0）以及各 rank 是否选了同一个 variant。再加一次 GSM8K 乱码探测和限池门禁（触发持续驱逐的那种配置）。
  - 可选：在 runner 里加一个默认关闭的环境变量开关，打开后每次 replay 做上面的比较并在违反时报错。这样 reviewer 或以后的 CI 都能复现这个检查。
  - 做完后更新描述的 Limits 段。

### R2　分支过期，CI 需要刷新

- **问题**：
  - PR 的 merge-base（`043cfc4d`）之后，main 又进了 300 个提交；最近一次 CI 跑的是 11-07 的 head。
  - 当前 10 项失败：Intel CPU 的 3 个 `build-test`（日志显示 "The operation was canceled"，是被取消的）、MLX 单测（挂在 `test_attention_patching.py`，与本 PR 无关）、几个 `*-finish` 收尾检查和 `pr-gate`。
  - 我用 `git merge-tree` 检查过：和最新 main 没有文本冲突。main 上 MLP sync 行仍是 13 列，PR 新加的第 14 列（下标 13）在合并后依然正确。
- **原理**：合并需要必需检查全绿；reviewer 一般也希望看到基于新 main 的结果。另外 PR 改的是调度器热路径上的共享数据结构，main 上同一文件又有改动（见 R1），刷新后再跑一遍 CI 更稳妥。
- **怎么改**：合并最新 main，推送触发 CI；确认新单测 `test_graph_variants_dp_consistency.py` 在 `base-a-test-cpu` 里通过；Intel CPU 如果还失败，先看是不是又被取消，必要时 `/rerun-failed-ci`。

---

## P2

### R3　覆盖范围：只有 decode/idle 保证所有 rank 选同一张图

- **问题**：PR 只在 decode/idle 模式下用全局上界选图。下面两类 variant 不在范围内：
  1. **DSV4.1 的 DSpark target verify**（HIP gfx95 和 NVIDIA SM100 以上都会创建）：`Dsv41CandidateGraphVariants.select()` 在 TARGET_VERIFY 时仍按本 rank 的长度选；DSpark 的 verify 元数据也是按本 rank 的已提交长度提前准备的（`decode_cuda_graph_runner.py` 的 `_early_variant_for` / `stage_verify_metadata_early`）。
  2. **NVIDIA megakernel** 的 verify 和 draft variants（`kernel_plugins/nvidia/binding.py`），PR 完全没有涉及。
- **原理**：
  - 引发 bug 的机制和模式无关：只要图里有跨 rank 的集合通信，并且这些通信在 capture 时登记了本图的 buffer 地址，各 rank replay 不同的图就会读到对方另一张图的旧数据。所以理论上这个规则应该对所有 variant、所有模式成立。
  - 一个具体的小坑：在 DSpark verify 步里，空闲 rank 的 forward mode 是 IDLE，PR 之后它会按全局上界选图；而忙碌 rank 是 TARGET_VERIFY，按自己的长度选。两者仍可能不同（PR 之前空闲 rank 总是选 fallback，同样可能不同），所以这一点不算 PR 引入的回归，但 PR 也没有解决。
  - 这些路径不影响 GLM-5.2 on MI355X（PR 的目标场景）：GLM-5.2 不是 `deepseek_v41`，AMD 上也没有 NVIDIA megakernel。
- **怎么改**（任选，越往后越彻底）：
  - 描述里写明"只覆盖 decode/idle；DSV4.1 的 DSpark verify 和 NVIDIA megakernel 的 variant 不在本 PR 范围内"。
  - 在 runner 里加一次性 warning：DP attention 下、存在 variant、当前模式又不在覆盖范围内时打印一次，方便以后定位。
  - 把 DSpark verify 也纳入：提前准备元数据时用 `max(本 rank 长度, 全局上界)` 选 variant，空闲 rank 用同一规则。改动面较大，可以作为后续 PR。

### R4　对 NVIDIA 的行为改变没有写进标题和描述

- **问题**：标题是 `[AMD] dsa`，但 `Dsv41CandidateGraphVariants` 在 NVIDIA SM100 以上的 DECODE capture 里同样存在。在 DP attention 下，PR 之后它改为按全局上界选图。
- **原理**：
  - 正确性上是安全的：全局上界 ≥ 本 rank 的最大长度，选到的 variant 上限只会更大，而上限更大的 variant 是更通用的。
  - 性能上有取舍：短上下文的 rank 会跟着最长的 rank 选更通用、可能更慢的 variant。
  - NVIDIA 的维护者 review 时多半会问到这一点。
- **怎么改**：在描述里写明"NVIDIA 上 DSV4.1 的 decode variant 在 DP attention 下也改为按全局上界选，正确性不变，短上下文 rank 可能用更通用的 variant"，并请 DSV4.1 的 owner 确认；有条件的话补一次 NVIDIA 上的冒烟测试。

### R5　描述里"没有 DP attention 时规则不变"不完全准确

- **问题**：描述写的是 "Without the sync (no DP attention) the per-rank rule is unchanged"。
- **原理**：`require_mlp_sync()` 在 `attn_dp_enabled` **或** `require_gathered_buffer()`（例如某些 MoE A2A 后端）时为真。后一种情况下只有一个 DP rank，走 `finalize_local()`，`metadata_ready` 为真，上界同样会写到 batch 上，选图改用上界：
  - 有 host mirror 时，上界就是本地精确最大值，结果和原来完全一样；
  - 没有 mirror 时，上界是"已提交长度 + 2 × 步宽"，比原来读设备最大值稍微保守（边界附近偶尔会选 sparse），不影响正确性，反而省掉了一次 `.item()` 同步。
- **怎么改**：描述改成"没有 MLP sync 时规则不变；只有一个 DP rank 时，上界就是本地值（有 host mirror 时与原规则完全相同）"。

### R6　没有预先回答设计上的替代方案

- **问题**：reviewer 很可能会问："既然 dense 只占 3.7% 的 replay，为什么不在 DP attention 下直接只用 sparse（或 fallback），省掉新加的同步列和一串字段传递？"
- **原理**：PR 的 +1.3% 主要来自去掉 verify 前的 `.item()`，而不是来自 dense。但对另外两种 variant，"一律用 fallback"代价更大：aiter 的分区桶注释里写了，在 batch 256、1.5K token 时，32 和 128 个分区比 16 个分区分别慢 1% 和 9%。而新加的列只是在已有的 all-gather 里多一个 int64，没有新的集合通信，也没有新的设备读取。
- **怎么改**：在描述里补一段设计取舍：
  - 为什么用"同步上界"而不是"一律用通用图"；
  - 代价是每步每个 rank 多一个 int64，搭在现有的 all-gather 上；
  - 性能取舍是短上下文 rank 跟随最长 rank 选图。
  - 如果能给一组"DP attention 下一律 sparse"的对照数据更有说服力。

---

## P3

### R7　测试检查源码文本

- **问题**：`test_bound_reaches_the_forward_batch` 用 `inspect.getsource(...)` 加正则，检查 `ScheduleBatch.copy`、`ForwardBatch.init_new` 和 `two_batch_overlap` 的源码里有没有某段文字。
- **原理**：这种测试不测行为，格式化或重构（比如换行、改变量名）就会误报，reviewer 常常不喜欢。
- **怎么改**：改成行为测试，例如构造一个最小的 `ScheduleBatch`（或 mock），调用 `copy()` 检查字段被带过去；TBO 那部分直接调用 `filter_batch` 检查子 batch 带有这个字段。

### R8　列号写死为 13

- **问题**：`all_gather` 里用 `tp0_info_cpu[:, 13]` 读新列。这个 PR 存续期间，main 已经两次往这一行里插列（#48447、#48492）。
- **原理**：和现有代码风格一致，而且已有测试 `test_sync_row_carries_the_bound` 断言新列在最后一列；但每次合并 main 都要人工确认没有新列插到它前面。
- **怎么改**：合并 main 时核对即可（目前 main 上仍是 13 列，下标正确）；可选在 `_get_local_tensor` 旁边补一行注释说明列序。

### R9　reviewer 跟进

- **问题**：11-09 ping 过 HaiShaw、hnyls2002、ch-wan，至今没有回应；合并需要 `layers/attention`（graph_variants.py）和 `managers`、`model_executor` 的 codeowner 批准。
- **怎么改**：R2 完成（合并 main、CI 绿）后再发一条简短状态，点名对应目录的 codeowner；如果 R1 的审计在 main 上做完了，一并附上结果。

---

## 已核实没有问题的点

| 检查项 | 结论 |
|---|---|
| 和最新 main 的文本冲突 | 没有（`git merge-tree`） |
| 新列的下标 | main 上 MLP sync 行仍是 13 列，PR 的新列是第 14 列，`tp0_info_cpu[:, 13]` 正确 |
| `skip_global_metadata=True` 时上界设为 `None` | 只在没做 all-gather（DWDP 或 attn-dp=1）时发生，此时各 rank 本来就独立，退回老规则没问题 |
| 空闲 rank、PD decode 的 prebuilt batch | 空闲 rank 的 idle batch 和 prebuilt batch 的 inner idle batch 都会经过 `_update_gather_batch` 拿到上界 |
| 上界能否传到 runner | main 上没有 `ModelWorkerBatch` 了，`tp_worker.py` 里 `ForwardBatch.init_new` 直接接收 `ScheduleBatch`；PR 在 `init_new` 里带上了字段，`copy()`（overlap 下进入结果队列的那份）也带上了；TBO 的子 batch 也带上了 |
| verify capture 只抓一张图是否安全 | `DSA_DENSE` 只有 `dsa_indexer.py` 在用，而且只在 decode/idle 生效；verify 模式下两张图完全相同 |
| verify capture 返回 `None` 后会不会误落到别的 variant | DSV4.1 工厂要求 `model_type == "deepseek_v41"`；aiter 的分区桶只在 DECODE capture；NVIDIA megakernel 在 AMD 上返回 `None`。GLM-5.2 的 verify 最终是单图 |
| `DsaGraphVariants` 是否只影响 AMD | 只在 `is_hip()` 时创建 |
| 其他调用 `_update_gather_batch` 的地方 | 只有一个正式调用点；两个构造假对象的测试 PR 都已补上新属性 |
| Intel CPU、MLX 的失败 | 前者是作业被取消，后者挂在 `test_attention_patching.py`，都与本 PR 无关 |
