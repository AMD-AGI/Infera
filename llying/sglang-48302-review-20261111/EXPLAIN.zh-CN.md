# sglang#48302 原理与执行过程详解（零基础版）

这份文档用具体例子讲清楚：这个 PR 要解决什么问题、为什么会出问题、它怎么修、修完后一次 decode 从调度到 replay 的数据怎么流动。配数据流图和接口调用流程图（纯文本，任何编辑器都能看）。

- PR：[sgl-project/sglang#48302](https://github.com/sgl-project/sglang/pull/48302)，head `7f393397`
- 意见清单见同目录 `REVIEW-LIST.zh-CN.md`

---

## 0. 一句话

**在 DP attention 下，各个 rank 必须在同一步 replay 同一张 decode 图；以前每个 rank 按自己的最长序列选图，可能选得不一样，导致输出乱码。这个 PR 让每个 rank 在每步已有的同步里多报一个数（本 rank 最长序列的上界），大家取最大值后用同一个值选图。** 顺带让 verify 模式只抓一张图，并去掉 verify 前的一次 GPU→CPU 同步。

---

## 1. 先认识五个概念

### 1.1 decode 图（CUDA/HIP graph）

decode 每步都执行同样的一串 kernel。SGLang 在启动时把这串 kernel **录制（capture）** 成一张图，之后每步直接 **重放（replay）**，省掉 CPU 一个个发 kernel 的开销。每种 batch size 各录一张。

### 1.2 graph variant（同一 batch size 录多张图）

有时候同一个 batch size 需要两种不同的 kernel 序列，于是录多张，叫 variant；每步 replay 前由 `select()` 决定用哪张。图的查找键大致是 `(batch size, LoRA variant, attention variant)`。这个 PR 涉及三种 variant：

| variant 类 | 用在哪 | 按什么选 | 选"大"了会怎样 | 选"小"了会怎样 |
|---|---|---|---|---|
| `DsaGraphVariants`（只在 HIP） | DSA 模型（GLM-5.x、DeepSeek-V3.2） | 最长序列 ≤ `index_topk`（2048）选 dense，否则 sparse | sparse 永远正确 | 选了 dense 而有序列 > 2048：那条序列只看到部分 KV，**静默出错** |
| `SeqLenBucketGraphVariants` | aiter 后端的 decode 分区桶（开环境变量时） | 最长序列落在哪个长度桶 | 分区多一点，稍慢 | 分区太少会**静默丢 KV** |
| `Dsv41CandidateGraphVariants` | DeepSeek-V4.1（HIP gfx95、NVIDIA SM100+） | 最长序列落在哪个上限 | 用更通用的 variant，稍慢 | 可能用错算法分支 |

**规律：按"偏大"的长度选永远安全，按"偏小"的长度选可能静默出错。** 这就是后面"上界"必须不小于真实值的原因。

### 1.3 DSA 的 dense 和 sparse

DSA（DeepSeek Sparse Attention）有一个 indexer：给每个 query 从历史 token 里挑出最相关的 `index_topk`（2048）个，attention 只看这些。

- **sparse 图**：完整跑 indexer（算 logits、取 top-k）。
- **dense 图**：如果所有序列都不超过 2048，top-k 等于"全选"，logits 就不用算了，只写 K。省一点计算。

### 1.4 DP attention 和 lockstep

以 GLM-5.2 的 decode（TP4、DP4 attention）为例：4 张 GPU，每张卡（rank）各自持有一批不同的请求，各算各的 attention；MoE 那部分则 4 张卡一起算，中间要做 all-gather、reduce-scatter 等集合通信。所以 4 个 rank 必须**同步前进（lockstep）**，每一步都同时进入同一个集合通信。没有请求的 rank 也要跑一个空的 IDLE 步陪着。

每一步开始前，调度器会做一次 **MLP sync**：每个 rank 把一行整数（本步 token 数、能不能用图等）放进一次 all-gather，大家拿到所有 rank 的行，据此做一致的决定（比如"只要有一个 rank 不能用图，大家都不用"）。

### 1.5 图里的集合通信会"登记 buffer 地址"

aiter 的 all-gather / reduce-scatter 和 custom all-reduce 为了快，在 capture 时就把**这张图的输入 buffer 地址**登记给其他 rank；replay 时直接按登记的地址去读对方的数据。也就是说，**图 A 里的通信读的是对方图 A 的 buffer，图 B 读的是对方图 B 的 buffer。**

---

## 2. 问题是怎么发生的：一个具体例子

某一步，4 个 rank 的情况如下（老规则：各自按本 rank 最长序列选；空闲 rank 总选 sparse）：

| rank | 本 rank 最长序列 | 老规则选的图 |
|---|---|---|
| 0 | 1,500 | dense（≤ 2048） |
| 1 | 9,000 | sparse |
| 2 | 空闲 | sparse |
| 3 | 300 | dense |

lockstep 下，4 个 rank 同时 replay，并在图里的 all-gather 处配对：

```text
              rank0 (dense 图)          rank1 (sparse 图)
  写入数据 →  dense 图的 buffer          sparse 图的 buffer
  all-gather：按登记地址读对方 ──────►   dense 图的 buffer   ← rank1 这一步根本没往这里写！
                                          （里面是上次 replay dense 时留下的旧数据）
```

rank0 从 rank1 那里读到的是旧数据，MoE 算出来就错了，输出变成乱码。PR 描述里的实测：GLM-5.2-MXFP4、MI355X、PD decode TP4 + DP4，200 条 GSM8K 里有 26 条乱码；把所有 rank 固定成同一个 variant 后乱码全部消失，证实了原因。

---

## 3. 修法总览

**每个 rank 在 MLP sync 那一行里多加一列：本 rank 这一步最长序列的上界。** all-gather 之后取所有 rank 的最大值，每个 rank 都用这个最大值选图。

同一个例子：

| rank | 上报的上界 |
|---|---|
| 0 | 1,500 |
| 1 | 9,000 |
| 2 | 0（空闲） |
| 3 | 300 |
| **全局最大** | **9,000 → 4 个 rank 都选 sparse** |

另一步，如果 4 个 rank 的上界是 1,500 / 1,800 / 0 / 300，全局最大 1,800 ≤ 2048，4 个 rank（包括空闲的）都选 dense。

代价：每步每个 rank 多一个 int64，搭在已有的 all-gather 上；没有新的集合通信，也不读设备。

---

## 4. 数据流图

```text
 每个 rank 的调度器（CPU）                                         每个 rank 的模型 runner（GPU）
 ─────────────────────────                                         ──────────────────────────

 本步的 ScheduleBatch
   │  _local_decode_graph_max_seq_len()
   │    有 seq_lens_cpu：取它的 max
   │    否则：max(req.seqlen) + 2×步宽
   │    空闲 / 非 decode：0
   ▼
 MLPSyncBatchInfo（本 rank 一行 14 列）
   │ 列 0..12：原有内容（token 数、能否用图、forward mode、drain 状态…）
   │ 列 13  ：decode_graph_max_seq_len   ← 新增
   ▼
 all_gather（所有 rank 的行拼成 tp0_info_cpu，形状 [rank 数, 14]）
   │
   ▼
 global_decode_graph_max_seq_len = tp0_info_cpu[:, 13].max()
   │  _update_gather_batch()
   ▼
 ScheduleBatch.dp_decode_graph_max_seq_len
   │  （copy() 也带上它：overlap 调度把 batch.copy() 放进结果队列）
   ▼
 ForwardBatch.init_new()（tp_worker 里）  ─────────────────────►  ForwardBatch.dp_decode_graph_max_seq_len
                                                                     │
                                                                     ▼
                                                                   DecodeCudaGraphRunner.replay()
                                                                     │ _resolve_attention_variant()
                                                                     │   → variants.select(forward_batch)
                                                                     │       → dp_decode_max_seq_len()：decode/idle 时返回全局上界
                                                                     ▼
                                                                   label（dense / sparse / 某个桶）
                                                                     │
                                                                     ▼
                                                                   replay graphs[(bs, ..., label)]
                                                                   （4 个 rank 的 label 相同）
```

MLP sync 那一行的列含义（PR 合并后）：

| 列 | 内容 | 汇总方式 |
|---|---|---|
| 0 | num_tokens | 列表 |
| 1 | num_tokens_for_logprob | 列表 |
| 2 | can_run_decode_cuda_graph | min（有一个不能用，大家都不用） |
| 3 | is_extend_in_batch | max |
| 4、5 | TBO 相关、local_forward_mode | 交给 TBO 计算 |
| 6、7 | prefill 图相关 | min / max |
| 8 | can_run_draft_cuda_graph | min |
| 9 | is_health_check | max |
| 10 | attention_pairs | 列表 |
| 11 | is_shutting_down | max |
| 12 | drain_state | 列表 |
| **13** | **decode_graph_max_seq_len（新增）** | **max** |

---

## 5. 接口调用流程图

### 5.1 每一步（调度 → replay）

```text
Scheduler 事件循环
 └─ get_next_batch_to_run()
     ├─ update_running_batch()
     │    └─ ScheduleBatch.prepare_for_decode()      seq_lens、seq_lens_cpu 各 +1（本步要算的位置）
     └─ dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)     （在 prepare_for_decode 之后，所以上界是本步的长度）
          └─ prepare_mlp_sync_batch_raw(local_batch, ...)
               ├─ MLPSyncBatchInfo(...,
               │     decode_graph_max_seq_len = _local_decode_graph_max_seq_len(local_batch, spec_algorithm))   ← 新
               ├─ 只有 1 个 DP rank：finalize_local()       全局值 = 本地值
               │  否则：all_gather()                         全局值 = 第 13 列的 max                              ← 新
               ├─ 需要时为空闲 rank 造 idle batch（PD decode 的 prebuilt batch 用它的 inner idle batch）
               └─ _update_gather_batch(batch_to_gather, info, ...)
                    └─ batch.dp_decode_graph_max_seq_len = 全局值（没做 all-gather 时为 None）            ← 新

 └─ run_batch(batch)
     └─ TpModelWorker：ForwardBatch.init_new(batch, model_runner)
          └─ 复制 dp_decode_graph_max_seq_len                                                             ← 新
     （overlap 调度下 batch.copy() 进入结果队列，copy() 也带上这个字段）
     └─ ModelRunner.forward(forward_batch)
          └─ DecodeCudaGraphRunner.can_run_graph() / replay()
               └─ _resolve_attention_variant(forward_batch, bs)
                    └─ attention_graph_variants.select(forward_batch)
                         ├─ dp_decode_max_seq_len(forward_batch)                                         ← 新
                         │    forward mode 是 decode 或 idle 且有全局值 → 返回全局值；否则 None
                         ├─ 有全局值：按它选（所有 rank 一致）
                         └─ 没有：老规则（按本 rank 的 seq_lens_cpu，DSA 还可能退到设备 max 的 .item()）
```

### 5.2 启动时（capture）

```text
DecodeCudaGraphRunner.__init__()
 ├─ capture_forward_mode = DECODE（无 spec）或 TARGET_VERIFY（MTP 等 spec）
 └─ attention_graph_variants =
      create_attention_graph_variants(hf_config, capture_forward_mode)                ← 新增参数
         ├─ HIP 且是 DSA 模型？
         │    ├─ KPool > 1 → None（交给 DSV4.1）
         │    ├─ capture_forward_mode 不是 decode → None                              ← 新：verify 只抓一张图
         │    └─ 否则 → DsaGraphVariants(index_topk)（dense + sparse 两张）
         └─ 否则 None
      or create_dsv41_candidate_graph_variants(...)     只对 deepseek_v41
      or attn_backend.get_decode_graph_variants(...)    aiter 分区桶，只在 DECODE
      （verify 且仍为 None 时，NVIDIA megakernel 还会再试一次；AMD 上返回 None）
```

为什么 verify 只需要一张图：dense 唯一的作用是让 indexer 的 `_should_skip_logits_computation()` 跳过 logits，而这个函数只在 decode/idle 模式下看 variant；verify 模式下两张图跑的完全是同一套 kernel。PR 之前 MTP 部署会把 verify 图录两遍，纯属浪费；而且 verify 时 `select()` 拿不到 host 长度，每次 replay 前都要 `seq_lens.max().item()` 做一次 GPU→CPU 同步。PR 之后 verify 只有一张图，不用选，这次同步也没了（实测每步 verify 前的 GPU 空闲从 0.59 ms 降到 0.03 ms）。

---

## 6. 上界怎么算（`_local_decode_graph_max_seq_len`）

| 情况 | 上报值 | 例子 |
|---|---|---|
| 没有 batch、不是 decode、batch 为空（含空闲 rank） | 0 | 空闲 rank 报 0 |
| 有 host 镜像 `seq_lens_cpu` | `seq_lens_cpu.max()`（精确） | 普通 decode：本 rank 3 条请求长度 [9, 1200, 4000] → 4000 |
| 没有 host 镜像（spec 的 acceptance 还没回来、或 merge 时丢了镜像） | `max(req.seqlen) + 2 × 步宽` | spec 每步最多 4 个 draft token，已提交最长 3990 → 3990 + 2×4 = 3998；普通 decode 步宽 1 → 3990 + 2 = 3992 |

为什么是"两步宽度"：overlap 调度下最多有一步还在 GPU 上跑、结果没回到 CPU（落后一步），再加上本步自己要新增的 token（又一步）。

为什么必须是**上界**：见 1.2 的表，偏大只损失一点性能，偏小可能静默出错。PR 描述里的审计：21,504 次 replay 中，上界从未小于真实最大值；所有 rank 每次都选了同一个 variant（老规则会在 1,644 次上不一致）；dense 仍然被用到（3.7%）。

---

## 7. 各种情况下的行为

| 场景 | 全局值 | 选图规则 |
|---|---|---|
| DP attention，decode / idle | 第 13 列的 max | 所有 rank 按全局值选，一致 |
| DP attention，verify（DSA，HIP） | 不使用 | 只有一张图，不用选 |
| DP attention，verify（DSV4.1 DSpark） | 不使用（idle rank 除外） | 仍按本 rank 长度，**不保证一致**（意见清单 R3） |
| 没做 all-gather（DWDP、attn-dp=1） | None | 老规则（各 rank 本来独立） |
| 只有 1 个 DP rank 但需要 MLP sync（某些 MoE A2A） | 本地上界 | 有 host 镜像时与老规则完全相同（意见清单 R5） |
| 不需要 MLP sync | None | 老规则，完全不变 |
| TBO（双 batch 重叠） | 子 batch 继承父 batch 的全局值 | 一致 |

---

## 8. 和 PR 描述里的数据对应

- **正确性**：对照组（镜像原样）400 条 20 条乱码；修复后 600 条 0 条。MoE gather/combine 放进图里时，对照组 200 条 53 条乱码，修复后 400 条 0 条。不开 MTP（只有 decode 图）时对照组 200 条 33 条，修复后 800 条 0 条。
- **性能**：+1.3%（50,797 → 51,477 tok/s/GPU），主要来自去掉 verify 前的 `.item()`；同时让"MoE gather/combine 放进图里"这个原本会出错的优化变得可用，开启后 +5.5%（53,609）。
- **范围限制**：端到端验证是在 v0.5.22-based build 上做的，main 上只跑了单测（意见清单 R1）。

---

## 9. 代码在哪里

| 文件 | 改了什么 |
|---|---|
| `managers/scheduler_components/dp_attn.py` | 新列 `decode_graph_max_seq_len`；`_local_decode_graph_max_seq_len()` 算本地上界；`all_gather` 取第 13 列 max；`_update_gather_batch` 写到 batch |
| `managers/schedule_batch.py` | `ScheduleBatch.dp_decode_graph_max_seq_len` 字段，`copy()` 带上 |
| `model_executor/forward_batch_info.py` | `ForwardBatch` 同名字段，`init_new` 带上 |
| `batch_overlap/two_batch_overlap.py` | TBO 子 batch 带上 |
| `layers/attention/graph_variants.py` | `dp_decode_max_seq_len()`；三种 variant 的 `select()` 优先用全局值；`create_attention_graph_variants` 增加 `capture_forward_mode`，verify 不建 DSA variant |
| `model_executor/runner/decode_cuda_graph_runner.py` | 把 `capture_forward_mode` 传给 `create_attention_graph_variants` |
| `test/registered/unit/layers/test_graph_variants_dp_consistency.py` | 11 个 CPU 单测 |
| 两个已有测试 | 给构造的假对象补 `global_decode_graph_max_seq_len` |
