# 讨论前材料：U17、U19、U11、U13

截至 2026-11-25 07:00 UTC。**第一次接触这些概念的话，先读"原理讲解"一节**（第 0 节之后），里面从零讲了并行方式、MoE、集合通信、CUDA graph、锁页内存、KV offload，以及四个 PR 各自的原理和小例子。

vLLM main 现在是 `69d3af837`（11-25），比 PR 基线 `8828834fb` 多约 390 个提交。本次只读代码和 GitHub 状态，没有跑测试。

| 项 | 当前状态 | 你需要做的 | 讨论时要定的 |
|---|---|---|---|
| U17 | 已在上游：[#69616](https://github.com/vllm-project/vllm/pull/69616) | **逐行审 diff，核对正文里的数字** | 没人 review 时找谁推动 |
| U19 | 已在上游：[#69617](https://github.com/vllm-project/vllm/pull/69617) | **逐行审 diff**；看与 #69254 的说法 | 是否主动去 #69254 留言；行很小时调用次数的上界 |
| U11 | fork #4，未上游；上游 draft #69348 是同一修法 | 读 #69348 的 diff（6 行） | 关不关 fork #4；在 #69348 留不留言、给哪些数据 |
| U13 | fork #1，未上游；上游 #68006 已修一半 | 审改写补丁（源码 +11 / −4） | 用不用改写版；集群重测的范围和方案 |

## 0. 一件需要你先确认的事

#69616 和 #69617 的正文里都有这句：

> the submitter reviewed every changed line and the measurements.

vLLM `AGENTS.md` 规定提交人必须确实逐行审过、能"端到端为改动辩护"，否则可能被自动封号。两个 PR 现在已公开，**请先把下面第 1、2 节的 diff 逐行看完**。如果有看不懂或不同意的地方，我们讨论时优先处理；需要改的话，我更新分支和正文。

---

## 原理讲解（先读这一节）

这一节从零讲四个 PR 背后的概念，每个 PR 配一个小例子。后面第 1–4 节讲的是"改了什么、要审什么"，读懂这一节再看会容易很多。

### A. 基础概念

**A1. 大模型推理在 GPU 上做什么**

模型由很多"层"串起来。一个请求先把整段输入（prompt）一次性算一遍，叫 **prefill**；之后每一步只生成一个新 token，叫 **decode**。每生成一个 token，都要把所有层再算一遍。

**A2. KV cache 和 block**

每一层在处理每个 token 时会算出一组中间结果（K 和 V），后面的 token 还要用，所以要存起来，这就是 **KV cache**。vLLM 把 KV cache 切成固定大小的 **block**（比如每 block 64 个 token），按 block 分配和回收。一个 block 占多少字节，取决于"这台 GPU 上有几层"×"每层每 token 多少字节"×"每 block 几个 token"。

**A3. 四种并行（一台机器 8 张卡怎么分工）**

| 名称 | 怎么分 | 比喻 |
|---|---|---|
| **TP**（张量并行） | 每一层的矩阵切成几份，每张卡算一份 | 一道大题几个人各算一部分 |
| **PP**（流水线并行） | 把层按顺序分段，每张卡负责一段（一个 **stage**） | 流水线，A 做完 1–20 层交给 B 做 21–40 层 |
| **DP**（数据并行） | 每张卡有完整模型，各自处理不同的请求 | 几个窗口同时办不同人的业务 |
| **EP**（专家并行） | 只用于 MoE 层，把"专家"分到不同的卡上 | 见 A4 |

我们的 recipe：prefill 用 **PP4**（4 张卡各管四分之一的层），decode 用 **DP4 + EP**（4 张卡各自处理不同请求，MoE 层的专家分到 4 张卡上）。

**A4. MoE、路由专家和共享专家**

GLM-5.2 是 **MoE**（Mixture of Experts）模型：每个 MoE 层里有几百个小网络，叫**专家**。

- **路由专家**：一个小网络（router）为每个 token 挑出最合适的 k 个专家（top-k），只算这 k 个。挑选结果是一张表 `topk_ids`：每行一个 token，每列是一个被选中的专家编号。
- **共享专家**：不用挑，**每个 token 都要算**。
- 最终输出 = 各个被选中专家的输出 × 权重，再加起来。

**A5. 集合通信：all-gather 和 reduce-scatter**

几张卡之间交换数据的标准操作：

- **all-gather（AG）**：每张卡拿出自己的一份，拼起来，每张卡都得到完整的一份。
  ```
  卡0: [a]  卡1: [b]   --all-gather-->   卡0: [a b]  卡1: [a b]
  ```
- **reduce-scatter（RS）**：每张卡有一份完整长度的数据，先按位置**相加**，再切开，每张卡拿回属于自己的那段。
  ```
  卡0: [x0 y0]  卡1: [x1 y1]  --reduce-scatter-->  卡0: [x0+x1]  卡1: [y0+y1]
  ```

**DP + EP 下一个 MoE 层的做法（"AG/RS 分发"，代码里叫 naive DP/EP）**：

1. **AG**：每张卡把自己那几个 token（连同 router 算好的 `topk_ids`）发给所有卡，于是每张卡都看到全部卡的 token。
2. 每张卡只算**自己手上那些专家**负责的部分，其余位置填 0。
3. **RS**：把所有卡的部分结果相加（每个 token 的完整结果 = 各卡贡献之和），再把每个 token 的结果送回它原来所在的卡。

关键点：**RS 是相加**。所以如果同一份计算被两张卡都做了，结果就会被加两次；如果谁都没做，就是 0。

**A6. RCCL 和 AITER**

- **RCCL**：AMD 的通用集合通信库（相当于 NVIDIA 的 NCCL），什么情况都能用。
- **AITER**：AMD 为推理写的高性能算子库。它有专门为"同一台机器、小数据量"优化的 AG/RS，比 RCCL 的通用实现快；它也提供"把共享专家和路由专家合在一个 kernel 里算"的融合实现。

**A7. CUDA graph（在 ROCm 上叫 HIP graph，vLLM 里统一叫 CUDA graph）**

decode 每一步要发射几千个小 kernel，如果每次都由 Python 一个个发，CPU 会成为瓶颈。CUDA graph 的做法是：

- **捕获（capture）**：启动时，按每种 batch 大小把整个 forward 跑一遍，GPU 不真正执行，只把"发射了哪些 kernel、参数是什么"录下来。
- **重放（replay）**：之后每一步直接让 GPU 按录像执行，**不再运行 Python**。

所以：**捕获时 Python 代码做的每一个"选哪个 kernel"的判断，会被永久固化进录像**。重放时再怎么改 Python 都没用。

两种模式：**FULL**（整个 forward 一张图）和 **PIECEWISE**（切成几段，注意力部分不进图）。

**A8. 锁页内存（pinned memory）和"注册"**

GPU 要直接用 DMA 读写主机内存，那块内存必须是**锁页**的（操作系统保证它不会被换出、物理地址不变）。把一块已有的普通内存变成锁页内存的操作叫**注册**（`hipHostRegister`）。注册要花时间，并且 AMD 驱动（amdkfd）对一台机器上**所有进程注册的总量**有一个上限（大约是物理内存总量），同一块内存被 4 个进程各注册一次，就算 4 份。

**A9. KV offload（把 KV cache 放到 CPU 内存）**

GPU 显存放不下所有 KV cache，vLLM 的 **OffloadingConnector** 会把暂时不用的 KV block 拷到 CPU 内存里存着（我们的 recipe 用了 1.9 TB），以后同样的前缀再来时直接拷回 GPU，省掉重算。CPU 上的存储单位叫 **chunk**（若干个 block）。

**TieringOffloadingSpec 的共享区域**：在 PP 下，几个 worker 进程（每个 PP stage 一个）和调度器进程**共用一个 mmap 文件**作为 CPU 存储。它按"行"组织，每行存一个 chunk，一行里按 worker 编号依次放每个 worker 的**槽（slot）**：

```
          worker0 槽  worker1 槽  worker2 槽  worker3 槽
第 0 行 [ ######## | ######## | ######## | ######## ]
第 1 行 [ ######## | ######## | ######## | ######## ]
...
```

每个 worker 只读写自己那一列的槽。

---

### B. U17 的原理：DP + EP 下共享专家被重复算或漏算

**B1. AITER 是怎么把共享专家"融合"进来的**

为了少发一次 kernel，AITER 把共享专家当成"每个 token 都必选的额外专家"，直接在 `topk_ids` 后面多加几列：

```
topk_ids 一行 = [ 路由专家1, 路由专家2, ..., 路由专家k,  共享专家列 ]
```

这几列的内容是启动时预先填好在一块**全局缓冲区（meta buffer）**里的，router 返回的 `topk_ids` 直接用这块缓冲区的前若干行。

**B2. 开了 EP 以后，共享专家由谁算**

开 EP 时每张卡都存了一份共享专家的权重。如果每张卡都对每个 token 算一遍共享专家，RS 一相加就变成了 `ep_size` 倍。所以 AITER 的做法是**轮流分配**：第 `row` 行的共享专家只由 `row % ep_size == ep_rank` 的那张卡算，其余卡在共享专家列填一个"无效专家"编号（哨兵），算的时候跳过。

这个规则的隐含前提是：**每张卡看到的行号代表同一个 token**。在 TP + EP 下这是成立的（所有卡处理同一批 token）。

**B3. DP 打破了这个前提（举例）**

DP2 + EP2，卡 0 有 token A0、A1，卡 1 有 token B0、B1。

router 之后，各卡用**自己的** meta buffer 打标记（√ = 本卡要算共享专家，× = 不算）：

```
卡0（ep_rank=0，偶数行打√）:   第0行 A0 √   第1行 A1 ×
卡1（ep_rank=1，奇数行打√）:   第0行 B0 ×   第1行 B1 √
```

AG 之后，两张卡都看到同样的 4 行，**标记原样带过来**：

```
gather 后（两张卡相同）:  A0 √   A1 ×   B0 ×   B1 √
```

两张卡都按这张表算：两张卡都算了 A0 和 B1 的共享专家，都没算 A1 和 B0。RS 相加后：

| token | 共享专家被算了几次 | 应该是 |
|---|---|---|
| A0 | 2 | 1 |
| A1 | 0 | 1 |
| B0 | 0 | 1 |
| B1 | 2 | 1 |

每一层都这样错一点，几十层累积下来，模型就输出乱码。这就是 J8 里 main 的 GSM8K 0 / 200。

**B4. U17 的修法**

AG 之后，每张卡**按 gather 后的新行号，用自己的规则重新打一遍标记**：

```
卡0（偶数行）:  A0 √   A1 ×   B0 √   B1 ×
卡1（奇数行）:  A0 ×   A1 √   B0 ×   B1 √
```

现在每个 token 恰好被一张卡算一次，而且两张卡的工作量一样。实现上就是把 `topk_ids` 的最后几列用本卡 meta buffer 的对应行覆盖掉，一行代码：`topk_ids[:, -cols:] = meta_ids[:rows, -cols:]`。

**B5. 为什么只在 ROCm、只在这两个地方改**

- 只有 AITER（ROCm）有这种融合方式，CUDA 上不存在这个问题。
- 另一个后端 MoRI 发现了这个问题，直接拒绝"融合共享专家 + EP"的组合；AG/RS 后端没有检查，所以默默算错。
- monolithic `prepare` 发的是 router 的原始打分，在接收方才路由，标记本来就是按 gather 后的行号打的，不需要改。

---

### C. U19 的原理：共享区域被每个 worker 整块注册，超过驱动上限

**C1. 每个 worker 要注册什么**

共享区域里，每个 worker 只用自己那一列槽。理想情况是**只注册自己的槽**。但内存是按行连续存放的，自己的槽在每一行里都是不连续的一小段，所以只注册自己的槽意味着**每行一次注册调用**；而整块注册只需要几次大调用（每次最多 64 GiB）。

**C2. vLLM 现在的选择规则**

```
每行能省下的字节 = 行宽 - 槽宽
如果 省下的 ≥ 3 MiB:  只注册自己的槽（每行一次调用）
否则:                 整块注册（几次大调用）
```

3 MiB 的门槛是在 NVIDIA 上为了"省下的内存值得多出来的调用次数"定的。

**C3. GLM-5.2 PP4 的数字**

- 每个 worker 的槽 = 796,416 字节 ≈ 0.80 MB
- 行宽 = 4 个槽 ≈ 3.19 MB
- 只注册自己的槽，每行省 3.19 − 0.80 ≈ 2.39 MB，**小于 3 MiB**
- → 规则选择"整块注册"：4 个 worker 每个都注册整块 1.25 TB，总共 5 TB
- 而 amdkfd 对整台机器的注册上限约等于物理内存（我们的机器 2,752 GiB）→ 注册到一半被驱动拒绝，失败的 worker 退回用普通内存，拷贝变慢

**C4. U19 的修法和代价**

ROCm 上门槛改成"每行省一页（4 KB）"，于是 GLM-5.2 会走"只注册自己的槽"：4 个 worker 合计只注册 1.25 TB，4 / 4 成功。

代价是调用次数：1 TB 区域、3.2 MB 一行，约 31 万次调用，实测每个 worker 约 40 秒（比整块注册的 10 分钟还快，因为整块注册要注册 4 倍的内存）。行越小调用越多，这是 reviewer 最可能追问的地方。

---

### D. U11 的原理：PP 各 stage 的槽宽不一致

**D1. 为什么各 stage 的 block 大小不同**

PP 把层分给 4 个 stage，各 stage 分到的层数不一定相同（总层数不能被 4 整除时，或按其他规则切分时），所以每个 stage 一个 block 的字节数不同：

| stage | 每 block 字节数 |
|---|---|
| 0 | 796,416 |
| 1 | 779,520 |
| 2 | 779,520 |
| 3 | 742,656 |

**D2. 共享区域要求所有进程算出同一个布局**

C 节说过，共享区域的一行由所有 worker 的槽拼成。调度器和 4 个 worker **各自独立**算这个布局（没有互相通信），布局包括：槽宽、行宽、一共几行、文件总大小。

vLLM 原来的做法是**每个进程用自己的 block 字节数当槽宽**：

```
stage0 以为： 行宽 = 4 × 796,416，文件大小 = X0
stage3 以为： 行宽 = 4 × 742,656，文件大小 = X3（比 X0 小）
```

后果：

- 文件由一个进程创建，其他进程等文件"长到自己期望的大小"再打开。期望更大的进程永远等不到，30 秒后超时，服务起不来（我们主线 s6 遇到的就是这个）。
- 即使大小碰巧对上，各进程认为的"第 n 行第 k 个槽的位置"也不同，会读写到别人的槽。
- 不用共享区域的默认模式下，调度器按 stage 0 的大小分配 chunk 编号，block 更大的 stage 实际能放的 chunk 更少，编号超出范围就越界（#69348 在 B300 上复现过）。

**D3. 修法（U11 和 #69348 一样）**

所有进程都用**所有 stage 里最大的**那个 block 字节数当槽宽。这样大家算出的布局完全一致。代价是小 stage 的槽有一部分空着（多用 2–7% 的主机内存）。

U11 和 #69348 的区别只在"最大值从哪里拿"：U11 自己加了一个字段去算，#69348 用的是 main 上刚合入的 #67998 已经提供的每 stage 信息。所以 #69348 更简洁，U11 没有必要再提。

---

### E. U13 的原理：AITER 的快速 AG/RS 在 CUDA graph 里从没生效

**E1. 背景**

decode 用 DP4 + EP，每个 MoE 层要做一次 AG 和一次 RS（A5），每步一共 156 次集合通信。这些都是小数据量的通信，AITER 的专用 AG/RS 比 RCCL 快。vLLM 用一个判断函数 `_can_use_aiter_ag_rs()` 决定用哪个，条件是：

1. 各卡 token 数相同（`sizes is None`），AITER 的实现要求这一点；
2. 当前是 FULL CUDA graph；
3. `batch_descriptor.uniform` 为真。

**E2. 为什么从没生效**

关键在 A7：判断只在**捕获**时运行，结果被固化进录像。

- **条件 2**：以前 vLLM 捕获 FULL graph 时，把模式标成 `NONE` 来跑 forward，于是判断为假 → RCCL 被录进去。U13 原版就是为了绕过这一点（"模式是 NONE 但正在捕获，也算 FULL"）。**上游 #68006 已经把捕获时的模式改成了 FULL，这一半问题没了。**
- **条件 3**：捕获时不传 `batch_descriptor`，vLLM 自动补一个默认值，它的 `uniform` 永远是 `False`；整个 vLLM 里没有任何地方把它设成 `True`。所以即使有了 #68006，判断依然为假。**这一半还没人修。**

**E3. U13 现在要做的**

去掉条件 3。理由（也是 PR 的核心论点）：

- `uniform` 说的是"**同一张卡上**，这批请求每个的 token 数都相同"，是批内的性质；
- AITER AG/RS 需要的是"**每张卡**交出的 token 数相同"，这由条件 1（`sizes is None`）保证；在 DP + CUDA graph 下，各卡会被补齐到同一个 token 数，以便都用同一张图；
- 所以条件 3 和 AITER AG/RS 的正确性无关。

这个论点我认为成立，但我没有查到当初为什么加 `uniform`（引入它的是 #48247），上游很可能会问，讨论时需要你判断。

**E4. 收益和为什么要重测**

以前测到的是每步 decode 快 2–3%（AITER 通信每步 6.1 ms 对 RCCL 7.8–10.2 ms）。但那些数都是在 #68006 之前测的；#68006 改了捕获时的模式，可能让别的"按 FULL 判断"的代码也开始生效，main 本身的速度可能已经变了，所以 A/B 要在新 main 上重做。


---

## 1. U17：#69616

### 现状
- 11-24 09:20 UTC 开，head `d9dd5161c`，标签 `bug`、`rocm`，自动请求了 7 位 reviewer：
  tjtanaa、AndreasKaratzas、mgoin、pavanimajety、zyongye、wzhao18、bnellnm。
- 还没有人工评论或 review；CI 没跑（fork 来的 PR 需要维护者评论 `/ci run` 或 `/amd-ci run`）。
- 改动的 3 个文件和调用方在最新 main 上仍未被改过，可以直接合并。

### 需要掌握的
- **问题**：开 EP 时，AITER 融合共享专家用一块预填的 `topk_ids` 缓冲区标记"每行的共享专家由哪个 EP rank 算"（`row % ep_size == ep_rank`），这在 TP+EP（各 rank token 相同）下是对的。DP 下 naive dispatch 把各 rank 的 `topk_ids` all-gather 起来，这些标记是按**源 rank 的本地行号**打的，gather 之后：有的 token 的共享专家在所有 rank 上都算了、被 combine 加了 `ep_size` 次，有的一次都没算。MoRI 后端对这个组合直接断言拒绝，AG/RS 后端没有检查。
- **修法**：dispatch 之后，用本 rank 的 meta 缓冲区按**gather 后的行号**重写共享专家那几列。所有 rank 看到的 gather 顺序相同，所以每个 token 恰好由一个 rank 算，负载也均匀。只在 ROCm 上生效。
- **证据**：J8（main `8828834fb`，GLM-5.2 MXFP4，DP4 + EP）main 0 / 200 → 196 / 200；v0.34.0 上 1 / 200 → 196 / 200。单测 + 新测试在 main 上失败 + pre-commit。
- **可能被问到的**：
  1. 为什么不像 MoRI 一样直接拒绝？答：拒绝也能保证正确，但会失去融合，每步慢约 2.4 ms（正文已写）。
  2. 为什么只在 `prepare` 和 `prepare_routed` 加？答：monolithic `prepare` 发的是 router logits，在接收方路由，本来就对。
  3. 越界怎么办？答：meta 缓冲区按 `max_num_batched_tokens * dp_size` 行分配，正好是 gather 后的上限；超出时断言，而不是静默算错。
  4. 测试用了模拟（不跑多卡），会不会被要求加 e2e？答：多卡 DP + EP 服务 CI 跑不了，正文已说明，并给了 e2e 结果。

### 要 review 的（源码 +46 / −2）
```diff
--- vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
 aiter_topK_meta_data: tuple[torch.Tensor, torch.Tensor] | None = None
+# Trailing shared-expert columns of the meta buffers that mark, under EP, which
+# rank computes each row's shared experts (0 without EP).
+aiter_topK_ep_shared_cols: int = 0
 ...
-    global aiter_topK_meta_data
+    global aiter_topK_meta_data, aiter_topK_ep_shared_cols
 ...
     aiter_topK_meta_data = (total_topk_weights, total_topk_ids)
+    aiter_topK_ep_shared_cols = n_shared_experts + 1 if is_EP else 0
+
+
+def localize_fused_shared_expert_ids(topk_ids: torch.Tensor) -> torch.Tensor:
+    """(docstring 略)"""
+    cols = aiter_topK_ep_shared_cols
+    if aiter_topK_meta_data is None or cols == 0:
+        return topk_ids
+    meta_ids = aiter_topK_meta_data[1]
+    if topk_ids.dim() != 2 or topk_ids.shape[1] != meta_ids.shape[1]:
+        return topk_ids
+    rows = topk_ids.shape[0]
+    assert rows <= meta_ids.shape[0], (...)
+    topk_ids[:, -cols:] = meta_ids[:rows, -cols:]
+    return topk_ids

--- vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
+def _localize_shared_expert_ids(topk_ids: torch.Tensor) -> torch.Tensor:
+    if not current_platform.is_rocm():
+        return topk_ids
+    from ...experts.rocm_aiter_moe import localize_fused_shared_expert_ids
+    return localize_fused_shared_expert_ids(topk_ids)
 (Modular.prepare 返回前、Monolithic.prepare_routed 两个返回分支各调用一次)
```
看的时候建议重点核对：
- [ ] `cols = n_shared_experts + 1` 与 `init_aiter_topK_meta_data` 里 `n_shared_experts + is_EP` 的列数一致（多出的 1 列是 EP 下的哨兵）。
- [ ] `topk_ids.shape[1] != meta_ids.shape[1]` 时直接返回：这是"没用融合共享专家的 `topk_ids`"的判断，是否足够可靠。
- [ ] 就地写 `topk_ids`：前提是 dispatch 后的 `topk_ids` 不是 meta 缓冲区的视图，否则会改坏所有层共用的 meta 缓冲区。我读代码核对过（11-25）：
  - `CudaCommunicator.all_gatherv` 的三条路径（AITER `custom_all_gather`、对称内存、`torch.empty` + pynccl）返回的都是新张量或通信库自己的缓冲区；
  - naive DP/EP 的 prepare 只在 DP > 1（AG/RS 后端，或"DP 但没开 EP"的回退）时被选中；
  - "DP 但没开 EP"时 `is_EP=False`、`cols = 0`，函数直接返回。

  剩下没核对的一点：如果 AITER `custom_all_gather` 返回的是它跨调用复用的内部缓冲区，就地写入也只影响本次已 gather 的数据，在下一次 gather 前已被消费，我认为没问题，但没有看 AITER 源码。
- [ ] 正文里的数字和你的原始记录一致：GSM8K 0 / 200 → 196 / 200，单测 108 / 106、177、10、47，MTP acceptance 3.555。

---

## 2. U19：#69617

### 现状
- 11-24 09:21 UTC 开，head `8bf00c65f`，标签 `rocm`、`nixl`（`nixl` 是自动打的，不太对，可以不管），自动请求了 ApostaC、orozery、Etelis。
- 没有人工评论，CI 没跑。文件在最新 main 上没被改过，可直接合并。

### 需要掌握的
- **问题**：多个 worker 共用一块 CPU offload 区域，每行由各 worker 的槽拼成。`pin_mmap_region` 只有在"只注册自己的槽每行能省 ≥ 3 MiB"时才这么做。GLM-5.2 PP4 每行 3.19 MB、槽 0.80 MB，只省 2.39 MB，于是 4 个 worker 各注册整块区域，同一块内存被注册 4 次。ROCm 上驱动在中途拒绝，部分 stage 退成可分页内存。
- **修法**：只在 ROCm 上把门槛降到"每行省一页"。
- **证据**：J1b 机制测量（v0.34.0 镜像，选择规则与 main 相同），1 TB 区域、4 个 worker 同时注册：现规则 2 / 4 失败、成功的 608 / 616 s；新规则 4 / 4、38.5–41.2 s。单测 814 / 812，新测试在 main 上失败。**没有进过 P4D4 端到端**（J3 / J3b 走的是 U26 的私有锁页路径）。
- **最可能被问的：代价没有上界。** 每行一次注册调用，约 8,000 次 / s / worker。行越小调用越多：0.5 MB 行的 1 TB 区域约 200 万次、约 4 分钟。正文提了备选：按"worker 数 × 区域大小"决定规则。讨论时要想好是否主动加一个上界（例如只在 `world_size × 区域大小` 超过某值时才切换），还是等 reviewer 提。
- **与 #69254 的关系**（正文已写）：#69254（@yuzhouo7，open，有 1 条评论、1 个 review）在 ROCm 上启动时按 amdkfd 系统内存上限检查注册总量。它在 `cpu/spec.py` 里**照抄了** `pin_mmap_region` 的选择规则来估算。两者都合入而它不跟着改的话，会把我们这种行按"整块 × worker 数"计。按它的公式估算，我们的主机上 1.25 TB 的 Tiering 区域会在启动时被拒；这是公式估算，没有实测。
- **#52296**（draft，08 月起没进展）：跨 worker 协调注册、失败时统一回滚。与 U19 互补。

### 要 review 的（源码 +7 / −1）
```diff
--- vllm/v1/kv_offload/cpu/gpu_worker.py
     stride, slot, page = region._row_stride, region.cpu_page_size, region.page_size
-    min_savings = max(page, MIN_HOST_REGISTER_SAVED_BYTES)
+    # ROCm: every worker of a shared region registering all of it can exceed the
+    # driver's limit on registered memory (4 PP workers x 1.25 TB), while
+    # registering only the worker's slots, one call per row, is affordable (a
+    # 1 TB region of 3.2 MB rows: ~40 s per worker on MI355X).
+    min_savings = (
+        page if current_platform.is_rocm() else max(page, MIN_HOST_REGISTER_SAVED_BYTES)
+    )
```
测试：原有的 `test_pin_mmap_region_selects_by_savings_not_row_count` 补了 `is_rocm → False`；新增 `test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[2/4]`。

看的时候建议重点核对：
- [ ] 注释里"driver's limit"现在可以更具体：#69254 指出那是 amdkfd 的全节点系统内存上限。要不要改注释、引用它。
- [ ] 正文里的数字：J1b 的 2 / 4、687.18 GB、608.0 / 616.2 s、313,805 次、38.5–41.2 s；主线 P4D4 里 rank 0 / 2 在 549.74 / 755.90 GB 失败。
- [ ] 正文写"On ROCm main still pins synchronously (`async_init` is CUDA-only)"，这是 11-19 的判断，我这次没有复核。

---

## 3. U11：fork #4（未上游）

### 现状
- 上游 [#69348](https://github.com/vllm-project/vllm/pull/69348)（@xsxszab，11-23 开，**draft**，0 评论 0 review，11-23 之后没更新）与 U11 是同一修法。
- 它依赖已合入 main 的 #67998：`KVCacheConfig.pipeline_stages`（每个 stage 的每块字节数）。
- U11 与最新 main 在 `vllm/v1/kv_cache_interface.py` 冲突（两边在同一处加字段）。
- #50653（@StellaH-ter，统一 CPU chunk 数，不统一槽宽）仍 open，10-29 之后没动；#69348 把她列为 co-author。

### 需要掌握的：两者的差别
| | U11 | #69348 |
|---|---|---|
| 每 stage 字节数从哪来 | 自己在 `get_kv_cache_configs` 里算最大值，新字段 `max_worker_bytes_per_block` | 用 #67998 已有的 `pipeline_stages` |
| `build_offloading_config` 的改法 | 取本 worker 与最大值中的大者，打 info 日志 | 直接取各 stage 的最大值 |
| 源码行数 | +38（3 个文件） | +6（1 个文件）+ 注释 |
| 验证 | MI355X，GLM-5.2 PP4：不修时 Tiering 第 4 个 stage 30 s 超时，修后 9 / 9，P4D4 上 4 个 PP worker 加入同一块 1,250 GB 区域 | 2 × B300，Qwen3-0.6B PP2，default 和 Tiering 两种 spec，都给了 main 失败 / PR 通过的日志 |

U11 能做的而 #69348 没做的：**没有**。#69348 更小，且建立在 main 的新接口上。

### 要 review 的
- #69348 的源码只有一处（6 行），在 `kv_connector/v1/offloading/config.py`：
  ```python
  worker_kv_bytes_per_block = max(
      (stage.bytes_per_block for stage in kv_cache_config.pipeline_stages),
      default=worker_kv_bytes_per_block,
  )
  ```
- 它正文的 Limitation 一节：多 KV group 的模型在 PP 下需要每个 stage 最大 group 的大小，`pipeline_stages` 没有这个信息（GLM-5.2 是 MLA，单 group，不受影响）。

### 讨论时要定的
1. 关闭 fork #4？（关闭只影响你的 fork，可以随时重开）
2. 是否在 #69348 留言：提供 ROCm + GLM-5.2 PP4 的实例（各 stage 每块 796,416 / 779,520 / 779,520 / 742,656 字节、Tiering 超时的日志），并提出在 MI355X 上帮测。它是 draft、没人 review，我们的数据可能帮它推进。**这需要在 MI355X 上把 #69348 的分支跑一遍**（聚合 PP4 + Tiering 的门禁，约 2 卡 30 min），否则只能说"我们的修法相同、有效"。
3. 等它合入后，P4D4 是否改用它、去掉 U11 的回移副本。

---

## 4. U13：fork #1（未上游）

### 现状
- 上游 #68006（11-20 合入）让 Model Runner V2 捕获 FULL graph 时直接以 `CUDAGraphMode.FULL` 跑 forward。U13 原来"`NONE` 且正在捕获就当 FULL"的分支因此多余。
- 但 `_can_use_aiter_ag_rs()` 仍要求 `batch_descriptor.uniform`，而 `vllm/` 下没有任何地方把它设为 `True`（捕获时 runner 不传 descriptor，`set_forward_context` 补的默认值 `uniform=False`）。**所以 main 上 AITER AG/RS 的快速路径至今仍不生效**，去掉这个要求的那一半还需要。
- 改写补丁 `U13-rework-on-afb35f85e.patch` 在最新 main `69d3af837` 上仍能直接应用。11-24 有一个提交（#53377，MNNVL allreduce）改了同一个文件，不涉及 AITER。
- 按关键字没有找到上游做同一件事的 PR（11-23 查的，讨论前我可以再查一次）。

### 需要掌握的
- **原有证据全部早于 #68006**：J8 decode A/B −2.0% 至 −3.1%（main `8828834fb`），J3b 日志确认生效（`f0c592bfd`）。#68006 之后，其他"按 FULL 判断"的代码在捕获期也开始生效了，main 一侧的基线变了，**原数字不能用于新 PR**。
- **`uniform` 要求原本想保护什么？** 我的判断是：`sizes is None` 已经表示组内各 rank 的 token 数相同，这正是 AITER 自定义 AG/RS 需要的条件；`uniform`（每个请求 token 数相同）是"批内"的性质，与跨 rank 的 AG/RS 无关。**这个判断是 U13 的核心论点，上游一定会问，请你想清楚是否同意。** 可以去看 #48247（引入 AITER AG/RS 的 PR）里加 `uniform` 的讨论，我没有查过。
- **CUDA 不受影响**：`use_aiter_ag_rs` 只在 ROCm + AITER 自定义 allreduce 时为真。

### 要 review 的（改写补丁，源码 +11 / −4）
```diff
--- vllm/distributed/device_communicators/cuda_communicator.py
         Requires:
-        - uniform batches
-        - FULL CUDAgraphs
+        - the same token count on every rank (sizes is None)
+        - a FULL CUDA graph
         """
 ...
+        # The GPU model runner captures a FULL graph without a batch descriptor,
+        # so the context's default one is never marked uniform. sizes is None
+        # already means every rank of the group has the same token count.
         if ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL:
             return False
-        bd = ctx.batch_descriptor
-        return bd is not None and bd.uniform
+        logger.info_once("Using AITER custom AG/RS for group %s.", self.unique_name)
+        return True
```
测试 `test_aiter_ag_rs_gate`：5 例（FULL → True；NONE、PIECEWISE、`sizes` 不等、无 forward context → False），forward context 用真实的默认 `BatchDescriptor(num_tokens=8)`。**只做过语法检查，没跑过。**

看的时候建议重点核对：
- [ ] 上面关于 `uniform` 的论点。
- [ ] FULL **重放**时也会走这个判断吗？重放不跑 Python，所以判断只在捕获时起作用，捕获时的选择被固化进 graph。这一点和原版 U13 一样。
- [ ] `info_once` 日志放在判断函数里，每次调用都会经过（`once` 保证只打一次），开销可以接受吗。

### 讨论时要定的：重测方案
需要在 MI355X 上重做，基于当时最新的 main + 对应 ROCm nightly：
1. 单测：`tests/distributed/test_comm_ops.py -k aiter`，以及新测试在 main 上失败（2 卡非独占，约 30 min，可与下一项同一作业）。
2. pre-commit。
3. **decode A/B**（main vs main + U13，两组 4 卡服务同时跑，同 J8 的做法，8 卡独占约 1 h）：每步时间、日志里 `Using AITER custom AG/RS` 只出现在 U13 一侧、GSM8K 门禁。
4. 要不要在 P4D4 上再跑一次（J3b 的做法），还是只做 A/B。

---

## 5. 明天讨论的议程建议
1. U17 / U19：你 review 后的问题；U19 要不要先加上界；要不要去 #69254 留言。
2. U11：关 fork #4；在 #69348 留言（以及要不要先在 MI355X 上跑它）。
3. U13：`uniform` 论点；用不用改写版；重测方案和集群资源。
