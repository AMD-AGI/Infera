# 讨论前材料：U17、U19、U11、U13

截至 2026-11-25 07:00 UTC。vLLM main 现在是 `69d3af837`（11-25），比 PR 基线 `8828834fb` 多约 390 个提交。本次只读代码和 GitHub 状态，没有跑测试。

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
