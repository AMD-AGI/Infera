# PR U17：DP + EP + AITER 融合共享专家 + AG/RS 分发时，每个 token 的共享专家只算一次

- 目标：vllm-project/vllm main。分支（仓库外 `/home/liyingli/bench_agentx/upstream-20261119/vllm`）：`u17-aiter-fused-shared-ep-main`
  （基于 main `8828834fb`，一个提交 `fa0bdbf97`，11-20 写），工作树 `wt/u17-aiter-fused-shared-ep-main`。还没推到 fork，没开 PR。
- 只影响 ROCm：`naive_dp_ep` 里的调用有 `current_platform.is_rocm()` 判断；`localize_fused_shared_expert_ids` 在没有 EP、没有融合共享专家、
  或分发的 `topk_ids` 只含路由列时什么都不做。不需要 CUDA 测试。
- 查重（11-20）：上游 open PR / issue 里没有这个问题。相关的：[#61941](https://github.com/vllm-project/vllm/pull/61941)（GLM-5.x MXFP4 默认开
  AITER 融合共享专家，会让更多部署走到这条路径）、[#63800](https://github.com/vllm-project/vllm/pull/63800)（EP 下 AITER MXFP4 MoE 只走本地专家）。
- main 上仍有这个问题（11-20 读代码确认）：`init_aiter_topK_meta_data(is_EP=True)` 仍按行号 `range(tp_rank, max_num_tokens, tp_size)` 标共享专家；
  路由（`AiterSharedRoutedFusedMoERouter`、`rocm_aiter_grouped_topk`）返回 meta 缓冲区前几行的视图（含共享列）；`naive_dp_ep.prepare` 把它原样交给
  `get_ep_group().dispatch` all-gather；MoRI 后端对这个组合直接断言拒绝（`config.py`），AG/RS 后端没有检查。
- 状态：代码 + 单测写完、已提交；**单测、pre-commit、main 上的端到端门禁都还没跑**（单测和门禁要 GPU，等用户安排；pre-commit 只要 CPU）。
  端到端证据目前只有 A 线在 v0.34.0 上的（同样的逻辑，A 线覆盖层 `overlay-ep/fused-shared-expert-ep.diff`）。

## 改动（3 个文件，+88 / −2）

- `rocm_aiter_moe.py`：`init_aiter_topK_meta_data` 记下 EP 时的共享列数（`n_shared_experts + 1`，含哨兵列）到 `aiter_topK_ep_shared_cols`；
  新增 `localize_fused_shared_expert_ids(topk_ids)`：用本 rank 的 meta 缓冲区、按 gather 后的行号覆盖共享列；行数超过 meta 缓冲区时断言
  （缓冲区本来就按 `max_num_batched_tokens × dp_size` 分配，正是为 gather 后的行数）。
- `naive_dp_ep.py`：modular 的 `prepare` 与 monolithic 的 `prepare_routed` 在 dispatch 之后调用它（只在 ROCm）。monolithic 的 `prepare` 分发的是
  router logits，路由在接收方做，标记本来就对，不用改。
- 与 A 线覆盖层的区别：A 线只改了 modular `prepare`、用两个全局量；这里收成一个函数，加了 `prepare_routed`，并且行数越界时断言而不是静默跳过。

## 验证数据

| 项 | 结果 | 出处 |
|---|---|---|
| 单测（`tests/kernels/moe/test_rocm_aiter_topk.py::test_fused_shared_expert_rows_owned_once_after_ep_dispatch[2/4]`，ROCm + AITER + GPU） | 待跑（要 GPU）。测试模拟 2 / 4 个 rank 的 all-gather 结果：不修时每个 token 的共享专家被 0 个或 ep_size 个 rank 算，修后正好 1 个；在 main 的代码上会失败（没有这个函数） | — |
| pre-commit | 待跑（CPU） | — |
| 聚合 GLM-5.2 MXFP4，DP4 + `--enable-expert-parallel` + 融合共享专家（默认），v0.34.0，不修 | GSM8K **1/200**，MTP 接受长度 1.03，输出全部 runaway | A 线 README 3.2（r2，v3-240，单次） |
| 同上，关掉融合（`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0`） | 198/200，步时 54.49 ms | A 线 r4 |
| 同上，融合 + A 线修复 | 196/200、16K 长上下文 32/32 + 31/32，步时 53.40 ms | A 线 r9 |
| 融合 + 修复 + gfx950 gluon logits + AITER AG/RS（U13） | 197/200、32/32 + 32/32，步时 50.82 ms | A 线 r9 |
| 对照：默认非 EP（融合） | 52.9 ms；本负载下 EP 不比非 EP 快，所以修复的价值是正确性 | A 线 3.2 |
| main 上的端到端门禁 | 待跑（要 4 卡）：DP4 + EP + 融合，GSM8K 200 + 长上下文，修前 / 修后各一次 | — |

## PR description (English, vLLM template)

**Title:** [ROCm][Bugfix][MoE] Keep fused shared experts computed once per token under DP + EP with AG/RS dispatch

<!-- markdownlint-disable -->

## Overview

With AITER fused shared experts and `--enable-expert-parallel` under data parallelism, the naive DP/EP (AG/RS) dispatch
all-gathers the combined `topk_ids`, including the shared-expert columns that mark which EP rank computes each token's
shared experts. Those marks refer to source-local row indices, so after the gather a token's shared experts run on
every rank or on none. GLM-5.2 MXFP4 at DP4 + EP on MI355X answers 1/200 GSM8K questions correctly. This PR re-marks
the gathered rows for the local rank after the dispatch.

## Claims

- DP + EP + AITER fused shared experts + the AG/RS dispatch produces correct output again (GSM8K 1/200 -> 196/200 on
  GLM-5.2 MXFP4, DP4 + EP, MI355X).
- No change without EP, without fused shared experts, with other all2all backends, or on CUDA.

## Validation

<TBD: unit test on MI355X (main + this PR, and the new test failing on main), pre-commit, end-to-end GSM8K on main>

End to end on vLLM v0.34.0 with the same fix as a patch (GLM-5.2-MXFP4, DP4 TP1, MTP3, fp8 KV, one 4-GPU server,
GSM8K 200 + 32 needle / 32 long-prefix GSM8K prompts at 16K tokens, single runs):

| | GSM8K | 16K needle / long-prefix GSM8K | MTP acceptance |
|---|---|---|---|
| EP + fused shared experts (current) | 1/200 (all outputs run away) | — | 1.03 |
| EP, fusion off (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0`) | 198/200 | — | — |
| EP + fused shared experts + this fix | 196/200 | 32/32, 31/32 | — |
| no EP (fused, default) | 195-199/200 | 32/32, 31-32/32 | 3.44-3.56 |

## Details

- **Root cause.** `init_aiter_topK_meta_data(is_EP=True)` fills the shared-expert columns of the meta buffer so that only
  rows with `row % ep_size == ep_rank` carry the real shared-expert ids and the rest a masked sentinel; with identical
  tokens on every rank (TP + EP) that computes each token's shared experts exactly once. The routers return views of
  that buffer, and under DP the naive dispatch all-gathers them: each gathered row keeps its source rank's mark for its
  source-local index, so a token whose source-local index matches its source rank is computed on all `ep_size` ranks
  (and summed `ep_size` times by the combine), and every other token on none. MoRI already rejects fused shared experts
  for this reason (`config.py`); the AG/RS path had no such check.
- **Fix.** `localize_fused_shared_expert_ids()` overwrites the shared-expert columns of the dispatched `topk_ids` with
  this rank's marks for the gathered rows. Every rank sees the same gathered order, so each token is assigned to exactly
  one rank (and the work stays balanced, one in `ep_size` rows per rank). It runs after the dispatch in
  `MoEPrepareAndFinalizeNaiveDPEPModular.prepare` and `MoEPrepareAndFinalizeNaiveDPEPMonolithic.prepare_routed`, on ROCm
  only. The monolithic `prepare` dispatches router logits and routes on the receiving rank, so it is already correct.
  The meta buffer is sized `max_num_batched_tokens * dp_size` rows for exactly this gathered size; a larger dispatch
  asserts instead of silently keeping the wrong marks.
- **Alternative.** Reject fused shared experts with the AG/RS backend under EP, as MoRI does. That keeps the output
  correct but loses the fusion (2.4 ms per decode step on this model).
- **CI.** The new case lives in `tests/kernels/moe/test_rocm_aiter_topk.py` (ROCm + AITER) and checks the marking
  directly, by building the gathered ids of 2 and 4 source ranks; the end-to-end check needs a multi-GPU DP + EP server.
- **AI assistance** was used to write this change, its test and this description; the submitter reviewed every changed
  line and the measurements.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
