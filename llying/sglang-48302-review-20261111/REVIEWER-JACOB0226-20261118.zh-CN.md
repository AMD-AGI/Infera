# sglang#48302 第一条人工 review（Jacob0226，11-17）的评估

- PR：[sgl-project/sglang#48302](https://github.com/sgl-project/sglang/pull/48302)；意见原文：[discussion_r4595135900](https://github.com/sgl-project/sglang/pull/48302#discussion_r4595135900)
- 审阅对象：PR head `4dae7a44`。核对用的是本地 `/perf_apps/liyingli/bench_agentx/upstream/sglang-main`（就在这个 head 上）
- 背景：这是 PR 收到的第一条人工 review，只有一条行内评论，针对 `graph_variants.py` 里 `Dsv41CandidateGraphVariants.select()` 新加的 5 行。同目录 `REVIEW-LIST.zh-CN.md` 的 R3、R4 已经预判过这个问题。
- **结论**：他的三点事实判断都成立。建议采纳他的方案一：从本 PR 删掉这段，DSV4.1 留给后续 PR。
- **更新（11-18）**：已按方案一改好，并在最新 nightly 镜像上做了 main 与 PR 的端到端 A/B，见 `AB-MAIN-VS-PR-20261118.zh-CN.md`。还没有 push。

## 一、意见原文与要点

> This hunk changes NVIDIA SM100 decode selection under DP attention, which the `[AMD]` title and the test plan don't cover. It also half-applies in a DSpark verify runner: IDLE ranks now take the bound while TARGET_VERIFY ranks keep local lengths, and the early-staging reason doesn't hold there, since `_wants_early_verify_metadata()` requires `not attn_dp_enabled`. Suggest dropping this hunk and doing DSV4.1 (decode + verify, bound + `verify_extra_tokens`) in a follow-up with a B200 run from the owners; or keep it, retitle without `[AMD]`, and add that run here.

1. 这段改动改变了 NVIDIA SM100 在 DP attention 下的 decode 选图行为，但标题是 `[AMD]`，测试计划也没有覆盖。
2. 在 DSpark 的 target-verify runner 里，这段改动只生效了一半：空闲（IDLE）rank 按全局上界选图，正在做 verify（TARGET_VERIFY）的 rank 仍按本地长度选。
3. 代码注释给出的理由（"verify 的元数据已经按本地长度提前准备好"）在 DP attention 下不成立。
4. 给了两条路：一是删掉这段，后续单独做 DSV4.1（decode 和 verify 一起做，用上界加 `verify_extra_tokens`），并请 owner 在 B200 上跑；二是保留，但去掉 `[AMD]`，并在本 PR 里补 B200 的结果。

## 二、逐条核实

| 论点 | 结论 | 依据 |
|---|---|---|
| 1. 改了 NVIDIA 行为，没有测试 | 成立，而且范围比他说的大 | `create_dsv41_candidate_graph_variants` 在 HIP 上以 `is_gfx95_supported()` 为条件启用，所以 MI355X 上的 DSV4.1 也会走这段。GLM-5.2 在 decode 时由 `DsaGraphVariants` 先返回，在 verify 时又不满足 `model_type == "deepseek_v41"`，因此 PR 的所有验证（单测除外）都没有跑到这段代码，**在任何硬件上都没有端到端测过** |
| 2. DSpark verify 只生效一半 | 成立 | 空闲 rank 通过 `dspark_verify.py` 的 `run_idle_participation()` 参与 verify；非 ragged 时，`dflash_info.py` 的 `prepare_for_verify()`（第 94–98 行）保留 `ForwardMode.IDLE`，于是 `dp_decode_max_seq_len()` 返回上界。忙碌 rank 是 `TARGET_VERIFY`，返回 `None`，按本地长度选。两边规则不同，跨 rank 一致这个前提没有建立 |
| 3. 提前准备的理由不成立 | 成立 | `decode_cuda_graph_runner.py` 的 `_wants_early_verify_metadata()`（第 1926–1939 行）要求 `not get_parallel().attn_dp_enabled`。上界只在 DP attention 下才存在，而这时恰好不会提前准备，所以注释和 PR 描述里的理由在唯一有意义的场景里都是错的 |

补充两点：

- **不是回退，但也没修好。** 和 main 相比，空闲 rank 原来回退到 `candidate_filtered`，现在按上界选；两种都可能和忙碌 rank 不一致。问题在于它没解决问题，还加了一条错误的注释。
- **reviewer 没提到的第二条不一致来源。** 走 ragged 路径时，空闲 verify 会把 forward mode 改成 `TARGET_VERIFY`、`seq_lens_cpu` 全设为 1，按本地规则会选上限最小的 variant。所以 verify 必须和 decode 一起设计，这正好支持他的方案一。

**更正**：`REVIEW-LIST.zh-CN.md` 的 R3 写过"DSpark 的 verify 元数据也是按本 rank 的已提交长度提前准备的"。这个前提在 DP attention 下不成立，PR 的代码注释和描述（"A DSV4.1 DSpark target verify keeps its own lengths, since its metadata is staged for that variant before the replay"）都沿用了这个说法，需要一起改。

## 三、是否采纳：选方案一（删掉这段）

- **对本 PR 的修复没有损失。** GLM-5.2 和 DeepSeek-V3.2 在 HIP 上用的是 `DsaGraphVariants` 和 aiter 的 `SeqLenBucketGraphVariants`，从来不会进 `Dsv41CandidateGraphVariants`，已经验证过的修复不受影响，也不需要重跑端到端验证。
- **方案二基本做不到。** 我们没有 B200。而且要做对还得把 verify 一起改（见第五节），范围明显扩大，合入会被拖慢，还要额外请 DSV4.1 owner 签字。
- **删掉以后 `[AMD]` 标题名副其实。** 剩下的选图行为变化只在 HIP（`DsaGraphVariants`）和 aiter（分区桶）上生效。MLP sync 新加的那一列在所有平台上都会传，但只有这两个类会读。
- **零风险。** DSV4.1 的行为和 main 完全一致。

## 四、怎么改

1. **`python/sglang/srt/layers/attention/graph_variants.py`**：删掉 `Dsv41CandidateGraphVariants.select()` 开头新加的 5 行，恢复成 main 的写法：

   ```diff
        def select(self, forward_batch: ForwardBatch) -> str:
   -        # Decode only: a DSpark target verify stages its metadata for the variant of
   -        # its own committed lengths (select_for_max_seq_len) before the replay.
   -        dp_max_seq_len = dp_decode_max_seq_len(forward_batch)
   -        if dp_max_seq_len is not None:
   -            return self.select_for_max_seq_len(dp_max_seq_len)
            lengths = getattr(forward_batch, "seq_lens_cpu", None)
   ```

2. **`test/registered/unit/layers/test_graph_variants_dp_consistency.py`**：删掉 `test_dsv41_candidates_follow_the_bound_in_decode_only`（第 125–152 行），测试数从 13 改成 12。不建议改成"断言 DSV4.1 不读上界"，那等于把一个已知的缺口固化成预期行为。
3. **PR 描述**：
   - 删掉 "NVIDIA: under DP attention, `Dsv41CandidateGraphVariants` on SM100+ also selects decode variants from the global bound ..." 这一条。
   - 把 "Covered" 那一条里关于 DSV4.1 的部分改成：DSV4.1 的 candidate variants（NVIDIA SM100+ 和 gfx95 都有）不在本 PR 范围内，DP attention 下仍按每个 rank 自己的长度选，留给后续 PR 同时处理 decode 和 DSpark verify。
   - 删掉"因为提前准备了元数据，所以 verify 保留本地长度"这个理由。
   - "Unit tests: ... (13 tests, CPU)" 改成 12。
4. **验证**：只需要在 v0.5.23 ROCm 镜像里重跑这个单测文件（做法同 `PROGRESS-20261111.zh-CN.md` 2.1）。删掉的代码在 GLM-5.2 上本来就跑不到，端到端结果不变。
5. **回复 reviewer**（英文草稿，确认后再发）：

   > Thanks, agreed on all three points. `_wants_early_verify_metadata()` excludes `attn_dp_enabled`, so under DP attention (the only case with a bound) a DSpark verify never stages early and the comment's reason doesn't hold; an idle rank in the verify runner stays `IDLE` and would take the bound while busy ranks select from their own lengths. I've dropped the `Dsv41CandidateGraphVariants` hunk and its test, so DSV4.1 selection is unchanged from `main`, and removed the NVIDIA note from the description. The same class is also used on gfx95 (MI355X), so the follow-up should cover DSV4.1 decode + DSpark verify (bound + `verify_extra_tokens`, including the ragged idle-verify path) on both B200 and MI355X; I'll open an issue to track it. GLM-5.2 / DeepSeek-V3.2 never reach this class (`DsaGraphVariants` takes the decode capture), so the validated fix is unaffected.

## 五、后续 PR 的要点（可以先开 issue 跟踪）

- DP attention 下，decode、IDLE 和 TARGET_VERIFY 统一按全局上界选 variant，`select_for_max_seq_len()` 照常加上 `verify_extra_tokens`。`dp_decode_max_seq_len()` 目前只在 decode/idle 返回上界，需要为 verify 放开（或给 DSV4.1 单独一个入口）。
- 上界的口径要对齐：`_local_decode_graph_max_seq_len()` 的值要不小于 verify 侧 `candidate_request_length_bound(reqs, verify_num_draft_tokens)` 算出的值。
- 覆盖 ragged 空闲 verify 路径（forward mode 是 `TARGET_VERIFY`，`seq_lens_cpu` 全是 1）。
- NVIDIA megakernel 的 verify/draft variants 是同类问题，可以一起评估。
- 在 B200 和 MI355X 上都要验证。
