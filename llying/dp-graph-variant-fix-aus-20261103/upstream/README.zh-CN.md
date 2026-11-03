# 修法对到 SGLang main（11-03 17:40，未提 PR）

- 基线：SGLang main `1ea5f8c6`（11-03 17:11）。本地分支 `fix/dsa-graph-variant-dp-consistency`（`/perf_apps/liyingli/bench_agentx/upstream/sglang-main`），
  提交 `7ae1a44`，补丁 `0001-…patch`（`git am` 可用）。
- `fix.diff`（v0522，6 个文件）在 main 上全部 hunk 直接打上（偏移 / fuzz 1）；在此基础上：
  - 抽出 `graph_variants.dp_decode_max_seq_len()`：只在 DECODE / IDLE 模式下读 `dp_decode_graph_max_seq_len`；
  - `SeqLenBucketGraphVariants`（aiter decode 分区桶）与 `Dsv41CandidateGraphVariants` 同样优先用这个上界（交接第 4 项要求）。
    DSv4.1 的 TARGET_VERIFY（DSpark）不用：它在回放前按本 rank 的已提交长度预先构建该 variant 的元数据（`_early_variant_for`），回放时必须选同一个；
  - `ForwardBatch.init_new` 直接读 `batch.dp_decode_graph_max_seq_len`（与相邻的 `dp_prefill_cuda_graph_max_prefix_len` 一致）。
- 在 main 上核对过的前提：`dsa_indexer._should_skip_logits_computation()` 的 dense 只在 decode / idle 生效，TARGET_VERIFY 两套 graph 执行相同 kernel；
  `MLPSyncBatchInfo` 的列顺序与 v0522 相同（新列是第 11 列），all-gather 与 all-reduce 两条路径下各 rank 的行都单独保留，取最大值即可。
- 单元测试 `test/registered/unit/layers/test_graph_variants_dp_consistency.py`（`register_cpu_ci`，11 项）：在 `infera-sglang:v0522-fdbafb60` 镜像里以
  `PYTHONPATH=<main>/python` 跑 main 的代码，11/11 通过（aiter 导入时要看到一张 GPU）。格式：ruff format 0.15.1、isort、ruff check（F401 等）通过。
- 未做：main 上的端到端验证（需要按 main 构建的 ROCm 镜像）；PR 说明里应引用 AUS 上 v0522 的验证结果（`../RESULT.zh-CN.md` 第 1 节）。
  **向上游提 PR 需用户确认**（对外操作）。
