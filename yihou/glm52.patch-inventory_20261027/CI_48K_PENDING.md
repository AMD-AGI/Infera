# Pending handoff for the next user session

Updated: **2026-11-24T15:09:11.189972+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Historical/unrefreshed and outside active scope. Base passed 27 functional jobs. AMD shard 6 exited 255 after starting tests. Historical log retrieval returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: Retain for explicit owner resumption. External author eaglewangao-us and maintainers own log recovery, AMD coverage and source-equivalence review. Preserve the frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Historical/unrefreshed and outside active scope. Base had 36 functional successes and two shards cancelled before tests. One annotation identifies github-actions[bot]. The finish log confirms cancellation caused the aggregate failure. Cancellation policy and integration remain open. Scoped MI355X validation applies to 9763ef56.

Tracking key: `pr46918-external-ci-integration`.

Next action: Retain for explicit owner resumption. External author yu-depend and maintainers own original-PR conflict, coverage and review. Keep the frozen workaround and 9763ef56 evidence alongside the separately owned continuation #49732.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base remains successful on e95ec5ab. Aggregate review, #45307 overlap, exact source/run archival and independent numerical qualification remain open. The path is optional and disabled in frozen llr37. Historical FlashInfer mismatch causality is unproven.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review and coordinate #45307 overlap. Preserve passing e95ec5ab. Owner archives exact source/run and independent numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Base and AMD pass for 9602092a. Matching inherited AMD logs cover 45 sparse-MLA and 13 disaggregation cases. Exact-head approval is present and aggregate review remains required. Exact-source, image and raw archival and independent numerical/layout/replay coverage remain open. Earlier OOM and cancellation are historical. GitHub reports the current head is mergeable.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve passing 9602092a. Maintainers assess repository review requirements. Owner retrieves the existing 8d72d2ea-to-9602092a source, image and raw evidence. Retain BF16 partials, seq15/non-target numerical, scaled-KV, DCP/LSE and replay qualifications. See cycles/20261124T150002Z/PR47278_GPU_VALIDATION.json.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)

## #47897: Integrated 5a06f9e7 preserves the three reviewed PR file blobs and both production caller blobs. Exact run/source/image/raw archival, production Gluon engagement and long-context numerical coverage remain open. Historical isolated speedups and short-context GSM8K retain their measured scope.

Tracking key: `pr47897-source-long-context`.

Next action: Owner and kernel reviewer archive exact-source raw evidence and qualify current-head Gluon engagement, unequal paired-row tables and lengths, common-prefix/tails, fallback and long-context accuracy. Use approved GPU allocation for concrete gaps. Keep frozen whole-stack performance separate.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)

## #47872: Historical/unrefreshed and owner-paused pending redesign. Historical passing CI and source/numerical qualifications remain archived.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Resume only on owner instruction. Preserve opt-in/fallback/tie cases, the measured 32k serving regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #49732: Historical/unrefreshed and outside active scope. Base attempt 3 passed 31 functional jobs. Maintainer review and integrated-image qualification remain. Historical isolated configuration checks and 9763ef56 GPU evidence have narrower source and module coverage.

Tracking key: `pr49732-base-review-validation`.

Next action: Retain for explicit owner resumption. Maintainers review the continuation with yu-depend/#46918 attribution. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Exact-head approval and author current-code A/B are present. Fresh CPU metadata confirms parent launch routes for H32/H96/H128, H64 fallback at seq15, 16 splits at seq16 and two splits at seq128 on 256 CUs. Unreachable geometry, rotation and reducer variants are absent. Seq15 and non-target numerical comparisons against the exact parent and immutable raw provenance remain open. ROCm/triton#985 is a suspected, unconfirmed explanation for async-copy results. The gated behavior is reviewer-nonblocking. Warmup timing remains author-reported.

Tracking key: `pr47278-review-20261117`.

Next action: Maintainers assess aggregate review. Owner retrieves raw source/image, per-run A/B and exact-parent numerical records, including scaled-KV, BF16 partials, DCP/LSE and layout/replay coverage. Local CPU and review evidence is in cycles/20261124T150002Z/PR47278_REVIEW_HANDOFF.json and cpu-review-result.json.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: Current main f4284f29 has pr-gate.yml SHA256 bae2b8efdf384503855df4ba5c8c6649dd62e0907cab59940a5b08b4ac790371, different from the helper's audited cooldown hash 63e133892c2ef42609c0fdd33f3ae524afad21fb792c9eb3bbf8518916915af1. Required minimum base 30e076fa is absent from the passing #47276/#47278 heads and present in #47897. No cooldown gate failed this cycle.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: Preserve both passing heads. If a future cooldown gate fails, the automation maintainer audits upstream policy and authorizes any helper update. Assess any concrete source-integration need separately.

- [Evidence](https://github.com/sgl-project/sglang/blob/f4284f29ba56fffe2d09c9e6c5ac5e218e6cc053/.github/workflows/pr-gate.yml)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47897: Ready/run-ci head 5a06f9e7 contains required 30e076fa. Base 45338553315 attempt 1 failed the tests-ran assertion because exact-commit CI approval is missing. Eighteen jobs skipped and zero functional/JIT/build jobs ran. Prior request 7123836485 reached successful handler 45339762031, which reran zero workflows. The guarded dry run returned run_attempt_already_reserved after the hourly slot reopened. No newer attempt exists.

Tracking key: `pr47897-current-head-base`.

Next action: Upstream maintainer approves exact 5a06f9e7 through /tag-and-rerun-ci. Verify handler and actual Base test execution after approval. Preserve the integrated head and use the guarded PR-wide helper for eligible later failures.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/job/138256395291)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/job/138256449433)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123836485)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45339762031/job/138260319757)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157)

## #47897: Exact-head caller source confirms generic gfx950 preshuffled calls build a persistent schedule and take Aiter, while qualifying verify calls take direct Aiter. Eligible unscheduled unit calls can reach Gluon. Current serving engagement and transfer of the historical performance gain remain unestablished.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: Owner and kernel reviewer map intended deployment calls and archive an exact-head dispatch trace with source and image identity. Preserve schedules and paired-row/tail/fallback behavior. Hand off any performance-sensitive dispatch change before publication.

- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L94)
