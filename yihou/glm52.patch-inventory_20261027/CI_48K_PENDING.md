# Pending handoff for the next user session

Updated: **2026-11-25T00:08:20.015515+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Historical and unrefreshed. Base passed 27 functional jobs. AMD shard 6 exited 255 after test start. Historical log retrieval returned HTTP 404. The first exception, source equivalence and AWQ and PP review remain open.

Tracking key: `pr45524-external-ci-review`.

Next action: Retain for explicit resumption. External author eaglewangao-us and maintainers own log recovery, AMD coverage and source-equivalence review. Preserve the frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Historical and unrefreshed. Base passed 36 functional jobs and two shards cancelled before tests. One annotation identifies github-actions[bot]. The finish log attributes the aggregate failure to cancellation. Cancellation policy and integration remain open. Scoped MI355X validation applies to 9763ef56.

Tracking key: `pr46918-external-ci-integration`.

Next action: Retain for explicit resumption. External author yu-depend and maintainers own original PR conflict, coverage and review. Preserve the frozen workaround and attribution alongside continuation #49732.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base remains green on e95ec5ab. Aggregate review, #45307 overlap coordination, exact-source and run archival, and independent numerical qualification remain open. The path is optional and disabled in llr37. Historical FlashInfer mismatch attribution remains unproven.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review and coordinate #45307 overlap while preserving e95ec5ab. The owner archives exact-source and independent numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base and AMD remain green on 9602092a. Matching inherited AMD logs cover 45 sparse-MLA and 13 disaggregation cases. Exact-head approval is present and aggregate review remains required. Source, image and raw artifact archival and independent numerical, layout and replay coverage remain open. Earlier OOM and cancellation are historical.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers assess aggregate review and preserve 9602092a. The owner archives candidate and parent 8d72d2ea source, image and raw runs. Retain BF16 partials, seq15 and non-target comparisons, scaled-KV, DCP/LSE and graph replay. See cycles/20261125T000002Z/PR47278_REVIEW_HANDOFF.md.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)

## #47897: Integrated 5a06f9e7 preserves the three reviewed PR file blobs and both production caller blobs. Aggregate review, exact source, image and raw archival, production Gluon engagement and long-context numerical coverage remain open. Historical isolated speedups and short-context GSM8K apply to their recorded inputs and sources.

Tracking key: `pr47897-source-long-context`.

Next action: Maintainers review the integrated head. The owner and kernel reviewer archive exact-source evidence and qualify Gluon engagement, unequal paired tables and lengths, common-prefix and tail cases, fallback and long-context accuracy. Use approved GPU allocation for concrete gaps. Keep frozen whole-stack throughput attached to the deployment.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)

## #47872: Historical and unrefreshed. Owner-paused pending redesign. Historical passing CI and source and numerical qualifications remain archived.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Resume on owner instruction. Preserve opt-in, fallback and tie cases, the 32k serving regression and exact-source and long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #49732: Historical and unrefreshed. Base attempt 3 passed 31 functional jobs. The copied watch entry carried maintainer review and integrated-image qualification. The local handoff separately records a November 17 merge. Configuration checks and 9763ef56 GPU evidence have narrower source and module coverage.

Tracking key: `pr49732-base-review-validation`.

Next action: Preserve yu-depend/#46918 attribution, integrated-image qualification and exact-source numerical evidence. Reconcile the historical lifecycle if active tracking resumes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Exact-head approval and the author-reported A/B are present. The measured revision has identical executable ASTs. CPU metadata reproduced parent routes for H32/H96/H128, H64 fallback at seq15, 16 splits at seq16 and two at seq128 on 256 CUs. Unreachable geometry, rotation and reducer variants are absent. ROCm/triton#985 remains a suspected cause of the gated async-copy combination. Compiler follow-up was reviewer-nonblocking and warmup remains author-reported.

Tracking key: `pr47278-review-20261117`.

Next action: Maintainers assess aggregate review. The owner archives source, image and per-run A/B and numerical records covering exact-parent non-target and seq15 comparisons, scaled-KV, BF16 partials, DCP/LSE and graph replay.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: Current main 2e4f4ca1 has pr-gate.yml SHA256 159d02b6a6a6b32688e773055712281a0e113b0ee1fab8b540f36ea94c9ff6bd, which differs from the helper's audited cooldown hash. The current minimum base 30e076fa is absent from the passing #47276 and #47278 heads and present in #47897. No cooldown gate failed this cycle.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: Preserve both passing heads. If a future cooldown gate fails, the automation maintainer audits upstream policy before any helper update. Assess source integration when a concrete need arises.

- [Evidence](https://github.com/sgl-project/sglang/blob/2e4f4ca1939d25f9c65377daabd5c19e237e97d0/.github/workflows/pr-gate.yml)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47897: Base 45338553315 attempt 1 failed the tests-ran assertion after the gate skipped tests for missing exact-commit CI approval. Zero functional, JIT or build jobs ran. Reserved request 7123836485 reached handler 45339762031, which started zero workflows. The guarded dry run again returned run_attempt_already_reserved. The final query found no approval status and still shows attempt 1.

Tracking key: `pr47897-current-head-base`.

Next action: An upstream maintainer reviews and approves exact head 5a06f9e7 for CI. Current upstream supports /approve-ci with the existing run-ci label. Verify commit approval, the current-head Base attempt and actual test starts before any later eligible helper recovery.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/job/138256395291)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/job/138256449433)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123836485)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45339762031/job/138260319757)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157)
- [Evidence](https://github.com/sgl-project/sglang/blob/2e4f4ca1939d25f9c65377daabd5c19e237e97d0/scripts/ci/utils/slash_command_handler.py#L2166)

## #47897: The verified caller blobs show the generic gfx950 preshuffled route builds a persistent schedule and uses Aiter. The qualifying target-verify route uses direct Aiter. CPU checks passed for eligible unscheduled Gluon routing and persistent-schedule forwarding. Production Gluon engagement and transfer of the historical performance gain remain unestablished.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: The owner and kernel reviewer archive current-head dispatch traces with source and image identity. Preserve persistent schedules and paired-row, tail and fallback behavior. Hand off any performance-sensitive dispatch change before publication.

- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L94)
