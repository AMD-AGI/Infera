# Pending handoff for the next user session

Updated: **2026-11-27T09:08:35.722343+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: All three required checks pass on 5818b0bf. Three AMD PP tests passed. Separate PP-size-1 NIXL setup bypassed the changed branch and aborted during Aiter FlyDSL GEMM warmup after duplicate LLVM registration of spirv-expand-step. Review, AWQ qualification, and equivalence between the frozen overlap result handler and the PR's PP chunk handler remain open.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us and requested PD maintainers own review. ByronHsu and ShangmingCai are among the requested reviewers. Qualify AWQ and both call sites before image substitution. If NIXL startup fails again, AMD CI owners can compare the loaded LLVM libraries and Aiter/FlyDSL build.

- [Evidence](https://github.com/sgl-project/sglang/pull/45524/checks)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44509301536/job/135516407197)
- [Evidence](https://github.com/sgl-project/sglang/blob/5818b0bf466c965c2dc8bc5aaa48810646fea32a/python/sglang/srt/disaggregation/prefill.py#L1606)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0/deploy/docker/patches/sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py#L48)

## #46918: Historical and unrefreshed. Base passed 36 functional jobs and two shards were cancelled before tests. The finish log attributes the aggregate result to cancellation. Cancellation policy and integration remain open. Scoped MI355X validation applies to 9763ef56.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend and maintainers own the original conflict, coverage, and review. Preserve the frozen workaround and attribution alongside #49732.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Head d334464b needs Base coverage, review, #45307 overlap coordination, exact-source and run archival, and independent numerical qualification. The path is disabled in llr37. The cause of the historical FlashInfer graph mismatch remains unestablished.

Tracking key: `pr47276-base-overlap-source`.

Next action: Verify Base execution after commit approval. Maintainers review and coordinate the #45307 overlap. The owner archives exact source and independent numerical evidence. Keep the e95ec5ab Base pass bound to its head.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Historical and unrefreshed. Inherited Base and AMD passes on merged head 9602092a include 45 sparse-MLA and 13 disaggregation cases in matching AMD logs. Source, image and raw artifact archival, and independent numerical, layout, and replay coverage remain open.

Tracking key: `pr47278-amd-validation-source`.

Next action: The owner archives the candidate and parent 8d72d2ea, image, and raw runs. Retain BF16 partials, seq15 and non-target comparisons, scaled-KV, DCP/LSE, and graph replay.

- [Evidence](https://github.com/sgl-project/sglang/commit/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)

## #47897: Head d6cb51b6 preserves both production callers. Review, exact-source, image and raw-run archival, production Gluon dispatch, and long-context numerical coverage remain open. Historical isolated speedups and short-context GSM8K retain their original source and test scope.

Tracking key: `pr47897-source-long-context`.

Next action: Maintainers review the current head. The owner and kernel reviewer archive exact-source evidence and qualify production dispatch, unequal paired tables and lengths, common prefixes, tails, fallback, and long-context accuracy on approved GPU allocation.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)

## #47872: Historical and unrefreshed. Owner-paused pending redesign. Historical passing CI and source and numerical qualifications remain archived.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Resume on owner instruction. Preserve opt-in, fallback and tie cases, the 32k serving regression, and exact-source and long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #49732: Historical and unrefreshed. Base attempt 3 passed 31 functional jobs. The copied entry carried maintainer review and integrated-image qualification. The curated handoff records a November 17 merge. Isolated configuration checks and 9763ef56 GPU evidence have narrow module and source coverage.

Tracking key: `pr49732-base-review-validation`.

Next action: Preserve yu-depend/#46918 attribution, integrated-image qualification, and exact-source numerical evidence. Reconcile the lifecycle if active tracking resumes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Historical and unrefreshed. Inherited approvals and the merge closed pre-merge review. The author reported A/B and cold-cache warmup. Prior checks established executable AST parity with the measured revision, parent routes for H32/H96/H128, H64 fallback at seq15, 16 splits at seq16 and two at seq128 on 256 CUs, and removal of unreachable variants. ROCm/triton#985 remains a suspected cause of the gated async-copy combination. The reviewer treated compiler follow-up as nonblocking.

Tracking key: `pr47278-review-20261117`.

Next action: The owner archives source, image, and per-run A/B and numerical records for exact-parent non-target and seq15 comparisons, scaled-KV, BF16 partials, DCP/LSE, and repeated graph replay including heads 48–63.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6052268784)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: On main c2693020, Base uses pull_request_target with commit approval, and the rate-limit gate counts pull_request_target history. The installed helper can admit an authorization skip as a successful gate. Its cooldown verifier pins an older gate digest and queries pull_request history. The upstream PR-wide handler leaves approval-gated Base runs alone.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: The automation maintainer updates launcher and helper admission to check bot commit approval and executed gate steps, updates cooldown event matching and policy verification, and tests PR-wide handler selection before unattended retries resume on this workflow.

- [Evidence](https://github.com/sgl-project/sglang/blob/c269302082327f0899ac3c470bf713fe1446a8fa/.github/workflows/pr-test.yml#L199)
- [Evidence](https://github.com/sgl-project/sglang/blob/c269302082327f0899ac3c470bf713fe1446a8fa/.github/workflows/pr-gate.yml#L109)
- [Evidence](https://github.com/sgl-project/sglang/blob/c269302082327f0899ac3c470bf713fe1446a8fa/.github/workflows/pr-gate.yml#L144)
- [Evidence](https://github.com/sgl-project/sglang/blob/c269302082327f0899ac3c470bf713fe1446a8fa/scripts/ci/utils/slash_command_handler.py#L575)

## #47897: Base 45555280464 attempt 1 at d6cb51b6500f executed Skip tests and failed the required finish assertion. Rechecked zero tests and builds, missing bot approval, no later attempt, and no reservation for this head. The inherited November 24 request on a superseded head started zero workflows.

Tracking key: `pr47897-current-head-base`.

Next action: An authorized maintainer posts /approve-ci while d6cb51b6500f is current. Verify the bot approval status, handler outcome, and actual Base test execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005783735)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005836787)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157)

## #47897: At d6cb51b6 the generic gfx950 preshuffled caller forwards a persistent schedule to Aiter, and qualifying target verify calls Aiter directly. Both caller functions matched main c2693020 in this cycle's AST comparison. Inherited CPU checks cover eligible unscheduled Gluon routing and schedule forwarding. Production Gluon engagement and transfer of the historical gain remain unverified.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: The owner and kernel reviewer archive current-head dispatch traces with source and image identity. Preserve persistent schedules, paired rows, tails, and fallback. Hand off performance-sensitive changes before publication.

- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L159)

## #47276: Base 45555151383 attempt 1 at d334464b7225 executed Skip tests and failed the required finish assertion. Rechecked zero tests and builds, missing bot approval, no later attempt, and no reservation for this head.

Tracking key: `pr47276-current-head-base`.

Next action: An authorized maintainer posts /approve-ci while d334464b7225 is current. Verify the bot approval status, handler outcome, and actual Base test execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005372702)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005424922)
