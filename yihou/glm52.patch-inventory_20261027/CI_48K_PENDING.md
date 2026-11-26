# Pending handoff for the next user session

Updated: **2026-11-26T12:07:40.330487+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: All three required checks pass at 5818b0bf. The AMD PP file passed. The separate NIXL startup at PP size 1 exited with code -9 before accuracy cases and bypassed the PP size greater than one branch. Review and AWQ qualification remain open. The frozen patch adds a producer event in the overlap result handler. The PR adds it in the non-overlap PP chunk handler. Deployment equivalence remains unverified.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us and PD maintainers own review. Retain the frozen patch for reproduction and qualify both paths before image substitution. If NIXL startup recurs, obtain process or cgroup termination evidence.

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

Next action: After commit approval, verify Base execution. Maintainers review and coordinate the #45307 overlap. The owner archives exact-source and independent numerical evidence. Keep the e95ec5ab Base pass bound to its head.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Historical and unrefreshed. Inherited Base and AMD passes on merged head 9602092a include 45 sparse-MLA and 13 disaggregation cases in matching AMD logs. Exact source, image and raw artifact archival, and independent numerical, layout, and replay coverage remain open.

Tracking key: `pr47278-amd-validation-source`.

Next action: The owner archives the candidate and parent 8d72d2ea, image, and raw runs. Retain BF16 partials, seq15 and non-target comparisons, scaled-KV, DCP/LSE, and graph replay.

- [Evidence](https://github.com/sgl-project/sglang/commit/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)

## #47897: Head d6cb51b6 preserves both production callers. Code review, exact-source, image and raw-run archival, production Gluon dispatch, and long-context numerical coverage remain open. Historical isolated speedups and short-context GSM8K retain their original source and test scope.

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

## #47278: Historical and unrefreshed. Inherited approvals on the merged commit and the merge closed pre-merge review. The author reported A/B and cold-cache warmup. Prior checks established executable AST parity with the measured revision, parent routes for H32/H96/H128, H64 fallback at seq15, 16 splits at seq16 and two at seq128 on 256 CUs, and removal of unreachable variants. ROCm/triton#985 remains a suspected cause of the gated async-copy combination. The reviewer treated compiler follow-up as nonblocking.

Tracking key: `pr47278-review-20261117`.

Next action: The owner archives source, image, and per-run A/B and numerical records for exact-parent non-target and seq15 comparisons, scaled-KV, BF16 partials, DCP/LSE, and repeated graph replay including heads 48–63.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6052268784)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: Both owned Base runs match the current SHA, fork, and branch under pull_request_target. Both heads contain minimum base 30e076fa. Main dd87c5fd still requires commit approval and uses pull_request_target cooldown history. The helper's owner_requested path checks the gate job conclusion and can accept an executed Skip tests step. The stale-gate verifier pins a different digest and queries pull_request history. Both observed attempts skipped cooldown.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: The automation maintainer updates launcher selection and helper admission checks for bot commit approval, validates the PR-wide handler behavior, and aligns cooldown event matching before unattended retries resume under this workflow.

- [Evidence](https://github.com/sgl-project/sglang/blob/dd87c5fd0b197649ae259c1aad5cfa517c06d274/.github/workflows/pr-test.yml#L199)
- [Evidence](https://github.com/sgl-project/sglang/blob/dd87c5fd0b197649ae259c1aad5cfa517c06d274/.github/workflows/pr-gate.yml#L109)
- [Evidence](https://github.com/sgl-project/sglang/blob/dd87c5fd0b197649ae259c1aad5cfa517c06d274/.github/workflows/pr-gate.yml#L143)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47897: Base 45555280464 attempt 1 denied admission for d6cb51b6500f. The required finish assertion failed. Zero tests or builds ran. The final recheck found no bot approval, later attempt, or current-head reservation. The historical November 24 request started zero workflows on the superseded head.

Tracking key: `pr47897-current-head-base`.

Next action: An authorized maintainer runs /approve-ci while d6cb51b6500f is current. Verify the bot approval status, handler outcome, and actual Base test execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005783735)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005836787)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157)

## #47897: At d6cb51b6 the generic gfx950 preshuffled path forwards a persistent schedule to Aiter and qualifying target verify calls Aiter directly. Both dispatch functions matched main dd87c5fd in a fresh AST comparison. Inherited checks cover eligible unscheduled Gluon routing and schedule forwarding. Production Gluon engagement and transfer of the historical gain remain unverified.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: The owner and kernel reviewer archive current-head dispatch traces with source and image identity. Preserve persistent schedules, paired rows, tails, and fallback. Hand off performance-sensitive changes before publication.

- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L159)

## #47276: Base 45555151383 attempt 1 denied admission for d334464b7225. The required finish assertion failed. Zero tests or builds ran. The final recheck found no bot approval, later attempt, or current-head reservation.

Tracking key: `pr47276-current-head-base`.

Next action: An authorized maintainer runs /approve-ci while d334464b7225 is current. Verify the bot approval status, handler outcome, and actual Base test execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005372702)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005424922)
