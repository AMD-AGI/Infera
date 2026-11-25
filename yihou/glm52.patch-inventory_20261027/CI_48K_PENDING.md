# Pending handoff for the next user session

Updated: **2026-11-25T06:07:40.408458+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Historical and unrefreshed. Base passed 27 functional jobs. AMD shard 6 exited 255 after test start and historical log retrieval returned HTTP 404. The first exception, source equivalence and AWQ and PP review remain open.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us and maintainers own log recovery, AMD coverage and source-equivalence review. Preserve the frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Historical and unrefreshed. Base passed 36 functional jobs and two shards cancelled before tests. One annotation identifies github-actions[bot]. The finish log attributes aggregate failure to cancellation. The cancellation policy and integration remain open. Scoped MI355X validation applies to 9763ef56.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend and maintainers own the original PR conflict, coverage and review. Preserve the frozen workaround and attribution alongside continuation #49732.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: The d334464b rebase retains the guarded gfx950 change and needs new Base coverage. Code review, #45307 overlap, exact-source and run archival, and independent numerical qualification remain open. The path is optional and disabled in llr37. Attribution of the historical FlashInfer graph mismatch remains unproven.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review and coordinate #45307 overlap. The owner archives exact-source and independent numerical evidence, keeping the e95ec5ab Base pass bound to that head.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Merged with green Base and AMD on 9602092a. Inherited matching AMD logs cover 45 sparse-MLA and 13 disaggregation cases. Exact source, image and raw artifact archival and independent numerical, layout and replay coverage remain open after merge.

Tracking key: `pr47278-amd-validation-source`.

Next action: The owner archives candidate and parent 8d72d2ea source, image and raw runs. Retain BF16 partials, seq15 and non-target comparisons, scaled-KV, DCP/LSE and graph replay.

- [Evidence](https://github.com/sgl-project/sglang/commit/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)

## #47897: d6cb51b6 preserves the three reviewed PR file blobs and both production caller blobs. Code review, exact-source, image and raw archival, production Gluon engagement and long-context numerical coverage remain open. Historical isolated speedups and short-context GSM8K retain their recorded source and test scope.

Tracking key: `pr47897-source-long-context`.

Next action: Maintainers review the rebased head. The owner and kernel reviewer archive exact-source evidence and qualify Gluon engagement, unequal paired tables and lengths, common-prefix and tail cases, fallback and long-context accuracy using approved GPU allocation for concrete gaps. Keep frozen throughput attached to the whole deployment.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)

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

## #47278: The exact-commit approvals and merge close pre-merge review. The author-reported A/B is present. Inherited checks establish identical executable ASTs to the measured revision, parent routes for H32/H96/H128, H64 fallback at seq15, 16 splits at seq16 and two at seq128 on 256 CUs. Unreachable geometry, rotation and reducer variants were removed. ROCm/triton#985 remains a suspected cause of the gated async-copy combination. Compiler follow-up was reviewer-nonblocking, and cold-cache warmup timings are author-reported.

Tracking key: `pr47278-review-20261117`.

Next action: The owner archives source, image and per-run A/B and numerical records for exact-parent non-target and seq15 comparisons, scaled-KV, BF16 partials, DCP/LSE and repeated graph replay including heads 48–63.

- [Evidence](https://github.com/sgl-project/sglang/commit/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6052268784)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: Both new heads contain minimum base 30e076fa, and both Base runs use pull_request_target. Fork, branch and full SHA matched. On live main 75ed36d9, pr-gate.yml still differs from the helper's audited hash. Upstream cooldown history selects pull_request_target while the helper selects pull_request. The helper also classified both missing-approval attempts as dry-run eligible because it treats a successful gate job as admission. Neither current failure came from the cooldown.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: The automation maintainer audits commit approval, executed gate outcomes, cooldown history and launcher selection before unattended retries resume. The current report uses valid attempt links with a GitHub query parameter to keep the existing launcher from submitting either unapproved attempt.

- [Evidence](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/.github/workflows/pr-gate.yml)
- [Evidence](https://github.com/sgl-project/sglang/blob/75ed36d976113c693f8e6f64925e119573a93fda/.github/workflows/pr-test.yml#L538)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47897: Base 45555280464 attempt 1 failed the tests-ran assertion because d6cb51b6 lacks commit approval. Zero test or build jobs executed. On the prior head, reserved request 7123836485 reached the handler and started zero workflows. The new attempt has no reservation.

Tracking key: `pr47897-current-head-base`.

Next action: An upstream maintainer approves d6cb51b6500f with /approve-ci using the existing run-ci label. Verify the bot's approval status, the new Base attempt and actual test execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005783735)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005836787)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123836485)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45339762031/job/138260319757)

## #47897: The rebased caller blobs are unchanged. The generic gfx950 preshuffled route builds a persistent schedule and uses Aiter, and the qualifying target-verify route uses direct Aiter. Inherited CPU checks passed for eligible unscheduled Gluon routing and persistent-schedule forwarding. Production Gluon engagement and transfer of the historical performance gain remain unestablished.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: The owner and kernel reviewer archive current-head dispatch traces with source and image identity. Preserve persistent schedules and paired-row, tail and fallback behavior. Hand off performance-sensitive dispatch changes before publication.

- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L94)

## #47276: Base 45555151383 attempt 1 failed the tests-ran assertion because d334464b lacks commit approval. The gate skipped tests and zero functional, JIT or build jobs executed. The new attempt has no reservation.

Tracking key: `pr47276-current-head-base`.

Next action: An upstream maintainer approves d334464b7225 with /approve-ci using the existing run-ci label. Verify the bot's approval status, a new Base attempt and actual test execution.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005372702)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555151383/job/139005424922)
