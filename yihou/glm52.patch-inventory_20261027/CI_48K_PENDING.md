# Pending handoff for the next user session

Updated: **2026-11-28T00:09:21.893136+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: All three required checks pass on 5818b0bf. GitHub still requires review. AWQ qualification remains open. The frozen result handler and PR PP chunk handler record events at distinct call sites. The affected AMD shard passed all three PP cases, then hit duplicate LLVM registration during PP1 NIXL warmup. PP1 bypasses the changed branch and no NIXL accuracy cases executed.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us and requested PD maintainers, including ByronHsu and ShangmingCai, own review. The deployment owner qualifies AWQ and the distinct producer-event call sites before replacing the frozen image patch. The AMD runner owner can inspect the FlyDSL image if PP1 NIXL coverage is needed.

- [Evidence](https://github.com/sgl-project/sglang/pull/45524/checks)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44509301536/job/135516407197)
- [Evidence](https://github.com/sgl-project/sglang/blob/5818b0bf466c965c2dc8bc5aaa48810646fea32a/python/sglang/srt/disaggregation/prefill.py#L1606)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0/deploy/docker/patches/sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py#L48)

## #46918: Historical and unrefreshed. Base passed 36 functional jobs and two shards were cancelled before tests. The finish log attributes the aggregate result to cancellation. Cancellation policy and integration remain open. Scoped MI355X validation applies to 9763ef56.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend and maintainers own the original conflict, coverage, and review. Preserve the frozen workaround and attribution alongside #49732.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Head 783420b6 needs Base coverage, review, #45307 overlap coordination, exact-source and run archival, and independent numerical qualification. The fused-indexer optimization is disabled in llr37. The cause of the historical FlashInfer graph mismatch remains unestablished.

Tracking key: `pr47276-base-overlap-source`.

Next action: Verify Base tests after commit approval. Maintainers review and coordinate #45307 overlap. The owner archives exact source and independent numerical evidence. Keep the prior Base pass bound to e95ec5ab.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Historical and unrefreshed. Inherited Base and AMD passes on merged head 9602092a include 45 sparse-MLA and 13 disaggregation cases in matching AMD logs. Source, image and raw artifact archival, and independent numerical, layout, and replay coverage remain open.

Tracking key: `pr47278-amd-validation-source`.

Next action: The owner archives the candidate and parent 8d72d2ea, image, and raw runs. Retain BF16 partials, seq15 and non-target comparisons, scaled-KV, DCP/LSE, and graph replay.

- [Evidence](https://github.com/sgl-project/sglang/commit/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)

## #47897: Head d6cb51b6 preserves the production callers. Review, exact-source, image and raw-run archival, production Gluon dispatch, and long-context numerical coverage remain open. Historical isolated speedups and short-context GSM8K retain their original source and test scope.

Tracking key: `pr47897-source-long-context`.

Next action: Maintainers review the head. On approved hardware, the owner and kernel reviewer archive exact-source evidence, trace production dispatch, and compare valid logits and top-k against Aiter for unequal paired tables and lengths, including a late common-prefix mismatch across a split boundary, tails, and fallback. Qualify long-context accuracy separately.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/gluon_paged_mqa_logits.py#L1328)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1665)

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

## #47276: Main 96367507 still uses pull_request_target and commit approval. It checks for the bot status and uses pull_request_target for cooldown history. The installed helper accepts an outer successful gate for owner_requested, pins an older cooldown digest and queries pull_request cooldown history. The upstream rerun handler skips approval-gated Base attempts. The check-focused attempt links keep the two approval-gated rows out of the legacy launcher's retry parser.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: The automation maintainer adds exact-commit bot approval and executed gate-step checks to the launcher and helper, aligns cooldown and event matching, and tests the PR-wide handler's workflow selection before unattended retries resume on this workflow.

- [Evidence](https://github.com/sgl-project/sglang/blob/963675078b11567a8ded2a34b6ed0a81d8d61537/.github/workflows/pr-test.yml#L196)
- [Evidence](https://github.com/sgl-project/sglang/blob/963675078b11567a8ded2a34b6ed0a81d8d61537/.github/workflows/pr-gate.yml#L109)
- [Evidence](https://github.com/sgl-project/sglang/blob/963675078b11567a8ded2a34b6ed0a81d8d61537/.github/workflows/pr-gate.yml#L144)
- [Evidence](https://github.com/sgl-project/sglang/blob/963675078b11567a8ded2a34b6ed0a81d8d61537/scripts/ci/utils/slash_command_handler.py#L578)

## #47897: Base 45555280464 attempt 1 on d6cb51b6 executed Skip tests and failed the required finish assertion. Verified zero tests and builds, no bot approval, zero current-head reservations and no later attempt. The November 24 request on a superseded head started zero workflows.

Tracking key: `pr47897-current-head-base`.

Next action: An authorized maintainer posts a new /approve-ci comment while d6cb51b6 is current. The handler is designed to rerun eligible approval-gated workflows. Verify the bot status, handler outcome, and actual Base test execution before considering further recovery.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005783735)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45555280464/job/139005836787)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157)

## #47897: The generic gfx950 preshuffled caller forwards a persistent schedule to Aiter. Qualifying target verification also uses Aiter directly. Three caller functions match main 96367507 at the executable AST level. The inherited isolated CPU test used mocked backends. Production Gluon engagement and transfer of the historical gain remain unverified.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: The owner and kernel reviewer archive current-head dispatch traces with source and image identity. Preserve persistent scheduling, paired rows, tails, and fallback. Hand off performance-sensitive changes before publication.

- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1665)
- [Evidence](https://github.com/sgl-project/sglang/blob/d6cb51b6500fb5033b5a0fb8789ebd04acee59a7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L159)

## #47276: Base 46022709382 attempt 1 on 783420b6 executed Skip tests and failed the required finish assertion. Verified zero tests and builds, no bot approval, zero current-head reservations and no later attempt. AMD Actions approval is separate.

Tracking key: `pr47276-current-head-base`.

Next action: An authorized maintainer posts a new /approve-ci comment while 783420b6 is current. Verify the bot status, handler outcome, and actual Base test execution before considering further recovery.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/46022709382/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/46022709382/job/140586203268)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/46022709382/job/140586275436)

## #47276: The monokernel caller and fused indexer share the changed gfx950 JIT logits entry point and top-k implementation. The inherited CPU argument check passed. AMD attempt 1 on 783420b6 concluded action_required and created zero jobs. gfx950 JIT execution and numerical coverage remain open.

Tracking key: `pr47276-amd-monokernel-coverage`.

Next action: The AMD maintainer clears the workflow's Actions approval and checks the MI35x suite for executed gfx950 cases. The kernel reviewer qualifies monokernel decode and target verification with the fused indexer disabled, including top-k against the standard path, on approved hardware if CI lacks those cases.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/46022715300/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/blob/783420b63715a00d7af1449325651b17878b6934/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L306)
- [Evidence](https://github.com/sgl-project/sglang/blob/783420b63715a00d7af1449325651b17878b6934/test/registered/amd/test_dsa_gfx950_fused_indexer.py#L59)
