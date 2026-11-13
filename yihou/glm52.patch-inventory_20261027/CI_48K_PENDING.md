# Pending handoff for the next user session

Updated: **2026-11-13T00:11:19.515649+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Base passed 27 functional jobs. Same AMD shard 6 exited 255 after starting tests; historical log retrieval returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve runner logs, recover coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Base has 36 functional successes and two shards cancelled without tests. One annotation identifies github-actions[bot]; finish log confirms cancellation caused the aggregate failure. Cancellation policy and original-PR integration remain unresolved. Scoped validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers own original-PR conflict, remaining coverage and review decisions, including the unexplained cancellation policy. Advance owned continuation #49732 separately; preserve frozen workaround and source-specific MI355X evidence at 9763ef56.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 27 repeated the FlashInfer MLA graph mismatch (1/6 elements, absolute difference 43); 10/36 B0 files passed, one failed and 25 were unexecuted. Helper request, reservation and successful handler verified; attempt 28 is active with 22 retained functional passes. B0 Run test started at 2026-11-13T00:07:24Z; no new functional result or failure is available. Test/helper blobs match main 7e31007a; no verified correction or patch causality established. Source/overlap/numerical qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow active attempt 28 without duplicate requests. Runner/attention owners capture the mismatching output component, dtype/tensors and graph-replay state; patch causality remains unproved. If Base ends unsuccessfully, recover through the helper no earlier than 2026-11-13T01:03:31Z, subject to fresh author-wide eligibility. Preserve optional llr37-disabled role, #45307 overlap coordination and exact-source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/28)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/131585568484)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6839612880)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/43333962496)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/131490641505)
- [Evidence](https://github.com/sgl-project/sglang/blob/7e31007a86d4cfe29c3e86ee93aaad51e7ff950e/test/registered/attention/unittests/mla/test_flashinfer.py)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Current-head AMD passed all 17 functional stages, finish, JIT unit tests and benchmark. Inherited 33-test MI355X report still needs exact-source/run archiving and BF16/DCP/LSE/combined-stack qualification.

Tracking key: `pr47278-amd-validation-source`.

Next action: Owner/kernel reviewers archive and verify matching reconciled-candidate evidence; preserve numerical and e2e limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947083003/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base passed 27 functional jobs. Matching inherited logs identify AMD MIOpen SQLite database-open error and PP KVTransferError; Xeon package/checkout mismatch remains. ARM64 passed. Five later MIOpen files, two PP files and 16/23 Xeon files were unexecuted.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, investigate PP peers and align installed/test revisions; preserve passing Base and numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K does not establish long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence and complete unequal paired-row/table-length, dispatch/fallback and long-context review.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: OWNER PAUSED pending redesign. Historical passing CI and source/numerical qualifications remain archived and were not queried.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Owner-paused; resume only on owner instruction. Preserve opt-in/fallback/tie, measured 32k regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Matching Xeon failure cannot import _paged_experts_num_resident from the installed package. Checkout/package alignment remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Author supplied reconciliation d5fb0d33 and candidate f5fc9afe, reporting scaled-KV compatibility, a 16-split cap and 33 MI355X passes. Source/run archival verification, scaled-KV/group-width/launch/reducer review and current-head e2e qualification remain.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Owner/kernel reviewers verify supplied evidence and preserve BF16 partials, DCP/LSE, GROUP_W, scaled-KV/cache/async-copy and matching launch/reducer semantics. Archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/commit/d5fb0d33dd664193bee70c21025f185766e7f492)
- [Evidence](https://github.com/sgl-project/sglang/commit/f5fc9afe5ef1e2dc619ad90d90d6de9b9bbfc335)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #49732: Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)
