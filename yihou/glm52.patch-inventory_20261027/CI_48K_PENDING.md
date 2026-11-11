# Pending handoff for the next user session

Updated: **2026-11-11T21:11:24.183978+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Base passed 27 functional jobs. AMD shard 6 exited 255 after starting tests; log retrieval again returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers retrieve runner logs, recover coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Base retains 36 functional successes and two shards cancelled without tests. Cancellation policy and source integration remain unresolved. Scoped validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers own original-PR conflict, coverage and review decisions. Progress owned continuation #49732 separately; preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 20 remains failed on FlashInfer MLA EAGLE draft graph mismatch (1/6 elements, absolute difference 43); nine files passed and 25/35 did not execute. Recovery refused by shared hourly interval. Source, overlap and numerical qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker retries after 2026-11-11T22:02:56Z with fresh helper checks. Runner owner captures tuple component/dtype, tensors and replay state. Preserve optional llr37-disabled role, #45307 overlap and exact-source/numerical review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/20)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/130604174070)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Base attempt 2 active after verified recovery. All three C shards started tests: B200 Run test passed but its job is still finishing; H100/GB300 tests remain active. All 28 prior functional passes verified retained. New suite/job outcomes remain pending.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker follows active attempt without duplication and verifies final test results. Preserve supplied candidate and separate source/numerical review constraints.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42947082865/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/43099736228)

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
