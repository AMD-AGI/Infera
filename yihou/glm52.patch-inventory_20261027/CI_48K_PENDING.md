# Pending handoff for the next user session

Updated: **2026-11-03T10:18:43.914937+00:00**.

These items remain open until supported by resolution evidence. Reporting them to the user does not resolve them.

## #33059: Two Base H100 shards and three AMD stages remain queued. The Kimi MIOpen database failure persists; neither Base nor AMD is complete.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: CI owners: finish active attempts and investigate database accessibility before choosing retry scope.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254739/job/125747856350)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated MI300 collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: One H100 shard fails VLM preprocessing/warmup; one remains queued. All 36 Base successes are retained. AMD passes, but required review remains incomplete.

Tracking key: `pr46820-base-review`.

Next action: CI owners: compare baseline execution and finish Base. Maintainers: complete common-code review while preserving CPU socket-contract evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: One H100 shard newly passed; one remains queued without execution. Base has 37 successes, including 36 retained. Aggregate completion and required review remain.

Tracking key: `pr46830-base-review`.

Next action: CI owners: finish the remaining shard. Maintainers: complete review while preserving historical DSA deadlock evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/job/125761043136)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830)

## #45524: Base has 35 A/B successes and four queued shards. AMD MLA setup hits a 300-second watchdog during Inductor compilation, then exit -9. ARM64/Xeon also expose fork_worker_preload API mismatch; historical-source equivalence remains unproven.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose compiler waiting and CPU runner/package mismatch, finish CI and review. Preserve the separate frozen producer-ordering patch; no worker retry or branch takeover.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758584/job/125761894917)

## #46918: Cancelled Base coverage and AMD PP transfer errors remain. Validation at 9763ef56 does not qualify the entire current head or integrated image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover coverage, diagnose transfers, complete review and validate image integration before retiring the frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734491/job/125360336390)

## #47276: Two H100 shards remain queued. Review, #45307 overlap coordination and final source/run archiving remain; the optional deployment role is unchanged.

Tracking key: `pr47276-base-overlap-source`.

Next action: Finish Base and review; coordinate landing order and archive exact numerical evidence. Keep the optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Two Base H100 shards remain cancelled. A helper request would also retry AMD, Extra, AMD Extra, MLU and skipped Mori AMD; the VLM timeout is not established as transient.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: use scoped Actions access for cancelled Base shards, or justify a PR-wide request after diagnosing the broader failures.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout persists and C is skipped. Inherited matching test/processor blobs and the similar #46820 failure do not substitute for controlled baseline execution. BF16/DCP/LSE and exact-source qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Compare actual baseline execution before classifying the timeout as transient. Preserve BF16 partials and DCP/LSE coverage through review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Current-head Base A9/B1 and AMD A1/B2 jobs now pass, as do AMD JIT unit/benchmark jobs; both workflows remain active and incomplete. The inherited API repair does not resolve full CI or numerical qualification.

Tracking key: `pr47897-architecture-api-ci`.

Next action: Finish active Base/AMD and review without duplicate retries. Preserve paired-row/table/length, gfx950 dispatch/fallback and exact-source constraints; CPU API checks do not establish long-context numerical equivalence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/commit/8a9f215cb422ab507f00537c695ecccf1cfbe329)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: API compatibility repair 8083ca7a is pushed and locally CPU-validated. New-head Base/AMD gates passed and test jobs started, but no complete functional suite has passed. Old-head results and inherited MI355X tests do not qualify the whole new head.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Finish new-head Base/AMD and ARM64/Xeon CI and review. Preserve opt-in behavior, min(length,2048) valid selections, underflow/overflow fallback and boundary ties. Retain source-specific MI355X evidence and the measured 32k regression; no new model/throughput claims.

- [Evidence](https://github.com/sgl-project/sglang/commit/8083ca7ad7fdcaf2388a88196559b870e70e762a)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/attempts/1)

## #33059: Xeon tests import req_types and disaggregation.mode that the installed PR package cannot provide. The job builds the fork branch while running tests from the merge checkout; no source-alignment repair was attempted.

Tracking key: `pr33059-cpu-source-alignment`.

Next action: CI maintainers: align test and installed-package revisions, then validate the failing imports. Avoid unrelated branch refactors or a blind PR-wide retry.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254658/job/125745674783)
