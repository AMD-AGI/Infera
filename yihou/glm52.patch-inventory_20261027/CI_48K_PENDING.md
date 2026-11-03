# Pending handoff for the next user session

Updated: **2026-11-03T11:39:39.254727+00:00**.

These items remain open until supported by resolution evidence. Reporting them to the user does not resolve them.

## #33059: Base has 32 successes, two running jobs and four queued; AMD has 15 stage successes, two queued and the persistent Kimi MIOpen database failure. Neither workflow is complete.

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

## #46830: Base remains at 37 successes: 36 retained and one executed in attempt 2, already observed last cycle. One H100 shard queues without execution; aggregate completion and required review remain.

Tracking key: `pr46830-base-review`.

Next action: CI owners: finish the remaining shard. Maintainers: complete review while preserving historical DSA deadlock evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/job/125761043136)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830)

## #45524: Base has 35 successes, three queued shards and a new test_awq.py 1800-second timeout. AMD MLA setup still hits compilation/watchdog failure; ARM64/Xeon expose fork_worker_preload mismatch. Historical-source equivalence remains unproven.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose AWQ startup/graph-capture timeout, AMD compiler waiting and CPU source mismatch; finish CI and review. Preserve the separate historical PP producer-ordering patch; current-PR equivalence remains uncertified. No worker retry or branch takeover.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758584/job/125761894917)

## #46918: Cancelled Base coverage and AMD PP transfer errors remain. Validation at 9763ef56 does not qualify the entire current head or integrated image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover coverage, diagnose transfers, complete review and validate image integration before retiring the frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734491/job/125360336390)

## #47276: One H100 shard runs and one queues; 36 Base successes are retained. Review, #45307 overlap coordination and exact source/run archiving remain. The fused-indexer path stays optional and disabled in frozen llr37.

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

## #47897: Base now has 16 successes and AMD 14 stage successes, but AMD Kimi fails on MIOpen database access and Xeon has a distinct source mismatch. Both GPU workflows remain incomplete; ARM64 and AMD JIT pass. The inherited API repair does not qualify full CI or numerics.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: investigate MIOpen database access, align Xeon test/package revisions and finish active CI; complete review. Preserve unequal paired-row/table/length coverage, gfx950 dispatch/fallback gates and exact source/run archiving. Inherited CPU repair and short-context GSM8K do not establish long-context numerical equivalence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317704/attempts/2)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Inherited repair 8083ca7a now has passing ARM64 and 13 Base job successes; AMD has 12 stage successes plus JIT unit/benchmark passes. Base/AMD remain active and incomplete, and Xeon fails a separate source mismatch. Prior MI355X evidence does not qualify the whole head.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI maintainers: align Xeon test/package revisions, finish active Base/AMD and review. Preserve opt-in behavior, exactly min(length,2048) valid selections, underflow/overflow fallback and boundary ties. Inherited MI355X evidence remains source-specific; retain the measured 32k regression and avoid new model/throughput claims.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114163/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/commit/8083ca7ad7fdcaf2388a88196559b870e70e762a)

## #33059: Xeon tests import req_types and disaggregation.mode that the installed PR package cannot provide. The job builds the fork branch while running tests from the merge checkout; no source-alignment repair was attempted.

Tracking key: `pr33059-cpu-source-alignment`.

Next action: CI maintainers: align test and installed-package revisions, then validate the failing imports. Avoid unrelated branch refactors or a blind PR-wide retry.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254658/job/125745674783)

## #47897: Xeon test_server_args.py imports _paged_experts_num_resident from the installed fork package. Both logged merge revisions contain the new test/parser; both PR heads contain neither. This is a test/package revision mismatch, separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon test_server_args.py imports _paged_experts_num_resident from the installed fork package. Both logged merge revisions contain the new test/parser; both PR heads contain neither. This is a test/package revision mismatch, separate from the repaired architecture API.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
