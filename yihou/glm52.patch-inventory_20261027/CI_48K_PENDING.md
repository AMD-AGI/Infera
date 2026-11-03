# Pending handoff for the next user session

Updated: **2026-11-03T09:42:05.227283+00:00**.

These items remain open until supported by resolution evidence. Reporting them to the user does not resolve them.

## #33059: Two Base H100 shards and three AMD stages remain queued. The Kimi MI35X MIOpen database failure is unchanged; no complete Base/AMD success exists.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: CI owners: finish active attempts and investigate the inaccessible database before choosing retry scope. Do not duplicate queued workflows.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254739/job/125747856350)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated MI300 collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: One H100 shard now fails VLM preprocessing/warmup; one remains queued. The 36 Base successes are retained. AMD passes, but required review remains incomplete despite one approval.

Tracking key: `pr46820-base-review`.

Next action: CI owners: diagnose against the same baseline and finish Base. Maintainers: complete common-code review; preserve scoped CPU socket-contract evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: Two H100 shards remain unexecuted; 36 Base successes are retained. Required aggregate and maintainer review remain.

Tracking key: `pr46830-base-review`.

Next action: CI owners: resolve runner capacity and finish Base. Maintainers: review while preserving historical DSA deadlock evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830)

## #45524: Base progressed to 33 successful A/B jobs with six queued. AMD test_mla_fp8.py still exits -9 during setup; root cause and historical-source equivalence remain unproven.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose the exit, finish CI and review. Preserve the separate frozen producer-ordering patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)

## #46918: Cancelled Base coverage and AMD PP transfer errors remain. Hardware evidence at 9763ef56 does not qualify the entire current head or integrated image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover cancelled coverage, diagnose transfers, complete review and validate image integration before retiring the frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734491/job/125360336390)

## #47276: Two H100 shards queue. Review, #45307 overlap coordination and final source/run archiving remain; the optional deployment role is unchanged.

Tracking key: `pr47276-base-overlap-source`.

Next action: Finish Base and review; coordinate landing order and archive exact numerical evidence. Keep the fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Two Base H100 shards remain cancelled. A helper request would also retry AMD, Extra, AMD Extra, MLU and skipped Mori AMD; the VLM timeout is not established as transient.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: use scoped Actions access for cancelled Base shards, or establish a justified PR-wide helper request after diagnosing the broader failures.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout remains and stage-C is skipped. A similar failure occurs in #46820 CUDA with identical test/processor blobs at both heads and baseline, but no controlled baseline execution establishes cause. BF16/DCP/LSE and exact-source qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Compare actual baseline execution before classifying the timeout as transient. Preserve BF16 partials and DCP/LSE coverage through review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Interactive follow-up 2026-11-03T09:42:05.227283+00:00: current-head lint passed. Base attempt 2 is queued; gate is success. Observed Base A/B/C and separate JIT job states: {"in_progress": 10, "queued": 4, "success": 1}. This is not a passing full Base/AMD or numerical qualification. Other hardware evidence retains its prior scope.

Tracking key: `pr47897-architecture-api-ci`.

Next action: Follow the newly requested CI attempt and inspect actual functional results; do not duplicate queued/running CI. Complete Base/AMD and review. Preserve unequal paired-row/table/length coverage, gfx950 dispatch/fallback gates and exact source/run archiving. CPU API checks do not replace numerical validation; short-context GSM8K does not establish long-context DSA equivalence.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/commit/8a9f215cb422ab507f00537c695ecccf1cfbe329)
- [Evidence](https://github.com/sgl-project/sglang/commit/d0d50da899fd3a682491f874fe4f46ffaf210920)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317500/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/job/125800705843)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897#issuecomment-6600214639)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Interactive follow-up 2026-11-03T09:42:05.227283+00:00: current-head lint passed. Base attempt 2 is queued; gate is success. Observed Base A/B/C and separate JIT job states: {"in_progress": 6, "queued": 4, "success": 5}. This is not a passing full Base/AMD or numerical qualification. Other hardware evidence retains its prior scope.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Let the active attempt finish, inspect real test failures and complete review; do not duplicate queued/running CI. Preserve opt-in behavior, exactly min(length,2048) valid selections, underflow/overflow fallback and boundary ties. Inherited MI355X evidence remains source-specific; retain the measured 32k regression and avoid new model or throughput claims.

- [Evidence](https://github.com/sgl-project/sglang/pull/47872)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653398/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653398/job/125789163937)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653371/job/125788859557)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653098/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872#issuecomment-6600081857)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41611873925)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653398/attempts/2)
