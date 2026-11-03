# Pending handoff for the next user session

Updated: **2026-11-03T08:17:17.040658+00:00**.

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

## #47897: The removed-API import was repaired at 8a9f215cb422ab507f00537c695ecccf1cfbe329 with isolated CPU validation. New-head Base/AMD are blocked by author cooldown and ran zero functional tests; lint remains running. CI validation is unresolved.

Tracking key: `pr47897-architecture-api-ci`.

Next action: After 09:30:43 UTC, recheck head, cooldown and PR-wide scope before a justified helper request. Complete functional CI and review; do not treat isolated CPU checks as GPU qualification.

- [Evidence](https://github.com/sgl-project/sglang/commit/8a9f215cb422ab507f00537c695ecccf1cfbe329)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317500/attempts/1)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Current-head lint passes, but Base/AMD failed only at the author cooldown gate and all functional suites were skipped. Prior isolated CPU evidence remains narrower than full CI; numerical and performance qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: After 09:30:43 UTC, recheck head and PR-wide scope before a justified helper request. Finish CI/review while preserving opt-in behavior, exact selection count, underflow/overflow and tie coverage, source-specific MI355X evidence and the recorded 32k regression.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653398/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653398/job/125789163937)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653371/job/125788859557)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653098/attempts/1)
