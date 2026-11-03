# Pending handoff for the next user session

Updated: **2026-11-03T07:42:52.254258+00:00**.

These items remain open until supported by resolution evidence. Reporting them to the user does not resolve them.

## #33059: Two Base H100 shards remain queued. AMD has a Kimi MI35X MIOpen database failure and three queued stages.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: CI owners: finish active attempts and investigate the inaccessible database before selecting retry scope. Do not duplicate queued workflows.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254739/job/125747856350)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Successful unrelated MI300 collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply the requested native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: 36 Base successes are retained; one H100 shard runs and one queues. AMD passes, but Base and upstream common-code review remain incomplete.

Tracking key: `pr46820-base-review`.

Next action: CI owners: finish Base. Upstream maintainers: complete review and required checks; retain scoped CPU socket-contract evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820)

## #46830: Two H100 shards remain unexecuted; 36 Base successes are retained. Required aggregate and maintainer review remain.

Tracking key: `pr46830-base-review`.

Next action: CI owners: resolve runner capacity and finish Base. Maintainers: review while preserving historical DSA deadlock evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830)

## #45524: Authorization is resolved, but CI remains active and MI300 test_mla_fp8.py setup ended with server exit -9. Root cause and historical-source equivalence remain unproven.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose the server exit, finish CI and review. Preserve the separate frozen producer-ordering patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)

## #46918: Cancelled Base coverage and AMD PP decode-transfer errors remain. Hardware evidence at 9763ef56 does not qualify the entire current head or integrated image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover cancelled Base coverage, diagnose transfer failures, complete review and validate image integration before retiring the frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734491/job/125360336390)

## #47276: Two H100 shards queue. Review, #45307 overlap coordination and final source/run archiving remain; the optional deployment role is unchanged.

Tracking key: `pr47276-base-overlap-source`.

Next action: Finish Base and review; coordinate which overlapping PR lands first and archive exact numerical evidence. Keep the fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Base lacks two cancelled H100 shards. The helper would also retry AMD and other unsuccessful workflows; unresolved AMD attribution prevents classifying that broad retry as appropriate.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: use scoped Actions access for cancelled Base shards, or diagnose AMD first and establish a justified PR-wide helper request.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout remains unexplained and stage-C is skipped. Exact-source archiving and BF16/DCP/LSE numerical qualifications remain separate from Base.

Tracking key: `pr47278-amd-validation-source`.

Next action: Compare the failing VLM test against the same baseline before repeated retries. Preserve BF16 partials and DCP/LSE coverage through review and source mapping.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: CPU/JIT/AMD failures involve absent get_jit_cuda_arch imports and model-registration fallout. Three JIT jobs remain queued. Integrating current-main architecture cleanup and validating the port remain undone.

Tracking key: `pr47897-architecture-api-ci`.

Next action: Yihou/delegate: reconcile reviewed upstream changes and validate focused CPU imports before another CI attempt; do not blindly rerun the deterministic failure.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41428924194/job/125761436979)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41428923967/job/125761555517)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Repair is published at d07820e8c744116ecb8f3df5495383423fe35f1d, but new-head lint/Base/AMD remain incomplete. Standard local imports are blocked by absent orjson; isolated three-case validation is narrower than full CI. Inherited numerical and performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Finish new-head CI and maintainer review. Preserve opt-in behavior, exact selection count, underflow/overflow and tie coverage. Retain source-specific MI355X evidence and the recorded 32k performance limitation.

- [Evidence](https://github.com/sgl-project/sglang/commit/d07820e8c744116ecb8f3df5495383423fe35f1d)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653398/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41597653098/attempts/1)
