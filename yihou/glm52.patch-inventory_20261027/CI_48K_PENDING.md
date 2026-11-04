# Pending handoff for the next user session

Updated: **2026-11-04T01:12:37.625462+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base attempt 3 remains failed: 37 successes (20 retained, 17 executed), plus dLLM speed 224.739/224.875 versus >250. Exact-source checks confirm executed merge 456e2e3486 uses >250 while current main uses >200. Base-only scope remains; excluded checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: CI owners: reconcile the executed and upstream test revisions before integration/retry; do not weaken tests or assume a transient cause. Native MI300 reviewer coverage remains a separate pending item.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41423254776/job/125745684582)
- [Evidence](https://github.com/sgl-project/sglang/blob/d091f25872bfc9cfa3dd3515e44c754008dab66a/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

## #33059: Native MI300 DSA/padded-row reviewer coverage remains unestablished. Unrelated MI300 collective tests do not supply it.

Tracking key: `pr33059-native-mi300-review`.

Next action: Yihou/reviewers: supply native MI300 coverage using allocated hardware if needed and finish review; preserve Ready state.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)

## #46820: Base remains unsuccessful: 36 retained successes, one VLM timeout and one health-check cascade with tests skipped. Shared-code AMD/Xeon/ARM64 pass; review and controlled baseline diagnosis remain.

Tracking key: `pr46820-base-review`.

Next action: CI owners: compare VLM preprocessing/warmup with baseline execution before retrying; recover the fail-fast sibling. Maintainers: complete shared-code review and required CI. Preserve CPU socket-contract coverage without adding GPU or throughput qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869472)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/125750869468)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/2)

## #46830: Attempt 3, triggered by RolaoDenthu before this cycle, retains 37 successes and queues one H100 shard without a runner or steps. No new functional success. Attempt 2 failed dLLM speed 235.376/236.910 versus >250; current main uses >200. Source reconciliation and review remain; a queued retry is not resolution.

Tracking key: `pr46830-base-review`.

Next action: Let the existing attempt 3 run; do not duplicate it. CI owners: inspect its executed source and result against the prior >250 assertion/current main >200 discrepancy. Maintainers: finish review and preserve historical DSA deadlock/consumer evidence; do not weaken tests or infer performance from queueing.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/job/125761043114)
- [Evidence](https://github.com/sgl-project/sglang/blob/d091f25872bfc9cfa3dd3515e44c754008dab66a/test/registered/dllm/test_dllm_batching_fdfo.py#L95)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467026948/attempts/3)

## #45524: Base has 35 attempt-2 successes, one queued shard, AWQ timeout and two health-check failures with tests skipped. AMD has 12 stage successes, one running and one queued, PP GSM8K auxiliary-data/KV-transfer failure and MLA compiler/watchdog failure. CPU fork_worker_preload mismatch persists; historical-source equivalence remains uncertified.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: diagnose AWQ timeout, PP auxiliary-data/KV-transfer failure, AMD MLA compiler/watchdog failure and CPU test/package mismatch against matching baselines; recover skipped coverage and finish review. Preserve the frozen PP producer patch; current-PR equivalence remains uncertified. Worker stays read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190843)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541289)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758584/job/125761894917)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190799)

## #46918: Two Base H100 shards remain cancelled before execution. Source-specific 9763ef56 validation does not qualify the entire current head or integrated image. AMD transfer CI was previously removed from this mixed finding under Base-only scope, without claiming repair.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: recover Base coverage, finish review and validate image integration. Preserve source-specific MI355X evidence and the frozen workaround until reviewed integration.

- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Base remains failed: 37 successes (36 retained, one executed in attempt 2), plus dLLM speed 222.256/222.160 versus >250. Fresh main uses >200; source reconciliation, review, #45307 overlap coordination and exact source/run archiving remain. Optional fused indexer stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: CI owners: reconcile the executed >250 assertion with upstream main >200 before integration/retry; do not weaken tests or assume transience. Maintainers: finish review, coordinate overlapping #45307 and rebase whichever lands second. Archive exact source/run and numerical evidence; keep the optional fused-indexer path disabled in frozen llr37.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467493468/job/125765401572)
- [Evidence](https://github.com/sgl-project/sglang/blob/d091f25872bfc9cfa3dd3515e44c754008dab66a/test/registered/dllm/test_dllm_batching_fdfo.py#L95)

## #47278: Two Base H100 shards remain cancelled. The inherited actual_failure_requires_triage helper refusal still matches the head, attempt and scoped AMD stage-B failure. Its recorded scope was six PR-wide workflows. No new helper request or retry was sent.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Maintainer: recover Base with scoped Actions access, or triage the AMD failure before another helper evaluation. Do not bypass the recorded refusal.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)

## #47278: AMD VLM preprocessing/warmup timeout and skipped C remain in scope because shared split-K/reduce interfaces serve other decode/DCP callers. No controlled baseline establishes a transient cause. BF16/DCP/LSE and source-mapping constraints remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: CI owners: compare actual baseline execution. Maintainers/yihou: preserve BF16 partials and DCP/LSE coverage through review, exact source/run archiving and combined-stack qualification.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/125359813491)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278)

## #47897: Base attempt 2 has 26 successes plus five JIT units and one benchmark passed; one Base shard queues without execution. AMD has 15 stage successes, one queued, the existing MIOpen database failure and a newly completed PP GSM8K auxiliary-data/KV-transfer failure ending in HTTP 500 and KeyError: answer. Xeon revision mismatch remains within shared-import scope. Complete CI and numerical qualifications remain pending.

Tracking key: `pr47897-architecture-api-ci`.

Next action: Allow the queued Base/AMD shards to run without duplicate retries. CI owners: compare AMD PP auxiliary-data transfer failure with matching baseline execution, investigate MIOpen database access and align Xeon test/package revisions; a similar symptom on another PR does not establish transience. Maintainers: finish review. Preserve unequal paired-row/table/length coverage, gfx950 dispatch/fallback gates and exact source/run archiving; short-context GSM8K is not long-context numerical qualification.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage through integration; reviewers assess remaining long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base attempt 1 is now completed/failure: 19 successful Base jobs, one VLM startup/preprocessing failure and two health-check cascades with tests skipped; C skipped. The last queued shard ran no functional tests. Three JIT units passed; two units and the benchmark were blocked by health checks. AMD has 15 stage successes, one queued, MIOpen failure and PP GSM8K auxiliary-data/KV-transfer failure. Xeon source mismatch persists; inherited MI355X evidence does not qualify the whole head.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: CI owners: compare Base VLM startup/preprocessing and AMD PP auxiliary-data transfer failures with matching baselines; recover skipped Base/JIT coverage, investigate MIOpen access and align Xeon revisions. Let the queued AMD shard run; do not assume transience or issue a blind PR-wide retry. Maintainers: finish review. Preserve opt-in behavior, exactly min(length,2048) selections, underflow/overflow fallback, valid boundary ties, exact source/run archiving, the measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248511)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248530)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125893452204)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125854248810)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991686)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/125860057676)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/125868991527)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/126316940864)

## #47897: Xeon test_server_args.py imports _paged_experts_num_resident from the installed fork package. Inherited source comparison identifies a merge-checkout/package revision mismatch, separate from the repaired architecture API; the matching failure persists.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed fork package. The cancelled sibling also failed imports of sglang.srt.managers.req_types and sglang.srt.disaggregation.mode. These matching logs retain the inherited merge-checkout/package revision mismatch; the architecture API repair does not resolve it.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then rerun the affected Xeon shard. Avoid unrelated parser backports or blind PR-wide retries.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853413002)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/125853412984)
