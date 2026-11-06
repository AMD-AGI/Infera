# Pending handoff for the next user session

Updated: **2026-11-06T11:42:27.479143+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 26 functional successes (A10/B12/C4), 15 retained. B300 C passed 1/1 files during this cycle. One B and one C shard queue without steps; JIT skipped. Full current-head Base and review remain required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker follows remaining coverage and recovers terminal failures when eligible, no earlier than 12:32:25 UTC under the current hourly interval. Maintainers complete review. Preserve native MI300/FlashInfer historical follow-up and source-specific MI355X limits without a new hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/job/127845066894)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base retains 24 functional successes; H100 C still fails Mooncake JIT with exit_code=127. PP passed, Mooncake setup failed, Eagle DP never ran. B200/GB300 C fail gates without tests; one B shard queues. AMD has two queued jobs and one executing tests. Disabled KV events bypass the changed send. Identical test/installer blobs passed on #33059; compiler stderr/environment equivalence remain unknown.

Tracking key: `pr46820-base-review`.

Next action: Worker retries when terminal and eligible, no earlier than 12:32:25 UTC under the current hourly interval. CI/Mooncake owners capture compiler stderr and check nvcc/dependencies if repeated. Complete full Base, shared-platform coverage and socket review; retain exact-source/default-unbounded-HWM limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127819512164)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127819512036)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/job/127845066965)

## #46830: Recovery request and handler succeeded; Base attempt 3 retains 21 functional successes and queues two B shards without new test steps. Attempt 2 FlashInfer graph mismatch (1/6 elements, max difference 43) left 25/36 files unexecuted; another B shard failed only a sibling-health gate. C/JIT unexecuted; AMD has three queued jobs. No established upstream repair. B200/source/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Worker follows attempt 3 without duplication. If repeated, CI/FlashInfer owners capture eager/graph outputs for runner_eagle_draft_decode_mla_cuda_graph_chain, topk=1. Complete DSA consumer/deadlock and CUDA/ROCm review, retaining B200 skipped-coverage/source limits and maintainer/fixture changes.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6683838515)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42211008978)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127685657167)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127685657236)

## #45524: Base has 20 functional successes and two queued B shards; C/JIT unexecuted. AMD has 11 stage successes, one exit-255 failure and one queued; log retrieval remains HTTP 404. ARM64/Xeon SPR passed; GNR queues. Historical AWQ timeout with 27/30 files unexecuted, PP/MLA/health-check failures, CPU fork_worker_preload revision mismatch and source equivalence remain pending.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers obtain AMD runner logs, recover coverage and complete review/source-equivalence assessment. Preserve historical findings and frozen producer patch. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards remain cancelled without steps after 36 functional successes. Annotation names github-actions[bot]; cancellation policy remains unknown. Finish failed on cancellation; main conflict persists. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile main conflict, recover cancelled shards, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/job/125361049407)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base has 20 functional successes, none retained. FlashInfer mismatch leaves 25/36 files unexecuted after ten passes and one failure. Two B shards queue; C unexecuted. JIT has five executed unit suites, one successful benchmark, one empty job and one benchmark blocked by a sibling-health gate. Optional deployment, overlap and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker retries once terminal and eligible after actionable priority PRs; current hourly interval ends no earlier than 12:32:25 UTC. CI/FlashInfer owners capture eager/graph outputs if repeated. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704234)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127755971622)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Annotation names github-actions[bot]; policy unknown. Required main is still absent; historical 31 unexecuted FDFO-shard files remain. The prior 11:02 UTC helper refusal is inherited. No new request this cycle; the hourly slot was used for priority #46830.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical validation of inherited kernel conflicts, preserving BF16 partials and DCP/LSE. Resume recovery after safe integration and retry eligibility; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)

## #47278: Inherited AMD attempt 2 passed 18 executed stage jobs, 14 retained. BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional jobs, four executed JIT unit suites and one benchmark; another JIT success ran 0/0. Identity-verified inherited AMD MIOpen database/PP failures and Xeon checkout/package mismatch remain; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row/table-length, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 20 functional successes, ten retained. FlashInfer mismatch left 25/36 files unexecuted; a second B shard failed only its sibling-health gate. One B shard queues; C unexecuted. JIT has five executed unit suites, two benchmarks and one empty job, four conclusions retained. AMD passed 17 stage jobs; ARM64/Xeon passed. Main conflict and source/long-context review remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker retries once terminal and eligible after actionable priority PRs. CI/FlashInfer owners investigate mismatch. Yihou/maintainers reconcile main conflict safely and finish review/source mapping, preserving opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962144)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962110)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918/job/127590668430)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required-main directive 50322ac7ba85df18111ec79d18a4c5c554df60c9 persists despite issue 21065 being closed; fresh ancestry remains diverged and main mergeability conflicting. Prior helper refusal retained. Inherited scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts require numerical validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main and kernel conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration and retry eligibility; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
