# Pending handoff for the next user session

Updated: **2026-11-06T10:40:58.464210+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 22 functional successes, 15 retained; one B shard queues without steps. C unexecuted and JIT skipped. Full current-head Base and review remain required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker follows B/C coverage. Maintainers complete review; preserve native MI300/FlashInfer historical follow-up and source-specific MI355X limits without introducing a hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 24 functional successes, 15 retained. H100 C fails Mooncake JIT exit 127; PP passed, Mooncake setup failed and Eagle DP never ran. B200 C failed its sibling-health gate. One B and one C shard queue; AMD has three queued jobs. Disabled KV events bypass the changed send. Main test/installer blobs match; compiler stderr/root cause remains unknown.

Tracking key: `pr46820-base-review`.

Next action: Worker requests owner_requested recovery once terminal and eligible. If repeated, CI/Mooncake owners capture compiler stderr and verify dependencies. Complete full Base, shared-platform coverage and socket review; preserve exact-source/default-unbounded-HWM limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127819512164)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127819512146)

## #46830: Base has 21 functional successes and the FlashInfer graph mismatch: 1/6 elements differ, maximum absolute difference 43; ten files passed, one failed, 25/36 unexecuted. One B shard queues; C/JIT unexecuted. AMD has three queued jobs. Main test/runner blobs match; no established repair. #47276 repeats the same mismatch. B200/source/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Worker requests owner_requested recovery once terminal and eligible. If repeated, CI/FlashInfer owners capture eager/graph outputs. Preserve fixture/maintainer commits and complete DSA consumer/deadlock and CUDA/ROCm review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127685657167)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704234)

## #45524: Base has 20 functional successes and two queued B shards; C/JIT unexecuted. AMD has 11 stage successes, one exit-255 failure and one queued; fresh log retrieval remains HTTP 404. ARM64/Xeon SPR passed; GNR queues. Historical AWQ timeout with 27/30 files unexecuted, PP/MLA/health-check failures, CPU fork_worker_preload revision mismatch and source equivalence remain pending reconciliation.

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

## #47276: Base has 20 functional successes, none retained from attempt 1. Newly inspected FlashInfer mismatch leaves 25/36 files unexecuted after ten passes and one failure. Two B shards queue; C unexecuted. JIT has five executed unit suites, one benchmark success, one empty job and one queued benchmark. Optional deployment, overlap and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows queues and requests owner_requested recovery once terminal and eligible. CI/FlashInfer owners investigate repeated mismatch. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127759704234)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Annotation names github-actions[bot]; policy unknown. Required-main failure and historical 31 unexecuted FDFO-shard files remain. Helper refused required_main_missing at 10:32:33 UTC; no reservation, request, handler or new attempt.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical validation of inherited kernel conflicts, preserving BF16 partials and DCP/LSE. Resume recovery after safe integration; archive exact source/runs and retain combined-stack limits.

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

## #47872: Base has 20 functional successes, ten retained; FlashInfer mismatch leaves 25/36 files unexecuted. Two B shards queue; C unexecuted. JIT has five executed unit suites, two benchmarks and one empty job, four conclusions retained. AMD passed all 17 stage jobs before this cycle; ARM64/Xeon passed. Main conflict and source/long-context review remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker requests owner_requested recovery once terminal and eligible. CI/FlashInfer owners investigate mismatch. Yihou/maintainers reconcile main conflict safely and finish review/source mapping, preserving opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962144)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918/job/127590668430)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required-main directive 50322ac7ba85df18111ec79d18a4c5c554df60c9 persists despite issue 21065 being closed; fresh ancestry comparison is diverged. Helper refused before reservation/comment. Live mergeability remains conflicting; inherited scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts require numerical validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main and kernel conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
