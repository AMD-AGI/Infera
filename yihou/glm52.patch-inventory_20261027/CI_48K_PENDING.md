# Pending handoff for the next user session

Updated: **2026-11-06T10:14:08.019763+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base has 22 functional successes, 15 retained; one B shard queues without steps. C unexecuted and JIT skipped. Full current-head Base and review remain required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker follows remaining B/C coverage. Maintainers complete review; preserve native MI300/FlashInfer historical follow-up and source-specific MI355X limits without introducing a new hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42134452467/job/127716852505)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers assess native MI300 follow-up alongside source-specific MI355X evidence and required review.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 24 functional successes, 15 retained. H100 C fails Mooncake JIT exit 127; PP passed, Mooncake setup failed and Eagle DP never ran. B200 C failed only the sibling-health gate. One B and one C shard queue; AMD has three queued jobs. Failed launch disables KV events and bypasses the changed send. Main test/installer blobs match; compiler stderr/root cause remains unknown. Full Base and socket review remain required.

Tracking key: `pr46820-base-review`.

Next action: Worker requests owner_requested recovery once Base is terminal and eligible. If repeated, CI/Mooncake owners capture compiler stderr and verify runner compiler/dependent tools. Complete shared-platform coverage and socket review; preserve exact-source/default-unbounded-HWM limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127819512164)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42133659395/job/127819512146)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Base has 21 functional successes and the inherited FlashInfer graph mismatch: 1/6 elements differ, maximum absolute difference 43; 10 files passed, one failed, 25/36 unexecuted. One B shard queues; C/JIT unexecuted. AMD has three queued jobs. Main 2fd45ed82 has identical test/runner blobs and only backend import relocation; no established repair. Fixture uses FlashInfer rather than changed DSA. B200/source/deadlock review remains.

Tracking key: `pr46830-base-review`.

Next action: Worker preserves queued coverage and requests owner_requested recovery once terminal and eligible. If repeated, CI/FlashInfer owners capture eager/graph outputs for runner_eagle_draft_decode_mla_cuda_graph_chain, topk=1. Preserve fixture/maintainer commits and complete DSA consumer/deadlock and CUDA/ROCm review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42125086323/job/127685657167)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962144)

## #45524: Base has 20 functional successes and two queued B shards; C/JIT unexecuted. AMD has 11 stage successes, one exit-255 failure and one queued; failed logs remain HTTP 404. ARM64/Xeon SPR passed; GNR queues. Historical AWQ timeout (27/30 files unexecuted), PP/MLA/health-check failures, CPU fork_worker_preload revision mismatch and source equivalence remain pending coverage reconciliation.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers obtain AMD shard 6 runner logs, recover coverage and complete review/source-equivalence assessment. Preserve historical findings and frozen producer patch. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903737/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/job/125766190802)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two B shards remain cancelled without steps after 36 functional successes. Annotation names github-actions[bot], but cancellation policy remains unknown. Finish failed on cancellation; main conflict persists. Validation at 9763ef56 does not qualify this head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers reconcile main conflict, recover cancelled shards, finish review and stock-image integration. Preserve frozen workaround and source-specific MI355X evidence. Worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/job/125361049407)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Base has 20 functional successes, two B shards queued and one executing tests; C unexecuted. JIT has five executed unit suites and one benchmark success, one empty 0/0 job and one queued benchmark. H100 JIT completed 73/73 files during this cycle. Optional deployment, overlap coordination and numerical/source qualifications remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker follows active Base and remaining benchmark. Preserve optional llr37-disabled role, coordinate #45307 overlap, complete numerical review and archive exact source/run evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42126248525/job/127755971674)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 8 cancelled before B2 started; 37 functional successes and six JIT conclusions retained, including one empty job. Annotation names github-actions[bot]; policy unknown. Required-main failure and historical 31 unexecuted FDFO-shard files remain. Helper again refused required_main_missing at 10:04 UTC; no reservation, request, handler or new attempt.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Yihou/kernel maintainer integrates required main with numerical validation of inherited kernel conflicts, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127656329036)

## #47278: Inherited AMD attempt 2 remains successful with 18 executed stage jobs, 14 retained. BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou finish review and archive exact evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, four executed JIT unit suites and one benchmark; another JIT success ran 0/0. Identity-verified inherited AMD MIOpen database/PP failures and Xeon checkout/package revision mismatch remain; ARM64 passed.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact evidence; preserve unequal paired-row/table-length, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 20 functional successes, ten retained; FlashInfer graph mismatch leaves 25/36 files unexecuted after ten passes and one failure. Two B shards queue; C unexecuted. Corrected JIT accounting: five executed unit suites, two benchmark jobs and one empty job; four conclusions retained. AMD has 16 stage successes and one queued; ARM64/Xeon passed. Main conflict, exact-source and long-context review qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker preserves active Base and requests owner_requested recovery once terminal and eligible. CI/FlashInfer owners investigate the mismatch; no HIP top-k repair is established. Yihou/maintainers reconcile main conflict and finish source/review work, preserving opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124401280/job/127735962144)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42124400918/job/127624251848)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains unresolved.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers install and test the same revision, then recover the shard; avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47278: Required-main directive 50322ac7ba85df18111ec79d18a4c5c554df60c9 persists despite issue 21065 being closed; fresh ancestry comparison is diverged. The helper refused before reservation/comment. Live mergeability remains conflicting; inherited scaled-KV, block-sizing, GROUP_W, cache/async-copy and split-K conflicts need numerical validation.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer reconciles required main and these conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Worker resumes recovery after safe integration; archive exact source/runs and retain combined-stack limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/8)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...a5d8b3af95d7915910547073b1771825981ccb98)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
