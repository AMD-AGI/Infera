# Pending handoff for the next user session

Updated: **2026-11-05T19:12:01.278067+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Gate and 26 functional jobs passed, none retained. Two newly observed C passes occurred before this cycle. Another C job is executing tests; three B shards remain runnerless. Full current-head Base remains required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve active CI, then recover eligible terminal failures until full Base passes. Maintainers: complete review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127451836780)

## #33059: Native MI300 coverage remains unestablished. Existing approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate. Owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 27 functional successes. FlashInfer graph failure left eight files unexecuted; two B shards remain runnerless. Direct harness bypasses metrics and logged launches disable KV events. Relevant source blobs still match main c03b6b3d; no verified correction. Full Base, socket review and default-unbounded-HWM reproduction limits remain.

Tracking key: `pr46820-base-review`.

Next action: Worker: preserve active Base, then recover eligible terminal failures. CI/FlashInfer maintainers: diagnose graph mismatch and supply a verified correction. Maintainers: finish socket-contract/source review; attribution does not waive coverage.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/job/127192592529)
- [Evidence](https://github.com/sgl-project/sglang/blob/c03b6b3df5fe9aa86c1d56c0cdbb30d7a9158f5c/test/registered/attention/unittests/mla/test_flashinfer.py)

## #46830: Maintainer head and resolved fixture review preserved. Base has 26 functional successes. FlashInfer graph mismatch matches inherited scheduled-main and #46820 evidence. Eight B files remain unexecuted, C failed health checks without tests, and two B shards queue. AMD and DSA/cross-platform review remain incomplete.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve active Base; prioritize this oldest-created priority run once terminal and eligible. CI/FlashInfer maintainers: supply a verified correction. Recover missing coverage and finish DSA consumer/deadlock and cross-platform review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127144025689)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127313248691)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41931046936/job/126928003134)

## #45524: Base VLM/AWQ failures and health-blocked siblings persist. AMD retains PP KVTransferError and MLA exit -9; CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, inspect startup failure, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base B shards were cancelled without steps; finish failure reflects cancellation. Cancellation policy remains unknown. Main is conflicted; historical 9763ef56 validation does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Inherited attempt 4 failed streaming tool-call parsing, leaving 37/39 B1 files unexecuted; B2 failed health checks without tests. Attempt 5 retains gate, 25 functional successes, five JIT units and one benchmark. B1/B2 remain runnerless; no new tests or proven patch causality.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: preserve attempt 5 and verify execution. Maintainers/yihou: retain optional llr37-disabled role, coordinate #45307, finish numerical review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6666058994)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42085652770)

## #47278: Inherited attempt 5 failed old FDFO >250 threshold at 233.83/230.53; 31 files were unexecuted. Attempt 6 retains gate, 37 functional successes, five JIT units and one benchmark; B2 has zero steps. Main contains correction, but numerical rebase conflicts remain.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: preserve attempt 6. Yihou/kernel maintainer: reconcile conflicts and integrate upstream correction with numerical validation preserving BF16 partials and DCP/LSE. Finish review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/6)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127168703264)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained; C passed 3/3 files. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou: finish BF16 partials and DCP/LSE review and archive source/run evidence. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, five JIT units and one benchmark, all predating this cycle. Scoped AMD MIOpen/PP failures and Xeon revision mismatch persist; review/source/numerical qualifications remain separate.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review/provenance/long-context qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 26 functional successes, 19 retained, plus five JIT units and one benchmark. B1 passed 39/39 files before this cycle; B2 remains runnerless. AMD passed; Xeon mismatch and review/source/correctness limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve queued B2 and passing AMD. CI maintainers: align Xeon revisions. Maintainers/yihou: finish review/source mapping; preserve opt-in, min(length,2048), exact fallbacks, valid ties, 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/127071308840)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/attempts/2)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from installed package. Checkout/package mismatch remains separate from repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 3 failed test_sharded_padded_cp_kv_layout_refuses_unified_memory: expected ValueError not raised. Cancelled sibling lacked req_types/disaggregation.mode. Logs install PR branch but execute merge ed0afb91 tests; new test is absent from head 8083ca7a.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon. Avoid unrelated backports; preserve active Base. This does not establish top-k causality or a new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47278: Fresh queries confirm CONFLICTING/DIRTY. Inherited source audit identified scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts. Numerical resolution remains unqualified and blocks safe FDFO correction integration. No kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE, and validate affected numerical cases before rewriting. Existing AMD success does not validate conflict resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/493c857736725a402fa42e1adc3f7e67ff300375...c1240a9fa3cd8cd02040c2007e719ef0be8a8816)
