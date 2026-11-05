# Pending handoff for the next user session

Updated: **2026-11-05T18:47:35.702964+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Attempt 2 passed its gate and 24 functional jobs, none retained. Two C jobs completed this cycle; H100 C is executing tests. Three B and two C shards remain runnerless. Full current-head Base remains required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve active CI, then recover eligible terminal failures until full Base passes. Maintainers: complete required review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127451836256)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127451836344)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127451836188)

## #33059: Native MI300 coverage remains unestablished. Approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate. The owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 27 functional successes. B0 failed FlashInfer MLA EAGLE graph comparison, leaving eight of 22 files unexecuted; two B shards remain queued. Direct harness bypasses scheduler metrics, and logged launches disable KV events. Three inspected files match main c03b6b3d; no correction established. Full Base, socket review and default-unbounded-HWM reproduction limits remain.

Tracking key: `pr46820-base-review`.

Next action: Worker: preserve active Base, then use eligible recovery until full Base passes. CI/FlashInfer maintainers: diagnose graph mismatch and supply a verified correction. Maintainers: finish socket-contract/source review. Historical attribution does not waive coverage.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/job/127192592529)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41931046936/job/126928003134)
- [Evidence](https://github.com/sgl-project/sglang/blob/c03b6b3df5fe9aa86c1d56c0cdbb30d7a9158f5c/test/registered/attention/unittests/mla/test_flashinfer.py)

## #46830: Maintainer head and resolved fixture review preserved. Base has 26 functional successes. FlashInfer graph mismatch matches inherited scheduled-main evidence and newly observed #46820 failure. Three inspected files match main c03b6b3d; no correction established. Eight files were unexecuted, C failed health checks without tests, and two B jobs queue. AMD and DSA/cross-platform review remain incomplete.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve active Base; prioritize this oldest-created priority run for eligible recovery once terminal. CI/FlashInfer maintainers: supply a verified correction. Recover missing coverage and finish DSA consumer/deadlock and cross-platform review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127144025689)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127313248691)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41931046936/job/126928003134)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820535/attempts/1)

## #45524: Base VLM/AWQ failures and health-blocked siblings persist. AMD retains PP KVTransferError and MLA exit -9; CPU revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, inspect startup failure, align CPU revisions, recover coverage and finish review. Preserve the frozen producer patch; worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base B shards were cancelled without steps; finish failure reflects cancellation. Annotation names github-actions[bot] without explaining policy. Main remains conflicted; historical 9763ef56 validation does not qualify the current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Attempt 4 failed streaming tool-call parsing, leaving 37/39 B1 files unexecuted; B2 failed health checks without tests. Verified PR-wide recovery produced attempt 5, retaining gate, 25 functional successes, five JIT units and one benchmark. Both retried B shards remain runnerless. No new test execution or proven patch causality.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: preserve attempt 5 and verify B1/B2 execution. Next author slot 19:35:08 UTC, subject to priority PRs and fresh eligibility. Maintainers/yihou: retain optional llr37-disabled role, coordinate #45307, finish numerical review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6666058994)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42085652770)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/127091952486)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/127091952546)

## #47278: Prior attempt 5 failed the old FDFO >250 threshold at 233.83/230.53; six files passed, one failed and 31 were unexecuted. Inherited request/handler reverified. Attempt 6 retains gate, 37 functional successes, five JIT units and one benchmark; B2 is runnerless with zero steps. Main contains the correction, but numerical rebase conflicts remain.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: preserve attempt 6 and verify B2 execution. Yihou/kernel maintainer: reconcile conflicts and integrate the upstream correction with numerical validation preserving BF16 partials and DCP/LSE. Finish review and source/run archiving; existing AMD success does not qualify a rewritten kernel or combined stack.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/6)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127168703264)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6664808469)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42075986020)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained; C passed 3/3 files. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou: finish BF16 partials and DCP/LSE review and source/run archiving. Passing AMD does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base remains successful with 27 functional jobs, five JIT units and one benchmark, all predating this cycle. Scoped AMD MIOpen/PP failures and Xeon revision mismatch persist; review/source/numerical qualifications remain separate.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair database access, compare PP baselines and align CPU revisions. Preserve passing Base and separate review/provenance/long-context qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base has 26 functional successes, 19 retained, plus five JIT units and one benchmark. B1 passed 39/39 files before this cycle; B2 remains runnerless. AMD remains successful; Xeon mismatch and review/source/correctness limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve queued B2 and passing AMD. CI maintainers: align Xeon revisions without unrelated backports or CI-only rebase. Maintainers/yihou: finish review/source mapping; preserve opt-in, min(length,2048), exact underflow/overflow fallback, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/job/127071308840)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/attempts/2)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 3 failed test_sharded_padded_cp_kv_layout_refuses_unified_memory: expected ValueError not raised. A cancelled sibling lacked req_types/disaggregation.mode. Logs install the PR branch but execute merge ed0afb91 tests; the new test is absent from head 8083ca7a.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon. Avoid unrelated backports and preserve active Base. This does not demonstrate top-k causality or establish a new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47278: Fresh queries confirm CONFLICTING/DIRTY. Inherited source audit identified scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts. Numerical resolution remains unqualified and blocks safe integration of the upstream FDFO correction. No kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE, then validate affected numerical cases before rewriting. Passing AMD on the existing head does not validate conflict resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/493c857736725a402fa42e1adc3f7e67ff300375...c1240a9fa3cd8cd02040c2007e719ef0be8a8816)
