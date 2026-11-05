# Pending handoff for the next user session

Updated: **2026-11-05T17:11:50.016798+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Attempt 2 passed its gate and 21 functional jobs, none retained from attempt 1. Four B jobs remain runnerless; C has not started. Full current-head Base remains required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve active CI, then recover eligible terminal failures until full Base passes. Maintainers: complete required review.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/2)

## #33059: Native MI300 coverage remains unestablished. Approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate. The owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base has 27 functional successes and three runnerless B jobs; all five C jobs passed. AMD has 14 successful stages and three queued; Xeon/ARM64 passed actual tests. Full Base and upstream review remain required. Inherited socket-contract evidence did not reproduce the historical hang with current default unbounded HWM.

Tracking key: `pr46820-base-review`.

Next action: Worker: recover eligible terminal failures until full Base passes. Maintainers: finish socket-contract/source review. Verify KV-events launch settings before attributing VLM failures.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/job/127373715710)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583151/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583066/attempts/1)

## #46830: Maintainer head and resolved fixture review are preserved. Base has 26 functional successes. H100 FlashInfer graph failure matches inherited scheduled-main evidence: 1/6 mismatch, difference 43. Three inspected files still match main e838360f; no merged correction established. Eight files were unexecuted; C failed fast without tests; two B jobs queue. AMD and cross-platform/DSA review remain incomplete.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve active Base and use eligible recovery once terminal. CI/FlashInfer maintainers: diagnose the matching upstream failure and supply a verified correction. Finish DSA consumer/deadlock and cross-platform review without weakening assertions.

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

## #47276: Attempt 4 retains 25 functional successes, five JIT units and one benchmark. B1 failed streaming tool-call parsing; one file passed, one failed and 37/39 were unexecuted. CUDA bypasses the gfx950 changes. B2 remains runnerless; root cause is unproved.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: preserve B2, then use eligible recovery after the priority group. Maintainers/yihou: preserve optional llr37-disabled role, coordinate #45307, and finish numerical review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/127091952486)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 5 retains 37 functional successes, five JIT units and one benchmark. B2 now executes Run test, started 16:54:30 UTC before this cycle; no new success yet. Historical FDFO failure left 31 files unexecuted.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: preserve executing B2 and verify final coverage. Yihou/kernel maintainer: reconcile scaled-KV/split-K conflicts with numerical validation, retaining BF16 partials, DCP/LSE and exact source/run qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127168703264)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)

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

## #47872: Base has 25 functional successes, 19 retained, plus five JIT units and one benchmark. B1 executes Run test, started 16:47:41 UTC before this cycle; B2 remains runnerless. AMD remains successful; Xeon mismatch and review/source/correctness limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve active Base and passing AMD. Maintainers/yihou: finish review and source/run mapping; preserve opt-in, min(length,2048), exact fallbacks, valid ties, measured 32k regression and source-specific MI355X limits.

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

## #47278: Fresh queries confirm CONFLICTING/DIRTY. Prior source audit identified scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts; numerical resolutions remain unqualified. No kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE, then validate affected numerical cases before rewriting. Passing AMD on the existing head does not validate conflict resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/493c857736725a402fa42e1adc3f7e67ff300375...c1240a9fa3cd8cd02040c2007e719ef0be8a8816)
