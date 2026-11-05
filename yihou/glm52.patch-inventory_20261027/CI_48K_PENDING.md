# Pending handoff for the next user session

Updated: **2026-11-05T16:42:41.250749+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Attempt 2 passed its gate and 21 functional jobs, none retained. Four B jobs remain runnerless; C has not started. Newly observed two-GPU B success passed 9/9 files before this cycle. Full current-head Base remains required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: preserve active CI, then recover eligible terminal failures until full Base passes. Maintainers: complete required review and merge assessment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127254428531)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. Approval supplies no MI300 evidence; this is historical follow-up, not a new hardware gate. The owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Current-head Base has 25 functional successes, one executing H100 C test job and four runnerless jobs. AMD has 14 successful stages and three queued; Xeon runs; ARM64 passed. Full Base and upstream review remain required. Inherited socket-contract evidence did not reproduce the historical hang with current default unbounded HWM.

Tracking key: `pr46820-base-review`.

Next action: Worker: recover eligible terminal failures until full current-head Base passes. Maintainers: finish socket-contract/source review. Verify KV-events launch settings before attributing new VLM failures.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/job/127373716077)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583066/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46820/files)

## #46830: Maintainer head and resolved fixture review are preserved. Base has 26 functional successes. The H100 FlashInfer MLA EAGLE graph failure also occurs on scheduled main c67efab4 with FlashInfer 0.7.2 and identical test/backend blobs: 1/6 mismatch, difference 43. The runner has a stock-runtime adaptation; the three inspected PR files match current main. Root cause/correction remain unknown. Eight of 22 files did not execute, C GB300 failed health check, two B jobs queue, and AMD/review remain incomplete.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve maintainer commits and active Base, then use eligible owner_requested recovery once terminal. CI/FlashInfer maintainers: diagnose the matching upstream failure and provide a verified correction. Maintainers: finish DSA consumer/deadlock and cross-platform review without weakening assertions.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127144025689)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127313248691)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41931046936/job/126928003134)
- [Evidence](https://github.com/sgl-project/sglang/blob/c67efab427ac9d7965123a3cea50bf1d3d899563/test/registered/attention/unittests/mla/test_flashinfer.py)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820535/attempts/1)

## #45524: Base VLM/AWQ failures and health-blocked siblings persist. AMD retains PP KVTransferError and MLA exit -9; CPU fork_worker_preload revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, inspect startup failure, align CPU revisions, recover missing coverage and finish review. Preserve the frozen producer patch; worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base B shards were cancelled without executed steps; finish failure reflects cancellation. One annotation names github-actions[bot] without explaining policy. Main remains conflicted; historical 9763ef56 validation does not qualify the current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Attempt 4 retains 25 functional successes, five JIT units and one benchmark. B1 failed streaming tool-call parsing; one file passed, one failed and 37/39 did not execute. CUDA bypasses the gfx950 changes. B2 is runnerless; root cause remains unproved.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: preserve B2, then use eligible owner_requested recovery after the priority group. Maintainers/yihou: preserve optional llr37-disabled role, coordinate #45307, and finish numerical review and source/run archiving.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/127091952486)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)

## #47278: Attempt 5 retains 37 functional successes, five JIT units and one benchmark; B2 remains runnerless. The inherited request/handler does not establish new test execution. Historical FDFO failure left 31 files unexecuted.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: preserve B2 and coverage gaps. Yihou/kernel maintainer: reconcile scaled-KV/split-K conflicts with numerical validation, retaining BF16 partials, DCP/LSE and exact source/run qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/5)
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

## #47872: Base has 25 functional successes, 19 retained, including all five C jobs. Two B jobs remain runnerless. Five JIT units and one benchmark passed; all completions predate this cycle. AMD remains successful; Xeon mismatch and review/source/correctness limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve active Base and passing AMD. Maintainers/yihou: finish review and source/run mapping; preserve opt-in, min(length,2048), exact fallbacks, valid ties, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/attempts/2)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 3 failed test_sharded_padded_cp_kv_layout_refuses_unified_memory: expected ValueError not raised. A cancelled sibling lacked req_types/disaggregation.mode. Logs install the PR branch but execute merge ed0afb91 tests; the new server-args test is absent from head 8083ca7a.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon. Avoid unrelated backports and preserve active Base. This failure does not demonstrate top-k causality or establish a new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47278: Fresh queries confirm CONFLICTING/DIRTY. Prior source audit identified scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts; numerical resolutions remain unqualified. No kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE, then validate affected numerical cases before rewriting. Passing AMD on the existing head does not validate conflict resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/493c857736725a402fa42e1adc3f7e67ff300375...c1240a9fa3cd8cd02040c2007e719ef0be8a8816)

## #33059: The preceding 16:00 UTC cycle failed at 16:13 UTC during shared ledger publication: remote rejected the push with fatal error in commit_refs_rules. Last confirmed successful publication is the 15:42 UTC cycle. This affects the handoff for all nine PRs, not their source branches; current-cycle publication is unverified.

Tracking key: `ci-watch-publication-failure`.

Next action: Launcher/publisher owner: retry normal publication, preserve the prepared commit and logs, and verify inventory and permanent-Draft tracker publication. Do not infer recovery from the current RUNNING record.
