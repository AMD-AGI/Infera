# Pending handoff for the next user session

Updated: **2026-11-05T09:43:30.190001+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base attempt 3 has 26 functional successes: 25 retained plus B1 completing 39/39 files before this cycle. B2 remains runnerless. Full coverage and historical LLaVA recovery remain unestablished; excluded checks are not claimed fixed or passed. Owner read-only hold persists.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: observe read-only. Owner/maintainers: coordinate merge with current-head approval and remaining B2 coverage; preserve MI355X source limits and native MI300 follow-up.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/127010869262)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. Current-head approval adds no MI300 test evidence. This is historical evidence/review follow-up, not a new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 coverage alongside source-specific MI355X evidence and approval. Worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Attempt 6 retains 37 successes with B2 queued and unexecuted. Historical attempt-5 FDFO failure remains explained by unchanged-path exclusion, matching pre-fix main and merged correction; it is not a passing result. Attempt-5 deferral no longer matches, but owner hold persists. Socket review, historical LLaVA limits and unpublished integration remain.

Tracking key: `pr46820-base-review`.

Next action: Owner/maintainers: assess merge using the incident report and existing approval. Worker: observe read-only; no retry or publication until the owner lifts the hold.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/6)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/126989078017)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41542824410/job/125611431950)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)

## #46830: Maintainer head a349c3b9 preserves the fixture fix, shortened comments and resolved thread. Base has 13 functional successes, one B shard executing tests and eleven runnerless; AMD has eleven successful stages, one executing tests and one queued. Maintainer review and historical source qualifications remain; COMMENTED is not approval.

Tracking key: `pr46830-base-review`.

Next action: Worker: follow active Base/AMD; use the helper for unexplained terminal failures when eligible. Maintainers: complete DSA deadlock/consumer review and retain historical source qualifications. Preserve the maintainer merge, fixture fix and resolved comment thread. Historical Qwen TTFT is not a DSA repair task or new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820535/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/job/127135692802)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#discussion_r4477545713)

## #45524: Base VLM/AWQ failures and fail-fast siblings persist. AMD PP logged KVTransferError; MLA startup exited -9. CPU fork_worker_preload revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, inspect MLA exit -9, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base jobs remain cancelled without execution; mergeable=false/DIRTY confirms main conflict. Source-specific 9763ef56 validation does not qualify current head/image. Excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Attempt 4 retains 25 functional and six JIT successes; B1/B2 remain runnerless. Review, #45307 overlap, numerical qualifications and source/run archiving remain. Optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Follow B1/B2 without duplication. Maintainers/yihou: finish review, coordinate #45307 landing order and archive exact source/run evidence; preserve numerical limits and optional deployment role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6651481153)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41981310976)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Attempt 4 is now failed, retaining 37 Base and six JIT successes. B2 passed six files then failed FDFO at 238.300/236.501 versus >250, leaving 31 files unexecuted. Verified CUDA FlashInfer exclusion, identical failing main test and merged >200 correction explain this failure; they do not establish complete coverage or a passing rerun.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Owner: use the FDFO evidence for maintainer merge assessment, explicitly disclosing 31 unexecuted files and the red Base result. Kernel maintainer/yihou: reconcile scaled-KV/split-K conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Retain review, exact source/run mapping and combined-stack qualifications. Worker: prioritize other unfinished Base runs; reassess changed heads/attempts rather than creating an indefinite hold.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127051341962)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41542824410/job/125611431950)
- [Evidence](https://github.com/sgl-project/sglang/pull/47989)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)
- [Evidence](https://github.com/sgl-project/sglang/blob/a5d8b3af95d7915910547073b1771825981ccb98/test/registered/dllm/test_dllm_batching_fdfo.py)
- [Evidence](https://github.com/sgl-project/sglang/blob/77e20bd8d5b132a1fcc57477593752cda87de6cc/test/registered/dllm/test_dllm_batching_fdfo.py)
- [Evidence](https://github.com/sgl-project/sglang/blob/a5d8b3af95d7915910547073b1771825981ccb98/python/sglang/srt/models/llada2.py#L1015)

## #47278: Inherited AMD attempt 2 passed 18 stage jobs, 14 retained; C passed 3/3 files. BF16/DCP/LSE review, source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou: finish BF16 partials and DCP/LSE review and source/run archiving. Preserve passing AMD evidence; it does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional and six JIT jobs. AMD MIOpen database-access and PP KVTransferError failures and Xeon revision mismatch remain in shared-import scope; review/source/numerical qualifications remain.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair MIOpen access, compare PP baselines and align Xeon revisions. Preserve passing Base and separate review/source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Attempt 3 retains 19 Base and three JIT successes; three B and three JIT jobs remain runnerless. AMD PP B remains queued. Xeon assertion/missing-module failures retain concrete installed-package versus tested-source mismatch. Review, provenance, correctness and performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Follow active Base/AMD. CI maintainers: align Xeon revisions. Maintainers/yihou: finish review; preserve opt-in, min(length,2048) valid selections, exact fallback, valid ties, source mapping, measured 32k regression and MI355X evidence limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 3 failed test_sharded_padded_cp_kv_layout_refuses_unified_memory: expected ValueError not raised. A cancelled sibling lacked req_types/disaggregation.mode. Logs install the PR branch but run merge ed0afb91 tests; the new server-args test is absent from head 8083ca7a.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same commit, then recover Xeon. Avoid unrelated module/parser backports and preserve active Base. The red check does not demonstrate top-k causality or establish a new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47278: Fresh mergeable=false/DIRTY confirms the conflict. Prior source audit identified scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts; numerical resolutions remain unqualified. No kernel edit occurred.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE paths, then validate affected numerical cases before rewriting. Passing AMD on the old head does not validate resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/493c857736725a402fa42e1adc3f7e67ff300375...c1240a9fa3cd8cd02040c2007e719ef0be8a8816)

## #46820: Signed integration 3af3189a86ac97eae65c13f7fceb82b0fe52fa44 remains unpublished. Active publication blocker is the owner read-only hold, not historical OAuth rejection; earlier evidence verified existing SSH as dorado269. No push or authentication change occurred.

Tracking key: `pr46820-upstream-integration-permission`.

Next action: Preserve integration. Only the owner can lift the hold; expiry/new head/attempt does not lift it. Later authorized publication requires fresh ownership/head checks, an exact lease and new CI.

- [Evidence](https://github.com/sgl-project/sglang/pull/46820)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)
