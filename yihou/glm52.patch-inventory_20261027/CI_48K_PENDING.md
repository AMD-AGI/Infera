# Pending handoff for the next user session

Updated: **2026-11-05T07:42:09.260624+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Base attempt 3 retains 25 functional successes. B1 is executing Run test since 07:20:06 UTC; B2 remains runnerless. Full coverage and historical LLaVA recovery are not established. Excluded checks are not claimed fixed or passed.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: follow running B1 and queued B2; diagnose/recover unexplained terminal failures when eligible. Maintainers: assess merge with current-head approval and remaining coverage. Preserve source-specific MI355X evidence and native MI300 follow-up, not an established new hardware gate. If LLaVA repeats, obtain result-wait stacks and runner worker-count/quota/memory.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773699215/job/127010869262)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. Current-head approval confirms no non-HIP impact but adds no MI300 test evidence. This is historical evidence/review follow-up, not an established new hardware gate.

Tracking key: `pr33059-native-mi300-review`.

Next action: Maintainers/yihou: assess native MI300 coverage alongside source-specific MI355X evidence and finish review. Allocate hardware only for a concrete coverage need.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Base attempt 5 retains 37 successes but fails the inherited FDFO >250 threshold. Disabled KV events bypass the changed send; pre-fix main also failed and merged c927e7c corrects this assertion. Exact-attempt retry is owner-deferred through 12:00 UTC. Validated integration remains unpublished; historical LLaVA recovery and shared review remain.

Tracking key: `pr46820-base-review`.

Next action: Maintainers/yihou: seek merge assessment using the documented unchanged-path exclusion, matching pre-fix main failure and merged correction; retain socket-contract review, existing approval and historical LLaVA uncertainty. Worker: prioritize other unfinished Base runs and reassess at changed head/attempt or expiry. Preserve signed integration 3af3189a86ac97eae65c13f7fceb82b0fe52fa44; later publication needs fresh ownership/head checks, an exact lease and new CI.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/126989078017)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)

## #46830: Base attempt 4 retains 26 successes with B2 runnerless; AMD has 16 successful stage jobs and two queued. New COMMENTED collaborator review requests two shorter comments and upstream maintainer review. An AST/token-equivalent local suggestion is prepared but unpublished; review/source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Worker: follow Base B2 and scoped AMD; recover unexplained terminal failures when eligible. Yihou/maintainer: address the two comment-shortening requests using the prepared suggestion with the next justified head update or at merge, preserving active CI meanwhile. Obtain upstream maintainer DSA deadlock/consumer review and retain source qualifications. Qwen TTFT is not a DSA repair task or new validation gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41773966755/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#pullrequestreview-5781043791)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830#discussion_r4477545713)

## #45524: Base VLM/AWQ failures and fail-fast siblings persist. AMD PP logged KVTransferError; MLA startup exited -9. CPU fork_worker_preload revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare failing baselines, inspect MLA exit -9, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base jobs remain cancelled without execution; fresh mergeable=false/DIRTY confirms main conflict. Source-specific 9763ef56 validation does not qualify the current head/image. Excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Previously accepted PR-wide recovery remains at Base attempt 4 with 25 retained functional successes and six retained JIT results. B1/B2 remain runnerless without steps. Review, #45307 overlap, numerical qualifications and source/run archiving remain; optional path stays disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: follow B1/B2 without duplication. The local hourly interval elapsed at 07:34:18.946155 UTC, but this Base remains active; use the next eligible terminal failure after priority PRs. Maintainers/yihou: finish review, coordinate #45307 landing order and archive exact source/run evidence; retain numerical limits and optional disabled llr37 role.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6651481153)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41981310976)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)

## #47278: Previously accepted recovery produced Base attempt 4, retaining 37 Base successes and six JIT results. B2 remains runnerless without steps. Requeueing does not establish coverage or recovery.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Follow active B2 without duplication. Diagnose a new terminal failure if one occurs; later helper recovery remains subject to the author-wide interval. Incidental PR-wide retries do not expand diagnostic scope.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6649758056)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41968562635)

## #47278: AMD attempt 2 passed 18 stage jobs, 14 retained; C passed 3/3 files before this cycle. BF16/DCP/LSE review, exact source/run archiving and combined-stack qualifications remain.

Tracking key: `pr47278-amd-validation-source`.

Next action: Maintainers/yihou: finish BF16 partials and DCP/LSE review and source/run archiving. Preserve passing AMD evidence at this head; it does not qualify a rewritten kernel or combined deployment.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616407/job/126709688128)

## #47897: Base passed 27 functional jobs and six JIT jobs. AMD MIOpen database-access and PP KVTransferError failures and Xeon revision mismatch remain in shared-import scope; review/source/numerical qualifications remain.

Tracking key: `pr47897-architecture-api-ci`.

Next action: CI maintainers: repair MIOpen access, compare PP transfer baselines and align Xeon revisions. Preserve passing Base and separate review/source/numerical qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317751/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125877749297)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317739/job/125844943832)

## #47897: Raw run-to-source archival mapping and long-context numerical qualification remain. Historical short-context GSM8K alone is insufficient.

Tracking key: `pr47897-source-long-context`.

Next action: Archive exact source/run evidence and preserve unequal paired-row, dispatch and fallback coverage; reviewers assess long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897)

## #47872: Base attempt 3 retains 19 Base and three JIT successes; three B and three JIT jobs remain runnerless. AMD PP B remains queued. Xeon attempt 3 failed with server-args assertion and missing-module evidence; merge tests versus installed PR-branch package remains the concrete source mismatch. Review, provenance, correctness and performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Follow active Base/AMD. CI maintainers: align installed/tested Xeon revisions. Maintainers/yihou: finish review; preserve opt-in, min(length,2048) valid selections, exact underflow/overflow fallback, valid ties, source/run mapping, measured 32k regression and source-specific MI355X limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47897: Xeon test_server_args.py cannot import _paged_experts_num_resident from the installed package. Checkout/package mismatch remains separate from the repaired architecture API.

Tracking key: `pr47897-cpu-source-alignment`.

Next action: CI maintainers: build/install and test the same revision, then recover the shard. Avoid unrelated parser backports.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41601317546/job/125839112259)

## #47872: Xeon attempt 3 failed test_sharded_padded_cp_kv_layout_refuses_unified_memory: expected ValueError not raised. A cancelled sibling executed test_io_struct.py/test_scheduler_pause_generation.py and lacked req_types/disaggregation.mode. Logs install the PR branch but run merge ed0afb91 tests; the new server-args test is absent from PR head 8083ca7a.

Tracking key: `pr47872-cpu-source-alignment`.

Next action: CI maintainers: build/install and run tests at the same commit, then recover Xeon. Avoid unrelated module/parser backports and preserve active Base. The red check does not demonstrate top-k patch causality or establish a new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294696)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/job/127071294654)

## #47278: Fresh mergeable=false/DIRTY confirms the conflict. The prior source audit identified scaled-KV, block sizing, GROUP_W, cache/async-copy and split-K conflicts; those numerical resolutions remain unqualified. No new main-source comparison or kernel edit was performed this cycle.

Tracking key: `pr47278-main-kernel-conflict`.

Next action: Yihou/kernel maintainer: reconcile preserved conflicts, retain BF16 partials and DCP/LSE paths, then validate affected numerical cases before rewriting the branch. Passing AMD on the old head does not validate the resolution.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/493c857736725a402fa42e1adc3f7e67ff300375...c1240a9fa3cd8cd02040c2007e719ef0be8a8816)

## #46820: Historical OAuth workflow-scope rejection is not the active blocker: earlier evidence verified existing SSH as dorado269. Signed integration 3af3189a86ac97eae65c13f7fceb82b0fe52fa44 remains intentionally unpublished under owner priority. No SSH authentication, push or new source validation occurred this cycle.

Tracking key: `pr46820-upstream-integration-permission`.

Next action: Worker: preserve the integration while other unfinished Base runs take priority; reassess at deferral expiry or changed head/attempt. Before later publication, recheck ownership/head and use an exact lease, then verify new CI. No account or workflow permission change is currently requested.

- [Evidence](https://github.com/sgl-project/sglang/pull/46820)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)
