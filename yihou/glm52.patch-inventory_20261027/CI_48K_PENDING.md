# Pending handoff for the next user session

Updated: **2026-11-05T10:43:00.899606+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: New head 31d19a21 includes requested main integration, but Base attempt 1 failed only the author cooldown. Zero functional tests ran. The owner hold is lifted; fresh full Base remains required. Prior-head passes and historical LLaVA findings do not qualify this head.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: after 12:18:15 UTC, recheck head/attempt/account policy and use stale_gate if eligible; verify request, handler and actual tests. Maintainers require full current-head Base before merge.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127185382203)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. Approval adds no MI300 evidence; this remains historical evidence follow-up, not a new hardware gate. The owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues authorized Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Published d796f865 integrates main 732d34b9 and the FDFO correction. Fresh Base has 10 successful A jobs, including four socket tests; B/C coverage remains incomplete. AMD/Xeon are active, ARM64 passed. Historical LLaVA/FDFO limits and socket review remain; historical attribution does not waive full Base.

Tracking key: `pr46820-base-review`.

Next action: Worker: preserve active attempts and recover eligible later failures until Base passes fully. Maintainers: retain existing approval and socket-contract/source qualifications; old publication and hold blockers are resolved.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/job/127185313427)

## #46830: Maintainer head a349c3b9 preserves the fixture fix and resolved thread. Base has 17 functional successes, two B shards executing and six queued. AMD has 14 successful stages and three queued. COMMENTED reviews are not approval; source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve active Base/AMD and recover eligible terminal failures. Maintainers: complete DSA deadlock/consumer review. Do not turn historical Qwen TTFT into a DSA repair or new author gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820535/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830/files)

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

## #47278: Base attempt 5 retains 37 functional and six JIT successes; B2 remains runnerless without executed steps. The inherited request/handler succeeded but no new functional recovery is established. Historical FDFO failure left 31 files unexecuted; matching upstream correction does not establish complete coverage.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Preserve queued B2 and inherited FDFO attribution. Yihou/kernel maintainer: resolve scaled-KV/split-K conflicts with numerical validation, preserving BF16 partials and DCP/LSE. Retain review, exact source/run mapping and combined-stack qualifications.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/attempts/5)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6654764283)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42004732891)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467616894/job/127051341962)
- [Evidence](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c)

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

## #47872: Base attempt 3 retains 19 functional and three JIT successes. B0 and two JIT jobs execute tests; two B shards and one JIT unit remain queued. AMD attempt 2 now passed 18 stage jobs, 16 retained, with PP passing 4/4 files this cycle. Xeon revision mismatch, review, provenance, correctness and performance limits remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Preserve active Base and passing AMD. CI maintainers: align Xeon package/test revisions. Maintainers/yihou: finish review while retaining opt-in, exact selection/fallback/tie contracts, source mapping, measured 32k regression and source-specific hardware limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114610/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114164/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41617114096/job/126847789904)

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
