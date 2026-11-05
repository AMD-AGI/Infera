# Pending handoff for the next user session

Updated: **2026-11-05T11:40:39.517649+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #33059: Published 31d19a21 includes requested main integration, but Base attempt 1 remains gate-only failure under the upstream cooldown; zero functional/JIT tests ran. Earliest stale_gate opportunity is 12:18:15 UTC, subject to live policy/account checks. Local hourly interval is already elapsed; full current-head Base is still required.

Tracking key: `pr33059-base-amd-incomplete`.

Next action: Worker: at/after 2026-11-05 12:18:15 UTC freshly check head, attempt, policy and account-wide runs, then use stale_gate if eligible and verify request/handler/tests. Maintainers require full current-head Base before merge. Retain approval, source-specific MI355X limits and native MI300 follow-up as historical evidence context, not a new hardware gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009672020/job/127185382203)

## #33059: Native MI300 DSA/padded-row coverage remains unestablished. Approval adds no MI300 evidence; this remains historical evidence follow-up, not a new hardware gate. The owner hold is lifted.

Tracking key: `pr33059-native-mi300-review`.

Next action: Owner/maintainers: assess native MI300 follow-up alongside source-specific MI355X evidence and approval; worker continues authorized Base recovery.

- [Evidence](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447)
- [Evidence](https://github.com/sgl-project/sglang/pull/33059#pullrequestreview-5780736705)

## #46820: Current-head Base has 14 completed functional successes; B3 executes tests and ten B shards are runnerless. C coverage is absent. Four socket tests are inherited current-head passes; AMD has 13 successful stages, Xeon queued and ARM64 passed. Verified main/FDFO integration and existing approval do not waive full Base or socket-contract/source review.

Tracking key: `pr46820-base-review`.

Next action: Worker: preserve active runs and recover eligible terminal Base failures until full current-head success. Maintainers: complete merge assessment with approval and socket-contract/source qualifications. Historical LLaVA/FDFO attribution does not waive coverage; verify KV-events launch settings before attributing any later VLM failure to this patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42009583307/job/127185313427)

## #46830: Maintainer head a349c3b9 preserves the fixture fix and resolved thread. Base now has 20 functional successes; five B shards are runnerless. AMD has 14 successful stages and three queued. COMMENTED reviews are not approval; DSA consumer review/source qualifications remain.

Tracking key: `pr46830-base-review`.

Next action: Worker: preserve active Base/AMD and maintainer commits; recover eligible terminal failures. Maintainers: complete DSA deadlock/consumer review with source qualifications. Historical Qwen tokenizer TTFT does not establish DSA causality or a new author validation gate.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820966/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41994820535/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46830/files)

## #45524: Base VLM/AWQ failures and fail-fast siblings persist. AMD PP logged KVTransferError; MLA startup exited -9. CPU fork_worker_preload revision mismatch and historical-source equivalence remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: External author eaglewangao-us/maintainers: compare baselines, inspect MLA exit -9, align CPU revisions, recover coverage and finish review. Preserve frozen producer patch; worker remains read-only.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758455/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758572/job/125766541226)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/40803758488/job/125761907669)

## #46918: Two Base B shards remain cancelled without executed steps; finish failure reflects cancellation. Live mergeable=false/DIRTY still identifies a main conflict. Source-specific 9763ef56 validation does not qualify the current head/image. Excluded AMD CI is not claimed repaired.

Tracking key: `pr46918-external-ci-integration`.

Next action: External author yu-depend/maintainers: reconcile main, recover Base coverage, finish review and validate image integration. Preserve scoped MI355X evidence and frozen workaround.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)
- [Evidence](https://github.com/AMD-AGI/Infera/blob/ce7a2217c0d782a303b1c9f191632fc964e5f2ba/yihou/glm52.patch-inventory_20261027/validation/b8_46918/RESULTS.md)

## #47276: Attempt 4 retains 25 functional and six JIT successes. B1 failed streaming tool-call parsing with None instead of add; one file passed, one failed and 37/39 files did not execute. Its CUDA Llama launch bypasses gfx950 changes; root cause remains unproved. B2 is runnerless, preventing retry while active. Review, #45307 overlap, numerical/source qualifications and optional llr37-disabled role remain.

Tracking key: `pr47276-base-overlap-source`.

Next action: Worker: preserve B2; after terminal Base, prioritize the first group then use owner_requested recovery if eligible and no established fix exists. Maintainers/yihou: retain optional role, disabled in frozen llr37; coordinate #45307 landing order, numerical review and exact source/run archiving. Do not expand the unrelated tool-call failure into kernel work.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/4)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6651481153)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41981310976)
- [Evidence](https://github.com/sgl-project/sglang/pull/45307)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41774179295/job/127091952486)
- [Evidence](https://github.com/sgl-project/sglang/blob/674ad71976e9154c15cc7011596cd0deb5405077/test/registered/openai_server/function_call/test_openai_function_calling.py)

## #47278: Base attempt 5 retains 37 functional and six JIT successes; B2 remains runnerless without executed steps. The inherited request/handler succeeded but no new functional recovery is established. Historical FDFO failure left 31 files unexecuted; matching upstream correction does not establish full coverage.

Tracking key: `pr47278-cancelled-base-retry-scope`.

Next action: Worker: preserve queued B2 and historical FDFO evidence including 31 unexecuted files. Yihou/kernel maintainer: reconcile scaled-KV/split-K conflicts with numerical validation, preserving BF16 partials and DCP/LSE; finish review and exact source/run archiving. Passing old-head AMD does not qualify conflict resolution or a combined stack.

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

## #47872: Base attempt 3 has 20 functional successes (19 retained), five successful JIT units (three retained) and a non-retained benchmark. JIT0 passed 144/144 files before this cycle. Two B and five C jobs remain runnerless. AMD remains successful with 18 stages (16 retained). Xeon mismatch and review/provenance/correctness/performance qualifications remain.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Worker: preserve active Base and passing AMD. CI maintainers: align Xeon installed/tested revisions without unrelated parser/module backports or a CI-only rebase. Maintainers/yihou: complete review; retain opt-in, min(length,2048) selections, exact underflow/overflow fallback, valid ties, source/run mapping, measured 32k regression and source-specific MI355X limits.

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
