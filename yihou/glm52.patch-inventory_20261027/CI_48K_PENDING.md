# Pending handoff for the next user session

Updated: **2026-11-24T00:07:14.229187+00:00**.

These items remain open until supported by resolution evidence or an explicit exclusion under the [CI scope policy](CI_SCOPE_POLICY.md). Reporting them to the user does not resolve them; scope exclusion is not test success.

## #45524: Historical/unrefreshed; outside the resumed scope. Base passed 27 functional jobs. Same AMD shard 6 exited 255 after starting tests; historical log retrieval returned HTTP 404. First exception, source equivalence and historical AWQ/PP review remain unresolved.

Tracking key: `pr45524-external-ci-review`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: External author eaglewangao-us/maintainers retrieve runner logs, recover coverage and complete source-equivalence review; preserve frozen producer patch.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42170903730/job/127742649611)
- [Evidence](https://github.com/sgl-project/sglang/pull/45524/files)

## #46918: Historical/unrefreshed; outside the resumed scope. Base has 36 functional successes and two shards cancelled without tests. One annotation identifies github-actions[bot]; finish log confirms cancellation caused the aggregate failure. Cancellation policy and original-PR integration remain unresolved. Scoped validation at 9763ef56 does not qualify current head/image.

Tracking key: `pr46918-external-ci-integration`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: External author yu-depend/maintainers own original-PR conflict, remaining coverage and review decisions, including the unexplained cancellation policy. Advance owned continuation #49732 separately; preserve frozen workaround and source-specific MI355X evidence at 9763ef56.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/41467734403/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918/files)

## #47276: Unchanged e95ec5ab Base attempt 3 passes all 35 functional/seven JIT jobs; inherited retention is 12 functional/one JIT results. Source hashes, reviews and executed-step timestamps match prior evidence. Final GraphQL is MERGEABLE/BLOCKED/REVIEW_REQUIRED. Maintainer review, #45307 overlap, exact source/run archival and independent numerical qualifications remain. Historical FlashInfer causality is unproven. Optional and disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review passing e95ec5ab and coordinate #45307 overlap. Preserve passing head; no freshness retry/rebase. Owner archives exact source/run mapping and independent numerical evidence. Optional and disabled in frozen llr37. Historical FlashInfer causality remains unproven without matching controls. Only a future cooldown-gate failure requires an authorized helper-policy audit.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44535642771)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/compare/50322ac7ba85df18111ec79d18a4c5c554df60c9...e95ec5abc8b4a9bf27b3dae160e42dc5beeb5f6e)

## #47278: Unchanged 9602092a Base/AMD attempt 1 fully pass: Base 33 functional/seven JIT and AMD 17 functional/two JIT jobs; execution predates this cycle. Inherited matching logs record 45 sparse-MLA passes and disaggregation four files/13 cases. kkHuang-amd APPROVED on exact head; aggregate remains MERGEABLE/BLOCKED/REVIEW_REQUIRED. Source/image/raw-run archival and independent numerical/layout/replay qualifications remain. Prior-head OOM/cancellation stay historical without proven causal recovery; no current textual conflict.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve passing 9602092a Base/AMD; no freshness retry/rebase. Maintainers assess remaining REVIEW_REQUIRED/BLOCKED requirements despite exact-head approval. Owner archives exact-parent 8d72d2ea source/image/raw evidence. Retain seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. Author reports current-code A/B with 10.3% lower ITL; TPOT/throughput overlap. Compiler causality is suspected/unconfirmed and reviewer-nonblocking; warmup is author-reported. Retrieve existing evidence before necessary authorized GPU work; no machine-pool skill here. No issue/review reply without owner authorization. Local follow-up: cycles/20261124T000002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Required source-family port, not byte-identical frozen overlay.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/job/136437583246)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136396984786)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7033217328)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44694630806)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/job/136148493015)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908940/attempts/2)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135554671697)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908317/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7008488632)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44519908888/job/135585592844)

## #47897: Reactivated for current f08bde1f. Exact run-to-source/image/raw archival and long-context numerical qualification remain. Gluon kernel bytes match historical 8a9f215c, while current main dispatch routes scheduled generic and qualifying verify calls to Aiter; inherited isolated/serving gains do not establish rebased-head engagement or performance. Short-context GSM8K does not establish long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Owner/reviewers archive exact sources and raw evidence, establish current-main Gluon engagement, and preserve unequal paired-row/table-length, common-prefix/tail and fallback qualification. Obtain additional GPU validation only through an authorized allocation if existing exact-source evidence is insufficient; this worker has no machine-pool skill. Do not attribute frozen whole-stack throughput to this patch.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/f08bde1f0ad9fea26a9cb629dfc512a540ad9fc0/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/f08bde1f0ad9fea26a9cb629dfc512a540ad9fc0/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)

## #47872: Historical/unrefreshed; outside the resumed scope. OWNER PAUSED pending redesign. Historical passing CI and source/numerical qualifications remain archived and were not queried.

Tracking key: `pr47872-new-head-ci-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Owner-paused; resume only on owner instruction. Preserve opt-in/fallback/tie, measured 32k regression and exact-source/long-context limits.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42722204156/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47872/files)

## #49732: Historical/unrefreshed; outside the resumed scope. Base attempt 3 passed all 31 functional jobs; CI gap recovered. Current-head maintainer review and integrated-image qualification remain. Isolated configuration checks and archived 9763ef56 GPU evidence do not establish full module/model coverage for this candidate.

Tracking key: `pr49732-base-review-validation`.

Next action: Outside the resumed scope; retain for explicit owner resumption only. Historical next step: Maintainers review and merge owned continuation preserving yu-depend/#46918 credit. Complete integrated-image qualification and exact-source numerical evidence.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/42974439450/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/49732)
- [Evidence](https://github.com/sgl-project/sglang/pull/46918)

## #47278: Mixed review/provenance item retained: kkHuang-amd exact-head 9602092a APPROVED/LGTM is confirmed, while aggregate REVIEW_REQUIRED/BLOCKED persists. Author supplied current-code A/B reporting 10.3% lower ITL; TPOT/throughput overlap. Source/reviews unchanged and Base/AMD fully passed. H64 gate/non-target fallback, unreachable-variant cleanup, seq15/16/128 routes and inherited CPU/CI evidence are documented locally. Immutable raw provenance, seq15/non-target exact-parent numerical, BF16 partials, scaled-KV/DCP/LSE and layout/replay qualifications remain. Triton link is suspected/unconfirmed and reviewer-nonblocking; warmup is author-reported.

Tracking key: `pr47278-review-20261117`.

Next action: Preserve passing 9602092a Base/AMD; no freshness retry/rebase. Maintainers assess remaining REVIEW_REQUIRED/BLOCKED requirements despite exact-head approval. Owner archives exact-parent 8d72d2ea source/image/raw evidence. Retain seq15/non-target numerical, BF16-partial, scaled-KV, DCP/LSE and layout/replay qualifications. Author reports current-code A/B with 10.3% lower ITL; TPOT/throughput overlap. Compiler causality is suspected/unconfirmed and reviewer-nonblocking; warmup is author-reported. Retrieve existing evidence before necessary authorized GPU work; no machine-pool skill here. No issue/review reply without owner authorization. Local follow-up: cycles/20261124T000002Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. Required source-family port, not byte-identical frozen overlay.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: Conditional future retry limitation, not an active test failure: freshly retrieved main c64f2256 pr-gate.yml SHA256 21e00a44276ae5de28b74b106ecf71c76330054e5db648a1bc0e3e0b86aef394 differs from helper audited 63e133892c2ef42609c0fdd33f3ae524afad21fb792c9eb3bbf8518916915af1. A cooldown-based stale_gate check would fail closed. Current #47276/#47278 gates pass; #47897 awaits maintainer workflow approval with zero Base jobs. Hourly interval elapsed; no retry CLI or actual helper refusal this cycle.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: If a future active-PR Base gate fails its author cooldown, the owner/automation maintainer must audit current gate semantics and authorize any helper update before stale_gate recovery. Do not edit the helper/config or bypass its refusal. Ordinary failed-job recovery retaining a passing gate is not blocked by this check.

- [Evidence](https://github.com/sgl-project/sglang/blob/c64f2256d58358ff65c53f0e38dc3e6bcfac8484/.github/workflows/pr-gate.yml)

## #47897: Freshly verified rebased f08bde1f has no current-head Base coverage: 45123546744 attempt 1 remains completed/action_required; public page explicitly awaits maintainer approval. Zero jobs/check-runs, executed gates, functional/JIT tests or retained results. run-ci present. Historical passing Base belongs to another head. Read-only helper eligibility returns base_run_not_unsuccessful; no CLI/request/reservation/handler/new attempt this cycle.

Tracking key: `pr47897-current-head-base`.

Next action: Upstream maintainer approves existing workflow run. Follow actual jobs and recover eligible terminal unsuccessful attempts through guarded helper after fresh head/config checks. Preserve head/schedule semantics; do not duplicate requests or infer test success from approval.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45123546744/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)

## #47897: Current-main integration preserves schedules by routing non-None persistent_schedule to Aiter. Audited production generic gfx950 preshuffled calls always construct that schedule; qualifying target-verify calls take direct Aiter before this wrapper. Unscheduled unit calls still reach Gluon. Real serving engagement and preservation of the historical patch gain on f08bde1f are unestablished. This is a performance/source qualification, not a demonstrated numerical failure.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: Owner/kernel reviewer maps intended deployment calls and archives a current-head engagement trace against exact main/source/image. Preserve existing schedule semantics and paired-row/tail/fallback correctness; hand off any proposed GPU/performance-sensitive dispatch change before publication. Retrieve existing evidence first; worker must not access GPUs through another route.

- [Evidence](https://github.com/sgl-project/sglang/blob/f08bde1f0ad9fea26a9cb629dfc512a540ad9fc0/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/f08bde1f0ad9fea26a9cb629dfc512a540ad9fc0/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/f08bde1f0ad9fea26a9cb629dfc512a540ad9fc0/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L94)
