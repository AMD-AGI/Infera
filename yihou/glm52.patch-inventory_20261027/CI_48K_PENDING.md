# Pending handoff for the next user session

Updated: **2026-11-24T06:20:43.550534+00:00**.

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

## #47276: Unchanged e95ec5ab Base attempt 3 passes 35 functional/seven JIT jobs; inherited retention is 12 functional/one JIT results. Fresh source/reviews/step timestamps match. MERGEABLE/BLOCKED/REVIEW_REQUIRED. Maintainer review, #45307 overlap, exact source/run archival and independent numerical qualifications remain. New minimum-base commit is absent, but passing head must be preserved. Historical FlashInfer causality is unproven; optional and disabled in frozen llr37.

Tracking key: `pr47276-base-overlap-source`.

Next action: Maintainers review passing e95ec5ab and coordinate #45307 overlap. Preserve passing head; no freshness retry/rebase. Owner archives exact source/run and independent numerical evidence. Optional and disabled in frozen llr37. Owner/maintainers assess any actual future integration need; future cooldown-gate recovery also needs authorized helper-policy audit.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44478753284/attempts/3)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276#issuecomment-7010787687)
- [Evidence](https://github.com/sgl-project/sglang/pull/47276/files)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47278: Unchanged 9602092a Base/AMD attempt 1 fully pass: Base 33 functional/seven JIT and AMD 17 functional/two JIT jobs; execution predates this cycle. Inherited matching logs record 45 sparse-MLA passes and disaggregation four files/13 cases. kkHuang-amd APPROVED on exact head; aggregate remains MERGEABLE/BLOCKED/REVIEW_REQUIRED. Source/image/raw-run archival and independent numerical/layout/replay qualifications remain. Prior-head OOM/cancellation stay historical without proven causal recovery; no current textual conflict.

Tracking key: `pr47278-amd-validation-source`.

Next action: Preserve passing head. Maintainers assess remaining aggregate review requirements despite exact-head approval. Owner archives exact-parent 8d72d2ea source/image/raw evidence and retains BF16 partials, seq15/non-target numerical, scaled-KV, DCP/LSE and layout/replay qualifications. Retrieve existing evidence before any necessary authorized GPU work. Current-code A/B is supplied, not wholly missing. Local requirements: cycles/20261124T060001Z/PR47278_GPU_VALIDATION.json. No issue/review reply without owner authorization.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473623/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/44765473872/job/136428385564)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)

## #47897: Current integrated 5a06f9e7 preserves all three PR file blobs and both production callers from f08bde1f. Exact run-to-source/image/raw archival and long-context numerical qualification remain. Gluon bytes match historical 8a9f215c, but scheduled generic and qualifying verify calls still route to Aiter. Historical isolated/serving gains do not establish this head engagement or performance; short-context GSM8K does not establish long-context DSA accuracy.

Tracking key: `pr47897-source-long-context`.

Next action: Owner/reviewers archive exact sources and raw evidence, establish current-main Gluon engagement, and preserve unequal paired-row/table-length, common-prefix/tail and fallback qualification. Obtain additional GPU validation only through an authorized allocation if existing exact-source evidence is insufficient; this worker has no machine-pool skill. Do not attribute frozen whole-stack throughput to this patch.

- [Evidence](https://github.com/sgl-project/sglang/pull/47897/files)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)

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

## #47278: Mixed review/provenance item retained: exact-head 9602092a APPROVED/LGTM is confirmed, while aggregate REVIEW_REQUIRED/BLOCKED persists. Author supplied current-code A/B reporting 10.3% lower ITL; TPOT/throughput overlap. H64 gate/non-target fallback, unreachable-variant cleanup, seq15/16/128 routes and inherited CPU/CI evidence are documented locally. Immutable raw provenance, seq15/non-target exact-parent numerical, BF16 partials, scaled-KV/DCP/LSE and layout/replay qualifications remain. Triton causality is suspected/unconfirmed and reviewer-nonblocking; warmup is author-reported.

Tracking key: `pr47278-review-20261117`.

Next action: Preserve passing 9602092a Base/AMD. Maintainers assess remaining aggregate review requirements. Owner retrieves and archives existing exact-parent source/image/raw evidence before requesting additional hardware work. Preserve numerical/layout/replay and compiler qualifications. Prepared follow-up: cycles/20261124T060001Z/PR47278_REVIEW_HANDOFF.json and PR47278_GPU_VALIDATION.json. No issue or review reply without owner authorization.

- [Evidence](https://github.com/sgl-project/sglang/pull/47278#pullrequestreview-6016992240)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-6947249189)
- [Evidence](https://github.com/sgl-project/sglang/pull/47278#issuecomment-7043695032)
- [Evidence](https://github.com/sgl-project/sglang/blob/9602092ac908c2077a400abe7aa3d1a1b26a9220/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py#L1618)
- [Evidence](https://github.com/ROCm/triton/issues/985#issuecomment-6617640183)

## #47276: Conditional future recovery limitation, not an active functional failure: main 6bd7db14 gate SHA256 bae2b8efdf384503855df4ba5c8c6649dd62e0907cab59940a5b08b4ac790371 differs from helper audit 63e133892c2ef42609c0fdd33f3ae524afad21fb792c9eb3bbf8518916915af1. New MIN_BASE_SHA 30e076fa is absent from passing #47276/#47278; owner explicitly requires preserving them. #47897 now contains it but needs maintainer commit approval. No cooldown failure or actual helper refusal; hourly slot available and no request made.

Tracking key: `pr47276-stale-gate-policy-drift`.

Next action: Preserve passing #47276/#47278. Owner/maintainers assess an actual future integration requirement before any rewrite. If a cooldown gate later fails, automation maintainer must audit current semantics and authorize helper changes; worker must not edit helper/config or bypass approval. Commit approval requires an upstream maintainer, not an author retry.

- [Evidence](https://github.com/sgl-project/sglang/blob/6bd7db148b96fe98d6560785290716674c995ee1/.github/workflows/pr-gate.yml)
- [Evidence](https://github.com/sgl-project/sglang/issues/21065)

## #47897: Published 5a06f9e7 contains live minimum-base 30e076fa. New Base 45338553315 attempt 1 completed/failure solely at Require that tests ran: gate explicitly says this head lacks run-ci approval. Ready/run-ci are present; gate success means Skip tests executed. Zero functional/JIT tests, retained results or builds; 18 skipped jobs. No author helper request/reservation/handler or retry attempt. Historical Base success and old f08bde1f action_required do not cover this head.

Tracking key: `pr47897-current-head-base`.

Next action: Upstream maintainer approves exact 5a06f9e7 through the authorized upstream process, then follow actual Base execution. Author /rerun-failed-ci and run-ci label alone approve nothing. Recover eligible terminal functional/cancelled attempts through the guarded PR-wide helper after fresh head/config/authorization checks.

- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/attempts/1)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/job/138256395291)
- [Evidence](https://github.com/sgl-project/sglang/actions/runs/45338553315/job/138256449433)
- [Evidence](https://github.com/sgl-project/sglang/commit/5a06f9e7f6b455c99dabc108f60d7a0648a317e7)

## #47897: Current-main integration preserves schedules by routing non-None persistent_schedule to Aiter. Audited production generic gfx950 preshuffled calls construct that schedule; qualifying target-verify calls take direct Aiter before this wrapper. Unscheduled unit calls still reach Gluon. Real serving engagement and preservation of historical patch gain on 5a06f9e7 are unestablished. This is a performance/source qualification, not a demonstrated numerical failure.

Tracking key: `pr47897-current-main-dispatch-engagement`.

Next action: Owner/kernel reviewer maps intended deployment calls and archives a current-head engagement trace against exact main/source/image. Preserve schedule semantics and paired-row/tail/fallback correctness; hand off any GPU/performance-sensitive dispatch change before publication. Retrieve existing evidence first; worker must not access GPUs through another route.

- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L106)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1646)
- [Evidence](https://github.com/sgl-project/sglang/blob/5a06f9e7f6b455c99dabc108f60d7a0648a317e7/python/sglang/srt/layers/attention/dsa/dsa_indexer_hip.py#L94)
