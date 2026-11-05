# SGLang #46820: inherited LLaDA FDFO CI performance threshold

Evidence checked on **2026-11-05 UTC**. This note explains the failed Base shard; it does not mark the PR's CI as passed.

## Maintainer merge request

**Level: P0**

**[Fix] [sgl-project/sglang#46820](https://github.com/sgl-project/sglang/pull/46820)**

This PR prevents a stalled KV metrics consumer from blocking scheduler progress by making metrics publication nonblocking. It is needed by downstream deployments. Please review it for merge, taking the following existing upstream CI correction into account.

At published head `ac205f829173029ce17c1898fd9c8d9ab2f861fd`, **37 functional Base jobs passed**. The remaining H100 job's observed failure is `TestBatchingFDFO.test_bs_1_speed`: **246.67 / 245.91 token/s** against the old `>250` assertion. The [same test on upstream main](https://github.com/sgl-project/sglang/actions/runs/41542824410/job/125611431950) also failed, at **241.52 / 239.63 token/s**. Both snapshots contain the identical test-file blob.

Upstream already merged [#47989](https://github.com/sgl-project/sglang/pull/47989), changing this exact host-CPU-sensitive assertion to **`>200`**. The recorded measurements exceed the corrected threshold. The failing launch has `kv_events_config=None`, so it returns before executing the metrics send modified by #46820. This supports classifying the observed failure as the already-corrected upstream test issue rather than a failure of the changed send path.

**Coverage:** the failing H100 shard stopped after six passing files and one failed file out of 38; the other 31 files were not executed in that attempt. The Base workflow remains red, and this report does not claim those files passed. The request is for maintainer assessment of the documented upstream failure, not a new PR-specific threshold change or an assertion that every test completed. Full evidence follows below.

## Conclusion for reviewers

The remaining Base failure on [SGLang #46820](https://github.com/sgl-project/sglang/pull/46820) is the same LLaDA FDFO single-request performance assertion that failed on an unpatched upstream `main` revision. The failing launch disables KV events, so execution returns before the nonblocking metrics send changed by this PR. Upstream has already merged [#47989](https://github.com/sgl-project/sglang/pull/47989), commit [`c927e7c`](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c), changing this exact CPU-sensitive assertion from `speed > 250` to `speed > 200`.

Both recorded #46820 measurements exceed the corrected threshold. This demonstrates that the upstream correction addresses the observed assertion failure; it is **not** a rerun or a claim that the entire suite now passes. No PR-specific threshold relaxation is proposed.

## Direct CI evidence

| Source | Immutable source / attempt | Observed failure |
|---|---|---|
| [Scheduled upstream main Base job](https://github.com/sgl-project/sglang/actions/runs/41542824410/job/125611431950) | main `9eec211b9a134419d92a47d442da5d171cb833ea`; run created November 2 at 23:06:53 UTC | `TestBatchingFDFO.test_bs_1_speed`: **241.51844971937498** and **239.62652504958703** token/s, both below the old 250 threshold. |
| [#46820 Base attempt 5, H100 B2](https://github.com/sgl-project/sglang/actions/runs/41467220762/job/126989078017) | PR head `ac205f829173029ce17c1898fd9c8d9ab2f861fd`; tested merge `142c4a7f565ca91f87a36e0b30c9e964b4c4e737` | The same test: **246.6744753146441** and **245.90637097983733** token/s, both below 250 and above 200. |
| [Merged upstream correction](https://github.com/sgl-project/sglang/commit/c927e7c394cd33a11a62395e8a588ce16cae117c) | #47989 merged November 2 at 23:41:42 UTC | Changes this exact assertion to `> 200`; explains that bs=1 dLLM decode is host-CPU bound, without overlap scheduling and with a synchronization per denoising step. |

The scheduled main run captured its revision **before** the correction merged. Its failing test executed later, on November 3. These timestamps do not imply that current main still contains the old assertion.

The scheduled-main revision and the PR's failing merge contain the **identical test-file Git blob** `9a204e046ae00eafd72777d19adaa79e8d72f74c`. The fixed assertion is visible at [commit c927e7c, line 97](https://github.com/sgl-project/sglang/blob/c927e7c394cd33a11a62395e8a588ce16cae117c/test/registered/dllm/test_dllm_batching_fdfo.py#L97); the upstream change also received [maintainer approval](https://github.com/sgl-project/sglang/pull/47989#pullrequestreview-5744841188). This evidence does not establish identical CPU conditions across the two jobs or exclude every possible performance effect; the changed-send exclusion below is the narrower, directly verified attribution.

The PR's B2 shard completed six passing test files and one failing file out of 38; 31 files were not executed in that attempt. The Base workflow retains 37 passing functional jobs and one failed functional shard. Passing files within B2 do not turn that shard green. Historical LLaVA uncertainty is separate and is not resolved by this FDFO diagnosis.

These numbers concern the LLaDA CI test, not GLM-5.2 deployment throughput or the historical 48k curve.

## Why this failure does not exercise our changed send

1. The failing FDFO launch log records `kv_events_config=None` and `enable_metrics=False`; the [test's launch arguments](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/test/registered/dllm/test_dllm_batching_fdfo.py#L31) do not enable KV events.
2. [`is_kv_publisher_rank`](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/python/sglang/srt/disaggregation/kv_events.py#L66) returns false without a KV-events configuration.
3. [`KvEventsPublisher.emit_kv_metrics`](https://github.com/sgl-project/sglang/blob/ac205f829173029ce17c1898fd9c8d9ab2f861fd/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py#L75) returns immediately when KV events are disabled, before reaching the send modified by #46820.

#46820 fixes scheduler blocking when an enabled metrics consumer stops draining its socket. That behavior still needs its own socket-contract tests and code review; the unrelated LLaDA speed failure is not evidence against it.

## Prepared integration and validation

A signed local rebase is preserved at `3af3189a86ac97eae65c13f7fceb82b0fe52fa44`, based on upstream main `7fb6ff4736b388b2d3a279328c638d255f0c23e8`, which includes #47989. It has **not been pushed**. Its PR patch is identical to the previous patch under stable patch-ID comparison: `dc1ebbba0ff435f519b5071d4673deb86254154b`. The PR diff still contains only the KV publisher and its regression test.

- Four socket regression tests passed in each of MsgPack and pickle modes: eight checks using real ZMQ and isolated package imports. The upstream blocking-send negative control raised `zmq.Again` as expected.
- Scoped lint and `git diff --check` passed.
- Normal pytest collection was blocked by the local environment's missing `pybase64`; no full CPU suite or GPU/model rerun is claimed.
- Existing signatures and authorship were preserved; the rewritten owned commits carry yihou's sign-off.

The earlier HTTPS push was rejected because that OAuth credential lacked `workflow` scope for inherited main workflow updates. On November 5 at approximately 05:51 UTC, the existing SSH credential successfully authenticated as the same `dorado269` account. No credential, scope, key, workflow, or global Git configuration was changed. Publication remains deferred while other unfinished Base runs receive priority; this is no longer described as an account-authentication failure.

The preserved local checkout is `runtime/ci-watch-20261103/cycles/20261105T050001Z/pr46820-repair` beneath this inventory. Publishing it later requires a fresh remote-head check and an exact force-with-lease. A new head starts new CI; the old head's 37 successes cannot be presented as results for the new revision.

## Recovery priority after the owner's November 5 instruction

Keep monitoring #46820, but do not spend the next recovery opportunities repeatedly running its unchanged, already-explained 250-threshold failure. The helper records a temporary deferral for the exact head/run/attempt, with an explicit expiry and evidence; it is not a permanent exclusion or a passing status. A changed revision or attempt receives a fresh assessment.

At the 05:50 UTC audit:

| PR | Base state | Action |
|---|---|---|
| #33059 | 25 passes; two H100 shards queued | Preserve the active attempt and queue position. |
| #46830 | 26 passes; one H100 shard queued | Preserve the active attempt and queue position. |
| #47276 | Attempt 3 cancelled; 25 retained Base passes and six retained JIT results | Next eligible unfinished owned Base candidate; recover through the existing helper. |
| #47278 | Attempt 4 queued; 37 retained Base passes | Follow the active attempt; its AMD workflow has passed, while the main merge conflict remains separate. |
| #47872 | Attempt 3 queued; 19 retained Base and three retained JIT passes | Follow the active attempt. |
| #47897 | Base passed | Preserve passing Base; remaining shared-code/review matters are tracked separately. |

The next author-wide retry opportunity is **2026-11-05 06:34:03 UTC / 14:34:03 Beijing**, subject to no intervening reservation and fresh eligibility checks. Pending or running workflows are not duplicated. External #45524 and #46918 remain read-only.

Current execution and review state is maintained in [PR_STATUS.md](PR_STATUS.md), [CI_48K_WATCH.md](CI_48K_WATCH.md), and the permanently Draft [team tracker #228](https://github.com/AMD-AGI/Infera/pull/228).

## Separate status of #33059 and #46830

Checked at **2026-11-05 06:06 UTC / 14:06 Beijing**: [#33059 Base attempt 3](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/3) has 25 passing functional jobs and two queued H100 jobs; [#46830 Base attempt 4](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/4) has 26 passing functional jobs and one queued H100 job. These unexecuted jobs cannot be treated as passes or attributed to the 250 threshold. Both current heads already contain `c927e7c`, so neither needs another rebase for this correction.

Continue their active runs and recover future failed/cancelled jobs through the existing hourly retry mechanism. If a later failure is shown to bypass our changed path and has a matching upstream failure and merged correction, produce the same evidence-based maintainer report and move recovery effort to other unfinished runs. Preserve the actual red/queued state and disclose any unexecuted coverage.
