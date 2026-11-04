# Current-main integration for the dLLM CI blocker

## Cooldown recovery and live progress: 2026-11-04T06:38:32.575416+00:00

Both bounded retry requests below were made sequentially after the logged
cooldown elapsed, with fresh upstream policy and account-wide checks. Both
handlers succeeded, both Base gates passed, and both runs reached attempt 2.
This supersedes the initial cooldown state recorded later in this document.
No code change was needed during this follow-up.

| PR | Current head | Base progress | Action |
|---|---|---|---|
| [#33059](https://github.com/sgl-project/sglang/pull/33059) | `201d823b3c6a` | [Base attempt 1: A9/B10=19 passed; three B jobs queued without execution, C not started. Gate passed.](https://github.com/sgl-project/sglang/actions/runs/41773699215/attempts/1) | Read-only current-head audit; passes increased from 15 to 19 since the 05:34 snapshot. No retry of this active run. |
| [#46830](https://github.com/sgl-project/sglang/pull/46830) | `76b75ed6ed9b` | [Base attempt 2: author cooldown gate passed; nine stage-A jobs in progress. No complete Base result yet.](https://github.com/sgl-project/sglang/actions/runs/41773966833/attempts/2) | After fresh upstream-policy, account-wide cooldown, head and attempt checks, sent a sequential bounded stale_gate retry via the helper: [request](https://github.com/sgl-project/sglang/pull/46830#issuecomment-6624357389). [Slash-command handler](https://github.com/sgl-project/sglang/actions/runs/41786487047) succeeded and the new Base attempt passed its gate. The supported command also retried other unsuccessful current-head workflows; no code change or rebase this invocation. |
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | `674ad71976e9` | [Base attempt 2: author cooldown gate passed; nine stage-A jobs in progress. No complete Base result yet.](https://github.com/sgl-project/sglang/actions/runs/41774179295/attempts/2) | After fresh upstream-policy, account-wide cooldown, head and attempt checks, sent a sequential bounded stale_gate retry via the helper: [request](https://github.com/sgl-project/sglang/pull/47276#issuecomment-6624386421). [Slash-command handler](https://github.com/sgl-project/sglang/actions/runs/41786705555) succeeded and the new Base attempt passed its gate. The supported command also retried other unsuccessful current-head workflows; no code change or rebase this invocation. |

The remaining six PRs were freshly checked at 06:33 UTC: #47897 Base remains
successful; #46820/#45524/#47872 retain their previously diagnosed failures;
#46918/#47278 retain cancelled Base shards. All nine are Ready with run-ci and
passing lint; no new review or merge was observed. Original ownership, source
and numerical validation qualifications remain unchanged.

## Initial integration snapshot

Initial current-head CI audit: **2026-11-04T04:37:15.907135+00:00**. All three PRs remain Ready with run-ci.
Integration is complete; new Base execution is not yet complete.

Upstream [#47989](https://github.com/sgl-project/sglang/pull/47989) already changed
the LLaDA2.0-mini batch-size-one speed assertion from >250 to >200 token/s for
slower host CPUs in the H100 CI lane. This metric differs from GLM 48k throughput.
The previous failed runs executed older test revisions. All three owned PRs now
include the upstream correction unchanged via rebase onto verified main
`4c7c775cf4adcbd2c7419d13b51ec838a17e559e`, containing upstream fix
`c927e7c394cd33a11a62395e8a588ce16cae117c`.

| PR | Head transition | Local validation | New Base attempt 1 |
|---|---|---|---|
| [#33059](https://github.com/sgl-project/sglang/pull/33059) | `795488cd19aa` → `201d823b3c6a` | 12 passed; lint passed. | [Gate passed; nine Base-A jobs in progress.](https://github.com/sgl-project/sglang/actions/runs/41773699215) |
| [#46830](https://github.com/sgl-project/sglang/pull/46830) | `ba484a37910f` → `76b75ed6ed9b` | 3 passed plus 2 subtests; lint passed. | [Author 120-minute cooldown gate; functional tests skipped. Earliest logged boundary 06:31:19 UTC, conditional on no newer counted run.](https://github.com/sgl-project/sglang/actions/runs/41773966833) |
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | `74c576b7db58` → `674ad71976e9` | Existing launch-selection dispatch assertions passed; no GPU numerics executed; lint passed. | [Author 120-minute cooldown gate; functional tests skipped. Earliest logged boundary 06:31:19 UTC, conditional on no newer counted run.](https://github.com/sgl-project/sglang/actions/runs/41774179295) |

## Code changes and validation

`git range-diff` confirms each original patch survived unchanged. #33059 and
#46830 additionally update one test import: ForwardMode now comes from its
upstream defining module, required by the current indirect-import lint rule.
#47276 needed no source adjustment beyond rebase.

Commands, from the corresponding SGLang checkout with its python directory on
PYTHONPATH and GPU visibility disabled:

- #33059: `python -B -m pytest -q -p no:cacheprovider test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py` — 12 passed.
- #46830: `python -B -m pytest -q -p no:cacheprovider test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py` — 3 passed plus 2 subtests.
- #47276: invoke the existing `TestGfx950FusedIndexerKernels.test_logits_launch_selection` method with the real loader, without GPU setUpClass — dispatch assertions passed; no GPU compilation, numerical kernel test, or performance measurement.
- All three: upstream pre-commit hooks on PR-changed files passed. The local invocation uses Ruff `--isolated` to prevent enclosing Infera settings from overriding SGLang defaults; no upstream lint configuration was changed.

All integrations used an isolated task-owned checkout, the configured yihou
identity, signed commits, fresh author/head checks and exact force-with-lease.
Original refs, before/after patches and validation logs remain archived locally.
No experiment pack-ups were staged or modified.

## Next action and remaining work

#33059 needs its active Base run to finish. #46830/#47276 need a bounded retry
after the upstream author cooldown; their current failure is a gate refusal,
not an executed functional regression. Both logs cite 04:31:19 UTC plus 120
minutes. Check fresh account-wide runs before using that boundary. The 90-minute
watcher owns subsequent checks/retries. A new push or a passing gate is not a
passing Base result.

The Nov 4 04:25 UTC review audit found no new comments, reviews, requested changes,
or inline threads since Nov 3 14:00 UTC. REVIEW_REQUIRED describes an unmet
upstream approval requirement, not nine new development requests. #46820 retains
an existing LGTM with a request for upstream-maintainer review of shared code.
#33059's native MI300 follow-up and kernel source/numerical evidence requirements
are existing validation work. VLM and other scoped CI failures still need
attribution; no new kernel feature or rewrite has been established as necessary.

The worker instructions now require safe integration of a verified already-merged
upstream fix rather than repeatedly handing the same actionable source mismatch
back to CI owners. Existing ownership, validation, retry and scope limits remain.
