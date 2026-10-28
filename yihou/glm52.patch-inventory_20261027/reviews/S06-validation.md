# S06 validation and maintenance, 2026-10-27

- PR: https://github.com/sgl-project/sglang/pull/45484
- Upstream base: `e4d4a9a7de5454cd8b854b21af723c2155a55a3d`.
- Signed commit: `d80a8f2da2023ce721092dc68587d0f6cf740949`.
- Fork branch: `dorado269/sglang:fix/offloader-keep-is-shuffled`.
- Local checkout: `../worktrees/sglang-main`, branch `yihou/s06-offloader-main-20261027`.
- Identity checked before commit: `yihou <yihou@amd.com>`; only `Signed-off-by` trailer.

## Scope and duplicate check

The production change is the original seven-line `is_shuffled` preservation fix, applied to current main. The registered test now derives directly from `unittest.TestCase`, avoiding unnecessary serving/model imports. It retains both test methods, absent/false controls, and value equality checks. No other production code changed.

Searched GitHub issues/PRs for `is_shuffled offload`, `offloader shuffled`, `offload attributes`, and `_CpuParamOffloader`. The existing #45484 owns this fix. Open #37793 changes CPU storage retention/expert streaming; merged #44139 changes cached-weight residency. Neither preserves this device-copy attribute. Their diffs and search results are in `../runtime/upstream-api/`.

## Local checks

Host: `smc300x-ccs-aus-a16-19`. CPU-only PyTorch `2.12.1+cpu`, Python `3.13.13`. This differs from main's full dependency pin and is targeted CPU validation, not the full CI environment. No server, model download, or GPU workload was launched.

From the checkout:

```bash
source ../../runtime/env.sh
export PYTHONPATH="$PWD/python"
../../runtime/cpu-venv/bin/python -m pytest -q \
  test/registered/unit/utils/test_offloader.py \
  -o cache_dir="$XDG_CACHE_HOME/pytest"
pre-commit run --files \
  python/sglang/srt/utils/offloader.py \
  test/registered/unit/utils/test_offloader.py
```

- Patched: **2 passed, 2 subtests passed**, 17.14 seconds. Log: `../runtime/s06-test.log`.
- Baseline: substituted `git show HEAD:python/sglang/srt/utils/offloader.py` before the new commit, retaining exactly the same test. **1 failed, 1 passed, 2 subtests passed**, 19.06 seconds. Failure: `AssertionError: False is not true` at the positive attribute assertion. Log: `../runtime/s06-baseline-test.log`.
- Restored the patched source in a `finally` block and checked byte equality. SHA256: `674392e0454b0ac50f2cce3f763c37003fbd10a81789fb5f8ac0a61150866baa`.
- All applicable pre-commit hooks passed on the final files; `git diff --check` passed. Log: `../runtime/s06-lint.log`.
- The checkout is clean after the signed commit.

Because this checkout is nested under Infera, Ruff initially inherited Infera's line-length configuration. A task-local `../worktrees/ruff.toml` stops that ancestor discovery and restores SGLang's default Ruff settings. Initial incidental formatting was discarded before validation/commit; the final production diff contains only the attribute fix. The isolation file is outside the upstream checkout and is not part of the PR.

## Upstream maintenance

Pushed with an exact lease on old head `8ddcf6b1d1a7f0a78402100813a10cc155dc7045`. Updated the PR body with explicit run IDs, CPU commands, and measurement limits. Requested maintainer CI/review in https://github.com/sgl-project/sglang/pull/45484#issuecomment-6436087321.

The new-head AMD gate job https://github.com/sgl-project/sglang/actions/runs/40466314882/job/122062458072 again reports missing `run-ci`; this is not an executed test failure. See `PR_STATUS.md` for the latest lint/review state.

No throughput gain is attributed to this patch. Historical offload rounds all carried it, so there is no unpatched end-to-end comparison. Historical simulated-acceptance throughput does not establish generated-text correctness.

## Approval-readiness follow-up (2026-10-27 08:06 UTC)

Head: `43dec26b032c08f4e9eee0c772db9f44d3afc275`, base: `449617bc94a5d4ae23ad62d0da2f842e711ca983`. Added forward-wrapper CPU and registered AMD/CUDA onload tests. Patched: 3 passed, 1 GPU skip, 2 subtests. Baseline: 2 expected attribute failures, 1 passed, 2 subtests. Logs: `runtime/s06-readiness-test.log`, `runtime/s06-readiness-baseline.log`, `runtime/s06-readiness-lint.log`. All applicable lint passed. GPU allocation and upstream CI remain pending.

Repeated CPU regression and unpatched-main control with PyTorch 2.14.1+cpu (the main pin): identical 3-pass / 2-expected-failure results. GPU collection remains skipped. Logs: `runtime/s06-torch214-test.log`, `runtime/s06-torch214-baseline.log`.
