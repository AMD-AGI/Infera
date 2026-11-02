# 48k-priority SGLang rebase and CI handoff — 2026-11-02

<!-- LINT-REPAIR-AUDIT -->
## Current lint repair and Ready status

Latest audit: 2026-11-02T10:06:49.041258+00:00. Four owned PRs were independently rebased onto main `b45a36531c3673048aeaf35322c3df7f18edd038`, which includes upstream XPU logger fix [#47837](https://github.com/sgl-project/sglang/pull/47837). Every range-diff preserves the PR patch, local full-tree Ruff and scoped hooks pass, and the full upstream Lint job passes at each new head. Each push used an explicit old-head lease and retained a local backup. No other author's branch was changed.

| PR | Current head | Local validation | Upstream lint / next action |
|---|---|---|---|
| [#46820](https://github.com/sgl-project/sglang/pull/46820) | `783c1a2192ec` | Full-tree Ruff, scoped repository hooks and 4 real-ZMQ CPU tests passed; range-diff unchanged | [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41424697358/job/125208358328); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41424697709/job/125208623306) sees Ready; missing maintainer-controlled `run-ci`. |
| [#46830](https://github.com/sgl-project/sglang/pull/46830) | `625403853ab6` | Full-tree Ruff, scoped repository hooks and 3 CPU tests plus 2 subtests passed; range-diff unchanged | [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425300647/job/125210347134); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425300984/job/125210545797) sees Ready; missing maintainer-controlled `run-ci`. |
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | `23a8de201bd5` | Full-tree Ruff, scoped repository hooks and CPU launch-selection assertions passed; 7 GPU tests skipped; range-diff unchanged | [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425379090/job/125210599208); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425379155/job/125210861058) sees Ready; missing maintainer-controlled `run-ci`. |
| [#47278](https://github.com/sgl-project/sglang/pull/47278) | `d3a13d1cf87f` | Full-tree Ruff, scoped repository hooks and 3 CPU contracts passed; 22 GPU tests skipped; range-diff unchanged | [Upstream Lint passed](https://github.com/sgl-project/sglang/actions/runs/41425638344/job/125211454851); [Base gate](https://github.com/sgl-project/sglang/actions/runs/41425638677/job/125211664882) sees Ready; missing maintainer-controlled `run-ci`. |
| [#33059](https://github.com/sgl-project/sglang/pull/33059) | `795488cd19aa` | User/author advanced main and marked Ready; preserved without a branch rewrite. [Author MI355X DP/MTP results](https://github.com/sgl-project/sglang/pull/33059#issuecomment-6573920447) cover the padded-row failure; tested/current DSA and regression-test blobs match. | [Lint passed](https://github.com/sgl-project/sglang/actions/runs/41423254991/job/125203622542). [Base](https://github.com/sgl-project/sglang/actions/runs/41423254776/job/125204140539) sees Ready and existing `run-ci`; account cooldown, not a Draft gate. Native MI300 and required CI/review remain. |

For #33059, the observed 120-minute cooldown began at 09:07:56 UTC; nominal eligibility is November 2 at 11:07:56 UTC if no later counted run intervenes. Expiry does not automatically rerun CI. For the other four, an authorized maintainer must add `run-ci` and trigger required checks. No new GPU test or benchmark was run for this lint repair; the 7/22 skipped kernel cases are not GPU passes. Historical 48k throughput still uses simulated MTP acceptance and is not patch-level evidence.

External [#45524](https://github.com/sgl-project/sglang/pull/45524) and [#46918](https://github.com/sgl-project/sglang/pull/46918) already have passing lint at their audited heads; their branches were not changed. The separate exact-head MI355X evidence for #46918 remains valid within its stated scope. Infera [#228](https://github.com/AMD-AGI/Infera/pull/228) remains permanently Draft for tracking.

## Earlier rebase audit, retained as history

The following 07:02 UTC snapshot is superseded by the audit above. Its Draft, missing-model-matrix and unresolved-XPU-lint statements describe the earlier state, not the current handoff. Validation commands and source boundaries remain useful.

Last audited: 2026-11-02T07:02:09.732231+00:00. The five owned PRs below were independently rebased onto main `23c6442cde7fb38230134ba0e3513ad2332aa5cf` (the current main when work began). Each has a local pre-rebase backup, a yihou Signed-off-by trailer and an exact old-head lease on its force push. Every range-diff reports an unchanged patch. No other author's branch was changed.

This is the 48k **priority queue**, not a complete reproducible serving stack. #33059, #46820 and #46830 represent historical baked fix families; #47276 and #47278 are later scoped K01-family ports, not byte-identical substitutes for the complete frozen overlay. See [deployment priorities](LLR37_DEPLOYMENT_PRIORITY.md) and [pack-up map](PR_PACKUP_MAP.md) for the remaining image, overlay, configuration and attribution dependencies. Historical stack throughput used simulated MTP acceptance; it does not establish patch-level gains or real-acceptance correctness.

## Historical status at 07:02 UTC

| PR | New head | Local validation | Base / upstream lint / next action |
|---|---|---|---|
| [#46820](https://github.com/sgl-project/sglang/pull/46820) | `840ddc2dab5e` | 4 real-ZMQ CPU regression tests and scoped pre-commit hooks passed; range-diff unchanged | [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41402282474): Ready; missing maintainer-controlled run-ci label. [Lint](https://github.com/sgl-project/sglang/actions/runs/41402282458/job/125136503294): inherited XPU F821; repair main, rebase and rerun. |
| [#46830](https://github.com/sgl-project/sglang/pull/46830) | `3c70f61f2973` | 3 CPU regression tests plus 2 subtests and scoped pre-commit hooks passed; range-diff unchanged | [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41402508388): Ready; missing maintainer-controlled run-ci label. [Lint](https://github.com/sgl-project/sglang/actions/runs/41402508194/job/125137192923): inherited XPU F821; repair main, rebase and rerun. |
| [#33059](https://github.com/sgl-project/sglang/pull/33059) | `ab4407a57c8a` | 12 CPU regression tests and scoped pre-commit hooks passed; range-diff unchanged; requested AMD model matrix remains outstanding | [Base attempt 1](https://github.com/sgl-project/sglang/actions/runs/41402667950): Draft gate; requested AMD model matrix outstanding. [Lint](https://github.com/sgl-project/sglang/actions/runs/41402667672/job/125137681053): inherited XPU F821; repair main, rebase and rerun. |
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | `d973ced62c78` | Scoped pre-commit and CPU launch-selection assertions passed; 7 GPU tests skipped; unchanged patch, inherited main #47300 device-alias fix | [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41402860632): Ready; missing maintainer-controlled run-ci label. [Lint](https://github.com/sgl-project/sglang/actions/runs/41402860388/job/125138284937): inherited XPU F821; repair main, rebase and rerun. |
| [#47278](https://github.com/sgl-project/sglang/pull/47278) | `607b3c9f76f1` | Scoped pre-commit and 3 CPU-only contract checks passed; 22 ROCm tests skipped; PR-touched files byte-identical to previous head | [Base attempt 2](https://github.com/sgl-project/sglang/actions/runs/41403072135): Ready; missing maintainer-controlled run-ci label. [Lint](https://github.com/sgl-project/sglang/actions/runs/41403072008/job/125138927796): inherited XPU F821; repair main, rebase and rerun. |

All five scoped pre-commit runs passed. All five full upstream lint runs failed on the same unchanged main XPU file; these are different scopes and are reported separately. No new GPU kernel/model test or serving benchmark was performed.

## Why Base still fails

The gate fetches the live PR draft flag and labels. The older Draft-gated results remained visible because promotion did not create a new run. Pushes generated fresh runs; supported `/tag-and-rerun-ci` comments then produced attempt 2 for all four Ready PRs. Their Draft gate now skips, but `Require run-ci label` fails. `jit-kernel` is not equivalent to `run-ci`.

The account has repository READ permission. The bot grants authors failed-workflow reruns, but reports `Permission denied: can_tag_run_ci_label is false`. A successful slash-command workflow or a thumbs-up reaction does not mean the label was granted or Base tests passed. [Observed bot log](https://github.com/sgl-project/sglang/actions/runs/41402484529).

A maintainer with CI-label permission must add `run-ci` and issue `/tag-and-rerun-ci` on #46820, #46830, #47276 and #47278. Do not dispatch a main-only workflow and call it a fork-head result.

#33059 already has `run-ci`, but remains legitimately Draft. HaiShaw's DeepSeek-V3.2 MI300/MI355 DP-attention/stage-c matrix remains outstanding. A CI-authorized maintainer can use the targeted `/rerun-test` facility for the requested suites at the exact head, or an authorized delegate can use an allocated pool node. After evidence and review gaps are resolved, mark Ready and rerun Base. Repeated Draft-gated retries cannot supply this evidence.

## Independent main lint regression

Merged #30980 adds `logger.info` to `python/sglang/srt/hardware_backend/xpu/graph_runner/xpu_graph_runner.py:116` without defining the logger. Ruff reports `F821`. The file is unchanged by these five PRs and has main blob `642e326950a6fc81124c3ca0c5fe32a21246f977`. The omission also persists on main `b0debe0a7aa827b34c1987a234c07eeedf14c68f`, checked after the rebases. [Source-owner report, CI logs and minimal repair](https://github.com/sgl-project/sglang/pull/30980#issuecomment-6571123990).

```bash
pre-commit run ruff --files python/sglang/srt/hardware_backend/xpu/graph_runner/xpu_graph_runner.py
```

Expected result on that main: F821 at line 116. The upstream author/maintainers should restore the module logger in one focused repair. Rebase these PRs and rerun repository lint after it lands; do not duplicate an unrelated XPU repair across five patch PRs.

## Validation commands and boundaries

```bash
# Run each command on its corresponding PR head.
PYTHONPATH=python python -m pytest -q test/registered/unit/managers/test_kv_metrics_backpressure.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py
PYTHONPATH=python python -m pytest -q -rs test/registered/amd/test_dsa_gfx950_fused_indexer.py
PYTHONPATH=python python -m pytest -q -rs test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py
# Repository lint for the PR-touched paths:
pre-commit run --files <paths-from-the-corresponding-PR>
```

CPU results are 4 tests for #46820, 3 tests plus 2 subtests for #46830, and 12 tests for #33059. Kernel suites collect but skip 7 and 22 cases respectively on CPU; those are not numerical GPU passes. The existing pure CPU contracts were additionally invoked directly without GPU fixtures:

```python
# On #47276, with PYTHONPATH=python:
import runpy
from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader
suite = runpy.run_path("test/registered/amd/test_dsa_gfx950_fused_indexer.py")
case = suite["TestGfx950FusedIndexerKernels"]("test_logits_launch_selection")
case.loader = loader
case.test_logits_launch_selection()
```

```python
# On #47278, with PYTHONPATH=python:
import runpy
import pytest
suite = runpy.run_path("test/registered/kernels/ops/attention/test_triton_sparse_mla_hip.py")
for name in ("test_cu_count_uses_tensor_device", "test_gfx950_fp8_gate_supports_tp8_prefill"):
    with pytest.MonkeyPatch.context() as monkeypatch:
        suite[name](monkeypatch)
suite["test_page_offset_i32_boundary"]()
```

#47278's two touched files are byte-identical to its pre-rebase head. #47276 inherits main #47300's `kDLCUDA` to `kDLGPU` wrapper fix; its other five touched files are byte-identical. This source comparison preserves the evidence boundary and is not a new hardware result. Overlapping #45307 remains open.

## External dependency and follow-up ownership

[#45524](https://github.com/sgl-project/sglang/pull/45524) remains Ready at `14c8591329db003317f0bfddc26335090ad92ea7`, owned by eaglewangao-us. Its [Base run](https://github.com/sgl-project/sglang/actions/runs/40803758455) fails missing `run-ci`. [Requested author rebase and maintainer labeling/rerun](https://github.com/sgl-project/sglang/pull/45524#issuecomment-6571065983). The current patch covers non-final PP-chunk producer ordering; the author dropped the cached-prefix change on October 29 after 72/72 baseline checks. No duplicate PR or branch mutation was made.

Yihou/delegate: rebase after the shared main lint repair, assess actual functional CI results and respond to review. CI maintainers: labels and targeted AMD dispatch. XPU author/maintainers: main logger repair. #33059 validation delegate: requested model matrix. Infera [#228](https://github.com/AMD-AGI/Infera/pull/228) remains permanently Draft, for view and track only. The daily ledger refresh publishes these heads and handoff actions; it does not automatically perform GPU validation or promote runtime PRs.

## Exact branch updates

| PR | Previous head | Pushed head |
|---|---|---|
| #46820 | `82176e70284c8d67b41de74b8e8005d148e57c8b` | `840ddc2dab5e6bad4db7686e7d610a97a3310539` |
| #46830 | `8bd9fe8c3b1612519e81891a9dc12c5237664eec` | `3c70f61f29731eb3b1a9da5e6b15be862b1f96cd` |
| #33059 | `ea88d74bfc2c934159e4fc73a258861f3c5d96e2` | `ab4407a57c8a7571dd48b4a89f6e5a0359d0248c` |
| #47276 | `23568fdf24f3c70b057f71ba0ca3aee94ab594b7` | `d973ced62c7836931590756af8a9e3e7f6357fc2` |
| #47278 | `375f052eda199e2b7fc56ae4adc61b79b685d384` | `607b3c9f76f1e146920384938a1aea9de523e0bd` |
