# PR #47872: candidate-underflow reproduction

This bundle preserves the 2026-11-02 CPU and subsequent MI355X GPU evidence for
the underflow repair in [SGLang #47872](https://github.com/sgl-project/sglang/pull/47872).
The GPU run reproduced the original failure and passed the repaired full kernel
pipeline and all five GPU tests. See [GPU validation](#gpu-validation).

The earlier CPU interpreter check runs the actual Triton `_topk_est` and
`_topk_select` functions with CPU tensors under `TRITON_INTERPRET=1`. Candidate
packing between those functions is implemented on the CPU. **That CPU check does
not launch the Gluon collector or validate GPU compilation or execution.** The
later GPU evidence is separate. Neither check establishes end-to-end model
accuracy, server integration, or performance.

## Pinned source and results

The source path in both revisions is
`python/sglang/kernels/ops/attention/dsa/triton_topk_transform.py`.

| Revision | Immutable source | SHA-256 of source file | Recorded result |
| --- | --- | --- | --- |
| Original submission | [`a899828aaf4fa1ee2013c072a19be9115b773c09`](https://github.com/sgl-project/sglang/blob/a899828aaf4fa1ee2013c072a19be9115b773c09/python/sglang/kernels/ops/attention/dsa/triton_topk_transform.py) | `29f9a820cb2ff2cd14963fceb4385222651703a6e94af59721d267c3b042e12b` | [original.json](original.json) |
| Underflow repair | [`1061c298389facbdeaafced9891a5c7c9b399331`](https://github.com/sgl-project/sglang/blob/1061c298389facbdeaafced9891a5c7c9b399331/python/sglang/kernels/ops/attention/dsa/triton_topk_transform.py) | `9b29d2a628eaaa8df4d77bac17db507475cafd866b676029730d9bb76c9bbe13` | [fixed.json](fixed.json) |

The CPU JSON records retain the measured case fields from the original local runs;
private filesystem locations were replaced with immutable source provenance.
They are not additional executions of this public script. Package versions were
not captured in those records.

## What fails and why the repair works

Both cases use one FP32 row of length 131072, top-k 2048, an 8192-entry candidate
buffer, and an identity page table with page size 64. Logits are zero except for
the first 256 or 2048 estimator probe positions, which contain 100.0. The actual
estimator chooses the integer key for 100.0 as the threshold in both cases.

| Case | Candidates | Original valid / padding | Fixed valid / padding | Expected |
| --- | ---: | ---: | ---: | --- |
| Biased sample: 256 high probe positions | 256 | 256 / 1792 | 2048 / 0 | 256 high entries plus 1792 distinct zero entries |
| Positive control: 2048 high probe positions | 2048 | 2048 / 0 | 2048 / 0 | All 2048 high entries |

The original selector falls back to the whole row only when the candidate buffer
overflows. A candidate count below the requested k also requires this fallback: the
repair takes it when `n_all < min(L, TOPK)` or `n_all > CAND`. When the candidate
count is between those limits, every excluded key is below the threshold and the
candidate buffer contains a valid top-k.

The check requires 2048 valid, unique slots and the reference top-k score sum.
Checking only the sum would miss this failure: omitting the required zero-valued
entries does not change that sum. Tied zero entries may be chosen in any order.
This reproducer uses finite, nonnegative logits and an identity page table; it
does not establish every supported shape, sign pattern, or page-table mapping.

## Run the interpreter reproduction

Use an existing CPU Python environment with PyTorch and a Triton installation
that provides `triton.experimental.gluon` and supports `TRITON_INTERPRET=1`.
Gluon must be importable because the unmodified source defines the collector,
although the reproducer does not launch it. No GPU or SGLang installation is
needed for this script. Obtain both pinned commits in a local SGLang checkout.

Run these commands from this bundle's directory. Set `SGLANG_SOURCE` to that
checkout; extraction does not change its working tree or branch.

```bash
SGLANG_SOURCE=/path/to/sglang
mkdir -p yihou-validation
git -C "$SGLANG_SOURCE" show a899828aaf4fa1ee2013c072a19be9115b773c09:python/sglang/kernels/ops/attention/dsa/triton_topk_transform.py > yihou-validation/original_module.py
git -C "$SGLANG_SOURCE" show 1061c298389facbdeaafced9891a5c7c9b399331:python/sglang/kernels/ops/attention/dsa/triton_topk_transform.py > yihou-validation/fixed_module.py

# Expected exit code: 1. The biased-sample case fails; the control passes.
PYTHONDONTWRITEBYTECODE=1 TRITON_INTERPRET=1 python reproduce.py \
  --module yihou-validation/original_module.py \
  --output yihou-validation/original.json

# Expected exit code: 0. Both cases pass.
PYTHONDONTWRITEBYTECODE=1 TRITON_INTERPRET=1 python reproduce.py \
  --module yihou-validation/fixed_module.py \
  --output yihou-validation/fixed.json
```

The script hashes the supplied source and attaches the corresponding pinned
commit when the hash matches either recorded revision. Unknown source hashes
remain explicitly unpinned. It imports the source without modifying it. The
output directory must already exist.

## Separate import-guard regression

The maintenance revision also moves the ROCm and opt-in checks before importing
the optional kernel. The new
[CPU guard test](https://github.com/sgl-project/sglang/blob/1061c298389facbdeaafced9891a5c7c9b399331/test/registered/unit/layers/attention/test_dsa_triton_topk_gate.py)
checks `(HIP, enabled)` equal to `(false, true)`, `(true, false)`, and
`(false, false)`. All three cases failed against the original implementation and
passed after the repair. These are separate from the two interpreter cases.

On the repaired checkout, the recorded related suite produced **32 CPU tests
passed and 5 GPU tests skipped**. Reproduce with the repository's test dependencies
installed, from a checkout of the fixed commit:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/attention/test_dsa_triton_topk_gate.py \
  test/registered/unit/layers/attention/test_index_topk_share.py \
  test/registered/amd/test_dsa_triton_topk_transform.py
```

The [five AMD tests](https://github.com/sgl-project/sglang/blob/1061c298389facbdeaafced9891a5c7c9b399331/test/registered/amd/test_dsa_triton_topk_transform.py)
include the new underflow case. The CPU-host skips are not GPU validation; the
separate allocated ROCm run below executes all five without skips. This bundle
supplies no new throughput claim; the PR's original serving speed measurements
used simulated MTP acceptance and do not measure this repaired revision.

## GPU validation

Run `kf47872-20261102T1038Z-topk-underflow`, completed on 2026-11-02 and recorded at
10:40:42 UTC, used one allocated AMD Instinct MI355X (`gfx950:sramecc+:xnack-`).
It ran the actual estimator, Gluon collector, and selector through
`topk_transform_paged_triton`, followed by the complete GPU test class from each
pinned revision. The node was idle before the run and the task's container had
exited afterward.

| Environment | Recorded version |
| --- | --- |
| Public image tag | `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021` |
| Immutable image | `lmsysorg/sglang-rocm@sha256:b28fe77ae3d1c6022da1b579687a5f5ecec08269081301a4887d2093cfd34ae9` |
| PyTorch | `2.11.0+rocm10.0.0` |
| HIP reported by PyTorch | `7.15.26333` |
| Triton | `3.8.0` |

| Revision | Actual GPU underflow case | Full revision-specific GPU suite | Record |
| --- | --- | --- | --- |
| Original `a899828aaf4fa1ee2013c072a19be9115b773c09` | **Incorrect:** 256 valid, unique slots; 1792 padding slots | 4 passed, 0 skipped; these original tests did not expose the underflow | [gpu-original.json](gpu-original.json) |
| Fixed `1061c298389facbdeaafced9891a5c7c9b399331` | **Correct:** 2048 valid, unique slots; no padding; all 256 high-score positions retained | 5 passed, 0 skipped, including the new underflow regression | [gpu-fixed.json](gpu-fixed.json) |

The GPU kernel hashes match the table above. The exact test file is
`test/registered/amd/test_dsa_triton_topk_transform.py`:

| Revision | Test SHA-256 |
| --- | --- |
| [Original test](https://github.com/sgl-project/sglang/blob/a899828aaf4fa1ee2013c072a19be9115b773c09/test/registered/amd/test_dsa_triton_topk_transform.py) | `91956f146d02f9b445a0cdb90f563be07ed49e8ae713f548dabb6c50f522e467` |
| [Fixed test](https://github.com/sgl-project/sglang/blob/1061c298389facbdeaafced9891a5c7c9b399331/test/registered/amd/test_dsa_triton_topk_transform.py) | `a070739c016740904684518729747f269163fc8d5b85d421ea0a76e16cfca806` |

The fixed kernel and test snapshots were independently matched to the committed
`1061c298389facbdeaafced9891a5c7c9b399331` files. The JSON records preserve measured
fields and add public commit/image provenance; private node names and paths are
omitted. Their `elapsed_seconds` fields are test-run wall time, not kernel
benchmarks.

### Reproduce on an allocated MI355X GPU

[run_topk_gpu.py](run_topk_gpu.py) loads the exact extracted module and test file.
To run newer files in the older public image, it creates SGLang package namespace
entries and substitutes only the CI-registration hook with a no-op. It does not
replace Torch, Triton, Gluon, kernels, or test assertions. This covers GPU kernel
compilation and execution, including randomized page tables in the test class;
it does not exercise SGLang server dispatch or a full model deployment.

Use an allocated MI355X environment with the recorded image, expose the assigned
GPU as device 0, and run from this bundle's directory. The source extraction uses
the `SGLANG_SOURCE` checkout and `yihou-validation` directory from the CPU example.

```bash
git -C "$SGLANG_SOURCE" show a899828aaf4fa1ee2013c072a19be9115b773c09:test/registered/amd/test_dsa_triton_topk_transform.py > yihou-validation/original_tests.py
git -C "$SGLANG_SOURCE" show 1061c298389facbdeaafced9891a5c7c9b399331:test/registered/amd/test_dsa_triton_topk_transform.py > yihou-validation/fixed_tests.py

# Expected exit code: 0, because the known original underflow is required here.
PYTHONDONTWRITEBYTECODE=1 TRITON_INTERPRET=0 python run_topk_gpu.py \
  --module yihou-validation/original_module.py \
  --tests yihou-validation/original_tests.py \
  --output yihou-validation/gpu-original.json --expect-underflow

# Expected exit code: 0; correct underflow result and all five tests must pass.
PYTHONDONTWRITEBYTECODE=1 TRITON_INTERPRET=0 python run_topk_gpu.py \
  --module yihou-validation/fixed_module.py \
  --tests yihou-validation/fixed_tests.py \
  --output yihou-validation/gpu-fixed.json
```

The runner rejects interpreter mode and requires a ROCm `gfx950` GPU. It fails if
the expected counterexample result is absent, any test fails, or any test is
skipped. This is an isolated exact-source GPU validation; it supplies no full
server, model accuracy, or performance result.
