## Motivation

With DP-attention / MAX_LEN padding, `Indexer._get_topk_paged` receives physical query and weight rows beyond the real query count `q_offset`. Aiter derives its logits row count from the query tensor, while top-k uses the real-row sequence lengths. Passing the padded inputs can therefore fail the top-k row-count check (`lengths.size(0) == B`).

The existing padding restoration also excludes HIP. Simply slicing the Aiter inputs would leave downstream consumers with fewer rows than the physical query tensor.

## Modifications

- Slice Aiter query and weight inputs to `q_offset`, preserving the real-row metadata contract already used by the other paged backends.
- Restore trailing `-1` top-k rows on HIP as well.
- Return an all-`-1`, int32 result directly when the real query count is zero. Passing an empty slice to Aiter is insufficient: its [SplitKV scheduling](https://github.com/ROCm/aiter/blob/7559e3411beecb04d2f64ac488d6990a3b03534e/aiter/ops/triton/attention/pa_mqa_logits.py#L711) divides by the query count. The empty path launches neither Aiter nor top-k.
- Add a registered CPU regression covering legacy and preshuffle page layouts, padded/unpadded decode, empty and padding-only queries, target verification, and draft-extend-v2. It checks actual row contents, valid top-k results, output shape/dtype, padding, and empty-query launch avoidance.

Related work: #34455 addressed a different metadata-construction problem and is closed without merging. #32175 contains CP-specific metadata padding in a broader release-based enablement PR; that approach must be reconciled with this caller's real-row contract if both are pursued. This updates the existing #33059 rather than opening another PR.

## Validation

Base: upstream main `449617bc94a5d4ae23ad62d0da2f842e711ca983`.
Local environment: Python 3.13.13, PyTorch 2.14.1+cpu, Transformers 5.19.0, compressed-tensors 0.18.0, apache-tvm-ffi 0.1.11.

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py
pre-commit run --files python/sglang/srt/layers/attention/dsa/dsa_indexer.py test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py
```

CPU run `pr33059-cpu-20261027`:

- Patched implementation: **12 passed**.
- The same tests with `dsa_indexer.py` restored to unpatched main: **10 failed, 2 passed**. The failures expose padded/empty-row handling; both unpadded controls pass.
- All applicable repository pre-commit hooks passed.

These are dispatch/shape contract tests with a stubbed Aiter kernel and top-k transform. They exercise the production `_get_topk_paged` method but **do not establish GPU kernel correctness or distributed model accuracy**.

## Remaining before approval

The AMD validation requested in the August 7 review is still outstanding. No current-main MI300/MI355 DP-attention accuracy result is claimed, and historical v0.5.15 deployment reports are not substituted for it. Please execute the requested suites on this head:

```bash
python test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py
python test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py
python test/registered/amd/test_deepseek_v32_basic.py
```

Complete AMD and NVIDIA CI execution is also pending. CPU coverage and skipped/gated checks do not resolve that review request.

## Performance

No performance claim. Historical GLM-5.2 stack throughput used simulated MTP acceptance and bundled unrelated changes; it is not evidence for this patch. The October 27 llr35 bisect implicated PP communication overlap and decode custom all-reduce, so stack-level success cannot establish this patch's correctness.

## Checklist

- [x] Registered targeted CPU regressions; verified failure on unpatched main.
- [x] Repository formatting and lint checks.
- [x] No new configuration or user-facing documentation needed.
- [ ] Current-head AMD DP-attention accuracy and complete upstream CI.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:no_entry_sign: [Run #40473090609](https://github.com/sgl-project/sglang/actions/runs/40473090609)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40473090630](https://github.com/sgl-project/sglang/actions/runs/40473090630)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:hourglass_flowing_sand: [Run #40473090503](https://github.com/sgl-project/sglang/actions/runs/40473090503)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->

## Draft blockers

The requested MI300/MI355 distributed DeepSeek-V3.2 DP-attention accuracy and stage-c validation remain outstanding. Current CI also has infrastructure/dependency failures; local CPU contracts do not replace model accuracy checks.

Kept as draft until these validation/review gates are satisfied.
