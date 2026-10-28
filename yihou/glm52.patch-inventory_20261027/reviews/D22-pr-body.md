## Problem and fix

On gfx950, GLM's BF16 `kv_b_proj` is automatically converted to FP8. The replicated-Q preparation accepts BF16/FP16 weights, so that conversion prevents `--dcp-replicate-q-proj` from preparing full-head Q weights and avoiding its per-layer Q all-gather.

Add opt-in `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1` to retain BF16 only for GLM target weights on DCP ranks. Default behavior, DCP1 prefill, other architectures, non-BF16 weights and non-gfx95 paths stay on their existing policy. Rebased D22 (`636cc68dd0`) onto main `8a455f1`.

## Evidence

Historical isolated `benchrun T10`, 34K context, recorded step-time reductions of 0.4/1.6/1.7/1.7 ms at N=8/16/24/32, at a cost of about 0.34 GB/GPU in weights. This is the archived combined DCP implementation, not a new benchmark of this main-based revision. No stack throughput is attributed to this flag. Historical serving throughput used simulated MTP acceptance and is not model-correctness evidence.

## Validation

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/models/test_glm_dcp_kv_b_dtype.py
pre-commit run --hook-stage pre-commit --files python/sglang/srt/environ.py python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py test/registered/unit/models/test_glm_dcp_kv_b_dtype.py
```

CPU policy matrix: **1 passed, 72 subtests passed**, covering platform, architecture, dtype, DCP width and flag state. This tests dispatch policy, not numerical equivalence or serving performance.

## Draft blockers

- Coordinate the final ROCm DSA DCP/Q-projection implementation in #44322/#42618; for PD also #46824 and #46826. This option does not enable that backend by itself.
- Fresh final-stack weight-memory, Q-rep activation, isolated latency and real-acceptance model accuracy checks. Revalidate both DCP1 and DCP>1 and default-off behavior; the existing DP2×DCP2 graph garbling remains unbisected.
- Maintainer agreement on the opt-in interface, upstream CI and review.
