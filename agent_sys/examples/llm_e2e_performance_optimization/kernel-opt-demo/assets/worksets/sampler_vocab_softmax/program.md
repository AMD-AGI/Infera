# Program: optimize the sglang sampler vocabulary softmax

**GPU**: gfx942 (AMD Instinct MI300X)
**Backend**: triton
**Provenance**: real hot kernel, traced from a live sglang v0.5.14 decode step

## Where this came from

An sglang server running `Qwen/Qwen3-0.6B` (TP1, triton attention backend) was
profiled with `torch.profiler` via `/start_profile` (`profile_by_stage=true`,
`num_steps=20`, `record_shapes=true`) while 8 concurrent requests decoded.
Ranking the DECODE-stage trace by GPU time gives:

| GPU time | share | calls | us/call | kernel |
|---------:|------:|------:|--------:|:-------|
| 1153.6 us | 15.05% | 20 | 57.68 | `reduce_kernel<..., ArgMaxOps<float>, ...>` |
| **1111.9 us** | **14.50%** | **20** | **55.59** | **`cunn_SoftMaxForwardGmem<4, float, float, float, ...>`** |
| 696.1 us | 9.08% | 20 | 34.81 | `reduce_kernel<..., func_wrapper_t<float>, ...>` |
| 252.3 us | 3.29% | 56 | 4.51 | `aiter::add_rmsnorm_quant_kernel<...,128,8,...>` |

The CPU-side op with `record_shapes` resolves the second row exactly:

```
aten::softmax   Input Dims = [[8, 151936], [], []]   types = ["float", ...]
aten::_softmax  Input Dims = [[8, 151936], [], []]
```

which is `python/sglang/srt/layers/sampler.py:183`:

```python
logits[:] = torch.softmax(logits, dim=-1)
```

So on a small model the decode step is dominated by the *sampler*, not by
attention or the MoE/GEMM path — three vocabulary-wide ATen reductions are ~39%
of decode GPU time.

## Objective

Make `sampler_softmax(logits, out)` in `sampler_softmax_kernel.py` faster than
the ATen baseline while staying numerically correct. The loop gates correctness
on SNR (>= 30 dB) plus an exact `allclose` and a rows-sum-to-1 check, before it
ever benchmarks a change.

## The headroom argument

Shape `[8, 151936]` fp32 moves `8*151936*4 = 4.86 MB` in and the same out:
**9.72 MB per call**. At the measured 55.59 us that is **~175 GB/s**. MI300X HBM
peak is ~5.3 TB/s. A bandwidth-saturating single-pass softmax should be
roughly an order of magnitude faster; the ATen path here is
`cunn_SoftMaxForwardGmem`, the *global-memory fallback* ATen selects when the
reduced dimension is too large for its shared-memory kernel — it makes several
passes over the row.

## Optimization ideas (not prescriptions — measure everything)

- One row per program with an online (streaming) max+sum pass, so the row is
  read once for statistics and once for the write, instead of ATen's multi-pass.
- Tune `BLOCK_SIZE`, `num_warps` (AMD wave size is 64), and `num_stages` for a
  151936-wide row; the row does not fit in LDS, so it must be tiled.
- Vectorized loads/stores over the contiguous vocab dimension.
- With only 1-32 rows there is very little parallelism in the batch dimension —
  a split-row (two-stage) reduction may be needed to fill 304 CUs, especially
  for `B1`.

## Modification rules

1. Keep the public `sampler_softmax(logits, out)` signature — the driver imports it.
2. `out` is pre-allocated and must be written in place; return it.
3. Keep the kernel in Triton (or plain torch); do not add a build step.
4. Do NOT edit `driver.py` or `graph_harness.py` — they are the measurement
   oracle and the loop blocks edits to them. Optimize the kernel, not the
   measurement.
5. Correctness bar is tight on purpose: probabilities feed multinomial sampling,
   so rows must still sum to 1 within 1e-4 and match a float64 reference to
   `atol=1e-6, rtol=1e-3`.
6. Verify your change yourself before finishing; the loop then runs its own
   canonical correctness + benchmark pass and keeps the change only if it is
   correct AND faster than the current best.
