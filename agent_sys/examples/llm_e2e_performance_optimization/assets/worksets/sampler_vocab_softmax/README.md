# Workset — `sampler_vocab_softmax`

A **workset** is everything needed to test, optimize and re-integrate one
operator, in one directory. It is stage 3's deliverable and stage 4's input
(series task book, `temp/mission.md` §Goal 3). Stage 3 does not exist yet, so
this one ships as package data and `ingest_workset` publishes it as a handoff.
When stage 3 lands, it produces a directory of this shape and this copy goes
away.

Every number below was **measured**, on 2026-08-31, on the host named in
`environment.md`. Nothing here is estimated.

## The operator

| | |
|---|---|
| Name | `sampler_vocab_softmax` |
| Framework | sglang v0.5.14 (commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb`) |
| Integration point | `python/sglang/srt/layers/sampler.py:183` |
| Source line | `logits[:] = torch.softmax(logits, dim=-1)` |
| Signature | `sampler_softmax(logits: Tensor[B, V] fp32, out: Tensor[B, V] fp32) -> Tensor` |
| Vocabulary `V` | 151936 (`Qwen/Qwen3-0.6B`) |
| Batch `B` measured | 1, 8, 32 |
| Semantics | row-wise softmax, written into a pre-allocated `out`, returned |

`out` is pre-allocated and written in place because the production call site is
`logits[:] = ...` — an in-place write into the logits buffer. A replacement that
allocates would not be substitutable there.

## Provenance — how this operator was chosen

An sglang server running `Qwen/Qwen3-0.6B` (TP1, triton attention backend) was
profiled with `torch.profiler` through `/start_profile`
(`profile_by_stage=true`, `num_steps=20`, `record_shapes=true`) while 8 requests
decoded concurrently. Ranking the DECODE-stage trace by GPU time:

| GPU time | share | calls | µs/call | kernel |
|---:|---:|---:|---:|:--|
| 1153.6 µs | 15.05% | 20 | 57.68 | `reduce_kernel<…, ArgMaxOps<float>, …>` |
| **1111.9 µs** | **14.50%** | **20** | **55.59** | **`cunn_SoftMaxForwardGmem<4, float, float, float, …>`** |
| 696.1 µs | 9.08% | 20 | 34.81 | `reduce_kernel<…, func_wrapper_t<float>, …>` |
| 252.3 µs | 3.29% | 56 | 4.51 | `aiter::add_rmsnorm_quant_kernel<…,128,8,…>` |

`record_shapes` resolves row 2 to a CPU op exactly:

```
aten::softmax    Input Dims = [[8, 151936], [], []]    types = ["float", "Scalar", ""]
aten::_softmax   Input Dims = [[8, 151936], [], []]
```

On a small model the decode step is dominated by the **sampler**, not by
attention or the GEMM path: three vocabulary-wide ATen reductions are ~39% of
decode GPU time, attention ~2%.

**Profile-report number, for cross-checking:** `55.59 µs/call`.

## Contents

```
README.md                     this file — the operator definition
environment.md                hardware, image, versions, extra installs
program.md                    the brief KernelForge is given: objective, headroom, rules
integration.md                where it plugs back into sglang, and what may not change
baseline_measurement.md       the 5×(≥10-iteration) baseline, and how to redo it
kernel/
  sampler_softmax_kernel.py   the PyTorch-naive implementation == the measured baseline
  driver.py                   one-shot correctness + benchmark + profile driver
  graph_harness.py            HIP-graph replay timing used by driver.py
  measure_baseline.py         the 5×10 protocol, re-runnable
```

## Correctness cases

Three, all in `driver.py`, all against a **float64** reference:

| case | shape | why this one |
|---|---|---|
| `B1_V151936` | `[1, 151936]` | single-sequence decode. Worst case for parallelism — one row over 304 CUs |
| `B8_V151936` | `[8, 151936]` | **the traced production shape** |
| `B32_V151936` | `[32, 151936]` | larger batch, where a per-row strategy starts to pay |

Bars, all three enforced before any timing is believed:

- SNR ≥ 30 dB against the float64 reference (the loop's own gate);
- `torch.allclose(atol=1e-6, rtol=1e-3)`;
- every row sums to 1 within `1e-4`.

The last one is not decoration: these probabilities feed `torch.multinomial`, so
a row that does not sum to 1 is a sampler that draws from the wrong
distribution — which is a correctness bug that no SNR threshold catches.

## Baseline

See `baseline_measurement.md` for the protocol and the raw numbers. Measured
2026-09-01, 5 rounds × 30 iterations, each round a fresh process:

| case | baseline median | spread over 5 rounds |
|---|---:|---:|
| `B1_V151936` | 54.52 µs | 2.2% |
| `B8_V151936` | 55.40 µs | 1.7% |
| `B32_V151936` | 60.05 µs | 1.9% |

`B8` at 55.40 µs against the trace's 55.59 µs/call is a **0.3% agreement**, and
that agreement is the point of the whole file: it establishes that this
standalone driver measures the same work the production path does. Without it
every later speedup number is unanchored.

An earlier run on 2026-08-31 gave 55.72 / 55.64 / 60.53 µs at a spread of 0.3%.
The medians agree to within 2%; the **spread differs by ~5×** and that is not
explained. Consequence for a reader: compare medians over ≥5 rounds, and treat
anything under ~1.05× as indistinguishable from noise.

## One-click use

```bash
cd kernel
python3 driver.py                                        # correctness, all 3 cases
python3 driver.py --bench-mode --warmup 10 --iters 30    # timing, all 3 cases
python3 measure_baseline.py                              # the full 5×10 protocol
```

`driver.py` prints only to stdout, in the contract KernelForge's forge-loop
parses (`SNR: <db> dB`, `case_snr:`, `case_allclose:`, `wall_ms:`, `case_ms:`).

## Boundary — what this workset does not carry

- **No re-integration patch.** Nothing here has been put back into sglang, and
  no end-to-end throughput or latency claim is made. That is stage 5.
- **No multi-GPU, no TP > 1.** Single card, single process.
- **fp32 only.** The traced call is fp32; bf16/fp16 logits are not covered.
- **Vocabulary is fixed at Qwen3-0.6B's 151936.** A different vocabulary is a
  different shape and may well want a different strategy.
- **The other two hot kernels in the table above are not in this workset.**
  `ArgMaxOps` is hotter and is not being optimized here.
