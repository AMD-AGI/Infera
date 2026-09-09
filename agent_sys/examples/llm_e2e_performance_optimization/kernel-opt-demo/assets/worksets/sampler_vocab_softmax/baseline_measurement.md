# Baseline — the number every speedup claim is measured against

Protocol: **5 rounds × 30 timed iterations**, each round a **fresh process**,
one MI300X card, HIP-graph replay timing. Re-run it with:

```bash
cd kernel && python3 measure_baseline.py --json baseline.json
```

## Measured 2026-09-01, `smc300x-ccs-aus-a17-31`, `HIP_VISIBLE_DEVICES=0`

Raw per-round medians, in µs:

| case | round 1 | 2 | 3 | 4 | 5 | **median** | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B1_V151936` | 54.48 | 54.52 | 54.16 | 55.36 | 54.76 | **54.52** | 2.2% |
| `B8_V151936` | 55.52 | 55.32 | 54.96 | 55.92 | 55.40 | **55.40** | 1.7% |
| `B32_V151936` | 60.29 | 59.93 | 59.61 | 60.73 | 60.05 | **60.05** | 1.9% |

Correctness on the same build, all three cases against a float64 reference:
SNR 135.01 / 123.51 / 124.51 dB, `allclose` true on all three.

## The cross-check, which is the point of this file

The live sglang DECODE trace measured `cunn_SoftMaxForwardGmem` at
**55.59 µs/call** at `[8, 151936]`. This standalone driver measures
**55.40 µs** for `B8_V151936`. That is **0.3%**.

Without that agreement, a speedup measured here would be a speedup over
*something*, with no evidence it is the thing production runs. With it, an *N*×
here is an *N*× on the traced kernel. This check is cheap and it is the one step
that must not be skipped when the workset is re-pointed at a different operator.

## Prior measurement, 2026-08-31, kept for comparison

Same host, same image, `HIP_VISIBLE_DEVICES=3`, 4 rounds:

| case | median | spread |
|---|---:|---:|
| `B1_V151936` | 55.72 µs | 0.3% |
| `B8_V151936` | 55.64 µs | 0.3% |
| `B32_V151936` | 60.53 µs | 0.4% |

**The two runs agree on the medians to within 2%** (`B8`: 55.64 → 55.40) and
**disagree on the spread by roughly 5×** (0.3% → ~2%). The medians are the
reproducible part; the spread is not, and the difference has not been explained.
Candidates not tested: a different card, a different set of neighbours on the
shared host, a different clock state. Recorded as an open question rather than
attributed.

## What this means for anyone comparing against these numbers

- **Compare medians of ≥5 rounds, never single measurements.** On 2026-08-31 a
  single re-measurement of an optimized kernel returned 21.67 µs where the
  median over four rounds was 18.9 µs, and the single sample was written up as a
  regression before the repeat showed it was an outlier.
- **The optimized side is noisier than the baseline.** Measured 2026-08-31, an
  optimized replacement spread ~8% round to round against this baseline's 0.3%.
  Cause not identified.
- **A speedup under ~1.05× is not distinguishable from noise at this spread.**
  If that is where a result lands, say so; do not report it as an improvement.
