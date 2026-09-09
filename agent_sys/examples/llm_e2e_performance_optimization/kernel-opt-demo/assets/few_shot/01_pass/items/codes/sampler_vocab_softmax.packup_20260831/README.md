# sglang sampler vocabulary softmax — KernelForge campaign

## Result

**It worked.** The ATen `cunn_SoftMaxForwardGmem` path was replaced by a
three-kernel segmented online softmax in Triton.

| | baseline | optimized | speedup |
|---|---:|---:|---:|
| `B1_V151936` | 54.52 µs | 21.31 µs | 2.558× |
| `B8_V151936` (traced shape) | 55.40 µs | 20.11 µs | 2.755× |
| `B32_V151936` | 60.05 µs | 23.80 µs | 2.524× |
| **mean case speedup** | | | **2.612×** |

Correctness: SNR **138.12 dB** against a float64 reference, `allclose` true on
all three cases, every row summing to 1 within 1e-4.

How I know: `results/verification.json` holds my own re-measurement — 5 rounds
per side, each round a fresh process, medians compared. KernelForge's own report
said 2.8328×; I measured 2.6123×. Both numbers are in the kit and I believe
mine, because the spread on the optimized side is large enough (19–21%) that a
single campaign's measurement can land either way. I did not resolve which
effect causes that spread.

Root cause of the original slowness, from the campaign's PMC data: ATen issues
**one workgroup per batch row**, so at `B8` exactly 8 of the MI300X's 304 CUs
had work — 97.4% idle. Per-workgroup bandwidth was flat at 20–22 GB/s across an
8× batch span, which says the problem was parallelism and not memory efficiency.
The fix issues ~1216 workgroups and pays for it by reading the logits twice.

## Purpose

A reproduction kit for a KernelForge campaign against the sglang sampler's
vocabulary softmax, `[B, 151936]` fp32, on MI300X / gfx942.

## Interface

- `results/optimized_kernel.py` — the artefact. Public entry point
  `sampler_softmax(logits, out)`, writes in place into `out`, returns it.
- `results/forge_result.json` — the campaign's own verdict.
- `results/verification.json` — my independent re-measurement.
- `results/candidates_index.jsonl` — one line per iteration, with the reason
  each reverted change was reverted.
- `REPRODUCE.md` — the entry point for re-running it.

A re-integrator must preserve the signature, the in-place write, fp32, and a
contiguous vocabulary dimension. See the workset's `integration.md`.

## Boundary

What is **not** established here:

- **No end-to-end claim.** This kernel is 14.5% of decode GPU time, so Amdahl
  bounds the decode-GPU saving at ~9.3% and the service-level effect is smaller
  again. Nobody measured it.
- **fp32 and V=151936 only.** No bf16, no other vocabulary, no TP > 1.
- **The 19–21% round-to-round spread on the optimized side is unexplained.**
  Candidates not tested: multi-kernel launch jitter, Triton JIT cache state,
  neighbour interference on a shared host.
- **No GPU clock lock and no exclusive reservation.** Other tenants were idle
  but present.
- The two hotter sibling kernels in the same sampler (`ArgMaxOps` at 15.05%)
  were not touched.
