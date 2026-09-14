# Kernel optimization — sglang sampler vocabulary softmax

## Purpose

A KernelForge campaign against one traced sglang decode hot kernel — the
sampler's vocabulary softmax, `[B, 151936]` fp32 — with the optimized kernel,
its correctness evidence, an independent re-measurement, and a reproduction kit.

## Interface

The packup is `items/codes/sampler_vocab_softmax.packup_20260831/`.
Start at its `REPRODUCE.md`.

A consumer wanting only the artefact takes
`items/codes/sampler_vocab_softmax.packup_20260831/results/optimized_kernel.py`.
Its public entry point is `sampler_softmax(logits, out)`: it writes in place
into the pre-allocated `out` and returns it, expects fp32 with a contiguous
vocabulary dimension, and adds no build step.

A consumer wanting to judge the claim reads `results/verification.json` — the
producer's own re-measurement — rather than `results/forge_result.json`, which
is the campaign's self-report.

## Boundary

- **Kernel level only.** No integration patch, no end-to-end latency,
  throughput or accuracy measurement. The kernel is 14.5% of decode GPU time, so
  the end-to-end effect is bounded well below the kernel speedup and was not
  measured.
- **One shape family.** fp32, `V = 151936`, `B ∈ {1, 8, 32}`, single GPU.
- **Two numbers disagree and neither was reconciled.** Forge reported 2.8328×;
  the next-day re-measurement was 2.6123×.
- **A 19–21% round-to-round spread on the optimized side is unexplained.**
- Nothing here reserves a GPU, and no clock was locked.
