# Forge Optimization Report

- Campaign: `76ecef370dbe43c69115863e7fc51fbc`
- Status: best verified result
- mean case speedup: 2.832760x
- Baseline raw mean: 0.0562 ms
- Search-start raw mean: 0.0562 ms
- Selected candidate raw mean (diagnostic; not monotonic): 0.0186 ms
- Correctness: PASS
- Best iteration: 5
- Commit: `381188448145f272d2bfca61847c54f44e208303`
- Optimization: fold combine kernel into normalize for small segment counts (B8/B32), keep 3-kernel path for B1

## Changed Files

- `sampler_softmax_kernel.py`

## Artifacts

- Patch: `best/iter_005/forge.patch`
- Validation: `best/iter_005/validation.txt`
- Benchmark: `best/iter_005/benchmark.json`
- Bundle: `best/iter_005`
