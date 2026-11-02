# B8: reproduce the absorbed-Q BMM compiler failure and validate SGLang #46918

This standalone bundle exercises the production `rocm_absorb_q_bmm` dispatcher
from [SGLang #46918](https://github.com/sgl-project/sglang/pull/46918), at exact head
`9763ef56bf57d03f8af4f9adc9eebb717cd0d9b1`, against
[issue #46829](https://github.com/sgl-project/sglang/issues/46829). The candidate
caps the hand-selected `waves_per_eu` and provides a configuration escape hatch.
The harness does not replace the production dispatcher, configuration selection,
or AITER kernel.

## Recorded environment and scope

| Component | Recorded version |
| --- | --- |
| GPU | One allocated AMD Instinct MI355X, `gfx950` |
| Container tag | `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261011` |
| Container digest | `sha256:fa0b69936d3e91cb42af9a2c499f35c4602e9c580a58b49214cb67b0f00b973b` |
| SGLang source under test | PR #46918 head `9763ef56bf57d03f8af4f9adc9eebb717cd0d9b1` |
| AITER source | `e7d2453f25e5aaeea72a3a40b284ba453e026624` |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| HIP | `7.2.26015` |
| Triton | `3.7.0` |

The container supplies the ROCm/compiler/AITER environment. The SGLang Python
source is the current candidate head above, imported from a separate checkout;
this is not a request to backport the change to SGLang 0.5.21. Reproducing the
compiler failure benefits from retaining the original compiler environment.
The harness records actual software versions and SHA256 fingerprints of the
imported SGLang dispatcher, AITER wrapper, and underlying Triton implementation.

CPU validation at this candidate passed **11 tests and 265 subtests**, using:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q -p no:cacheprovider \
  test/registered/unit/models/test_rocm_mla_absorb_fusion_gate.py
```

Run this CPU command from the candidate checkout in an environment with its test
dependencies. GPU results must be read from the accompanying validation report
and each run's `summary.json`; a test command or this procedure alone is not
evidence of a GPU pass.

## Prerequisites

- Use one allocated GPU, exposed as exactly one visible device, without live
  experiments sharing it. No model weights, network service, or distributed
  workers are needed. The harness rejects non-ROCm and non-`gfx950` devices.
- Prepare a clean candidate checkout at the exact SHA above. The harness checks
  both the commit and tracked Python changes, then verifies the imported SGLang
  module comes from this checkout. It enables AITER and rejects a Torch fallback.
- Keep the candidate, AITER source copy, and runtime artifacts under paths whose
  normalized names contain `yihou`, preserving the bundle's artifact-path guard.
  Use a mounted data disk or NFS filesystem with sufficient capacity. The
  examples below use `/data/yihou/b8`; replace that prefix consistently.
- Create a run-owned copy of the recorded AITER source tree from the image,
  including its Python package and C++ sources. Set both `AITER_ROOT_DIR` and
  `PYTHONPATH` to that source root. `AITER_ROOT_DIR` alone does not relocate the
  imported Python package; leaving `PYTHONPATH` unchanged can import a different
  installation or allow writes beside the original package. Do not substitute an
  empty directory for AITER sources.
- Set `AITER_CONFIG_GEMM_BF16` explicitly to the existing source CSV. The recorded
  AITER version otherwise attempts a default CSV write under the system temporary
  directory. An explicit existing CSV avoids that side effect.
- Use a fresh output directory for each invocation. Previous results are never
  overwritten or removed. Keep launcher logs and container runtime artifacts on
  the same approved artifact filesystem.

## Setup and commands

Inside the allocated environment, after preparing the AITER source copy and
candidate checkout:

```bash
mkdir -p /data/yihou/b8/launcher-tmp /data/yihou/b8/launcher-cache
export AITER_ROOT_DIR=/data/yihou/b8/aiter-source
export AITER_CONFIG_GEMM_BF16=/data/yihou/b8/aiter-source/aiter/configs/bf16_tuned_gemm.csv
export PYTHONPATH=/data/yihou/b8/aiter-source
export TMPDIR=/data/yihou/b8/launcher-tmp
export XDG_CACHE_HOME=/data/yihou/b8/launcher-cache
export PYTHONDONTWRITEBYTECODE=1
```

Keep the GPU visibility assigned by your allocator. The harness does not alter
it. Use the environment's Python executable and the location of this bundle:

```bash
python /data/yihou/b8/validation/b8_46918/validate_b8.py matrix \
  --checkout /data/yihou/b8/sglang-candidate \
  --out /data/yihou/b8/run-first \
  --rows 256 --seeds 104729
```

The initial three-arm check isolates the reported failure. A short follow-up
checks the cap transition, decode boundary, and a second seed at the failing
shape:

```bash
python /data/yihou/b8/validation/b8_46918/validate_b8.py matrix \
  --checkout /data/yihou/b8/sglang-candidate \
  --out /data/yihou/b8/run-boundaries \
  --rows 128,129,257,256 --seeds 130363
```

Optional additional shapes are `2,64,200`. The default matrix is
`256,128,64,257` with seed `104729`. Compilation timeout is 300 seconds per
subprocess, configurable with `--timeout`. Each successful case uses 20 warmups
and seven GPU-event measurements of 100 production calls. Environment startup
and cold compilation dominate the short validation; no full-model loading is
required.

## Controlled comparison and evidence

All arms use the same candidate and production dispatcher:

| Arm | `SGLANG_ROCM_ABSORB_BMM_MAX_WAVES_PER_EU` | `SGLANG_ROCM_ABSORB_BMM_DISABLE_HAND_CONFIG` | Expected M=256 selection |
| --- | --- | --- | --- |
| `default` | `2` | `true` | `config=None`, historical workaround |
| `capped` | `2` | `false` | Hand configuration, `waves_per_eu=2` |
| `uncapped` | `0` | `false` | Hand configuration, `waves_per_eu=4`, historical failure |

Each arm/shape/seed has a separate process, Triton cache, runtime cache
directories, stdout/stderr, and core dumps disabled. AITER sources are shared
while AITER JIT output is isolated. An uncapped crash does not stop the other
arms. Timeout handling terminates only the process group launched for that case.

The exact issue inputs are BF16 Q storage `(256,64,256)` sliced to `[:,:,:192]`,
FP8 E4M3FN weights `(64,192,512)`, and FP32 scales of one. Production dispatch
passes transposed weights to AITER. Nearby shapes vary only the first Q dimension.
Inputs are generated on CPU with identical seeds; content hashes must match
before pairwise output comparisons.

`result.json` is written before the first production call, preserving config and
source/input/software fingerprints if compilation aborts. Successful cases save
BF16 CPU outputs and timing samples. `summary.json` adds paired output diagnostics
and timing ratios. A zero exit code means the required default and capped arms
completed; it is not a blanket numerical or end-to-end correctness approval.

## Interpretation and limits

1. Reproducing the reported bug requires the uncapped M=256 arm to fail with the
   original LLVM `Sequence.h` / `iota_range` assertion. An import error or unrelated
   exception does not reproduce that bug. If uncapped succeeds, retain version
   details and record that the issue was not reproduced in that environment.
2. For a tested shape/seed, capped and default should compile and return finite
   BF16 outputs of the expected shape. Exact equality is reported when present.
   Nonzero differences require investigation; the harness does not invent a
   tolerance to declare correctness.
3. The FP32 diagnostic uses BF16 activations and dequantized stored FP8 weights.
   AITER internally quantizes activations too, so this is not a quantization-matched
   oracle. Compare capped/default outputs and their reference errors together.
   Material differences warrant an AITER quantization-reference follow-up.
4. M=128 exercises a smaller hand configuration; M=129 probes the three-to-two
   wave cap. M=257 should select `config=None` in every arm. The four-wave
   selection at M=256 is asserted, so a different CU layout is not silently
   treated as the original failure case.
5. Steady GPU-event timings cover activation quantization, dispatch, and output
   allocation after warmup. They do not establish deployment throughput, model
   quality, or a token-rate improvement. Stack-level 48k throughput used simulated
   MTP acceptance and must not be cited as evidence for this isolated patch.

The public script omits private host-specific exclusions. The launcher remains
responsible for allocating authorized hardware and avoiding live experiments.
