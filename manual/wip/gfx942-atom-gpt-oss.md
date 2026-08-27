# ATOM + gpt-oss-120b on gfx942

**ATOM means to run this model on MI300X, and no published image does.** The
engine source has first-class gfx942 handling for MXFP4 MoE — an arch dispatch
that deliberately selects the gfx94x path, an `e4m3fnuz` branch, a CDNA4 scale
swizzle applied to gfx942 as well as gfx950 — and upstream's
[recipes/GPT-OSS.md](https://github.com/ROCm/ATOM/blob/main/recipes/GPT-OSS.md)
advertises "fits on a single MI300X/MI355X GPU". But in all three ATOM tags
published between June and July 2026, that path is wired to a mismatched kernel
dependency and the server dies before serving a token. The `gfx942` overlay on
the gpt-oss rows in `tests/e2e/pd_{mixed,disag}/atom/matrix.py` therefore skips.

This is an upstream defect, not a missing feature, and not something the infera
image layer causes: the base images fail the same way before `Dockerfile.atom`
adds anything.

## Why gfx942 lands on a path gfx950 never touches

`FusedMoEMethod.__init__` in `atom/model_ops/moe.py` picks the kernel family
from the arch, not from a tuning table:

```python
self.use_triton = gfx.startswith("gfx94") or (
    gfx.startswith("gfx95") and envs.ATOM_USE_TRITON_GEMM
)
```

gfx950 gets the aiter asm/FlyDSL kernels; gfx94x is routed to
`atom/model_ops/fused_moe_triton.py`. In every image measured, that module's
imports are satisfied on gfx950 only because gfx950 never executes them — the
import block is guarded by `if envs.ATOM_USE_TRITON_GEMM or
envs.ATOM_USE_TRITON_MOE:`, which is false by default on gfx950 and required on
gfx942. Upstream CI can be green on MI355X while this path is broken.

## What each published tag does

Measured on an MI300X node (8×gfx942, 304 CU), `gpt-oss-120b` MXFP4, tp2 + EP.

| Tag (rocm/atom) | Dies at | Cause |
|---|---|---|
| `…atom0.1.4_20260612` | weight load | ATOM calls the old `triton_kernels` API; the image ships the AMD 1.0.0 rewrite |
| `…atom0.1.4_202607091539` | weight load | ATOM imports `swizzle_scales`, absent from the image's aiter |
| `…atomesh_202607121715` | first forward | aiter's MoE scale shuffle has no gfx942 branch; forcing one faults the kernel |

### June — ATOM against a rewritten `triton_kernels`

`fused_moe_triton.py` wants `triton_kernels.routing` and `matmul_ogs`. The
installed `triton_kernels 1.0.0+amd.rocm7.2.0` has neither: routing became
`topk.py` + `compaction.py`, `matmul_ogs` became `matmul`, and `RoutingData`,
`GatherIndx`, `ScatterIndx` and `compute_expt_data` are gone entirely. ATOM
anticipated the `matmul` rename (there is a `try`/`except ImportError` for it)
but not the rest, and the whole block sits inside
`except (AttributeError, ImportError): logger.error(...)` — so the failure is
logged, module-level names stay unbound, and the crash surfaces much later.

Not shimmable. The new `matmul` takes `a_ragged_metadata` and a tensor
`gather_indx`; ATOM passes a `routing_data` positional and a `GatherIndx`
dataclass. That is an API generation apart, which is presumably why upstream's
next build dropped the `triton_kernels` dependency instead of adapting to it.

The non-triton escape (`ATOM_USE_TRITON_MOE=0`) does not help either. gpt-oss
MoE carries biases, so aiter requires a bias-capable stage1, which means FlyDSL
or CK-Tile — the 1-stage asm table has no `Swiglu` entry for gfx942 *or* gfx950:

- **FlyDSL** is chosen when activations quantise to fp4x2, i.e. at large M. It
  emits a 128-bit `llvm.amdgcn.raw.ptr.buffer.load.lds`, a CDNA4 instruction, and
  the compile ends in `LLVM ERROR: Do not know how to expand this operator's
  operand!`.
- **CK-Tile** is the small-M path, and the one whose comment names GPT-OSS. It
  can be forced at every M by raising `GPTOSS_SWIGLU_MXFP4_BF16_BOUND` so
  activations stay bf16, or entered via `ATOM_MOE_GU_ITLV=1`. Both reach it, and
  both die identically: `Cannot find Symbol with name: …MoeFlatmmKernel…`. The
  host symbol is present in `module_moe_cktile2stages.so` — it is the *device*
  code that is missing. The gfx942 slice of that fat binary is 2,432 bytes
  against 82,128 for gfx950. Tried with tiles `[64,256,256]` and `[16,128,256]`.

So on gfx950 the model runs on FlyDSL, and the CK-Tile fallback that only gfx942
can reach was never built for it.

### July — ATOM against an older aiter

The July build is the fix for the above: `fused_moe_triton.py` was rewritten to
drop `triton_kernels` and call aiter directly, with explicit gfx942 handling
(`swizzle_scales_cdna4` for `gfx942` and `gfx950`, `float8_e4m3fnuz` quant
dtype). It imports `swizzle_scales` from
`aiter.ops.triton.moe.moe_op_gemm_a8w4`, which the bundled aiter does not define
anywhere. Same class of skew, opposite direction.

One knob is needed before that error is even reachable: `_swizzle_mxfp4` asserts
`ATOM_USE_TRITON_GEMM or ATOM_USE_TRITON_MOE`, but the arch dispatch above sets
`use_triton` without setting either, so ATOM's own default aborts weight loading
on gfx942 with a bare `AssertionError`. `ATOM_USE_TRITON_MOE=1` gets past it.

### atomesh — consistent at last, and still faulting

`…atomesh_202607121715` pairs matching halves: ATOM no longer imports
`swizzle_scales`, and `aiter.ops.triton.moe.moe_routing.routing` exists. Weights
now load far enough to reach aiter's shared scale shuffle, which ends here:

```python
if arch == "gfx1250":
    tiled = _shuffle_scale_tile_gfx1250(...)
elif (arch or get_arch()) == "gfx950":
    tiled = _shuffle_scale_tile_gfx950(...)
scale = tiled.transpose(-1, -2)   # gfx942 falls through; `tiled` is unbound
```

`UnboundLocalError`. Its docstring lists gfx950 and gfx1250 only, so the gfx942
case was dropped when this dispatch moved out of ATOM into aiter — ATOM's own
July table still says gfx942 should get the CDNA4 layout.

Two patches were tried and both abandoned:

1. **Route gfx942 to the gfx950 tiler** (what ATOM's table asks for). Weights
   swizzle, torch.compile succeeds, and the profiling forward dies with
   `Memory access fault by GPU node-2` — the kernel does not address scales in
   the CDNA4 layout on this arch.
2. **Return the scales untiled with no layout label** — the combination ATOM
   already emits for shapes that are not swizzle-aligned, so the kernel has a
   path for it. Identical memory access fault.

Neither layout works, which puts the remaining fault inside the gfx942
instantiation of the aiter Triton MoE kernel itself. Guessing further would mean
patching kernel internals with no gfx950 coverage to protect the change, so the
row stays skipped.

## What would unblock it

An ATOM image whose aiter carries a working gfx942 `a16w4`/`a4w4` MoE kernel and
a gfx942 branch in `shuffle_scale_moe`. Two of the three failures above were
already fixed in the following build, so this is moving; re-test on the next tag
rather than assuming the skip is permanent. When it lands, the overlay likely
needs only `ATOM_USE_TRITON_MOE=1`.

## Reproducing

`Dockerfile.atom` takes the base tag as a build arg, and `run_tests.sh` forwards
build args on both tiers, so re-testing the next ATOM release needs no new
tooling:

```bash
INFERA_E2E_GFX_ARCH=gfx942 \
INFERA_E2E_BUILD_ARGS="ATOM_BASE_IMAGE=rocm/atom:<tag>" \
  tests/run_tests.sh e2e atom mixed
```

Two things to do first, or the run says nothing. Drop the `gfx942` skip from
`tests/e2e/pd_mixed/atom/matrix.py`, otherwise the case never executes; and put
`ATOM_USE_TRITON_MOE=1` in that overlay, otherwise weight loading stops at the
`_swizzle_mxfp4` assertion described above rather than at whatever the release
actually fixed.

Candidate knobs are worth probing outside the matrix before they are frozen into
an overlay — `ATOM_USE_TRITON_MOE=0` for the aiter route,
`GPTOSS_SWIGLU_MXFP4_BF16_BOUND` to hold activations at bf16 so the CK-Tile
branch is reached at every batch size, `ATOM_MOE_GU_ITLV=1` for the interleaved
entry into the same branch.
