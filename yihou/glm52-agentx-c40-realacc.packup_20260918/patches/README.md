# patches

Three things the run needs that are not in the tracked tree or the base image.
Only the first is required for the acceptance result.

## `0001-engine-sh-extra-args-env.patch` — **required**

**What.** Uncommitted change to `bench/glm5p2_pd/engine.sh`. Adds role-scoped
`PREFILL_EXTRA_ARGS` / `DECODE_EXTRA_ARGS` (extra engine flags) and
`PREFILL_EXTRA_ENV` / `DECODE_EXTRA_ENV` (extra container env, validated against
`^[A-Za-z_][A-Za-z0-9_]*=`), and makes `--enable-aiter-allreduce-fusion`
conditional on `AITER_ALLREDUCE_FUSION` instead of unconditional.

**Why.** `--disable-custom-all-reduce` has to reach the decode engine and only
the decode engine. Without this hatch the script itself would have to be edited
per experiment, which makes "one variable at a time" unenforceable. The AITER
change exists because sglang's CLI exposes only the positive
`--enable-aiter-allreduce-fusion` (a `store_true` with no `--no-` form), so there
is otherwise no way to A/B it; the only other code path that disables it,
`--enable-deterministic-inference`, changes far more besides.

**How.** `git apply patches/0001-engine-sh-extra-args-env.patch` at the repo root.

**Context.** All three additions default to the previous behaviour, so the patch
is a **no-op** unless a config opts in. It is unapplied upstream and uncommitted
here.

## `Dockerfile.yihou.hicache` + `pr37152.sources.yihou.diff` — carried, inert here

**What.** A thin image layer applying the three **source** hunks of upstream
sglang PR **#37152**, "[ROCm] Widen the HiCache JIT copy rounds and enable the
K-only host pool":

- `kernels/jit/csrc/kvcacheio/hicache.cuh` — `pick_group_bytes()` picks the
  widest of 128/64/32/16 B that both divides the element and splits across lanes
  into a 4/8/16 B hardware package. Previously the JIT transfer kernels tiled in
  128 B rounds only, so any element size not divisible by 128 — notably MLA's
  576 B fp8 KV row — could not use the JIT path at all.
- `kernels/ops/kvcache/hicache.py` — `_tiles_across_lanes()`, the Python mirror
  that screens element sizes before JIT.
- `srt/mem_cache/pool_host/mha.py` — `MHATokenToKOnlyPoolHost.can_use_jit`
  widened from `_is_cuda` to `(_is_cuda or _is_hip)`.

The PR's two test files are not applied.

**Why a thin layer and not a rebuild.** All three targets are plain source files
under `python/sglang/`, and the `.cuh` is JIT-compiled at runtime against a
content-addressed cache (sha256 over source bytes + generated ninja file +
compiler-env fingerprint), with no seeded `~/.cache/sglang` or `*.hsaco` baked
into the image. So editing the source in place is enough to make the next run
rebuild the kernel. A full rebuild would take 40–60 min and risk drifting the
7 DSA patches the base carries.

**How.** `docker build -f Dockerfile.yihou.hicache .` on each node with both
files in the build context. The Dockerfile applies at `--fuzz=0` behind a
`--dry-run` guard, asserts three markers, drops `__pycache__`, and fails the
build on any deviation. **Verification is deliberately marker-grep at build time
and import-based at runtime** — `docker build` has no GPU, so importing anything
under `sglang.srt` there dies with `No HIP GPUs are available`
(`fp8_kernel.py:50` → `torch.cuda.get_device_properties(0)`), and the `.cuh` has
no import path at all.

**Context you need before reusing this.**

- The PR is **open upstream and never merged** as of 2026-09-18, so it is in no
  nightly image. **Upstream's own AMD ROCm CI on it is red**, cause unknown. It
  was applied because the user asked for it.
- **It does nothing in this run.** `config.sh` sets `PREFILL_HICACHE=0`, so no
  HiCache copy kernel executes. Its effect belongs to the T2 sweep, which runs
  `config.full.sh` with `PREFILL_HICACHE=1`, ratio 1.5.
- Source: `/home/yihou/dev/git/infera.glm52.view/agentx.c72.oai176219_20260918.packup_20260918/patches/`.

## `Dockerfile.yihou.nextnfix` + `apply_nextn_fusion_fix.sh` — in the base, optional

**What.** Gives `GlmMoeDsaForCausalLMNextN` the
`fused_shared_experts_architecture` override it is missing, so the MTP **draft**
does not silently lose shared-experts fusion. Upstream fork commit `4350d37c5b`.

**Why it is here.** The base image used by this run already contains it, so
reproducing this exact run needs it. Include it if you are building up from plain
`infera-sglang:v0519-yihou-0917`.

**Why it is optional.** It is a real, independent defect — confirmed on this
stack by code read and by the log signature — but it is **not** what fixed the
garbled decode. R06 in `glm52-mtp-garbled-decode-rootcause.packup_20260918`
removed it and still produced correct output with `--disable-custom-all-reduce`
alone. Pass `--build-arg BASE=infera-sglang:v0519-yihou-0917` to skip it.

**How to confirm it is live**, from the decode log rather than the source:
`Shared experts fusion optimization enabled` appears **2×** and `Config does not
support fused shared expert(s)` **0×**. Under an unpatched image the draft logs
the disabled line.
