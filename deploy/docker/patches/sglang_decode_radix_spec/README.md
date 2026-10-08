# Experimental Decode radix cache with EAGLE/NEXTN

Stock SGLang v0.5.18 (the default mi35x image base) rejects Decode radix cache
whenever speculative decoding is active. Infera forwarding the radix flag does
not bypass that engine validation.

`patch_decode_radix_spec.py` supplies the engine half of the runtime opt-in.
`Dockerfile.sglang` applies it by default; the feature itself remains off unless
`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` is set in the worker environment. Only
EAGLE/NEXTN with top-k 1 bypass this one speculative-decoding rejection. Existing
HiSparse, DCP, backend and cache-builder checks are unchanged. Infera's model
compatibility checks still run before automatic flag forwarding.

The script recognizes the v0.5.18 mutable-args and v0.5.19 resolved-view hook
layouts, fails on unknown/partially patched guards, and is idempotent. It removes
old module bytecode and recompiles with checked source hashes. The source tests
execute the releases' actual admission branches; they do not instantiate a GPU
KV cache or validate inference accuracy.

Use `--build-arg APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=0` for a custom compatible
base where this patch is inappropriate. `Dockerfile.sglang.gfx942` does not apply
this patch. Drop/rework the patch when the pinned base natively supports the
required combination.

The experimental runtime evidence is GLM-5.2 on the previously patched v0.5.19
build, not a fresh GPU validation of the v0.5.18 default image. Hybrid SWA/SSM and
EAGLE3 are outside this feature's supported scope. In particular this does not
backport [SGLang #40857](https://github.com/sgl-project/sglang/pull/40857), whose
ownership fix addresses hybrid-SWA multi-turn corruption. That PR was still
open when checked on 2026-10-08; relaxing admission is not proof of memory safety
for those other configurations.
