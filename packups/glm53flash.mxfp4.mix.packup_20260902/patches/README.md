# Patches

## `patch_glm5next_shared_experts_fusion_quant_guard.py`

**WHAT.** Adds the `quant_blocks_shared_experts_fusion(quant_config)` guard to
`Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason`
(`sglang/srt/models/glm5_next.py`), as the gate's **first** statement. Two edits:
one to the existing `deepseek_common.utils` import block, one to the gate body.
It mirrors `deepseek_v2.py:3069` verbatim; `deepseek_v4.py:3289` uses the same
helper.

**WHY.** Without the guard, GLM-5.3-Flash-MXFP4 cannot load at all on gfx950 —
the gate answers "fuse", a BF16 shared expert is renamed into routed slot 288 of
an MXFP4-packed `FusedMoE`, and `_load_w2` dies with `256 must match 512`. Full
evidence in `../results/root_cause.md`; the patch's own docstring is the
authoritative write-up and includes the upstream survey.

**HOW.** Run inside the image, after the sglang overlay is in place:

    python3 patches/patch_glm5next_shared_experts_fusion_quant_guard.py

It locates sglang through `importlib`, is **idempotent** (re-running is a no-op),
and is **all-edits-or-none**: an anchor that is missing or no longer unique
writes nothing and exits 1, because a half-applied fix crashes in the same place
the unpatched tree does. It also refuses to write if
`quant_blocks_shared_experts_fusion` is absent from
`models/deepseek_common/utils.py`, since the added import would otherwise be an
`ImportError` at model load rather than a clean build failure.

**CONTEXT — and an important scoping note.**

**This patch is NOT required to reproduce the result in this packup.** Every
number here was produced with the runtime flag `--disable-shared-experts-fusion`
instead, which `layers/moe/utils.py:459` short-circuits on **before** ever
calling the gate. The flag and the patch are interchangeable in effect; the
patch additionally makes the engine correct for anyone who forgets the flag.

It is **anchor-verified but not runtime-verified**: it applied cleanly to a copy
of the c821c425 tree and the result byte-compiled (the verify run left a
`__pycache__/glm5_next.cpython-313.pyc`, omitted from this copy), but **no server
was ever brought up on the patched engine**. Verification fixtures are in
`_verify/`
(`glm5_next.py.orig` = the pristine c821c425 file, and the applied result under
`_verify/sglang/srt/models/`). Drift and no-helper fixtures existed in the
workspace and are omitted here as scratch.

**Upstream status: unfixed, and there is nothing to take.** `refs/pull/36607/head`
is exactly `c821c425` with zero commits beyond it; the only downstream movement
is `c767511e` (2026-09-01), which reverts #36607 wholesale rather than guarding
it. See `../results/root_cause.md`.

## What is NOT here

The infera-repo build-time patches and Dockerfile changes this run needed are in
`../repo-changes/`, not in this directory — they are repo deliverables, not
one-off engine fixes.
