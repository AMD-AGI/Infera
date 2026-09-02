# `patches/sglang_carried/` — patches we hold, and deliberately do not build

**No Dockerfile copies or runs anything in this directory.** That is the whole
point of it, and it is the reason these scripts do not live beside their
siblings in `patches/sglang_rocm/`.

Every other patch directory is consumed the same way:

```dockerfile
COPY deploy/docker/patches/sglang_rocm/ /tmp/sglang-rocm-patches/
RUN set -eu; \
    for f in /tmp/sglang-rocm-patches/*.py; do echo "[sglang-patch] $f"; python "$f"; done; \
```

The glob is `*.py` and the loop is unconditional, so **dropping a file into one
of those directories wires it into every image that copies the directory** —
`Dockerfile.sglang` and `Dockerfile.sglang.gfx942` for `sglang_rocm/`. There is
no per-file opt-out. A patch that is ready to read but not ready to build
therefore cannot be stored there without changing two images.

This directory is the opt-out. Nothing globs it, so a script here is carried,
reviewable, and inert.

## When a patch belongs here

When all three hold:

- the defect is established well enough to write the fix down,
- the fix has **not** been executed on hardware, or its effect has not been
  measured, and
- wiring it would perturb something in flight — a benchmark campaign, an
  alignment comparison, a packup whose numbers are already reported.

The last one is the common case, and it is why "carried" is a status and not a
euphemism for "unfinished". A silently changed image invalidates every number
measured before and after it; an unwired script costs nothing and loses nothing.

## Moving one out

Move the file into the directory of the image that should carry it
(`sglang_rocm/`, `sglang_dsa/`, …) and update its row in
`../../patch.upstream.status.md` from *carried, not applied* to the image list
it is now baked by. The glob picks it up from there — no Dockerfile edit is
needed, which is exactly why the move must be deliberate.

Before moving one out, it needs what it did not have when it was parked: a run
on hardware, and a measurement of the thing it claims to change.

## Contents

| script | what it fixes | why it is not wired |
|---|---|---|
| `patch_glm_moe_gate_bias_fp32.py` | GLM MoE `e_score_correction_bias` is allocated bf16 under aiter+quant and downcast again at the aiter router boundary; GLM's bias band collapses from 238 distinct fp32 values to 8 in bf16 (Flash: 282 → 11), reordering `noaux_tc` top-k routing | verified as text and as predicate logic only — **never executed on a GPU**, and no accuracy or throughput delta measured. Wiring it would rebuild the engine image mid-campaign and confound an alignment comparison whose validity rests on both arms running the same code |
