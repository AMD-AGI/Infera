# DeepSeek-V4 on MI325X (gfx942)

```{admonition} One-pager
:class: tip
**What:** run the DeepSeek-V4 family (Pro / Flash, FP4 / FP8) on MI325X
(gfx942 / CDNA3). **Why:** the runnable configurations differ by engine, so
Infera enforces one fixed support contract and auto-applies the knobs each
combination needs (set-if-unset). **The one nuance:** FP4 dsv4 runs on **vLLM**
only; FP8 dsv4 runs on **SGLang / ATOM** only; **Flash-FP8 needs MTP**, which
Infera turns on for you.
```

Infera detects a local DeepSeek-V4 checkpoint at startup (variant from model
dimensions, quant from `quantization_config`), enforces the support matrix, and
sets the functional env + CLI a supported combination needs. Unsupported
combinations **fail fast** with an actionable message instead of running degraded
or silently wrong. Infera does **not** patch third-party engines.

## Support matrix

On gfx942 (MI325X), per variant × quant × engine:

| Variant | Quant | vLLM | SGLang | ATOM |
|---|---|:---:|:---:|:---:|
| **Pro** | FP4 | ✅ native | ❌ | ❌ |
| **Pro** | FP8 | ❌ | ✅ | ✅ |
| **Flash** | FP4 | ✅ native | ❌ | ❌ |
| **Flash** | FP8 | ❌ | ✅ (MTP) | ✅ (MTP) |

✅ supported · ❌ fails fast (use the engine that supports the combo) · **(MTP)**
= speculative decoding applied automatically (see below).

## Why these rules

- **FP4 → vLLM only.** gfx942 has no native FP4 MoE kernel. vLLM's
  `triton_unfused` MoE backend upcasts FP4 → bf16 *in-kernel*, so it runs
  unpatched. SGLang and ATOM have no native FP4 path, and Infera does not patch
  third-party engines — so FP4 dsv4 on SGLang/ATOM fails fast. Use vLLM for FP4,
  or hand SGLang/ATOM an FP8 checkpoint instead.
- **FP8 → SGLang / ATOM.** FP8 dsv4 runs natively on SGLang and ATOM. It is not
  validated on vLLM, so FP8 dsv4 on vLLM fails fast. Use SGLang or ATOM for FP8.
- **Flash-FP8 needs MTP.** The gfx942 dsv4-**Flash** compressed-MQA *decode*
  kernel is defective: prefill (and the first token) is correct, but subsequent
  decode diverges. Routing decode through a speculative (EAGLE / MTP) path avoids
  the broken kernel. Infera enables MTP automatically for Flash-FP8; Pro-FP8 is
  correct without it and does not get it.

## What Infera sets automatically

All of the below are applied **set-if-unset**: if you already set the env var or
pass the CLI flag, **your value always wins** — Infera never overrides it. The
knobs are functional (correctness / bring-up), not tuning.

**SGLang (FP8)** — env:

| Env | Value | Why |
|---|---|---|
| `HSA_NO_SCRATCH_RECLAIM` | `1` | gfx942 firmware requirement — distributed init aborts without it. |
| `SGLANG_USE_ROCM700A` | `0` | Select the gfx942-correct ROCm path. |
| `SGLANG_HACK_FLASHMLA_BACKEND` | `unified_kv_triton` | The default tilelang MLA backend fails to compile on gfx942. |
| `AITER_BF16_FP8_MOE_BOUND` | `0` | AITER MoE numeric bound for the FP8 path. |

SGLang (FP8) — CLI: `--attention-backend dsv4 --disable-shared-experts-fusion`.
For **Flash** additionally:
`--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.

**ATOM (FP8)** — env: `HSA_NO_SCRATCH_RECLAIM=1` (same gfx942 firmware
requirement). For **Flash** additionally, CLI: `--method mtp
--num-speculative-tokens 3`.

**vLLM (FP4)** — nothing to inject; the `triton_unfused` MoE path runs natively.

```{admonition} What Infera does NOT set
:class: note
Infera does not touch memory or throughput knobs — `--cpu-offload-gb`,
`--max-total-tokens`, `--max-running-requests`, `--mem-fraction-static`. Size
those for your own hardware and load.
```

## Image

gfx942 needs the dedicated SGLang image `Dockerfile.sglang.gfx942` (MI30x base).
The default `Dockerfile.sglang` targets MI355X / gfx950 and will not run on
gfx942. **ATOM** and **vLLM** use their standard images.

## Related

- The launcher logic lives in `infera.engine.dsv4_gfx942`, called from each
  engine's `python -m infera.engine.<name>` startup.
- [Feature matrix](feature_matrix.md) — where dsv4-on-MI325X sits against the
  other engine features.
