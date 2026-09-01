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

Infera detects a local DeepSeek-V4 checkpoint at startup, enforces the support
matrix, and sets the functional env + CLI a supported combination needs.
Unsupported combinations **fail fast** with an actionable message instead of
running degraded or silently wrong. Infera does **not** patch third-party engines.

Variant comes from the model's dimensions (Pro: 7168 hidden / 61 layers; Flash:
4096 / 43). Quant is **the routed experts' dtype**, which is not what
`quantization_config` reports: every dsv4 checkpoint declares `quant_method:
fp8`, because that describes the attention half. The experts are declared
separately in a top-level `expert_dtype`, and where that key is absent Infera
reads the dtype off the safetensors headers rather than trusting the config.

## Support matrix

On gfx942 (MI300X / MI325X), per variant × quant × engine:

| Variant | Quant | vLLM | SGLang | ATOM |
|---|---|:---:|:---:|:---:|
| **Pro** | FP4 | ✅ native | ❌ | ❌ |
| **Pro** | FP8 | ❌ | ✅ | ✅ |
| **Flash** | FP4 | ✅ native | ❌ | ❌ |
| **Flash** | FP8 | ❌ | ✅ (MTP) | ✅ (MTP) |

✅ supported · ❌ fails fast (use the engine that supports the combo) · **(MTP)**
= speculative decoding applied automatically (see below).

## Which checkpoint to hand which engine

The matrix above is about the experts' dtype, so in practice it is a choice
between three published checkpoints:

| Checkpoint | Size | Experts | Serve on |
|---|---:|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` | 806 GiB | MXFP4 | vLLM |
| `deepseek-ai/DeepSeek-V4-Flash` | 149 GiB | MXFP4 | vLLM |
| `sgl-project/DeepSeek-V4-Flash-FP8` | 274 GiB | block-FP8 | SGLang, ATOM |

**On a 192 GiB card, Flash-FP8 is the only dsv4 checkpoint SGLang and ATOM can
serve**, and the reason is capacity rather than support. Both engines could in
principle unpack MXFP4 experts to FP8 at load — each needs a source patch to do
it — but for Pro the result does not fit: 186.5 GiB of unpacked weights a card at
tp8, plus ~9.25 GiB the runtime holds outside PyTorch, against 191.98 usable. The
upstream `sgl-project/DeepSeek-V4-Pro-FP8` is those same weights already unpacked
(~1.6 TB, 200 GiB a card at tp8) and lands in the same place. Flash-FP8 needs
34.2 GiB a card at tp8 and 68.5 at tp4, so no patch and no arithmetic is
involved.

A 256 GiB MI325X changes the Pro answer and nothing else here.

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

```{admonition} The MTP rule is inherited from MI325X and unverified on MI300X
:class: warning
Both halves of the bullet above — that the Flash decode kernel is broken, and
that the depths below are the right way around it — arrive from the MI325X
bring-up, and nothing on the MI300X fleet has measured either. They are carried
because they are the only recipe that exists and they are at least self-consistent
(SGLang requires `num_draft_tokens == num_steps + 1` for dsv4; the checkpoint
ships one draft layer, `num_nextn_predict_layers: 1`, which the extra steps
re-run). A Flash row that comes back **correct with the speculative flags
removed** would falsify the premise, and is worth reporting rather than
discarding.
```

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
