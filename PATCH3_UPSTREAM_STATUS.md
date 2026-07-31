# Patch 3 — already covered upstream, do NOT open a PR

**Patch:** `deepseek_nextn_glm52_mtp_bf16.diff`
**File:** `python/sglang/srt/models/deepseek_nextn.py`
**Status:** superseded by merged upstream PR — keep locally, do not upstream.

## What patch 3 does

One line, on the `v0.5.15.post1` release branch we build against:

```python
- ckpt_prefix = f"model.layers.{config.num_hidden_layers}"
+ ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"
```

GLM-5.2's MTP/NextN `eh_proj` is bf16 while the rest is quark-mxfp4. Quark's
`exclude_layers` records the excluded MTP weight at submodule granularity, so
matching on the bare layer prefix `model.layers.{N}` misses it and `eh_proj`
gets built as mxfp4 → weight-load shape crash. Appending `.eh_proj` makes the
`should_ignore_layer` check match and keeps it bf16.

## Why no upstream PR

Upstream **`main` already fixes exactly this bug, more completely**, via merged
PR **#30265 "[AMD] Fix GLM-5.2 MTP Quark excludes"**
(https://github.com/sgl-project/sglang/pull/30265 — merged 2026-07-08, merge
commit `07ef650`):

- adds a dedicated `GlmMoeDsaForCausalLMNextN` class;
- extracts `_resolve_nextn_quant_config()`;
- remaps the **full** MTP exclude set for GLM — `eh_proj`, `enorm`, `hnorm`,
  `shared_head.norm`, decoder blocks (`model.decoder.*`), and the fused-MoE
  experts prefix `model.decoder.mlp.experts`.

The quark matcher (`_is_equal_or_regex_match`) is exact-string unless the target
is `re:`-prefixed, which is why #30265 had to add every runtime name. Our
one-liner only remaps `eh_proj`, so it is a **strict subset** of #30265 — a
narrow backport onto the release branch, not an improvement over `main`. The PR
body says so directly: "Mapping only the leaf names is not enough."

## Version relationship (verified)

- Our base tag `v0.5.15.post1` → commit `0b3bb0c`. At that commit
  `deepseek_nextn.py` still has the old inline block patch 3 edits, and
  `glm4_moe.py` has **no** `GlmMoeDsaForCausalLMNextN` → the release branch does
  **not** contain #30265.
- `compare(v0.5.15.post1 … 07ef650)` = diverged, ahead 65 / behind 25 — the
  release line was cut without #30265; `main` has it.

## Action

Keep patch 3 in the local set (our base genuinely needs it). Do not open a PR.
If/when we rebase onto an sglang `main` that already contains #30265, drop
patch 3 entirely — it will no longer apply and is no longer needed.
