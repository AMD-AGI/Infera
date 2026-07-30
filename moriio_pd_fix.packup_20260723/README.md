# vLLM v0.25.1 MoRIIO PD page-len fix — DeepSeek-V4 & GLM-5.1

**Ran:** 2026-07-22 to 2026-07-23
**Author:** c_huggingface
**Status:** ✅ **PASS** — both models produce correct, coherent cross-node PD output.

## Goal

infera's vLLM-MoRIIO PD-disaggregation produced GARBAGE output on the new v0.25.1
image (DeepSeek-V4-Pro and GLM-5.1-FP8), while Mooncake PD was correct on the same
nodes. The task: find the root cause and make **cross-node GLM PD via MoRIIO** emit
correct output. This kit captures the root-cause hunt, the one-line fix, and the
before/after proof for both models.

**Success criterion:** cross-node MoRIIO PD produces correct + coherent output at
temp=0 for GLM-5.1-FP8 (the target model), on the v0.25.1 image, via the
infera-native launch (etcd + infera.server router + infera.engine.vllm workers).

## Result

Root cause = a genuine **vLLM v0.25.1 MoRIIO bug** in `moriio_layout.py`, NOT the
infera router protocol. `get_layer_transfer_geometry` derived the MLA per-block
transfer size/stride from tensor `.shape` instead of the authoritative
`spec.page_size_bytes`. One bug, two symptoms; one fix.

| Model | Config | Before fix | After fix | Verdict |
|-------|--------|-----------|-----------|---------|
| DeepSeek-V4-Pro | TP4 2-node MoRIIO WRITE PD | France→"a good idea" (facts garble, structure survives) | France→**Paris**, China→**Beijing** | ✅ |
| GLM-5.1-FP8 | TP4 2-node MoRIIO WRITE PD | France→"is is is" (total garbage) | France→**Paris**, China→**Beijing**, 2+2→**4**, sky→**Rayleigh** | ✅ |

- DSv4 = `fp8_ds_mla` padded page (ratio 1) → dropped tail = UE8M0 scale.
- GLM = per-kernel-block=1 (ratio 16) → transferred only 1/16 of each block.
- Fix (`patch_moriio_pagelen.py`): MLA `block_len = spec.page_size_bytes`,
  `block_stride = page_size_bytes // element_size`. Mirrors Mooncake. No-op for
  contiguous fp16/bf16 K/V.

## How to reproduce

See `REPRODUCE.md`. TL;DR: bring up 1P1D on two nodes (infera-native), run the
temp=0 probe → garbage; apply `patches/patch_moriio_pagelen.py`; rerun → correct.

## Folder map

- `REPRODUCE.md` — step-by-step reproduction (bug → fix → correct)
- `environment.md` — exact HW/SW/image digest/git SHA/paths/secrets
- `scripts/` — the infera-native launch + probe scripts (verbatim) + the patch
- `patches/` — `patch_moriio_pagelen.py` + its what/why/how/context note
- `results/evidence.md` — the before/after geometry tables + probe outputs
- `notes.md` — the full debug narrative: 5 hypotheses, wrong turns, ruled-out list
- `logs/` — gzipped engine + router logs (DSv4 + GLM, prefill/decode/router)
- `working_process_raw.md` — the raw iteration-by-iteration log (unedited source)
