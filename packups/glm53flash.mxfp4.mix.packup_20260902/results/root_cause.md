# Root cause — GLM-5.3-Flash-MXFP4 cannot load without `--disable-shared-experts-fusion`

Referenced from `ladder.md`. The engine-side fix and its full anchor rationale
are in `../patches/patch_glm5next_shared_experts_fusion_quant_guard.py` (read its
docstring — it is the authoritative write-up); this file is the evidence trail.

## Symptom

Bare `sglang.launch_server`, vendor image, vendor flags verbatim, unmodified
model dir, TP4. Dies at weight load, shard 4 of 120 (`tqdm` reports
"3/120 Completed"):

```
RuntimeError: The size of tensor a (256) must match the size of tensor b (512)
              at non-singleton dimension 1
```

Call path, from `logs/n0433/rung0.log.gz`:

```
models/glm5_next.py:1811                       load_weights
layers/moe/fused_moe_triton/layer.py:983       weight_loader
layers/moe/fused_moe_triton/layer.py:1013      _weight_loader_physical
layers/moe/fused_moe_triton/layer.py:1293      _weight_loader_impl
layers/moe/fused_moe_triton/layer.py:582       _load_model_weight_or_group_weight_scale
layers/moe/fused_moe_triton/layer.py:789       _load_w2   -> expert_data.copy_(loaded_weight)
```

## The discriminating log line

`Shared experts fusion optimization enabled.`

- present in `logs/n0433/rung0.log.gz` (the failing run) — count 1
- absent in every passing run, including the live MIX worker on n01-33 — count 0

## Tensor-level evidence

Instrumented `_load_w2` (`logs/n0433/rung0_wtrace.log.gz`). Neighbouring routed
experts, then the one that fails:

```
[WTRACE] layer_id=10 wname='...experts.w2_weight' shard=w2 eid=8   param=(289, 4096, 256) loaded=(4096, 1024) qm=QuarkFusedMoEMethod
[WTRACE] layer_id=10 wname='...experts.w2_weight' shard=w2 eid=89  param=(289, 4096, 256) loaded=(4096, 1024) qm=QuarkFusedMoEMethod
[WTRACE] layer_id=10 wname='...experts.w2_weight' shard=w2 eid=288 param=(289, 4096, 256) loaded=(4096, 2048) qm=QuarkFusedMoEMethod
```

The arithmetic, all of it checked against the checkpoint's `config.json`:

- `n_routed_experts = 288` → valid ids are 0..287. **`eid=288` is the shared
  expert**, renamed into a routed slot.
- `param` dim0 = **289** = the routed buffer grown by one to hold it.
- `moe_intermediate_size = 2048`. TP4 → 512. MXFP4 packs two 4-bit values per
  byte → **256**. That is `param` dim2.
- The routed experts arrive already packed: `loaded=(4096, 1024)`, TP4-sharded
  from 4096 → 1024, which halves again to 256 per-rank after unpacking bookkeeping.
- The shared expert arrives **BF16**: `loaded=(4096, 2048)`, TP4 → **512**.

512 vs 256. Exactly the 2:1 MXFP4 packing ratio.

## The code defect

`Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason`
(`glm5_next.py:1414`) takes a `quant_config` argument and **never reads it**. Its
branches cover `n_shared_experts`, the device (CUDA or gfx95 AITER), the SM
level, EP size and the DeepEP backend. Nothing about quantization.

Its two sibling families both have the guard:

- `deepseek_v2.py:3069` — `if quant_blocks_shared_experts_fusion(quant_config)`
- `deepseek_v4.py:3289` — same

The helper lives at `models/deepseek_common/utils.py:155` and duck-types
`quant_config.can_fuse_shared_expert()`. `QuarkConfig.can_fuse_shared_expert()`
(`layers/quantization/quark/quark.py:1001`) reads
`quantization_config.exclude`, which for this checkpoint lists
`model.layers.N.mlp.shared_experts.{gate,up,down}_proj` for every MoE layer,
and returns **`False`** — the correct answer, computed and never asked for.

The gate answering "fuse" makes the loader rename `mlp.shared_experts` into
routed slot `n_routed_experts` at `glm5_next.py:1735`, and the copy above fails.

## Where the defect came from

PR **#36607** commit `bd1cc98b` *"[AMD] Enable GLM shared-expert fusion on
gfx95"*. Its entire production change is nine lines swapping `if not _is_cuda`
for `if not (_is_cuda or _use_aiter_gfx95)`. Before it, ROCm always returned a
disable reason, so **this bug was unreachable on AMD**; that commit opened the
door and added no quantization guard. Its 56 accompanying test lines
(`TestGlm5NextGate`) cover backend / SM / EP / DeepEP and pass
`quant_config=None` in every case — the quant dimension is untested for this
model.

## Upstream status

- **Unfixed at #36607's head.** `refs/pull/36607/head` is exactly `c821c425`
  with zero commits beyond it.
- The only downstream movement is `c767511e` (2026-09-01, on #36507), which
  **reverts #36607 wholesale** — 22 files, removing the gfx95 enablement rather
  than guarding it. So there is no upstream fix to take, and rebasing onto
  post-revert #36507 would drop ROCm support entirely.
- `quant_blocks_shared_experts_fusion` appears upstream in three places only —
  its definition, `deepseek_v2`, `deepseek_v4` — and never in `glm5_next`.

## Same class, elsewhere

Recorded from the upstream-search pass; the numbers below are quoted from those
threads, not measured here.

| ref | what |
|---|---|
| **#37268** | The identical failure on NVIDIA — GLM-5.3-NVFP4 on H100 logs `Shared experts fusion optimization enabled.` then dies in `_load_w13` with 3072 vs 6144. Same 2× ratio. **Accepted workaround is the same flag.** |
| **#37325** | Fixes #37268 in exactly the shape our patch uses: add a gate branch, keep the enforce override, extend the same test module. |
| **#25261** | Same class for GLM5 AutoRound INT4 — and that one produced **silent wrong output** rather than a crash, because INT4 shapes happen to line up. MXFP4's 2:1 packing is why we got a loud failure instead of a quiet one. |
| **#8456** | Listed by the upstream search as a third prior fix of this class. Not read first-hand here. |
| **#37057** | *"[AMD] Enable GLM shared-expert fusion on gfx942"* (CLOSED) — extends the same gate and repeats the omission. It measures fusion at **+21.18 %** output tok/s (333.572 → 404.226) on **8×MI300X / gfx942 / TP8 / FP8**. **That is not our configuration** and is quoted only so nobody assumes the guard is free. |

**For this checkpoint the guard is free**: fusion does not load at all, so the
choice is not fast-vs-slow but **runs-vs-does-not-run**.

## Still open

The vendor model card (`../research/GLM-5.3-Flash-MXFP4.README.md`) publishes a
launch command with **no** `--disable-shared-experts-fusion` and reports a
successful 4×MI350 validation. Why that loads is **unknown**.

The check that would settle it: the exact sglang ref the card's
`-v "$PWD/python/sglang:..."` bind-mount pointed at, and whether that
validation's log contains `Shared experts fusion optimization enabled.` If the
ref predates `bd1cc98b`, fusion was simply unreachable on ROCm and there was
nothing to guard. Neither fact is in the card.

## Not fixed, and out of scope

The MTP / NextN draft layer 45 also has BF16 routed experts, and
`model.layers.45.mlp.experts` is absent from `exclude` (only deeper per-expert
entries are present, and those can never match a `FusedMoE` prefix). This is
independent of fusion and only reachable with speculative decoding on, which
this recipe does not use.
