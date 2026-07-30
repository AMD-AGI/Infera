# Notes — gotchas, wrong turns, why the launch is minimal

Format: what / why / how / context. Read before re-attacking GLM on ATOM.

## The single most important gotcha (correctness)

**A "ready" server tells you NOTHING about correctness**, and GLM has a
model-specific trap on top: it is a **THINKING model**. With thinking ON (the
default) and a small `max_tokens`, the entire budget is spent inside the reasoning
preamble, so `message.content` comes back empty/truncated — which looks like a
failure but is NOT. Send `chat_template_kwargs={"enable_thinking": false}` and a
generous `max_tokens` (≥200) to get the direct answer. Every "PASS" here is a
temp=0 probe with thinking disabled and the reply shown verbatim.

## The disk trap (operational — nearly crashed the node)

The ATOM image is 45GB. `docker load -i .../dsv4_repro_atom_img.tar` on chi2866
(only ~34-77GB free on `/`, shared with a 263GB foreign titan layer) drove `/` to
**100% full and hung ssh**. The daemon did eventually register the image and
reclaim the temp space, but this is a landmine. Rules for this node:
- The image is ALREADY loaded (`infera/engine-atom:kimi`). Do NOT `docker load`/
  `pull` it again.
- Do NOT `docker system prune` / `image prune` broadly (foreign images).
- Keep the model RO-mounted (writable layer stays tiny); watch `df -h /` and stop
  if free approaches ~8GB.

## Why the launch is minimal + the no-MTP finding (what / why)

ATOM loaded `GlmMoeDsaForCausalLM` (model_type `glm_moe_dsa`) with no
unregistered-arch error and allocated the MLA chunked-prefill workspaces, so it
handles the GlmMoeDsa MLA+DSA path natively. The working launch is just
`--kv_cache_dtype fp8 -tp 4` + `HSA_NO_SCRATCH_RECLAIM=1`.

The DSv4 reference (gfx942) REQUIRED `--method mtp --num-speculative-tokens 3`
because plain decode on gfx942 was broken (only the first/prefill token emitted).
For GLM on gfx950 that does NOT apply, two independent reasons:
1. **No MTP weights:** GLM's config says `num_nextn_predict_layers=1` but the
   weight index has **0** mtp/nextn tensors — there are no draft weights to run
   speculative decode with. `--method mtp` would have nothing to load.
2. **gfx950 plain decode is correct:** decode advances fully (multi-token,
   coherent, clean stop) — the gfx942 broken-plain-decode bug does not reproduce
   on this hardware. So plain decode is both necessary (no MTP option) and
   sufficient (it's correct).

No `ATOM_USE_TRITON_MOE` / dp-attn / tbo / FP4 knobs were needed.

## Flags vs the DSv4 reference (summary)
- Kept: `--kv_cache_dtype fp8`, `HSA_NO_SCRATCH_RECLAIM=1`.
- Changed: `-tp 4` (card4-7) not `-tp 8`.
- Dropped: `--method mtp --num-speculative-tokens 3` (see above).
- Added: nothing model-specific — GLM ships `chat_template.jinja` so
  `/v1/chat/completions` works out of the box (DSv4 needed a custom template path).

## Smaller traps
- **Nested-curl JSON is a trap** — probe via the urllib `.py`, not inline
  `curl -d '{...}'` through `docker exec bash -c "..."` (quoting mangles the body).
- **Card discipline on chi2866** — jump host with foreign titan on card0-3; use
  card4-7 only. Never touch foreign cards/containers, never `scancel` slurm holds.
- **`--server-port` not `--port`** — for the infera-native launch
  (`python -m infera.engine.atom`), ATOM's `--port` is the torch-dist MASTER_PORT;
  the OpenAI HTTP port is `--server-port` (see the atom adapter docstring).

## Relationship to the rest of the GLM-on-every-engine effort

- **vLLM:** GLM-5.1 cross-node PD via MoRIIO — DONE (packup
  `moriio_pd_fix.packup_20260723`; needed `patch_moriio_pagelen.py`).
- **SGLang:** single-node mix — DONE (packup `glm_sglang_mix.packup_20260723`).
- **ATOM:** THIS kit — single-node — DONE, no code fix, correct out of the box on
  gfx950 (no MTP).
- All three verified commands are also wired into the automated suite as e2e cases
  (`GLM_5_1_FP8` in vLLM pd_disag/MoRIIO, and sglang + atom pd_mixed).
