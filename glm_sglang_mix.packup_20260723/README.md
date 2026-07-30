# GLM-5.1-FP8 on SGLang — single-node PD-mix bring-up

**Ran:** 2026-07-23
**Author:** c_huggingface
**Status:** ✅ **PASS** — GLM-5.1-FP8 comes up on SGLang single-node (TP4) and
temp=0 factual probes are all correct.

## Goal

We already have GLM-5.1 running correctly on the vLLM engine (cross-node PD via
MoRIIO). The broader intent is to get **GLM-5.1 running on every engine**. This
kit is the first SGLang step: bring GLM-5.1-FP8 up on **SGLang single-node
PD-mix** (one server, prefill+decode together — not disaggregated) and prove it
produces correct output at temp=0.

**Success criterion:** the server registers and, at temp=0, answers factual
prompts correctly and coherently (France→Paris, China→Beijing, 2+2→4) — not
fluent garbage.

## Result

Came up on `lmsysorg/sglang:v0.5.14-rocm720-mi35x` (TP4, card4-7). The minimal
launch is `--tp-size 4 --trust-remote-code --mem-fraction-static 0.85
--reasoning-parser glm45` with `SGLANG_USE_AITER=1`. GLM's `GlmMoeDsaForCausalLM`
auto-selects the DSA attention path, so the DSv4-specific flags from the older
reference recipe are dropped ON PURPOSE (they fight the auto-config).

| Model | Config | temp=0 probes | Verdict |
|-------|--------|---------------|---------|
| GLM-5.1-FP8 | SGLang single-node mix, TP4 | France→**Paris**, China→**Beijing**, 2+2→**4** | ✅ |

Single-node mix uses **no RDMA / ionic / MoRIIO / Mooncake** — it's one server on
one node.

## How to reproduce

See `REPRODUCE.md`. TL;DR: `docker run` the sglang image → run
`scripts/glm_sglang_mix.sh` inside it → poll `/health` (~15 min, ~8-10 min silent
JIT/tuning window is normal) → run `scripts/glm_probe.py` → all PASS.

## Folder map

- `REPRODUCE.md` — step-by-step (container → launch → probe → teardown)
- `environment.md` — exact HW/SW/image/paths/secrets + sglang's auto-config
- `scripts/` — `glm_sglang_mix.sh` (launch) + `glm_probe.py` (temp=0 correctness)
- `results/evidence.md` — the verbatim probe replies + sglang's chosen config
- `results/agent_result_raw.md` — the raw bring-up report (unedited source)
- `notes.md` — gotchas: the silent JIT window, DSv4-vs-GLM flag divergence, why minimal
- `logs/glm_sglang_mix.log.gz` — full server log (gzipped)
