# GLM-5.1-FP8 on vLLM — single-node PD-mix bring-up

**Ran:** 2026-07-23
**Author:** c_huggingface
**Status:** ✅ **PASS** — GLM-5.1-FP8 comes up on vLLM single-node (TP4) and
temp=0 factual probes are all correct.

## Goal

Fills the last gap in the GLM-on-every-engine matrix. GLM-5.1 was already verified
on vLLM cross-node PD (MoRIIO), SGLang single-node mix, and ATOM single-node. This
kit adds **vLLM single-node PD-mix** (one server, prefill+decode together, no
cross-node KV transfer). Secondary benefit: it exercises the freshly-built image
`infera/engine-vllm:test-local` (which contains the MoRIIO page-len fix) on GLM —
though single-node mix does NOT use the KV-transfer path, so the fix is a no-op
here; this is purely a brings-up-and-is-correct check.

**Success criterion:** vLLM serves GLM-5.1-FP8 and, at temp=0, answers factual
prompts correctly and coherently (France→Paris, China→Beijing, 2+2→4).

## Result

Came up cleanly on `infera/engine-vllm:test-local` (id e91a6d7d3a91, TP4, card0-3);
health 200, CUDA graphs captured, no OOM at gpu-util 0.85, ready in ~6-7 min. The
launch is a plain `vllm serve` with **no kv-transfer-config** (that's the
pd_disag path). fp8 KV + aiter (via env) + `--reasoning-parser glm45`.

| Model | Config | temp=0 probes | Verdict |
|-------|--------|---------------|---------|
| GLM-5.1-FP8 | vLLM single-node mix, TP4, fp8 KV | France→**Paris**, China→**Beijing**, 2+2→**4** | ✅ |

## How to reproduce

See `REPRODUCE.md`. TL;DR: `scripts/glm_vllm_mix.sh` (docker run + `vllm serve`)
→ poll `/health` → `scripts/glm_probe.py` → all PASS → `docker rm -f`.

## Folder map

- `REPRODUCE.md` — step-by-step (container+serve → probe → teardown)
- `environment.md` — exact HW/SW/image/paths/secrets
- `scripts/` — `glm_vllm_mix.sh` (launch), `glm_probe.py` (temp=0)
- `results/evidence.md` — verbatim probe replies + launch cmd
- `results/agent_result_raw.md` — raw bring-up report (unedited source)
- `notes.md` — gotchas: thinking-model probe, no kv-transfer, port collision
- `logs/glm_vllm_mix.log.gz` — full server log (gzipped)
