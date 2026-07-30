# GLM-5.1-FP8 on ATOM — single-node bring-up

**Ran:** 2026-07-23
**Author:** c_huggingface
**Status:** ✅ **PASS** — GLM-5.1-FP8 comes up on ATOM (TP4, single-node) and
temp=0 factual probes are all correct.

## Goal

Third engine in the GLM-on-every-engine effort. GLM-5.1 already runs correctly on
vLLM (cross-node PD via MoRIIO) and SGLang (single-node mix). This kit brings
GLM-5.1-FP8 up on **ATOM single-node** and proves temp=0 correctness. Going in it
was genuinely unknown whether ATOM supports the GlmMoeDsa architecture — it does.

**Success criterion:** ATOM loads GlmMoeDsaForCausalLM and, at temp=0, answers
factual prompts correctly and coherently (France→Paris, China→Beijing, 2+2→4) —
not empty, not first-token-only, not garbage.

## Result

Came up on `infera/engine-atom:kimi` (TP4, card4-7). ATOM loaded
`GlmMoeDsaForCausalLM` (model_type `glm_moe_dsa`) cleanly — no unregistered-arch
abort — allocated the MLA chunked-prefill workspaces (so it handles the GlmMoeDsa
MLA + DSA path), captured cudagraphs, and served correct output. The launch is
minimal: `--kv_cache_dtype fp8 -tp 4` with `HSA_NO_SCRATCH_RECLAIM=1`.

| Model | Config | temp=0 probes | Verdict |
|-------|--------|---------------|---------|
| GLM-5.1-FP8 | ATOM single-node, TP4, fp8 KV | France→**Paris**, China→**Beijing**, 2+2→**4** | ✅ |

**No MTP needed** (unlike the DSv4/gfx942 reference): GLM ships no MTP/nextn draft
weights, and gfx950 plain decode is correct — the gfx942 broken-plain-decode bug
does not reproduce. Single-node uses no RDMA / MoRIIO / Mooncake.

## How to reproduce

See `REPRODUCE.md`. TL;DR: the atom image is already on the node (do NOT re-load
it) → run `scripts/glm_atom_mix.sh` → wait for "ready" → run `scripts/glm_probe.py`
→ all PASS → `scripts/down.sh`.

## Folder map

- `REPRODUCE.md` — step-by-step (container+server → probe → teardown)
- `environment.md` — exact HW/SW/image/paths/secrets + the disk-tight caveat
- `scripts/` — `glm_atom_mix.sh` (launch), `glm_probe.py` (temp=0), `down.sh`
- `results/evidence.md` — verbatim probe replies + launch cmd
- `results/agent_result_raw.md` — raw bring-up report (unedited source)
- `notes.md` — gotchas: thinking-model probe, no-MTP, gfx950-vs-gfx942, disk trap
- `logs/glm_atom_mix.log.gz` — full server log (gzipped)
