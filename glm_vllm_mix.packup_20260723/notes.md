# Notes — gotchas, why the launch is minimal

Format: what / why / how / context. Read before re-attacking GLM on vLLM.

## The single most important gotcha (correctness)

**A "ready" server tells you NOTHING about correctness**, and GLM adds a
model-specific trap: it is a **THINKING model**. With thinking ON (default) and a
small `max_tokens`, the entire budget is spent inside the reasoning preamble, so
`message.content` comes back empty/truncated — looks like a failure, is NOT. Send
`chat_template_kwargs={"enable_thinking": false}` and a generous `max_tokens`
(≥200) for the direct answer. Every "PASS" here is a temp=0 probe with thinking
disabled and the reply shown verbatim.

## Single-node mix uses NO kv-transfer connector (what / why)

This is PD-**mixed**: one server does prefill+decode on one node. There is NO
cross-node KV movement, so `--kv-transfer-config` / MoRIIO / Mooncake are NOT set.
That is the whole difference from the `pd_disag/vllm` path (which DOES use a
connector — Mooncake by default, MoRIIO for the GLM regression case). Consequently
the MoRIIO page-len fix baked into this image is a **no-op** for this run — the run
only confirms the image serves GLM correctly, it does not exercise the fix.

## Why these flags (what / why)

Mirrors the known-good GLM vLLM recipe: `--kv-cache-dtype fp8` (GLM MLA is fp8),
`--reasoning-parser glm45` (splits reasoning cleanly), aiter via env
(`VLLM_ROCM_USE_AITER=1` + fusion-shared-experts), `--distributed-executor-backend
mp`, `--no-enable-prefix-caching`, `--gpu-memory-utilization 0.85` (no OOM),
`--max-model-len 9472`. Did NOT pass `--moe-backend aiter` — aiter is already
active via env in the plain-serve path. TP4.

## Smaller traps
- **Port collision:** host port 8000 collided with a pre-existing `glm_pd`
  container on chi2879. The fix: don't publish `-p`; probe via `docker exec`
  localhost inside the container. (Cosmetic — the server itself was fine.)
- **`vllm serve --help` errors in-image without a GPU** — it does device inference
  at parse time; harmless, run the real command instead.
- **Nested-curl JSON is a trap** — use the urllib probe, not inline `curl -d`.
- **CUDA-graph / torch.compile silent window** (~3-7 min) is normal; don't kill.

## Relationship to the rest of the GLM-on-every-engine effort

- **vLLM cross-node PD (MoRIIO):** DONE (packup `moriio_pd_fix.packup_20260723`;
  needed `patch_moriio_pagelen.py`).
- **vLLM single-node mix:** THIS kit — DONE, no kv-transfer, correct.
- **SGLang single-node mix:** DONE (`glm_sglang_mix.packup_20260723`).
- **ATOM single-node:** DONE (`glm_atom_mix.packup_20260723`).
- All four verified commands are wired into the automated suite as e2e cases
  (`GLM_5_1_FP8`: vLLM pd_disag MoRIIO + vLLM/sglang/atom pd_mixed).
