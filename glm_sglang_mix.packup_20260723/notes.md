# Notes — gotchas, wrong turns, why the launch is minimal

Format: what / why / how / context. Read this before re-attacking GLM on SGLang.

## The single most important gotcha

**A running server tells you NOTHING about correctness.** SGLang will happily
report 200 / tokens-per-second on a broken KV or wrong-parser config while
emitting fluent garbage. ALWAYS run the temp=0 factual probe
(`scripts/glm_probe.py`) before believing the run works. Every "PASS" in this kit
is a temp=0 probe with the reply shown verbatim.

## Why the launch is minimal (the key insight)

`GlmMoeDsaForCausalLM` (GLM's MoE + DSA lightning indexer) is a
`DeepseekV2ForCausalLM` subclass, so SGLang routes it through the DeepSeek MLA +
DSA path and **auto-selects the right backends** (`attention_backend=dsa`,
`page_size=64`, tilelang prefill/decode, `kv_cache_dtype=bfloat16`). The correct
move is to let it — the working launch is just
`--tp-size 4 --trust-remote-code --mem-fraction-static 0.85 --reasoning-parser glm45`
with `SGLANG_USE_AITER=1`. See `environment.md` for the full auto-config.

## Flags DROPPED vs the reference recipe (what / why)

The reference (`sglang_single_r4_20260707_080726`) was a **DeepSeek-V4** run. Its
DSv4-specific flags are wrong for GLM and were removed:

- **`--attention-backend dsv4` / `--page-size 256`** — DSv4-model specific.
  GlmMoeDsa auto-selects `dsa` + `page_size=64`; forcing dsv4/256 fights the
  auto-config. **Removing them is REQUIRED, not optional.**
- **`--swa-full-tokens-ratio` / `--disable-radix-cache` /
  `--disable-shared-experts-fusion` / `--cuda-graph-max-bs 128`** and the large
  `SGLANG_OPT_*` env block — DSv4 throughput perf-tuning; unnecessary for a
  correctness bring-up. Left off to keep the surface minimal.

Added: **`--reasoning-parser glm45`** (splits GLM's `reasoning_content` cleanly).
Kept: `SGLANG_USE_AITER=1`. Changed: `--tp-size` 8→4 (one node, 4 cards).

## The silent "is it hung?" window (what / why / how)

Bring-up takes ~15 min, with an **~8-10 min silent window** after weight load and
before CUDA-graph capture: tokenizer init + **tilelang eager JIT compilation** +
**aiter GEMM autotuning**, all CPU-side with no log output. It looks exactly like
a hang. It is NOT — `py-spy dump` on the scheduler PID shows forward progress
through the JIT/tuning stack. **Do not kill a "stuck" server in this window.**
After it, CUDA-graph capture runs 52/52 (~3.5 min) and then `/health` goes 200.

Sequence: weight load (142 fp8 shards) → [silent JIT + GEMM tuning ~8-10 min] →
CUDA-graph capture 52/52 (~3.5 min) → ready.

## Smaller traps

- **`--trust-remote-code` is required** — GLM ships custom model code; without it
  the config/model won't load.
- **Nested-curl JSON is a trap** — probe via the urllib `.py`, not inline
  `curl -d '{...}'` through `docker exec bash -c "..."` (quoting mangles the body
  → HTTP 500/000 that looks like a hang but isn't).
- **Card discipline on chi2866** — it is the jump host with foreign `titan`
  training on card0-3; this run used card4-7 only. Never touch foreign cards or
  containers (`titan`/`zirui`/`primus-*`), never `scancel` slurm holds.

## Relationship to the rest of the GLM-on-every-engine effort

- **vLLM:** GLM-5.1 cross-node PD via MoRIIO — DONE (separate packup
  `moriio_pd_fix.packup_20260723`; required the `patch_moriio_pagelen.py` fix).
- **SGLang single-node mix:** THIS kit — DONE, no code fix needed, correct out of
  the box once the DSA auto-config is left alone.
- **atom:** next.
- The verified SGLang command here is also wired into the automated suite as a
  `pd_mixed/sglang` case (GLM_5_1_FP8), so it becomes a standing regression.
