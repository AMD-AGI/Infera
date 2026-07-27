# Notes — 01 single-node mix

## The load-bearing fix: DSA-ROCm envs (why the launch has them)

GLM-5.2 uses DSA (DeepSeek Sparse Attention). sglang's DSA top-k defaults to a **CUDA-only
`deepseek_v4` JIT kernel that will not build on gfx950**. Without rerouting it, startup fails.
The three envs route it through ROCm-friendly paths:
- `SGLANG_OPT_USE_TILELANG_INDEXER=1` — tilelang indexer instead of the CUDA kernel.
- `SGLANG_OPT_USE_TOPK_V2=0` — skip the failing topk_v2 JIT kernel.
- `SGLANG_OPT_USE_JIT_NORM=0` — HIP non-JIT norm.
Plus `SGLANG_USE_AITER=1` and `SGLANG_ROCM_FUSED_DECODE_MLA=0`, and the flags
`--nsa-prefill-backend tilelang --nsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3`.

**Why:** these are not perf tuning — they are correctness/bring-up requirements on gfx950.

## Gotchas / wrong turns

- **Image version matters.** Only sglang **0.5.15+** knows GLM-5.2's `head_dim=192`. Older rocm
  images (0.5.12 / 0.5.14) crash at weight load with `IndexError 18432 vs 16384`. rc6 = 0.5.15.post1.
- **Silent cold-start window is NOT a hang.** Sequence: weight load (282 shards) → aiter JIT
  (norm/rope/cache modules) → cudagraph capture (~55 s) → ready. Looks stalled for minutes; watch
  VRAM climb to ~267 GB/card (mem-fraction 0.85) as proof of progress. Do not kill.
- **200/health ≠ correct.** sglang happily serves fluent garbage if KV/parser config is wrong.
  Always run the temp=0 factual probe. (GLM-5.2 is a reasoning model, so it emits chain-of-thought
  before the answer — the probe substring-matches the fact, which is present.)
- **Do NOT copy the DSv4 example kit's flags** (`--attention-backend dsv4`, `--page-size 256`,
  `--disable-radix-cache`, big SGLANG_OPT_* block). Those are DeepSeek-V4-tuned and fight GLM's
  auto-config. GLM wants the minimal DSA recipe above; it auto-selects `attention_backend=dsa`,
  `page_size=64`.

## Reproducibility gaps (honest)

- The single-node run logged to container stdout (`docker logs`), and the container was removed at
  teardown — so no persistent log file is included here. The result numbers in `results/` are the
  captured bench output. To regenerate logs, re-run `scripts/launch.sh` (redirect to a file if you
  want them persisted).
- `mem-fraction-static 0.85` was used; `cuda-graph-max-bs 64` matched conc=64. Both are in
  `scripts/launch.sh`.
