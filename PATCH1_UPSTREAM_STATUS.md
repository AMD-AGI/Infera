# Patch 1 — upstreamed to sglang (HIP/aiter DSA padding fix)

**Patch:** `dsa_indexer_hip_dp_padded_rows.diff`
**File:** `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
**Status:** real, current, unreported-on-HIP bug → **PR opened against sglang `main`.**

## What patch 1 fixes

`Indexer._get_topk_paged` dispatches the paged-MQA-logits kernel across
backends. Under attn-tp / DP-attention (or MAX_LEN padding), `q_fp8` carries
padding rows up to the largest per-rank token count, while `seqlens`
(`dsa_seqlens_expanded`) is sized to the real count `q_offset`.

- The CUDA backends (`deepgemm_paged_mqa_logits_split` / `_native`,
  `cutedsl_paged_mqa_logits`) all take `q_offset=` and slice internally.
- The **aiter (HIP)** branch passes `q_fp8`/`weights` **unsliced** and sizes its
  `logits` from `q_fp8.shape[0]` → top-k sees `score.shape[0] != lengths.shape[0]`
  → assert `Expected lengths.size(0) == B`.
- The padding-restore after `topk_transform` is gated `if not _is_hip`, so HIP
  never restores either.

Fix: slice `q_fp8[:q_offset]` / `weights[:q_offset]` in the aiter branch (match
the CUDA contract), and drop the `not _is_hip` guard on the restore since every
backend now returns a `q_offset`-sized result. ~5 lines, no kernel change.

## Local vs upstream form

- **Local (this repo, base `v0.5.15.post1`):** the patch also carries debug
  scaffolding — `GLM52_P1V2` markers, `_p1v2_*` locals, and the
  `SGLANG_DEBUG_DSA_ROWS` / `_DSA_DEBUG_ROWS` logging block. Those are local-only
  and were **stripped** for the upstream PR.
- **Upstream PR (against `main`):** re-written against current `main`, whose
  `_get_topk_paged` has drifted (added cutedsl / dg_native branches,
  `_mask_init_and_local_tokens`). The clean diff is the two edits above with an
  upstream-neutral comment. Verified to compile.

## Upstream context

- **PR #32762 "[NPU] Fix DSA eager padding mismatch in PD MTP warm-up"** — same
  bug class, NPU platform, OPEN + CI-red. Our PR is the HIP/aiter sibling; cited.
- **PR #30839** (merged) — the eager-fallback path that first exposes the
  seed-unavailable case; cited as background.
- No PR covers the HIP/aiter path. HIP is still broken on `main`.

## PR

- **https://github.com/sgl-project/sglang/pull/33059** — "Fix DSA indexer aiter
  (HIP) padding mismatch under DP-attention" (OPEN, against `main`, 1 file
  +16/-4). Opened from fork `dorado269/sglang`, branch
  `fix-dsa-indexer-hip-aiter-dp-padding`.

## Local validation (base-branch form)

Single-node dp8+ep8 + EAGLE steps=3: 4/4 correctness, conc=64 1k/1k **256/256
zero failed** (see memory `project-glm52-dpa-mtp-fix`). The upstream port is the
same logic adapted to `main`; it was **not** re-run on `main`-based hardware from
the PR session — CI / a maintainer run is the confirmation on that base.
