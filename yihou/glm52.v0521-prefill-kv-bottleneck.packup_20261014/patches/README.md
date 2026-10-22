# Patches (runtime bind-mounts, reduced leg only; image unchanged)

Diff base for both = the image's own sources, `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt`
(extracted from `infera-sglang:v0521-yihou-1010-mc4340`, ID 9d755dd6). `sha256.image-vs-mounts.txt` proves the
expert8 `.orig` files are byte-identical to the image files and records the sha256 of every mounted file and of the
model's `config.json`. Both patches make a **non-faithful model** (quality is meaningless; correctness is waived anyway
by `SGLANG_SIMULATE_ACC_LEN=3.61`). Attention/KV geometry is untouched, so prefill→decode KV transfer stays
layout-compatible between a reduced leg and a real 256-expert leg.

## 1. expert8 (`expert8/`) — rounds x8d, x8p, x8d2, x8pc

- **What:** keep routed experts 0..7 of every MoE layer (target and the NextN/MTP draft), slice the router gate and
  `e_score_correction_bias` to 8 rows, route every token to all 8 (`route_all_eight`: sigmoid weights, optional
  renormalise, ids 0..7). Files: `models/yihou_expert8.py` (new), hooks in `models/deepseek_v2.py` (custom routing on
  the TopK, log `YIHOU_EXPERT8 role=… routed=8 topk=8 ep=… local=…`), `model_loader/loader.py` (filtered safetensors
  iterator; experts ≥ 8 never read), `layers/moe/topk.py` (bypass for the custom routing function).
  Overlay `config.expert8.json` over `$MODEL/config.json`: `n_routed_experts 256→8`, `ep_size 1→4`,
  `yihou_synthetic_expert8: true` (`config.expert8-vs-model.keydiff.txt`; the model already has `num_experts_per_tok 8`,
  `n_group 1`, `topk_group 1`, which the patch asserts).
- **Why:** free MoE weight memory on ONE leg so that leg gets a bigger KV pool (user directive 2026-10-13 05:15), as in the
  2026-09-24 "26k" runs (8 experts on both legs, v0.5.19).
- **How:** `yihou-expert8-nextn.patch` (unchanged from the hicache-ceiling / `yihou-expert8-pd.packup_20260924` work)
  applied to the v0.5.21 image sources → `expert8/tree/python/sglang/srt/...` (mounted `:ro` by
  `scripts/harness/mounts.yihou.expert8.tsv`); unified diff vs the image: `expert8-vs-image.v0521.diff` (148 lines).
  Harness overlay `scripts/harness/config.yihou.expert8leg.sh`: `E8_LEG=decode|prefill`, that leg gets EP4 (2 local
  experts/rank), `--disable-shared-experts-fusion` (the patch raises with fused shared experts) and
  `--max-total-tokens ${E8_MAX_TOTAL_TOKENS:-3670016}` (3,670,016 = the 26k runs' cap; x8pc sets 2433088).
- **Context / checks:** prefill-leg logs show 304 `YIHOU_EXPERT8` lines; decode-leg 312 incl. 8 `role=draft` (the draft
  model is reduced too). audit v3 checks `{ROLE}_EXPERT8` (fusion off on that leg only) and `{ROLE}_MAX_TOTAL_TOKENS`.

## 2. expertn / 128 experts (`expertn/`) — round x128p (prefill leg only)

- **What:** keep routed experts 0..127, slice gate + correction bias to 128 rows; routing is the model's own sigmoid
  top-8 over 128 (no custom routing). Files: `yihou_expertn.py` (new module), `loader.yihou-expertn.py` (image
  `loader.py` + one hunk after `use_safetensors = resolved_source.use_safetensors`). Overlay `config.expert128.json`:
  `n_routed_experts 256→128`, `yihou_synthetic_keep_experts: 128` (`config.expert128-vs-model.keydiff.txt`). Diff vs
  image: `expertn-vs-image.v0521.diff` (66 lines).
- **Why:** user ~16:00 "prefill侧expert砍半" — a less extreme weight reduction than 8 experts (prefill KV 3,318,400).
- **How:** built by `scripts/session/build_expertn.py <N> <image srt dir>` from `yihou_expertn.build-src.py`
  (= `$N/yihou_expertn.py`, byte-identical to the mounted one) and `scripts/session/loader_hunk.py`; mounted by
  `mounts.yihou.expert128.tsv`; overlay `config.yihou.expertnleg.sh` (`EN_LEG=prefill` only — the loader hook refuses the
  draft model; EP stays 1; `--disable-shared-experts-fusion` because sglang only fuses for 256/384 routed experts; no KV cap).
- **Context:** log `YIHOU_EXPERTN keep=128 checkpoint loaded=… skipped=…`. Both x128p benches had exactly 180
  `InvalidInferenceResultError` (empty content) — not investigated; other rounds 0-5.

## Not patches but required (all rounds, both legs)

`scripts/harness/mounts-v0521/src/{kv_events_publisher.yihou-noblock.py, forward_mla_rocm.yihou-noabsorbcfg.py,
conn.yihou-regbase.py}` via `mounts.yihou.v2-regbase.tsv` — the session baseline (documented in the baseline kit and
`glm52.v0521-cap-c44c48.packup_20261013/patches/`).
