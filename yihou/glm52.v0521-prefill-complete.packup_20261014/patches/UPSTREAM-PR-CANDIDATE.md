# Upstream PR candidate (NOT filed) — sgl-project/sglang

**Title:** `[PD] Fix prefill_complete validation rejecting attention-DP (attn_dp_size) deployments`

**Status:** candidate only. Upstream main at `536d0653703e9977132d8b179251b113423945b3` (2026-10-14T02:00:42Z) still has
`attn_dp_size = cfg.dp_size if cfg.enable_dp_attention else 1` at `python/sglang/srt/arg_groups/pd_disaggregation_hook.py:214`
(checked with `gh api .../contents/...?ref=main`). `gh search prs/issues` for "prefill_complete" found #40703 (feature, merged),
#43442 (merged), #43811 (open), #43970 (open, NPU) — none touches this check.

**Problem:** `--disaggregation-decode-allocation-policy prefill_complete` with `--tp-size 4 --dp-size 4 --enable-dp-attention`
fails at startup: "does not yet support attention tensor parallelism greater than one". `handle_deprecated_dp_attention` has
already moved the width to `attn_dp_size` and cleared `enable_dp_attention`, so the validator computes attention-DP = 1.
The same expression ignores `--attn-dp-size N` given directly.

**Proposed change (upstream form, drop the YIHOU comment):**
```python
-    attn_dp_size = cfg.dp_size if cfg.enable_dp_attention else 1
+    attn_dp_size = cfg.attn_dp_size
```
(After `handle_deprecated_dp_attention`, `attn_dp_size` is authoritative for both spellings; our local mount kept a fallback to the
old expression to stay byte-minimal against the image. Needs a check that no other resolution path leaves `attn_dp_size == 1`
with attention DP on.)

**Evidence:** GLM-5.2-MXFP4, MI355X, same-node 1P1D TP4/attention-DP4 per leg, image `0.5.21.dev20261008+g11b1fb21de`:
stock → startup ValueError (`results/rounds/pc48/prefill-0.startup-ValueError.log`); patched → both legs report
`disaggregation_decode_allocation_policy=prefill_complete`, two 3600 s AgentX C48 runs completed (A1 3773 ok / 1 err,
A2 4027 ok / 3 err).

**Before filing:** rebase onto current main, add a unit test in the arg-resolution tests (TP4 + attention DP4 + prefill_complete
must validate), run the fast CI, and follow the repo's own skill `open-source-pr` (no assistant co-author trailer upstream).
