# mHC dispatch in vLLM 9c1d45c2 (image of the PD stack) — code read 2026-11-21 [F, source/vllm-9c1d45c2]

Layer seam = `MHCPreDelayedOp.forward_hip` (vllm/model_executor/layers/mhc.py:~395-515), called from
`models/deepseek_v41/amd/model.py` (attention seam ~L475/506/524 with attn_norm, FFN seam ~L559 with ffn_norm).
Inputs per seam: residual [T, hc_mult=4, 5120] bf16, fn [24, 4*5120] fp32 (hc_mult3 = 4*(2+4)), hc_scale, hc_base,
eps values, sinkhorn_repeat (20), pre_mix [T,4], previous sublayer_out [T,5120] + post_layer_mix [T,4] + comb_res_mix [T,4,4],
norm_weight [5120] bf16. Outputs: residual_out [T,4,5120], post_mix [T,4,1], comb_mix [T,4,4], layer_input [T,5120],
next_pre_mix [T,4].

Dispatch order (gfx950):
1. `HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM` and norm fused (`fuse_seam_norm` = that flag) and supported:
   a. `fn_packed` present and bf16 norm (`use_mega_mhc`: hc_mult 4, no ubatching, `is_mega_mhc_available()`):
      **aiter HIP Mega-MHC** `aiter.ops.mega_mhc.mega_mhc` (one launch: post + gate GEMM + Sinkhorn + collapse + RMSNorm;
      split-K scratch per stream). If the seam writes MXFP8 for the next GEMM (`attn_seam_mxfp8` when fused_wqa_wkv takes
      MXFP8 input; `ffn_seam_mxfp8` only with aiter_mega_moe): `mhc_mega_post_pre_delayed_rms_norm_mxfp8` (amd/mhc.py).
   b. else `mhc_fused_post_pre_delayed_rms_norm_aiter` = aiter Triton `mhc_fused_post_pre_delayed_rmsnorm`.
2. no norm: aiter unfused `mhc_pre_delayed_aiter`; post folded only while T < 1024 on gfx950
   (`mhc_fused_post_pre_delayed_prefers_unfused`, aiter_ops.py:5711).
3. else TileLang / native reference.
Overlap (amd/mhc.py): side-stream gate projection only for T <= 128 (`MHC_OVERLAP_MAX_TOKENS`) and when
`supports_mhc_overlap` (CDNA4, fused flag, hc_mult 4, no ubatching).
Mono decode layer (`MonoDecodeLayer`, needs --monokernel) would absorb the decode seam; our PD configs drop --monokernel
(dp2_engine_v3.sh `args_drop --monokernel`).
Other mHC sites: model entry broadcast (x path) and final `mhc_post` (MHCPostOp) before the head; DSpark draft has its own
boundary (check trace).
To confirm from the live trace: kernel names (expect mega_mhc / mhc_fused_post_pre_delayed_rmsnorm Triton), calls per
step per layer (2 seams x 40 layers), token counts per step per role.
