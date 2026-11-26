# mHC round-4 MXFP8 overlay A/B with vLLM tool 3 at shape A (2026-11-26, n02-21 GPUs 6,7)

- A = stock image, container `yihou-inproc-v3a`. B = the same image with
  `/data/yihou/dsv41_kf_20261120/mhc/overlay_md/mega_mhc.py` bind-mounted read-only over `aiter/ops/mega_mhc.py`
  (container `yihou-inproc-v3b`, sha256 3fdbd728... vs stock ff092f6d...).
- Image `infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2`. On n02-21 this tag has id sha256:57ab3dc0... (created
  2026-11-26 03:04, revision 0007cd8087f1), not n03-33's 51ceb17d. Its `vllm` .py tree hash equals the n03-33
  copy (529a96ef...).
- Shape A uniform: DP2 TP1, C448 (224/rank), ISL 173000, OSL 1024, DSpark 5 synthetic 3.51, seed 1234, warmup 10,
  random KV fill, llying decode preset. Order A1 B1 A2 B2 A3 B3, same GPUs, sequential (`chain_ab.sh`, `chain_ab.out`).
- Overlay engaged: `module_mega_mhc_kfr4_h5120_2e4ee0851291.so` was built in B's JIT dir (B1 log shows the build) and
  was present after every B run.

| run | step p50 ms (rank 0 / rank 1) | step mean ms | max step ms | TPOT ms | realized accept | iterations | target / draft replays |
|---|---|---|---|---|---|---|---|
| A1 | 102.300 / 102.298 | 102.468 | 106.9 | 29.921 | 3.514781414237936 | 299 | 299 / 299 |
| B1 | 103.132 / 103.133 | 103.413 | 109.7 | 30.197 | 3.514781414237936 | 299 | 299 / 299 |
| A2 | 103.356 / 103.358 | 103.694 | 110.3 | 30.279 | 3.514781414237936 | 299 | 299 / 299 |
| B2 | 102.915 / 102.920 | 103.197 | 108.5 | 30.134 | 3.514781414237936 | 299 | 299 / 299 |
| A3 | 102.749 / 102.749 | 102.919 | 107.0 | 30.052 | 3.514781414237936 | 299 | 299 / 299 |
| B3 | 103.575 / 103.579 | 103.699 | 107.8 | 30.280 | 3.514781414237936 | 299 | 299 / 299 |

Per arm (step p50, rank 0): A mean 102.80 ms (range 102.30-103.36, spread 1.06 ms = 1.0 %), B mean 103.21 ms (range
102.92-103.58, spread 0.66 ms). Delta B vs A +0.39 %, inside A's own run-to-run spread. All six points pass
`verify_point.py`.

Why no difference is expected at this shape (code read plus trace, first-hand):
- The overlay's fused fork runs only when the selected variant is `mid256` or `mid384` (`_FUSED_VARIANTS`; the
  `kf_fused = None` fallback at overlay lines 524-535). Otherwise the stock launch is used.
- Stock `_select_stream_variant` picks `mid2048` for 1025-2048 tokens. One verify step here is 224 x 6 = 1344 tokens.
- The profiled runs show the same mega_mhc kernels in both arms, all with template argument 10 = index of `mid2048` in
  `_STREAM_VARIANTS`: three instantiations, 43 + 39 + 3 calls per step.

| profiled arm (window steps 20-24) | mega_mhc ms/step rank 0 / rank 1 |
|---|---|
| A (p_v3ab_A_u173k) | 4.232 / 4.268 |
| B (p_v3ab_B_u173k) | 4.251 / 4.155 |

Observation, not investigated: after the run, GPU 6 showed about 1 % VRAM / 1 % use. The only GPU processes left on
the node belonged to kf's `yihou-dsv41kf-tune-mxscale` container (HIP_VISIBLE_DEVICES=5, created 05:01:45 UTC, during
this A/B). Whether it touched GPU 6 during the runs is not established.
