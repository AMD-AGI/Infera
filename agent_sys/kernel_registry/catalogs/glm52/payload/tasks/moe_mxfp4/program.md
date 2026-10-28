> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Program: GLM-5.2-MXFP4 decode routed MoE (a4w4 MXFP4, aiter FlyDSL) on MI355X

**GPU**: gfx950 (AMD Instinct MI355X, 256 CUs, 256 MB MALL, HBM ~8 TB/s) — **Backend**: aiter FlyDSL
(the production GEMMs are FlyDSL kernels; HIP/Triton/CK replacements are allowed, see rules).

## Objective

Make `glm52_moe_mxfp4` in `moe_mxfp4_kernel.py` faster for the GLM-5.2 decode leg while staying
correct. It reproduces sglang v0.5.21's production MoE for this deployment (Quark MXFP4 scheme ->
aiter runner -> `aiter.fused_moe`), called 75× per TARGET_VERIFY (one per MoE layer). It is the
largest decode kernel: **24.6 %** of the distribution-weighted decode iteration.

## Shapes (per rank; driver cases)

TP4 / attention-DP4 / EP1. MoE rows per rank after the DP all-gather: M = 4 × 6 × bs_graph =
24·bs, bs_graph ∈ {8, 10, 12, 14} (P = 2 / 42 / 49 / 7 %) → M = 192 / 240 / 288 / 336.

| tensor | shape | dtype / layout |
|---|---|---|
| hidden_states | [M, 6144] | bf16 |
| topk_ids | [M, 9] | int32; column 8 is always 256 (fused shared expert) |
| topk_weights | [M, 9] | fp32 (workset blob: routed 8 sum to 1.0, shared = 1.0) |
| w13 | [257, 1024, 3072] | MXFP4 packed e2m1 (`float4_e2m1fn_x2` view), `shuffle_weight((16,16))`; rows [0,512) gate, [512,1024) up |
| w13_scale | [257, 1024, 192] | UE8M0 uint8, `e8m0_shuffle` layout |
| w2 | [257, 6144, 256] | MXFP4, `shuffle_weight((16,16))` |
| w2_scale | [257, 6144, 16] | UE8M0, `e8m0_shuffle` |
| output | [M, 6144] | bf16, Σ_k w_k · expert_k(x) |

Every case routes to **all 257 experts** (blob check), so each call streams the whole per-rank
expert set: 257 × (3,145,728 + 196,608 + 1,572,864 + 98,304) B = **1.288 GB per call**.

Scored cases: **T60_bs10 and T72_bs12** (91 % of the mass). T48_bs8 and T84_bs14 are timed and must
stay correct but are unscored. Each graph replay runs the op over 4 distinct weight sets (5.2 GB);
case_ms = replay / 4.

## Production call path (image historical-sglang-image-d)

sglang 11bfd868, aiter e7d2453. Files (md5): `quark_w4a4_mxfp4_moe.py` dc08c0e3…,
`moe_runner/aiter.py` f8a239d0…, `kernels/ops/moe/moe_sorting_small.py` dc8a07d7…,
`aiter/fused_moe.py` b948cc86….

1. Weight load (once): `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:946-992`
   `process_weights_after_loading` (gfx950, activation silu → not the GU-interleaved branch at 933-944):
   `e8m0_shuffle` of both scales, `shuffle_weight(w, (16,16))` of both weights, `is_shuffled=True`.
   → `prepare_weights()` (driver setup, not timed). I = 512 is 128-aligned, so
   `intermediate_pad = 0` (`get_moe_weight_sizes`, `moe/utils.py:1054`, AITER_PADDING_SIZE 128).
2. Forward: `apply_weights` (`quark_w4a4_mxfp4_moe.py:1017-1068`: weights viewed as
   `float4_e2m1fn_x2`, `gate_mode=SEPARATED` on gfx95, `swiglu_limit=0`, quant PER_1X32)
   → `moe_runner/aiter.py:259-395` `AiterRunnerCore.run`:
   * `smallm_moe_gfx950` (line 336) is **not** taken: it requires hidden 4096 and topk 10/11.
   * `apply_aiter_small_moe_sort_patch` (installed in `__init__`, line 257) wraps `fused_moe`, but
     is inert here: it needs `topk_ids.numel() <= 256` (ours: 1728–3024).
   * `expert_mask=None` (EP1: `StandardDispatcher.expert_mask_gpu` is only built when
     `moe_ep_size > 1`), `moe_sorting_dispatch_policy = SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY`
     = 2 (default, `environ.py:908`; deployment value not seen — assumption), activation Silu.
   → `aiter.fused_moe.fused_moe` → `_fused_moe_impl` → `get_2stage_cfgs(get_padded_M(M), …)` looks
   up the tuned row (`aiter/configs/model_configs/*` merged into `tuned_fmoe.csv`):

| case | padded M | stage1 kernelName1 | stage2 kernelName2 | block_m |
|---|---|---|---|---|
| bs8, bs10 | 256 | flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4 | flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_persist_nt_sbm32 | 32 |
| bs12, bs14 | 512 | flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4 | flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_nt_sbm32 | 32 |

GPU kernels per call (torch profiler, eager, 4 rotating weight sets, µs per call):

| kernel | bs8 | bs10 | bs12 | bs14 |
|---|---|---|---|---|
| opus_moe_sorting P0_v2 + P23 (sort, zero output) | 9.5 | 9.4 | 10.4 | 10.3 |
| activation MXFP4 quant + scale sort | 11.7 (fused_mx_quant_moe_sort_kernel) | 9.1 (dynamic_per_group_scaled_quant + mxfp4_moe_sort_kernel) | 9.5 | 9.2 |
| `mfma_moe1_silu_mul_afp4_wfp4_fp4_t32x64x256_…_fp4q` (gemm1 + SiLU·up + MXFP4 quant) | 138.3 | 138.7 | 139.8 | 140.9 |
| `gemm2_a4w4_port_…_bm32_…_atomic…` (gemm2 × topk weight, bf16 atomics) | 78.1 | 77.2 | 76.9 | 77.4 |
| sum | 237.5 | 234.3 | 236.6 | 237.8 |

Effective bandwidth: gemm1 reads 859 MB in ~139 µs (6.2 TB/s), gemm2 430 MB in ~77 µs (5.6 TB/s).
GEMM1+GEMM2 are 91–92 % of the op; sort + quant are ~19 µs.

**AITER_USE_FLYDSL_MOE_SORTING=1** (set in the deployment, set by the driver) only switches
`moe_sorting` to the FlyDSL sorter when `dispatch_policy == 0` (`aiter/fused_moe.py:614-633`). With
sglang's default policy 2 the opus multi-phase sorter runs (verified: `_moe_sorting_impl`,
`use_opus=True`). Policy 0 was measured as an A/B: same kernel time within noise (graph: bs10
227.7 vs 233.8 µs, bs12 223.0 vs 229.9 µs, same process).

**routed_scaling_factor (2.5)** is not applied inside this op: `aiter.fused_moe` has no such
parameter, the Quark MXFP4 scheme file does not define `fuse_routed_scaling_factor_in_topk`, and
`deepseek_v2.py:1165-1170` skips `final_hidden_states *= routed_scaling_factor` when `_use_aiter`.
Where the factor ends up on the aiter path (routing weights vs. elsewhere) is outside this op and
was not established here; the driver feeds the workset blob weights unchanged.

## Pristine baseline (this driver, redacted-node, GPU 1, 2026-10-20)

| case | run 1 | run 2 | in-model anchor (v0519 image) |
|---|---|---|---|
| T48_bs8 (unscored) | 0.2363 ms | 0.2359 ms | ~0.179 ms |
| **T60_bs10** | **0.2340 ms** | **0.2332 ms** | ~0.188 ms |
| **T72_bs12** | **0.2362 ms** | **0.2348 ms** | ~0.199 ms |
| T84_bs14 (unscored) | 0.2366 ms | 0.2365 ms | ~0.203 ms |
| median_ms (P-weighted) | 0.235286 | 0.234251 | |

Run-to-run spread of median_ms 0.44 %. A separate process on the same GPU measured 2–3 % lower
graph times for the same configuration (breakdown run: bs12 229.9 µs) — treat ~3 % as the
noise floor until it is characterised.

**Gap to the anchors is not explained.** Measured and ruled out as the cause: MALL warmth
(1 vs 4 weight sets: ≤ 2 %), sorting policy (≤ 3 %). The anchors come from the older v0519 image
(sglang a9fb1c3238 / aiter 4ad9983) whose kernel choice for this shape was not available here;
the measurement that would answer it is the per-kernel breakdown of that profile, or this driver
run inside that image. Note the shape of the difference: the anchors grow with bs (179→203 µs),
the pristine kernel is flat (~235 µs) — the current op is bandwidth-bound on the 1.29 GB weight
stream at all four sizes.

## Correctness

Gate: SNR ≥ **30 dB** vs the fp32 oracle (min over 4 cases × 2 activation seeds). Pristine:
46.29–46.30 dB (bs8/bs10, padded-M 256 config), 50.55–50.71 dB (bs12/bs14). The output is not
bitwise reproducible (bf16 atomics in gemm2; run-to-run SNR 55–56 dB).

### Oracle adaptations (fixed in the oracle, not the kernel) — each measured first-hand

Starting point: the workset reference `moe_mxfp4_routed_topk9_e257_h6144_i512.json`.

1. **Activation quant scale rule.** The reference uses `scale = 2^(floor(log2 amax) − 2)` with
   ties to the smaller magnitude. aiter's HIP activation quantizer
   (`dynamic_per_group_scaled_quant<bf16, fp4>`, used by `fused_dynamic_mxfp4_quant_moe_sort`)
   uses **`scale = 2^ceil(log2(amax/6))`** (never clips) and **round-to-nearest-even** elements:
   its e8m0 bytes equal `ceil(log2(amax/6))+127` for 100 % of 55,296 groups, and with that scale +
   RNE all 1,769,472 elements match bit-exactly (`_src/probe_quant2.py`). The reference rule
   mismatches 144,574 elements; Quark 'even' + RNE mismatches 59,524.
2. **Intermediate (GEMM2 input) quant.** Quantized inside the stage1 kernel (`fp4q`). End-to-end
   SNR vs production with the inter rule varied (activation rule fixed to 1.): ceil6+RNE 46.3 /
   50.7 dB; Quark-even+RNE 21.1 / 21.3 dB; workset reference 16.9 / 16.8 dB; no quant 16.2 dB
   (bs10 / bs12, `_src/oracle_variants.log`). The oracle uses ceil6+RNE for both GEMM inputs. The
   workset reference as written scores 13.0 / 12.8 dB against production.
3. **Confirmed unchanged:** low nibble = even element; gate rows first (up-first scores 2.0 dB);
   SiLU(gate)·up; topk weight applied after GEMM2; shared expert id 256 treated as a normal expert.
4. **Weights:** the checkpoint quantizer is Quark `scale_calculation_mode: even`. aiter's
   `dynamic_mxfp4_quant` (used to build the weights) matches Quark even-round scale + RNE bit-exactly
   on 1,769,472 elements, including a grid-midpoint stress set (`_src/probe_quant.py`).

## Modification rules

1. Keep `glm52_moe_mxfp4(hidden_states, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids)`
   and its semantics/return (bf16 [M, 6144], a4w4 numerics: MXFP4 activations into both GEMMs).
   It must stay droppable into sglang's aiter MoE runner / aiter `fused_moe`.
2. Weight-layout changes are allowed **only** in `prepare_weights` (it maps to
   `process_weights_after_loading`, runs once at load, untimed); its inputs stay the checkpoint
   layout. Any per-call work must be inside `glm52_moe_mxfp4`.
3. Must be CUDA/HIP-graph capturable on the current stream (the driver verifies capture by SNR).
   Shapes other than the four cases (other M, prefill sizes) must keep working.
4. Do NOT edit `driver.py`, `graph_harness.py` or `data/`. No caching of results across calls,
   no dependence on the driver's fixed inputs (routing, weights).
5. Do not modify `REDACTED_LOCAL_PATH` in the container; copy aiter/FlyDSL kernel source into the
   kernel file (or next to it) if you need to change it.

## Environment notes

The driver sets (before importing aiter): `SGLANG_USE_AITER=1`, `AITER_USE_FLYDSL_MOE_SORTING=1`,
`AITER_CONFIG_FMOE=REDACTED_LOCAL_PATH` (container runs non-root: aiter's runtime
re-merge into root-owned `REDACTED_LOCAL_PATH` fails; the image's pre-merged copy has the same 2870
rows as a fresh merge, row order differs; the lookup for our key hits the primary table, which has
no duplicate rows), and `FLYDSL_RUNTIME_CACHE_DIR=REDACTED_LOCAL_PATH` (the image's
prebuilt FlyDSL cache pkls are root-only mode 600, so kernels are compiled from the same source on
first use; equality with the prebuilt binaries is assumed, not verified). `--profile-run` builds
one weight set (randn, `dynamic_mxfp4_quant`, shuffles) before the 6 timed-op launches ×6.
