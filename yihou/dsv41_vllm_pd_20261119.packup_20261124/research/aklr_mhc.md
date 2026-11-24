# AKLR (AMD-AGI/AKLR) — mHC optimizations relevant to DeepSeek-V4.1-Flash

Date: 2026-11-20. Method: `gh` CLI, read-only (PR/issue search over all states with terms mhc, mHC, hyper,
hyperconnection, hc_pre, hc_post, sinkhorn, deepseek_v4, dsv4, dsv41, v4.1, V4; `git/trees/main?recursive=1`;
file contents and commit history of `aklr/ops/mhc`). Repo is PUBLIC and accessible; default branch `main`.
Everything below is read first-hand from PR bodies / files unless marked *(inferred)*.

## 0. Bottom line

- **Hidden size matters.** DeepSeek-V4.1-Flash: `hidden_size 5120`, `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6`
  (read from `/perf_apps/chaox/models/DeepSeek-V4.1-Flash/config.json`).
- **On AKLR `main` there is NO mHC kernel at hidden 5120.** The `aklr/ops/mhc` kernels on main are at hidden 4096
  (`mhc_fused_post_pre`, FlyDSL) and hidden 7168 (`mhc_fused_post_pre` HIP + FlyDSL; `mhc_pre` definition only,
  kernel withdrawn). Which DeepSeek models those sizes belong to is not stated in the schemas (`models` field is null)
  *(inferred: 4096 = DeepSeek-V4-Flash, 7168 = DeepSeek-V4-Pro — not verified)*.
- **The only AKLR work that targets V4.1-Flash's mHC (hidden 5120) is open PR #88** (sglang 0.5.23 plugin, Triton/GEAK
  kernels, not merged, not CI-run, no e2e A/B through AKLR). It is sglang-only; no vLLM route.
- #288 (open, 11-19) states for the V4.1 sglang unfused path: "The mHC boundary, the indexer's logits and the top-k
  select have no op." #289 (closed 11-20) defined V4.1 monokernel ops that include the mHC boundary but had no kernel.
- #271 (merged 11-20) makes AKLR serve aiter's `mhc_pre` / `mhc_fused_post_pre` entry points via aiter's plugin
  provider. With no 5120 kernel on main, V4.1-Flash calls would stay on aiter *(inferred from the dispatch design;
  not tested)*.

## 1. AKLR hits (mHC)

| PR | Title | Author | State / dates | Op(s), shapes, dtype, GPU | Claimed speedup + method | Integration |
|---|---|---|---|---|---|---|
| [#88](https://github.com/AMD-AGI/AKLR/pull/88) | feat(plugins): DeepSeek-V4 mHC boundary + RMSNorm: ops, sglang 0.5.23 routes and MI355X kernels | zhaoan12-prc | OPEN, created 2026-11-06 | 3 new ops in sglang plugin `deepseek_v4`: `deepseek_v4_hc_boundary_norm@1.0` (bf16 norm), `deepseek_v4_hc_boundary_norm_mxfp8@1.0` (norm + MXFP8 codes/scales), `deepseek_v4_hc_pre_norm@1.0` (DSpark draft boundary). **Hidden 5120**, Triton (GEAK-authored), MI355X/gfx950. One launch per boundary replacing sglang's `hc_boundary_fused_deferred` + `rmsnorm_with_sinkhorn` (draft: 5-launch chain); uses a self-resetting grid barrier. | AKLR validate bench on 633 exported workloads (DSv4.1-Flash FP4, sglang 0.5.23, InferenceX MI355X AgentX recipes c32 TP2 and c64 TP2+DP2-attn, DSpark block 5): boundary_norm 1.02–1.48x (184/314 kept), mxfp8 1.02–1.30x (199/292), draft pre_norm 1.83–2.34x (27/27); kept only if >= 1.02x; dropped decode sizes 260–288 and prefill 572–16384 tokens. Body also says the kernel set is "part of the DeepSeek-V4.1-Flash c32 set measured at +9.52% P90 interactivity on sglang 0.5.23" — but "No end-to-end A/B through the AKLR plugin; the gains quoted are the GEAK overlays of the same kernels." gsm8k plugin on/off: c32 0.9674/0.9636, c64 0.9629/0.9621. | sglang 0.5.23 routes only (`deepseek_v4_platform_stage.forward_hc_pre_from_prev_fused_boundary`, `NonCudaDSparkMixin._forward_hc_pre_from_prev`). Falls back on TP4 all-reduce fusions (`_can_fuse_mhc`), multi-node/EP/EPLB/MoRI, non-gfx950. CI and `ci:gpu` not run. No vLLM. |
| [#40](https://github.com/AMD-AGI/AKLR/pull/40) | feat(dsv4): import AITER quantization, mHC and CSR prefill kernels | zihaoanllm | MERGED 2026-11-05 (opened 11-04) | `mhc_fused_post_pre@1.0` HIP kernel, hidden 7168 with norm weight, MI355X/gfx950 ("mHC decode" from TP8 and TP8/EP8 campaigns) | 1.218–1.275x vs AITER `mhc_fused_post_pre`, 6/6 Points, AKLR validator graph-ratio lower bound. On main now: n_tokens 1,2,4,7,8,16 recorded 1.232–1.244x, ~9 us. | AKLR op only. |
| [#200](https://github.com/AMD-AGI/AKLR/pull/200) | feat(flydsl): mHC pre kernel for hidden size 7168 | zhenhuang12 | MERGED 2026-11-14 (opened 11-13) | `mhc_pre@1.0`, hidden 7168, no norm weight, FlyDSL, gfx950; split-K matrix-core projection + Sinkhorn second launch; n_tokens 1–128 | 0.9775–1.0877x vs AKLR baseline (16 rounds/Point lower bound); passed n_tokens 2–32 (1.05–1.09x), demoted 1, 64, 128. KernelForge campaign. | **Kernel later removed by #253** (2026-11-18): vs aiter v0.1.29 best ratio 1.02 (< 1.05 gate); only the kernel-less schema remains on main. |
| [#202](https://github.com/AMD-AGI/AKLR/pull/202) | feat(flydsl): fused mHC post-pre kernel for hidden size 7168 with norm weight | zhenhuang12 | MERGED 2026-11-14 (opened 11-13) | `mhc_fused_post_pre@1.0`, hidden 7168, norm weight, FlyDSL, gfx950; k_mm (post + projection + Gram partials) + k_fin (gates, Sinkhorn, layer_input); single fused launch for n<=4; n_tokens 1–128 | vs baseline 1.2755–1.5289x; vs incumbent HIP kernel (#40) 0.99–1.15x at n 1–16. KernelForge campaign. On main now: 1.299–1.515x, 7.3–12.0 us. Review fix: Gram form clamped at 0 (NaN when streams cancel), per-call partials buffer. | AKLR op; served through aiter provider after #271. |
| [#278](https://github.com/AMD-AGI/AKLR/pull/278) | fix(mhc): clamp the Gram form at 0 before its rsqrt (hidden 4096) | kyle-256 | MERGED 2026-11-20 (opened 11-19) | `mhc_fused_post_pre` hidden 4096 FlyDSL kernel (uuid 2827fc56 -> fc71f9d7), n_tokens 1–4096 | Correctness fix (NaN on cancelling streams: 1034/6400 tokens at T64, 7616/51200 at T512 before). Validate: 13 Points 1.14–1.51x. KB page lists 1.100–1.510x vs aiter, CI, date 2026-10-28. | AKLR op. |
| [#271](https://github.com/AMD-AGI/AKLR/pull/271) | feat(plugins): serve aiter's routed GEMM, MoE, MLA and mHC ops through the aiter provider, on aiter main and FlyDSL 0.5 | kyle-256 | MERGED 2026-11-20 (opened 11-18; #245 was the closed predecessor) | New selectors incl. `mhc_fused_post_pre` (+ `mhc_pre` in `aklr/plugins/aiter/provider/mhc.py`); declines aiter's preshuffled weight/residual layouts | No mHC-specific number. | Plumbing: AKLR plugs into aiter's `aiter.plugin` (ROCm/aiter#8075), so any framework calling aiter mHC can reach AKLR kernels. |
| [#289](https://github.com/AMD-AGI/AKLR/pull/289) | feat(plugins): DeepSeek-V4.1 layer ops at the monokernel's launches, for the decode steps it does not take | zhenhuang12 | CLOSED 2026-11-20 (opened 11-19), not merged | Op definitions `deepseekv41_mono_k1` (includes mHC boundary into attention + Sinkhorn) and `deepseekv41_mono_k2_attention` (includes mHC boundary into FFN); definitions/recording only, no kernel | None | sglang cookbook build only; #288 says "the monokernel is not planned for now". |
| [#288](https://github.com/AMD-AGI/AKLR/pull/288) | feat(plugins): DeepSeek-V4.1 sparse MLA, MXFP8 linear and A8W4 MoE on sglang's unfused path | zhenhuang12 | OPEN (opened 11-19) | Not mHC; states mHC boundary has no op on the V4.1 path | — | — |
| #286, #290, #296 | V4.1-Flash op definitions / vLLM routes (sparse MLA, MXFP8 linear, wo_a) | zhenhuang12 | OPEN (11-19) | No mHC files in their file lists | — | — |
| #270, #194, #253, #13, #32, #168, #251 | refactors / stack bumps / KB | various | mixed | Touch mHC schemas only incidentally (#253 withdrew the #200 kernel; #270 notes mhc will be checked at flashinfer tolerances) | — | — |

Issues: [#268](https://github.com/AMD-AGI/AKLR/issues/268) (OPEN, zufayu, 11-18, "Enabling AKLR today would split the
caller's graph on 24 of aiter's 40 routed ops") lists `mhc_fused_post_pre` in a table; #60 (RFC e2e accuracy) not mHC.
No issue specific to DSv4.1 mHC.

Main-tree mHC files (`git/trees/main`): `aklr/ops/mhc/{api,baseline,contract,reference}.py`;
`impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size4096/fc71f9d721054df9.json`;
`impls/flydsl/mhc_fused_post_pre_has_norm_weight_hidden_size7168/{138d8923d8bbd221 (definition), a2a5a2db565a78df}.json`;
`impls/flydsl/mhc_pre_hidden_size7168/ea2e03a44d502ad6.json` (definition only);
`impls/hip/mhc_fused_post_pre_has_norm_weight_hidden_size7168/d89d285ad73a24cc.json`;
`aklr/plugins/aiter/provider/mhc.py`; `harness/skills/kb/cases/mhc_fused_post_pre.md`. Nothing at hidden 5120, nothing
under `aklr/plugins/vllm` for mHC.

## 2. Passing hits elsewhere (one line each, titles only, not read in depth)

ROCm/aiter (mHC):
- #8055 MERGED 11-16 yhl-amd: [gfx1250] Mega-MHC three-way column split for 65..80 token tiles at H = 5120.
- #7854 MERGED 11-12 zufayu: [RFC #7240] Route the three remaining DSV4.1 ops that already have a guard.
- #7710 MERGED 11-10: mega_mhc dispatch raw FP32 fn to the ASM seam on gfx942. #7573 MERGED 11-06: gfx1250 Mega-MHC tiles.
- OPEN gfx950 perf: #8146 (two-kernel packed prefill, hidden 7168), #8151, #8171 (fused post+pre GEMM settings 65–1023 rows),
  #7181 (decode with Gram statistics), #7151, #7343 / #7114 (fn-resident mHC pre GEMM, prefill), #6030 (tile_m=16), #8155
  (mhc_post_reduce), #8238, #6957 (mHC pre cross-batch consistency on gfx950).

vllm-project/vllm (mHC / DSv4.1 ROCm):
- #68027 MERGED 11-16: [ROCm][Bugfix] Probe AITER mHC support when the op is built, not at import.
- #67853 MERGED 11-16: [ROCm][DSv4.1][Perf] Always use the unfolded layer-0 mHC entry on gfx942.
- OPEN: #65421 asm decode layers for 17–256-row decode steps (gfx950, TP4); #64207 GLM-5.3-Flash MonoKernel mHC Sinkhorn
  off critical path; #66140 mHC post for SM100; #65526 mHC broadcast refill.

## 3. Open questions (not answered here)

- Whether our vLLM DSv4.1 path calls aiter `mhc_pre` / `mhc_fused_post_pre` (and so could be reached by #271's provider)
  or vLLM's own mHC kernels — needs a code read of the vLLM image in use.
- #88's kernels at hidden 5120 have no vLLM route; porting them would need a vLLM seam.
