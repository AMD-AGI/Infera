# AMD (ROCm, MI355X) DCP in sglang, and GLM-5.2 (DSA) + DCP on AMD

All sglang claims are read from `origin/main` at `fda84f80b7` (2026-10-21) in `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp`. InferenceX claims are read from `origin/main` at `6046414d2` (2026-10-20). Benchmark numbers are copied verbatim from PR bodies or CI logs. I did not run anything on a GPU.

## 1. Short answer

- **GLM-5.2 + DCP on AMD is not merged.**
  - The merged DSA DCP work, #43735 (`8bf145d119`, 2026-10-13), only has a DCP path on the `trtllm` (FlashInfer) DSA backend. It is documented as "CUDA SM100/SM103".
  - On ROCm, `main` forces that backend when `--dcp-size > 1`. A reviewer reports that this crashes during startup on MI355X (details in §3).
- **The AMD port is open, not merged.** The PR chain is:
  - #44322: DCP on the triton sparse-MLA backend.
  - #45205: HiCache on top of #44322.
  - #42618: an older, wider alternative that has CHANGES_REQUESTED.
  
  These PRs carry GSM8K, passkey and InferenceX agentic-replay numbers for GLM-5.2-MXFP4 on MI355X with TP4/EP4 + DCP4 + MTP (§4).
- **There is no official InferenceX/AgentX result yet for sglang + DCP + GLM-5.2 on AMD.**
  - InferenceX PR #4064 adds the DCP4 points. It is OPEN and its results table is all "TBD".
  - Its only sweep failed: the image predates #43735/#44322, and the DCP4 c40 point died with `UnboundLocalError: local variable 'lse' referenced before assignment`.
- **AMD DCP that is merged on `main`:**
  - Kimi-K3 on the aiter backend (#34432, #34438, #34537, #40750), plus PD decode DCP over Mori (#37358).
  - Qwen3.5 on the triton backend (#25090), with an AMD nightly test.
  - I found no AMD DCP test or recipe for Kimi-Linear. All its DCP tests are registered on CUDA B200 only (§5).

## 2. Merged AMD DCP commits (on `origin/main`)

| Commit | PR | Date | What it enables (from PR body or code) |
|---|---|---|---|
| `833bce9df5` | #34432 [AMD][DCP 1/N] (billishyahao) | merged 2026-09-11 | Adds DCP to `aiter_backend.py`: `_forward_decode_dcp`, `_forward_verify_dcp` and `_plan_dcp_decode_metadata`. The aiter Gluon MLA kernel returns the LSE. The Kimi-K3 override accepts `aiter` decode under DCP. Example command: Kimi-K3 `--tp-size 8 --dcp-size 8 --prefill-attention-backend aiter --decode-attention-backend aiter --kv-cache-dtype fp8_e4m3`. |
| `6cc661f223` | #34438 [AMD][DCP 2/N] | merged 2026-10-04 | aiter asm PS fp8 prefill for Kimi-K3 (heads padded 12→16), plus asm chunked-KV prefill. Requires aiter commit `3dfe0b46f`. |
| `99c9d65b32` | #40750 (chien-an-chen) | merged 2026-10-01 | Opt-in asm persistent `mla_decode_fwd` for DCP decode, controlled by env `SGLANG_AITER_MLA_DCP_DECODE_BACKEND` (default `gluon` at the time). Body: "Kimi-K3 AgentX DCP8 on MI355, `BACKEND=asm`: decode ITL dropped from Gluon ~96 ms to ASM ~38 ms." |
| `4ab4d90d78` | #34537 [AMD][DCP 3/N] | merged 2026-10-08 | asm CP round-robin kernel for DCP target verify. Renames the env var to `SGLANG_AITER_MLA_DECODE_BACKEND`; the old name still works as an alias. Adds `SGLANG_AITER_MLA_VERIFY_BACKEND` and `SGLANG_AITER_MLA_AUTO_KV_SPLITS`. **Both backend vars now default to `asm`** (`python/sglang/srt/environ.py:1015-1019` on main; `git log -S` shows they were introduced by this commit). |
| `8ac19cc19f` | #39066 | 2026-09-21 | Kimi-K3 deferred KDA gate fix, plus a DCP cookbook update. |
| `9dc4c5d891` | #40907 | 2026-09-25 | Restores non-DCP Mamba checkpoint donation (HiCache agent-mode cache hit). |
| `5fab1f0f51` | #42018 | 2026-10-01 | Lets the aiter DCP ASM decode test run without aiter installed. |
| `50a144057c` | #37358 [AMD][PD] (inkcherry) | merged 2026-10-13 | Mori KV transfer for PD decode with DCP (token-granular relayout into DCP shards). |
| `b2c8f7a22e` / `308d89e042` | #25090 / #29409 | 2026-06-24 / 07-01 | Qwen3.5 triton-backend DCP on AMD, with its own nightly. |
| `2ded8a6aea`, `eb61cb2823`, `a362ba9da3` | #35611, #33480, #27928 | Jun–Aug | DeepSeek-V4 **prefill** context parallelism on AMD. This is prefill CP, not DCP. |

Verbatim numbers from these PRs:

- **#34438 accuracy** (Kimi-K3, 8xMI355X, tp8, fp8 KV, gsm8k 200): asm 0.990/0.985 vs `flash_attn_varlen_func` 0.985/0.985 at cap 131072. At ISL 650000 the TTFT is 1.535 s (cap 65536) vs 2.047 s (`MHA_ONE_SHOT`).
- **#34537 accuracy** (Kimi-K3, 8xMI355X, tp8, fp8 KV, DSPARK, gsm8k 200, 20-shot): `dcp_base` 0.990, `dcp_verify_asm` 0.985, `dcp_both_asm` 0.990, `tp_default` 0.990.
- **#34537 speed** (ISL 300k / OSL 2000, conc 4):
  - DCP8, gluon → asm + auto splits: "Output throughput 280.67 tok/s → **318.45 tok/s** (+13.5%)", "P90 TPOT 9.77 ms → **8.03 ms** (-17.8%)".
  - TP8: 331.26 → 335.80 tok/s.
- **#37358 validation:**
  - Kimi-K3 PP8/TP1 prefill + TP8/DCP8 decode: "GSM8K: 39/40 (97.5%)".
  - "32K input, 1 output token, concurrency 16, 128 requests: 16,699 input tok/s and 31.11s median TTFT."

## 3. GLM-5.2 / DSA + DCP: what `main` contains and the exact gaps

**Merged: #43735, `8bf145d119`** "[DCP] Decode context parallelism for DSA (GLM-5.2 / GLM-5.3) with MTP" (kpham-sgl, merged 2026-10-13T18:35Z).

The guard, `python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py:18-34` on main, has these checks:

- It raises `"DSA DCP does not support HiSparse or HiCache."` and `"DSA DCP does not support PD disaggregation."`.
- It requires `--speculative-eagle-topk 1`.
- It sets **unset** `dsa_prefill_backend`/`dsa_decode_backend` to `"trtllm"` on every platform; there is no `is_hip` branch.
- It defaults `dcp_replicate_q_proj=True` under a2a/fi_a2a.

Other constraints on main:

- `parallel_hook.py`: `--dcp-size > 1 does not support --enable-prefill-cp.`
- `deepseek_v2.py` overrides: DCP is rejected when `index_kpool > 1`.
- `dsa_backend.py`: the DCP code (`localize_dcp_page_table` and `dcp_trtllm_sparse_attention` at lines 3880 and 3926) exists only inside `_forward_trtllm` (defined at line 3747). No other DSA impl (triton, tilelang, aiter, flashmla) has a DCP branch. `dsa_dcp.py` calls `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`.
- The documentation (`docs/docs/advanced_features/dcp.mdx`, "DCP × DSA") says: "Requirements: CUDA SM100/SM103 with the `trtllm` DSA backends ..., `fp8_e4m3` or `bf16` KV cache, `attn_tp_size % dcp_size == 0` ... HiSparse, prefill CP, HiCache, PD disaggregation, unified memory and `index_kpool > 1` are not supported."
- **#43735 does not mention AMD anywhere in its body.** Its only test is CUDA: `test/registered/dcp/test_glm53_dcp2.py`, `register_cuda_ci(... runner_config="4-gpu-b200")`, GLM-5.3-NVFP4 TP4 + DCP2 + MTP, GSM8K threshold 0.88.
- `_check_dsa_backend_constraints` (`overrides.py:550`) has no HIP + DCP check on main.

**Accuracy reported in #43735** (NVIDIA only: GLM-5.3 FP8, 8xB300, TP8):

| Config | GSM8K | AIME26 pass@1 | gen tok/s | accept len |
|---|---|---|---|---|
| TP8 | 0.912 | 99.17% | 1140 | - |
| TP8 + DCP2 | 0.917 | 99.17% | 1041 | - |
| TP8 + DCP2 + MTP | 0.919 | 95.83% | 1876 | 3.51 |
| TP8 + DCP8 | 0.914 | 95.00% | 838 | - |
| TP8 + DCP8 + MTP | 0.919 | 95.00% | 1784 | 3.62 |

**What happens on ROCm with `main` today.** This is from a reviewer comment, not from my own run.

Jacob0226 on #44322 (2026-10-20, https://github.com/sgl-project/sglang/pull/44322):

> "`--dcp-size > 1` on ROCm resolves to a DSA backend that cannot run. `_dsa_dcp_overrides` ... sets unset DSA backends to `trtllm` on every platform ... On MI355X (GLM-5.2-MXFP4, TP4, `--dcp-size 4`, no backend flags), the server loads weights for ~7 min, then fails in decode graph capture: `'RotaryEmbedding' object has no attribute 'cos_sin_cache'` (`forward_mla_rocm.py:907`). `tilelang` fails with `not enough values to unpack` (line 913)."

If you force `triton` explicitly on main, #44322 reports (in its "PR before" column, meaning #43735 + 811248c80b) that the math prompt returns `"Comments"` and passkey output is "garbled". Its stated causes: the triton sparse-MLA kernel returns no LSE, and the HIP FP8 latent KV write ignores DCP sharding.

**The GLM-5.2 cookbook does not mention DCP.** `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx:134,140` gives the MI355X MXFP4 recipe as TP4, `--kv-cache-dtype fp8_e4m3`, triton DSA backends and MTP. It is "validated ... with `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`", with HiCache for TP4/EP4.

## 4. Open PRs: GLM-5.2 + DCP on AMD

All three are by EricKing626 and are OPEN.

### #44322 "[Feat] [AMD] dsa: Enable DCP on the triton sparse-MLA backend"

- Created 2026-10-14, updated 2026-10-21, +667/-65, head `68dbb49164`.
- Builds on the merged #43735.
- **What it adds:**
  - Base-2 LSE output from `triton_sparse_mla_decode_splitk` (both the fused and the split-K reduce kernels).
  - Triton decode, verify and paged extend routed through `localize_dcp_page_table`.
  - The ROCm fused rope + cache write is disabled under DCP.
  - The FP8 latent KV write is sharded (`slot % dcp_size == dcp_rank`, written at row `slot // dcp_size`).
  - On HIP, the MTP draft KV is sharded like the target. This fixes a c16 memory-access fault seen "about 10M cumulative prefill tokens in".
  - The owned-top-k scan is bounded, prefill indexer rows are split across ranks, and the prefix is gathered when that is cheaper.
- **Fixes made after review, now in the diff (commit 6404b6f15c):**
  - `_dsa_dcp_overrides`: `backend = "triton" if get_platform().is_hip else "trtllm"`.
  - `_check_dsa_backend_constraints` raises `"... DSA backend has no DCP path on ROCm; use --dsa-prefill-backend triton --dsa-decode-backend triton with --dcp-size > 1."`.
- **New env var:** `SGLANG_DSA_DCP_OWNED_PREFILL_RATIO = EnvFloat(3.0)`.
- **Review state:**
  - Jacob0226 submitted CHANGES_REQUESTED on 2026-10-20. The author replied on 2026-10-21 that both blockers are addressed; there is no re-review on record yet.
  - CI state in the body: Base run in progress, Extra ❌, AMD ❌ (runs 39555605256 / 39555605042 / 39555605251).
- **Correctness table** (MI355X, `amd/GLM-5.2-MXFP4`, TP4/EP4, FP8 KV, MTP EAGLE 5 steps / 6 draft tokens):

  | check | no DCP | PR before | PR after |
  |---|---|---|---|
  | math prompt (expect `29`) | 29 | `"Comments"` | 29 |
  | passkey 8K / 34K / 96K | 3/3 | garbled | 3/3 |
  | GSM8K (400 questions, few-shot) | 0.953 | — | 0.953 |
  | MTP accept length | 3.11 | — | 3.47 |
  | KV capacity (tokens) | 2.83M | — | 8.81M (2.20M rows per rank x 4) |

- **Performance** ("InferenceX agentic replay on this head, TP4/EP4 + DCP4 with MTP (simulated AL 3.61), 1 h per point". The TP4 baseline is commit 33dd8f6a2e and uses HiCache.):

  | Conc | Config | P90 Interactivity (tok/s/user) | TPUT per Chip (tok/s/GPU) | Chip hit | CPU hit |
  |---|---|---|---|---|---|
  | 16 | TP4 | 53.07 | 17,396 | 71.1% | 22.7% |
  | 16 | TP4 + DCP4 | 53.49 | 18,175 | 96.6% | — |
  | 24 | TP4 | 14.07 | 10,872 | 18.6% | 57.5% |
  | 24 | TP4 + DCP4 | 37.79 | 25,302 | 96.9% | — |
  | 40 | TP4 | 7.19 | 5,144 | 31.2% | 0.9% |
  | 40 | TP4 + DCP4 | 21.31 | 27,087 | 94.8% | — |
  | 48 | TP4 | 6.30 | 4,781 | 27.3% | 0.9% |
  | 48 | TP4 + DCP4 | 3.26 | 13,252 | 79.4% | — |

  The author's note on c48: "the DCP4 pool fills up (chip hit rate 79%) and DCP rejects HiCache, so evicted prefixes are recomputed ... P90 interactivity falls below it."

### #45205 "[Feat] [AMD] hicache: Enable HiCache for DSA DCP with a packed MTP draft"

- Created 2026-10-19. Stacked on #44322. Adds L1+L2 HiCache only; L3 is still rejected.
- New test `test/registered/dcp/test_glm52_dcp4_hicache_amd.py`, `register_amd_ci(est_time=1800, suite="nightly-amd-8-gpu-mi35x", nightly=True)`.
- The reviewer (Jacob0226, CHANGES_REQUESTED 2026-10-20) raised two issues: the branch sits on main rather than #44322, and the e2e test "is registered on a suite no workflow runs".
- **Correctness:** "GSM8K 0.94 (normal) / 0.935 (device pool forced to evict)". "KL ≤ 0.25: measured 0.10–0.18".
- **Performance** (`glm5.2-fp4-mi355x-sglang-agentic-mtp` agentic replay with InferenceX, MI355X TP4/EP4 + DCP4, AL 3.61, 3600 s, `--hicache-size 180`):

  | Conc | Config | P90 Interactivity | P90 E2E Norm. Interactivity | TPUT/GPU | P90 TTFT (s) | P90 E2E (s) | CPU hit |
  |---|---|---|---|---|---|---|---|
  | 16 | #44322 | 49.22 | 40.46 | 17,976 | 1.175 | 28.14 | — |
  | 16 | + HiCache | 49.88 | 41.93 | 18,198 | 1.054 | 28.45 | 0.01% |
  | 24 | #44322 | 36.34 | 30.80 | 25,029 | 1.155 | 46.75 | — |
  | 24 | + HiCache | 37.27 | 31.90 | 24,975 | 1.189 | 47.33 | 0.07% |
  | 40 | #44322 | 19.56 | 15.50 | 25,925 | 2.714 | 89.83 | — |
  | 40 | + HiCache | 25.55 | 22.27 | 29,339 | 1.600 | 72.01 | 2.69% |
  | 48 | #44322 | 6.05 | 4.75 | 18,169 | 9.936 | 181.77 | — |
  | 48 | + HiCache | 19.90 | 17.52 | 31,935 | 2.155 | 83.39 | 9.23% |

  The #44322 baseline rows here differ from the #44322 body (for example, c48 is 18,169 here vs 13,252 there). These are separate runs; the PRs do not reconcile them.

### #42618 "[ROCm] Support decode context parallel (DCP) for GLM-5 / DeepSeek-V3.2"

- Created 2026-10-05, last updated 2026-10-15, +1499/-58.
- A different design: the indexer K cache is sharded, and each rank computes a local top-k, then all-gathers and merges them. #43735 instead replicates the index-K cache.
- Jacob0226 CHANGES_REQUESTED (2026-10-13). His points:
  - The checks are not HIP-scoped and break the NPU DSA+DCP path (#37787).
  - The new Triton kernels are not exercised in CI.
  - Per-forward state is kept in a module global.
- Whether #42618 is superseded by #44322 is UNVERIFIED; neither PR says so.
- Accuracy (GLM-5.2-MXFP4, MI355X): TP4 + DCP4 + MTP GSM8K 0.938; with HiCache 0.942; passkey 9/9; "Accept length under DCP4 + MTP is 4.73 ... (4.74)".
- Agentic table, TP4 + DCP4:
  - c24: 35.2 tok/s/user, 24,243 tok/s/GPU
  - c40: 25.3 tok/s/user, 29,786 tok/s/GPU
  - c48: 20.2 tok/s/user, 33,835 tok/s/GPU
- KV pool: TP4 3,009,408 → TP4 + DCP4 10,891,520 tokens (3.6×).

### Other related open PRs (not AMD GLM)

- #39215: DSA DCP on SM120.
- #44341: env-gated gathered-prefix DSA extend under DCP.
- #39330 / #39638 / #39639: DSA DCP on Hopper.
- #34212: skip QuickAllReduce allocation for DCP groups (AMD).
- #41388: Kimi-K3 aiter fp8-Q asm MLA.
- #42439: DCP verify prefix on MLA ASM.
- #44660: KV sharding with aiter + MoRI.

## 5. Which models have AMD DCP today (on `main`)

- **Kimi-K3: yes.**
  - Override `kimi_k3.py:41-119`: decode backend under DCP "must be 'cutedsl_mla' or 'aiter'". `_require_kimi_k3_aiter_gluon_dcp_support()` raises if the aiter Gluon kernel is missing.
  - Cookbook `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx:98-112`: "MI350X/MI355X unified Balanced recipe uses TP8/DCP8 with AITER" on image `v0.5.19-rocm720-mi35x-20260916`. The recipe is marked "Final Verification In Progress" (config `verified: false`, `docs/src/snippets/configs/moonshotai/kimi-k3.jsx:1282-1340`).
- **Qwen3.5 (triton backend): yes.** `test/registered/dcp/test_qwen3p5_triton_dcp.py` is registered with `register_amd_ci(... suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35-triton-dcp", nightly=True)`, TP4/DCP4, GSM8K ≥ 0.90. It is wired in `.github/workflows/nightly-test-amd.yml:1864`.
- **Kimi-Linear: no AMD evidence.** Its DCP tests (`test_kimi_linear_dcp4.py`, `test_kimi_linear_dcp_dspark4.py`, `test_kimi_linear_pd_dcp4.py`, `test_kimi_linear_unified_memory_dcp_blackwell.py`) are all `register_cuda_ci(... b200)`.
- **GLM-5.2 / GLM-5.3 / DeepSeek-V3.2 (DSA): not on AMD** (see §3). GLM-5.3 DCP is CUDA-only, tested on B200.

## 6. How to run

**Kimi-K3 on MI355X, merged** (cookbook cell, `kimi-k3.jsx:1313-1340`):

```
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 \
sglang serve --model-path moonshotai/Kimi-K3 --trust-remote-code --tp-size 8 --dcp-size 8 \
  --dcp-comm-backend a2a --prefill-attention-backend aiter --decode-attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

Relevant env vars:

- `SGLANG_AITER_MLA_DECODE_BACKEND` and `SGLANG_AITER_MLA_VERIFY_BACKEND`: `asm` (default) or `gluon`. The old name `SGLANG_AITER_MLA_DCP_DECODE_BACKEND` still works as an alias.
- `SGLANG_AITER_MLA_AUTO_KV_SPLITS` (default 0).

**GLM-5.2-MXFP4 on MI355X: requires the unmerged #44322, plus #45205 for HiCache.** The flags are taken from #42618's command, #44322's scope note ("pass `--dsa-prefill-backend triton --dsa-decode-backend triton`"), the #45205 test, and the InferenceX recipe:

```
sglang serve --model-path amd/GLM-5.2-MXFP4 --trust-remote-code --tp 4 --ep 4 --dcp-size 4 \
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  [--enable-hierarchical-cache --hicache-size 180 --hicache-write-policy write_through   # only with #45205]
```

- Base env from the InferenceX recipe: `SGLANG_OPT_USE_TOPK_V2=true`.
- Optional (#44322): `SGLANG_DSA_DCP_OWNED_PREFILL_RATIO`.
- Constraints:
  - `--speculative-eagle-topk 1` only.
  - No PD disaggregation, HiSparse or prefill CP.
  - No HiCache without #45205.
  - No L3 storage.

## 7. InferenceX / AgentX status for sglang + DCP + GLM-5.2 on AMD

- On InferenceX `origin/main`, `glm5.2-fp4-mi355x-sglang-agentic-mtp` (`inferencex-e2e/configs/amd-master.yaml:1417-1431`, image `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`) has **no DCP arm**. It has only TP4/EP4 + HiCache (c4–16) and TP8/EP1 (c1–4).
- **InferenceX PR #4064**, "[AMD][AgentX] Add GLM-5.2 MXFP4 MI355X SGLang TP4/EP4 DCP4 points" (EricKing626, OPEN, created 2026-10-14):
  - Adds the arm `{ tp: 4, ep: 4, dcp-size: 4, kv-offloading: none, conc-list: [16, 24, 40, 48] }`.
  - Its body says it "Requires an SGLang ROCm image containing sgl-project/sglang#43735 plus ... #44322 ... The image in `amd-master.yaml` is unchanged and must be bumped before merge."
  - Its results table is all **TBD**.
  - Its unofficial sweep run 38612691278 (2026-10-14) **failed**. The DCP4 c40 job (116020377585) died in decode graph capture with `UnboundLocalError: local variable 'lse' referenced before assignment`, followed by "Server did not become healthy". The other points were cancelled.
- The only GLM-5.2 + DCP agentic numbers on AMD are therefore the replay tables in sglang PR bodies (§4). These were run by the PR author through InferenceX's harness; they are not official InferenceX results.
- **For comparison, ATOM (not sglang):**
  - A TP4+DCP4 arm for `glm5.2-fp4-mi355x-atom-agentic-mtp` was added per the InferenceX `perf-changelog.yaml:7949` entry (PR #3205).
  - The current `amd-master.yaml:1470-1490` search space has no DCP entry, although the comment above it still describes C16–C48 DCP4.
  - Open InferenceX PR #4308 says it will "Restore the TP4+DCP4 arm at C16-C48 with ATOM_DCP_PREFILL_QREP=1".

## 8. Open items (UNVERIFIED)

- Whether #44322 and #45205 now pass AMD CI and get maintainer approval. CI was red or in progress as of 2026-10-21, and re-review after the author's 2026-10-21 fixes is pending.
- Whether any published ROCm image contains #44322. #4064 says the current pinned image does not; no newer image was checked.
- Whether #42618 will be closed in favour of #44322.
- Non-MXFP4 GLM-5.2 (FP8) DCP on MI355X, and GLM-5.2 DCP on MI300X/MI325X: no PR or measurement found.

Scratch file: `/var/tmp/yihou/research_glm52_dcp/pr45205.diff`. One stray copy was also written to `/var/tmp/yihou_45205.diff`, which is outside the scratch directory; I left it in place because the rules forbid deleting outside `yihou/`.