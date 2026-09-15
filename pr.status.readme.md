# Which SGLang/AITER patches the GLM-5.2 decode measurements were built on

Scope: the build used by
`packups/glm52_tp8_ep8_dpa_c256_isl70k_osl10k_decode_profiling_yihou.packup_20260915-0130`
(and, same image, the TP8/EP8 sweep packup next to it).

Everything below was checked first-hand against `../rocm-llm-bench/Dockerfile`, the packup's
container probe, and the GitHub API (`gh api ... /compare`, `/pulls`) on 2026-09-15.

## 1. Pinned versions

| | |
|---|---|
| Base image | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` |
| Actual image | `sha256:b5aa5bd3d828…` (local `rocm-llm-bench:latest`, pinned by digest) |
| **SGLang** | **`402df1e1e453e1e85ec0f5ac4052d36598cc691a`**, authored **2026-09-06 09:23Z**, `Take the DSA rope+cache fusion only on decode-shaped forwards (#49)` — repo `xiaobochen-amd/sglang`, **not upstream** |
| **AITER** | **`2c71811b32c8ce2e1266aedaec199df7d90f597d`** — repo `xiaobochen-amd/aiter` |
| torch / triton / ROCm | `2.9.1+rocm7.2.0` / `3.7.0+amd.rocm7.2.0` / 7.2.0 |
| Extra stack | `flydsl 0.3.2`, `tilelang 0.1.7.post3` |

Divergence from upstream:

- sglang: merge-base `0111b2903` (2026-08-18, upstream `#34890`) — **ahead 116 / behind 1470**
- aiter: merge-base `0200ada1a` (2026-08-28, `ROCm/aiter`) — **ahead 16 / behind 187**

## 2. Which branch is the live one

**The GLM-5.2 optimisation branch is `dev_glm52_0907`, not `main`.**

- The pinned commit is a **direct ancestor of `dev_glm52_0907`** (ahead 5 / behind 0). Those 5
  commits are lint + test registration only — **no functional change**. The pin therefore *is*
  that branch's state.
- `main` is the upstream-tracking / upstreaming branch. It forked from `dev_glm52_0907` at the
  same upstream merge-base and has since synced 1359 upstream commits; the 116 ROCm commits were
  **re-landed there under new SHAs** (compare reports main `behind 121`). A newer `main` does not
  mean more optimisations — it is the same work on a newer upstream.
- `rocm-llm-bench` states this explicitly: *"Pin sglang to the fork's dev_glm52_0907"*.
- AITER: the pin **equals `xiaobochen-amd/aiter` main tip** (ahead 0 / behind 0). Nothing merged
  since.

## 3. Patches in the pinned build

Three layers.

### (a) Upstream `release/v0.5.18` cherry-picks (~8, 5-digit PR numbers)

Kimi-K3 tool calls, HiCache DCP, xgrammar segfault, W4AFP8 requant, … — **none on this decode
path**.

### (b) Fork PRs #1–#49 (~35 merged)

Correctness / enablement (not performance): #1 GLM-5.2 bias in fp32, #2 EAGLE verify silently
greedy on HIP, #12/#15 DSA split-op and head-gate on HIP, #14 stale custom-AR pointers across DP
graphs, #16/#23 Quark shared-experts and NextN arch name, #19 DSA k-only rope q/k aliasing,
#27 EAGLE top-k predicate read off host, #35 staged large pageable H2D copies, #49 restrict the
rope+cache fusion to decode-shaped forwards.

Performance:

| PR | What | Active in this run? |
|---|---|---|
| sglang #31 + #40 (needs aiter #11) | gfx950 **FlyDSL sparse MLA** prefill/decode; decode at 8 q heads padded to 16 | yes — `--dsa-decode-backend flydsl` |
| sglang #4/#7/#10/#21 (aiter #2/#3/#5/#8/#10) | **DSA top-k with fused page-table transform**; PAGED prefill via aiter instead of gather (2.86x op-level); made the default | yes — `--dsa-topk-backend aiter` |
| sglang #11 | drop constant work from the MLA absorb bmms | yes |
| sglang #13 | **AITER AllReduce fusion** | yes — `--enable-aiter-allreduce-fusion` |
| sglang #22 | capture the draft-extend CUDA graph for DSA | yes (graph-ON + EAGLE) |
| sglang #25 | stop rebuilding `q_all` for target-verify | yes |
| sglang #26 | fused DSA metadata kernels on HIP | yes |
| sglang #36 | unified Triton MoE router on ROCm | yes — EP8 + `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1` |
| sglang #37 | gfx950 four-kernel fused DSA indexer decode | present; interaction with the flydsl decode backend **not verified line-by-line** |
| sglang #38 | build the DSA fp8 q in one kernel instead of two | yes |
| sglang #43 | split-row top-p renorm, 4.35x at CONC=1 decode | yes (spec sampling) |
| sglang #47 | restore rope+KV fusion under flydsl, fold the q absorb into it | yes |
| sglang #18/#20/#28/#29/#34 | HiCache (fp8 JIT path, quota 16, kernel io backend, all-layer load) | **no — HiCache is off** |
| aiter #11 | gfx950 FP8 sparse MLA kernels | yes |
| aiter #13 | faster bf16 a16w16 skinny GEMM on gfx950 | yes |
| aiter #14 | `fp8_mqa_logits` BLOCK_M=4 + measured selection, 1.30x at production shapes | yes |

Cross-check against the packup's top-10 (kernel-name match, not a line-by-line source audit):
`main_kernel` 24.4% ← #31/#40 + aiter#11; `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` ←
aiter#14; `cross_device_reduce_2stage` ← #13; `hgemm_bf16_*` ← aiter#13;
`_fused_fp8_bmm_rope_cat_and_cache_mla` ← #38/#47. **6 of the top 10 kernels come from this set.**

### (c) This experiment's own patch

`evidence/code_snapshot/code.diff` — opt-in `--profile` plumbing in `bench/profile_decode.py` plus
a new `bench/profiling_yihou.py`. **Profiling only, no performance change**, with a pristine
differential run as evidence (`realized_accept_length` bit-identical, wall-clock within 0.06%).

## 4. Landed after the pin (none of it in these measurements)

| Where | PR | What | State |
|---|---|---|---|
| main | #54 | **AITER ASM MLA decode with EAGLE + CUDA graphs** | open |
| main | #55 | carry the real top_k into the EAGLE draft proposal | merged 09-11 (run used `--speculative-eagle-topk 1`, so likely a no-op here) |
| main | #57/#58/#59 | EAGLE: reject recycled / non-finite draft probabilities | merged, correctness |
| main | #56 | re-land of the fused DSA indexer (same work as #37) | merged |
| dev_glm52_0907 | #50 | gfx950 Triton sparse-MLA prefill + decode (alternative to `main_kernel`) | open |
| dev_glm52_0907 | #51 | GLM-5.2 NextN draft fused MoE → per-channel FP8, **9.3% ITV** | open |
| dev_glm52_0907 | #52 | slim the fused indexer's dead configuration space | open |
| aiter | #15/#16/#18/#19 | tuned BF16 GEMM configs for GLM-5.2 decode shapes; HGEMM pipeline depth 3; TP4/EP4 and TP8/EP1 MXFP4 fused-MoE tuning; decode GEMM update | open |
| aiter | #17/#20 | MegaMoE small-token tuning; pre-allocated MLA scratch for CUDA-graph-safe non-persistent decode | open |

The open AITER PRs land squarely on this experiment's top-10 (`hgemm_bf16_*`, `mfma_moe1/2`,
`main_kernel`) and are the most direct source of further gain.

## 5. Coverage of the upstreaming list

Against the tracked upstream PRs:

| upstream PR | Level / gain | What | In the pin? |
|---|---|---|---|
| **37134** | P0, long-generation correctness | ROCm EAGLE verify silently sampling greedy | ✅ fork #2 / `bb5be0c8a` |
| **38583** | P0, +10% ITV | gfx950 four-kernel fused DSA indexer decode | ✅ fork #37 / `22f52acd9` (re-landed on main as #56) |
| **38340** | P1, +1.5% ITV | fuse the MLA q absorb into RoPE + KV-write (gfx950) | ✅ `2942c768e` (fork #47); the pin's HEAD `#49` is its decode-shape restriction |
| **39155** | P0, +9.3% ITV | GLM-5.2 NextN draft fused MoE → per-channel FP8 | ❌ — this is fork #51, still open |
| **37152** | P0, HiCache enable +10% Tput | widen HiCache JIT copy rounds, enable the K-only host pool | ⚠️ partial — fork #18/#20 touch the same two files but at +26/-4 vs upstream's +70/-33; an earlier, smaller version |
| **35233** | P0, HiCache 1/2 | fix registered HiCache host pointer aliases (merged upstream 09-14) | ❌ — no fork commit touches `kvcacheio/transfer.cu` / `common_extension*.cc` (file-level judgement, medium confidence) |

**3 of 6 fully in, 1 partial, 2 absent.**

Two notes:

- Both HiCache items (35233 / 37152) are irrelevant to these numbers — HiCache was never enabled.
- The one performance item we could have had and did not is **39155 (9.3% ITV)**. It targets
  exactly the speculative-decode cost this packup measured (`eagle_draft` 6.0% +
  `eagle_draft_extend` 4.0% of wall). It is the single best candidate to merge and re-measure.

## 6. Discrepancies worth knowing

1. **The Dockerfile's patch-list comment is stale.** It names only sglang #31/#36/#37/#38/#40/#41
   and aiter #11/#13/#14, while the pinned SHA also carries #43, #47, #49 and the whole #1–#29
   wave. Read the commit range, not the comment.
2. **`AITER_COMMIT=d9e5ef7ce…` in the container env is not what is installed.** It is a leftover
   build arg from the base image; the Dockerfile removes `/sgl-workspace/aiter` and installs the
   fork at `2c71811b3`. Trust `git -C /aiter rev-parse HEAD`.
3. The packup's `environment.md` says the wrapper feature is at `23f472d2`, but
   `evidence/code_snapshot/git_head.txt` records `e38dadae`. Does not affect any SGLang-side
   conclusion, but the packup is internally inconsistent there.
