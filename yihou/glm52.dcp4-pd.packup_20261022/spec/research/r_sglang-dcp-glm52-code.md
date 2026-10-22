# GLM-5.2 / GLM-5.3 (DSA) + Decode Context Parallelism (DCP) in SGLang: what is supported, how to run it, and what has been measured

Sources: sglang `origin/main` as fetched; InferenceX `origin/main` = `6046414d2d` (2026-10-20); `gh`, read on 2026-10-21. Scratch files are only under `/var/tmp/yihou/research_glm52_dcp/`.

## 0. Bottom line

1. **NVIDIA (merged).** PR #43735 is merged into main as `8bf145d119` (2026-10-13). It supports DSA + DCP only on CUDA SM100/SM103 (B200/B300/GB-class), using the `trtllm` DSA backends. MTP (EAGLE chain, topk 1) works with it. HiCache, HiSparse, PD disaggregation, prefill CP and `index_kpool > 1` are rejected at launch.
   - The PR's own numbers are for **GLM-5.3 FP8 on 8×B300**. The CI test is **GLM-5.3-NVFP4** on 4×B200.
   - I found **no GLM-5.2 + DCP accuracy or perf number** for NVIDIA anywhere: not in the PR, not in tests, not in the cookbook or AgentX snippets.
2. **AMD (not merged).** ROCm GLM-5.2 + DCP lives only in open PRs:
   - **#44322**: triton sparse-MLA DCP, building on #43735. It carries InferenceX agentic-replay numbers on MI355X with GLM-5.2-MXFP4, TP4/EP4 + DCP4.
   - **#45205**: HiCache under DSA DCP on ROCm, stacked on #44322.
   - **#42618**: an older, different design (sharded index-K), currently at CHANGES_REQUESTED.
   - On current main, ROCm + `--dcp-size > 1` with no backend flags **crashes during decode graph capture**. A reviewer reproduced this on MI355X (§3.4).
3. **InferenceX (no official result).** No sglang GLM-5.2 + DCP AgentX result exists on InferenceX main.
   - PR #4064 (open) adds GLM-5.2 MXFP4 MI355X SGLang TP4/EP4/DCP4 points. Its unofficial sweep (run 38612691278) **failed** on the old image `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`. That image predates #43735 and #44322, and the PR body itself says the image must be bumped.
   - The only GLM-5.2 + DCP arm that ever landed on InferenceX main is **ATOM**, not sglang: added in #3205, removed in `1f00e7a01`. Open PR #4308 restores it.

---

## 1. Merged NVIDIA support: PR #43735 (`8bf145d119`)

- Title: "[DCP] Decode context parallelism for DSA (GLM-5.2 / GLM-5.3) with MTP". Author kpham-sgl, co-author vincentzed, merged 2026-10-13T18:35:36Z.
- It builds on #42536, which was closed unmerged (2026-10-13).
- `git log 8bf145d119..origin/main -i --grep='dcp\|context parallel'` finds only `0d130f1915` (Qwen3.8 prefill CP). So **#43735 has no merged DSA-DCP follow-up yet**.

### 1.1 Design, as stated in the PR body and in `docs/docs/advanced_features/dcp.mdx:175-196`

- **Target latent KV**: striped by position across the DCP ranks.
- **Index-K cache**: replicated on every rank. Each rank scores the whole sequence and takes the global top-k.
- **Attention**: one Triton kernel maps the top-k to rank-local rows (`transform_index_page_table_dcp`). The TRT-LLM sparse kernel returns the LSE, and the existing MLA DCP merge combines the partial outputs.
- **Shared phases**: decode, target verify and extend all use this same path.
- **MTP draft (nextn)**: keeps a replicated KV pool over the widened slots and attends without DCP. Code:
  - `deepseek_v2.py:2046-2047`: `self.dcp_size = 1 if (is_nextn and self.use_dsa) else get_parallel().attn_dcp_size`
  - `dsa_backend.py:537-541`: the draft worker uses `dcp_size, dcp_rank = 1, 0`.
- **Index-K memory cost** (doc): "about 6% at DCP2 and 17% at DCP4 on GLM-5.3".

### 1.2 Backends and hardware (code)

- **DCP attention path**: `dsa_backend.py:3879-3936`. When `self.dcp_size > 1` it calls `localize_dcp_page_table` and then `dcp_trtllm_sparse_attention` (`python/sglang/srt/layers/attention/dsa/dsa_dcp.py`). That function calls `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(..., return_lse=True)` with `backend="trtllm-gen"`.
- **Backend default**: `arg_groups/model_overrides/deepseek_v2.py:18-34`, `_dsa_dcp_overrides`, applied at line 100. It sets `dsa_prefill_backend` and `dsa_decode_backend` to `"trtllm"` when unset, **on every platform**: there is no `is_hip` / `is_sm100` check in this function.
- **Q projection default**: the same function sets `dcp_replicate_q_proj=True` when the comm backend is `a2a` or `fi_a2a`.
- **Hardware requirement**: stated only in the doc, `dcp.mdx:184`: "CUDA SM100/SM103 with the `trtllm` DSA backends (selected automatically), an `fp8_e4m3` or `bf16` KV cache, `attn_tp_size % dcp_size == 0`". I did not find a code-level guard for SM100.
- **Other DSA backends** (flashmla, tilelang, aiter, triton): the PR body says "other DSA backends need to return the LSE under DCP" and lists this under "Fast follows". I found no explicit runtime rejection of other backends on CUDA in `dsa_backend.py`.
- **Hopper**: not supported on main. Open PR #39330 (martin-batista, last updated 2026-10-04) does Hopper DSA DCP via `flashmla_kv`.
- **NPU**: separate path, #37787 (merged); `dsa_npu_indexer.py:220-231`.

### 1.3 Supported and rejected combinations on main

| Feature | Status | Source |
|---|---|---|
| MTP / EAGLE | Supported only as a chain: `--speculative-eagle-topk 1`, draft tokens = steps + 1 | `deepseek_v2.py:25-26`; `dcp.mdx:184` |
| HiSparse, HiCache | Rejected: "DSA DCP does not support HiSparse or HiCache." | `deepseek_v2.py:21-22`; generic `hicache_hook.py:111-150` |
| PD disaggregation | Rejected: "DSA DCP does not support PD disaggregation." | `deepseek_v2.py:23-24` |
| Prefill CP | Rejected: "--dcp-size > 1 does not support --enable-prefill-cp." | `parallel_hook.py` (in `handle_decode_context_parallelism`) |
| `index_kpool > 1` | Rejected | `deepseek_v2.py:164-169` |
| Unified memory | Rejected for non-MLA; the doc says it is unsupported for DSA | `kv_cache_hook.py:719-730`; `dcp.mdx:184` |
| LMCache | Rejected | `kv_cache_hook.py:457-460` |
| Prefill (breakable) CUDA graph | Disabled when `dcp_size > 1` | `cuda_graph_hook.py:212-215` |
| Decode CUDA graph | Not disabled by any rule I found. The CI test runs with default settings. | — |
| Radix cache | No DSA-DCP guard found; radix pages at `page_size * dcp_size` | `overrides.py:213-219`; actual testing **UNVERIFIED** |
| PP, DP attention | No DSA-DCP-specific guard found; no test exercises them | **UNVERIFIED** |
| Quantization | PR results use FP8 (`zai-org/GLM-5.3`); CI test uses NVFP4 (`--quantization modelopt_fp4`); KV cache must be fp8_e4m3 or bf16 | `dcp.mdx:184`; test file |

### 1.4 CLI flags (`arg_groups/fields/parallel.py`)

- `--dcp-size`, alias `--decode-context-parallel-size` (lines 52-58), default 1.
- `--dcp-comm-backend {ag_rs,a2a,fi_a2a}` (lines 127-139). When unset it resolves to `fi_a2a` where supported (Blackwell, single MNNVL domain), otherwise `a2a` on CUDA/ROCm/XPU.
  - `fi_a2a` on non-CUDA is rejected in `parallel_hook.py` (`handle_decode_context_parallelism`).
- `--dcp-replicate-q-proj` / `--no-dcp-replicate-q-proj` (lines 140-149).
- **Group layout**: DCP groups are contiguous slices of each TP group (`distributed/parallel_state.py:2711-2733`).
- **Page size**: on CUDA, DSA page size is set to `64 * index_kpool` (`deepseek_v2.py:170-174`).

Doc command (`dcp.mdx`, DSA section), quoted verbatim:
```
sglang serve --model-path zai-org/GLM-5.3 --trust-remote-code --tp-size 8 --dcp-size 2 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
```

### 1.5 Tests on main

- **`test/registered/dcp/test_glm53_dcp2.py`**
  - Registration: `register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")`. The `extra-b` stage is in the label-gated PR Test Extra workflow (`.github/workflows/pr-test-extra.yml:1-14,176`). It is not nightly.
  - Model and args: `RadixArk/GLM-5.3-NVFP4`, TP4, `--quantization modelopt_fp4 --dcp-size 2`, `mem_fraction_static=0.8`.
  - Fixture: `DsaMtpServerBase` (`python/sglang/test/server_fixtures/dsa_mtp_fixture.py`) with EAGLE steps 5, topk 1, 6 draft tokens.
  - Assertions: `gsm8k_accuracy_thres = 0.88` (class override). From `DsaMtpEvalConfigDefaults`: `gsm8k_accept_length_thres = 3.7`, 500 questions, 20 shots.
  - **No GLM-5.2 + DCP test exists on main.**
- **`test/registered/dcp/test_dsa_dcp_attention.py`**: kernel-level test; `register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")`.
- **CI results**: in the PR's last Extra run (38341425464), the `extra-b-test-4-gpu-b200` shards (0/1/2) concluded `success`. The job logs were not retrievable, so whether the GLM-5.3 DCP2 test itself ran and passed is **UNVERIFIED**.

### 1.6 Numbers reported in PR #43735 (verbatim)

Setup: "`zai-org/GLM-5.3` (FP8), 8xB300, `--tp 8 --mem-fraction-static 0.88 --max-running-requests 64`, branch at `28bb3d6074`." GSM8K uses `few_shot_gsm8k`, 8-shot, 1319 questions. AIME26 uses `sgl-eval run aime26 --n-repeats 4` with the GLM-5.2-FP8 preset.

| Config | GSM8K acc | AIME26 pass@1 (avg of 4) | AIME26 pass@4 | AIME26 gen tok/s | MTP accept len |
|---|---|---|---|---|---|
| TP8 | 0.912 | 99.17% | 100% | 1140 | - |
| TP8 + DCP2 | 0.917 | 99.17% | 100% | 1041 | - |
| TP8 + DCP2 + MTP | 0.919 | 95.83% | 100% | 1876 | 3.51 |
| TP8 + DCP8 | 0.914 | 95.00% | 100% | 838 | - |
| TP8 + DCP8 + MTP | 0.919 | 95.00% | 100% | 1784 | 3.62 |

- **Spot checks** (GSM8K, 300-400 questions): `ag_rs` DCP2 + MTP 0.907; with an 8192-token-per-rank pool, DCP2 0.903 and DCP2 + MTP 0.907; without head folding, DCP8 0.943 and DCP8 + MTP 0.913.
- **Open kernel issue** (PR comments): vincentzed reports a FlashInfer 0.7.0.post1 `return_lse` corruption with 64 heads and short KV at ≥1024 rows per call (for example "16,384 | 10 of 10 | 31,326,208 | inf"). kpham-sgl merged without head folding and says this "only DCP8 extend can hit". The PR says it "will be filed with FlashInfer". I found no fix on main (**UNVERIFIED** beyond the PR text).
- There is no AgentX or InferenceX perf result in #43735.

---

## 2. Cookbook / AgentX snippets on main

- `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx`, `GLM-5.3.mdx`, `docs/src/snippets/configs/zai-org/glm-5.2*.jsx` and `docs/src/snippets/agentx/zai-org/glm-5.2.jsx` contain **0 occurrences of "dcp"**.
- The AgentX GLM-5.2 snippet (generated 2026-10-20 from InferenceX main `6046414d2d`) lists hardware b200/b300/gb200/gb300/h200 only. It has no MI355X and no DCP.

---

## 3. AMD / ROCm: GLM-5.2 + DCP (all open PRs)

### 3.1 sglang #44322: "[Feat] [AMD] dsa: Enable DCP on the triton sparse-MLA backend"

- Author EricKing626. OPEN, head `68dbb49164`, 26 commits, updated 2026-10-21. Labels: amd, deepseek, run-ci, jit-kernel, memory-pool.
- **Scope**: builds on #43735 with the replicated index-K design. What it adds:
  - triton `triton_sparse_mla_decode_splitk(..., return_lse=True)`;
  - DCP-sharded HIP FP8 latent KV write;
  - the DSA MTP draft sharded like the target (gated on `is_hip()`);
  - DCP-split prefill indexer rows, and gathered prefill when beneficial.
- **Run instructions** (PR body): "On ROCm, pass `--dsa-prefill-backend triton --dsa-decode-backend triton`, because #43735 defaults to `trtllm` when they are unset."
  - Commit 6404b6f15c in this PR changes the default to triton on HIP and raises for other DSA backends.
- **Correctness**, verbatim. Setup: MI355X, `amd/GLM-5.2-MXFP4`, TP4/EP4, FP8 KV, MTP (EAGLE, 5 steps, 6 draft tokens).

| check | no DCP | PR before | PR after |
|---|---|---|---|
| math prompt (expect `29`) | 29 | `"Comments"` | 29 |
| passkey 8K / 34K / 96K | 3/3 | garbled | 3/3 |
| GSM8K (400 questions, few-shot) | 0.953 | — | 0.953 |
| MTP accept length (4 prompts x 400 tokens, greedy) | 3.11 | — | 3.47 |
| KV capacity (tokens) | 2.83M | — | 8.81M (2.20M rows per rank x 4) |

- **Performance**, verbatim. Labelled "InferenceX agentic replay on this head, TP4/EP4 + DCP4 with MTP (simulated AL 3.61), 1 h per point"; "Table baseline: 10/19/2026 commit: 33dd8f6a2e". The PR says the TP4 rows "without DCP but with HiCache are for reference only".

| Conc | Config | P90 Interactivity (tok/s/user) | TPUT per Chip (tok/s/GPU) | Chip Cache Hit | CPU Cache Hit | Theoretical Cache Hit |
|---|---|---|---|---|---|---|
| 16 | TP4 | 53.07 | 17,396 | 71.1% | 22.7% | 97.0% |
| 16 | TP4 + DCP4 | 53.49 | 18,175 | 96.6% | — | 97.0% |
| 24 | TP4 | 14.07 | 10,872 | 18.6% | 57.5% | 96.2% |
| 24 | TP4 + DCP4 | 37.79 | 25,302 | 96.9% | — | 97.3% |
| 40 | TP4 | 7.19 | 5,144 | 31.2% | 0.9% | — |
| 40 | TP4 + DCP4 | 21.31 | 27,087 | 94.8% | — | 97.1% |
| 48 | TP4 | 6.30 | 4,781 | 27.3% | 0.9% | — |
| 48 | TP4 + DCP4 | 3.26 | 13,252 | 79.4% | — | 96.5% |

- The PR states that at c48 the DCP4 pool fills up and, because DCP rejects HiCache, evicted prefixes are recomputed.
- **Review history**:
  - Jacob0226 requested changes on 2026-10-20 (backend resolution crash, missing registered kernel tests).
  - The author replied on 2026-10-21 that both are fixed in 6404b6f15c and d72a59643f, plus a bug fix in 528444fe38.
  - No re-approval is visible yet. The PR body's CI slots show Extra ❌ and AMD ❌/⌛.

### 3.2 sglang #45205: "[Feat] [AMD] hicache: Enable HiCache for DSA DCP with a packed MTP draft"

- Author EricKing626. OPEN, head `3db5d71761`, stacked on #44322. Status: CHANGES_REQUESTED by Jacob0226 ("This branch sits on main, not on #44322 … the e2e test is registered on a suite no workflow runs").
- **New test**: `test/registered/dcp/test_glm52_dcp4_hicache_amd.py`, `register_amd_ci(est_time=1800, suite="nightly-amd-8-gpu-mi35x", nightly=True)`.
  - Server args: `amd/GLM-5.2-MXFP4`, TP4, `--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --dcp-size 4`, plus HiCache.
  - Thresholds: `gsm8k_threshold = 0.9`, `kl_threshold = 0.25`.
- **Reported correctness**: GSM8K "0.94 (normal) / 0.935 (device pool forced to evict)"; KL "measured 0.10–0.18".
- **Performance**, verbatim. Setup: InferenceX agentic replay, MI355X TP4/EP4 + DCP4, golden simulated AL 3.61, 3600 s window, `--hicache-size 180`; the baseline is #44322 without HiCache.

| Conc | Config | P90 Interactivity | P90 E2E Norm. Interactivity | TPUT per Chip (tok/s/GPU) | P90 TTFT (s) | P90 E2E (s) | CPU Cache Hit |
|---|---|---|---|---|---|---|---|
| 16 | #44322 | 49.22 | 40.46 | 17,976 | 1.175 | 28.14 | — |
| 16 | +HiCache | 49.88 | 41.93 | 18,198 | 1.054 | 28.45 | 0.01% |
| 24 | #44322 | 36.34 | 30.80 | 25,029 | 1.155 | 46.75 | — |
| 24 | +HiCache | 37.27 | 31.90 | 24,975 | 1.189 | 47.33 | 0.07% |
| 40 | #44322 | 19.56 | 15.50 | 25,925 | 2.714 | 89.83 | — |
| 40 | +HiCache | 25.55 | 22.27 | 29,339 | 1.600 | 72.01 | 2.69% |
| 48 | #44322 | 6.05 | 4.75 | 18,169 | 9.936 | 181.77 | — |
| 48 | +HiCache | 19.90 | 17.52 | 31,935 | 2.155 | 83.39 | 9.23% |

The #44322 rows here (for example c40 19.56 / 25,925) differ from the #44322 PR table (21.31 / 27,087). They are separate runs, and neither PR explains the gap.

### 3.3 sglang #42618: "[ROCm] Support decode context parallel (DCP) for GLM-5 / DeepSeek-V3.2"

- Author EricKing626. OPEN, head branch `amd/dcp-glm-dsa`, last commit 2026-10-14. Jacob0226 requested changes on 2026-10-13.
- **Design differs from #43735/#44322**: index-K is sharded, with a cross-rank top-k merge. MTP IndexShare is off under DCP.
- **Reported results**, GLM-5.2-MXFP4 on MI355X:
  - GSM8K: TP4 0.935/0.930; TP4+DCP4 0.939; TP4+DCP4+MTP 0.938; with HiCache 0.942.
  - KV pool: TP4 3,009,408 vs TP4+DCP4 10,891,520 tokens.
  - Agentic table: c24 TP4+DCP4 35.2 tok/s/user, 24,243 tok/s/GPU; c48 20.2 tok/s/user, 33,835 tok/s/GPU.
- Its relation to #44322 (whether it is superseded) is not stated in either PR (**UNVERIFIED**).

### 3.4 Current-main behaviour on ROCm (reviewer comment on #44322, 2026-10-20)

Quoted verbatim:

> "On MI355X (GLM-5.2-MXFP4, TP4, `--dcp-size 4`, no backend flags), the server loads weights for ~7 min, then fails in decode graph capture: `'RotaryEmbedding' object has no attribute 'cos_sin_cache'` (`forward_mla_rocm.py:907`). `tilelang` fails with `not enough values to unpack` (line 913)."

---

## 4. InferenceX side (`/home/yihou/dev/git.16-19/external.InferenceX`, `origin/main` `6046414d2d`)

- **Configs on main**: no config entry with `dcp-size > 1` exists for GLM-5.2 in `configs/amd-master.yaml` or `configs/nvidia-master.yaml`. Every DCP>1 entry is Kimi-K3 (vLLM, ATOM, Dynamo-vLLM).
  - The current `glm5.2-fp4-mi355x-sglang-agentic-mtp` uses image `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`, with a TP4/EP4 + HiCache arm and a TP8 arm.
- **InferenceX PR #4064** (OPEN, EricKing626, head `a600ec97d6`): "[AMD][AgentX] Add GLM-5.2 MXFP4 MI355X SGLang TP4/EP4 DCP4 points".
  - Adds `override_tp4_dcp4_c16/c24/c40/c48` with `decode-context-parallel-size: 4`, `KV_OFFLOADING: none`.
  - Search-space entry: `{ tp: 4, ep: 4, dcp-size: 4, kv-offloading: none, conc-list: [16, 24, 40, 48], ... }`.
  - Body: "Requires an SGLang ROCm image containing sgl-project/sglang#43735 plus … #44322 … The image in `amd-master.yaml` is unchanged and must be bumped before merge." Its Results table is all "TBD".
  - The unofficial sweep (run 38612691278, 2026-10-14) concluded `failure`. The TP4/DCP4/EP4 c40 job failed; c48 eval-only failed with `UnboundLocalError: local variable 'lse' referenced before assignment` at `forward_mla_rocm.py:969` during decode CUDA-graph capture. Other jobs were cancelled. The log is saved at `/var/tmp/yihou/research_glm52_dcp/run38612691278_failed.log`.
- **ATOM GLM-5.2 FP4 MI355X TP4+DCP4 (not sglang)**:
  - Added by #3205 (`a260318ab`, 2026-09-18): "Add a TP4+DCP4 arm (dcp-size 4) at conc [16, 24, 32, 40, 48]".
  - Removed in `1f00e7a01` (2026-09-26), which dropped "the ATOM LMCache DRAM-offload arms of … GLM-5.2 on MI355X".
  - The comment at `amd-master.yaml` (~line 1466) still mentions DCP4, but the search-space has only TP4 and TP8.
  - Open PR #4308 says "Restore the TP4+DCP4 arm at C16-C48 with ATOM_DCP_PREFILL_QREP=1".
  - I found no AgentX score for the ATOM DCP arm in the PR bodies I read (**UNVERIFIED** whether any official run ingested it).

---

## 5. Not found or UNVERIFIED

- Any GLM-5.2 + DCP accuracy or perf on NVIDIA. The only NVIDIA numbers are for GLM-5.3 (§1.6).
- Any official InferenceX AgentX leaderboard entry for sglang + DCP + GLM-5.2. The AMD agentic numbers above are self-reported in sglang PR bodies (#44322, #45205, #42618), labelled "InferenceX agentic replay".
- Whether DSA DCP works with PP, DP attention or radix cache: there are no guards and no tests.
- Whether the FlashInfer 64-head `return_lse` issue from #43735 has been fixed.

The "nvidia dynamo + pp" part of the original request was outside this task's scope and is not covered here.