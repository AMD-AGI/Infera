# Completeness critic: GLM-5.2 + DCP in SGLang (NVIDIA and AMD), AgentX results, and InferenceX Dynamo + PP

Sources read on 2026-10-21: sglang `origin/main` at `fda84f80b7`, InferenceX `origin/main` at `6046414d2`, and `gh` (read-only). Scratch files are only under `/var/tmp/yihou/research_glm52_dcp/critic/`; nothing was deleted.

## (a) Gaps found in the five reports

1. **"No GLM-5.2 + DCP number exists for NVIDIA" is incomplete.** The open-PR searches only covered the AMD PRs (#44322, #45205, #42618) and the Hopper PR #39330. They did not cover #45499, #39215, #36990, #39768 or #44122. Four of those five (all but #44122) contain GLM-5.2 + DCP measurements, on GB300, RTX PRO 6000 and H-class GPUs.
2. **The PP + DCP combination was not covered.** sglang #39768 runs GLM-5.2 with PP8 prefill feeding DCP2 decode. This links the user's two questions and none of the reports found it.
3. **HiCache + DCP on NVIDIA for DSA models was not covered.** #44122 and #45499 are the NVIDIA counterparts of the AMD #45205.
4. **#4064's other "Run Sweep" runs were not checked.** One verifier saw run 38611271497 marked success and did not inspect it. That would matter if it were a passing DCP sweep.
5. **The current CI state of #44322 and its link to #42618 were open.**
6. **No search for an InferenceX PR adding NVIDIA GLM-5.2 + DCP.**
7. **No check for AgentX + DCP material on sglang main beyond the cookbook.**

## (b) Findings that close these gaps

### 1. Open sglang PRs with GLM-5.2 + DCP numbers on NVIDIA

All of these are OPEN with reviewDecision REVIEW_REQUIRED. Bodies are saved under `/var/tmp/yihou/research_glm52_dcp/critic/pr*.json`.

**sglang #39215**, "[Feat] attention: Enable DSA DCP on SM120 with MTP support"
- Author Yuening-wa, head `e22c8d8813`, updated 2026-10-21. CI slots in the body: Base, Extra and AMD all ❌.
- Accuracy (verbatim): "GLM-5.2 NVFP4, TP8/PP1, natural MTP5; GSM8K, 1,319 questions, 5-shot, temperature 0". No DCP 1,282/1,319 = 97.19%. DCP4 1,286/1,319 = 97.50%.
- **AgentX-style performance** (verbatim setup): "8× RTX PRO 6000, GLM-5.2 NVFP4, TP8/PP1, AgentX fast, MTP5 gold (controlled acceptance length 3.61), 3,600-second target profiling window, no offloading".

| C | Config | Throughput (total tok/s/GPU) | P90 TTFT (ms) | P90 TPOT (ms) | KV usage | Prefix hit (trace/server) |
|---|---|---|---|---|---|---|
| 4 | TP8 + MTP5 | 462.65 | 262,653.06 | 172.53 | 99% | 98.00% / 26.80% |
| 4 | TP8 + MTP5 + DCP4 | 1,981.70 | 6,565.19 | 28.93 | 73% | 97.46% / 97.32% |
| 8 | TP8 + MTP5 | — ("exceeded the KV capacity") | — | — | — | — |
| 8 | TP8 + MTP5 + DCP8 | 2,555.40 | 9,123.28 | 63.87 | 74% | 97.52% / 97.26% |

- The PR's own summary: "At C4, DCP4 improved throughput/GPU by **4.28×** and reduced P90 TPOT by **83.2%**". No InferenceX run ID is given.

**sglang #45499**, "[Feature] hicache: support DCP with EAGLE/MTP speculative decoding"
- Author nvpohanh, created 2026-10-21, head `44a18f776b`. CI slots: Base, Extra and AMD all ❌.
- Motivation (verbatim): "AgentX-style long-context agentic workloads on GLM-5.2 / DSA models benefit from DCP and HiCache at the same time. On 8xB200, TP8+DCP8 gives about 6x the logical KV capacity of TP8."
- What it changes: it lifts the main-branch rejection of HiCache under DSA DCP (`_dsa_dcp_overrides`), for EAGLE/NEXTN with topk 1. Only HiCache L2 is enabled. L3 and HiSparse stay rejected.
- GPU validation setup: "GLM-5.2-FP8, 1x GB300 node with 4 GPUs, TP4 + DCP4", `--kv-cache-dtype fp8_e4m3`, DSA backends resolved to `trtllm`, DCP comm `fi_a2a`, EAGLE 3/1/4.
- Capacity: "1,276,224 physical tokens/rank, so `max_total_num_tokens=5,104,896` logical".
- GSM8K, measured with "sgl-eval 0.1.3, 200 ex., GLM-5.2 preset, thinking on, seed 0":

| Config | GSM8K | Mean MTP accept len (device-hit / after load-back) |
|---|---|---|
| No HiCache | 97.5% | 3.07 / 3.18 |
| HiCache write_through | 98.5% | 3.08 / 3.03 |
| HiCache write_back | 97.5% | 3.01 / 3.14 |

- It contains **no throughput or AgentX numbers.**

**sglang #44122**, "[DCP] HiCache L2 for DSA models with MTP under decode context parallelism"
- Author kpham-sgl, head `3b1a17f0a9`, updated 2026-10-16.
- Validation is on **GLM-5.3** FP8 on 8xB300 with TP4 DCP2 + MTP: "HiCache, normal pool: … GSM8K 0.930 (0.925 with fp8 KV)".
- #45499's body says #44122 "hits `AssertionError: index page of 8448 bytes does not split over page_size=512 tokens`" at DCP8 for GLM-5.2. That claim comes from #45499's author; I did not reproduce it.

**sglang #36990**, "Feat[DCP] Support decode context parallelism for DSA models with the flashmla_kv backend"
- Author Zhiy-Zhang, last updated 2026-09-17. The body says the PR lacks the `run-ci` label.
- Scope (verbatim): "DCP support is available for DSA models such as GLM-5.2. On Hopper-architecture hardware, only the flashmla_kv backend is currently supported."
- Model: `GLM-5.2-W4AFP8`, `--tp 8`, `--attention-backend nsa --dsa-*-backend flashmla_kv`, fp8 KV.
- GSM8K: baseline TP8 0.948; "cp-dcp" (`--attn-cp-size 8 --enable-prefill-cp --dcp-size 8 --enable-cp-decode-attn-tp`) 0.951; "tp-dcp" (`--dcp-size 8`) 0.944.
- Speed is reported only as images, on the `Inferact/codex_swebenchpro_traces` workload. I did not read the images, so those numbers are **UNVERIFIED**.

### 2. GLM-5.2 with PP prefill and DCP decode: sglang #39768

- "[PD][GLM-5.x] Improve PP prefill to DCP decode KV transfer". Author hrpang0203, OPEN, depends on #36990, updated 2026-10-12.
- Stated target topology: "GLM-5.2 with PP8 prefill and single-node DCP2+DP4 decode over Mooncake".
- Commands in the body:
  - Decode: `--tp-size 8 --dp-size 4 --dcp-size 2 --enable-dp-attention … --dsa-decode-backend flashmla_kv`, on `GLM-5.2-W4AFP8`.
  - Prefill: `SGLANG_PP_LAYER_PARTITION=10,10,10,10,10,10,10,8`, `--tp-size 1 --pp-size 8`, on `GLM-5.2-FP8`.
- Numbers (verbatim), output tok/s TP8 → PP8:
  - LongBench v2, all 503 prompts: 378.00 → 1092.81.
  - 100K cold input, C32: 52.80 → 181.40.
  - GSM8K (1,314 prompts): 2263.50 → 2229.57.
- The PR says the cold-input rows come from separate runs with different limits. This is not an AgentX run, and no GPU type is stated in the body (**UNVERIFIED**; the flashmla_kv backend suggests Hopper, but that is not confirmed).

### 3. InferenceX #4064: no successful DCP sweep exists

- Runs on branch `amd/glm5.2-fp4-mi355x-sglang-enable-dcp`:
  - 38611271497 (success) and 38602017032 (success) ran only `check-changelog` and `reuse-sweep-gate`. Every benchmark job is `skipped` and there are no artifacts.
  - 38610743146 failed at `check-changelog`.
  - 38612691278 is the only run that benchmarked, and it failed. A verifier already confirmed that both its c40 and c48-eval DCP4 jobs hit the `lse` UnboundLocalError.
- The PR's last activity is 2026-10-14T19:03:30Z, a bot comment linking the visualizer `inferencex.semianalysis.com/inference?unofficialRun=38612691278`.
- **Conclusion: no InferenceX run, official or unofficial, has produced a SGLang + DCP + GLM-5.2 data point.**

### 4. #44322 status now, and its relation to #42618

- Reviews on #44322: Jacob0226 CHANGES_REQUESTED at 2026-10-20T03:31Z, then three EricKing626 COMMENTED entries at 2026-10-21T00:40–01:20Z. There is no re-review.
- CI on head `68dbb49164`:
  - AMD run 39555605251 is a completed failure. Failing jobs: `stage-b-test-1-gpu-small-amd-mi35x` shards 0 and 1, `stage-b-test-1-gpu-small-amd (mi300)`, and `stage-b-test-large-8-gpu-mi35x-disaggregation-amd`.
  - Extra run 39555605042 failed at `call-gate / pr-gate`.
  - Base run 39555605256 is still in progress.
  - Whether these failures are caused by the PR is **UNVERIFIED**; I did not read the logs.
- No comment on #44322 or #42618 says #42618 is superseded. #42618 is still OPEN.
- The links in #44322 and #45205 point only to sglang CI runs. **Their agentic replay tables have no InferenceX run ID**, so they cannot be reproduced from InferenceX records.

### 5. No InferenceX PR for NVIDIA GLM-5.2 + DCP

`gh search prs --repo SemiAnalysisAI/InferenceX dcp` returns, for GLM, only ATOM (MI355X) PRs (#2975, #3171, #3205, #3359, #4045, #4308) and SGLang #4064. Every other DCP PR is Kimi-K3. So the GLM-5.2 + DCP AgentX results from NVIDIA (#39215, on RTX PRO 6000) exist only in an sglang PR body.

### 6. No AgentX + DCP content on sglang main outside the cookbook

`git grep -il agentx origin/main` (excluding the agentx snippets) finds cookbook pages, `speculative_hook.py`, `environ.py` and two tests. Of these, only `DeepSeek-V4_1.mdx` and `environ.py` also mention dcp; none of them is about GLM.

### 7. Dynamo + PP: nothing new since the existing report

A search of InferenceX PRs updated on or after 2026-10-18 for PP turned up nothing beyond what the existing report already covers (#4300 merged; #4333, #4045, #3748 and #4322 open). The Dynamo + PP section needs no change.

## Updated bottom line

- **On sglang main**, GLM-5.2/5.3 + DCP is merged only for CUDA SM100/SM103 with the trtllm DSA backend (#43735). Its only measurements are on GLM-5.3, and the only CI test is GLM-5.3-NVFP4.
- **GLM-5.2 + DCP numbers exist only in open sglang PR bodies:**
  - AMD MI355X MXFP4: #44322, #45205, #42618 (InferenceX agentic replay, author-run).
  - NVIDIA RTX PRO 6000 NVFP4: #39215 ("AgentX fast" replay, C4 DCP4 1,981.70 vs 462.65 tok/s/GPU).
  - NVIDIA GB300 FP8: #45499 (GSM8K 97.5–98.5%; accuracy and acceptance length only, no throughput).
  - Hopper W4AFP8: #36990 (GSM8K 0.944–0.951) and #39768 (PP8 prefill to DCP2+DP4 decode).
- **No official or successful InferenceX AgentX result exists for SGLang + DCP + GLM-5.2.** The only GLM-5.2 DCP data in InferenceX CI comes from ATOM.