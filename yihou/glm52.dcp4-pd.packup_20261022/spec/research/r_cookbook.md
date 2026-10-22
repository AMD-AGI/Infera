# SGLang cookbook (origin/main, `fda84f80b7`): what it says about DCP, PP and AgentX for GLM-5.2, GLM-5.3, GLM-5.3-Flash and Kimi-K3

## Summary

- **GLM-5.2 and GLM-5.3 cookbook pages have no DCP deployment at all.** `git grep -i 'dcp|decode-context|dcp-size'` finds nothing in `GLM-5.2.mdx`, `GLM-5.3.mdx`, `GLM-5.3-Flash.mdx`, `configs/zai-org/glm-5.{2,3}.jsx`, `glm-5.3-flash.jsx`, any `*-benchmarks.jsx`, or `agentx/zai-org/glm-5.{2,3}.jsx`. `git log origin/main -S'dcp'` and `-S'DCP'` over these paths return only the two GLM-5.3-Flash commits (d1f14431fd, which added DCP4, and b98a2d1096, which removed it). So DCP was never in the GLM-5.2 or GLM-5.3 cookbook.
- **GLM-5.2/5.3 + DCP is documented outside the cookbook**, in `docs/docs/advanced_features/dcp.mdx:175-196` ("DCP × DSA (GLM-5.2, GLM-5.3)"), added by 8bf145d119 / PR #43735 (merged 2026-10-13). It is CUDA SM100/SM103 only, with the `trtllm` DSA backends. It is not AMD.
- **The only AMD + DCP recipe in the cookbook is Kimi-K3** on MI350X/MI355X (TP8/DCP8, ROCm/AITER, Unified Balanced). It is marked `verified: false` / `verificationStatus: "in-progress"`, and the page says its published speed numbers do not validate the DCP8 recipe.
- **The GLM-5.2 AgentX section has no AMD hardware and no DCP.** It has 31 deployments on GB300/GB200/B300/B200/H200. It has PP in one place: GB300 Dynamo disagg with prefill `--pipeline-parallel-size 4` (InferenceX PR #4300, now merged). The H200 disagg cells use prefill CP (`--attn-cp-size 8 --cp-strategy interleave`), which is not DCP.
- **The GLM-5.3 AgentX section** (added by #45513) has one deployment: B300 NVFP4 TP8, InferenceX #3930. No DCP, no PP, no AMD.
- **The AgentX snippets contain no throughput, interactivity or score numbers.** They hold launch commands, concurrency lists, run links and a "golden AL" per point (the synthetic speculative acceptance length InferenceX used).
- **GLM-5.3-Flash had a DCP4 option, removed by b98a2d1096 / PR #40036.** Reason, per the PR: "until it is re-measured on the current TP4/EP1 + flashinfer_trtllm recipe".

---

## 1. Pages and snippets found

`git grep -il 'glm-5\|glm5\|kimi-k3' origin/main -- docs/` finds these relevant files:
- `docs/cookbook/autoregressive/GLM/{GLM-5.2,GLM-5.3,GLM-5.3-Flash}.mdx`
- `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`
- `docs/src/snippets/configs/zai-org/{glm-5.2,glm-5.2-benchmarks,glm-5.3,glm-5.3-benchmarks,glm-5.3-flash,glm-5.3-flash-benchmarks}.jsx`
- `docs/src/snippets/configs/moonshotai/kimi-k3.jsx` (benchmarks: `kimi-k3-benchmarks.jsx`)
- `docs/src/snippets/agentx/zai-org/{glm-5.2,glm-5.3}.jsx`
- `docs/docs/advanced_features/dcp.mdx` (not cookbook, but it holds the GLM DCP content)

Kimi-K3.mdx has no AgentX section: `grep -i agentx` returns nothing.

## 2. GLM-5.2 cookbook

### 2.1 Deployment matrix: no DCP; AMD cells are TP-only

`GLM-5.2.mdx:3` lists hardware as "H200, B200, B300, GB300, and AMD MI300X/MI325X/MI355X".

AMD guidance on the page:
- `GLM-5.2.mdx:133`: "FP8 (`zai-org/GLM-5.2-FP8`) runs single-node at `tp=8` on all three … Use the DSA tilelang backend (`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`) and add `--chunked-prefill-size 131072` plus `--watchdog-timeout 1200`".
- `GLM-5.2.mdx:134`: MXFP4 (`amd/GLM-5.2-MXFP4`) at `tp=4` with `--kv-cache-dtype fp8_e4m3` and the triton DSA backends. The page says it "has not yet been benchmarked on GLM-5.2, so the Deploy panel marks it unverified."
- `GLM-5.2.mdx:140`: "Five-step MTP is validated for `amd/GLM-5.2-MXFP4` on MI355X with `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`. Choose **Low-Latency** for TP8/EP1, or **High-Throughput** for TP4/EP4 with HiCache (`--enable-hierarchical-cache --hicache-ratio 1.0`)."
  - The config pins a different image: `cfg glm-5.2.jsx:100` has `"mi355x|mxfp4": "lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260923"`. The page and config disagree on the image date.

MI355X FP8 cells, all `verified: true`:
- Low-latency, `glm-5.2.jsx:894-907`: `--tp 8 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --chunked-prefill-size 131072 --mem-fraction-static 0.80 --watchdog-timeout 1200`
- Balanced, `:910-925`: `--tp 8 … --chunked-prefill-size 32768 --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 128 --max-running-requests 80 --watchdog-timeout 1200`
- High-throughput, `:928-942`: `--tp 8 … --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 --max-running-requests 256 --watchdog-timeout 1200`

MI355X MXFP4 cells, all `verified: false`:
- Low-latency, `:1007-1027`: env `SGLANG_OPT_USE_TOPK_V2=1`; `--trust-remote-code --tp 8 --ep-size 1 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --chunked-prefill-size 131072 --mem-fraction-static 0.80 --watchdog-timeout 1200`
- High-throughput, `:1050-1073`: env `SGLANG_OPT_USE_TOPK_V2=1`; `--tp 4 --ep-size 4 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --enable-hierarchical-cache --hicache-ratio 1.0 --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 --max-running-requests 256 --watchdog-timeout 1200`
- "mtp-314", `:1075-1105`. Its comment says "validated AgentX recipe". Flags: `--trust-remote-code --tp 4 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --chunked-prefill-size 131072 --mem-fraction-static 0.80 --cuda-graph-max-bs-decode 160 --max-running-requests 160 --watchdog-timeout 1800`.
  - The option label at `:192-196` is "EAGLE / MTP 3-1-4 (agentic · MI355X MXFP4)". Its enableReason reads: "Validated on MI355X gfx950 with amd/GLM-5.2-MXFP4 (InferenceX AgentX sweep, GSM8K em_strict 0.971) … Pair with SGLANG_SIMULATE_ACC_LEN=2.99 for benchmarking".
  - It came from b5d1453ed2 (#34379, "[AMD] GLM 5.2 MXFP4 SGLANG COOKBOOK") and 288627e400 (#39230).
  - This is a Deploy-panel option. It is not part of the AgentX section.

### 2.2 GLM-5.2 AMD benchmark numbers in the cookbook (FP8 only, no MTP, no DCP)

Source: `glm-5.2-benchmarks.jsx:197-227`. The comment there says "Measured on lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618 … No spec-decoding". All rows are random, ISL 8192 / OSL 1024.

| MI355X FP8 cell | concurrency | TTFT (ms) | TPOT (ms) | tok/s/GPU |
|---|---|---|---|---|
| low-latency | 1 | 634 | 13.56 | 81 |
| low-latency | 16 | 5411 | 23.60 | 621 |
| balanced | 64 | 19526 | 46.50 | 1098 |
| balanced | 256 | 117866 | 56.12 | 1044 |
| high-throughput | 1024 | 432058 | 106.44 | 1269 |

There are no MXFP4 benchmark rows. The only accuracy figure is the "GSM8K em_strict 0.971" string at `glm-5.2.jsx:196`.

### 2.3 Context-parallel content on the page is prefill CP, not DCP

- `GLM-5.2.mdx:268-287` ("3.5 Context Parallelism") covers prefill CP: `--attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave`, plus LayerSplit `--enable-dsa-cache-layer-split` on the PD prefill node.
- `GLM-5.2.mdx:271` says zigzag CP is "temporarily unavailable for GLM-5.2".

### 2.4 GLM-5.2 AgentX section

The page text is at `GLM-5.2.mdx:70-81`. The data is in `agentx/zai-org/glm-5.2.jsx`.

Header, `glm-5.2.jsx:1-3`: "Launch commands mirrored from SemiAnalysis InferenceX AgentX submissions (SemiAnalysisAI/InferenceX main@6046414d2d1a plus open PRs #3930, #4333, #4341, #4342, #4343, #4344, #4346, #4350)". `generatedOn` is 2026-10-20.

Coverage:
- Hardware: gb300, gb200, b300, b200, h200. **No AMD.**
- Checkpoints: `nvidia/GLM-5.2-NVFP4`, `zai-org/GLM-5.2-FP8`.
- 31 deployments. This matches PR #45513's table: "GLM-5.2 | 16 lanes | 31 deployments | 71 points".

DCP: none in any cell (`grep -i dcp` returns nothing).

PP and CP cells:

| Cell id | Shape | InferenceX PR / run | Concurrency |
|---|---|---|---|
| `glm5.2-fp4-gb300-dynamo-sglang-agentic-disagg#0,#1,#2,#3,#5` | "Disagg 1P5D / 1P3D / 1P1D / 3P1D / 4P1D · prefill PP4 · decode TP4 / DEP8 / DEP16" | #4300, verified, run 39187209967/attempts/2 | [30], [40], [56], [216,264], [256] |
| `glm5.2-fp8-h200-dynamo-sglang-agentic-mtp-1p1d-hicache#0` | "Disagg 1P1D · prefill TP8 · decode TP8·DPA8"; prefill uses `--attn-cp-size 8 --cp-strategy interleave` | #3021, run 34643185284/attempts/1 | [8,12,16] |
| `glm5.2-fp8-h200-dynamo-sglang-agentic-mtp-2p2d#0` | "Disagg 2P2D · prefill TP8 · decode TP8·DPA8" (recipe name `…-pcp8-tp8-dp8-mtp.yaml`) | #2684 | [8,12,16] |

- PR #4300 recipe: `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/gb300-fp4/agentx/disagg-variants.yaml`; image `lmsysorg/sglang:nightly-dev-20261016-a93f16f0`.
- gh: SemiAnalysisAI/InferenceX#4300 is "[NVIDIA][AgentX] GB300 GLM-5.2 NVFP4 Dynamo+SGLang: PP4-prefill disagg and TP4/TP8 agg points", MERGED 2026-10-20T08:20:54Z.

GB300 PP4 prefill worker command (disagg#0, golden AL 3.61, kv=hicache), as mirrored in the cookbook. The `--pipeline-parallel-size 4` line is at `glm-5.2.jsx:685`:
```
python3 -m dynamo.sglang --model-path nvidia/GLM-5.2-NVFP4 --served-model-name nvidia/GLM-5.2-NVFP4 --host 0.0.0.0 --port 30000 --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 --request-plane tcp --active-decode-blocks-threshold None --active-prefill-tokens-threshold None --active-prefill-tokens-threshold-frac None --chunked-prefill-size 16384 --context-length 1048576 --disable-cuda-graph --disaggregation-transfer-backend nixl --enable-hierarchical-cache --fp4-gemm-backend flashinfer_cutedsl --hicache-io-backend direct --hicache-size 100 --hicache-write-policy write_back --max-prefill-tokens 16384 --max-running-requests 16 --mem-fraction-static 0.83 --model-loader-extra-config {"enable_multithread_load":true} --moe-runner-backend flashinfer_cutedsl --pipeline-parallel-size 4 --quantization modelopt_fp4 --router-mode kv --router-replica-sync --router-session-affinity-ttl-secs 3600 --speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-moe-runner-backend flashinfer_trtllm --speculative-num-draft-tokens 6 --speculative-num-steps 5 --tensor-parallel-size 1 --trust-remote-code --watchdog-timeout 1000000 --weight-loader-prefetch-checkpoints
```
Decode (5 workers): same model and transfer flags, with `--disaggregation-mode decode … --cuda-graph-max-bs-decode 24 --max-running-requests 24 --mem-fraction-static 0.9 … --tensor-parallel-size 4 --speculative-num-steps 5 --speculative-num-draft-tokens 6`. Prefill env includes `MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True`.

Other cells are TP / DEP / DPA aggregated or disaggregated: B300 PRs #3836, #3890, #3953; B200 PRs #2808, #2863, #3400, #4286; GB200 PR #3401; H200 PR #3112.

No numbers: the data schema has no throughput, latency or score fields. `_agentx.jsx:583-584` renders only the per-point golden AL: "SemiAnalysis measured this point with a synthetic speculative-decoding acceptance length of {goldenAL} (InferenceX golden AL via `SGLANG_SIMULATE_ACC_LEN`)". Actual results are only at the linked InferenceX runs (e.g. https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39187209967). I did not read those runs; their numbers are UNVERIFIED here.

### 2.5 Commit 721953c6f6 (#44065, 2026-10-14)

"[Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main (GLM-5.2 NVFP4 B300 agg TP4, Dynamo frontend counts)". It touches `agentx/zai-org/glm-5.2.jsx` (+813/−…) and the Qwen/DeepSeek agentx files. Its effect is the B300 "Agg TP4" lane, PR #3953 in the current data. No DCP in that file at that commit or now, per the `-S` search above.

## 3. GLM-5.3 cookbook

- No DCP and no PP anywhere in `GLM-5.3.mdx`, `glm-5.3.jsx` or `glm-5.3-benchmarks.jsx`.
- AMD cells exist (TP only): MI355X FP8/BF16, MI325X, MI300X at `glm-5.3.jsx:821-1059`, with images `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260920` and `…-mi30x-20260920` (`:101-103`).

AgentX section, added by 1689cb2d52 / PR #45513 (2026-10-21; `GLM-5.3.mdx` +13 lines):
- Hardware: B300 only. Checkpoint: `nvidia/GLM-5.3-NVFP4`.
- One cell: `glm5.3-fp4-b300-sglang-agentic-mtp#0`, "Agg TP8", "MTP 3-1-4", 8 GPUs, submitted via the SGLang router.
- Image: `lmsysorg/sglang:v0.5.20-cu130`.
- Recipe: `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/glm5.3/sglang/b300-fp4-mtp/agentic.yaml`.
- InferenceX PR #3930, "verified-pr", run 39462624600. gh shows the PR still OPEN.
- Concurrency 8, 10, 13, 14, 22, 24, 25, 28; golden AL "2.99". HiCache (`--hicache-size 300 --hicache-write-policy write_back`) from c22 up.
- Worker parallelism: `--tensor-parallel-size 8 --expert-parallel-size 1 --quantization modelopt_fp4 --speculative-num-steps 3 --speculative-num-draft-tokens 4`.

PR #45513's eight overlaid InferenceX PRs, per its body table:

| PR | Model | Status | DCP / PP |
|---|---|---|---|
| #3930 | GLM-5.3 B300 | verified | none |
| #4346 | DeepSeek-V4 B300, Mooncake EFA | in progress | not checked |
| #4344 | Muse Spark GB300 | in progress | not checked |
| #4342, #4343, #4341, #4350 | Muse Spark retunes | in progress | not checked |
| #4333 | Qwen3.5 GB300 | in progress; retires `-agg-pareto` / `-pp-pareto` | not checked |

All are NVIDIA hardware; none are AMD. I did not inspect the DCP/PP content of the non-GLM lanes.

## 4. GLM-5.3-Flash: DCP4 option added, then removed

- **Added** by d1f14431fd (#36544), "…DCP4 overlay".
- **Removed** by b98a2d1096 (#40036, Khoa Pham, 2026-09-17), "[Docs] GLM-5.3-Flash cookbook: temporarily remove the DCP option".
  - PR motivation, verbatim: "Temporarily remove the Context Parallelism (DCP 4) row from the GLM-5.3-Flash cookbook until it is re-measured on the current TP4/EP1 + flashinfer_trtllm recipe."
- Removed page text: "The **Context Parallelism** row offers DCP4, validated on 4x GB300 with both KV pairings on the earlier TP4/EP4 recipe and queued for re-measurement on the current command; it requires the current release image, which carries the TileLang LSE fix. Other platforms and attention backends are unvalidated, and draft-extend v2 is unsupported under DCP."
- Removed flags: `--dcp-size 4 --dcp-comm-backend a2a --dcp-replicate-q-proj`, gated to `hw === "gb300"` ("DCP is validated only on 4x GB300 for now.").
- Removed benchmark rows (GB300, low-latency, sglang `d6ab04bdf1`, random ISL 1024 / OSL 256, concurrency 16, 80 prompts):

| KV/DSA pairing | TTFT (ms) | TPOT (ms) | tok/s/GPU | Notes text |
|---|---|---|---|---|
| fp8-trtllm + DCP4 | 534.1 | 7.31 | 2100.76 | "1,680.61 aggregate output tok/s at a 3.937 accept length — about 10% below the non-DCP FP8 Low Latency row" |
| bf16-tilelang + DCP4 | 510.0 | 8.0 | 1957.25 | "1,565.8 aggregate output tok/s at a 3.90 accept length" |

- No AMD DCP was ever in the GLM-5.3-Flash page. Its current MI355X cells (`glm-5.3-flash.jsx:507-594`) have no DCP.

## 5. Kimi-K3: the only DCP recipes in the cookbook, including AMD

Strategy text, `Kimi-K3.mdx:69-70`: "**Low-Latency** — no DCP … **Balanced** — the accuracy-preserving default: PP2 × DCPEP8 on B200 …, TP16/DCP16 on GB200, TP8/DCP8 on B300/GB300, TP8/DCP8 ROCm/AITER on MI35x."

AMD section, `Kimi-K3.mdx:98-112` ("AMD AITER with DCP8"):
- The pinned `v0.5.19-rocm720-mi35x-20260916` image records SGLang `e7f7447333`, "which includes AITER DCP support (#34432) and the DCP KV-free fix (#38941). No source overlay is required for DCP."
- "Keep `SGLANG_ROCM_K3_KDA_FUSED_BACKEND` unset with this image … This updated recipe remains **Final Verification In Progress**; the recorded speed numbers use their original configurations and do not validate the new image or DCP8 recipe."

MI355X cell, `kimi-k3.jsx:1313-1340`. The MI350X cell at `:1283-1310` is identical. Both are `verified: false`, `verificationStatus: "in-progress"`:
```
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1
--model-path {{MODEL_NAME}} --trust-remote-code --tp-size 8 --dcp-size 8 --dcp-comm-backend a2a --prefill-attention-backend aiter --decode-attention-backend aiter --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```
Image: `kimi-k3.jsx:518-519`.

B300 Balanced cell, `kimi-k3.jsx:1146-1160`, `verified: true`: `--trust-remote-code --tp-size 8 --dcp-size 8 --mem-fraction-static 0.85 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3`.

GB300 / GB200 / B200 DCP cells (all in-progress):
- GB300 Balanced, `:1547-1564`: TP8/DCP8, env `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`.
- GB200: `--dcp-size 16` at `:1621` and `:1644`.
- B200: PP2 × DCPEP8 at `:1218` and `:1251`.

Benchmark rows, `kimi-k3-benchmarks.jsx`, all random ISL 8192 / OSL 1024:

| Cell (version) | Quant / spec | Conc | TTFT (ms) | TPOT (ms) | tok/s/GPU | Lines |
|---|---|---|---|---|---|---|
| B300 Balanced TP8/DCP8 (`v0.5.18 @ 71de97b2`) | mxfp4, none | 64 | 11635 | 40.19 | 1395 | 63-68 |
| B300 Balanced TP8/DCP8 | mxfp4, dspark | 64 | 12038 | 24.47 | 1987 | 71-76 |
| B300 Balanced TP8/DCP8 | nvfp4, none | 64 | 11063 | 41.55 | 1373 | 79-84 |
| B300 Balanced TP8/DCP8 | nvfp4, dspark | 64 | 11664 | 22.17 | 1946 | 87-92 |
| MI350X Balanced (`v0.5.19 @ 12771786`) | mxfp4, dspark | 1 | 604 | 5.39 | 186 | 98-107 |
| MI350X Balanced | mxfp4, dspark | 16 | 6035 | 15.32 | 864 | 98-107 |
| MI350X Balanced | mxfp4, dspark | 64 | 23720 | 31.45 | 913 | 98-107 |
| MI350X Balanced | mxfp4, none | 1 | 590 | 18.55 | 58 | 110-119 |
| MI350X Balanced | mxfp4, none | 16 | 6331 | 32.80 | 462 | 110-119 |
| MI350X Balanced | mxfp4, none | 64 | 18665 | 70.39 | 813 | 110-119 |

- MI350X DSPARK rows pin `SGLANG_SIMULATE_ACC_LEN=5` (comment at `:95-97`).
- Per `Kimi-K3.mdx:110-112`, the MI350X rows do **not** validate the DCP8 recipe.

Other DCP notes on the Kimi-K3 page:
- Peak Capacity preset, `Kimi-K3.mdx:436`: "`dp = k` + `--dcp-size 8` | … concurrency ceiling +72% at the same engine throughput, ~1.8× ITL".
- HiCache L3 drops DCP: `:307`.
- Blackwell DCP rules: `:205-211`.
- Deep PP prefill: `:339-342`.

## 6. GLM DCP support outside the cookbook (relevant to GLM-5.2 + DCP)

`docs/docs/advanced_features/dcp.mdx:175-196`, "DCP × DSA (GLM-5.2, GLM-5.3)", added by 8bf145d119 / PR #43735 (merged 2026-10-13T18:35:36Z). Requirements, verbatim:

> "CUDA SM100/SM103 with the `trtllm` DSA backends (selected automatically), an `fp8_e4m3` or `bf16` KV cache, `attn_tp_size % dcp_size == 0`, and, with speculative decoding, the checkpoint's own MTP draft as a single EAGLE chain … HiSparse, prefill CP, HiCache, PD disaggregation, unified memory and `index_kpool > 1` are not supported in this combination."

So there is no AMD path. The PR body says "other DSA backends need to return the LSE under DCP" as a fast follow.

Example command, `dcp.mdx:186-196`:
```
sglang serve --model-path zai-org/GLM-5.3 --trust-remote-code --tp-size 8 --dcp-size 2 --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
```

Accuracy and throughput from PR #43735's body: `zai-org/GLM-5.3` FP8, 8×B300, `--tp 8 --mem-fraction-static 0.88 --max-running-requests 64`. GSM8K is `few_shot_gsm8k` 8-shot (not sgl-eval); AIME26 is sgl-eval.

| Config | GSM8K | AIME26 pass@1 | pass@4 | AIME26 gen tok/s | MTP accept len |
|---|---|---|---|---|---|
| TP8 | 0.912 | 99.17% | 100% | 1140 | - |
| TP8 + DCP2 | 0.917 | 99.17% | 100% | 1041 | - |
| TP8 + DCP2 + MTP | 0.919 | 95.83% | 100% | 1876 | 3.51 |
| TP8 + DCP8 | 0.914 | 95.00% | 100% | 838 | - |
| TP8 + DCP8 + MTP | 0.919 | 95.00% | 100% | 1784 | 3.62 |

These were measured on GLM-5.3, not GLM-5.2. No GLM-5.2 + DCP measurement appears in the cookbook or in this PR body.

## 7. Cross-check in InferenceX (origin/main)

- `inferencex-e2e/configs/amd-master.yaml:1463-1486` (`glm5.2-fp4-mi355x-atom-agentic-mtp`, framework **atom**, not SGLang, image `rocm/atom-dev:nightly_202609211553`).
  - Its comment says: "large concurrency (C16-C48) adds DCP4 to shard the decode KV and keeps the LMCache DRAM offload".
  - The search-space on main lists only `{ tp: 4, … conc-list: [2, 4, 8, 10] }` and `{ tp: 8, … conc-list: [1, 2, 4] }`, with no `dcp-size` entry.
- The SGLang MI355X GLM-5.2 agentic recipes (`glm5.2/sglang/mi355x-fp{4,8}-mtp/agentic.yaml`, `mi325x-fp8-mtp`) contain no dcp key, only `tensor-parallel-size` 4 or 8. None of them is mirrored in the cookbook's GLM-5.2 AgentX section.
- Every `dcp-size: 8` entry in `amd-master.yaml` is Kimi-K3 (atom / vllm lanes, lines 731-759 and 1665-1728).

I did not check open (unmerged) InferenceX PRs for GLM-5.2 SGLang + DCP on AMD.

## Open / UNVERIFIED

- Actual AgentX scores and throughput for any cited InferenceX run (e.g. 39187209967, 39462624600) were not fetched. The cookbook does not reproduce them.
- Whether any open SGLang or InferenceX PR adds GLM-5.2 + DCP on AMD (ROCm DSA backend returning LSE) was not searched beyond the files above.
- The GLM-5.2 MI355X MXFP4 MTP image date disagrees between page and config: `…-20260930` at `GLM-5.2.mdx:140` vs `…-20260923` at `glm-5.2.jsx:100`. I don't know which is intended.

Scratch copies of the extracted snippets are in `/var/tmp/yihou/research_glm52_dcp/` (`agentx-glm-5.2.jsx`, `agentx-glm-5.3.jsx`, `cfg-*.jsx`, `bench-*.jsx`, `kimi-k3*.jsx`).