# InferenceX × GLM-5.2/5.3 × DCP/PP: configs, PRs and results (checked 2026-10-21)

## TL;DR

1. **No SGLang + DCP + GLM-5.2 config is merged in InferenceX `origin/main`, and InferenceX CI has produced no successful SGLang DCP result.**
   - `git grep` over every `*glm5*` recipe finds no `decode-context-parallel-size` / `dcp-size`. The only CP flags are `attn-cp-size` (H200 prefill CP, and `attn-cp-size: 1` on GB200). Details in §2.
   - The SGLang MI355X key today has only TP4/EP4+HiCache and TP8/EP1 arms (`amd-master.yaml` L1417-1431 on origin/main).
2. **The one SGLang + DCP + GLM-5.2 + AMD config is OPEN PR SemiAnalysisAI/InferenceX#4064** (author EricKing626, opened 2026-10-14, head `a600ec97`).
   - It adds `glm5.2-fp4-mi355x-sglang-agentic-mtp` TP4/EP4 + DCP4 points at c16/24/40/48.
   - Its PR-body results table is all "TBD".
   - Its only CI sweep (run 38612691278) **failed**: the c40 and c48-eval DCP4 jobs died during CUDA graph capture with `UnboundLocalError: local variable 'lse' referenced before assignment` in `forward_mla_rocm.py:969`. The other jobs were cancelled.
   - The PR itself says it needs an image with sglang#43735 plus sglang#44322, and the master image was not bumped (still `v0.5.20-rocm720-mi35x-20260930`).
3. **The actual SGLang + DCP4 + GLM-5.2 + MI355X numbers are in sglang PRs, not in InferenceX.**
   - sgl-project/sglang#44322 (OPEN) and #45205 (OPEN, stacked on #44322) both have tables measured with the InferenceX agentic replay. They are quoted in §4.
   - These are author-run numbers, not InferenceX official or unofficial CI runs.
4. **The DCP + GLM-5.2 + MI355X results that do exist in InferenceX CI are all ATOM.**
   - PR #3205: merged, sweep run 35378449449.
   - PR #4308: open, run 39540045682.
   - PR #4045: open, ATOM PD with PP4 prefill and DCP4 decode, run 38674332505.
   - The ATOM DCP4 arm was later **dropped from origin/main** by #3463 (1f00e7a01) and is proposed again in #4308.
5. **NVIDIA Dynamo + PP is confirmed: PR #4300, merged 2026-10-20 (bc6a04461).**
   - Key `glm5.2-fp4-gb300-dynamo-sglang-agentic-disagg` uses PP4 prefill workers (tp1, pp4) with TP4 or DEP8/DEP16 decode.
   - The sweep is run 39187209967. Numbers are in §5.
6. **I found no vLLM + DCP + GLM-5.2 config.** vLLM DCP configs on origin/main are Kimi-K3 only. The only GLM vLLM file is `benchmarks/single_node/speedbench/glm52_fp4_b300_vllm.sh` (SPEED-Bench). GLM-5.3 on MI355X is TileRT (`glm5.3-fp8-mi355x-tilert-agentic`), with no DCP.

---

## 1. Repo layout: where configs and results live

- **Agent docs:** `CLAUDE.md` just points to `AGENTS.md` (origin/main).
- **Master configs:** `inferencex-e2e/configs/{amd,nvidia}-master.yaml`. Each config key sets image, model, runner, framework and a search space (tp/ep/dcp-size/pp/conc-list/kv-offloading/srt-recipe).
- **Recipes:**
  - Single node: `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/<model>/<engine>/<hw>/agentic.yaml`, one `override_*` variant per point.
  - Multi node: `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/<model>/<engine>/<hw>/agentx/*.yaml`.
- **Change record:** `inferencex-e2e/perf-changelog.yaml`. AGENTS.md says an entry triggers result collection.
- **Results:** no result files are committed in the repo (apart from `experimental/token_position_decode_slo/glm-5/results/*`, which is GLM-5 vLLM and not relevant). Results live in three places:
  - GitHub Actions artifacts `bmk_agentic_<...>` (one JSON per point) on each "Run Sweep" run.
  - The visualizer, `inferencex.semianalysis.com/inference?unofficialRun=<run_id>`, linked by a bot comment on each PR.
  - Staged or official data in InferenceX-app.
- **Metrics:** the PR tables use "P90 Interactivity (tok/s/user)", "TPUT per Chip (tok/s/GPU)" and cache hit rates. I did not find a single scalar "AgentX score".
- **How I extracted numbers:** I read them from the artifact JSONs, using fields `request_metrics.latency.intvty.p90`, `request_metrics.throughput.per_gpu.total_tput_tps`, `latency.ttft.p90`, `latency.e2el.p90` and `server_metrics.cache.*`.
- **Workload** (from the artifact JSON): dataset `semianalysisai/cc-traces-weka-062126`, 3600 s window.

## 2. Every GLM-5.2/5.3 config key on origin/main (6046414d)

**AMD (`amd-master.yaml`)**

| Key (line) | Engine | HW | Quant | Parallelism / arms | DCP | MTP | Image |
|---|---|---|---|---|---|---|---|
| `glm5.2-fp8-mi325x-sglang-agentic-mtp` (L1105) | sglang | MI325X | fp8 | TP8/EP1, c1-8 | no | EAGLE | `lmsysorg/sglang:v0.5.20-rocm720-mi30x` |
| `glm5.2-fp4-mi355x-sglang-agentic-mtp` (L1417) | sglang | MI355X | MXFP4 (`amd/GLM-5.2-MXFP4`) | TP4/EP4 + HiCache c4-16; TP8/EP1 c1,2,4 | **no** | EAGLE 5/1/6 | `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930` |
| `glm5.2-fp8-mi355x-sglang-agentic-mtp` (L1448) | sglang | MI355X | fp8 | TP8/EP1 none c1,2,4,10; TP8 + HiCache c8,12,16 | no | EAGLE 5/1/6 | `...v0.5.19-rocm720-mi35x-20260917` |
| `glm5.2-fp4-mi355x-atom-agentic-mtp` (L1470) | ATOM | MI355X | MXFP4 | TP4 c2-10; TP8 c1-4 | **no on main** (the comment still mentions DCP4, but the arm was deleted) | mtp | `rocm/atom-dev:nightly_202609211553` |
| `glm5.3-fp8-mi355x-tilert-agentic` (L1556) | TileRT | MI355X | fp8 | disagg 1P1D TP8, c1 | no | mtp | `ghcr.io/tile-ai/tilert-rocm-decode:0.1.6` |

**SGLang MI355X FP4 recipe flags** (`.../glm5.2/sglang/mi355x-fp4-mtp/agentic.yaml`):
- Base args: `kv-cache-dtype fp8_e4m3`, `dsa-prefill-backend triton`, `dsa-decode-backend triton`, `chunked-prefill-size 32768`, `mem-fraction-static 0.85`, `speculative-algorithm EAGLE`, `num-steps 5`, `eagle-topk 1`, `num-draft-tokens 6`.
- Env: `SGLANG_OPT_USE_TOPK_V2=true`.
- TP4 points add `hicache-size 180` and `write_through`.

**NVIDIA (`nvidia-master.yaml`)**, all engine sglang (via Dynamo or plain) except the TRT key:
- `glm5.2-fp8-h200-dynamo-sglang-agentic-mtp-{2p2d,1p1d-hicache,agg}` (L6732/6800/6871)
- `glm5.2-{fp4,fp8}-{b300,b200}-sglang-agentic-mtp` (L6960-7042)
- `glm5.2-fp4-b200-dynamo-sglang-agentic-{agg,disagg}`
- `glm5.2-fp4-gb200-dynamo-sglang-agentic-{agg,disagg,mtp,mtp-agg}`
- `glm5.2-fp4-gb300-dynamo-sglang-agentic-{agg,disagg}` (L7382/7419)
- `glm5.2-fp4-gb300-dynamo-trt-agentic-disagg-mtp` (L7540)
- `glm5.2-fp4-b300-dynamo-sglang-agentic-agg-tp4` (L9558)

No NVIDIA GLM key uses DCP. Context-parallel only appears as:
- **H200 prefill CP:** `attn-cp-size: 8`, `enable-prefill-cp: true`, `cp-strategy: interleave`, `enable-dsa-cache-layer-split: true` in `glm5.2/sglang/h200-fp8/agentx/disagg-{1p1d-pcp8...,2p2d-pcp8...}.yaml` (L62/L65).
- **GB200:** `attn-cp-size: 1` in `gb200-fp4/agentx/disagg-mtp-variants.yaml:71`.

## 3. ATOM DCP history (AMD, a different engine from SGLang)

- **#2975** (closed WIP): ATOM `_0902` image, a single DCP point at C16. Superseded by #3171, which was in turn re-created as #3205.
- **#3205** (merged 2026-09-19, a260318ab):
  - Adds `{ tp: 4, dcp-size: 4, kv-offloading: dram, lmcache 0.4.5, conc [16,24,32,40,48] }`.
  - Script flag `--decode-context-parallel-size $DCP_SIZE`.
  - MTP K4/AL 3.33 for C16-40, K3/2.99 for C48.
  - Image `rocm/atom-dev:nightly_202609151450`.
- **#3359** (merged, 71d471256): image `nightly_202609211553`, keeps the DCP4 arm, adds `--index_cache_dtype fp4 --block-size 64`.
- **#3463** (merged 2026-09-26, 1f00e7a01): *"Drop the 17 search-space entries without srt-recipe ... and the ATOM LMCache DRAM-offload arms of Kimi-K3, MiniMax-M3 and GLM-5.2 on MI355X."* This deleted the GLM-5.2 ATOM DCP4 arm. Its comment in amd-master L1463-1468 still says "adds DCP4".
- **#4308** (OPEN): restores the TP4+DCP4 arm at C16-C48 with `ATOM_DCP_PREFILL_QREP=1`, LMCache 256 GiB per rank and image `nightly_202610181506`.
- **#4045** (OPEN): ATOM PD with PP4 prefill feeding TP4+DCP4 decode.

**ATOM results from sweep run 35378449449** (#3205; `rocm/atom-dev:nightly_202609151450`; TP4 ep1; LMCache dram)

| conc | dcp | P90 intvty (tok/s/user) | tput/GPU (tok/s/GPU) | P90 TTFT (s) | GPU hit | theoretical hit |
|---|---|---|---|---|---|---|
| 10 | 1 | 79.46 | 13,659 | 1.196 | 0.951 | 0.974 |
| 16 | 4 | 48.17 | 17,290 | 2.060 | 0.951 | 0.970 |
| 24 | 4 | 35.83 | 24,840 | 2.296 | 0.955 | 0.973 |
| 32 | 4 | 26.67 | 28,800 | 2.852 | 0.953 | 0.974 |
| 40 | 4 | 22.96 | 29,370 | 3.101 | 0.947 | 0.972 |
| 48 | 4 | 17.91 | 32,870 | 3.401 | 0.935 | 0.972 |

The same run's TP8 points: c1 230.2 / 2,152; c2 204.2 / 2,356; c4 173.9 / 3,360 (P90 intvty / tput per GPU).

**ATOM results from run 39540045682** (#4308 head `c919625c`; `nightly_202610181506`; overall conclusion = failure)

The DCP4 points that succeeded:

| conc | P90 intvty (tok/s/user) | tput/GPU (tok/s/GPU) | P90 TTFT (s) |
|---|---|---|---|
| 24 | 46.62 | 27,860 | 1.682 |
| 32 | 35.59 | 34,910 | 1.711 |
| 40 | 33.42 | 37,080 | 1.974 |
| 48 | 26.46 | 41,670 | 2.384 |

DCP4 c16 failed. TP4 c1/c8/c10 and TP8 c1/c2 also failed.

**ATOM PD results from run 38674332505** (#4045, `nightly_202610131249`)

| Topology | conc | GPUs | P90 intvty (tok/s/user) | tput/GPU (tok/s/GPU) | P90 TTFT (s) |
|---|---|---|---|---|---|
| 1P1D, PP4 prefill + TP4/DCP4 decode | 56 | 8 | 58.41 | 31,371 | 2.39 |
| same | 72 | 8 | 48.73 | 38,510 | 4.06 |
| same | 96 | 8 | 42.24 | 40,837 | 17.13 |
| same | 112 | 8 | 44.11 | 36,779 | 38.36 |
| 1P2D, DCP4 decode | 56 | 12 | 79.43 | 22,379 | — |
| 1P3D, TP4 decode, no DCP | 12 / 16 / 24 | 16 | 187.69 / 176.48 / 148.40 | 3,980 / 6,473 / 10,051 | — |

## 4. SGLang + DCP4 + GLM-5.2 MXFP4 + MI355X

### InferenceX PR #4064 (OPEN, CI failed)

**What it adds**
- Recipe variants `override_tp4_dcp4_c{16,24,40,48}`, each with:
  - `tensor-parallel-size: 4`, `expert-parallel-size: 4`, `decode-context-parallel-size: 4`
  - `max-running-requests` = 2×conc
  - `cuda-graph-max-bs-decode` = min(2×conc, 64)
  - `KV_OFFLOADING: none`
- Recipe comment: "DSA DCP rejects HiCache".
- Master entry: `{ tp: 4, ep: 4, dcp-size: 4, kv-offloading: none, conc-list: [16, 24, 40, 48], spec-decoding: mtp }`.
- `infx/srt_slurm/single_node.py` now validates DCP for sglang too.

**Its sweep, run 38612691278 (failure)**
- DCP4 c40 and c48-eval failed with the `lse` UnboundLocalError described in the TL;DR.
- c16, c24 and all non-DCP points were cancelled.
- The run used image `v0.5.20-rocm720-mi35x-20260930`.
- Log saved at `/var/tmp/yihou/research_glm52_dcp/run_38612691278_failed.log`.

### sgl-project/sglang#44322 (OPEN, head 68dbb491)

"[Feat] [AMD] dsa: Enable DCP on the triton sparse-MLA backend", built on #43735.

**Correctness table (verbatim)**
- GSM8K 0.953 without DCP and 0.953 with the PR.
- MTP accept length 3.11 → 3.47.
- KV capacity 2.83M → 8.81M tokens.

**Performance table (verbatim).** Labelled "InferenceX agentic replay on this head, TP4/EP4 + DCP4 with MTP (simulated AL 3.61), 1 h per point", baseline commit 33dd8f6a2e:

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

The PR states, under Scope: "On ROCm, pass `--dsa-prefill-backend triton --dsa-decode-backend triton`".

### sgl-project/sglang#45205 (OPEN, stacked on #44322)

"HiCache for DSA DCP with packed MTP draft". Measured on `glm5.2-fp4-mi355x-sglang-agentic-mtp` with the InferenceX agentic replay, TP4/EP4+DCP4, AL 3.61, `--hicache-size 180`:

| Conc | Config | P90 intvty | TPUT per Chip |
|---|---|---|---|
| 16 | #44322 | 49.22 | 17,976 |
| 16 | + HiCache | 49.88 | 18,198 |
| 24 | #44322 | 36.34 | 25,029 |
| 24 | + HiCache | 37.27 | 24,975 |
| 40 | #44322 | 19.56 | 25,925 |
| 40 | + HiCache | 25.55 | 29,339 |
| 48 | #44322 | 6.05 | 18,169 |
| 48 | + HiCache | 19.90 | 31,935 |

GSM8K 0.94 / 0.935 (the second with the device pool forced to evict).

### SGLang upstream state on origin/main (verified)

- sglang#43735 (merge commit 8bf145d1, merged 2026-10-13) is an ancestor of `origin/main`.
- The flag is `--dcp-size`, with alias `--decode-context-parallel-size` (`python/sglang/srt/arg_groups/fields/parallel.py:56`).
- `arg_groups/model_overrides/deepseek_v2.py:18-33` (registered for `GlmMoeDsaForCausalLM` among others):
  - rejects HiCache/HiSparse and PD disaggregation under DSA DCP
  - requires `--speculative-eagle-topk 1`
  - defaults the DSA backends to `trtllm` when they are unset
- #44322 and #45205 are not merged, so on main ROCm DSA+DCP still lacks the triton LSE path. That is per the #44322 description; I did not run it.

### Baseline: current SGLang MI355X FP4 key without DCP

Run 37326853365 (#3724 reuse; image `...mi35x-20260930`):

| Arm | conc | P90 intvty (tok/s/user) | tput/GPU (tok/s/GPU) |
|---|---|---|---|
| TP4/EP4 + HiCache | 4 | 159.7 | 7,200 |
| | 8 | 108.9 | 14,360 |
| | 10 | 92.12 | 16,660 |
| | 12 | 84.93 | 15,950 |
| | 14 | 63.20 | 18,040 |
| | 16 | 42.01 | 18,330 |
| TP8/EP1 | 1 | 271.5 | 3,087 |
| | 2 | 249.5 | 3,118 |
| | 4 | 222.7 | 4,345 |

## 5. NVIDIA Dynamo + SGLang + PP (GB300, GLM-5.2 NVFP4)

PR #4300, merged (bc6a04461):
- Image `lmsysorg/sglang:nightly-dev-20261016-a93f16f0`, Dynamo `1.6.0.dev20261016`, NIXL KV transfer, HiCache `write_back`, EAGLE 5/1/6, AL 3.61.
- PR body cites sgl-project/sglang#39378 for the PP4 prefill.
- Prefill args (`gb300-fp4/agentx/disagg-variants.yaml`): `tensor-parallel-size: 1`, `pipeline-parallel-size: 4`, `chunked-prefill-size 16384`, `disable-cuda-graph`, `moe-runner-backend flashinfer_cutedsl`.
- Changelog note: c216/c264 split the PP4 prefill layers 20,20,20,18.

Sweep run 39187209967 (artifacts):

| Topology | conc | GPUs | P90 intvty (tok/s/user) | tput/GPU (tok/s/GPU) | P90 TTFT (s) |
|---|---|---|---|---|---|
| 1×PP4 P + 5×TP4 D | 30 | 24 | 247.54 | 8,942 | 1.83 |
| 1×PP4 P + 3×TP4 D | 40 | 16 | 185.15 | 16,924 | 2.25 |
| 1×PP4 P + 1×DEP8 D | 56 | 12 | 131.86 | 28,407 | 3.16 |
| 3×PP4 P + 1×DEP8 D | 216 | 20 | 94.83 | 55,488 | 5.49 |
| 3×PP4 P + 1×DEP8 D | 264 | 20 | 89.65 | 63,647 | 9.43 |
| 4×PP4 P + 1×DEP16 D | 256 | 32 | 114.64 | 44,453 | 4.07 |
| agg TP4 | 1 | 4 | 283.47 | 5,181 | 0.93 |
| agg TP8 | 1 | 8 | 345.82 | 2,860 | 0.88 |

## 6. Caveats and UNVERIFIED items

- The "TPUT per Chip" in PR tables and the `per_gpu.total_tput_tps` I extracted have the same scale. **UNVERIFIED** that the visualizer computes them identically.
- The sglang #44322/#45205 tables were measured by the PR author. No InferenceX run ID is given for them, so they are **not reproducible from InferenceX CI records**.
- The baseline used in #44322 (TP4 c24 = 14.07 / 10,872) differs from InferenceX run 37326853365 (TP4 c16 = 42.01 / 18,330; no c24 point). The images and commits differ; I did not investigate why.
- Visualizer pages (`inferencex.semianalysis.com`) and InferenceX-app staged data were not read directly. Every number above comes from Actions artifacts or PR bodies.

Scratch files (artifact downloads, the failed log, `scripts/tab.py`) are under `/var/tmp/yihou/research_glm52_dcp/`.