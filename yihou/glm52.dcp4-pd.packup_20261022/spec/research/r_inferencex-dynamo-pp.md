# InferenceX: NVIDIA Dynamo + pipeline-parallel (PP) AgentX experiments, as of 2026-10-21

All InferenceX references are to `origin/main` at `6046414d2`, fetched 2026-10-20. I took the numbers from the `bmk_agentic_*` and `eval_results_all` artifacts of each PR's pinned GitHub Actions run, downloaded under `/var/tmp/yihou/research_glm52_dcp/run_<id>/`. My summarizer is `/var/tmp/yihou/research_glm52_dcp/scripts/summ.py`.

How to read the result columns:
- **tok/s/GPU** is `request_metrics.throughput.per_gpu.total_tput_tps` (input plus output tokens per GPU). Output-only values are given where relevant.
- **Interactivity** is `latency.intvty`, in tok/s/user. Its values equal 1/ITL (for example, ITL p50 0.01034 s gives 96.7).
- **TTFT** and **E2EL** are in seconds.
- InferenceX AgentX has no single "AgentX score". A point is described by throughput against interactivity or latency, plus a GSM8K `em_strict` eval gate. The sign-off verifier also reports which points sit on the Pareto frontier.

---

## 0. Every PP>1 experiment found

The list comes from four searches: `git log --grep` for pp/pipeline, a grep of master configs for `pp: [2-9]`, a grep of recipes for `pipeline-parallel-size: [2-9]`, `PP_LAYER` and `pp-async`, and `gh pr list` searches for pp4, PP2, "PP prefill", "pipeline parallel", "dynamo pp" and tp8pp2.

| PR | State / date | Model / HW | Engine | Where PP is used |
|---|---|---|---|---|
| **#4300** | MERGED 2026-10-20 | GLM-5.2 NVFP4 / GB300 | Dynamo + SGLang | PP4 **prefill only** (disagg) |
| #2812 | MERGED 2026-09-09 | Qwen3.5-397B NVFP4 / GB300 | Dynamo + SGLang | PP4 prefill only |
| #3231 | MERGED 2026-10-05 | Qwen3.5-397B FP8 / GB300 | Dynamo + SGLang | TP1xPP4 prefill only |
| #3840 | MERGED 2026-10-09 | Qwen3.5-397B FP8 / GB300 | Dynamo + SGLang | PP4 prefill + DEP16 decode |
| #4333 | OPEN (2026-10-20) | Qwen3.5 NVFP4 / GB300 | Dynamo + SGLang | PP4 prefill; replaces the #2812 keys |
| #2391 | MERGED 2026-07-29 | Kimi-K3 MXFP4 / B200 | "dynamo-vllm" label, but direct `vllm serve` | Aggregated TP8xPP2 |
| **#2672** (commit `bf317e636`) | MERGED 2026-08-19 | Kimi-K3 FP4 / B200 | dynamo-vllm (vLLM frontend) | Aggregated TP8xPP2 + DCP8 |
| #2700 | MERGED 2026-08-21 | Kimi-K3 / GB200 | dynamo-vllm | Aggregated TP8xPP2/DCP8 |
| #3045 | MERGED 2026-09-19 | Kimi-K3 / GB200 | dynamo-vllm | Telemetry plus CUDA Graph memory reserve on the TP8/PP2 recipes (no topology change) |
| #3087 | MERGED 2026-10-02 | Kimi-K3 / B200 | dynamo-vllm | Aggregated TP8/PP2/DCP8, more concurrency points |
| #3747 (closed), #3748 / #4322 (open) | 2026-10 | Kimi-K3 / GB200 | Dynamo + vLLM | Aggregated TP8 PP2 DCP8 |
| #4045 (non-NVIDIA, see §6) | OPEN 2026-10-14 | GLM-5.2 MXFP4 / MI355X | ATOM | PP4 prefill + TP4/DCP4 decode |
| #2678 (non-NVIDIA) | CLOSED 2026-08-19 | Kimi-K3 / MI355X | vLLM | Aggregated TP8xPP2 |

Infrastructure: #2166 (merged 2026-07-11) added the `pp` master-YAML field (`PP_SIZE`, single-node GPU count = `TP * PP_SIZE * PCP_SIZE`).

Not counted as PP: the Qwen3.5 B300/GB300 aggregated recipes set `pp-max-micro-batch-size`, but the grep found no `pipeline-parallel-size > 1` in them.

---

## 1. #4300 (commit `bc6a04461`): GB300 GLM-5.2 NVFP4, Dynamo + SGLang, PP4-prefill disagg

**Metadata**
- PR: https://github.com/SemiAnalysisAI/InferenceX/pull/4300. Created 2026-10-19, merged 2026-10-20T08:20:54Z, label `full-sweep-fail-fast`, author nvpohanh.
- Image: `lmsysorg/sglang:nightly-dev-20261016-a93f16f0@sha256:0b3e3970…` for both keys. The previous image was `nightly-dev-20260914-4358a161` (see the `nvidia-master.yaml` diff in `bc6a04461`).
- Recipe `frameworks.sglang: 0.0.0.dev1+gc15edf314`. Local `git log` shows `c15edf314d` dated 2026-10-16 and `a93f16f0cf` dated 2026-10-15.
- Dynamo: the disagg key installs the wheel `ai-dynamo 1.6.0.dev20261016`; the agg key uses the bundled `1.6.0.dev20261015`.
- KV transfer: NIXL. Runner: `cluster:gb300-nv`, 4 GPUs per node.

**SGLang dependency.** The PR body says "The disaggregated points move to one PP4 prefill worker per node (sgl-project/sglang#39378)".
- sglang#39378 "Fix disagg PP MTP for GLM-5.2" merged 2026-09-19 as `3a64faa1f2`. My `git merge-base` checks show it **is** in `a93f16f0` (the new image) and **is not** in `4358a161` (the old image).
- What #39378 says it fixes: PP + speculative decoding was rejected outside NPU; the last PP stage was missing `embed_tokens`; `dsa_topk_indices` was dropped across the PP ring; NIXL needed to map heterogeneous PP prefill stages to unpipelined decode MLA pools.
- Its negative control on the parent commit: `AssertionError: Pipeline parallelism is not compatible with overlap schedule, speculative decoding`.
- Its accuracy check: PP4 prefill + TP4 decode, steps 3/3, GSM8K 94.69%, decode accept length 3.745.

**Why PP was chosen:** the #4300 description, changelog and comments give no performance rationale. They only state the topology change and cite #39378.

**PP is prefill only.** Every disagg point has prefill `tp 1, pp 4, ep 1, dp-attn false` and decode `pp 1`. The 2 aggregated points (TP4, TP8) use no PP.

**Key prefill flags** (`inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/gb300-fp4/agentx/disagg-variants.yaml`, base `roles.prefill`, quoted verbatim):

```
tensor-parallel-size: 1
pipeline-parallel-size: 4
chunked-prefill-size: 16384
max-prefill-tokens: 16384
max-running-requests: 16
disable-cuda-graph: true
moe-runner-backend: flashinfer_cutedsl
fp4-gemm-backend: flashinfer_cutedsl
mem-fraction-static: 0.83
context-length: 1048576
enable-hierarchical-cache: true
hicache-write-policy: write_back
hicache-io-backend: direct
speculative-algorithm: EAGLE
speculative-moe-runner-backend: flashinfer_trtllm
speculative-num-steps: 5
speculative-eagle-topk: 1
speculative-num-draft-tokens: 6
router-session-affinity-ttl-secs: '3600'
disaggregation-transfer-backend: nixl
```

**Settings that are absent:**
- No `pp-async-batch-depth` and no `pp-max-micro-batch-size` (grep finds neither in this recipe).
- No `PYTHONOPTIMIZE`.
- Decode adds `router-decode-active-request-weight: '8192'`, `chunked-prefill-size: 64` and `stream-interval: 30`.

**Per-point overrides:**
- **c216 and c264:** `SGLANG_PP_LAYER_PARTITION: 20,20,20,18`. The recipe comment says: "The default 19,19,20,20 split also gives the last stage the LM head and MTP draft layer."
- **c56 and c264:** `chunked-prefill-size: 8192` and `max-prefill-tokens: 8192`.
- **c264:** `hicache-size: 185`, prefill `mem-fraction-static: 0.88`, decode 0.94.
- **c30 and c40:** `hicache-size: 100`.
- **DEP decode:** `moe-a2a-backend: flashinfer`, `moe-runner-backend: flashinfer_trtllm`, `disable-radix-cache: true`, `cuda-graph-max-bs-decode: 32`.
- **Golden AL:** 3.33 at 4 steps and 3.61 at 5 steps (`perf-changelog.yaml` #4300 entry).

**Topology:** prefill workers are always PP4 (TP1, EP1, one 4-GPU node each).

| Recipe override | Conc | Prefill workers | Decode | Nodes / GPUs |
|---|---|---|---|---|
| `override_1p5d_pp4_tp4_c30` | 30 | 1 | 5x TP4 | 6 / 24 |
| `override_1p3d_pp4_tp4_c40` | 40 | 1 | 3x TP4 | 4 / 16 |
| `override_1p1d_pp4_dep8_c56` | 56 | 1 | 1x DEP8 (TP8, DP8, EP8, dp-attn) | 3 / 12 |
| `override_3p1d_pp4_dep8_c216` | 216 | 3 | 1x DEP8 | 5 / 20 |
| `override_3p1d_pp4_dep8_c264` | 264 | 3 | 1x DEP8 | 5 / 20 |
| `override_4p1d_pp4_dep16_c256` | 256 | 4 | 1x DEP16 | 8 / 32 |

**Results.** Pinned run [39187209967](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39187209967) (attempt 2, head `8f965ad4`), pinned with `/use 39187209967` by nvpohanh on 2026-10-20.

| Conc | Topology | GPUs | tok/s/GPU total | tok/s/GPU output | TTFT p50 / p90 | Interactivity p50 / p90 | E2EL p50 / p90 | Server cache hit | GSM8K em_strict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | agg TP4 | 4 | 5,180.6 | 37.25 | 0.361 / 0.933 | 330.62 / 283.47 | 3.39 / 17.11 | 0.980 | – |
| 1 | agg TP8 | 8 | 2,859.6 | 19.39 | 0.363 / 0.884 | 381.03 / 345.82 | 3.00 / 14.21 | 0.980 | – |
| 30 | 1P(PP4)+5D TP4 | 24 | 8,942.3 | 54.17 | 0.680 / 1.833 | 319.02 / 247.54 | 2.40 / 9.95 | 0.939 | 0.9697 |
| 40 | 1P(PP4)+3D TP4 | 16 | 16,924.2 | 109.05 | 0.954 / 2.254 | 228.61 / 185.15 | 3.22 / 13.46 | 0.936 | 0.9651 |
| 56 | 1P(PP4)+1D DEP8 | 12 | 28,406.5 | 202.53 | 1.097 / 3.156 | 139.05 / 131.86 | 4.90 / 19.87 | 0.922 | 0.9689 |
| 216 | 3P(PP4)+1D DEP8 | 20 | 55,487.9 | 392.99 | 1.676 / 5.485 | 101.41 / 94.83 | 6.61 / 28.91 | 0.915 | 0.9606 |
| 264 | 3P(PP4)+1D DEP8 | 20 | 63,647.2 | 458.15 | 2.072 / 9.432 | 96.70 / 89.65 | 7.90 / 32.99 | 0.906 | 0.9727 |
| 256 | 4P(PP4)+1D DEP16 | 32 | 44,452.7 | 313.82 | 1.375 / 4.069 | 124.70 / 114.64 | 5.46 / 23.58 | 0.920 | 0.9651 |

- The c264 point used the `semianalysis_cc_traces_weka_062126` dataset. Its mean ISL was 138,651.7 and mean output 1,005.3; it ran 3,629.9 s with 33,086 successful requests and 0 errors.
- Sign-off verifier ([comment](https://github.com/SemiAnalysisAI/InferenceX/pull/4300), 2026-10-20), Check 14: "6 frontier points at P90 (c30, c40, c56, c256, c216, c264; agg TP4/TP8 c1 are dominated)".
- Cookbook: the configs are in the sglang cookbook through sglang#45111 and #45311. On sglang `origin/main`, `docs/src/snippets/agentx/zai-org/glm-5.2.jsx` marks them `"status": "verified"` with `"run": "39187209967/attempts/2"`, `"pr": "4300"` (lines 252–256 onward; `--pipeline-parallel-size 4` at line 685).

**Comparison with the non-PP layout it replaced.** These are the pre-#4300 points from #3659, run [36997573694](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36997573694), image `nightly-dev-20260914-4358a161`.

These are **not like-for-like**. Per the #3659 changelog, the earlier disagg points used MTP 2-1-3 (#4300 uses 5-1-6); the image, Dynamo version and HiCache settings also differ.

| Conc | Topology (no PP) | GPUs | tok/s/GPU | TTFT p50 / p90 | Interactivity p50 / p90 |
|---|---|---|---|---|---|
| 48 | 1P DEP4 + 2D TP4 | 12 | 21,138.9 | 3.472 / 11.16 | 152.74 / 124.1 |
| 45 | 1P DEP4 + 4D TP4 | 20 | 13,403.0 | 2.921 / 8.957 | 215.18 / 175.52 |
| 45 | 1P DEP4 + 6D TP4 | 28 | 9,634.8 | 2.745 / 8.099 | 239.33 / 193.33 |
| 192 | 2P DEP8 + 1D DEP16 | 32 | 27,575.0 | 3.266 / 10.378 | 101.01 / 91.45 |
| 1 / 2 / 4 | agg TP8 | 8 | 2,908.5 / 2,977.1 / 4,201.9 | 0.289 / 0.249 / 0.257 (p50) | 373.2 / 371.55 / 358.92 (p50) |

My arithmetic on equal GPU counts:
- **12 GPUs:** PP4 1P1D c56 gives 28,406.5 tok/s/GPU at 139.05 interactivity. Non-PP 1P2D c48 gave 21,138.9 at 152.74.
- **32 GPUs:** PP4 4P1D c256 gives 44,452.7 at 124.70. Non-PP 2P1D c192 gave 27,575.0 at 101.01.

Neither PR attributes the difference to PP.

---

## 2. Qwen3.5-397B on GB300, Dynamo + SGLang, PP4 prefill

### #2812 (NVFP4, merged 2026-09-09)
- Key `qwen3.5-fp4-gb300-dynamo-sglang-agentic-pp-pareto`. Image `lmsysorg/sglang:nightly-dev-cu13-20260831-bb5e6198`. Dynamo router pinned to commit `61c37bc33c3e…`. KV transfer: mooncake.
- Topology: PP4-prefill / DEP4-decode at c704 (3P+2D) and c565 (3P+4D). Prefill uses MTP 5 steps at c704 and 4 steps at c565; golden AL 4.41 and 3.97.
- Prefill flags (`qwen3.5/sglang/gb300-fp4/agentx/disagg-pp-pareto-variants.yaml`, lines 53–132), verbatim:
  ```
  PYTHONOPTIMIZE: '1'
  SGLANG_PP_LAYER_PARTITION: 16,16,16,12
  tensor-parallel-size: 1
  pipeline-parallel-size: 4
  disable-overlap-schedule: true
  max-prefill-tokens: 32768
  chunked-prefill-size: 32768
  hicache-write-policy: write_back
  hicache-ratio: 0.9
  pp-async-batch-depth: 1
  disable-cuda-graph: true
  ```
- **The decode role sets `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE: '1'`.** The #4300 sign-off checklist says enabling this variable is now prohibited going forward.
- On `PYTHONOPTIMIZE=1`, the sign-off verifier wrote: "prefill sets `PYTHONOPTIMIZE=1` to bypass a stale pre-merge PP+spec assert … runtime support is merged upstream in sgl-project/sglang#35758".
- The PR body says the image "includes the support merged in sgl-project/sglang#36248". My local check disagrees: `git merge-base --is-ancestor b77cac06a9 bb5e619860` is false. The #36248 merge commit is dated 2026-08-30T23:11 −07:00; image commit `bb5e619860` is dated 2026-08-31T08:25 +08:00. sglang#35758 (`5f216fc33f`) **is** an ancestor.
- No rationale for choosing PP is stated.
- Results, run [34345050756](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34345050756) (`/reuse-sweep-run`):

| Conc | Topology | GPUs | tok/s/GPU | TTFT p50 / p90 | Interactivity p50 / p90 |
|---|---|---|---|---|---|
| 704 | 3P PP4 + 2D DEP4 | 20 | 131,367.6 | 4.559 / 7.671 | 90.37 / 80.31 |
| 565 | 3P PP4 + 4D DEP4 | 28 | 87,274.5 | 3.701 / 6.570 | 163.66 / 150.21 |
| 44 | agg TP2 (two variants) | 2 | 25,227.5 and 37,527.7 | 87.17 and 49.31 (p50) | 507.54 and 418.45 (p50) |
| 7 | agg TP8 | 8 | 4,459.9 | 2.465 / 10.874 | 561.89 / 515.59 |

- Evals: GSM8K 0.973–0.977, per the verifier.

### #3231 (FP8, merged 2026-10-05)
- Key `qwen3.5-fp8-gb300-dynamo-sglang-agentic-pp-pareto`. Image `nightly-dev-cu13-20260913-fa663e72`. Router `ad6f7d54…`.
- Prefill (`gb300-fp8/agentx/disagg-variants.yaml`): TP1, PP4, `disable-overlap-schedule: false`, `pp-async-batch-depth: 1`, `PYTHONOPTIMIZE: '1'`, `SGLANG_PP_LAYER_PARTITION 16,16,16,12`, chunk 32768, MTP 5/6.
- Pinned run [37052235829](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37052235829):

| Conc | Topology | GPUs | tok/s/GPU | TTFT p50 / p90 | Interactivity p50 / p90 |
|---|---|---|---|---|---|
| 100 | 1P PP4 + 1D DEP4 | 8 | 50,282.4 | 2.423 / 5.112 | 122.07 / 110.63 |
| 200 | 1P PP4 + 1D DEP4 | 8 | 88,801.3 | 6.558 / 11.735 | 98.77 / 88.67 |
| 600 | 3P PP4 + 2D DEP4 | 20 | 99,738.4 | 6.503 / 11.851 | 77.35 / 71.07 |
| 1000 | 5P PP4 + 3D DEP4 | 32 | 101,235.1 | 7.144 / 13.347 | 73.46 / 67.54 |
| 44 | agg TP4 (three variants) | 4 | 12,462.9 / 17,955.0 / 25,852.1 | 89.0 / 52.9 / 31.6 (p50) | – |
| 1 | agg TP4 / TP8 | 4 / 8 | 4,597.9 / 2,456.8 | – | – |

- Evals: GSM8K 0.9719–0.9765, per the verifier.

### #3840 (FP8 "midcurve", merged 2026-10-09)
- PP4 prefill + DEP16 decode, MTP6. The PR body says: "PP4 prefill roles set `PYTHONOPTIMIZE=1`, which disables Python assertions, to allow MTP with overlap scheduling on this nightly".
- Pinned run [37833971361](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37833971361):

| Conc | Topology | GPUs | tok/s/GPU | TTFT p50 / p90 | Interactivity p50 / p90 |
|---|---|---|---|---|---|
| 180 | 1P PP4 + DEP16 | 20 | 37,083.0 | 6.02 / 10.46 | 210.39 / 192.52 |
| 300 | 2P PP4 + DEP16 | 24 | 53,890.7 | 4.925 / 8.609 | 178.37 / 158.72 |
| 400 | 3P PP4 + DEP16 | 28 | 61,358.7 | 3.843 / 6.256 | 155.91 / 136.93 |
| 200 | 1P PP4 + DEP4 | 8 | 85,310.0 | 7.007 / 13.8 | 98.70 / 88.19 |
| 1000 | 5P PP4 + 3D DEP4 | 32 | 101,367.3 | 7.135 / 13.24 | 73.85 / 67.87 |

- Evals: GSM8K 0.9742–0.9780.

### #4333 (OPEN, NVFP4 refresh, not merged)
- Image `lmsysorg/sglang:dev-qwen35-agentx-gb300-42ad537de600@sha256:9b16…`. Dynamo built from source at `b208989fbe`. It retires the #2812 keys.
- The PR body says no recipe uses `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE`.
- PP points from run [39355392627](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39355392627), the only run on the PR:

| Conc | Topology | GPUs | tok/s/GPU | TTFT p50 / p90 | Interactivity p50 / p90 |
|---|---|---|---|---|---|
| 704 | 3P PP4 + DEP8 | 20 | 144,838.5 | 3.493 / 7.018 | 115.87 / 109.20 |
| 488 | 2P PP4 + DEP16 | 24 | 96,605.5 | – | 276.15 (p50) |
| 530 | 3P PP4 + DEP16 | 28 | 94,852.5 | – | 253.07 (p50) |
| 660 | 3P PP4 + DEP16 | 28 | 114,194.8 | – | 216.79 (p50) |
| 760 | 3P PP4 + DEP16 | 28 | 126,473.8 | – | 207.38 (p50) |
| 840 | 3P PP4 + DEP16 | 28 | 134,200.5 | 4.016 / 8.417 | 206.69 / 191.70 |

- The non-PP points in the same run (1P TP1 + 1D TP2, and agg TP4/TP8) range from 9,122.9 to 64,562.9 tok/s/GPU. All rows are in my summarizer output.

---

## 3. #2672 (commit `bf317e636`): Kimi-K3 FP4 B200, TP8 x PP2 + DCP8, aggregated

- PR #2672, merged 2026-08-19. Key `kimik3-fp4-b200-dynamo-vllm-agentic-dspark`.
- Commit message: "Nine points on TP8 x PP2 with DCP8 and a Mooncake external tier, concurrency 1 through 96. Concurrency 1 to 32 drafts with DSpark level 7, 48 and above runs unspeculated." The changelog entry says: "Add new B200 multi-node configs with PP2, DCP8."
- Engine label is `dynamo-vllm`, but the current recipe runs `dynamo.install: false` with `frontend.type: vllm` (`kimik3/vllm/b200-fp4/agentx/agg-variants.yaml`).
- Current recipe flags (verbatim):
  ```
  tensor-parallel-size: 8
  pipeline-parallel-size: 2
  decode-context-parallel-size: 8
  dcp-comm-backend: a2a
  kv-cache-dtype: fp8
  max-num-batched-tokens: 8192
  max-model-len: 1048576
  attention-backend: TOKENSPEED_MLA
  kv-transfer-config: MooncakeStoreConnector (kv_both)
  ```
  Env: `VLLM_USE_DIRECT_DCP_A2A/Q_GATHER/KV_GATHER=1`.
- Why PP: no rationale in this PR (the body is empty). The earlier #2391 changelog gives one for the original TP8xPP2: "the native MXFP4 checkpoint (2.8T total params, ~1.4TB weights) does not fit one 8xB200 node, so TP8 shards attention/dense and PP2 splits the 93 layers".
- Results, run [32225002756](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32225002756), image `vllm/vllm-openai:nightly-dev-x86_64-cu13.0.1-728d3ad`, 16 GPUs. The result JSON records `decode_dcp_size 1`. #3043 is titled "correct DCP metadata" (I did not read it).

| Conc | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 72 | 96 |
|---|---|---|---|---|---|---|---|---|---|
| tok/s/GPU | 972.2 | 1,030.3 | 1,559.9 | 2,736.5 | 4,374.6 | 6,998.1 | 7,896.7 | 8,624.0 | 9,079.1 |
| Interactivity p50 | 463.81 | 234.97 | 232.08 | 204.21 | 107.52 | 55.71 | 26.59 | 16.02 | 11.96 |

- Follow-up #3087, run [36879331968](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36879331968), image `nightly-ac9126e5…`, TP8/PP2/DCP8: c1 986.0 @ 241.0; c32 8,518.1 @ 79.22; c56 9,976.4 @ 37.83; c72 10,584.9 @ 25.32; c80 10,451.0 @ 21.51; c96 8,854.0 @ 11.76 (tok/s/GPU @ interactivity p50).
- Earlier #2391, run [30385435921](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/30385435921): TP8xPP2, no DCP, no spec decoding, image `vllm/vllm-openai:kimi-k3`. Results: c1 348.3; c16 1,051.0; c32 723.8 tok/s/GPU (TTFT p50 177.98 s at c32).

**GB200 comparison (#2700).** Run [32424103771](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32424103771) has both layouts on 16 GPUs, with different images (`…-728d3ad` for TP8/PP2/DCP8, `…-75c2eef` for TP16/DCP16). The changelog says the PR replaces "the non-DSpark DCP16 candidates … with TP8 x PP2/DCP8 Mooncake candidates".

| Conc | TP8 PP2 DCP8 | TP16 DCP16 |
|---|---|---|
| 16 | 4,764.0 @ 121.49 | 2,300.1 @ 194.28 (TTFT p50 48.47 s) |
| 48 | 8,131.0 @ 28.22 | 4,853.7 @ 16.06 |

Open #4322 (Kimi GB200, image `nightly-e68efc68`) reports "internal" numbers in its body that I did not verify against artifacts: "Agg: c48 10,319 @ 31.2" for TP8 PP2 DCP8.

---

## 4. Related non-NVIDIA PP experiment: #4045, GLM-5.2 MXFP4 on MI355X with ATOM (OPEN)

- Key `glm5.2-fp4-mi355x-atom-disagg-agentic-mtp`. Image `rocm/atom-dev:nightly_202610131249`. Head `9f774fe3f`. Not merged.
- Prefill: one "CPP4" worker, PP4 x TP1, `VLLM_PP_LAYER_PARTITION "20,20,20,18"`, `pipeline-parallel-size: 4`. Decode: TP4 + `decode-context-parallel-size: 4` for 1P1D and 1P2D; plain TP4 for 1P3D.
- It needs an srt-slurm patch (`604-atom-tp-from-pp.patch`, NVIDIA/srt-slurm#604) so the PP4 prefill does not launch as `-tp 4 --pipeline-parallel-size 4`.
- The PR body reports run [39028510364](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39028510364) "all green". I did not download its artifacts.

| Point | tok/s/GPU | TTFT p50 / p90 (s) | GSM8K strict |
|---|---|---|---|
| 1P1D C56 | 31,067 | 1.45 / 2.85 | – |
| 1P1D C72 | 38,487 | 1.93 / 4.00 | – |
| 1P1D C96 | 42,884 | 3.73 / 11.35 | – |
| 1P1D C112 | 39,245 | 14.78 / 26.70 | – |
| 1P1D C128 | 39,933 | 16.02 / 37.22 | 0.962 |
| 1P2D C56 | 24,030 | 1.34 / 2.71 | 0.965 |
| 1P3D C12 | 3,974 | 0.94 / 1.80 | – |
| 1P3D C16 | 6,481 | 0.94 / 2.00 | 0.972 |
| 1P3D C24 | 10,073 | 1.06 / 2.19 | 0.966 |

A later comment, `/reuse-sweep-run 38674332505` by seungrokj, points at the earlier layout's run.

---

## 5. Patterns in the sources (no interpretation)

- **SGLang PP is used for disaggregated prefill only.** Every Dynamo+SGLang PP point (GLM-5.2 #4300, Qwen3.5 #2812/#3231/#3840/#4333) is PP4 x TP1 prefill with PP1 decode.
  - On origin/main, the only SGLang PP+spec path #39378 allows is "single-layer EAGLE with PP only for disaggregated prefill on the DeepSeek/GLM model path … Decode/null PP + speculative decoding … remain rejected" (sglang#39378 body).
- **vLLM PP (Kimi-K3) is used aggregated:** TP8 x PP2, plus DCP8 from #2672 onward.
- **Rationale for PP:** only #2391 states one (weights do not fit one 8xB200 node). #4300, #2812, #3231 and #3840 state none.
- **PP-specific SGLang settings seen:**
  - `SGLANG_PP_LAYER_PARTITION`: GLM 20,20,20,18; Qwen 16,16,16,12.
  - `pp-async-batch-depth: 1`: Qwen only.
  - `PYTHONOPTIMIZE=1`: Qwen only.
  - `disable-cuda-graph: true` on prefill: GLM and Qwen FP4.
  - Chunked prefill: 16384 for GLM (8192 at c56/c264), 32768 for Qwen.
- **Engine versions:**
  - GLM-5.2 PP needs an image containing sglang `3a64faa1f2` (#39378). `a93f16f0` contains it; `4358a161` does not.
  - Qwen PP images: `bb5e6198`, `fa663e72`, and the dev image `42ad537de600`.

Scratch artifacts are under `/var/tmp/yihou/research_glm52_dcp/`: `run_*`, `prs/*.json`, `perf-changelog.yaml`, `nvidia-master.yaml` and `pr4045-disagg-variants.yaml`.