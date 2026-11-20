# SGLang support and performance for DeepSeek-V4.1-Flash (and V4) - research note

Date: 2026-11-19. Author: research teammate (sglang dimension). Phase 1, research only, no GPU runs.

Evidence tags: `[1st: <file:line | URL | PR#>]` = code/file read or official source read by me;
`[2nd: <source>]` = second-hand (third-party write-up, search snippet); `[inf]` = my inference.
"local" in a number's source means the PR author's own run, not an InferenceX-published run.

Saved raw material (all inside this workspace):
- `research/data/inferencex_api_dsv41flash_all_20261119.json` - InferenceX public API dump, model DeepSeek-V4.1-Flash, 227 rows
  (`curl https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash`, fetched 2026-11-19 08:11Z).
- `research/data/inferencex_dsv41flash_table_20261119.tsv` - flattened table of the above.
- `research/data/inferencex_api_DeepSeek-V4-Pro_20261119.json` - same API, model DeepSeek-V4-Pro (935 rows).
- `research/data/sglang_cookbook_DeepSeek-V4_1_main_20261119.mdx` - sglang main cookbook (main = a735253e, 2026-11-19T07:55Z).
- `research/data/retired_dsv4pro_mi355x_sglang_disagg-umbp-dspark_at_1f87891d.yaml` - retired InferenceX MI355X SGLang PD recipe (DSv4-Pro).
- `source/sglang_main_files/deepseek_v4_hook.py`, `speculative_hook.py` - sglang main a735253e.
- `source/InferenceX` - shallow clone of SemiAnalysisAI/InferenceX main db3fc796 (2026-11-18).

---

## 0. Decision summary (most important first)

1. **No published SGLang PD (disaggregated) number exists for DeepSeek-V4.1-Flash on MI355X.** InferenceX has only the
   aggregated key `dsv41flash-fp4-mi355x-sglang-agentic-dspark`; the only MI355X SGLang PD key ever published was for
   DeepSeek-V4-Pro (`dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`, MoRI), retired 2026-10-30 by InferenceX #4664
   [1st: InferenceX perf-changelog.yaml; gh pr 4664; API dump]. Searched: InferenceX configs/changelog/open PRs, sglang PRs, cookbook.
2. **Published MI355X SGLang AgentX frontier (2026-11-17, image `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`)**:
   top throughput point = TP2 + DP2-attention c64, **181,052 tok/s/GPU (total) at P90 interactivity 73.3 tok/s/user**
   [1st: API dump, run 43344632854]. MI355X vLLM published top is higher: DP2 c384 **427,451 @ 15.8** and DP2 c128 271,329 @ 45.1
   (vllm nightly-60bd6dd2, 2026-11-18) [1st: API dump, run 43861172833].
3. **Open (unpublished, local) SGLang MI355X points push higher**: DP4+EP4 c256 **229,660 @ 78.4** (InferenceX #5034, local, staged run 43712971238);
   DP2 c128 **239.8k @ 50.5** (InferenceX #5096, local) [1st: PR bodies].
4. **On GB300 the SGLang PD curve dominates agg by ~3x**: Dynamo+SGLang PD 7P(DEP2)+1D(DEP16) c5540 **992,956 @ 161.8**,
   6P+1D(DEP32) c3018 **474,138 @ 321.5**, vs best GB300 SGLang agg 340,890 @ 75.8 (c160) [1st: API dump]. Different hardware
   (NVL72 domain, Mooncake over MNNVL) - do not transfer the ratio to MI355X [inf].
5. **SGLang code on main allows DSv4.1 DSpark PD with `mori`, `mooncake` or `nixl`, static verify, DCP=1, CP=1 on decode**
   [1st: source/sglang_main_files/deepseek_v4_hook.py:399-424]. DSv4.1 rejects PP, TBO, HiSparse, non-DSpark spec decode
   [1st: deepseek_v4_hook.py:377-397].
6. **Known ROCm PD caveats on SGLang main**: FlashMLA sparse prefill is force-disabled on HIP because it "breaks the disaggregation
   nightly" [1st: deepseek_v4_hook.py:120-130]; decoder SWA bounded replay is NOT auto-enabled for PD on ROCm ("PD on ROCm has not
   been run with it") [1st: deepseek_v4_hook.py:469]; paged-KV PD over MoRI "can answer long prompts wrong when OPUS sparse prefill
   is on" [1st: sglang PR #44031 body, open]; HiCache "has not been validated on ROCm for this model" [1st: cookbook 3.5].
7. **AgentX scoring uses a simulated DSpark acceptance length**: thinking-on golden AL **3.51 at block size 5**; golden table only
   has draft lengths 1-5 ("6-8 rejected by the image") [1st: InferenceX infx/golden_al_distribution/deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml].
   Throughput numbers below are therefore at AL 3.51 regardless of engine draft quality.
8. **Workload character (from published MI355X rows)**: mean input tokens per request 134k-327k, server GPU prefix-cache hit rate
   0.956-0.981; input tok/s is ~99% of total tok/s (c64 DP2: input 179,733 vs output 1,319 tok/s/GPU) [1st: API dump metrics].

---

## 1. Metric definitions (InferenceX API, agentic_traces)

- `tput_per_gpu` = `total_tput_tps` / GPUs = (input + output) tokens/s per GPU; this is the y-axis "tok/s/GPU" quoted in
  InferenceX PRs (e.g. #5034 "184,058 tok/s/GPU" for DP2 c64) [1st: API dump; PR #5034].
- `p90_intvty` = P90 low tail of per-user interactivity (1/ITL), tok/s/user [1st: yanfwang summarize.py docstring; API fields].
- Profile window 3600 s; AIPerf trace replay; "conc" = live sessions. Throughput runs simulate DSpark acceptance
  (`SGLANG_SIMULATE_ACC_LEN`, match-expected, real-draft-token) at golden AL 3.51; GSM8K evals use real verification
  [1st: perf-changelog entry for dsv41flash-fp4-b200-sglang-agentic-dspark; dsv41flash #3696 entry].
- Precision for all rows: checkpoint FP4 routed experts + FP8 dense ("fp4" in InferenceX) [1st: API `precision`].

---

## 2. Performance numbers

### 2.1 MI355X, SGLang, DeepSeek-V4.1-Flash - InferenceX published (API, date 2026-11-17)

Image `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`, run https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43344632854,
aggregated (no PD), DSpark block 5, no KV offload. [1st: API dump]

| TP/EP/DPA | conc | tok/s/GPU (total) | output tok/s/GPU | P90 intvty | median intvty | P90 TTFT s | server cache hit |
|---|---|---|---|---|---|---|---|
| TP4/EP1 | 1 | 8,014 | 47.9 | 775.2 | 806.5 | 0.69 | 0.980 |
| TP2/EP1 | 1 | 15,083 | 91.0 | 606.1 | 617.3 | 0.62 | 0.981 |
| TP4 | 2 | 8,493 | 56.5 | 714.3 | 800.0 | 0.60 | 0.976 |
| TP2 | 2 | 15,059 | 105.0 | 485.4 | 617.3 | 0.73 | 0.974 |
| TP4 | 4 | 10,343 | 70.3 | 625.0 | 800.0 | 0.63 | 0.969 |
| TP2 | 4 | 19,130 | 131.3 | 606.1 | 621.1 | 2.41 | 0.965 |
| TP4 | 8 | 17,200 | 118.0 | 540.5 | 746.3 | 0.95 | 0.967 |
| TP2 | 8 | 32,240 | 227.6 | 387.6 | 518.1 | 1.41 | 0.966 |
| TP4 | 16 | 29,846 | 193.3 | 434.8 | 609.8 | 0.95 | 0.959 |
| TP2 | 16 | 57,345 | 364.2 | 336.7 | 396.8 | 4.04 | 0.958 |
| TP2 | 32 | 110,904 | 639.5 | 197.6 | 238.1 | 10.16 | 0.963 |
| TP2 + DP2 attn | 32 | 110,744 | 647.8 | 131.9 | 190.8 | 1.70 | 0.965 |
| **TP2 + DP2 attn** | **64** | **181,052** | 1,319.4 | **73.3** | 97.4 | 2.45 | 0.956 |

### 2.2 MI355X, SGLang, DeepSeek-V4.1-Flash - unpublished / local numbers (same model)

| Point | tok/s/GPU @ P90 intvty | Image / commit | Source | Notes |
|---|---|---|---|---|
| DP4+EP4 attn c256 | 229,660 @ 78.4 | v0.5.23-rocm10-mi35x-20261111 | InferenceX #5034 (open), local 3600 s | "+24.8% over published top"; staged run 43712971238 |
| DP4+EP4 c128 | 156,468 @ 97.3 | same | #5034 local | |
| DP2 c128 | 223,139 @ 57.1 | same | #5034 local | |
| DP2+EP2 c128 | 196,822 @ 62.2 | same | #5034 local | |
| TP2 (no DPA) c64 | 170,887 @ 58.5 | same | #5034 local | vs DP2 c64 178,230 @ 71.9 same image -> DPA +4.3% tput, +23% P90 [inf from these two rows] |
| TP4 c1 | 8,061 @ 782.6 | same (sgl-amd-mono 7113024f) | #5034 local | published (older image) 7,234 @ 641.0 |
| DP2 c80 | 195.2k @ 61.7 | v0.5.23-rocm10-mi35x-20261115 | InferenceX #5096 (open), run 43860734741 | |
| DP2 c96 | 218.9k @ 56.0 | sglang 6c29aa5954 + 2 unmerged patches | #5096 local | |
| DP2 c128 | 239.8k @ 50.5 | same | #5096 local | |
| TP2 c16, max-running 2 | 51.9k @ 471.9 (TTFT p50 2758 ms) | same | #5096 local | vs online 57.3k @ 336.7 (618 ms): cap trades TTFT/tput for P90 |
| TP2 c32 (TP only) | 112,107 total/GPU, P90 133.5 (mean of 2 runs) | yfw/sglang-rocm:v0.5.23-rocm10-mi35x-20261101-idxshard | /perf_apps/yanfwang/dsv41-idxshard-agentx-20261103/results (my run of their summarize.py) | AgentX A/B of sglang #48152 |
| TP2 c64 (TP only) | 171,008 total/GPU, P90 58.9 | same | same | #48152 on/off ratio 1.0017 tput, 1.0000 P90 -> no effect |

### 2.3 MI355X comparators (same model, other engines, InferenceX published) [1st: API dump]

| Engine | Shape | conc | tok/s/GPU @ P90 | Date | Image |
|---|---|---|---|---|---|
| vLLM | TP2+DP2 attn | 128 | 271,329 @ 45.1 | 2026-11-18 | vllm/vllm-openai-rocm:nightly-60bd6dd2 |
| vLLM | TP2+DP2, UMBP offload | 256 | 401,896 @ 23.7 | 2026-11-18 | same |
| vLLM | TP2+DP2, UMBP offload | 384 | 427,451 @ 15.8 | 2026-11-18 | same |
| ATOM | TP2/EP2+DPA | 256 | 389,350 @ 37.8 | 2026-11-13 | (see API) |
| ATOM | TP2/EP2+DPA | 128 | 264,138 @ 52.1 | 2026-11-13 | |
| ATOM | TP2+DPA | 64 | 195,738 @ 90.8 | 2026-11-13 | |
| TileRT (PD: vLLM prefill + TileRT decode) | P TP4, D TP8 | 1 | 2,829 @ 1333.3 | 2026-11-07 | ghcr.io/tile-ai/tilert-rocm-decode:0.1.6 |

Observation only: at P90 ~73-91 tok/s/user, SGLang DP2 c64 (181k) and ATOM DP2 c64 (196k) are close; the vLLM/ATOM high
end extends to c256-c384 at P90 <= 38. No SGLang MI355X point above c64 is published.

### 2.4 Other hardware, SGLang, DeepSeek-V4.1-Flash (InferenceX published) [1st: API dump]

| HW | Framework | Shape (P / D) | conc | tok/s/GPU @ P90 | P90 TTFT s | Date |
|---|---|---|---|---|---|---|
| GB300 | dynamo-sglang PD | 6x DEP2 / 1x DEP32 (12+32 GPU) | 3,018 | 474,138 @ 321.5 | 2.35 | 2026-11-13 |
| GB300 | dynamo-sglang PD | 7x DEP2 / 1x DEP16 (14+16) | 5,540 | 992,956 @ 161.8 | 11.12 | 2026-11-13 |
| GB300 | dynamo-sglang PD | 9x DEP2 / 1x DEP16 (18+16) | 7,480 | 1,071,944 @ 105.6 | 12.31 | 2026-11-13 |
| GB300 | sglang agg | TP2/EP2+DPA | 160 | 340,890 @ 75.8 | 2.31 | 2026-11-03 |
| GB300 | sglang agg | TP2/EP2+DPA | 64 | 224,816 @ 150.8 | 1.35 | 2026-11-03 |
| B300 | sglang agg | TP2/EP2+DPA, offload on | 256 | 503,971 @ 65.9 | 2.07 | 2026-11-11 |
| B300 | sglang agg | TP2/EP2+DPA | 64 | 220,251 @ 143.9 | 1.68 | 2026-11-11 |
| B200 | sglang agg | TP4/EP4+DPA, offload on | 256 | 364,592 @ 110.4 | 1.54 | 2026-11-09 |
| GB200 | sglang agg | TP4/EP4+DPA | 256 | 355,296 @ 108.6 | 1.61 | 2026-11-06 |

GB300 PD rows: images `lmsysorg/sglang:nightly-dev-cu13-20261111-314bce7c`, run 43029773811; server cache hit drops to 0.79-0.90
(vs ~0.96 agg) and `allocated_cpu_dram_gb` = 385 [1st: API dump]. Earlier (superseded) GB300 small-PD points, TP2/EP2 workers,
Mooncake: 1P2D c64 80,750 @ 210.4; 1P1D c96 143,459 @ 132.7; 1P1D + prefill HiCache c160 187,463 @ 99.0 (TTFT 14.0 s); 2P1D HiCache
c256 179,107 @ 68.3 [1st: perf-changelog entry, InferenceX #3598, 2026-10, image nightly-dev-20260928-81f27fb3].

### 2.5 DeepSeek-V4-Pro on MI355X, SGLang PD vs agg (different model; only MI355X SGLang PD data that exists) [1st: API dump, Pro]

PD = `mori-sglang` (MoRI KV transfer, UMBP DRAM tier on prefill), image `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026`, 2026-10-28.
Agg = `sglang`, image `v0.5.22-rocm720-mi35x-20261028`, 2026-10-29. DSpark draft len 3 (golden AL 3.01, Pro-0813) on PD.

| conc | PD shape (P / D) | PD tok/s/GPU @ P90 | Agg shape | Agg tok/s/GPU @ P90 |
|---|---|---|---|---|
| 1 | TP4 / TP4 | 2,133 @ 230.9 | TP8 | 2,364 @ 248.8 |
| 16 | TP4 / TP8 | 8,982 @ 183.8 | TP8 | 12,586 @ 116.8 |
| 32 | TP4 / TP8 | 18,057 @ 153.1 | TP8 | 23,552 @ 83.5 |
| 48 | TP4 / TP8 | 23,174 @ 123.2 | TP8 | 30,896 @ 73.0 |
| 256 | DEP8 (MegaMoE) / TP8+DPA | 64,946 @ 76.5 | DEP8 | 72,744 @ 30.9 |
| 384 | same | 75,897 @ 65.3 | DEP8 | 82,903 @ 18.3 |
| 512 | same | 79,806 @ 60.6 | - | - |

Observation: at equal conc the Pro PD points have lower tok/s/GPU but 1.6-3.6x higher P90 interactivity; whether PD wins at
equal interactivity needs the Pareto overlay, not done here. The InferenceX blog "MI355X DeepSeek-V4-Pro on SGLang: 110.5x Throughput
per GPU in 26 Days" exists [2nd: search result, inferencex.semianalysis.com/blog/mi355x-deepseek-v4-pro-sglang-110x-in-26-days; not read].

### 2.6 Kernel / feature-level gains on MI355X (SGLang, DeepSeek-V4.1-Flash)

From the AMD tracker issue sgl-project/sglang#49327 (open, last updated 2026-11-18) [1st: gh issue view 49327]; each gain is the
author's measurement against the stated baseline.

| PR | Lever | Gain (MI355X unless noted) | Status |
|---|---|---|---|
| #44353 | `--monokernel-backend amd` / `--enable-amd-monokernel` (sgl-amd-mono) | output tput +11.4%, TPOT -12.0% (22-point random sweep geomean, TP4) | merged 2026-10-15 |
| LMSYS blog 2026-10-18 | AMD monokernel, AgentX replay, 4x MI355X, DSpark | P90 per-user +45-56% at 1/4/8 sessions; 331.6 -> 516.0 tok/s at 4 sessions [1st: lmsys.org/blog/2026-10-18-sglang-low-latency-mode, partially extracted; exact c4 values also in 2nd: alphasignal] | - |
| sgl-amd-mono#95 | DSpark step glue / Markov sampling | output tput +1.7% at 1-8 users (TP4) | merged 2026-11-03 |
| sgl-amd-mono#106+#115 | fewer launches per layer | output tput +8.0% at 1-8 users (TP4) | open |
| sgl-amd-mono#117 | asm layer kernels for 49-256 rows (ported from vllm#65421) | TPOT -11.1% @64 rows, -11.5% @96 verify rows (TP4) | draft |
| #47923 | fastokens tokenizer in ROCm image | mean TTFT -26 to -29%, tput +2.2-2.6% (~92k-token prompts, TP4) | merged 2026-11-11 |
| #44006 | FP4 indexer row-group kernel (decode + verify) | tput +0.7/+1.2/+2.4% at 68k/200 c1/c4/c16 (TP2) | merged |
| #42911 | fused prefill index-Q RoPE/fake-quant/FP4 | tput +0.7/+2.2/+3.3% (TP2) | merged |
| #49280 | QuickReduce-first size threshold (DSpark verify all-reduces) | verify step -4.0% c32, -6.9% c64; tput +3.0-3.9% (TP4) | draft |
| #46850 | FlyDSL INT4 QuickReduce | tput +1.6-2.7% at 60k/128k (TP4) | draft |
| #48641 | exact decoder SWA tail in HIP breakable prefill graphs | tput +4.1%, mean TTFT -13.5% (random sweep, TP4, DSpark) | draft |
| #48152 | HIP prefill indexer TP row shard | single-request TTFT -11.9% @512k, -21.4% @1M; 8k/1k unchanged | open (AgentX A/B: null, see 2.2) |
| #44265 | prefill Mega-mHC | mean TTFT -5.5% geomean 8k/2k (TP4) | open |
| #47317 | ROCm DSpark verify kernels | decode step -16% at bs1, TP8 (MI350X screen) | draft |

Other: GB300 encoder-only prefill for PD (#45265, open): +5.6% req/s at AgentX c160 1P1D and p90 TTFT 13.7 -> 1.8 s; at
10P+DEP16 c6400 906k vs 909k tok/s/GPU (tie) [1st: PR #45265 body]. Release v0.5.23 claims "75% faster single-user decode on
MI350X" for the plugins [2nd: search snippet of github.com/sgl-project/sglang/releases/tag/v0.5.23; not read].

---

## 3. Best MI355X SGLang recipe(s), verbatim

### 3.1 InferenceX `dsv41flash-fp4-mi355x-sglang-agentic-dspark` (main db3fc796)
Master entry [1st: source/InferenceX/inferencex-e2e/configs/amd-master.yaml:1176-1191]:
```yaml
dsv41flash-fp4-mi355x-sglang-agentic-dspark:
  image: lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111
  ...
      - { tp: 2, ep: 1, dp-attn: false, kv-offloading: none, spec-decoding: mtp, conc-list: [1, 2, 4, 8, 16, 32], srt-recipe: agentic.yaml }
      - { tp: 4, ep: 1, dp-attn: false, kv-offloading: none, spec-decoding: mtp, conc-list: [1, 2, 4, 8, 16], srt-recipe: agentic.yaml }
      - { tp: 2, ep: 1, dp-attn: true, kv-offloading: none, spec-decoding: mtp, conc-list: [32, 64], srt-recipe: agentic.yaml }
```
Recipe base and the top-throughput override (`benchmarks/single_node/srt-slurm-recipes/dsv41flash/sglang/mi355x-fp4-mtp/agentic.yaml`) [1st]:
```yaml
      args:
        served-model-name: deepseek-ai/DeepSeek-V4.1-Flash
        trust-remote-code: true
        tensor-parallel-size: 2
        speculative-algorithm: DSPARK
        speculative-dspark-block-size: 5
        enforce-shared-experts-fusion: true
        fp8-gemm-backend: aiter
        cuda-graph-max-bs-decode: 64
        enable-decoder-swa-bounded-replay: true
        cuda-graph-backend-prefill: disabled
        reasoning-parser: auto
        tool-call-parser: auto
        watchdog-timeout: 3600
        enable-metrics: true
      env:
        PYTHONNOUSERSITE: '1'
        PYTHONUNBUFFERED: '1'
        SGLANG_TIMEOUT_KEEP_ALIVE: '900'
        SGLANG_DEFAULT_THINKING: '1'
        SGLANG_DSV41_REASONING_EFFORT: high
        SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE: '1'
        SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT: per_rank
        SGLANG_USE_AITER: '1'
        SGLANG_USE_ROCM700A: '0'
        TRITON_HIP_USE_ASYNC_COPY: '0'
        ROCM_QUICK_REDUCE_QUANTIZATION: INT8
        SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL: 'true'
        SGLANG_OPT_HIP_FLYDSL_MXFP8: '1'
        SGLANG_HIP_FP4_INDEXER_ROWGROUP: '1'
        TORCH_BLAS_PREFER_HIPBLASLT: '1'
        HSA_NO_SCRATCH_RECLAIM: '0'
        GPU_MAX_HW_QUEUES: '2'
...
override_tp2_c64_dp:
  frontend:
    type: sglang-router
    args:
      policy: cache_aware
      request-id-headers: x-correlation-id
      dp-aware: true
      connect-timeout-secs: 900
      request-timeout-secs: 14400
      disable-health-check: true
      disable-retries: true
  roles:
    agg:
      gpus: 2
      args:
        tensor-parallel-size: 2
        data-parallel-size: 2
        enable-dp-attention: true
        enable-dp-lm-head: true
        enable-prefill-delayer: true
        enable-dp-attention-local-control-broadcast: true
        tokenizer-worker-num: 2
        prefill-delayer-token-usage-low-watermark: 0.7
        load-balance-method: total_requests
        mem-fraction-static: 0.85
        chunked-prefill-size: 32768
        prefill-decode-interval: 16
        swa-prefix-tails: 4096
        max-running-requests: 64
      env:
        SGLANG_SHARED_EXPERT_TP1: '1'
        SGLANG_DP_USE_GATHERV: '1'
        SGLANG_DP_USE_REDUCE_SCATTER: '1'
        SGLANG_AITER_CAPTURE_COPY_IN: '1'
        GPU_MAX_HW_QUEUES: '5'
        SGLANG_HIP_DSV41_INDEX_K_PAGE8: '1'
  benchmark:
    env:
      CONC: '64'
      AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID: 'true'
      AGENTIC_WARMUP_GRACE_PERIOD: '3600'
```
Low-conc points (c1-c32) additionally use `monokernel-backend: amd`, `schedule-policy: lpm` (c<=8), `prefill-decode-interval`
192/96/8, `swa-prefix-tails` 128..2048 and `max-running-requests` caps below conc (TP2 c4 = 1, c16 = 4, c32 = 8); TP2 c32 adds
`tokenizer-backend: fastokens` and drops INDEX_K_PAGE8 (monokernel refuses it) [1st: same file; changelog #5029].

### 3.2 SGLang cookbook (main a735253e), DP attention c128-c256 (newest documented MI355X high-throughput shape)
DP2 command verbatim [1st: research/data/sglang_cookbook_DeepSeek-V4_1_main_20261119.mdx, section 3.6]:
```bash
SGLANG_DEFAULT_THINKING=1 \
SGLANG_DSV41_REASONING_EFFORT=high \
SGLANG_USE_AITER=1 \
SGLANG_USE_ROCM700A=0 \
TORCH_BLAS_PREFER_HIPBLASLT=1 \
TRITON_HIP_USE_ASYNC_COPY=0 \
ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true \
SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 \
SGLANG_HIP_DSV41_INDEX_K_PAGE8=1 \
HSA_NO_SCRATCH_RECLAIM=0 \
GPU_MAX_HW_QUEUES=5 \
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_DP_USE_GATHERV=1 \
SGLANG_DP_USE_REDUCE_SCATTER=1 \
SGLANG_AITER_CAPTURE_COPY_IN=1 \
sglang serve \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --trust-remote-code \
  --engram-config '{"placement": "cpu"}' \
  --tp 2 \
  --attn-dp-size 2 \
  --ep-size 1 \
  --enable-dp-lm-head \
  --enable-prefill-delayer \
  --enable-dp-attention-local-control-broadcast \
  --tokenizer-worker-num 2 \
  --prefill-delayer-token-usage-low-watermark 0.7 \
  --load-balance-method total_requests \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion \
  --fp8-gemm-backend aiter \
  --enable-decoder-swa-bounded-replay \
  --cuda-graph-backend-prefill disabled \
  --mem-fraction-static {mem-fraction-static} \
  --chunked-prefill-size 32768 \
  --prefill-decode-interval {prefill-decode-interval} \
  --swa-prefix-tails {swa-prefix-tails} \
  --max-running-requests 64 \
  --cuda-graph-max-bs-decode 64 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --watchdog-timeout 3600
```
Deltas for c128-c256 (cookbook table, from sglang #50677 merged 2026-11-17, mirrors InferenceX #5034):

| Deployment | DP2, c128 | DP2 + EP2, c128 | DP4 + EP4, c128 | DP4 + EP4, c256 |
|---|---|---|---|---|
| `--tp`, `--attn-dp-size` | 2 | 2 | 4 | 4 |
| `--ep-size` | 1 | 2 | 4 | 4 |
| `SGLANG_DP_SHARED_EXPERT_LOCAL` | unset | 1 | 1 | 1 |
| `--prefill-decode-interval` | 24 | 32 | 32 | 40 |
| `--chunked-prefill-size` | 32768 | 32768 | 32768 | 65536 |
| `--max-running-requests` | 128 | 128 | 128 | 256 |
| `--cuda-graph-max-bs-decode` | 64 | 64 | 64 | 128 |
| `--prefill-delayer-token-usage-low-watermark` | 0.7 | 0.7 | 0.7 | 0.85 |

All columns: `--mem-fraction-static 0.85 --swa-prefix-tails 4096 --schedule-policy shortest-prefill-first
--engram-config '{"placement": "cpu", "replication": "per_node"}'`, router `sgl-router --policy cache_aware --dp-aware` with server
`--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' --load-publish-endpoint auto`.
InferenceX #5034 also serves every point with `--image-processor-backend pil` because with `auto` "in 3 of 6 TP2 DPA startups in CI
that rank never got the lock" and failed the 7200 s health check [1st: PR #5034 body].

---

## 4. PD recipes and PD support facts

### 4.1 SGLang main code constraints (a735253e) [1st: source/sglang_main_files/deepseek_v4_hook.py]
- l.399-424: DSv4.1 PD + speculative requires `SGLANG_RAGGED_VERIFY_MODE` static, `--disaggregation-transfer-backend` in
  {mooncake, nixl, mori}, `dcp_size == 1`, and no attention CP on decode; prefill CP only with `--enable-prefill-cp` + interleave.
  Error text: "DP attention is supported when both servers use the same block size and target/draft KV layout."
- l.377-397: DSv4.1 does not support speculative decoding other than DSpark, HiSparse, the unified KV layout (`is_unified_kv_triton`),
  two-batch overlap, pipeline parallelism.
- l.120-130: on HIP `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` is set False by default "until the sparse kernel is validated there"
  (it "breaks the disaggregation nightly").
- l.336-366: encoder SWA bounded replay rejects DP attention on ROCm, PD decode, PD prefill on ROCm.
- l.469: decoder SWA bounded replay auto-enable excludes PD on ROCm ("PD on ROCm has not been run with it"); explicit
  `--enable-decoder-swa-bounded-replay` is still validated against the incompatibility list (l.428-462).
- l.50-65: MegaAttention (`flashmla_mega`) default only on CUDA SM100; ROCm resolves `flashmla`. PD handshake checks KV layout
  (sglang #45075, merged) -> prefill and decode must agree on `--dsv4-attn-backend` / `SGLANG_DSV4_KV_LAYOUT` [1st: PR #45075; cookbook 3.4].
- Cookbook 3.4: "PD runs with DSpark ... prefill and decode servers must both pass [speculative flags]"; HiCache card "not offered on
  MI355X: hierarchical cache has not been validated on ROCm for this model yet" (3.5).
- From PR #44031 body (open): "DeepSeek-V4.1 DSpark PD needs the same TP size on both servers (a check on main)" [1st: PR body; code
  location not checked by me].

### 4.2 DSv4.1 PD on MI355X in SGLang - only correctness evidence, no throughput
sglang #44031 (open, AMD, FP8 unified KV for V4.1 with DSpark/PD/UMBP), 1P1D over MoRI, `--speculative-algorithm DSPARK
--speculative-dspark-block-size 5 --cuda-graph-backend-prefill disabled --fp8-gemm-backend aiter --kv-cache-dtype fp8_e4m3
--max-running-requests 32 --mem-fraction-static 0.70`, image `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261025` + branch:
P TP2 / D TP2 one node: GSM8K 96.51%, AL 3.50; P TP4+CP4 / D TP4: 96.29%, AL 3.49. Body also states "With paged KV, PD over MoRI
on current main can answer long prompts wrong when OPUS sparse prefill (SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL) is on"; "Decode with TP8
and no DP attention is blocked by the aiter fp8 decode kernel (no gqa=8 instance)" (for the unified layout) [1st: PR #44031 body].
Local AMD material: /perf_apps/yangwu/dsv4-3045-clean3p3d (3P3D runs 3077/3085) - REVIEW.md shows throughput "missing" and
decode faults pending review; no usable numbers [1st: file read]. /perf_apps/yanfwang/dsv41_pd_validate_1029 validates two ROCm
kernel fixes (64-bit candidate-block offsets, bounded prefill-schedule JIT) on image `sglang-0.5.22-rocm10-mi35x-86f72e5-dsv41fix2`;
no PD throughput [1st: validate.sbatch].

### 4.3 GB300 Dynamo + SGLang PD recipe for DSv4.1-Flash (published, the only DSv4.1 SGLang PD recipe)
File `benchmarks/multi_node/srt-slurm-recipes/dsv41flash/sglang/gb300-fp4/agentx/disagg-dep16-hicache-variants.yaml` [1st].
Prefill (per worker, DEP2, 2 GPUs): `tensor-parallel-size 2, data-parallel-size 2, expert-parallel-size 2, enable-dp-attention,
enable-dp-lm-head, enable-dp-attention-local-control-broadcast, moe-a2a-backend megamoe, dsv4-attn-backend flashmla_mega,
disaggregation-transfer-backend mooncake, chunked-prefill-size 65536, max-prefill-tokens 32768, stream-interval 32,
max-running-requests 128, swa-prefix-tails 2048, enable-decoder-swa-bounded-replay, enable-encoder-swa-bounded-replay,
enable-hierarchical-cache, hicache-ratio 1.2, hicache-mem-layout page_first_direct, hicache-io-backend direct,
hicache-write-policy write_back, mem-fraction-static 0.85, speculative-algorithm DSPARK, speculative-dspark-block-size 5,
speculative-num-steps 5`; env `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1, SGLANG_DSV4_KV_LAYOUT=v41_new,
SGLANG_OPT_DSV41_BF16_INDEX_LOGITS=1, SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1, SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True, MC_FORCE_MNNVL=1,
SGLANG_DISAGGREGATION_{BOOTSTRAP,WAITING}_TIMEOUT=3600, SGLANG_ENABLE_PREFILL_WAR_READ_DONE=1`.
Decode (one DEP16 worker, 4 nodes): `tp/dp/ep 16, enable-dp-attention, enable-dp-lm-head, enable-fp32-lm-head, moe-a2a-backend megamoe,
dsv4-attn-backend flashmla_mega, disaggregation-transfer-backend mooncake, disaggregation-decode-extra-slots 256, stream-interval 32,
swa-prefix-tails 2048, enable-decoder-swa-bounded-replay, mem-fraction-static 0.75, enable-eplb, ep-num-redundant-experts 16,
load-balance-method total_tokens, DSPARK block 5`; env `SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS=1, SGLANG_OPT_DEEPGEMM_MEGA_GATE=1`.
Frontend: Dynamo KV router, session-affinity TTL 14400 s on prefill, round-robin on decode. Variants: 7P c5540 (decode
max-running 1536, cuda-graph-max-bs-decode 96), 9P c7480 (3072 / 192, total_requests), 6P + DEP32 c3018 (576 / 18, 32 redundant
experts, `SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1`). Changelog notes: Mooncake external linker for prefill L3 was replaced by local
host-DRAM HiCache because "layer-wise L3 reads block the forward pass and stall prefill after the client's turn burst" (#4595);
decode batch cap set "at the rank's equilibrium batch so the queue moves to TTFT instead of decode step time" (#4673) [1st: changelog].

### 4.4 Retired MI355X SGLang PD recipe (DeepSeek-V4-Pro-0813, MoRI + UMBP) - closest AMD PD template
Saved at `research/data/retired_dsv4pro_mi355x_sglang_disagg-umbp-dspark_at_1f87891d.yaml` (InferenceX commit 1f87891d, parent of the
#4664 retirement). Key base settings [1st: that file]: `sglang-router` `policy consistent_hashing, dp-aware, decode-policy round_robin,
cache-threshold 0.3`; UMBP standalone server on the prefill node (`UMBP_DRAM_CAPACITY 1.5e12`, 2 MiB hugepages);
both roles `disaggregation-transfer-backend mori, disaggregation-ib-device rdma0..7, kv-cache-dtype fp8_e4m3, attention-backend dsv4,
page-size 256, swa-full-tokens-ratio 0.1, speculative-algorithm DSPARK, speculative-num-steps 1, context-length 1048576`;
env `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton, SGLANG_MORI_QP_PER_TRANSFER=4, SGLANG_MORI_NUM_WORKERS=4,
MORI_IO_QP_MAX_SEND_WR=32767, MORI_SHMEM_MODE=ISOLATION, SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32, SGLANG_ENABLE_SPEC_V2=1`;
prefill eager (`disable-cuda-graph: true`) with `enable-unified-cache-external-linker` (mori) and `optimistic-prefill-attempts 2`;
decode with CUDA graph and simulated acceptance env. Per-point: c1 P TP4/D TP4 block 6 with `SGLANG_DSV4_UNIFIED_KV_FP8=1`;
c16-48 P TP4/D TP8; c192-512 prefill DEP8 MegaMoE (`--moe-a2a-backend megamoe`, `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1`,
`MORI_SHMEM_MODE=STATIC_HEAP` 8G) and decode TP8+DPA (changelog #3664, #4186). Lessons recorded in changelog: prefill and decode must
carry identical speculative flags (v0.5.15 hard-fails "state component count mismatch"; v0.5.14 silently corrupted EAGLE verify,
GSM8K 0.96 -> 0.88) (#2305); router request timeout 14400 because an abort "landing mid-RDMA-write strands the mori TransferStatus and
wedges that prefill DP's transfers permanently" (#3664); decode must get `--decode-policy round_robin` (#3256/#3664) [1st: changelog].
This is DeepSeek-V4 Pro (V4 KV layout, compress ratios 4/128); V4.1 uses ratios 1/2 [1st: PR #44031 body] - flags will not all carry over [inf].

### 4.5 Other SGLang PD-relevant PRs for V4.1 (status 2026-11-19) [1st: gh search]
Merged: #43734 NIXL admitted for DSpark PD; #45075 MegaAttention default + KV layout in PD handshake; #44623 V4.1 external linker pools.
Open: #44031 (AMD FP8 unified KV + DSpark + PD + UMBP, adds MoRI allowance - note main already lists mori), #45265 encoder-only prefill
for PD (Mooncake only), #46727 encoder-only decode tails through DSpark verify, #40567 DSpark dynamic verify on PD decode,
#46576 CPU-buffer PD (NIXL, H20), #43465 AMD prefill context parallel, #44013 AMD interleave prefill CP + breakable graphs,
#50111 HiCache page_unified for V4/V4.1 pools.

---

## 5. SGLang levers that mattered for DSv4/DSv4.1, and vLLM equivalents

vLLM column sources: InferenceX vLLM MI355X recipe `benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`
and amd-master.yaml comments (main db3fc796), vLLM tracker vllm-project/vllm#66587 [1st]. "unknown" = not checked in vLLM source.

| SGLang lever | Evidence / number | vLLM equivalent |
|---|---|---|
| DSpark block 5 (`--speculative-algorithm DSPARK`) | scored at golden AL 3.51 (thinking on); cookbook: step cost does not move with block size | yes: `speculative-config {"method":"dspark","num_speculative_tokens":5,...}`; vLLM also has adaptive verification (#58151, +3.5-13.2% vs fixed 5 at c16-c256, TP4 EP) - disabled in the InferenceX ROCm recipe |
| AMD monokernel (sgl-amd-mono) for <= 48 rows | +11.4% tput / -12% TPOT (#44353); P90 +45-56% (LMSYS blog); TP4 c1 P90 641 -> 782.6 (#5034 local) | yes: `--monokernel` (vllm #60397/#61082/#62131), <= 48 rows, EP off |
| DP attention (TP2 -> DP2) at c32-c64 | same image: DP2 c64 178.2k@71.9 vs TP2 c64 170.9k@58.5 (#5034 local) | yes (`dp-attn: true` arms in vLLM keys) |
| DP4 + EP4 at c256 | 229.7k@78.4, +24.8% over published top (#5034 local) | EP: unknown for ROCm vLLM DSv4.1 at DP4; InferenceX #5098 (open) title "DEP2 (MegaMoE)" for vLLM MI355X |
| `SGLANG_DP_SHARED_EXPERT_LOCAL=1`, `SGLANG_SHARED_EXPERT_TP1`, gatherv / reduce-scatter DP collectives | used in all DPA points; no isolated number found | unknown |
| `--prefill-decode-interval` + `--schedule-policy lpm / shortest-prefill-first` + prefill delayer | trades TTFT for P90 ITL (#3926); no isolated tput number | partial/unknown: InferenceX vLLM PRs #4873 (open, "prefill delayer") and #4956 (closed, "prefill cadence") imply a vLLM-side mechanism; not code-checked |
| `--max-running-requests` below conc | TP2 c16 cap 2: 471.9@51.9k vs 336.7@57.3k (#5096 local) | yes: `max-num-seqs` caps used in vLLM recipe |
| Engram tables on host (`--engram-config placement cpu`, per_node replication) | frees HBM for KV; no isolated number | yes: `engram-config {"cpu_offload": true}` at TP2 |
| Decoder SWA bounded replay | default/required in recipes; PD on ROCm not run with it (code) | yes on ROCm: vllm #59039 (amd-master comment) |
| `--swa-prefix-tails` sizing (128..4096) | per-conc table; no isolated number | unknown |
| fastokens tokenizer | TTFT -26 to -29%, tput +2.2-2.6% (#47923) | unknown; vLLM recipe sets `VLLM_USE_RUST_FRONTEND=1` (not the same thing, equivalence not checked) |
| FP4 indexer row-group, INDEX_K_PAGE8, OPUS sparse prefill, FlyDSL MXFP8 | small % each (2.6) | vLLM: MXFP4 sparse-logits indexer (#58671), MXFP8 GEMM (#58510), FlyDSL dense MXFP8 (#67377); 1:1 mapping unknown |
| QuickReduce INT8 / INT4 | #46850 +1.6-2.7%, #49280 +3-3.9% | yes: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (recipe sets NONE); FlyDSL INT4 QR (#59478) |
| cache_aware dp-aware router keyed by correlation id | used for DPA points; no A/B number | vLLM keys use `vllm-router 0.1.14`; policy equivalence unknown |
| HiCache host tier (PD prefill, GB300) | 1P1D c160 187k@99 with HiCache (GB300); not validated on ROCm for V4.1 (cookbook) | KV offload: UMBP (#64715) published on MI355X agg (c256/c384); SimpleCPU / Mooncake on B200 |
| MoRI KV transfer for PD | allowed for DSv4.1 DSpark PD on main (code l.413) | unknown here (vLLM dimension) |
| MegaMoE prefill DEP8 (Pro), MegaAttention / MegaGate (CUDA) | Pro MI355X PD c192-512; GB300 PD | unknown on ROCm vLLM |
| Encoder-only prefill for PD (#45265, open) | GB300: +5.6% req/s at c160, TTFT 13.7 -> 1.8 s; tie at max tput | unknown |

Null result worth recording: sglang #48152 (indexer TP row shard) on AgentX TP2 c32/c64 MI355X, 2 runs per arm: tok/s/GPU ratio
1.0008 / 1.0017, P90 ratio 1.0033 / 1.0000 [1st: my run of /perf_apps/yanfwang/dsv41-idxshard-agentx-20261103/scripts/summarize.py].

---

## 6. AKLR / agent_sys knowledge base

- AKLR micro-bench: `AMD-AGI/SIKL` main, `aklr/benchmarks/models/deepseekv4/{README.md, model_bench.py, profile_decode.py,
  profile_prefill.py}` [1st: gh api contents]. Supports DeepSeek-V4-Flash and V4-Pro only (V4-Flash: 43 layers, 256 experts top-6,
  indexer top-k 512); runtime target `lmsysorg/sglang:v0.5.19-rocm724-mi35x`; rejects `--speculative-algorithm`; forces
  `--disable-cuda-graph` unless `--full-step`; `AITER_BF16_FP8_MOE_BOUND=16`. DeepSeek-V4.1-Flash (40 layers, 384 experts per cookbook)
  is NOT in the README's support list [1st: README "other architectures are rejected before loading weights"]. `gh search code` for
  `deepseek_v41` / `V4.1` in AMD-AGI/SIKL returned nothing (index may not cover the repo).
- Knowledge repo (/home/yihou/.cache/infera-agent-sys/knowledge): no DSv4.1-specific handoff. Mentions: AKLR export stats 2026-09-29
  "DeepSeek-V4.1 Flash 25 defs, 0 solutions" (kbk-tool-perop-microbench README:179); PD bottleneck-finder worked example is DSv4-Pro
  8k/1k 1P1D/2P1D on MI355X - "its numbers do not transfer" (pdx-bottleneck-localization README:160,202); capacity-probe handoff notes
  "the PD bootstrap only checks page_size, kv_cache_dtype and one DSV4.1 layout" (pdx-capacity-probe-truncation README:227) [1st: grep].

---

## 7. Gaps

- No DSv4.1-Flash SGLang PD throughput on MI355X anywhere found (InferenceX main/changelog/open PRs, sglang PRs/issues/cookbook,
  local /perf_apps dirs listed in the brief).
- InferenceX #5034 / #5096 numbers are author-local runs; the staged run 43712971238 data were not fetched (staging site only).
- Pro MI355X PD vs agg Pareto at equal interactivity not computed.
- LMSYS low-latency blog read only partially (HTML extraction); 331.6 figure confirmed only to "331." by me, full value from 2nd-hand.
- sglang release notes v0.5.23 not read; AMD ROCm blogs for DSv4.1 not found (searched WebSearch once).
- vLLM-equivalence column: most "unknown" entries need a vLLM source check (other teammate's dimension).
- "Same TP size on both PD servers" check location in sglang main not located by me.
- Whether MoRI-based PD + DSpark + DP-attention on MI355X for V4.1 actually runs end-to-end on main is unverified (code allows it;
  no run evidence beyond #44031's TP2/TP2 and TP4/TP4 GSM8K).

---

## Verification (adversarial, 2026-11-19)

Independent re-check of the 12 headline claims. Sources re-fetched by the verifier: InferenceX public API
(`/api/v1/benchmarks?model=DeepSeek-V4.1-Flash` and `?model=DeepSeek-V4-Pro`, fetched 2026-11-19 ~08:20Z), InferenceX main
`db3fc796` (`inferencex-e2e/configs/amd-master.yaml`, `inferencex-e2e/perf-changelog.yaml`, golden AL yaml, GB300 recipe),
InferenceX PRs #4664/#5034/#5096, sglang main `a735253e` `deepseek_v4_hook.py` + cookbook `DeepSeek-V4_1.mdx`, sglang PR #44031,
sglang issue #49327, vllm issue #66587, LMSYS blog 2026-10-18. Verifier scratch copies: `verify_sglang/`.

Confirmed (no change): claims 1 (no MI355X DSv4.1 SGLang PD; only agg key; Pro key `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`
retired by #4664, merged 2026-10-30T00:39Z), 2 (181,052 @ 73.3, run 43344632854), 3 (vLLM DP2 c128 271,329 @ 45.1; DP2 c384 427,451 @ 15.8
with kv-offload backend `umbp` 1.2.3.post1 per amd-master.yaml `dsv41flash-fp4-mi355x-vllm-agentic-dspark-umbp`), 5 (GB300 PD/agg numbers
and `override_7p1d_dep2_dep16_b96_c5540` / `override_9p1d_dep2_dep16_b192_c7480`), 6 (hook :377-424; note it additionally rejects
"the unified KV layout" (`is_unified_kv_triton`) and TRT-LLM + encoder SWA replay for V4.1), 7, 8, 9 (the "6-8 rejected" comment refers to
the AL-collection image `vllm/vllm-openai:deepseekv41-flash-0909`, TP4), 12 (Pro rows exact). The LMSYS blog figure 331.6 -> 516.0 tok/s
(1.56x, 4 sessions, 4x MI355X) is now confirmed first-hand (Gap list item "331 only partially" is resolved).

Corrected:

- **Claim 4 (#5034 "+24.8% over published top")**: the +24.8% baseline is NOT the current published top 181,052. #5034's table uses
  published TP2 DPA c64 = **184,058 @ 75.2** from run 39596624542 on the older image `v0.5.22-rocm10-mi35x-20261021`;
  229,660 / 184,058 = +24.8%. Against the current published top (181,052, v0.5.23 image) the same local point is +26.8% [inference: arithmetic].
  Other numbers confirmed: TP4 DPA EP4 c256 229,660 @ 78.4; TP2 DPA c64 178,230 @ 71.9; TP2 c64 170,887 @ 58.5 (all local, PR #5034 body);
  DP2 c128 239.8k @ 50.5 (local, PR #5096 body; local runs used sglang `6c29aa5954` + AITER `285579b6` plus two sglang patches not in that PR).
  Both PRs OPEN as of 2026-11-19.
- **Claim 10 (workload ranges)**: the ranges hold only for the MI355X **SGLang** rows (mean input 134,288-326,947 tokens; server GPU
  prefix-hit 0.9557-0.9808). Across all MI355X published DSv4.1-Flash rows (ATOM, vLLM, TileRT, SGLang) the ranges are mean input
  **113,745-326,947** and GPU prefix-hit **0.883-0.981** (low end: vLLM DP2+UMBP c384, run 43861172833). Input share ~99% of total holds
  (e.g. vLLM c384 424,074 / 427,451). [1st: InferenceX API re-fetch]
- **Claim 11 (monokernel "largest single SGLang lever at low concurrency")**: the cited numbers are confirmed (#44353: output tput +11.4%,
  TPOT -12.0%, 22-point random sweep geomean, TP4; blog: P90 per-user +45-56% at 1/4/8 sessions; vLLM `--monokernel` per #66587 rows
  #60397/#61082/#62131). The superlative is not supported as stated: #49327 also lists #47317 (draft) "Decode step -16% at batch 1, TP8
  (MI350X screen)", a larger single-PR figure at batch 1 on a different metric/screen. Treat "largest" as [inference], not first-hand.
