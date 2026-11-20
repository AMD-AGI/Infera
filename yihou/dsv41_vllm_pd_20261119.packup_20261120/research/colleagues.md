# Local / colleague prior work on DeepSeek-V4.1-Flash (read-only survey, 2026-11-19)

Scope: everything on local NFS about DSv4.1-Flash (and DSv4-Pro where it bears on PD), plus transferable lessons from
the sibling Muse vLLM project. Research only: nothing was run on a GPU, nothing outside our workspace was modified.
Evidence tags: `[fh: path]` = first-hand read of that file/log/result JSON; `[fh-cmt: path]` = first-hand read of a
comment/changelog entry written by someone else (the claim itself is theirs, i.e. second-hand content);
`[inference]` = arithmetic or reading on top of first-hand numbers; `[2h]` = second-hand.
Helper used to tabulate result JSONs (read-only): `scripts/colleagues/sum_results.py` in this workspace.

All AgentX numbers below are InferenceX/InferenceY `request_metrics.throughput.per_gpu.*` from 3600 s official-mode runs
(`semianalysis_cc_traces_weka_062126`, agentx-harness 1.0.12 where visible), hardware MI355X on the OpenAI `smci355-ccs-aus-*`
cluster, precision "fp4" (MXFP4 routed experts, MXFP8 dense), DSpark 5 drafts with simulated AL 3.51 (golden, thinking_on),
aggregated (single-server, not PD) unless stated. `tot` = (input + output) tok/s/GPU = the scored metric
(`per_gpu.total_tput_tps`) [fh: Muse research/agentx_harness.md §0.1, citing InferenceX request_metrics.py];
`comp` = prompt tokens actually computed (not cache hits) per GPU per second = `server_metrics.tokens.prompt_by_source.computed`
/ duration / GPUs [inference]; `hit` = overall prefix-cache hit rate from server metrics.

---

## 0. Bottom line (decision-relevant)

1. **No local DSv4.1-Flash PD (disaggregated) run exists, on any engine.** All DSv4.1-Flash AgentX results found are
   single-server aggregated TP2/TP4 (sglang, vLLM). The only PD AgentX artefact is a DSv4-**Pro** sglang 1P1D result that
   is not a valid score (1323/1723 OK, 450 s window) [fh: /perf_apps/yanfwang/InferenceY-c192/dsv4_p1x8ep8dpa_d1x8dpa_conc192_*.json].
   Searched: all dirs listed in §1-§7.
2. **Best local DSv4.1-Flash AgentX score: sglang TP2 + DP-attention 2, c64 = 182,689 tot tok/s/GPU** (1326 out tok/s/GPU,
   hit 0.966), image `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261025`, 2026-10-26 [fh: /perf_apps/yanfwang/InferenceX_CI/archive/slurm_job-3244/logs/dsv41flash_tp2_conc64_*dpatrue*.json].
   sglang TP2 (no DPA) c64 on v0.5.23-20261103 = 172.9-174.5k [fh: archive jobs 4561/4562/4591/4598].
3. **Best local DSv4.1-Flash vLLM AgentX score: TP2 c32 = 110.1-110.5k tot tok/s/GPU** (632-636 out), vLLM
   `0.33.0rc1.dev573+g40941f2c4` + yanfwang indexer-TP-shard build, 2026-11-04 [fh: archive jobs 4356/4357/4369/4396].
   Official CI vLLM TP2 c32 on `nightly-rocm100-7d0b4e57` = 107,989 (reproduced 107,583 by chaox) [fh: /perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/].
   **No local vLLM c64 measurement.** The Nov-3 InferenceY snapshot says vLLM TP2 c64 = 151,896 and c128 = 150,683
   "tok/s/chip" [fh-cmt: amd-master.yaml comment], and changelog #3571 says TP2 c64 = 111k total tok/s/GPU in run
   36528242520 [fh-cmt: perf-changelog.yaml] - two different numbers for the same arm on different images; neither is local.
4. At equal concurrency (c32, TP2) vLLM (110k) and sglang (112k) are within ~2% [inference from rows in §8]. The c64 gap
   (sglang 173-183k vs no local vLLM number) is unmeasured locally.
5. **Workload shape (from result JSONs):** ISL mean 135k (c64) - 177k (c32), OSL mean ~1.0k, prefix hit 0.96-0.98,
   computed prompt tokens only 3.1k (c32) - 8.9k (c64) tok/s/GPU [fh/inference: §8]. The scored metric is >99% input tokens
   [fh: job 4572 per_gpu input_tput_tps 111,814 of total 112,468 = 99.4%].
6. **Prefill and decode capacity probes (sglang, not AgentX):** single-request cold prefill TTFT TP2 262,144 tok = 4.39 s,
   1,040,000 tok = 19.6 s; TP4 262,144 tok = 2.95 s, 1.04M = 12.3-14.7 s [fh: dsv41-dense-indexer-20261104/e2e/logs/job_4509.out, job_4510.out].
   Decode step (sglang TP2+DP2, DSpark 5) is ~flat in context (B=16/rank: 24.5 ms at 16K -> 22.3 ms at 262K) and grows
   with batch (16K ctx: 14.6 ms at 4 req/rank -> 33.1 ms at 32 req/rank) [fh: dsv41_kvlen_bench_1028/analysis.txt].
7. **DeepInfer (DeepSeek's own engine) port to MI355X** (jihhe, EP8/DP8, synthetic benchmark, forced-uniform routing):
   prefill 79.9k tok/s/GPU at 32k ctx; decode 14.9 / 27.2 / 39.4 ms/step at 8 / 64 / 128 req/rank; open acceptance
   regression [fh: /perf_apps/jihhe/dsv41di/integ/amd/README.md "Latest numbers"]. Not AgentX, not comparable to §8.
8. **vLLM DSv4 PD on ROCm needed 3 MoRIIO patches** (hybrid per-layer block sizes, non-contiguous fp8_ds_mla KV register,
   sparse backend-name handshake), verified only on vLLM 0.23.x + DSv4-**Pro** 1P1D; the infera vLLM PD example uses
   MooncakeConnector with prefix caching OFF and 8k1k random data [fh: infera.dsv41/deploy/docker/patches/vllm-dsv4/*.py,
   examples/deepseek_v4/engine/pd_mooncake/vllm/engine.sh]. Whether these are still needed on a Nov-2026 nightly: not checked.

---

## 1. /perf_apps/yanfwang (Yanfei Wang) - the main body of DSv4.1-Flash work

No README/notes `.md` exists in any `dsv41*` dir [fh: find *.md]. Facts below come from scripts, configs, logs and result JSONs.
Cluster: OpenAI `smci355-ccs-aus-*` (partition `Compute-Group01`), Slurm + pyxis squash images, launched through InferenceY
`runners/manual_launch.py` [fh: dsv41-idxshard-agentx-20261103/scripts/run_lane.sh].

### 1.1 InferenceX_CI/ (archive of 69 Slurm jobs + squash images)
- `archive/slurm_job-<id>/` holds result JSON, server log (`*_agg_w0.out`), `sweep_<id>.log`, `sbatch_script.sh`,
  `<id>.json` (job manifest). Every DSv4.1 AgentX number in §8 comes from here (39 result JSONs).
- `squash/` images (reusable on the same cluster): `lmsysorg_sglang-rocm_v0.5.22-rocm10-mi35x-20261028`,
  `..._v0.5.23-rocm10-mi35x-20261101`, `..._v0.5.23-rocm10-mi35x-20261103`, `yfw_sglang-rocm_*-mono91{base,pr}`,
  `yfw_sglang-rocm_v0.5.23-...-20261101-idxshard`, `rocm_mori-dev_sglang-0.5.2{2,3}-...-dsv41fix*`,
  **`yfw_vllm-openai-rocm_nightly-40941f2c-idxshard-a6f60c99.sqsh`** (the only local vLLM DSv4.1 build) [fh: ls].
- Lesson: `config.yaml` in archive jobs is mode 0600 (unreadable to us); flags must be taken from the server log
  `non-default args` line [fh].

### 1.2 dsv41-idxshard-agentx-20261103 (sglang PR #48152 indexer TP row-shard A/B; AgentX)
- What: two lanes (A/B on two nodes), image `yfw/sglang-rocm:v0.5.23-rocm10-mi35x-20261101-idxshard` (= 20261101 release +
  sglang #42968 + #48152), TP2 c32/c64, recipe `dsv41flash/sglang/mi355x-fp4-mtp/agentic.yaml` with/without
  `SGLANG_OPT_DSV4_INDEXER_TP_SHARD` [fh: InferenceY-lane-A/inferencex-e2e/configs/yfw-idxshard.yaml].
- Result: on/off ratio of means 1.0005 (c32 out tok/s/GPU) and 1.0025 (c64), i.e. no throughput effect; n=2 per arm
  [fh: `python3 scripts/summarize.py` output over results/]. Numbers in §8 (jobs 4243-4314).
- Assets: `scripts/run_lane.sh` (queue-driven lane runner: one Slurm job per point via `manual_launch.py`, pins all arms
  of an A/B to the node of the first run, fast-fail rule, result copy with marker-file freshness check),
  `scripts/summarize.py` (result-JSON table + on/off ratios), `queues/lane{A,B}.txt`.

### 1.3 dsv41-dense-indexer-20261104 (HIP dense prefill indexer candidates C1-C4)
- What: kernel candidates for the DSv4.1 FP4 paged MQA-logits indexer (`c4_kernel_v2_hybrid.py`: row-group a wave,
  page-8 layout matching DSv4.1's 8-row candidate blocks), overlay patches in `e2e/patch/` (fp4_indexer_hip.py,
  topk_tilegate_hip.py, yfw_c4_*.py, yfw_rowgroup.py, low_ratio_backend_hip.py, deepseek_v4_backend_hip_radix.py),
  microbench `e2e/scripts/{mb.py,combo_mb.py}`, indexer input dumps at 1.04M context `profile/dumps_44{10,21}/*.pt`.
- TTFT (single request, prefix cache "shon" warm/measured, image `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261103`)
  [fh: e2e/logs/job_4509.out (TP4), job_4510.out (TP2)]:
  - TP2: 262,144 tok 4.39 s (stock) vs 4.35 s (arm 1234); 1,040,000 tok 19.58 s vs 18.57 s.
  - TP4: 262,144 tok 2.95 s vs 2.91 s; 1,040,000 tok 12.3-12.6 s (arms), a second 1234 batch 14.7 s.
  - First ("warm") request is much slower (TP2 262k: 44.4 s stock, 33.8 s arm 1234; 1.04M: 76.2 s stock) - JIT/compile on
    first use [fh]; mechanism not verified.
- bench8k (TP2, 4/8/32 requests, short run, not AgentX): conc 1 out 516 -> 754 tok/s, AL 5.17-5.65 (real acceptance),
  conc 16 813 vs 826 tok/s [fh: job_4510.out].
- Accuracy: needle 6/6 both arms at 1.04M; gsm8k 5-shot 0.914 (stock) / 0.910 (1234); 20-shot 0.915 / 0.921 [fh: job_4509.out].
- AgentX A/B stock vs all4 (TP2, c32/c64, same image): all 8 runs within 1% of each other (§8 jobs 4528-4598); which
  fingerprint is which arm was not resolved (dryrun dirs exist: agentx/dryrun/{stock,all4}-c{32,64}) [fh].
- Microbench: L2 dump 1.04M ctx, 16,384 rows: score+topk c4 9.26 ms vs c24 8.45 ms, set-equal 16383/16384 rows
  [fh: e2e/logs/mb_4518.out].

### 1.4 dsv41_kvlen_bench_1028 (decode step vs batch and context; sglang)
- Server: `server.sh` = sglang TP2 + DP-attention 2 with the production recipe `override_tp2_c64_dp` flags
  (DSPARK block 5, `--enable-decoder-swa-bounded-replay`, `--cuda-graph-backend-prefill disabled`,
  `--chunked-prefill-size 32768`, `--prefill-decode-interval 16`, `--swa-prefix-tails 4096`, `--max-running-requests 64`,
  `--mem-fraction-static 0.85`, host Engram table `per_rank`, `SGLANG_SHARED_EXPERT_TP1=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`,
  etc.), image `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261028`, model `/perf_apps/data/models/DeepSeek-V4.1-Flash`
  [fh: server.sh, bench.sbatch].
- Result table [fh: analysis.txt] (step ms p50 per DP rank; accept = real acceptance length):

  | B/rank | context | step ms p50 | KV tok/rank | accept |
  |---|---|---|---|---|
  | 16 | 16,384 | 24.50 | 277,248 | 5.29 |
  | 16 | 65,536 | 23.70 | 1,065,216 | 5.46 |
  | 16 | 131,072 | 22.60 | 2,113,536 | 5.74 |
  | 16 | 262,144 | 22.30 | 4,210,432 | 5.72 |
  | 4 | 16,384 | 14.60 | 69,888 | 5.38 |
  | 8 | 16,384 | 17.90 | 139,520 | 5.72 |
  | 8 | 524,288 | 17.70 | 4,203,264 | 5.47 |
  | 32 | 16,384 | 33.10 | 555,008 | 5.43 |

  Fits written by the author: B=16: step = 24.17 - 0.500 x (KV Mtok/rank); L=16K: step = 12.81 + 0.656 x req/rank.
- Reusable: `client.py` (prefill-then-decode two-phase probe with UTC markers), `analyze.py` (windowed per-rank step time
  from sglang `Decode batch` log lines) - sglang log format only.

### 1.5 dsv41_pd_validate_1029 (kernel-fix validation, not a PD run)
- Validates two ROCm fixes baked into `rocm/mori-dev:sglang-0.5.22-rocm10-mi35x-86f72e5-dsv41fix2-20261029`:
  (1) 64-bit row offsets in candidate-block kernels - "decode logits of a bs=384 DSpark verify graph on a 1M-context
  server ([2304, 1052672] fp32) fault from row 2041"; (2) bounded JIT variants of `_prefill_schedule_prep_kernel` and
  aiter `_prefill_row_plan_kernel` (ROCm/aiter#5925) [fh: validate.sbatch, validate_cand64.py docstring].
- Outcome: **all 5 checks failed before running** - `RuntimeError: HIP_VISIBLE_DEVICES contains more devices than
  ROCR_VISIBLE_DEVICES` (and pytest collection errors) [fh: logs/*.log, job_3619.out]. Lesson: when setting
  `ROCR_VISIBLE_DEVICES` per process, unset `HIP_VISIBLE_DEVICES` (kvlen server.sh does exactly that) [fh].
- PD relevance [inference]: a large decode batch (384 x 6 verify rows) at 1M context hit an int32 offset overflow on the
  sglang path; any PD decode worker with a big batch should be checked for the same class of bug on vLLM.

### 1.6 dsv41-idxshard-sdma-20261104, dsv41-upstream-20261105, dsv41-ruler
- `sdma_allgather.py`: start/wait all-gather on mori SDMA queues (prototype for the indexer row-shard gather). Docstring:
  one queue per peer saturates ~61 GB/s XGMI; 2/4 queues 1-15% slower; start+wait ~13 us host [fh-cmt: docstring].
  Also `image1103_src/` (pip.txt, sglang_git.txt, srt.tar of the 1103 image), `mori866*` builds.
- `dsv41-upstream-20261105/`: mori builds + `wheels/flydsl-0.4.2-cp312-*.whl` [fh: ls].
- `dsv41-ruler/`: RULER long-context sets (cwe/fwe/niah_multikey...) at 131,072 / 262,144 / 524,288 / 1,040,000 tokens
  (`data/*.jsonl`, `*.ids`) [fh: ls]. No result file found.

### 1.7 InferenceY-c192 (DSv4-Pro sglang PD, not DSv4.1)
- Result JSON: DSv4-Pro-0813, sglang-disagg, 1P (TP8 EP8 DPA) + 1D (TP8 DPA), UMBP-linker DRAM KV offload, mori, c192:
  21,582 tot tok/s/GPU over 16 GPUs, out 166/GPU, intvty p50 104, TTFT p50 4.49 s, hit 0.893, **1323 of 1723 requests
  successful, duration 450 s** -> not a valid official score [fh: dsv4_p1x8ep8dpa_d1x8dpa_conc192_*.json, 2026-10-01].
- `KLAUD_DEBUG.md`: playbook of InferenceX recipe-bump CI failures (changelog append-only rule etc.) [fh].

### 1.8 InferenceY snapshots (configs as of ~2026-11-03..05) - pointers only, upstream is covered elsewhere
Paths: `dsv41-idxshard-agentx-20261103/InferenceY-lane-{A,B}/inferencex-e2e/`, `dsv41-dense-indexer-20261104/agentx/InferenceY-lane-{A,B}/inferencex-e2e/`.
- `configs/amd-master.yaml` keys `dsv41flash-fp4-mi355x-{vllm,sglang,atom}-agentic-dspark` (aggregated only, no
  DSv4.1-Flash MI355X PD key); DSv4-Pro PD keys `dsv4-fp4-mi355x-sglang-disagg-agentic-umbp-dspark`,
  `dsv4-fp4-mi355x-atom-disagg-agentic-lmcache-dspark` [fh].
- vLLM MI355X recipe `benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`: moe-backend
  aiter, block-size 128, attention-config `{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}`, monokernel,
  DSpark 5 probabilistic (adaptive verification off: "trims requests on device, which the ROCm DeepSeek-V4 indexer does
  not support"), env `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True`,
  `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `VLLM_USE_RUST_FRONTEND=1`; TP2 overrides use Engram `cpu_offload:true`;
  TP2 c32 = `max-num-batched-tokens 8192, max-num-seqs 8`; TP2 c64 = `16384`, no seq cap [fh].
- Multi-node DSv4/DSv4.1 disagg recipes for other HW exist (e.g. `multi_node/srt-slurm-recipes/dsv41flash/vllm/gb300-fp4/agentx/disagg-variants.yaml`,
  `dsv4/vllm/gb200-fp4/agentx/disagg-1p1d-dep8-dep8-c{32..384}-dspark{5,6}.yaml`) - the only vLLM PD AgentX recipes seen
  locally, all NVIDIA [fh: find].
- Changelog facts relevant to vLLM on MI355X [fh-cmt: perf-changelog.yaml, PR links given there]:
  `--no-swa-bounded-replay` once needed (HSA memory fault at first prefix hit with replay start, #3326), later removed
  when vLLM #59039 landed (#3959); `VLLM_USE_BREAKABLE_CUDAGRAPH=1` required (#2962); adaptive verification unsupported
  by the ROCm indexer (#2962); TP2 needs Engram tables on host (94.4 GiB/rank resident) (#3326); indexer buffers scale
  ~4.4 MiB per batched token (#3326); mono decode only for steps <= 48 rows (= c8 with 5 drafts) (#3690/#3959).

---

## 2. /perf_apps/chaox (Chao Xu) - vLLM TP2 c32 baseline + kernel extraction

- `run_DeepSeek-V4.1-Flash-vllm.sh`: a dev-container launcher (docker run + sshd for Cursor), image
  `vllm/vllm-openai-rocm:nightly-rocm100-5555a2d8...`, model `/perf_apps/chaox/models/DeepSeek-V4.1-Flash`. No serve/bench
  logic. Note it `docker rm -f`s a same-name container - do not reuse as is [fh].
- `DeepSeek_V4.1_Flash/conc32/baseline/`: **reproduction of official CI job 114309765517** (vLLM TP2 c32, image
  `nightly-rocm100-7d0b4e57`, 2026-10-14): output 1270.24 vs 1263.1 tok/s (server total, 2 GPUs), intvty p50 163.7 vs 160.7,
  TPOT p50 6.11 vs 6.22 ms, mean ISL 173,006 vs 173,270, KV cache 28,905,651 tokens, hit 0.960 -> "REPRODUCED within 2%"
  [fh: BASELINE_REPRO.md; result JSONs tabulated in §8]. Assets: `launch_server.sh` (exact CI vLLM args/env),
  `run_bench.sh` (InferenceX `srt_agentic.sh` @ 827e6c2 with the full env contract), `image_env.sh` (restores image ENV
  for ssh shells), `official_ci/` (CI artifacts + server logs).
  Gotcha: HIP device index != rocm-smi index on that node [fh-cmt].
- `conc32/discovery/`: `SOURCE_MAP.md` (27 kernel families, 147 source files with paths in the image), `CRITIQUE.md`,
  `CAPTURE_PLAN.md` + `capture_server.sh` (profiling plan) [fh]. Key first-hand-in-their-doc facts:
  - Layer roles: layers 0-1 SWA only; 2/8/14 ratio-2 compressor + dense MXFP4 indexer; 20 ratio-1 candidate writer
    (top-2048 blocks); 24/28/32/36 candidate consumers; 40-42 DSpark draft (SWA, 128 experts top-3) [fh-cmt: SOURCE_MAP.md].
  - Mono decode gate: decode-only steps <= 48 padded rows (<= 8 requests x 6), FULL graph only; mixed prefill+decode
    steps run PIECEWISE on the regular path; quick-reduce INT4 at >= 108 rows (>= 17 requests); aiter indexer top-k when
    rows >= 128 [fh-cmt: SOURCE_MAP.md, CRITIQUE.md, with vLLM file:line refs].
  - CI c32 server metrics: num_requests_running p50 = 8, mean 8.8, p90 16 [fh-cmt: CRITIQUE.md B.1].
- `DeepSeek_V4.1_Flash/kernels/`: UT packages extracted from vLLM TP2 **conc4** serving (image `nightly-rocm100-7d0b4e57`,
  vLLM 0.31.1rc1.dev253): mono_layer_r{1,2}, mono_ffn, mxfp8_block32_gemm, sparse_mla, fused_moe_a8w4, quick_allreduce_int4,
  each with GPU-time share. At conc4: decode 93.1% / prefill 6.9% of GPU time; prefill shares: fused_moe_a8w4 25.95%,
  sparse_mla 17.63%, mxfp8 GEMM 14.78%, quick-allreduce 13.06% [fh-cmt: kernels/README.md]. Also `vll_kernels.tar.bz2`.
- Models: `/perf_apps/chaox/models/DeepSeek-V4.1-Flash` (48 safetensors + README, inference/, tech report) [fh: ls].
  A second copy is at `/perf_apps/data/models/DeepSeek-V4.1-Flash` (used by yanfwang's kvlen bench) [fh: ls, bench.sbatch];
  also `/perf_apps/data/models/DeepSeek-V4-Flash`, `DeepSeek-V4-Pro(-0813)`.
- Other: `agentx-kernel-extraction-standalone/`, `RP_CHECKLIST_SKILL.md`, `GEAK/` (not inspected in depth).

## 3. /perf_apps/jihhe/dsv41di (DeepInfer on MI355X)
- `integ/` = DeepSeek's DeepInfer engine (open-sourced; supports DSv4.1-Flash; DSpark, SWA bounded replay "YoCo",
  attention DP / balanced CP, EP+EPLB) ported to gfx950 by replacing every SM100 kernel lib with same-name shims that
  resolve to `nacre_di` kernels or torch references [fh: integ/README.md, amd/README.md].
- State 2026-10-23: full model serves at EP8/DP8 on one node with CUDA graphs, DSpark nextn 5, 1M max_model_len.
  L5 (`examples/benchmark.py`, 32k ctx, random prompts, forced-uniform routing, nacre 4756bd326) [fh: amd/README.md]:

  | req/rank | prefill tok/s/GPU | decode ms/step | acceptance |
  |---|---|---|---|
  | 8 | 79,951 | 14.9 | 72.7% |
  | 64 | 79,925 | 27.2 | 71.8% |
  | 128 | 53,151 (steady 79.9k) | 39.4 | 77.0% |

  Real routing at 64 req: 35.6 ms/step. Open: acceptance regression 94.3% -> 76.9% between nacre revisions [fh].
- `results/agentx_vs_ours_1022*.png` + `plot_agentx_vs_ours.py`: a comparison chart vs AgentX numbers (not opened).
- `eplb-dg/`: EPLB code (not inspected).

## 4. /perf_apps/lihuzhan/dsv4_4n_trace
- Training (Primus/Megatron) profile of DeepSeek-V4-Flash BF16 on 4 x 8 MI355X (PP4, EP8), one iteration trace (163 MiB)
  + logs + config [fh: MANIFEST.md, RECIPE.md]. Not inference; only useful for op-shape cross-reference.

## 5. /perf_apps/yangwu/dsv4-3045-clean3p3d (DSv4-Pro sglang 3P3D investigation)
- Two attempts at a clean 3 prefill + 3 decode node rerun with image `rocm/mori-dev:sglang-v3-direct-get-mrsplit-20261020`,
  DSv4-Pro-0813, max-running 96: run_3077 `FAILED_OR_INVALID` (docker run exit 125 on the decode node), run_3085
  `FAILED_ALLOCATION_HELD` ("GPU already in use" by a foreign container on a prefill node) [fh: run_*/validation.json].
  No perf numbers. REVIEW.md: goal was to test whether external concurrent jobs triggered a GPU fault [fh].
- Lesson [fh]: their orchestrator records pre/postflight GPU ownership, docker events and kernel logs per node - a model
  for shared-cluster PD hygiene (`run_3085/*flight_*.json`, `ownership_*.jsonl`).

## 6. infera repo (this worktree) - DSv4 assets
- `examples/deepseek_v4/` (DSv4-**Pro** reproduction kit, MI355X): mix (atom/sglang/vllm), `pd_mooncake/{atom,sglang,vllm}`,
  `pd_mori/sglang`; vLLM PD is 1P1D TP8 only, `MooncakeConnector` producer/consumer, `--no-enable-prefix-caching`,
  `--disable-hybrid-kv-cache-manager`, `--kv-cache-dtype fp8`, `--max-model-len 9472`, random 8k1k bench; cold start ~30 min
  [fh: README.md, engine/pd_mooncake/vllm/engine.sh]. No AgentX numbers in the repo.
- `deploy/docker/patches/vllm-dsv4/`: active `patch_moriio_dsv4_{hybrid_blocksize,noncontig_register,sparse_backend}.py`
  (vLLM 0.23.x, DSv4-Pro MoRIIOConnector 1P1D) + `patch_aiter_flydsl_moe_memref_bufres.py`; `legacy/` (upstreamed).
  Upstream status "none found" for all three MoRIIO patches [fh: deploy/docker/patch.upstream.status.md L339-346].
- `infera/engine/dsv4_gfx942.py`, `manual/wip/mi325-deepseek-v4.md` (gfx942 support contract: FP4 DSv4 on vLLM only),
  `manual/examples/sglang_{1p1d,mix}_dsv4.md`, `scripts/dsv4_parity.py`, projection configs
  `infera/projection/configs/models/megatron/deepseek_v4_flash.yaml`, `speculative/defaults/dspark.yaml` [fh: grep -l].

## 7. Sibling Muse vLLM project - transferable lessons
Source: `/home/yihou/dev/git.16-19/infera.muse/yihou/muse_vllm_20261118/{CLAUDE.md, research/agentx_harness.md,
research/SYNTHESIS.md, working_process.md}` [fh]. Muse is a different model; only engine/harness lessons transfer.

Harness contract (vLLM):
- Score = `request_metrics.throughput.per_gpu.total_tput_tps`, engine-agnostic; GPU denominator = TP x PP x PCP from env
  (DCP adds none); for vLLM DP layouts `TP` env must equal total GPUs of the server or per-GPU is wrong [fh-cmt: AH §0, §3.9].
- Simulated acceptance on vLLM = `speculative-config` JSON `"rejection_sample_method":"synthetic","synthetic_acceptance_length":<AL>`
  (not env); disabled if `enable_adaptive_verification` true. DSv4.1 golden = AL 3.51 for DSpark 5 (thinking_on) [fh: our
  jobs' server args show exactly `{"method":"dspark","num_speculative_tokens":5,...,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51}`].
  vLLM synthetic draw is per-request min-variance; sglang is per-step batch-wide; same mean [fh-cmt: AH §1.3].
  Verify after a run: AL = 1 + d(vllm:spec_decode_num_accepted_tokens)/d(vllm:spec_decode_num_drafts).
- `AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:'`; GPU KV capacity is read only from server log line
  `GPU KV cache size: N tokens` placed in RESULT_DIR; vLLM has no `/server_info` unless `VLLM_SERVER_DEV_MODE=1`;
  use `vllm:cache_config_info` labels as the live gate [fh-cmt: AH §1.5, §3].
- Engine ids `vllm-disagg`, `dynamo-vllm`, `llmd-vllm` all map to vllm for synthetic acceptance [fh-cmt: AH §1.2] -> a
  vLLM PD run can use the same simulation mechanism; must be set on the decode (and draft-carrying) role.
- Fresh server per concurrency point; never flush caches mid-run; dataset HEAD gate `23f152f6...`; agentx-harness pin
  (1.0.12 locally vs 1.0.14 upstream) must be recorded [fh-cmt: AH §1.1, §6].
ROCm / vLLM pitfalls seen there:
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is effectively mandatory for non-compiled models (ROCm auto-list contains only
  `DeepseekV41ForCausalLM`, so for DSv4.1 it is auto on that image) [fh-cmt: vllm_code_map.md L42, SYNTHESIS B2].
- Any KV connector (SimpleCPUOffload or native Offloading) changed the draft attention path and broke fp8 draft KV
  ("Mixed FP8 and non-FP8 tensors"); workaround draft `kv_cache_dtype auto` [fh-cmt: working_process.md 07:00-07:04Z].
  Relevance [inference]: a PD KV connector (MoRIIO/Mooncake/NIXL) is also a KV connector - check DSpark draft KV dtype.
- Host KV tier = one mmap `/dev/shm/vllm_offload_*.mmap`; two tiers per node SIGKILLed (host OOM); rule <= 1 TiB per
  node [fh-cmt: working_process.md 07:56Z-08:07Z].
- Shared AITER_JIT_DIR between two servers raced (custom all-reduce .so "file too short", hang) - one JIT dir per server
  [fh-cmt: working_process.md 06:20Z].
- `prefill-schedule-interval` only acts in DP engine core; no shortest-prefill-first policy (fcfs|priority only) in that
  image [fh-cmt: SYNTHESIS §0.5].
- ROCM_AITER_FA context-loss bug with prefix caching + chunked prefill fixed by vLLM #59098 (11-16) - images older than
  that can be "faster and wrong" [2h: SYNTHESIS §0.1].
- Ops: crusoe no-idle policy (30 min all GPUs <=10% -> cancel); container runtime dirs not on NFS (exit 139); never
  `pkill -f` inside `bash -lc`; graph capture ~30 min [fh-cmt: CLAUDE.md].
- Reusable scripts (Muse paths, need DSv4.1 edits): `scripts/serve_vllm.py`, `agentx_run_vllm.py`, `client_launch_vllm.py`,
  `run_point*.sh`, `kit_dataset_gate.py`, `rsum.py`, `analyze_trace.py`, `prof_when_profiling.sh`, `mon_runs.sh` [fh: ls].

---

## 8. Every measured number found (AgentX, DSv4.1-Flash, MI355X, aggregated single server)

All rows: 3600 s profile, cc-traces-weka-062126, DSpark 5 synthetic AL 3.51. Date = result-file mtime (UTC).
`tot`/`out`/`comp` in tok/s/GPU. Path prefix `A=/perf_apps/yanfwang/InferenceX_CI/archive/slurm_job-`,
`C=/perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/`.

| path | date | engine / image | TP, DPA | conc | ok | tot | out | comp | hit | intvty p50 | TTFT p50 s | ITL p50 ms | ISL mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3244 | 10-26 | sglang v0.5.22-20261025 | 2, DPA | 64 | 9721 | **182,689** | 1326.3 | 8,852 | 0.966 | 99.9 | 0.669 | 10.01 | 135,279 |
| A3272 | 10-26 | sglang yfw ...20261025-all3f | 2, DPA | 64 | 9811 | 183,600 | 1331.0 | 8,899 | 0.966 | 99.3 | 0.691 | 10.07 | - |
| A3282 | 10-26 | sglang v0.5.22-20261025 | 2, DPA | 32 | 4587 | 112,029 | 652.2 | 4,208 | 0.974 | 197.6 | 0.573 | 5.06 | - |
| A3848/3860/3880/3889 | 10-31 | sglang yfw mono91 base/pr | 4 | 8 | ~1433 | 16,402-16,549 | 114.5-114.8 | ~430 | 0.974 | 613-633 | 0.34-0.36 | 1.58-1.63 | - |
| A3849/3867/3888/3892 | 10-31 | sglang yfw mono91 base/pr | 2 | 8 | ~1375 | 29,743-30,027 | 215.4-216.6 | 793-827 | 0.973 | 357-365 | 0.56-0.60 | 2.74-2.80 | - |
| A3907/3919/3934/3965 | 11-01 | sglang yfw mono91 base/pr | 4 | 16 | ~2715 | 29,306-29,377 | 189.1-190.8 | ~890 | 0.969 | 503-516 | 0.36-0.38 | 1.94-1.99 | - |
| A3933/3958/3975/3997 | 11-01 | sglang yfw mono91 base/pr | 2 | 16 | ~2535 | 54,186-54,292 | 342.8-343.3 | ~1,630 | 0.970 | 234-245 | 0.84-0.91 | 4.08-4.28 | - |
| A4243/4244/4309/4310 | 11-03 | sglang yfw 20261101-idxshard off/on | 2 | 32 | ~4570 | 111,785-112,429 | 650.3-653.1 | ~3,090 | 0.972 | 198-199 | 0.53-0.56 | 5.02-5.05 | - |
| A4267/4274/4313/4314 | 11-03 | sglang yfw 20261101-idxshard off/on | 2 | 64 | ~9170 | 170,300-172,027 | 1237-1251 | ~6,200 | 0.964 | 80-82 | 0.33-0.34 | 12.3-12.5 | - |
| A4356/4396 (idx off) | 11-04 | **vLLM 0.33.0rc1.dev573+g40941f2c4** yfw idxshard build | 2 | 32 | 4467/4444 | 110,495 / 110,103 | 635.8 / 631.8 | ~2,590 | 0.976 | 218.6 / 214.9 | 0.34 / 0.51 | 4.57 / 4.65 | 177,236 |
| A4357/4369 (idx on) | 11-04 | same | 2 | 32 | 4453/4467 | 110,322 / 110,414 | 632.9 / 635.9 | ~2,585 | 0.976 | 217.5 / 219.9 | 0.36 / 0.33 | 4.60 / 4.55 | - |
| A4528/4529/4572/4573 | 11-05 | sglang v0.5.23-20261103 stock vs dense-idx all4 | 2 | 32 | ~4575 | 111,903-112,468 | 651.8-654.8 | ~3,090 | 0.972 | 199-203 | 0.52-0.56 | 4.93-5.04 | 176,029 (4528) |
| A4561/4562/4591/4598 | 11-05 | same | 2 | 64 | ~9280 | 172,905-174,535 | 1254-1266 | ~6,280 | 0.964 | 81.5-83.5 | 0.33 | 12.0-12.3 | 134,672 (4591) |
| C official_ci/bmk_result | 10-14 | vLLM nightly-rocm100-7d0b4e57 (CI job 114309765517) | 2 | 32 | 4476 | 107,989 | 635.1 | 4,362 | 0.960 | 163.7 | 0.259 | 6.11 | 173,006 |
| C baseline/results | 10-14 | same, chaox repro | 2 | 32 | 4464 | 107,583 | 631.6 | 4,351 | 0.960 | 160.7 | 0.196 | 6.22 | 173,270 |

Notes on the table:
- vLLM jobs 4356-4396 server args [fh: A4356/*_agg_w0.out "non-default args"]: TP2, block 128, gpu-mem 0.9,
  max-num-batched-tokens 8192, **max-num-seqs 8**, capture 1024, monokernel, moe aiter, indexer mxfp4 + sparse logits,
  Engram cpu_offload, max-model-len 1M; GPU KV cache 52.3-52.6M tokens; engine init 886-891 s.
  Arm difference only `indexer_tp_shard_min_rows` 1e9 (off) vs 512 (on).
- Non-AgentX numbers (TTFT probes, decode-step probe, DeepInfer L5) are in §1.3, §1.4, §3 and are not comparable to this table.
- Invalid / not DSv4.1: InferenceY-c192 DSv4-Pro PD (§1.7). No other DSv4/DSv4.1 result JSON found in the searched dirs.

## 9. Reusable assets (paths)

| asset | path | what / caveat |
|---|---|---|
| vLLM DSv4.1 serve args (CI-exact) | /perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/baseline/launch_server.sh | TP2 c32, image nightly-rocm100-7d0b4e57 |
| AgentX client env contract | /perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/baseline/run_bench.sh | InferenceX srt_agentic.sh @827e6c2 |
| InferenceX bench tree (Oct) | /perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/bench_tools/InferenceX | stale vs upstream |
| vLLM kernel source map + profiling plan | /perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/discovery/{source_map,capture_plan,CRITIQUE.md} | file:line refs into the image |
| Kernel UT packages (vLLM conc4) | /perf_apps/chaox/DeepSeek_V4.1_Flash/kernels/ (+ vll_kernels.tar.bz2) | copy before running; image-pinned |
| vLLM DSv4.1 recipe (Nov snapshot) | /perf_apps/yanfwang/dsv41-dense-indexer-20261104/agentx/InferenceY-lane-A/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml | per-conc overrides |
| sglang DSv4.1 recipe (Nov snapshot) | same tree .../dsv41flash/sglang/mi355x-fp4-mtp/agentic.yaml | override_tp2_c64_dp = best local score config |
| DSv4/DSv4.1 disagg recipes (NV, sglang/atom MI355X DSv4-Pro) | same tree .../benchmarks/multi_node/srt-slurm-recipes/{dsv4,dsv41flash}/ | only PD AgentX templates found |
| Lane runner + summarizer | /perf_apps/yanfwang/dsv41-idxshard-agentx-20261103/scripts/{run_lane.sh,summarize.py} | Slurm/pyxis, OpenAI cluster |
| Squash images incl. vLLM idxshard build | /perf_apps/yanfwang/InferenceX_CI/squash/*.sqsh | pyxis only |
| Run archive (server logs, args) | /perf_apps/yanfwang/InferenceX_CI/archive/slurm_job-*/ | config.yaml unreadable (0600) |
| Decode-step probe (sglang) | /perf_apps/yanfwang/dsv41_kvlen_bench_1028/{server.sh,client.py,analyze.py,bench.sbatch} | client.py engine-agnostic; analyze.py sglang-log only |
| TTFT / needle / gsm8k harness logs | /perf_apps/yanfwang/dsv41-dense-indexer-20261104/e2e/ | scripts in e2e/scripts |
| Indexer kernel candidate + patches | /perf_apps/yanfwang/dsv41-dense-indexer-20261104/{c4_kernel_v2_hybrid.py,e2e/patch/} | sglang HIP path |
| 1.04M-context indexer dumps | /perf_apps/yanfwang/dsv41-dense-indexer-20261104/profile/dumps_44{10,21}/ | microbench inputs |
| Kernel-fix validators | /perf_apps/yanfwang/dsv41_pd_validate_1029/validate_*.py | never ran successfully (env bug) |
| SDMA all-gather prototype | /perf_apps/yanfwang/dsv41-idxshard-sdma-20261104/sdma_allgather.py | mori cco, intra-node |
| flydsl 0.4.2 wheel, mori builds | /perf_apps/yanfwang/dsv41-upstream-20261105/ | |
| RULER long-context data | /perf_apps/yanfwang/dsv41-ruler/data/ | 131k-1.04M |
| Model weights | /perf_apps/chaox/models/DeepSeek-V4.1-Flash ; /perf_apps/data/models/DeepSeek-V4.1-Flash | read-only NFS |
| DeepInfer MI355X port | /perf_apps/jihhe/dsv41di/integ (branch amd-integ) | not vLLM; reference for EP8/DP8 decode |
| infera DSv4 PD kit + MoRIIO patches | /home/yihou/dev/git.16-19/infera.dsv41/examples/deepseek_v4 ; deploy/docker/patches/vllm-dsv4/ | DSv4-Pro, vLLM 0.23.x |
| Muse vLLM harness scripts | /home/yihou/dev/git.16-19/infera.muse/yihou/muse_vllm_20261118/scripts/ | Muse-specific paths/flags |
| PD AgentX handbook | ~/.claude/skills/pd-agentx-tuning | GLM-5.2 / sglang origin |

## 10. Gaps
- No DSv4.1-Flash PD run on any engine locally; no vLLM PD (MoRIIO/Mooncake/NIXL) on DSv4.1 at all.
- No local vLLM measurement above c32; no local vLLM TP4 AgentX point; no local vLLM DP-attention / EP point.
- Arm mapping of the dense-indexer AgentX A/B (stock vs all4) not resolved (fingerprints only).
- yanfwang's vLLM idxshard image source/patch not located (only the squash file).
- `/perf_apps/chaox/{GEAK,agentx-kernel-extraction-standalone,mono_k2_r2}`, `/perf_apps/jihhe/dsv41di/eplb-dg` and the
  DeepInfer AgentX comparison PNG were not inspected in depth.

## Verification (adversarial, 2026-11-19)

Independent re-read of the cited files. Only refuted/corrected items are listed with the correct fact; all other
checked claims (best sglang TP2+DPA c64 = 182,689; sglang TP2 c64 = 172,905-174,535; vLLM TP2 c32 = 110,103-110,495
with the stated args, KV 52.55M tokens, init 886 s; official CI 107,989 vs chaox 107,583; InferenceY 151,896 /
111k comments; sglang decode-step table; DeepInfer L5 numbers; MoRIIO patches + Mooncake example; Muse lessons)
matched their sources.

1. **CORRECTED - "only PD AgentX result is DSv4-Pro sglang 1P1D c192".** There are additional DSv4-Pro sglang-disagg
   AgentX result JSONs: `/perf_apps/yanfwang/smoke_c4/InferenceY/dsv4_p1x4_d1x4_conc4_*sglang-disagg*_conc4.json`
   and `/perf_apps/yanfwang/smoke_c4/runs/{new_1300,ref_1301,new2_1302}_conc4.json` (2026-10-01, model
   `/models/DeepSeek-V4-Pro-0813`, `"disagg": true`, 1P1D TP4, c4, 88-90 of 132-134 requests OK, 464-500 s,
   2,332-2,459 tot tok/s/GPU) - smoke runs, not valid scores. The DSv4.1-Flash part holds: no `"disagg": true`
   DSv4.1-Flash JSON found by `grep -rl '"disagg": *true'` over /perf_apps/yanfwang, /perf_apps/chaox/DeepSeek_V4.1_Flash,
   /perf_apps/jihhe/dsv41di, /perf_apps/lihuzhan/dsv4_4n_trace, /perf_apps/yangwu/dsv4-3045-clean3p3d.
2. **CORRECTED - sglang TP2 c32 range on v0.5.23-20261103.** Cited jobs 4528/4529/4572/4573 give 112,381 / 112,332 /
   112,468 / 111,903 -> **111,903-112,468** (per_gpu.total_tput_tps). 111,785 comes from jobs 4243-4310 (yfw
   20261101-idxshard image), not from the cited jobs.
3. **CORRECTED - computed (non-hit) prompt tokens.** Recomputed as
   (server_metrics.tokens.prompt_total - prompt_by_source.gpu_cache_hit) / duration / num_gpus, which agrees with
   input_tput_tps x (1 - overall_cache_hit_rate): 3244 (c64 DPA) **6,131**; 3272 (c64 DPA) 6,199; 3282 (c32 DPA)
   2,862; 4561 (c64) 6,272; 4591 (c64) ~6,304; 4572 (c32) ~3,118; 4356 (vLLM c32) ~2,593 tok/s/GPU. So the range is
   **~2.6k-6.3k**, not 3.1k-8.9k; the §8 `comp` values 8,852 (A3244), 8,899 (A3272) and 4,208 (A3282) do not follow
   from these fields. Input share of the score: 99.27% (3244) to 99.42% (4572).
4. **CORRECTED - sglang TP4 1.04M-token TTFT.** job_4509.out: phase `ttft_shon` all arms 12.32-12.93 s; phase
   `ttft_shoff` arm 1234 14.69-14.72 s and arm stock 16.56-16.59 s. "12.3-14.7 s" omits the stock shoff value
   (16.6 s). TP2 (job_4510.out, ttft_shon): 262,144 tok stock 4.39 s; 1,040,000 tok 18.5-19.6 s across arms (stock
   19.6 s). Whether the probe is cold (no prefix reuse) is not stated in the log lines read - unverified.
5. Minor note (not a refutation): run 36528242520 is attributed in the #3571 changelog entry to "#3555's run".
