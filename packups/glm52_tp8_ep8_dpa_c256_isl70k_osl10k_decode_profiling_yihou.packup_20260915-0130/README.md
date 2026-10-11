# GLM-5.2 decode profiling — TP8 / EP8 / dpa=on / C=256 / ISL 70000 / OSL 10000

**Ran:** 2026-09-14, 08:40Z – 10:55Z (single day, single host `smci355-ccs-aus-n06-25`)
**Author:** yihou
**Status:** **PASS** — profiling capability implemented, both CUDA-graph modes verified, top-10 produced.

## Goal

Give the **scheduler-free** GLM-5.2 decode benchmark (`bench/profile_decode.py`, which drives
SGLang's `TpModelWorker` + `EAGLEWorkerV2` directly — no HTTP server, no Scheduler) a profiling
capability that works **with CUDA graphs both enabled and disabled**, then profile the
**TP8 / dpa=on / EP8 / C=256** point from
`../glm52_tp8_dpa_ep8_vs_ep1_sweep_yihou.packup_20260914-0740/` and rank the top 10 time consumers.

Five ordered goals, all met: analyse SGLang's in-server profiling → enumerate every relevant
config/env var and design a both-modes scheme → implement → measure at C=256 → report the top 10.

## Success criteria and result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| profiling default-off, behaviour unchanged when off | byte-compatible with the published packup | differential run vs pristine: identical file set, identical `result_yihou.json` key set and values apart from wall-clock | ✅ |
| both CUDA-graph modes produce usable output | both | graph-ON 86 device kernels / graph-OFF 104 | ✅ |
| per-rank chrome trace + top-10 table + CPU/GPU basis stated | yes | `results/ANALYSIS_top10_yihou.md` | ✅ |
| no parameter tuned, no validation relaxed | — | ISL/OSL/accept/mem-fraction/max-running-requests unchanged | ✅ |
| harness still the one that produced the published number | `realized_accept_length` bit-identical | `3.6134393063583814` on **both** full runs | ✅ |
| end-to-end TPOT sanity (not a measurement) | within 5 % of 26.2519 ms | 26.3154 ms (**+0.24 %**) | ✅ |

## Headline results

### Top 10 device kernels — graph ON, the configuration the published number used

rank 0, window = measured iterations 1384–1394 (10 iterations, context ≈ 75000), sorted by
**self device time**, `row_kind == device_kernel` only. Denominator = 887.79 ms of kernel time.

| # | ms | % | cum % | n | µs/call | kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 216.96 | **24.4** | 24.4 | 1580 | 137.3 | `main_kernel` → **DSA sparse attention (TileLang JIT)** |
| 2 | 113.74 | **12.8** | 37.2 | 791 | 143.8 | `ncclDevKernel_Generic_1` (TP all-reduce) |
| 3 | 75.58 | 8.5 | 45.8 | 750 | 100.8 | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t64x128x256` |
| 4 | 75.45 | 8.5 | 54.3 | 810 | 93.2 | `aiter::cross_device_reduce_2stage<bf16,8>` |
| 5 | 58.75 | 6.6 | 60.9 | 220 | 267.0 | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` |
| 6 | 52.24 | 5.9 | 66.8 | 790 | 66.1 | `hgemm_bf16_128x192x64x3_SPK4` |
| 7 | 50.32 | 5.7 | 72.4 | 750 | 67.1 | `mfma_moe2_afp4_wfp4_bf16_cshuffle` |
| 8 | 35.21 | 4.0 | 76.4 | 780 | 45.1 | `_fused_fp8_bmm_rope_cat_and_cache_mla_BLOCK_M_64` |
| 9 | 20.56 | 2.3 | 78.7 | 20 | 1027.9 | `aiter::allgather_vec<bf16,8>` |
| 10 | 18.92 | 2.1 | 80.8 | 790 | 24.0 | `hgemm_bf16_128x128x64x5_SPK1` |

By functional category: **attention 31.0 % > communication 23.6 % > MoE 14.2 % > dense GEMM 8.0 %.**

### Per-iteration time budget

| layer | ms/iter | % of wall |
|---|---:|---:|
| wall (whole 2768-iteration run) | 95.070 | 100 |
| GPU bracketed by DeviceTimer (CUDA events, whole run) | 92.856 | **97.7** |
| named kernels, self device time | 88.779 | 93.4 |
| outside the timer brackets (CPU / launch / sync) | 2.214 | **2.3** |

**GPU-bound: 97.7 % of wall is GPU-occupied.** Stage split `target_verify` 90.1 % /
`eagle_draft` 6.0 % / `eagle_draft_extend` 4.0 % — all of speculative decoding costs 10 %.

### CUDA graph ON vs OFF — single-variable, `realized_accept_length` identical

| per iteration | graph ON | graph OFF | delta |
|---|---:|---:|---:|
| wall | 95.070 ms | 135.306 ms | **+42.3 %** |
| **named kernel self device time** | **88.779 ms** | **86.685 ms** | **−2.4 %** |
| non-kernel time | 6.29 ms | 48.62 ms | +673 % |
| kernel launches / iter | 2417 | 2432 | ×1.006 |
| `cuda_runtime` host calls / iter | 205 | 5392 | **×26.3** |

**CUDA graph does not make kernels faster here — it removes ≈40 ms/iter of launch overhead.**

## How to reproduce
See `REPRODUCE.md`. TL;DR: on an idle 8× MI355X host with the image present, run
`scripts/create_container_yihou.sh`, then `scripts/run_profile_yihou.sh`. One full run is ~6 min.

## Folder map
- `REPRODUCE.md` — ordered, copy-pasteable steps
- `environment.md` — hardware/software the numbers came from, and the gaps
- `notes.md` — gotchas, the traps that would have produced wrong numbers, open questions
- `results/ANALYSIS_top10_yihou.md` — **the full analysis**; also the graph-visibility probe results
- `research/rocm_profiling_env.md` — complete SGLang / PyTorch / ROCm profiling env-var inventory
- `research/implementation_notes.md` — what was added to the bench and why
- `spec/` — mission book, task file, design doc
- `evidence/runs/` — 5 C=256 runs; `evidence/aborted/` — 2 failed attempts with failure tails
- `evidence/code_snapshot/` — the exact `bench/*.py` that ran, plus the pristine copy for diffing
- `env/`, `notes_src/working_process.md`, `MANIFEST.sha256`

## What was deliberately left out (originals intact at `../../glm52_decode_profiling_yihou_20260914-0851/`, 173 MB)
- **Per-step JSONL** (108 files, 40.8 MB) — per-iteration forensics only.
- **The 58 MB SGLang source copy** (`research/sglang_src_402df1e1e/`) — it is just
  `/sglang/python/sglang` from the pinned image, re-obtainable with one `docker cp`, and contains a
  compiled `.so`.
- **Smoke-run traces and the 8× 3.55 MB graph-capture traces** — debugging artifacts at C=8.
- **Short-window chrome traces** — the short-window pair is kept for its *derived* numbers (it is the
  context-position control); its raw traces add nothing the mid-window pair does not.
Three chrome traces **are** included (0.83 / 7.59 / 15.9 MB) because each underpins a distinct
conclusion: the top-10, the graph ON/OFF comparison, and the `main_kernel` identification.
