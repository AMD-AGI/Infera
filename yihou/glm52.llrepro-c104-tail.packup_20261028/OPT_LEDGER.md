# GLM-5.2 PD 4+4 (135 -> 136) optimization ledger — tried / not tried / can try

Maintained by session infera-glm52-pd-5f. One row per optimization. Goal: THROUGHPUT ONLY (latency is not a constraint;
user 10-26). Score = AgentX total tok/s per GPU (8 GPUs), sim acceptance 3.61 (no correctness claim). Run-to-run spread
at C80 ~ +-0.9 % (3 runs each of llr14 and llr21 configs).

Status: **KEPT** (in current best) | **NEUTRAL-KEPT** (within noise, kept) | **REJECTED** (measured worse) |
**FAILED** (could not run / crashed) | **N/A** (does not apply to PP4 prefill + DPA4 decode) | **TODO** (scheduled) |
**OPEN** (candidate, not scheduled) | **NO** (decided not to do, reason given).

Current best: llr37 stack at C104, mean of 3 = 48,204 (48,713 / 48,103 / 47,795).
Sources: [OWN] our own idea; [LLY] llying kit/handoff; [IX] InferenceX Dynamo+SGLang; [ATOM] ATOM recipe;
[R] Infera e186e8e5 REPORT.md item P<n>; [PR] Infera PR #209 item <n> (teammate analysis 10-26).

## A. Applied / measured (in order)

| # | Optimization | Source | Run | Result (score; vs previous) | Status |
|---|---|---|---|---|---|
| 1 | Reproduce llying 4+4 C80 (PP4xTP1 prefill, DPA4 decode, R2 router on) | LLY | llr1 | 37,392 (llying 37,310) | base |
| 2 | index_share_for_mtp_iteration true (both legs) | OWN | llr3 | 37,743 (+0.9 %, TPOT -3.4 %) | KEPT |
| 3 | decode detokenizer workers 4 | OWN | llr4 | 37,601 (-0.4 %) | REJECTED |
| 4 | decode mem 0.95 (no AR/KF) | OWN/R P12 | llr2 | 35,997 (qps +2.4 %, ISL -6 %) | ambiguous |
| 5 | KF decode kernels (sparse MLA, indexer logits, MoE) | OWN (kernel session) | llr5 | 40,777 (+8.0 %) | KEPT |
| 6 | decode custom all-reduce on | OWN | llr6 | 40,937 (+0.4 %, TPOT -4.8 %) | KEPT |
| 7 | decode SGLANG_OPT_USE_TOPK_V2=false (KF top-k engages) | OWN | llr7 | 41,679 (+1.8 %) | KEPT |
| 8 | decode mem 0.95 (with AR+KF) | R P12 | llr8 | HSA out of resources | FAILED |
| 9 | decode graph capture 11 sizes (max 96, per-rank cap stays 64) | IX/R P4 | llr9c | 37,990 (-8.9 %, padding) | REJECTED |
| 10 | decode mem 0.93 | R P12 | llr10 | 41,453 (neutral, TTFT p90 -8 %); crashed once at heavier load (llr17) | NEUTRAL-KEPT |
| 11 | prefill --pp-async-batch-depth 1 | LLY / R P6-equivalent | llr11 | 42,506 (+2.5 %) | KEPT |
| 12 | decode radix cache + router session-affinity both | ATOM/R P1 | llr12 | 40,814 (-4.0 %, DP-rank imbalance) | REJECTED |
| 13 | decode --stream-interval 30 | IX/R P3 | llr13 | 42,892 (+0.9 %) | NEUTRAL-KEPT |
| 14 | decode SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8 | IX/R P2 | llr14 | 43,043 (+0.35 %); config mean 42,689 (3 runs) | NEUTRAL-KEPT |
| 15 | prefill --pp-async-batch-depth 2 | OWN | llr15 | 42,411 (-1.5 %) | REJECTED |
| 16 | prefill HiCache io direct (page_first_direct) | IX/R P7 | llr16 | prefill 0.21 batch/s (stopped) | REJECTED |
| 17 | prefill chunk 8192 -> 4096 | OWN | llr17/17b | 0.93: crash; 0.92: 42,225 (neutral) | REJECTED |
| 18 | decode eager (graph off) for profiling | OWN | llr18 | start-up crash (topk.hip:184) | FAILED |
| 19 | rocprofv3 --attach for verify-graph kernels | OWN | llr19 | no output written | FAILED |
| 20 | prefill SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1 | OWN (code read) | llr20b | AssertionError with prefill MTP | FAILED |
| 21 | prefill SGLANG_PP_COMM_OVERLAP=1 | OWN (code read) / R P6 | llr21/21r/22 | mean 43,399 (+1.7 % vs llr14 mean; TTFT p90 6.8 s vs 9.5 s) | KEPT |
| 22 | YIHOU_KF_FUSED_INDEXER=1 (rows <= 48) | OWN (kernel session) | llr22 | no-op on this image (patch target module absent) | FAILED (needs port) |
| 23 | concurrency 80 -> 88 | IX/ATOM/R P13 | llr23 | 44,233 (+1.9 %) | KEPT |
| 24 | concurrency 96 | IX/ATOM/R P13 | llr24 | 44,432 (+0.45 % vs C88; saturating: decode KV 0.90, prealloc waits 58 % of time) | KEPT (base) |
| 25 | decode per-rank CUDA graph cap 64 -> 80 (--max-running-requests 320 --cuda-graph-max-bs-decode 80) | user / PR-14 | llr25 | HSA out of resources 28 min into profiling (5.85 GB free after capture at mem 0.93) | FAILED (retry only with lower mem or after freeing memory, e.g. with draft FP8) |
| 26 | MTP draft experts online FP8 (SGLANG_GLM_NEXTN_MOE_PTPC=1) | ATOM PR-9 (FP8 variant) | llr26 | 45,066 (+1.4 % vs llr24; TPOT 17.6 -> 17.2, step p50 55.3 -> 54.2 ms); PTPC-FP8 log lines confirmed, shared-expert fusion off for the draft | KEPT (new best; InferenceX rule for FP8 draft UNKNOWN) |
| 27 | MTP K3 (EAGLE 3/1/4, SIM 2.99) | user / ATOM / R P10 | llr27 | **46,565 (+4.8 % vs llr24)**; step p50 55.3 -> 44.6 ms, TPOT 17.6 -> 15.4; graph mem 10.5 -> 7.5 GB, 13.5 GB free after capture | KEPT (new best; note acceptance 2.99 vs 3.61 is a modeling choice) |
| 29 | K3 + draft FP8 (combine 26 + 27) | combo | llr29 | **47,014 (+1.0 % vs llr27)**; step p50 44.6 -> 43.3 ms, TPOT 15.4 -> 14.9, qps 3.041 -> 3.063; 13.5 GB free after verify capture | KEPT (new best) |
| 32 | llr29 + aiter BF16 GEMM retune csv (75 shapes, merged c96 + k3) mounted in model_configs | user / D | llr32 | **47,447 (+0.9 % vs llr29)**; step p50 43.3 -> 42.1 ms, TPOT 14.9 -> 14.7, qps 3.063 -> 3.104; aiter merged the csv (log); 49 other BF16 shapes (N=1024 K=6144, N=6144 K=512) still untuned -> tuning on 190 (aiter_retune/ptpc/) | KEPT (new best) |
| 33 | llr32 + decode mem 0.93 -> 0.95 | user 10-24 (0.95 not a bad point) | llr33 | **47,957 (+1.1 % vs llr32)**; KV tokens 2.92 M -> 3.04 M (+4.2 %), prealloc-wait frac 0.44 -> 0.36, TTFT p90 14.65 -> 13.28 s, qps 3.104 -> 3.114; no HSA errors | KEPT (new best) |
| 34 | llr32 with MTP draft experts online MXFP4 (our patch mxdraft.yihou, quark_mxfp4) instead of FP8 | user 10-26 / PR-9 | llr34 |  47,282 (-0.3 % vs llr32 47,447, on llr32 base mem 0.93); step p50 42.1 -> 42.4 ms | NEUTRAL (FP8 draft kept; MXFP4 real acceptance is also lower) |
| 35 | llr33 + 2nd aiter BF16 tuned csv (49 shapes untuned under draft FP8: N=1024 K=6144, N=6144 K=512; tuned on 190) | D | llr35 |  47,843 (-0.2 % vs llr33 47,957); qps 3.114 -> 3.125, step p50 42.75 -> 42.3 ms, 0 'not found tuned config' lines left | NEUTRAL (kept in the stack: no untuned BF16 GEMM left; llr36 builds on it) |
| 36 | llr35 + decode mem 0.95 -> 0.96 (prealloc-wait still 0.36 at 0.95) | user 10-24 | llr36 |  47,861 (+0.04 % vs llr35 47,843); KV tokens 3.04 M -> 3.11 M, prealloc-wait 0.35 -> 0.32, step p50 42.4 ms; no HSA errors | NEUTRAL (more KV no longer converts to score; stay at 0.95) |
| 35r | replicate llr35 (spread of the current stack) | — | llr35r |  47,950 (llr35 47,843; mean 47,897, spread ±0.1 %); step p50 42.3 ms | stack level ~47.9 k (llr33 47,957 is the same stack minus the 2nd csv) |
| 37 | llr35 at CONC 96 -> 104 | IX/ATOM/R P13 | llr37 | **48,713 (+1.7 % vs stack mean 47,897)**; qps 3.126 -> 3.209 (+2.7 %), ISL 121.8 k -> 120.5 k, TTFT p50 5.0 -> 6.0 s, TPOT 14.7 -> 15.0; decode KV usage 0.86 -> 0.92, prealloc-wait 0.35 -> 0.58 | KEPT (new best) |
| 38 | llr35 + prefill PP4 chunk 8192 -> 16384 (kit default 16384; 4096 was neutral in A-17) | OWN | llr38 |  46,862 (-2.2 % vs C96 mean 47,897); prefill PP0 batches 1.93 -> 1.34/s, TTFT p50 5.0 -> 6.5 s, KV transfer latency p50 2.0 -> 3.0 s | REJECTED (chunk 8192 stays) |
| 39 | llr37 at CONC 104 -> 112 | IX/ATOM/R P13 | llr39 |  45,668 (-6.3 % vs C104); decode KV 0.95, prealloc-wait 0.90, GPU prefix hit 0.43 -> 0.39, TTFT p50 10.0 s | REJECTED (C104 stays; saturation between 104 and 112) |
| 37r | replicate llr37 (C104) | — | llr37r |  48,103 (llr37 48,713; C104 mean 48,408 = +1.1 % vs C96 mean 47,897); qps 3.213 (llr37 3.209; C96 3.126 -> +2.8 %), ISL 118.8 k | C104 confirmed (score spread ±0.6 %, qps ±0.1 %) |
| 40 | llr37 + decode --enable-dp-lm-head (P8; image help: vocab parallel over attn-TP, avoids all-gather across DP groups) | R P8 | llr40 |  46,898 (-3.1 % vs C104 mean 48,408); qps 3.211 -> 3.148, decode step p50 42.4 -> 43.0 ms, TPOT 15.0 -> 15.2; flag active (server_args); prefill GPU prefix hit 0.43 -> 0.33 (not explained) | REJECTED |
| 41 | llr37 + decode --enable-dp-attention-local-control-broadcast (image help: removes an all-ranks gloo sync per scheduler iteration) | code read | llr41 |  48,027 (-0.8 % vs C104 mean 48,408, inside the C104 spread 48,103-48,713); step p50 42.5 ms unchanged, qps 3.176 | NEUTRAL (not carried) |
| 42 | llr37 + KF gfx950 fused decode indexer (rows <= 48), overlay copy kfov_fused.yihou with a target alias: this image keeps Gfx950FusedIndexer in sglang.kernels.ops.attention.dsa.hip_gfx950.fused_decode, so the original target (srt.layers.attention.dsa.gfx950_fused_indexer) was never imported - that is why A-22 was a no-op (0 'fused_indexer rows' log lines in llr22); file md5 8bbdae09 = the KF-validated one; double-patch guard added; smoke = llr42 startup log (patched / rows= -> KF / CHECK lines) | KF session request 10-27 | llr42 |  47,674 (-1.5 % vs C104 mean 48,408; C104 runs 48,103-48,713); qps 3.211 -> 3.196, decode step p50 42.4 -> 43.1 ms (slower), TPOT 15.0 -> 15.2; smoke OK: 'patched Gfx950FusedIndexer.run' x4 (one per rank), rows 4..48 -> KF, CHECK rows=48 identical 48/48 on all 4 ranks | NOT KEPT (no e2e gain; step time did not drop) |
| 43 | llr37 at CONC 104 -> 108 (peak between 104 and 112) | P13 | llr43 |  47,322 (-2.2 % vs C104 mean 48,408); prealloc-wait 0.73, TTFT p50 8.0 s; step p50 41.7 ms | REJECTED (C104 is the peak: C96 47.9 k, C104 48.4 k, C108 47.3 k, C112 45.7 k) |
| 44 | llr37 WITHOUT prefill SGLANG_PP_COMM_OVERLAP=1 (llying 10-27: main cause of garbled output with real acceptance) - cost of the correct setting | llying bisect | llr44 |  41,457 (-14.4 % vs C104 mean 48,408); qps 3.211 -> 2.853, prefill PP0 batches 1.93 -> 1.61/s, TTFT p50 6.3 -> 9.2 s, decode running/rank 12 -> 10, prealloc-wait 0.60 -> 0.84, transfer-req 5.6 -> 7.6, step p50 40.5 ms; ISL 115.3 k | COST of the correct setting at C104 (much larger than the +1.7 % it gave at C80 in A-21; mechanism not measured) |
| 45 | llr44 + decode custom all-reduce OFF (2nd garble contributor per llying) - cost of the correct setting | llying bisect | llr45 |  42,186 (llr44 41,457: +1.8 %, inside run-to-run spread; qps 2.853 -> 2.896); decode step p50 40.5 -> 43.4 ms (+7 %), prefill PP0 batches 1.64/s, TTFT p50 7.6 s | garble-free candidate at C104 ~41.5-42.2 k (-13 to -14 % vs 48.4 k); custom AR cost not visible in the score because prefill binds |
| 46 | llr45 at CONC 96 (concurrency of the correct setting) | — | llr46 |  42,543 (llr45 C104 42,186: +0.8 %, flat); qps 2.831, ISL 119.6 k (llr45 115.6 k), prealloc-wait 0.42, prefill PP0 1.72 batches/s | C96 ~= C104 for the garble-free settings |
| 47 | llr45 at CONC 112 | — | llr47 |  39,475 (-6.4 % vs llr45 C104); prealloc-wait 0.94, TTFT p50 11.8 s, prefill PP0 1.55 batches/s | REJECTED (garble-free peak at C96-C104) |
| 45r | replicate llr45 (garble-free, C104) | — | llr45r |  37,919 (llr45 42,186: -10 %); prefill bootstrap-req mean 15.7 (llr45 8.9), KV xfer p90 8.3 s; 'Decode transfer failed ... timed out after 1800 s in KVPoll.WaitingForInput' lines: llr37 3, llr37r 4 (overlap ON) vs llr44 12, llr45 14, llr46 5, llr47 21, llr45r 17 (overlap OFF) | garble-free settings are UNSTABLE run-to-run (37.9-42.5 k); more 1800-s transfer timeouts with overlap off (correlation only, mechanism not measured) |
| 48 | llr45 at CONC 88 | — | llr48 |  **44,245** (garble-free best so far; C96 42,543, C104 42,186 / 37,919, C112 39,475); qps 2.947, TTFT p50 4.8 s, decode KV 0.81, prealloc-wait 0.23, 1800-s transfer timeouts: 1 | garble-free peak moves down to <= C88 |
| 37r2 | 2nd replicate of llr37 (overlap ON): drift control | — | llr37r2 |  47,795; C104 x3 = 48,713 / 48,103 / 47,795, mean 48,204 (spread ±1 %); 2 transfer timeouts; step p50 42.4 ms | no drift; base = C104 mean 48,204 |
| 49 | garble-free settings at CONC 80 | — | llr49 | CANCELLED (user 10-28: performance phase, correctness not a constraint) | — |
| 48r | replicate llr48 | — | llr48r | CANCELLED (user 10-28) | — |
| 50 | llr37 + decode ROCM_QUICK_REDUCE_QUANTIZATION=INT4 (P9) | ATOM/Dynamo P9 | llr50 |  48,010 (-0.4 % vs C104 mean 48,204, inside spread); qps 3.209, step p50 42.8 ms (base 42.4); env reached the container config, activation not logged (debug-level) | NEUTRAL (not carried) |
| 51 | llr37 + decode mem 0.96 at C104 (KV 0.92, prealloc-wait 0.58 at C104) | P25-adjacent | llr51 |  48,806 (+1.2 % vs C104 mean 48,204; highest single run); qps 3.208 (base ~3.20, unchanged), ISL 120.8 k (base 118.8-120.5 k), prealloc-wait 0.58 -> 0.51, no HSA errors | CANDIDATE (gain at the edge of the ±1 % spread, qps flat) -> replicate llr51r |
| 52 | llr37 + decode DP controller blocking poll (P18 patch dpc_poll.yihou: zmq poll(1000 ms) after the NOBLOCK drain) | OWN / P18 | llr52 |  47,402 (-1.7 % vs C104 mean 48,204); step p50 42.3 ms unchanged; controller no longer in the decode top-CPU list (patch active) | NEUTRAL (not carried) |
| 53 | llr37 + prefill mem 0.88 -> 0.91 (larger GPU radix pool; kit default 0.88, prefill args are last-wins) | OWN | llr53 |  STOPPED at start (06:36Z) - machines released to the user 10-28 06:45Z | NOT RUN |
| 51r | replicate llr51 (mem 0.96 at C104) | — | llr51r |  not run (machines released) | NOT RUN |
| 28 | PR #209 MLA bmm per-M configs (3 JSONs mounted) | ATOM PR-5 / R P15 | llr28 | 44,847 (+0.9 % vs llr24, inside the ±0.9 % run-to-run spread); step p50 55.3 -> 55.4 ms, TPOT 17.6 -> 17.8; the 3 mounts are in engine-config.env (container mounts not inspected during the run) | NEUTRAL (not carried) |
| 30 | K3 with SIM 2.91 (acceptance sensitivity) | — | llr30 | CANCELLED (user 10-26: not a performance item) | NO |
| 31 | K3 + decode per-rank graph cap 64 -> 80 (--max-running-requests 320 --cuda-graph-max-bs-decode 80) | user (A-25 retry) | llr31 | 46,406 (-0.3 % vs llr27; no OOM, 9.5 GB free after capture); per-rank running mean 12 / p90 18, so the cap of 64 does not bind at C96 | NEUTRAL (not carried) |

## A2. Acceptance and correctness notes (10-26)

- All e2e scores use SIGNALED acceptance (SGLANG_SIMULATE_ACC_LEN): 3.61 at K5 (InferenceX), 2.99 at K3 (ATOM PR #209 value). Not measured on our stack.
- Real acceptance, pool node 190, single TP4/DP4 server, 12 short generic prompts, greedy, bf16 draft, K5: 3.547 / 3.550; MXFP4 draft K5: 3.16-3.42 (draft memory 6.92 -> 5.07 GB). Same server, real accept len (acc.yihou/results/summary.md): AgentX-style synthetic prompts (Shakespeare filler, ignore_eos, 40 req): K5 2.81 greedy / 2.67 sampled, K3 2.61 / 2.32; real coding prompts (16 req): K5 3.92 / 3.81, K3 3.24 / 3.12. Outputs sane.
- Correctness: on this image `--enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion` (kit decode flags) with custom all-reduce gave garbage output on 190 (bf16 and MXFP4 draft); without them and with --disable-custom-all-reduce the output was sane. Which flag is responsible: NOT isolated. Our e2e runs carry these flags; simulated acceptance hides it (user 10-24: performance first, correctness later).

## B. ATOM / Dynamo analysis (REPORT P1-P28) not covered in A

| Item | Status | Reason / plan |
|---|---|---|
| P5 decode chunked-prefill 64 | OPEN | no evidence it binds; low priority |
| P6b prefill max-running 16 | NO | prefill #running-req is always 0 in our logs (cap does not bind, INFERENCE) |
| P8 enable-dp-lm-head / prefill-delayer | DONE (A-40 dp-lm-head -3.1 %, rejected) | no evidence; low priority |
| P9 INT4 quick reduce | DONE (A-50 neutral) | accuracy risk; decode AR messages small (INFERENCE) |
| P10 MTP K3 (EAGLE 3/1/4, sim acceptance 2.99 as ATOM) | DONE (A-27, kept) | user 10-26: schedule it; fewer verify rows (4 vs 6) and less draft work may beat higher acceptance (3.61) |
| P11, P14, P16, P19-P24, P27 (DCP-only) | N/A | decode is DPA4; list for a future DCP leg |
| P15 MLA absorbed-bmm per-M configs | DONE (A-28, neutral) | our image already has its own per-M config (PR #209 item 5); compare with PR's swept table |
| P17 avoid decode re-tokenization | OPEN | PR #209: +0.5 % on ATOM; needs router/protocol change |
| P18 GC / API process | DONE (A-52 DP controller blocking poll: neutral) | not shown on our critical path; decode DP controller at 99.9 % CPU unexplained |
| P25 empty_cache allocator patch | OPEN | ported (ec_port.yihou), not run; mem 0.93 gave only +2.2 % KV |
| P26 CPU tier async loads | OPEN (measure first) | PR #209: sync loads ~50 % of PP stage on ATOM; measure HiCache load time per PP stage |
| P28 P:D ratio / DEP decode | NO | topology fixed at 4+4 (user) |

## C. PR #209 items new to us (teammate analysis 10-26)

| Item | Status | Plan |
|---|---|---|
| PR-9 MTP draft experts MXFP4 (online quant) | DONE (A-34, neutral vs FP8) | user 10-26: try it (InferenceX disallows it; this is to show draft-model optimization matters) |
| PR-13 TOPK_FORCE_PATH=one (aiter top-k single-block) | NO (no-op here) | aiter reads it (topk_per_row_kernels.cu:2650) but sglang in our image never calls aiter top_k_per_row (grep 10-26) |
| PR-5 / P15 swept MLA bmm table | DONE (A-28, neutral) | microbench vs our heuristic at our M |
| PR-14 graph bound by concurrency | see A-25 (llr25) | user 10-26: 64 too small, try 80 |
| C4 DPA gather/reduce kernel (aiter IPC vs RCCL) | OPEN | needs a trace that sees inside the verify graph |
| PR-2 double /metrics render | OPEN (measure first) | expected ~0 throughput |
| PR-12 MC_NUM_QP_PER_EP=1 | NO (for now) | stability only; links ~8 % busy |
| PR-19 router --breaker-failure-threshold 0 | NO (no breaker-open events in llr37/llr37r router logs) | only if breaker-open events appear in our router logs |

## D. Other candidates

| Item | Status | Plan |
|---|---|---|
| Prefill BF16 GEMM retune (garble-free stack is prefill-bound) | OPEN (measured) | llr45 prefill log: 13,164 'not found tuned config' lines, 3,523 distinct (M,N,K): N/K = 32/6144, 2624/6144, 4096/2048, 6144/16384, 24576/6144, 28672/512; M = chunk token counts (arbitrary, up to 8192), so exact-M tuning does not apply; needs bucketed M via aiter padded-M lookup (get_padded_m) - not investigated yet |
| Re-tune aiter GEMM configs (tuned csv) for our decode/prefill shapes at the current operating point | IN PROGRESS on pool OAI node crsuse2-m2m-190 (image copied 10-26): BF16 MoE of the draft: aiter tuner has NO candidate kernels for bf16 no-quant E=257 (stage1 tasks 0) -> not tunable (draft FP8 is the route); BF16 GEMM a16w16 (65 shapes, --with-hipblaslt) DONE 12:10Z (prep_next.yihou/aiter_retune/c96/yihou_glm52_dpa_bf16_tuned_gemm.csv, 65 rows, 0 key overlap with image CSVs); K3 adds 10 untuned M (80-256, N 6144 K 16384) -> tuned 13:26Z (k3/yihou_glm52_k3_bf16_tuned_gemm.csv); next: merge, mount, e2e A/B vs llr27 | 1) log the GEMM shapes actually hit (aiter untuned-shape logging) during a run; 2) offline tune on free GPUs with aiter's tuner; 3) mount the tuned csv; one e2e A/B |

## E. Experiment schedule (10-26; base = llr24 C96 since it did not regress vs C88)

| Order | Run | Change (one variable vs llr24) | Prep | ETA |
|---|---|---|---|---|
| 1 | llr25 | decode per-rank graph cap 80 | none | FAILED (OOM) |
| 2a | llr26 (running) | MTP draft MoE experts online FP8 (decode env SGLANG_GLM_NEXTN_MOE_PTPC=1; flag-only, glm4_moe.py:119-147,1323-1430) | ready | after llr25 |
| 2b | llr26b | MTP draft experts online MXFP4 (needs ~15-25-line sglang patch: widen Qwen3.5-only gates model_config.py:2014-2031, weight_utils.py:220-248, add glm_moe_dsa excludes quark.py:684-702; --speculative-draft-model-quantization quark_mxfp4); InferenceX disallows MXFP4 draft | patch to write | after 2a |
| 3 | llr27 (queued) | MTP K3: EAGLE 3/1/4 via PREFILL/DECODE_EXTRA_ARGS (last wins), SIM_ACCEPT_LEN 2.99 | ready: prep_next.yihou/llrk3.site.env.draft; KF indexer fused2 only at bs%3==0 (6-row grouping), else single-row KF; gfx950 fused path for bs<=12 | after llr26 |
| 4 | offline | MLA bmm config microbench (stock aiter buckets vs PR #209 table; note: our decode q-up goes through the FUSED bmm+rope+cache kernel whose config the PR does not patch; v-up uses the batched kernel) | DONE 10-26 06:4xZ (r1.log): PR #209 table bit-exact and faster at most M: v-up verify M48-144 16.0-16.3 -> 5.0-13.3 us (x1.2-2.9), draft M8-24 x1.36-1.46; fused q-up draft x1.16-1.49, verify x0.93-1.25 (slower at M48/114-126). Est. verify saving ~11 us/layer x 78 = ~0.9 ms/step (~1.6 %, INFERENCE; draft model is 1 layer). Next: e2e A/B with the 3 JSONs mounted (needs multi-mount decode hook) |
| 5 | llr28.. | aiter re-tune: BF16 MoE of the MTP draft (all 10 token buckets hit '2stage default') and BF16 GEMMs (o_proj 16384->6144 at 47 M values etc. hit 'not found tuned config -> torch'); gemm_moe_tune.py / gemm_a16w16_tune.py --with-hipblaslt on GPU0-3; dedupe keys before mounting (duplicate key = start-up raise); needs DECODE_MOUNT_EXTRA2 kit hook | plan ready (prep_next.yihou/aiter_retune) | after microbench |
| 7 | llr2x | DP controller busy-poll fix (blocking zmq poll) | needs patch + 2nd decode mount hook; expected ~0 (INFERENCE: 1 core of 236, GPU-bound) | low |
| 6 | llr2x | replicate the best of the above | — | — |

## F. Status at machine release (2026-10-28 06:45Z) and next plan

- 135/136 released (all yihou-llr containers stopped, queues stopped, STOP_QUEUE.yihou present). Self-check cron deleted.
- Best stack: llr37 config (llr35 + C104): mean of 3 = 48,204 (48,713 / 48,103 / 47,795). Best single run: llr51
  (+ decode mem 0.96) 48,806, NOT replicated (qps flat vs base; ISL +1.5 %).
- Correctness (not this phase's goal, user 10-28): the stack garbles with real acceptance (llying: PP comm overlap,
  decode custom AR). Without both: C88 44,245 (single run); C104 unstable (37.9-42.2 k) with 1800-s KV transfer timeouts.
- Next, in order (one variable vs the llr37 stack):
  1. llr51r: replicate decode mem 0.96 at C104 (keep if >= +1 % again).
  2. llr53: prefill --mem-fraction-static 0.91 (site ready).
  3. Prefill BF16 GEMM retune: 3,523 untuned shapes with arbitrary chunk M; read aiter get_padded_m to tune buckets.
  4. C4 DPA gather/reduce: trace inside the verify graph first.
  5. P26: measure HiCache load time per PP stage; async loads if large.
  6. KF fused indexer 8dcc5a1 once KF puts it in the overlay (1748c11 was live but -1.5 %).
  7. P17 decode re-tokenization (router/protocol change; +0.5 % on ATOM).
