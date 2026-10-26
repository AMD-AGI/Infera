# GLM-5.2 PD 4+4 (135 -> 136) optimization ledger — tried / not tried / can try

Maintained by session infera-glm52-pd-5f. One row per optimization. Goal: THROUGHPUT ONLY (latency is not a constraint;
user 10-26). Score = AgentX total tok/s per GPU (8 GPUs), sim acceptance 3.61 (no correctness claim). Run-to-run spread
at C80 ~ +-0.9 % (3 runs each of llr14 and llr21 configs).

Status: **KEPT** (in current best) | **NEUTRAL-KEPT** (within noise, kept) | **REJECTED** (measured worse) |
**FAILED** (could not run / crashed) | **N/A** (does not apply to PP4 prefill + DPA4 decode) | **TODO** (scheduled) |
**OPEN** (candidate, not scheduled) | **NO** (decided not to do, reason given).

Current best: llr27 (K3, C96) = 46,565; before: C96 llr24 = 44,432 (llr21 config; C88 44,233; C80 mean of 3 = 43,399). Base for new runs: llr24 (C96).
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
| 29 | K3 + draft FP8 (combine 26 + 27) | combo | llr29 | running (started 13:48Z) | — |
| 28 | PR #209 MLA bmm per-M configs (3 JSONs mounted) | ATOM PR-5 / R P15 | llr28 | 44,847 (+0.9 % vs llr24, inside the ±0.9 % run-to-run spread); step p50 55.3 -> 55.4 ms, TPOT 17.6 -> 17.8; the 3 mounts are in engine-config.env (container mounts not inspected during the run) | NEUTRAL (not carried) |
| 30 | K3 with SIM 2.91 (acceptance sensitivity) | — | llr30 | CANCELLED (user 10-26: not a performance item) | NO |
| 31 | K3 + decode per-rank graph cap 64 -> 80 (--max-running-requests 320 --cuda-graph-max-bs-decode 80); llr25 OOMed at K5, K3 frees ~7.6 GB | user (A-25 retry) | llr31 | queued after llr29 | — |

## A2. Acceptance and correctness notes (10-26)

- All e2e scores use SIGNALED acceptance (SGLANG_SIMULATE_ACC_LEN): 3.61 at K5 (InferenceX), 2.99 at K3 (ATOM PR #209 value). Not measured on our stack.
- Real acceptance, pool node 190, single TP4/DP4 server, 12 short generic prompts, greedy, bf16 draft, K5: 3.547 / 3.550; MXFP4 draft K5: 3.16-3.42 (draft memory 6.92 -> 5.07 GB). Same server, real accept len (acc.yihou/results/summary.md): AgentX-style synthetic prompts (Shakespeare filler, ignore_eos, 40 req): K5 2.81 greedy / 2.67 sampled, K3 2.61 / 2.32; real coding prompts (16 req): K5 3.92 / 3.81, K3 3.24 / 3.12. Outputs sane.
- Correctness: on this image `--enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion` (kit decode flags) with custom all-reduce gave garbage output on 190 (bf16 and MXFP4 draft); without them and with --disable-custom-all-reduce the output was sane. Which flag is responsible: NOT isolated. Our e2e runs carry these flags; simulated acceptance hides it (user 10-24: performance first, correctness later).

## B. ATOM / Dynamo analysis (REPORT P1-P28) not covered in A

| Item | Status | Reason / plan |
|---|---|---|
| P5 decode chunked-prefill 64 | OPEN | no evidence it binds; low priority |
| P6b prefill max-running 16 | NO | prefill #running-req is always 0 in our logs (cap does not bind, INFERENCE) |
| P8 enable-dp-lm-head / prefill-delayer | OPEN | no evidence; low priority |
| P9 INT4 quick reduce | OPEN | accuracy risk; decode AR messages small (INFERENCE) |
| P10 MTP K3 (EAGLE 3/1/4, sim acceptance 2.99 as ATOM) | TODO | user 10-26: schedule it; fewer verify rows (4 vs 6) and less draft work may beat higher acceptance (3.61) |
| P11, P14, P16, P19-P24, P27 (DCP-only) | N/A | decode is DPA4; list for a future DCP leg |
| P15 MLA absorbed-bmm per-M configs | TODO (offline microbench first) | our image already has its own per-M config (PR #209 item 5); compare with PR's swept table |
| P17 avoid decode re-tokenization | OPEN | PR #209: +0.5 % on ATOM; needs router/protocol change |
| P18 GC / API process | OPEN | not shown on our critical path; decode DP controller at 99.9 % CPU unexplained |
| P25 empty_cache allocator patch | OPEN | ported (ec_port.yihou), not run; mem 0.93 gave only +2.2 % KV |
| P26 CPU tier async loads | OPEN (measure first) | PR #209: sync loads ~50 % of PP stage on ATOM; measure HiCache load time per PP stage |
| P28 P:D ratio / DEP decode | NO | topology fixed at 4+4 (user) |

## C. PR #209 items new to us (teammate analysis 10-26)

| Item | Status | Plan |
|---|---|---|
| PR-9 MTP draft experts MXFP4 (online quant) | TODO | user 10-26: try it (InferenceX disallows it; this is to show draft-model optimization matters) |
| PR-13 TOPK_FORCE_PATH=one (aiter top-k single-block) | NO (no-op here) | aiter reads it (topk_per_row_kernels.cu:2650) but sglang in our image never calls aiter top_k_per_row (grep 10-26) |
| PR-5 / P15 swept MLA bmm table | TODO (offline) | microbench vs our heuristic at our M |
| PR-14 graph bound by concurrency | see A-25 (llr25) | user 10-26: 64 too small, try 80 |
| C4 DPA gather/reduce kernel (aiter IPC vs RCCL) | OPEN | needs a trace that sees inside the verify graph |
| PR-2 double /metrics render | OPEN (measure first) | expected ~0 throughput |
| PR-12 MC_NUM_QP_PER_EP=1 | NO (for now) | stability only; links ~8 % busy |
| PR-19 router --breaker-failure-threshold 0 | OPEN | only if breaker-open events appear in our router logs |

## D. Other candidates

| Item | Status | Plan |
|---|---|---|
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
