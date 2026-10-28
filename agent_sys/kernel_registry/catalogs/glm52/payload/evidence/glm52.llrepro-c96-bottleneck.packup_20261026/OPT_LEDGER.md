> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# GLM-5.2 PD 4+4 (REDACTED_NODE -> REDACTED_NODE) optimization ledger — tried / not tried / can try

Maintained by session historical-e2e-session. One row per optimization. Goal: THROUGHPUT ONLY (latency is not a constraint;
user 10-26). Score = AgentX total tok/s per GPU (8 GPUs), sim acceptance 3.61 (no correctness claim). Run-to-run spread
at C80 ~ +-0.9 % (3 runs each of llr14 and llr21 configs).

Status: **KEPT** (in current best) | **NEUTRAL-KEPT** (within noise, kept) | **REJECTED** (measured worse) |
**FAILED** (could not run / crashed) | **N/A** (does not apply to PP4 prefill + DPA4 decode) | **TODO** (scheduled) |
**OPEN** (candidate, not scheduled) | **NO** (decided not to do, reason given).

Current best: C96 llr24 = 44,432 (llr21 config; C88 44,233; C80 mean of 3 = 43,399). Base for new runs: llr24 (C96).
Sources: [OWN] our own idea; [BASE] contributor kit/handoff; [IX] InferenceX Dynamo+SGLang; [ATOM] ATOM recipe;
[R] Infera e186e8e5 REPORT.md item P<n>; [PR] Infera PR #209 item <n> (teammate analysis 10-26).

## A. Applied / measured (in order)

| # | Optimization | Source | Run | Result (score; vs previous) | Status |
|---|---|---|---|---|---|
| 1 | Reproduce contributor 4+4 C80 (PP4xTP1 prefill, DPA4 decode, R2 router on) | BASE | llr1 | 37,392 (contributor 37,310) | base |
| 2 | index_share_for_mtp_iteration true (both legs) | OWN | llr3 | 37,743 (+0.9 %, TPOT -3.4 %) | KEPT |
| 3 | decode detokenizer workers 4 | OWN | llr4 | 37,601 (-0.4 %) | REJECTED |
| 4 | decode mem 0.95 (no AR/KF) | OWN/R P12 | llr2 | 35,997 (qps +2.4 %, ISL -6 %) | ambiguous |
| 5 | KF decode kernels (sparse MLA, indexer logits, MoE) | OWN (kernel session) | llr5 | 40,777 (+8.0 %) | KEPT |
| 6 | decode custom all-reduce on | OWN | llr6 | 40,937 (+0.4 %, TPOT -4.8 %) | KEPT |
| 7 | decode SGLANG_OPT_USE_TOPK_V2=false (KF top-k engages) | OWN | llr7 | 41,679 (+1.8 %) | KEPT |
| 8 | decode mem 0.95 (with AR+KF) | R P12 | llr8 | HSA out of resources | FAILED |
| 9 | decode graph capture 11 sizes (max 96, per-rank cap stays 64) | IX/R P4 | llr9c | 37,990 (-8.9 %, padding) | REJECTED |
| 10 | decode mem 0.93 | R P12 | llr10 | 41,453 (neutral, TTFT p90 -8 %); crashed once at heavier load (llr17) | NEUTRAL-KEPT |
| 11 | prefill --pp-async-batch-depth 1 | BASE / R P6-equivalent | llr11 | 42,506 (+2.5 %) | KEPT |
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
| 22 | INFERA_KERNEL_FUSED_INDEXER=1 (rows <= 48) | OWN (kernel session) | llr22 | no-op on this image (patch target module absent) | FAILED (needs port) |
| 23 | concurrency 80 -> 88 | IX/ATOM/R P13 | llr23 | 44,233 (+1.9 %) | KEPT |
| 24 | concurrency 96 | IX/ATOM/R P13 | llr24 | 44,432 (+0.45 % vs C88; saturating: decode KV 0.90, prealloc waits 58 % of time) | KEPT (base) |
| 25 | decode per-rank CUDA graph cap 64 -> 80 (--max-running-requests 320 --cuda-graph-max-bs-decode 80) | user / PR-14 | llr25 | running | TODO |

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
| P25 empty_cache allocator patch | OPEN | ported (ec_port.registry), not run; mem 0.93 gave only +2.2 % KV |
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
| Re-tune aiter GEMM configs (tuned csv) for our decode/prefill shapes at the current operating point | TODO (user 10-26) | 1) log the GEMM shapes actually hit (aiter untuned-shape logging) during a run; 2) offline tune on free GPUs with aiter's tuner; 3) mount the tuned csv; one e2e A/B |

## E. Experiment schedule (10-26; base = llr24 C96 since it did not regress vs C88)

| Order | Run | Change (one variable vs llr24) | Prep | ETA |
|---|---|---|---|---|
| 1 | llr25 | decode per-rank graph cap 80 | none | running, ~07:40Z |
| 2 | llr26 | MTP draft MoE experts online MXFP4 (FP8 fallback) | code read in progress (wf_a4f9f885-346) | after llr25 |
| 3 | llr27 | MTP K3: EAGLE 3/1/4, SIM_ACCEPT_LEN 2.99, both legs | code read in progress (KF 6-row assumptions) | after llr26 |
| 4 | offline | MLA bmm config microbench (our heuristic vs PR #209 table) on idle GPU0 of REDACTED_NODE | script in progress | parallel with e2e |
| 5 | llr28.. | aiter GEMM re-tune: shape capture -> offline tune on idle GPU0-3 -> mounted csv A/B | plan in progress | after microbench |
| 6 | llr2x | replicate the best of the above | — | — |
