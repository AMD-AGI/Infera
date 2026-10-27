# GLM-5.2-MXFP4 PD on MI355X — config / tuning findings (no code), for a possible Infera recipe PR

Score = AgentX 3600 s `total_tput_tps` per GPU (tok/s/chip), **simulated MTP acceptance** (3.61 at K5, 2.99 at K3): no
correctness claim. n = 1 per row unless stated; run-to-run spread at C80 ~ +-0.9 % (3 runs each of two configs).
Sources: `NFS/llrepro_20261023/OPT_LEDGER.md` (10-27 copy, newer than the pack-up copy) and the pack-ups named per block.

**Correctness warning (llying 10-27, Infera 27740072 / 19a9c82d):** the llr35 stack garbles with real acceptance; the
main cause is `SGLANG_PP_COMM_OVERLAP=1` (row C-21); decode custom all-reduce adds short-prompt garbles (C-6); long
contexts still degrade without either (open, round 2 running). Do not put C-6 or C-21 in a recipe until that is resolved.

## A. Current 4+4 stack (prefill crsuse2-m2m-135 GPU4-7 PP4xTP1, decode 136 GPU4-7 TP4/DP4 DPA; llying kit + I07 hooks; image v0522-fdbafb60)

Ownership of the idea: [OWN] yihou, [LLY] llying kit, [IX] InferenceX Dynamo+SGLang, [ATOM] ATOM / Infera #209.

| # | knob (leg) | run | result vs parent | status | idea |
|---|---|---|---|---|---|
| C-1 | llying 4+4 C80 reproduction (base) | llr1 | 37,392 (llying 37,310) | base | LLY |
| C-2 | `index_share_for_mtp_iteration=true` (both; needs S05 fix) | llr3 | 37,743 (+0.9 %, TPOT -3.4 %) | kept | OWN |
| C-3 | decode detokenizer workers 4 | llr4 | 37,601 (-0.4 %) | rejected | OWN |
| C-4 | decode mem 0.95 (no AR/KF) | llr2 | 35,997, qps +2.4 %, ISL -6 % | ambiguous | OWN |
| C-5 | KF decode overlay (code: K01) | llr5 | 40,777 (+8.0 %) | kept | OWN |
| C-6 | decode custom all-reduce ON (`DECODE_CUSTOM_AR=1`) | llr6 | 40,937 (+0.4 %, TPOT -4.8 %) | kept, **implicated in garbling** | OWN |
| C-7 | decode `SGLANG_OPT_USE_TOPK_V2=false` (lets KF top-k engage) | llr7 | 41,679 (+1.8 %) | kept | OWN |
| C-8 | decode mem 0.95 with AR+KF | llr8 | HSA out of resources | failed | OWN |
| C-9 | decode graph 11 capture sizes, max 96 | llr9c | 37,990 (-8.9 %, padding) | rejected | IX |
| C-10 | decode mem 0.93 | llr10 | 41,453 (neutral, TTFT p90 -8 %) | neutral-kept | ATOM report |
| C-11 | prefill `--pp-async-batch-depth 1` | llr11 | 42,506 (+2.5 %) | kept | LLY |
| C-12 | decode radix cache + router session affinity | llr12 | 40,814 (-4.0 %, DP-rank imbalance) | rejected | ATOM |
| C-13 | decode `--stream-interval 30` | llr13 | 42,892 (+0.9 %) | neutral-kept | IX |
| C-14 | decode `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` | llr14 | 43,043 (+0.35 %); 3-run mean 42,689 | neutral-kept | IX |
| C-15 | prefill `--pp-async-batch-depth 2` | llr15 | 42,411 (-1.5 %) | rejected | OWN |
| C-16 | prefill HiCache io `page_first_direct` | llr16 | prefill 0.21 batch/s | rejected | IX |
| C-17 | prefill PP4 chunk 8192 -> 4096 | llr17b | 42,225 (neutral; mem 0.93 crashed) | rejected | OWN |
| C-20 | prefill `SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1` | llr20b | AssertionError with prefill MTP | failed | OWN |
| C-21 | prefill `SGLANG_PP_COMM_OVERLAP=1` | llr21/21r/22 | mean 43,399 (+1.7 % vs llr14 mean; TTFT p90 6.8 vs 9.5 s) | **main garbling cause (llying 10-27)** | OWN |
| C-23 | concurrency 80 -> 88 | llr23 | 44,233 (+1.9 %) | kept | IX/ATOM |
| C-24 | concurrency 96 | llr24 | 44,432 (+0.45 %; decode KV 0.90, prealloc waits 58 %) | kept | IX/ATOM |
| C-25 | decode per-rank graph cap 64 -> 80 (`--max-running-requests 320 --cuda-graph-max-bs-decode 80`) | llr25 / llr31 | OOM at mem 0.93 / with K3 46,406 (-0.3 %, cap never binds: per-rank p90 18) | not carried | PR #209 |
| C-26 | MTP draft experts online FP8 (`SGLANG_GLM_NEXTN_MOE_PTPC=1`) | llr26 | 45,066 (+1.4 % vs llr24; step p50 55.3 -> 54.2 ms) | kept | ATOM (FP8 variant) |
| C-27 | MTP K3 (EAGLE 3/1/4 on both legs; sim 2.99) | llr27 | 46,565 (+4.8 % vs llr24; step p50 55.3 -> 44.6 ms) | kept | ATOM |
| C-28 | PR #209 MLA bmm per-M configs (3 JSON mounts) | llr28 | 44,847 (+0.9 %, inside spread) | neutral, not carried | ATOM |
| C-29 | K3 + draft FP8 | llr29 | 47,014 (+1.0 % vs llr27) | kept | combo |
| C-32 | aiter BF16 GEMM csv #1 (code/data: A01) | llr32 | 47,447 (+0.9 %) | kept | OWN |
| C-33 | decode mem 0.93 -> 0.95 | llr33 | 47,957 (+1.1 %; KV +4.2 %, prealloc-wait 0.44 -> 0.36) | kept | OWN (user) |
| C-34 | draft MXFP4 (code: S14) instead of FP8 | llr34 | 47,282 (-0.3 %) | neutral | PR #209 idea |
| C-35 | aiter BF16 csv #2 (A01) | llr35 / llr35r | 47,843 / 47,950 (mean 47,897, ±0.1 %) | neutral, kept | OWN |
| C-36 | decode mem 0.96 | llr36 | 47,861 (+0.04 %) | neutral (stay 0.95) | OWN |
| C-37 | concurrency 96 -> 104 | llr37 | **48,713 (+1.7 % vs stack mean; qps +2.7 %, TTFT p50 5.0 -> 6.0 s)** | kept (best at 10-27 06:45Z) | IX/ATOM |
| C-38 | prefill PP4 chunk 8192 -> 16384 | llr38 | 46,862 (-2.2 %) | rejected | OWN |
| C-39 | concurrency 104 -> 112 | llr39 | running at time of writing | — | — |

Not applicable / not done (ledger B-D): decode chunked-prefill 64, dp-lm-head, INT4 quick reduce, avoid re-tokenization,
GC, CPU-tier async loads (open); `TOPK_FORCE_PATH=one` is a no-op on this image (sglang never calls aiter
`top_k_per_row`); `MC_NUM_QP_PER_EP=1` (stability only, links ~8 % busy).

Bottleneck at C96 (llr24/llr35r, `bneck`): decode-bound — decode GPU ~100 %, VRAM 287/288 GiB, KV usage mean 0.86-0.90,
prealloc waits 35-58 % of samples; prefill GPU 61-81 %.

## B. Cross-node 1P1D P4D4 on v0.5.21 (prefill 135 GPU0-3, decode 136 GPU0-3; `tune.p4p4.yihou.md`)

21,032 (v0.5.19 same-node, C40) -> 25,524 A1 / 25,376 A2 (g3x1, C56). Steps (each bundles several changes, mostly not isolated):

| knob | evidence | note |
|---|---|---|
| Per-GPU NIC pinning `--disaggregation-ib-device '{"0":"ionic_0",...}'` + `MC_TE_FILTERS` + router `PD_DP_RANK_AFFINITY=1` | ionic rails are isolated (ionic_i reaches only ionic_i); 0 affinity-503s at C32/C40 | required on this fabric (`glm52-1p1d-samerail-c32-c40.packup_20260918/notes.md` §1) |
| Cross-node RDMA instead of same-node HIP IPC | transfer_kv p50 ~7.5 s -> ~1.1 s; per call 2.6 -> 20 GiB/s; score +14-17 %, TTFT p50 15 -> 4.5 s | same C56 config |
| Prefill mem 0.85 -> 0.88 -> 0.90 (0.92) | q1 22,306 -> pf90 A2 23,127; pf92 A1 23,003 | prefill device KV is the lever (`glm52.v0521-CONCLUSIONS.yihou.md` §2); ceiling ~0.92-0.93 [inferred] |
| Decode mem 0.95, max_running 96 | bundled (q1 22,306 at C48) | not isolated |
| OffloaderV2 on prefill (`--offload-*` group 3 / num-in-group 1 / prefetch step 1, cpu) + S06 | prefill KV +21.7 % (G3), G4 +15.6 %, G2 +32.9 %; g3x1 25,524 vs g4x1/g4x2 24,976 / 24,398 (A1); correctness 23/23 | needs S06 (#45484) |
| HiCache via GTT host pools (#42553, prefill `HSA_USERPTR_FOR_PAGED_MEM=0`), 260 GB/rank | did not shorten transfer; HiCache 50 vs 450 GB = -9.1 % (different node/build) | #42553 merged |
| Decode tokenizer / detokenizer workers 4/4 | tw90 no consistent effect | kept as a rule (memory file) |
| Concurrency | same-node optimum C48-C52; cross-node C56 > C60/C64 (G4 23,056/22,670; G3 23,905/22,860) | ISL mix shifts with C (closed loop) |
| Did not help | runshare (S08), `prefill_complete` policy (inconsistent, >10 % below q1), `write_through_selective` (-4.8 %), OffloaderV2 prefetch step 2, chunk 65536 (A2 OOM) | |

## C. DCP4 decode (v0522 overlay 93034eba91)

| knob | evidence | source |
|---|---|---|
| Decode TP4 EP4 DP1 + `--dcp-size 4`, env `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`, `_SPLIT_DECODE_INDEXER=1`, `_SPLIT_INDEXER_TRITON=1`, router `PD_DP_RANK_AFFINITY=false` | DCP4 PD runs at all (needs D-series) | `glm52.dcp4-pd.packup_20261022` |
| Mooncake `-destpin` image (llying's dest-local rail, pinned HCA, `MC_IB_TIMEOUT=18`), decode registers 4 rails | needed for cross-node DCP fan-out | `glm52.dcpx-xnode-dcp4decode.packup_20261022` |
| Prefill carries decode EAGLE flags (prefill MTP; llying) | dcpx7b 21,302 -> dcpx8c 19,997 (bundled with stats logging); value = draft-input correctness | `glm52.dcpx-pack-fix-best.packup_20261023/opt.readme.md` §3 |
| Decode graph max bs 128 -> 96 with 13 capture sizes | free mem 24.12 -> 28.47 GB; 24,753 -> 25,197 (+1.8 %, inside spread) | same §4 |
| Decode mem 0.85 -> 0.87 | no gain at C56; >0.87 needs S12 | same §5 |
| DPA4 decode on the same v0522 stack (dpax1) | 26,321 vs DCP4 dcpx11 25,197; KV 12.2 M vs 8.87 M tokens | same §6 |
| PP4 prefill + DCP4 decode, C sweep | C56 27,426 / C64 29,817 / C72 31,128 / C80 31,076 | `glm52.ppdc-pp4-dcp4.packup_20261023` |
| DCP4 decode mem 0.88 with S12 | 27,685 (same-node llying kit) | `glm52.dcp4-decode-gmu.packup_20261023` |

## D. Older (v0.5.19, September)

- Decode `--disable-custom-all-reduce` cured garbled decode with real acceptance on v0.5.19 (R05/R06); reason simulated
  acceptance was adopted for perf work (`glm52-mtp-garbled-decode-rootcause.packup_20260918`).
- Same-node P4D4 on one host needs I03 (port stride, start gate) and I06 (`DECODE_PID_OFFSET`) — harness, not recipe.
