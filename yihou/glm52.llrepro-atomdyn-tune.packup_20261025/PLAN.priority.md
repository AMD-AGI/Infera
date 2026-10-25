# Priority plan after the ATOM / Dynamo+SGLang review (task llrepro, 2026-10-24 10:5xZ)

User (10-24): get the DCP session's ATOM (~48K) / Dynamo+SGLang (~55-64K) analysis, combine with our own state, rank
by (probability of a gain x expected size), then develop / debug / run them one by one.

Sources: Infera e186e8e5 `yihou/glm52.atom-dynamo-vs-sglang.analysis_20261023/REPORT.md` (P1-P28, none measured on our
stack; DCP session confirms nothing newer, 10-24); our llr7r trace report
`/shared_nfs/yihou_glm52/llrepro_20261023/analysis_llr7r.yihou/REPORT.md`; llr7 queue series (`qseries.yihou.py`).
Reference: llr7 = 41,679 (llying 4+4 kit + KF + decode custom AR + TOPK_V2=false), C80, 3600 s, sim AL 3.61, A1 only.

Measured facts the ranking leans on: decode KV usage mean 0.81-0.83, >= 0.95 25-28 % of the time; from mid-run usage
0.89-0.91, transfer-req 13 -> 25, running 54 -> 33, prefill batches/s 2.1 -> 1.2; decode step 51 ms (verify <= 42 ms,
contents unrecorded); prefill PP stages 71-76 % busy (88-101 ms CPU-handshake gaps); RDMA 8 % of line rate.

| rank | id | change (one variable vs current best) | targets | why ranked here | status |
|---|---|---|---|---|---|
| 1 | A1 | decode graph capture set: user 10-24 "64 too small, 96 with a larger step" -> llr9b `--cuda-graph-max-bs-decode 96 --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 40 48 64 80 96` (llr9 with 64 stopped before AgentX) | decode memory headroom (prereq for A2) | measured at start-up: per-rank capture is capped at 64 (= max_running 256 / dp 4) in both llr7 (27 sizes 1..64) and llr9b (11 sizes 1..64; 80/96 dropped); free after capture TP0 10.46 -> 11.35 GB | llr9b running |
| 2 | A2 | decode mem 0.92 -> highest stable (0.94/0.95), only after A1 shows the freed GB; empty_cache port (`ec_port.yihou/`) as fallback | decode KV capacity (the clearest measured limiter) | ATOM runs decode mem 0.94; IX C264 0.94 | queued |
| 4 | B | decode radix: `--disaggregation-decode-enable-radix-cache` + router `--session-affinity both` (2 changes; without `both` the decode DP pick ignores prefix, verified code read wf_f01a52a6-e4e) | transfer volume, prealloc, TTFT | experimental with DPA; risks: index_share seed metadata on reused prefix (disaggregation/utils.py:130-138), KF overlay, rank imbalance from sticky sessions; real-acceptance smoke first | code read done |
| 3 | C | prefill `--pp-async-batch-depth 1` (overlap schedule is force-disabled for pp>1, overrides.py:1625; PP loop = event_loop_pp_disagg_prefill) ; later SGLANG_PP_COMM_OVERLAP=1 (needs a prefill-env hook), SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1 | PP inter-step gaps 26-29 % | llying measured +8.5 % prefill capacity (e8227d61) with GSM8K correctness passed (7a9ba815 handoff) | code read done |
| 5 | D | decode `--stream-interval 30` | decode host CPU (threads at 90-100 %) | IX GB300; cheap; effect size unknown | queued |
| 6 | E | `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` on decode | admission reserve | IX GB300; code read: default 4096 tokens/request reserve in decode admission (decode.py:494,1534,1976,2100), ~1-2 % of KV at 10 req/rank, so small expected | queued |
| 7 | F | verify-graph kernel breakdown (rocprofv3 or graph-off profile) -> kernel targets (P15 MLA bmm configs, gfx950 fused indexer rows <= 48, KF at 180k contexts) | decode step 42 ms in verify | diagnostic; kernel session owns kernels | after A-D |
| 8 | G | INT4 quick reduce (P9), MTP K3 (P10) | AR time / verify rows | correctness / comparability risk | later |
| - | - | decode prefix-hit / gpu_hit difference llr (0.37-0.49) vs ppda (0.70-0.74) | - | unexplained; analysis only | open |

Each run: one variable, guard_llr.yihou.sh (stops our AgentX if the decode stops logging batches / logs a fatal),
qps + TTFT/TPOT + queue series + ISL change, not score alone.
