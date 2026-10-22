# R2 verification (adversarial) — R2-decode-kv-occupancy-logs.yihou.md + author summary

Date 2026-10-12. Verifier re-derived everything below independently (own parser `/tmp/yihou_r2v/dec.py`, own
TSV/JSON readers); the original report is not edited. Tags: [M] first-hand, [S] second-hand, [I] inference.

## Adversarial verification

### Confirmed (re-derived first-hand)

1. **Decode log `pre-allocated usage` = Σ extend_end over transfer-queue reqs (excluding host_staged) / max_total_num_tokens** [M]
   `srt/managers/scheduler_components/metrics_reporter.py:997` uses
   `disagg_decode_prealloc_queue.num_tokens_pre_allocated`; `srt/disaggregation/decode.py:1841-1847` sums
   `decode_req.req.extend_end for … in self.transfer_queue.queue if not decode_req.host_staged`;
   `decode.py:2208` sets `extend_end = req.kv.kv_committed_len` at allocation (full prompt reserved under `early`).
2. **`#token`/`kv_used_tokens` = pool − (available + evictable)** [M] `pool_stats_observer.py:233-236`, `:166`.
3. **`pending_prealloc_token_usage` is never assigned** [M] `grep -rn "pending_prealloc\|prealloc_token_usage"` over the
   whole package → only `observability/metrics_collector.py:146,617-618,1627` (default 0.0, gauge decl, export).
4. **`decode_sum_seq_lens` = Σ seq_lens of running batch** [M] `metrics_reporter.py:86-90,1061`.
5. **Per-DP running cap 16, decode pool 2,762,176 tok/DP, decode_log_interval 40, host pool 10,016,064 tok (450 GB)** [M]
   e2g decode log `max_running_requests=16`, `max_total_num_tokens=2762176`, `'decode_log_interval': 40`; prefill log
   `Allocating kv hierarchical KV host pool: 10016064 tokens, 450.00 GB`.
6. **§2 e2g A2 row** [M] own parse, window 07:30:53+3600 s (benchmark.log `Phase profiling … started`): n 8824; usage
   .32/.55/.70/max .89; pre .03/.24/.51/.67; running .22/.45/.61/.80; pre>.3 6.5 %, >.5 1.13 %; prealloc>0 0; retracted
   max 0. Exact match.
7. **§2 X1 row** [M] own parse 07:05:39+3600 s: n 8792 (report 8795, boundary second), usage .39/.59/.73/.87, pre
   .09/.38/.64/.73, pre>.5 4.24 %, prealloc>0 0, max running 16 (0.09 % of samples), max transfer 15. Match.
8. **§2 e1c row (only binding window)** [M] n 7899; usage max 1.00; pre>.5 3.14 %; prealloc>0 55/7899 = 0.70 %
   (prealloc max 2); retracted 0; pool 2,226,368. Match.
9. **Retractions = 0** [M] — stronger than the report's gauge-based claim: the `#retracted-req` field is a queue-length
   snapshot, but `scheduler.py:4209-4260` logs `"KV cache pool is full. Retract requests."` on every retraction, and
   that string occurs 0 times in e1c/e2g/e2j decode logs (the only non-batch `retract` hit is the server_args dump).
10. **§3 X1 TSV split** [M] own recompute: non-running = (kv_used − decode_sum_seq_lens)/2,762,176 > 0.5 in 5.15 %,
    > 0.3 in 16.9 %, max 0.710; min kv_available 363,136; prealloc-queue and retracted gauges max 0; 2,876 samples.
11. **§5 prefill residences X1** [M] mean Σ_DP gauge ÷ λ(4373/3630 s): inflight 5.24 s, queue 2.25 s, running 3.04 s. Match.
12. **Scores** [M] `request_metrics.throughput.per_gpu.total_tput_tps`: X1 21,773; e2j R2 21,681 (+0.42 %); e2g A2 22,114;
    A1 19,640 (A2/A1 = +12.6 %); e1c 21,049; e2e 19,565; e2f 19,864; e2i R1 20,371; e2j R1 19,335; e2b4 20,347;
    e2c C44 20,524; e2c C40 21,696; e2d 21,383. GPU hit A2 0.926 → X1 0.870 (`server_metrics.cache.gpu_cache_hit_rate`).
13. **v0.5.19 21,032 and +5.1 %** — 22,114/21,032 = 1.051 [M arithmetic]; 21,032 itself [S] (ledger R028
    `working_process.md:600-601`; the 0923 packup is not on this host, not re-read).

### Refuted / needs correction

R-a. **Summary §5 "0.5.21 上开 index_share：21,383，没有收益"** — protocol-inconsistent. e2d is a **fresh** first run
   (ledger R049 `working_process.md:921-922`, [S]) with the full target config (audit/*-server-info.json [M]: prefill
   mem .85 + hicache_size 450, decode mem .95, `index_share_for_mtp_iteration: true`). Under the summary's own
   "same-protocol (fresh) comparison" rule, it beats every target-config fresh run with index_share=false
   (19,335-20,371) by +5-11 %, and beats v0.5.19 fresh 21,032 (also index_share TRUE) by +1.7 %. It is only "no gain"
   vs the *warm* e2c C40 21,696. Confounds: n = 1, and e2d's decode leg also ran the ROCm debug agent
   (`HSA_ENABLE_DEBUG=1`) + seedguard patch ([S] `working_process.md:923-925`). The ledger itself names index_share as
   the leading candidate for the fresh-stack penalty ([S] `:975-977`). Verdict: index_share is **open**, possibly the
   single most important untested lever — not "no gain".
R-b. **Summary §3 "目标配置 fresh 首跑只有 19.3-20.4k … 比宽松配置 fresh e1c 21,049 还低"** — incomplete: omits e2d
   (target config, fresh, 21,383 > e1c). The counter-example holds only for index_share=false.
R-c. **Summary §2 "e1c = 0.5.21 宽松配置" as the version-only comparison** — e1c ran with `enable_dsa_fused_indexer: False`
   on both legs [M audit json], i.e. 0.5.21 *without* its only default-on new feature. So e1c vs 0.5.19 measures
   "0.5.21 minus fused indexer, index_share off, hicache ratio 1.5, decode .85" — it neither isolates the version nor
   includes its new default. The "+0.1 %" number is right; the label is not.
R-d. **Summary "e2e is target config"** — minor: e2e prefill had `chunked_prefill_size 16384` (others 8192) [M audit json].
R-e. **Report §6 "Decode is not the limiter at C44"** — not refuted, but under-supported: it rests on TPOT and pool
   occupancy, not on decode busy fraction, and it is not reconciled with R042 (C40 decode busy 93-96 %) [S]. A
   smaller running batch with lower TPOT is equally consistent with a decode leg that is still near-saturated by
   per-step fixed cost. Mark [I], not established.

### Unverifiable here

- C48 projection "non-running > 50 % on worst DP ~8-10 % of time" [I, no data].
- "one serialized transfer worker per rank; HW-queue stall" (R032-R035) [S].
- v0.5.19 HiCache "≈154 GB", accept 3.61, cross-node −5.6 %, triton+index_share +12.5 % [S, packup not on host].
- Whether `inflight` residence is wire time or wait-for-decode-ack (report itself flags open).
- Host-pool drain during warmup (report flags open).

### Important points the report / summary missed

1. **The score is ~99 % input tokens** [M]: A2 per_gpu input 21,957 of 22,114 total. So score ≈ λ × mean ISL (cached
   tokens included). X1 vs e2j R2: λ ×1.031 (4373/4242) × ISL ×0.975 (143,345/147,086) = ×1.005 ≈ observed +0.42 %.
   C44's gain was eaten mostly by a *shorter request mix*, not only by prefill/KV-send residence. Any C44/C48 comparison
   must normalise for ISL mix (aiperf's served-request mix is load-dependent), otherwise ±3 % swings are noise.
2. **`--disaggregation-decode-allocation-policy prefill_complete` exists in this image** [M]
   `arg_groups/fields/disagg.py:59-64` ("retains KV on prefill until prefill finishes, then admits it on decode; set the
   same policy on both roles"). This is the built-in answer to Track B (non-running decode KV would drop to ~0). Its
   validator `arg_groups/pd_disaggregation_hook.py:204-249` rejects: attention-TP > 1 (we have tp 4 = dp 4 → OK), decode
   radix/HiCache (OK), host_receive_threshold > 0 (we use 0.0 → OK), prefill HiCache other than L2 `write_back`
   (needs check of `hicache_write_policy`), staging buffers, Rust server; `serving_hook.py:498-503` additionally requires
   optimistic prefill. R2 does not mention it (R1/R5 do); its cost is moving KV residency to the prefill pool, which
   R2 shows is already under pressure at C44 (GPU hit .92 → .87) — this trade-off is the key Track-B question.
3. **Fresh vs warm asymmetry undermines the C44 inference**: X1 is a 3rd-run warm C44, but the C44 references
   (e2b4/e2c) are fresh; there is no fresh-vs-fresh or warm-vs-warm C44-vs-C40 pair except e2j R2 → R3, which is n = 1
   and confounded by stack age.
4. **No per-DP prefill busy fraction** (token-rate 13-20 % of peak is not utilisation; long-context attention cost is
   superlinear in context, so cached-heavy 150k-token chunks can saturate prefill at low new-token rate).
5. **Non-running KV includes per-running-request reservations** (`num_reserved_decode_tokens 512`, EAGLE draft slots,
   page round-up). Small (~14 × 512 ≈ 7k tok/node), but the ">50 % non-running" metric slightly overstates transfer KV.

### Net verdict

The R2 occupancy numbers, code citations and X1 score reproduce exactly. The decision-relevant errors are in the
22k-attribution summary: index_share is mis-classified as "no gain" (fresh-vs-fresh it is the only fresh run without
the ~10 % penalty), e1c is not a clean "0.5.21" point (fused indexer OFF), and the C44 throughput story omits that the
score scales with ISL mix. Track B should evaluate the existing `prefill_complete` policy before writing a new cap.
