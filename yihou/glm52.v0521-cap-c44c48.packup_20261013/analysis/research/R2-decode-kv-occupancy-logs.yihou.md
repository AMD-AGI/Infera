# R2 — Decode KV-pool occupancy under AgentX (C40 / C44), from server logs + X1 live /metrics

Date 2026-10-12 (analysis 08:08-08:40 UTC). Evidence tags: [M] first-hand (code line / log / command run here),
[S] second-hand (ledger/doc), [I] inference.

## 0. Answers (short)

1. **Peak prealloc/transfer share.** Under the `early` allocation policy, requests in the decode transfer queue
   hold their full prompt KV while prefill computes and sends it. Per DP rank, that share of the pool reached
   **p99 0.46-0.66 and max 0.67-0.84** (C40: max 0.67-0.82; C44: max 0.71-0.84). The 5-s X1 split (C44 warm)
   gives non-running KV p99 0.64 and max 0.71 of the DP pool [M].
   Non-running KV held **> 50 % of a DP's pool 0.3-1.6 % of the time at C40 warm, 1.7-5.1 % at C40 fresh, and
   2.8-4.2 % at C44 (5.15 % in the X1 5-s sampler)** [M].
   **This never hurt running requests**, because the pool was never full:
   - Per-DP used fraction stayed at max 0.79-0.96 (p99 0.69-0.81) [M].
   - The prealloc queue was empty 100 % of the time (admission never bound) [M].
   - Retractions were 0 in every v0.5.21 window at mem 0.95 [M].
   - The only window where admission bound is e1c (decode mem 0.85, pool 2.23M): prealloc>0 0.7 % of the time,
     usage max 1.00, still 0 retractions [M].
2. **C48 projection.** Decode KV is unlikely to **hard**-bind at C48 (mem 0.95). The tails will look like e1c,
   though: some time with prealloc>0 at high usage on the worst DP, and non-running KV > 50 % more often, about
   8-10 % of the time on the worst DP [I].
   Raising the load from C40 to C44 did not grow running KV (run/pool p50 0.21-0.24, flat). Only the
   transfer-queue KV grew: mean node #transfer 5.0 → 10.1, usage p50 0.32 → 0.39 [M]. The bench offers only
   ~3 % more requests/s, so a C48 run should mostly add transfer-queue residence, not decode work [I].
3. **Prefill side is now the TTFT bottleneck; KV send ("inflight") is its largest part.** Little's law closes
   to within 0.1-0.2 s (decode #run/λ vs e2e−TTFT) [M]. TTFT mean ≈ decode transfer-queue residence + ~0.8 s.
   Transfer-queue residence: 4.2-5.0 s at C40 warm → **8.35 s at C44** (both fresh and warm) [M]. The
   prefill-side split is in §5.
   Prefill compute is not saturated by tokens: about 13-20 % of peak per DP by token rate [M/I]. But at C44, half
   of the batches are full 8192-token chunks, the GPU-cache hit drops 0.92 → 0.87, and host-pool hits rise
   0.02 → 0.06 [M]. A decode-side cap on transfer requests would therefore mostly **move** the wait (to the
   decode prealloc queue or the prefill queue). It would not shorten it [I].

**X1 result (C44, 3rd run on the warm e2j stack)** [M] — `runs/e2j/agentx-c44-R3/agentx_conc44.json`:
- Score **21,773 tok/s/chip** (174,186 tok/s); ok 4,373, err 4 (InvalidInferenceResultError).
- TTFT mean/p50/p90 9.02/5.45/21.63 s; TPOT mean 11.69 ms.
- vs e2j R2 C40 (21,681, TTFT 5.21 s, TPOT 12.40 ms): **+0.4 %, below the 22,114 bar.**
- TPOT is *better* at C44 because the decode batch is smaller (mean node #running 14.2 vs 14.9). The extra
  lanes sat in the prefill/transfer path.

## 1. Method

- **Log formats.**
  - Decode line [M] (`metrics_reporter.py:937,997-1002`): `Decode batch, #running-req: N, #token: T,
    token usage: U, accept len…, pre-allocated usage: P, #prealloc-req, #transfer-req, #retracted-req, …,
    gen throughput, #queue-req`. It is logged per DP every `decode_log_interval=40` forwards.
    - `#token` = `full_num_used` = pool − free − evictable (`pool_stats_observer.py:233-236`).
    - `pre-allocated usage` = Σ `extend_end` over transfer-queue requests ÷ `max_total_num_tokens`
      (`disaggregation/decode.py:1842-1847`), i.e. the KV pre-allocated for requests still receiving KV.
    - Running KV ≈ `token usage − pre-allocated usage` (0.01 resolution).
  - Prefill line [M]: `Prefill batch, #new-seq, #new-token, #cached-token, token usage, #running-req,
    #queue-req, #pending-token, #bootstrap-req, #inflight-req, #optimistic-req, cuda graph, input throughput`.
    It carries no host-pool usage; that comes from /metrics.
  - Pool size [M]: decode 2,762,176 tok/DP (e2b4 2,773,376; e1c 2,226,368); prefill 2,270,656 tok/DP; HiCache
    host pool 10,016,064 tok/DP (450 GB).
- **/metrics semantics** [M]:
  - `kv_used_tokens` = `full_num_used` (`pool_stats_observer.py:166`).
  - `decode_sum_seq_lens` = Σ `seq_lens` of the running batch (`metrics_reporter.py:86-90,1061`).
  - The non-running part `kv_used − decode_sum_seq_lens` = transfer-queue pre-allocations + page round-up +
    reserved.
  - **`pending_prealloc_token_usage` is declared but never assigned anywhere in the source**
    (`grep -rn pending_prealloc` → `metrics_collector.py:146,617,1627` only), so it is always 0. Do not use it.
- **"Admission blocked" proxy**: `#prealloc-req > 0` [I]. It also includes handshake waits; a stricter view is
  prealloc>0 and usage ≥ 0.85.
- **Windows** = AgentX profiling phase (start from `Phase profiling … started` in each `benchmark.log`, 3600 s).
- **TTFT decomposition** = Little's law, residence = mean(Σ_DP gauge)/λ, with λ = profiled ok requests ÷ duration.

**Scripts** (all new; `/shared_nfs/yihou_glm52/v0521cap_20261012/logs/r2-scripts/`):

| script | what it does | outputs (`out/`) |
|---|---|---|
| `parse_server_logs.yihou.py` | log → per-DP CSV | `dec.<round>.csv`, `pre.<round>.csv` |
| `analyze_windows.yihou.py` | decode/prefill tables per window | `windows.md` |
| `analyze_x1_metrics.yihou.py` | X1 TSV split per DP | `x1.summary.txt`, `x1.decode_split.csv` |
| `little_ttft.yihou.py` | Little's-law TTFT decomposition | `little.md` |
| `prefill_little.yihou.py` | prefill-side residence (X1 TSV, e2g A2 aiperf timeslices) | `prefill_little.txt` |
| `prefill_load.yihou.py` | prefill computed-token load | `prefill_load.md` |
| `extract_server_metrics.yihou.py` | aiperf `server_metrics_export.json` → CSV (from the interrupted attempt; output correct and reused) | `sm.e2g-A2.csv` |

## 2. Decode pool occupancy per window (per-DP log samples) [M]

The four "p50/p90/p99/max" columns are fractions of the per-DP pool.
**Every window had #prealloc p50/p90/max 0/0/0, prealloc>0 0.0 % of the time, and max retracted 0**, except
e1c (see the first row).

| window | n | usage | pre-alloc usage | running = usage−pre | %t pre>.3 / >.5 | #run/DP p50/p90/max | #transfer/DP p50/p90/max | DP usage spread p50/p90 |
|---|---|---|---|---|---|---|---|---|
| e1c-r2 C40 (dec mem .85)* | 7899 | .41/.70/.93/**1.00** | .02/.30/.63/.87 | .30/.60/.85/1.00 | 10.0 / 3.1 % | 4/6/13 | 1/4/12 | .35/.57 |
| e2b4 C44 fresh | 7294 | .41/.62/.75/.88 | .07/.36/.61/.71 | .24/.48/.66/.75 | 14.8 / 3.0 % | 4/7/16 | 1/6/**21** | .25/.41 |
| e2c C44 fresh | 7137 | .40/.62/.81/**.96** | .05/.34/.63/**.84** | .26/.50/.70/.92 | 13.0 / 2.8 % | 4/7/16 | 1/5/17 | .28/.51 |
| e2c C40 warm | 8634 | .33/.53/.69/.79 | .03/.25/.47/.71 | .23/.45/.63/.78 | 6.2 / 0.8 % | 3/6/15 | 1/4/12 | .25/.42 |
| e2g A1 C40 fresh | 7573 | .36/.57/.72/.91 | .05/.32/.55/.71 | .23/.47/.64/.78 | 10.7 / 1.7 % | 4/7/14 | 1/5/15 | .26/.41 |
| **e2g A2 C40 warm (22,114)** | 8824 | .32/.55/.70/.89 | .03/.24/.51/.67 | .22/.45/.61/.80 | 6.5 / 1.1 % | 3/6/15 | 1/4/13 | .26/.45 |
| e2i R1 C40 fresh | 7740 | .34/.60/.78/.88 | .03/.33/.66/.82 | .23/.48/.64/.77 | 10.8 / 5.1 % | 3/7/15 | 1/5/13 | .25/.43 |
| e2i R2 C40 warm | 8633 | .32/.55/.73/.83 | .03/.26/.56/.76 | .23/.46/.67/.82 | 6.7 / 1.6 % | 3/6/15 | 1/4/14 | .27/.46 |
| e2j R1 C40 fresh | 7319 | .35/.55/.72/.85 | .04/.28/.58/.81 | .24/.47/.64/.80 | 8.9 / 1.6 % | 4/7/15 | 1/5/11 | .27/.43 |
| e2j R2 C40 warm | 8466 | .32/.54/.69/.83 | .02/.23/.46/.70 | .24/.46/.65/.80 | 5.4 / 0.3 % | 3/6/16 | 1/4/12 | .26/.44 |
| **X1 e2j R3 C44 warm** | 8795 | .39/.59/.73/.87 | .09/.38/.64/.73 | .21/.45/.61/.78 | 16.5 / 4.2 % | 3/6/16 | 2/6/15 | .26/.39 |
| c44c C44 (decode MAF 09:21:20) | 899 | .09/.40/.70/.80 | .00/.20/.55/.70 | .07/.16/.40/.59 | 6.6 / 2.5 % | 2/4/8 | 0/4/11 | .11/.37 |
| c44d C44 (prefill MAF 10:25:09) | 1332 | .28/.62/.83/.93 | .03/.31/.55/.77 | .12/.48/.78/.83 | 10.1 / 1.6 % | 3/8/16 | 1/7/14 | .20/.38 |

\* e1c is the only window with admission binding: prealloc>0 0.7 % of the time (0.5 % at usage ≥ 0.85), and
prealloc max 2.

**Running-slot cap.** `max_running_requests=16` per DP (`max_running_requests=16` in the log). #running reached
16 on a DP in only 0-0.09 % of samples; ≥12 in 0.2-0.9 % [M]. Running+transfer peaked at 16-25 per DP.

**Old C44 rounds before the fault** [M]:
- c44c: decode MAF at 09:21:20 (6 min into profiling). The last samples per DP 2-3 s before the fault show
  usage 0.32/0.58/0.62/0.69, max 0.80 in the prior 120 s. DP3 had 1 running and 8 transfer requests holding 0.64
  of its pool, so the pool was not full.
- c44d: prefill MAF at 10:25:09. Decode usage was ≤ 0.48 with 0 transfer requests at the fault.
- c44b and c44e decode logs contain no `Decode batch` lines.
- No evidence that either fault was KV-pool exhaustion [I].

## 3. X1 (C44 warm) — 5-s /metrics split, 07:05:40-08:05:40 UTC [M] (`out/x1.summary.txt`)

Same column layout as §2: the four "p50/p90/p99/max" columns are fractions of the DP pool.

| DP | used | running | non-running | %t non-run >.3 / >.5 | non-run ÷ used max (used ≥.3) | #transfer p50/p90/max | min free tok |
|---|---|---|---|---|---|---|---|
| all | .392/.597/.728/.869 | .217/.460/.636/.781 | .085/.397/.641/.710 | 16.9 / 5.15 % | 1.00 | 2/6/15 | 363,136 |
| 0 | .380/.563/.653/.708 | .212/.439/.524/.641 | .086/.371/.612/.645 | 14.0 / 3.3 % | .97 | 2/6/13 | 807,488 |
| 1 | .398/.663/.765/.824 | .195/.494/.682/.781 | .062/.449/.674/.710 | 18.9 / 8.3 % | 1.00 | 1/6/13 | 486,464 |
| 2 | .404/.625/.788/.869 | .229/.450/.639/.776 | .096/.426/.631/.631 | 21.0 / 7.5 % | .99 | 2/6/13 | 363,136 |
| 3 | .380/.569/.660/.720 | .230/.441/.577/.609 | .081/.332/.510/.585 | 13.8 / 1.4 % | .97 | 2/8/15 | 773,824 |

**Node sums.**
- used/pool p50/p90/p99/max 0.39/0.53/0.62/0.67; non-running 0.13/0.31/0.45/0.58.
- #running total p50/p90/max 14/23/36, mean 14.5; #transfer total 9/20/35, mean 9.8.
- #prealloc always 0; retracted 0; `pending_prealloc_token_usage` 0 (never set).

**Per-request context.**
- Running-request context Σseq_len/#run: p50 164k, p90 328k, mean 189k tokens.
- Non-running tokens per transfer request: p50 146k, p90 305k.
- Bench ISL: mean 149k, p50 103k, p90 285k, p95 514k, max 780k (= 0.28 of one DP pool); 258 of 4,857 requests
  are > 500k (profile_export.jsonl).

**Trend over the hour** (10-min bins, node mean).

| minutes | used | non-run | #run | #transfer | worst-DP used |
|---|---|---|---|---|---|
| 0-10 | .28 | .12 | 11.9 | 10.1 | .38 |
| 30-40 | .44 | .19 | 16.6 | 11.9 | .60 |
| 50-60 | .43 | .16 | 15.3 | 10.5 | .58 |

**Prefill leg (X1).**
- cache_hit_rate p50 0.97-0.98 per DP.
- HiCache host pool rose from 0.34-0.41 at profiling start to **1.00 on all DPs**: first full at t = 2,662 /
  2,847 / 3,082 s (DP3/2/0); DP1 was already full before the window.
- At sampler start (warmup) the host pool read 0.79-1.00, and it *fell* to 0.34-0.41 by profiling start. That
  host-pool release mechanism is open.
- e2g A2 for comparison (aiperf timeslices): host used 0.30-0.35 → 0.88-1.00 over the hour [M].

## 4. Bench outcome + Little's-law TTFT decomposition [M] (`out/little.md`)

| window | tok/s/chip | ok/err | λ req/s | ISL mean | TTFT mean/p90 s | TPOT ms | e2e s | mean #run | mean #transfer | #run/λ | e2e−TTFT | #transfer/λ | λ·e2e | GPU/CPU hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e2b4 C44 fresh | 20,347 | 4138/1 | 1.141 | 142k | 9.01/21.5 | 14.19 | 23.07 | 16.2 | 9.5 | 14.19 | 14.06 | 8.35 | 26.3 | .911/.032 |
| e2c C44 fresh | 20,524 | 4179/2 | 1.151 | 142k | 8.03/20.0 | 15.11 | 22.82 | 16.7 | 8.3 | 14.51 | 14.79 | 7.19 | 26.3 | .907/.040 |
| e2c C40 warm | 21,696 | 4237/3 | 1.167 | 148k | 5.61/11.9 | 12.17 | 18.13 | 14.6 | 5.5 | 12.47 | 12.52 | 4.75 | 21.2 | .929/.017 |
| e2g A1 C40 fresh | 19,640 | 3993/4 | 1.100 | 142k | 7.61/20.3 | 14.33 | 21.74 | 15.3 | 7.6 | 13.91 | 14.13 | 6.89 | 23.9 | .941/.017 |
| e2g A2 C40 warm | 22,114 | 4286/1 | 1.181 | 149k | 5.30/11.5 | 12.05 | 17.60 | 14.3 | 5.3 | 12.14 | 12.30 | 4.47 | 20.8 | .926/.021 |
| e2i R2 C40 warm | 21,921 | 4258/1 | 1.176 | 148k | 5.83/13.6 | 12.28 | 18.24 | 14.5 | 5.9 | 12.33 | 12.41 | 5.02 | 21.4 | .920/.025 |
| e2j R1 C40 fresh | 19,335 | 3938/2 | 1.086 | 141k | 7.20/20.4 | 14.66 | 21.81 | 15.7 | 7.1 | 14.44 | 14.61 | 6.56 | 23.7 | .938/.016 |
| e2j R2 C40 warm | 21,681 | 4242/1 | 1.171 | 147k | 5.21/10.6 | 12.40 | 17.97 | 14.9 | 5.0 | 12.69 | 12.75 | 4.24 | 21.0 | .924/.021 |
| **X1 C44 warm** | **21,773** | 4373/4 | 1.207 | 143k | 9.02/21.6 | **11.69** | 20.66 | 14.2 | 10.1 | 11.78 | 11.64 | **8.35** | 24.9 | .870/.064 |

**Readings.**
- #run/λ matches e2e−TTFT within 0.2 s in every window, which validates the method [M].
- TTFT − #transfer/λ ≈ 0.7-1.0 s throughout (pre-admission + first token) [M].
- λ·e2e = 21-26 requests in system at CONC 40/44. AgentX lanes spend about half their time idle between turns
  (`--trace-idle-gap-cap-seconds 300` in `benchmark_command.txt`) [M command / I semantics].
- So +4 lanes added only +3 % λ (1.171 → 1.207) and nearly doubled transfer-queue residence.
- Fresh-vs-warm: fresh runs have TPOT +2 ms (14.2-15.1 vs 12.0-12.4) and lower ISL mean (142k vs 148k). This
  matches the open fresh-stack penalty [S: R046/R049].

## 5. Prefill side [M] (`out/prefill_little.txt`, `out/prefill_load.md`)

Residence = mean Σ_DP gauge ÷ λ, from unbiased time-sampled gauges:

| phase (prefill gauge) | e2g A2 C40 warm | X1 C44 warm |
|---|---|---|
| queue (`num_queue_reqs`) | 0.97 s | 2.26 s |
| bootstrap | 0.02 s | 0.02 s |
| forward/chunking (`num_running_reqs`)† | 2.06 s | 3.05 s |
| **KV send (`num_prefill_inflight_queue_reqs`)** | **2.95 s** | **5.25 s** |
| decode transfer-queue residence | 4.54 s | 8.19 s |

† The prefill `num_running_reqs` gauge can hold a stale last-batch snapshot (`metrics_reporter.py:1436-1441`
comment), so the three prefill phases can sum to more than the decode transfer residence (they also overlap in
time) [I].

- The largest prefill-side phase, and the one that grows most C40 → C44, is **KV send (inflight)**.
- Consistent with [S] R032-R035 (one serialized transfer worker per rank; peer copies stall behind prefill
  kernels via shared HW queues; lever `MC_HIP_STREAM_PRIORITY=high` on the mc4340prio image, unvalidated).

Prefill load [M]:
- Computed (uncached) tokens are only ~1.2-2.6k tok/s per DP, i.e. 10-20 % of the logged peak input
  throughput of ~12.8k tok/s/DP [M]. By token rate, prefill compute is far from saturated [I].
- At C44, though, 52 % of prefill batches are full 8192-token chunks (vs 39-42 % at C40). Cached fraction drops
  0.960 → 0.947. GPU hit drops 0.92 → 0.87 while host hit rises 0.02 → 0.06. The host pool hits 100 % on all
  DPs by t≈45-50 min [M].
- Prefill GPU KV (2.27M/DP) and HiCache host (10M/DP) are both under pressure at C44 warm [I].

## 6. Implications for Tracks A/B [I]

- **Track B.** "Running requests always keep ≥ 50 % of the decode pool" is violated briefly but routinely:
  non-running > 50 % of a DP pool 0.3-5 % of the time today, likely ~8-10 % on the worst DP at C48. It has
  never cost anything yet: no retraction, no admission block, free tokens ≥ 363k per DP. Capping
  transfer-queue KV at 50 % of the pool would have engaged only in those few-% windows. It would delay the
  KV send of a long prompt, not save decode work. Its value is insurance against e1c-style admission stalls
  and retraction at C48 with 500k+ prompts, not throughput.
- **Track A.**
  - The C44 limiter is the prefill→decode path: KV send 5.3 s, plus queue 2.3 s, plus a larger uncached
    fraction. Decode is not the limiter at C44: TPOT improved, the KV pool was never full, and the running
    batch was flat.
  - Levers ranked by this evidence:
    - (1) KV-send efficiency (stream priority / HW queues, R035);
    - (2) prefill GPU cache capacity (mem-fraction 0.85 → higher) to recover the 0.92 GPU hit;
    - (3) HiCache host pool larger than 450 GB, or a fix for its fill-to-100 %, if (2) is insufficient.
  - Raising decode mem-fraction (0.96/0.97) buys decode headroom that C44 does not need. It may matter at C48
    tails.
- **Open:**
  - why the host pool drains during warmup and refills to 100 % during profiling;
  - true prefill busy fraction (needs per-batch forward time, not token rate);
  - whether the inflight residence is transfer wall time or waiting for decode acknowledgement.
