# Gap analysis: ppda1 (34,288) vs llying 4+4 C80 (37,310), 2026-10-23

Produced by a 16-agent read-only workflow (diff readers + adversarial verifiers + planner). Paths cite the local copies used at the time (/tmp/yihou/gap/...).

# Plan to close the 4+4 C80 gap: ours ppda1 34,288 vs llying 37,310 tok/s/GPU (+8.8%)

## What the gap is made of
All of the gap comes from the time to first token (TTFT). Decode speed is the same on both sides.

| | ours | llying |
|---|---|---|
| Completed requests | 8,605 | 9,264 |
| Requests per second | 2.372 | 2.553 |
| TTFT mean / p90 | 9.10 / 22.95 s | 5.53 / 10.83 s |
| Time after first token (e2e minus TTFT) | 19.36 s | 19.31 s |

- **Source:** both `agentx_conc80.json` files. VERIFIED.
- **Where the extra TTFT sits:** waiting for the decode side to pre-allocate KV (`per_stage_req_latency_seconds{stage=decode_bootstrap}`, per DP rank).
  - Ours: 3.69 / 7.45 / 5.20 / 3.60 s.
  - Llying: 1.19 / 0.96 / 1.65 / 1.22 s.
  - That is about 3.8 s more for us, matching the 3.6 s TTFT gap. VERIFIED (`server_metrics_export.csv` lines 1035-1050, `/tmp/yihou/gap-wf/hist_cmp.txt`).
- **Not the pool size:** the decode KV pool is the same on both sides (3.04M vs 3.00M tokens per rank).
- **The queue sits on one full rank while others have room:**
  - The rank with a waiting queue is itself at ≥0.9 KV usage in 75.8% of cases.
  - In 93.4% of cases another rank is below 0.8 usage with an empty queue (`/tmp/yihou/gap-wf/pq3.py`). VERIFIED.
  - Our 4 decode ranks each received exactly 2,379 requests. Llying's received 2708 / 2923 / 2152 / 2418. VERIFIED (histogram counts). So our router splits by request count, not by KV load. The mechanism is INFERRED.
- **Prefill compute, KV transfer and per-token decode time are equal on both sides.**

## One-variable experiments on top of ppda1, in order

### E1. Router R2: balance decode DP ranks by KV demand
- **Change:** add `-e INFERA_R2_DECODE_DEMAND=on` to the router `docker run` (dryrun line 29). Alternatively add the CLI flag `--decode-input-demand on`, but only if `python3 -m infera.server --help` inside our image lists it.
- **Status of the flag in our router:**
  - In the repo Rust source, `rust/router/src/routing_experiments.rs:83-97` reads `INFERA_R2_DECODE_DEMAND`. It accepts off / shadow / on (`:17-24`) and defaults to off. VERIFIED.
  - I found no CLI wiring for it in this repo's `python/` or `rust/`. Neither the flag name nor `DP_RANK_AFFINITY` appears there (grep, VERIFIED), so the repo snapshot is older than the image's router.
  - Pre-check, read-only: `docker run --rm <image> python3 -m infera.server --help | grep -i demand`, and look for the R2 env in the router log at start-up.
- **Expected direction:** decode_bootstrap drops toward about 1.2 s, TTFT p90 drops, and requests per second rise. INFERRED.
  - Llying's README credits R2 with sending new requests to other ranks (lines 635, 283: P8D4 went from 19.3k to 23.1k with R2).
  - The README also says balancing helps little at the KV-full 4+4 C80 point (line 876). So treat the expected effect as "medium, unmeasured".
- **What confirms it:**
  - decode_bootstrap per rank, and prefill_bootstrap.
  - `num_decode_prealloc_queue_reqs`: sum mean (now 10.7) and per-rank max (now 20/41/37/30).
  - Fraction of time some rank is above 0.9 KV usage (now 71%).
  - Requests completed per rank should no longer be exactly equal.
  - TTFT p90 and tok/s/GPU.
- **Risks:**
  - No correctness risk, since this is routing only.
  - No OOM risk.
  - Since our decode does not report `kv_block_size` (it is null), check that R2 still costs by tokens rather than falling back (README line 960). Verify this from the router log or the shadow metrics. Optionally do a short pre-run with `=shadow`.

### E2. `SGLANG_OPT_USE_TOPK_V2=true` on both legs
- **Change:** replace `SGLANG_OPT_USE_TOPK_V2=false` with `true` in the prefill (line 18) and decode (line 26) docker env.
- **Why it is live on our stack:** without the explicit setting, the overlay forces it False on HIP (`model_hook.py:433-436`). With it true, the PAGED and RAGGED top-k paths switch to the v2 kernel (`dsa_topk_backend.py:55-143`). Our servers report `dsa_topk_backend=sgl-kernel`. VERIFIED.
- **Expected direction:** throughput up.
  - Llying's own same-node A/B gave +8.6% with ITL p50 25.8 → 23.6 ms (mix README:111-138, run-to-run noise ±3.5%). That was the mixed example at C48, not PD at C80.
  - Our gap is in TTFT, not ITL. So it is open whether this attacks our gap at all; it could also be part of why llying's ITL matches ours at a higher load.
- **What confirms it:** ITL p50/p90, prefill forward stage time, tok/s/GPU.
- **Risks:**
  - **Correctness:** an older note says v2 is "garbage on gfx950" (`examples/sglang_1p1d_glm5.2/README.md:299`). Llying later measured GSM8K 0.967-0.98 on their image. On our overlay plus patches it is unverified, so a GSM8K run with real acceptance is required before adopting.
  - No OOM risk expected.
- **Order:** run after E1, because it does not target the measured bottleneck. Run it first instead if a slot frees up before E1's pre-check is done.

### E3. Router R1 and session affinity, plus the client header (as one bundle, only if E1 leaves a gap)
- **Change:** `--pd-prefill-guard-release completion` (or `-e INFERA_PD_PREFILL_GUARD_RELEASE=completion`) plus `--session-affinity prefill`.
- **R1 guard release:** a code read on fdbafb60 shows it only changes load accounting, and with one prefill worker it cannot change the prefill pick. Expected impact none.
  - Open caveat: `drain_prefill_early()` (`routing_experiments.rs:79-80`) is also switched by it, and that path (`disagg.rs:231`) was not read.
- **Session affinity:** also expected to have no effect with one prefill worker. INFERRED.
- **Client header:** `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true` exists only on llying's client. It feeds session affinity. Add it together with this bundle so both protocols match.
- **Metrics:** `infera_router_session_*` (all 0 today), TTFT.

### E4. Decode `--disable-custom-all-reduce`
- **Change:** add the flag to the decode command (line 26).
- **Expected:** small, sign unknown. It affects decode step time, which is already equal.
- **What confirms it:** ITL, decode forward time.
- **Risk:** none for correctness; no OOM risk.

### E5. Decode `--max-running-requests 256 --cuda-graph-max-bs-decode 256`, with mem 0.92 to match llying
- **What is known:**
  - The running cap rarely binds: busiest rank ≥24 in 3 of 726 samples. VERIFIED.
  - But the cap also changes `disaggregation_decode_extra_slots`: 48 for us (24 per worker ≤ 32) vs 0 for llying (`pd_disaggregation_hook.py:133-139`). Read in code; effect unmeasured.
- **Expected:** low.
- **Risks:**
  - About 6.6 GB more graph memory per rank (llying measures 13.14 vs our 6.56 GB). At mem 0.95 that cuts KV by about 2% or risks OOM.
  - Changing 256/256 together with mem 0.92 is three knobs at once. If you insist on one variable, change only `--max-running-requests 256` and keep graph bs 64. Batches above 64 per rank never occur, so the risk is low.

### E6. Prefill `HSA_NO_SCRATCH_RECLAIM=1`
- **Expected:** low. Prefill forward time is already equal.
- **Metric:** prefill forward stage time.

## Differences that do not matter, or only in theory
- **R1 guard release and session affinity in the 1P topology:** cannot change any pick. Code read, VERIFIED (`policy.rs:743-754`; llying's `workers.json` has one prefill worker).
- **Decode KV pool and mem 0.95 vs 0.92:** pool is +1.4% for us, so not a cause. VERIFIED.
- **Decode running cap and graph bs for the wait itself:** not the cause of the pre-allocation wait. The queued rank is at the cap in 0.3% of cases, and every batch size we run is captured. VERIFIED.
- **Mooncake 4 vs 8 rails:** our per-request transfer is faster (2.7-4.8 vs 1.9-3.2 GB/s) and the transfer stage time is equal. VERIFIED.
- **Prefill config:** same KV, same flags, same forward and queue times. HiCache 260 vs 250 GB and our conn.py patch show no gap in the prefill stages. VERIFIED from the stage histograms.
- **`INFERA_PD_DP_RANK_AFFINITY=false`:** not read anywhere in this repo's `rust/` or `python/` (grep). In the image's router its effect is unknown, so it is not ruled out.
  - Possible extra E1b: drop the env. Note that `dp_rank_affinity=false` combined with exactly equal per-rank counts is consistent with a count-based pick.
- **Decode `fake_output` stage, 58 ms vs 0.5 ms:** cause unknown. It is about 1% of TTFT, so it does not explain the gap.
- **`index_share` true vs false, detokenizer 4 vs 1:** not tested. ITL is equal, so a low effect is expected.

## Things that weaken the comparison
- **Images differ:** `v0522-1020-yihou-dcp-mc4340-destpin` + overlay 93034eba91 vs `v0522-fdbafb60`. The router binaries differ too: llying's flags in fdbafb60 `config.rs` vs our unverified image router.
- **Llying's router flags in the C80 run are INFERRED:** they come from kit `up.sh:51` and README:57. There is no router log or metrics from that run.
- **Server-side numbers on llying's side include warmup:** the `.prom` snapshots are cumulative (counts above the client's request count). Server-side TTFT and the cache-hit metrics are not comparable. Use only the client-side numbers and the stage histograms (both also include warmup).
- **Pre-allocation queue numbers are measured differently:** llying uses `decode_kv.py` on log lines in 10 s bins (sum over ranks); we sample `/metrics` every 5 s. The gap (max 52 vs 25, mean 10.7 vs 1.6) is far larger than that method difference.
- **Metadata disagreement:** `agentx_conc80.json` reports `prefill_dp_attention` false for us vs true for llying, and pp=1 / tp=4 on both. That contradicts PP4 on both sides; it is metadata only, and the launch commands should be trusted.
- **Run-to-run noise:** about ±3.5% (llying mix README:123-125). E2's +8.6% and E1's effect need repeats, or at least comparable windows.
- **Small workload offset:** our mean input is 1.05% shorter. This follows from the speed difference rather than causing it. INFERRED.

Scratch files are in `/tmp/yihou/gap-wf/`. Nothing was modified.