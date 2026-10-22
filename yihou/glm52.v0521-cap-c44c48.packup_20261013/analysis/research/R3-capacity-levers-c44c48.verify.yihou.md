# R3 verification (adversarial) — R3-capacity-levers-c44c48.yihou.md

Date 2026-10-12. Read-only. No GPU, container or host action. Helper scripts: `/tmp/x1chk*.yihou.py`,
`/tmp/ax*.yihou.py` (they parse the TSV and the AgentX JSONs). src = `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt`,
RUNS = `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs`, H = `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`.

## Adversarial verification

Verdict: the code and log claims on HiCache sizing, registration and decode memory hold. Two claims that feed
decisions are refuted:
- **e2f has the best C40 TTFT.** This is the evidence for lever #1, and it is false.
- **The running cap of 16 per DP is hit in under 0.1 % of samples at C44.** At C44 it did bind on DP3, and requests queued behind it.

The answer to the user's question ("v0.5.21 ≈ 0") holds only with caveats: two confounds are unstated.

### Confirmed [M]

1. **`--hicache-size` is GB (1e9) per rank and covers only the MLA KV host pool.**
   - The error text is at `src/mem_cache/host_memory.py:116-118`.
   - The unit is set by the conversion `int(host_size * 1e9 // self.size_per_token)` at `src/mem_cache/pool_host/base.py:212-213`.
   - e2g `prefill-0.log`, all 4 DPs: `Allocating kv hierarchical KV host pool: 10016064 tokens, 450.00 GB`.
2. **The DSA indexer host pool is separate and comes on top.**
   - `src/mem_cache/pool_host/dsa.py:197-219` runs its own `host_memory_budget_bytes` check and logs `Allocating %.2f GB host memory for DSA indexer`.
   - e2g log at 05:43:55, all DPs: `Allocating 103.13 GB host memory for DSA indexer (layout=page_first)`.
   - The ratio 103.13 / 450 = 0.2292 matches the claimed factor of 1.2292.
3. **Registration chunking and the 549 GB limit.**
   - `src/mem_cache/pool_host/common.py:140-185`: chunk = floor(CHUNK_GB·2^30 / granularity) × granularity. The MLA and indexer buffers pass a page granularity (`mla.py:398-402`, `dsa.py:292`).
   - On HIP, `ALLOC_MEMORY_FUNCS` defaults to `alloc_with_host_register` (`common.py:325-332`).
   - The default is 256 (`src/environ.py:770`). The harness sets 512 (`H/config.yihou.E2-target.sh:12`).
   - 549e9 B is 511.3 GiB, which fits in one chunk; 550e9 B does not. The "S ≤ 549" arithmetic is right.
   - Support for the H1 mechanism: `common.py:283-288` notes that on HIP the device address differs from the host address, and the kernel pointer table holds one pointer per tensor.
4. **Per-rank budget = (available − 10 GiB) / ranks_per_host** [M]: `pool_host/base.py:29`, `:89-110`.
   - Caveat [I]: each rank snapshots `psutil.available` on its own, so the 4 concurrent allocations are not checked against each other. At S = 540 every rank still passes (≈ 663 GB needed vs ≈ 720 GB budget). The code guard therefore does **not** protect the 4-rank total; the RAM table is the only guard.
5. **Arithmetic of the RAM table.**
   - 2,885,471,972 kB = 2,751.8 GiB. This matches `allocated_cpu_dram_gb: 2752` in the AgentX JSON.
   - Left over: 691 GiB at S = 450 and 462 GiB at S = 500. Both are correct.
   - No `--memory` flag appears in `H/dryrun.e2g-102.yihou.txt`. `--shm-size` is set at `H/bench-harness/engine.sh:115`.
6. **Decode memory waterfall** (e2g `decode-0.a1.log`, DP0):
   - Graph bs list = `[1..8,10,12,14,16]`.
   - Target-verify capture: `mem usage=4.10 GB, avail mem=11.77 GB`. After draft-extend: `avail mem=10.89 GB`.
   - `max_running_requests=16`, `available_gpu_mem=10.89 GB`. Server args show `'max_running_requests': 64` and `extra_slots': 32`.
   - Small error in the R3 table: target-verify capture began at `avail mem=15.88 GB`, not 13.99, so 15.88 − 4.10 = 11.77. The end value 10.89 is right.
7. **Extra transfer slots and graph buckets at max_running 96.**
   - `src/arg_groups/pd_disaggregation_hook.py:135-149`: extra = 2 × per_worker if per_worker ≤ 32. So max_running 96 → 24 per DP → 48 extra slots.
   - Note: from per_worker > 32 (max_running > 128) extra_slots drops to **0**.
   - The spec-decode bs list (`src/arg_groups/cuda_graph_hook.py:641-647`) adds 18, 20, 22, 24, i.e. +4 buckets.
8. **DP attention adjustments** (`src/arg_groups/parallel_hook.py:299-314`): chunked_prefill is divided by attn_dp (32768 → 8192), and schedule_conservativeness is multiplied by 0.3.
9. **Scores and TTFT** (`request_metrics.throughput.per_gpu.total_tput_tps` and `latency.ttft.mean`), matching the R3 lineage:

   | run | tok/s/chip | TTFT mean (s) |
   |---|---|---|
   | e1c | 21,049 | 5.14 |
   | e2b4 | 20,347 | 9.01 |
   | e2c C44 | 20,524 | 8.03 |
   | e2c C40 (flushed) | 21,696 | 5.61 |
   | e2d | 21,383 | 6.04 |
   | e2e | 19,565 | 6.94 |
   | e2f | 19,864 | 5.57 |
   | e2g A1 | 19,640 | 7.61 |
   | e2g A2 | 22,114 | 5.30 |
   | e2i R1 | 20,371 | 6.37 |
   | e2i R2 | 21,921 | 5.83 |
   | e2j R1 | 19,335 | 7.20 |
   | e2j R2 | 21,681 | 5.21 |
   | X1 | 21,773 | 9.02 |

   - Fresh range 19.3-20.4k and warm range 21.7-22.1k: confirmed. X1 vs e2j R2 is +0.4 %: confirmed.
10. **Per-round server args** (prefill/decode log strings):
    - e1c: `hicache_size 0`, index_share false, fused indexer False, decode mem 0.85.
    - e2d: index_share **true**.
    - e2b4: fused False (no "fused DSA indexer enabled" line).
    - e2c and e2g: fused `None`, which resolves to enabled. Both logs show `gfx950 fused DSA indexer enabled` ×4. Resolution code: `src/arg_groups/fields/exec_.py:340-352`, `src/layers/attention/dsa/utils.py:68-100`.
11. **X1 host pool full and decode admission idle** (TSV, all samples):
    - `hicache_host_used_tokens` reached 10,016,064 on DP1-3 and 10,016,000 on DP0 (64 short; full at page granularity).
    - Decode `num_decode_prealloc_queue_reqs` max 0, `num_retracted_reqs` max 0, `pending_prealloc_token_usage` max 0 on all DPs.
    - Per-DP `token_usage` max 0.71 / 0.82 / 0.87 / 0.72. The "0.87" claim is right.
12. **Throughput model.** I re-solved the R3 §3 fixed point and reproduced the table exactly: C44 22.4k at 9 s; C48 24.1k at 9 s.
    - TTFT 9 → 8 s at C44 gives +2.4 %. The "≈ 2.5 %/s" claim is consistent.
    - g = C/λ − e2e is 16.3 s (A2) and 15.8 s (X1). Consistent.
13. **xferfix and harness facts.**
    - `conn.yihou-xferfix.py` has 7 regbase hits and 6 XFER_SHARD hits.
    - `env.e2f-102.yihou.sh` used launch/room on the E2-target base.
    - `config.yihou.mc4340.sh:23-28` rejects nodes other than 102/136, and `:38` asserts the image tag.
    - `run.v2.yihou.sh:19-26` parses YIHOU_OVERRIDES.
    - `config.sh:66` sets DECODE_MAX_RUNNING; `engine.sh:264` passes `--max-running-requests`.

### Refuted / needs correction

- **R-1: "e2f TTFT 5.57 s, best of all C40, fresh or warm" (lever #1 evidence).** False [M, JSONs above].
  - e1c is 5.14 (fresh, no xferfix), e2j R2 is 5.21 and e2g A2 is 5.30.
  - e2f is best only among fresh *target-config* runs, and even there e1c (fresh, relaxed, no xferfix) beats it.
  - The fresh-run TTFT range is 5.1-7.6 s without xferfix. So e2f's 5.57 is not attributable to xferfix from scores alone.
  - Weak partial support [M, biased sampler]: the mean `#inflight-req` per prefill batch line is 1.01-1.29 in e2f versus 1.60-1.97 in e2e.
  - Lever #1's expected "+4…+6 %" rests on an untested TTFT delta.
- **R-2: "per-DP running hit its cap of 16 in < 0.1 % of samples" / "the cap is reached rarely" (lever #4 framed as pure C48 insurance).** Refuted for DP3 at C44.
  - TSV in the 07:05:40-08:05:40 window: DP3 running ≥ 16 in 5/719 samples = **0.70 %**, ≥ 12 in 0.83 %.
  - X1 decode log, 07-08 UTC: DP3 `#running-req: 16` in 8/2,282 batch lines = **0.35 %**. Other DPs: 0.
  - In those lines the decode **`#queue-req` was 1-5** (07:59:26-07:59:46). Transferred requests did wait ~20 s for a running slot, so the cap bound briefly at C44 already.
  - Still small, but at C48 with a 1.5× DP imbalance it is likelier. Lever #4 is a C48 enabler more than insurance [I].
- **R-3: "host pool 100 % full by about minute 45".** Only partly right.
  - Relative to profiling start (t0 + 1,073 s), DP3/2/0 reached ≥ 99.9 % at 2,661 / 2,847 / 3,082 s, i.e. minutes 44-51.
  - DP1 was already ≥ 95 % at sampler start.
  - Relative to bench start, the times are 61-69 min. The two time bases are easy to mix up.
- **R-4: "C44 TTFT 8.0-9.0 vs 7.2-7.6 s at C40 fresh" (§2).** The C40 fresh range is really 5.6-7.6 s (e2f 5.57, e2i R1 6.37, e2e 6.94).
- **R-5: the citation for the graph-bs cap.** The cited `memory_hook.py:193-198` [S, R042] only materializes the bs list; it does not cap it by max_running. The cap is real (the log shows bs ending at 16), but the code site is not where cited. `pool_configurator.py:1092-1094` derives per-worker max_running. I did not trace the exact filter.

### Unverifiable here (no first-hand check possible or attempted)

- 102 `/proc/meminfo` at 08:19 (no host action allowed). The MemTotal arithmetic is consistent.
- The `mc4340prio` image (sha256:aacc9aeb) exists only on 102, and the R043 microbench numbers (200.6 → 11.0 ms).
- The v0.5.19 lineage (+12.5 % from triton + index_share; 19,799; ~18,700). The packup server-info JSONs were not opened. The ledger (`working_process.md:601`) confirms 21,032 was C40, index_share TRUE, on **node 137**.
- Post-capture free memory at 0.96/0.97 (7.9 / 5.2 GB) is an extrapolation. It is plausible: 0.01 × ~288 GiB ≈ 2.9 GiB.
- The prefill mem 0.88 gain (+168k tok/DP, +7.4 %) is plausible: cell 55,227 B/token (116.79 GiB / 2,270,656), and 41.49 − 0.03 × 288 ≈ 32.9 GB free. The runtime peak is unmeasured.

### Caveats on the answer to the user's question ("what got us to 22k / what did v0.5.21 bring")

- **"v0.5.21 ≈ 0" is confounded twice.**
  1. e1c (v0.5.21, 21,049) ran **without** index_share, while v0519's 21,032 had it. If index_share was worth several % in v0519, v0.5.21 at parity *without* it implies v0.5.21 is slightly **positive**, not 0.
  2. v0519 ran on node 137 and e1c on 102: an unstated node confound. Both are single fresh runs, with ISL noise of ±3 %.
  - Say "within noise, no evidence of a gain", not "≈ 0".
- **index_share is described inconsistently.** e2d (index_share TRUE, fresh) = 21,383 is listed as "the only fast fresh run", yet "index_share TRUE at C40" is listed under "no measurable gain". The ledger (R049) attributes e2d's speed to its fresh-but-different decode (debug agent), so the index_share effect on v0.5.21 is **unresolved**, not zero.
- **The fused indexer** is on by default in e2c/e2g. Its +0.9 % (e2b4 → e2c) is one sample each, at C44 fresh. Agreed: not significant.

### Questions R3 did not answer

1. **Is `inflight` residence wire time or waiting for the decode-side ack/poll?** R2 left this open (R2 §6 "Open"). R3 still ranks the KV-send path #1. If the wait is decode-side (decode transfer-queue residence 8.19 s > prefill inflight 5.25 s), xferfix and stream priority will not help. Q1 should log per-transfer wire time (`YIHOU_XFER_LOG`) next to decode poll latency.
2. **Why the e1c fresh run (HiCache 154 GB, decode 0.85, no xferfix) had the lowest TTFT of any run (5.14 s).** It is a direct counterexample to "transfer path is the C40/C44 TTFT limiter", and an untested hint that the big host pool costs something on the prefill path, such as host load-back or registration.
3. **Track B (keep ≥ 50 % of the decode KV pool for running requests) is not addressed.** X1's per-DP non-running share exceeded 0.5 of the pool 1.4-8.3 % of the time (R2 §3). R3 does not say how a cap on prealloc/transfer would interact with lever #1 (it would lengthen exactly the transfer residence R3 wants to cut).
4. **How the second node's host RAM and image are handled.** If the pending job 208970 lands on 136, the mc4340 image already exists there (`config.yihou.mc4340.sh:25` has its ID), but R041-era RCCL init failures on 136 (ledger R011) apply.
5. **Q1 bundles three changes** (xferfix + prefill 0.88 + max_running 96) at one sample each, with ±3 % ISL noise. The score cannot attribute them. R3 admits this but gives no minimum detectable effect and no replicate plan.
6. **DP0's host pool never reached 100 % while DP1 was full at sampler start**, and the host pool draining during warmup is unexplained (R2 open). Is the "full pool" partly an artifact of the 3rd-run stack (X1) rather than load?
7. **extra_slots drops to 0 above max_running 128** (`pd_disaggregation_hook.py:142`). Any future max_running push past 128 silently removes the transfer headroom. R3 does not flag it.
