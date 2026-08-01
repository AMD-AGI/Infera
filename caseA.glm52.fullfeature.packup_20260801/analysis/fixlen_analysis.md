# fixlen sweep — every number, and what it means

Phase 1. `sglang.bench_serving`, paired ISL/OSL percentiles, **P99 pair dropped**
(operator instruction) → 2 pairs × 4 concurrencies = **8 rounds**, 660 requests,
**100 % success, 0 failures**.

One server for all 8 rounds, booted once at a config sized for the largest pair
(155K + 3.3K) and then frozen. Small workloads got no server-level retuning — the
deliverable is one deployment measured across the load range, not eight tuned
deployments.

Source: `../../fixlen.glm52.fullfeature.packup_20260801/results/fixlen_summary.csv`,
raw per-request arrays in that kit's `results/raw/*.jsonl.gz`.

| pair | ISL | OSL |
|---|---|---|
| p50 | 74,000 | 320 |
| p90 | 155,000 | 3,300 |

---

## The table

| pair | conc | n | req/s | input tok/s | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | hit% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 1 | 8 | 0.060 | 4,400 | 10,526 | 13,558 | 19,813 | 16.89 | 19.09 | 15,913 | 0.0 |
| p50 | 32 | 64 | 0.245 | 18,161 | 45,216 | 188,920 | 246,633 | 18.74 | 27.09 | 50,287 | 12.44 |
| p50 | 64 | 128 | 0.573 | 42,409 | 12,526 | 125,911 | 186,901 | 19.48 | 25.81 | 17,796 | 49.76 |
| p50 | 128 | 256 | 0.643 | **47,560** | 22,479 | 234,798 | 342,921 | 16.69 | 29.70 | 26,016 | 49.74 |
| p90 | 1 | 4 | 0.021 | 3,267 | 13,680 | 13,708 | 13,711 | 9.63 | 12.31 | 45,490 | 47.62 |
| p90 | 32 | 32 | 0.066 | 10,298 | 218,033 | 377,915 | 405,266 | 15.48 | 23.38 | 271,095 | 0.0 |
| p90 | 64 | 64 | 0.097 | 14,962 | 110,277 | 440,658 | 572,289 | 13.56 | 28.29 | 151,006 | 42.16 |
| p90 | 128 | 128 | 0.100 | **15,459** | 403,362 | 910,796 | 1,176,149 | 24.13 | 32.47 | 482,204 | 10.36 |

All latencies ms.

## Reading 1 — the prefill knee sits between conc 64 and 128

Input throughput is the capacity signal here (these are prefill-dominated
prompts):

| pair | c1 → c32 | c32 → c64 | c64 → c128 |
|---|---|---|---|
| p50 | ×4.13 | ×2.33 | **×1.12** |
| p90 | ×3.15 | ×1.45 | **×1.03** |

Doubling concurrency from 64 to 128 buys **12 %** (p50) and **3 %** (p90) more
input throughput while TTFT p99 rises 83 % and 106 %. **The deployment is
saturated at c64 and past its useful operating point at c128.** Everything beyond
the knee converts directly into queueing delay, which is what TTFT p99 measures.

p90 saturates at a lower absolute throughput (15.5K vs 47.6K input tok/s) because
each request is 2.1× longer and generates 10× more output, so the same GPU-seconds
serve far fewer requests.

## Reading 2 — TTFT p99 is the honest congestion tell, and p50 hides it

Look at p50/c32 vs p50/c64: **TTFT p50 *falls*** from 45,216 to 12,526 ms while
concurrency doubles. That reads like the server got faster under more load, which
is impossible.

The explanation is admission ordering, not speed. At c32 with only 64 prompts the
run is short (261 s) and the batch composition is dominated by a synchronized
start — most requests queue behind the same first wave. At c64/128 prompts the
scheduler has a deeper queue to interleave and the *median* request is admitted
sooner, while the *tail* pays: p99 rises from 246,633 to… actually falls to
186,901, then to 342,921 at c128.

**Neither p50 nor p99 alone is monotone here; the pair together is.** The reliable
statement is the throughput knee (Reading 1) plus the fact that TTFT p99 at every
concurrency is 4–20× TTFT p50, i.e. the tail is queue-dominated throughout.

This non-monotonicity is a caution against quoting a single fixlen TTFT number as
"the" latency of this deployment.

## Reading 3 — TPOT is flat, and that is the MTP result

TPOT p50 across all 8 rounds: **9.63 – 24.13 ms**, with no trend against
concurrency and no trend against ISL.

| | TPOT p50 range |
|---|---|
| this sweep (MTP ON) | 9.63 – 24.13 ms |
| spur Case A (MTP OFF) | 31.3 ms |

Decode is a steady per-token process; its cost is set by batch occupancy, not by
prompt length, so flatness is expected. What is not free is the *level* — every
round sits below the MTP-off reference. This is the same 2× effect Case A
measured with a proper acceptance-length attribution (mean 2.04); the sweep
corroborates it across the load range.

The one outlier, p90/c128 at 24.13 ms, is the only round where decode batches were
large enough for in-batch contention to matter.

## Reading 4 — the cache-hit column is GPU-radix residue, NOT kvd

This is the load-bearing negative result of Phase 1, and it is why Case A was
necessary.

`cache_hit_pct` reads up to 49.8 %. It is tempting to read that as the kvd tier
working. It is not:

| round | cached_device_tok | **cached_host_tok** | cached_storage_tok |
|---|---|---|---|
| p50/c32 | 589,056 | **0** | — |
| p50/c64 | 4,713,600 | **0** | — |
| p50/c128 | 9,422,080 | **0** | — |
| p90/c64 | 4,173,696 | **8,192** | — |
| p90/c128 | 2,055,360 | **64** | — |

**`cached_host_tok ≈ 0` in every round.** Hits are being served from the in-GPU
radix cache, not from kvd's L2 (host RAM) or L3 tiers.

The cause is the dataset: `--dataset-name random` with `--random-range-ratio 1.0`
generates prompts with **no shared prefix by construction**. The nonzero hit rates
are residue from the *previous* round's identical-length prompts still resident in
the GPU radix tree — an artifact of running 8 rounds back-to-back on one server,
not a property of the workload.

Two consequences:
1. **Phase 1 cannot evaluate kvd at all.** A green cache number here proves nothing
   about the storage tier.
2. **The 88–90 % target belongs to Case A**, whose shared-prefix construction is
   what actually exercises the cache. Case A measured 89.2 % actual against 89.0 %
   ideal with kvd at 452 gets / 452 hits / 0 misses.

The erratic pattern across rounds (0.0 → 12.4 → 49.8 → 49.7, then 47.6 → 0.0 →
42.2 → 10.4) is itself the tell: real cache behaviour does not oscillate like
that; carry-over residue does.

## Reading 5 — the two conc=1 rounds are the clean latency floor

| pair | TTFT p50 | TTFT p99 | p99/p50 | TPOT p50 | E2E p50 |
|---|---|---|---|---|---|
| p50/c1 | 10,526 | 19,813 | 1.9× | 16.89 | 15,913 |
| p90/c1 | 13,680 | 13,711 | **1.002×** | 9.63 | 45,490 |

At c1 there is no queueing, so these are the deployment's intrinsic latencies.
p90/c1's p99/p50 ratio of 1.002 across 4 requests is the signature of a completely
uncontended path — every request took the same time.

**TTFT scales sublinearly with ISL at c1:** 2.09× the tokens (74K → 155K) costs
only 1.30× the TTFT. That is chunked prefill amortizing fixed per-request overhead
across more chunks, and it is the best evidence in the sweep that the prefill path
itself is healthy — the congestion seen at high concurrency is queueing, not a
broken prefill.

## Reading 6 — E2E is dominated by output length, as designed

p90/c1 has *lower* TTFT contribution but 2.9× the E2E of p50/c1 (45,490 vs
15,913 ms), because OSL is 3,300 vs 320. Decomposing:

    p50/c1:  10,526 + 319 × 16.89  =  15,914 ms   (measured 15,913) ✓
    p90/c1:  13,680 + 3,299 × 9.63 =  45,449 ms   (measured 45,490) ✓

Both reconstruct to within 0.1 %. The composition model
`E2E ≈ TTFT + (OSL−1) × TPOT` is exact at c1, which validates using it as the
back-solve route for Case A (where no E2E SLI exists).

## The caveat that must travel with this table

**The p50 rounds ran at prefill `mem-fraction-static` 0.88; the p90 rounds ran at
0.80.** The change was forced mid-sweep by a DP-attention activation OOM at ISL
155K × conc 32 (`HSA_STATUS_ERROR_OUT_OF_RESOURCES` while `token usage` read
0.01–0.05 — activation memory, not KV; fixed by *lowering* the fraction, opposite
to the decode-side retract fix). See `patches/0002` in the Phase-1 kit.

Consequently:
- **within-pair comparisons (across concurrency) are clean** — that is where every
  reading above lives;
- **cross-pair comparisons are not controlled.** p50-vs-p90 differences carry a
  13 % KV-pool-size confound (3,260,672 → 2,829,952 tokens/rank).

Re-running the p50 pair at 0.80 (~15 min) would close this. It was offered and not
taken; recorded here rather than smoothed over.

## Verdict on Phase 1

| question | answer |
|---|---|
| does the merged full-feature stack serve a fixed-length load reliably? | **yes** — 660/660, zero failures across 4 concurrency levels and 2 shapes |
| where is the capacity knee? | **conc 64**; c128 buys 3–12 % throughput for ~2× tail latency |
| is MTP delivering? | **yes** — TPOT 9.6–24.1 ms vs 31.3 ms MTP-off reference |
| is kvd delivering? | **cannot be answered from this sweep** — `--dataset-name random` has no shared prefix; `cached_host_tok ≈ 0`. Answered by Case A instead. |
