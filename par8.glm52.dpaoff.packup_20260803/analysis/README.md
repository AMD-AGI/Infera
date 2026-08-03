# Analysis — par8: Case A shape at reduced load, prefill DP-attention OFF

Format follows `../../caseA.glm52.fullfeature.packup_20260801/analysis/`.

| file | what |
|---|---|
| [`sli_percentiles.md`](sli_percentiles.md) | full percentile ladders, recomputed from raw samples; TTFT-by-input-size; the error class |
| [`yaml_vs_measured.md`](yaml_vs_measured.md) | every YAML knob vs. what was measured, with the conversion rule |
| [`routing_and_kvaware.md`](routing_and_kvaware.md) | per-rank pick distributions and why kv-aware steered nothing |
| this file | headline, the confound, the verdict |

---

## Headline

**The run is clean and the deployment is fast — but its single most quotable
number is confounded, and the run cannot answer the question it was partly
meant to answer.**

| | par8 (this run) | Case A full |
|---|---|---|
| deployment | prefill **TP8 / DPA off**, chunk 16384 | prefill **DPA 8/8**, chunk 65536 |
| load | 8 / 32 / 24 | 32 / 128 / 48 |
| requests | 2,884 sent / **2,850 done** | 2,988 / 2,952 |
| success | **98.8 %** | 98.8 % |
| duration | 4,005 s | 4,006 s |
| **TTFT p50** | **1,365 ms** | 4,378 ms |
| **TTFT p90** | **4,903 ms** | 8,940 ms |
| TTFT p99 | 9,066 ms | 16,492 ms |
| TPOT p50 / p90 | **14.8 / 17.7 ms** | 14.8 ms *(summary only)* |
| E2E p50 / p90 | 7.4 s / 36.9 s | *not recorded* |
| cache hit | 88.9 % (eff. 100.0 %) | 89.2 % |
| accept len | 2.02 | 2.02 |
| in-flight max / mean | 22 / **12.1** | 30 / 15.4 |

*(par8 figures are sustain-phase; Case A's are whole-run as published.)*

## ⚠ The headline TTFT number is confounded — read this before quoting it

**Two variables moved together**: offered load (32/128/48 → 8/32/24) *and*
prefill DP-attention (on → off, with the chunk change that follows from it).
The −69 % TTFT p50 is their **combined** effect.

**This data cannot apportion it.** Mean in-flight fell 15.4 → 12.1 (−21 %), so a
meaningful share of the TTFT improvement is simply less queueing.

Where the variable *is* isolated — the solo pair at concurrency 1
(`../../solo.glm52.latencyfloor.packup_20260802` vs
`../../solo.glm52.dpaoff.packup_20260802`):

| | DPA on | DPA off | delta |
|---|---|---|---|
| TTFT p50 | 1,165 ms | **805 ms** | **−31 %** |
| TTFT p90 | 2,657 ms | 2,086 ms | −21 % |

A controlled −31 % at concurrency 1. **Whether DPA-off retains, exceeds, or
loses that advantage at this run's concurrency is unknown** — the aggregate-KV
cost of DPA-off (−85.6 % in the solo kit's accounting) is a *capacity* cost,
and capacity costs only bite under load. **The missing run is par8.yaml with
prefill DPA on.** Until it exists, quote the solo pair for the DPA effect and
quote this run only as "this deployment at this load."

## What this run does establish, cleanly

**1. Prefill scales with input size, with no pathology.**

| input | 0–50K | 50–100K | 100–160K | 160–220K | 220–300K |
|---|---|---|---|---|---|
| TTFT p50 | 623 ms | 1,036 ms | 1,815 ms | 3,411 ms | 5,863 ms |

Monotone, 9.4× over a 4.5× size span, **no stall bucket**. The DPA-off *solo*
run had two cache-stall outliers (11.4 s at 77K input while 215K served in
6.1 s); that signature is absent here.

**2. Decode is not the bottleneck.** TPOT p99/p50 = **1.45** — an exceptionally
tight ladder. A contended decode leg shows a fat upper tail. Prefill is the
binding resource at this load.

**3. The workload reproduced its spec.** Input p50/p90/p99 within ~1 % of the
sampler; cache hit 88.94 % against an ideal of 88.98 % (**100.0 % efficiency**,
0.2 % eviction).

**4. The load cap never bound.** In-flight peaked at **22 of 24**, so the
workload set the load, not backpressure. The window is valid.

**5. MTP is healthy.** Per-request accept len mean 2.02 (max 3.0); engine-side
mean 2.763 over 1,594 batches. Not the degenerate 4.00.

## The structural finding

**kv-aware routing performed no cache steering in this run — on either leg —
and one half of that is permanent under the mission's own constraints.**

| leg | targets | cache_hits | what routing actually did |
|---|---|---|---|
| prefill | **1** (`dp_size=1`) | mean 1,201, 99.7 % non-zero | nothing to choose |
| decode | 8 | **0 / 2,884** | pure least-loaded (10.6–14.3 %) |

- **Prefill**: DPA-off collapses 8 routing targets to 1. Inherent to the config.
- **Decode**: MTP forces `ChunkCache` (upstream raises on
  `--disaggregation-decode-enable-radix-cache` + `--speculative-algorithm`), so
  the router's KV view for that worker is permanently empty. **Verified on the
  wire**: decode's kv-event socket emitted 0 messages in 15 s while prefill's
  emitted 30.

Consequence: `decode_prefix_len` is always 0, so **every turn re-transfers the
entire prompt KV** — and a prefill-side cache hit does not reduce it (it only
reorders the send). Full trace in [`routing_and_kvaware.md`](routing_and_kvaware.md).

**Upstream gives no reason for the MTP × decode-radix exclusion.** See
`../notes.md`.

## Verdict against the stated bars

| bar | target | measured | verdict |
|---|---|---|---|
| success rate | ≥ 0.97 | **0.988** | **PASS** |
| TTFT p90 | < 30,000 ms | **4,903 ms** | **PASS**, 6.1× margin |
| E2E p50 | < 4,500 ms | **7,400 ms** | **FAIL**, 1.64× |
| full cycle | ramp+sustain | 4,005 s | **PASS** |
| load not capped | in-flight < 24 | max 22 | **PASS** |

The E2E miss is **not a regression**: `e2e_p50_ms: 4500` is a latency-floor spec
that the solo kits showed is met only at concurrency 1. At 12 mean in-flight it
is the wrong bar, and it fails identically in every loaded run on this stack.

## What to run next

| question | run |
|---|---|
| **How much of the TTFT win is DPA-off?** | par8.yaml, prefill **DPA on**, chunk 65536 — the missing control |
| Does kv-aware cache steering help under load? | same run; gives 8 prefill targets back |
| Does full-prompt KV re-transfer cost us? | instrument `transfer_total_bytes` vs TTFT; no config change |
| Are the 61 `accept len: 4.00` batches real loops? | dump output token IDs for those batches |
| Does `temperature: 0.0` reach the engine? | two probes, `temperature` 0.0 vs 1.0, compare outputs (see `../notes.md`) |
