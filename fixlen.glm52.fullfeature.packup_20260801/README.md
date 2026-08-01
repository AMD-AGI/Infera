# Fixed-length sweep on the full-feature GLM-5.2 deployment

**Ran:** 2026-08-01 09:10 – 11:20 UTC
**Author:** yihou
**Nodes:** `chi2879` (prefill) + `chi2867` (decode), 8 × MI355X gfx950 each, ionic RoCE
**Image:** `infera/engine-sglang:merged-e` (branch `yihou.dev.glm52.merged.experiment` @ `b92a1e8`)
**Status:** **PASS** — 8/8 rounds completed, 100 % success in every round, after two real
bugs were found and fixed.

## What this is

Phase 1 of the mission in `spec/mission.kv.liying.mtp.bench.md`: benchmark the **most
complete** infera GLM-5.2 deployment — `kvaware + kvd + mtp + pd + dpa`, **all on at
once** — with sglang's own `bench_serving`, before the agentic Case A run (Phase 2).

**This is the first time this configuration has ever run.** Prior kits validated the
merged branch at `ctx=32768` (correctness only, no perf), and validated Case A with
**MTP off** on a different cluster. `ctx=262144` **and** MTP together, behind the **Rust**
router, is new — so this was bring-up, not replay, and it behaved accordingly.

## Sweep design

Paired ISL/OSL percentiles from the Case A profile, **P99 dropped** by instruction, ×
four concurrencies = **8 rounds**, all against **one server** sized for the largest pair
and then frozen. Small workloads got **no** server-level retuning: the deliverable is one
deployment measured across a load range, not eight tuned deployments.

| pair | ISL | OSL | conc |
|---|---:|---:|---|
| p50 | 74,000 | 320 | 1 / 32 / 64 / 128 |
| p90 | 155,000 | 3,300 | 1 / 32 / 64 / 128 |

## Result

Full table: `results/fixlen_summary.csv`. Raw per-request samples (`ttfts`, `itls`) in
`results/raw/*.jsonl.gz`, gzipped, sufficient to recompute any percentile.

| pair | conc | n | input tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| p50 | 1 | 8 | 4,400 | 10,526 | 19,813 | 16.89 | 15,913 |
| p50 | 32 | 64 | 18,161 | 45,216 | 246,633 | 18.74 | 50,287 |
| p50 | 64 | 128 | 42,409 | 12,526 | 186,901 | 19.48 | 17,796 |
| p50 | 128 | 256 | **47,560** | 22,479 | 342,921 | 16.69 | 26,016 |
| p90 | 1 | 4 | 3,267 | 13,680 | 13,711 | 9.63 | 45,490 |
| p90 | 32 | 32 | 10,298 | 218,033 | 405,266 | 15.48 | 271,095 |
| p90 | 64 | 64 | 14,962 | 110,277 | 572,289 | 13.56 | 151,006 |
| p90 | 128 | 128 | **15,459** | 403,362 | 1,176,149 | 24.13 | 482,204 |

**Every round: 100 % success** (8/8, 64/64, 128/128, 256/256, 4/4, 32/32, 64/64, 128/128).
Zero failed requests across all 660.

### Three things the numbers say

**Prefill saturates, and you can see where.** At p50, input throughput runs
4.4K → 18.2K → 42.4K → 47.6K tok/s across conc 1 → 32 → 64 → 128; the last step is **+12 %
for 2× the concurrency**. At p90 it is starker: 15.0K → 15.5K from c64 to c128, **+3 %**,
with duration nearly doubling. Beyond c64 at 155K ISL the system is queueing, not serving.

**TTFT p99 is the queueing tell, not a latency defect.** p90/c128 reaches 1,176 s p99
against a 403 s p50. Requests are not slow — they are *waiting*. TPOT stays flat at
9.6–24.1 ms everywhere.

**MTP is doing real work.** TPOT 9.6–24.1 ms against the MTP-off spur baseline's 31.3 ms,
with `accept len` observed at 1.52–3.70 (healthy; **4.00 would be bad news** — see
`notes.md` §1). Decode is not the constraint at any point in this sweep.

## The two bugs this phase found

Both are in `patches/`, each with what / why / how / context.

**1. `MC_GID_INDEX` was hardcoded and is node-dependent** (`patches/0001`). The decode leg
died at init on all 8 DP ranks with `Mooncake Transfer Engine initialization failed`,
while prefill on the other node had zero errors — and that asymmetry was the diagnosis.
Both nodes expose a link-local and a routable RoCE v2 GID per port, but at **different
indices**: chi2879 → 1, chi2867 → **2**. Now discovered, not assumed.

**2. Prefill HSA OOM at long ISL is an *activation* problem, and the fix goes the
"wrong" way** (`patches/0002`). At ISL 155K × conc 32 prefill aborted with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` while `token usage` read **0.01–0.05** — the KV pool
was nearly empty, so it was never KV exhaustion. DP-attention at dp8 holds per-rank chunk
activations, and a 155K prompt is 19 chunks. Fix: **lower** `mem-fraction-static`
0.88 → 0.80 — the opposite of the decode-side retract fix. Afterwards: zero
`HSA_STATUS_ERROR` for the rest of the sweep.

A third, operational: **kvd's L3 tier fills the node's root disk** (`patches/0003`).
`--long-path` is container-local, so a 512 GB budget on an 838 GB disk took `/` to 100 %
and made every `docker exec` fail. Now 64 GB.

## The numbers that discriminate

A green run proves little. These would have gone red if a fix were absent:

**kvd is serving, not merely wired.** A latency win proves nothing — sglang's in-GPU radix
cache serves a repeated prefix without touching L3. Restarting the prefill engine empties
that cache while the kvd daemon and its L3 keep running:

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| after first reuse run | 0 | 0 | 102 | 0 |
| **after restart + replay** | **102** | **102** | **102 (unchanged)** | 0 |

`sets` staying put is the load-bearing part: reads, not re-writes.

**kv-aware routes per DP rank, and the weights visibly work.** Over 1,120 pick decisions
all 8 ranks were used on both legs, with prefill `cache_hits` reaching 2,422 blocks:

    prefill (w=20.0):  dp0 175 · dp1 106 · dp2 64 · dp3 35 · dp4 41 · dp5 39 · dp6 59 · dp7 41
    decode  (w=2.0):   dp0 141 · dp1  59 · dp2 61 · dp3 62 · dp4 60 · dp5 59 · dp6 58 · dp7 60

Prefill **skewed** (chasing locality at w=20.0), decode **near-uniform** (routing by load
at w=2.0). That contrast is the configuration doing its job, on a live path, under the
Rust router — where group E's bigram kv-event fix had never previously carried traffic.

## Read this before trusting any number

* **The cache-hit column is not a cache result.** `--dataset-name random` builds every
  prompt independently, so there is no shared prefix by construction. The 0–50 % values
  are the previous round's radix residue — confirmed by `host_cached_tokens ≈ 0` in every
  round, i.e. GPU-radix hits, not kvd. **The 88–90 % target belongs to Case A.**
* **p50 and p90 are not comparable to each other.** p50 ran at prefill memfrac 0.88, p90
  at 0.80 after the fix — 13 % less KV pool. Within-pair comparisons are clean.
* **The server was never reset between rounds.** Deliberate (one frozen deployment), but
  each round inherits the previous round's radix tree; per-round kvd snapshots bound it.
* **No kvd-off A/B**, so no performance claim is made for kvd.
* **The kvaware weight sweep was not run** — OOM debugging consumed its budget.

Full list: `notes.md` §11.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable: clean nodes → legs → gate → 8 rounds |
| `environment.md` | hardware, fabric, image digests, SHAs, external paths, secrets needed |
| `notes.md` | the traps in the order they bite — the most re-read file |
| `patches/` | the three fixes, each with what / why / how / context |
| `scripts/` | every script that ran, verbatim, plus the result extractor |
| `results/` | `fixlen_summary.csv` + `raw/` (gzipped per-request samples, kvd snapshots) |
| `logs/` | engine + sweep logs, gzipped — includes both crash scenes |
| `env/` | per-node `collect_env.sh` snapshots |
| `spec/` | the originating mission file, verbatim |
| `notes/plan_as_written.md` | the plan before execution, kept so the deltas are visible |

## Related

- `glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/` — the merged branch + image
  this runs on (correctness at ctx=32768; no perf numbers)
- `liying_rest_pr56.packup_20260801/` — group E, incl. the Rust-router bigram fix that
  this run is the first to exercise under live traffic
- `../infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731/` — Case A with
  **MTP off** on spur; the baseline this phase's TPOT is compared against
- **Phase 2 (Case A on this deployment) is the sequel** and is not in this kit.
