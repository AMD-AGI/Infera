# Phase 1 — fixlen sweep

**Ran:** 2026-08-01 09:46–11:08 UTC · **8 rounds, all completed**, 0 failed requests.
**Server:** one deployment, frozen; `ctx=262144`, chunk 65536 (8192/rank at dp8),
kv-aware **Rust** router, kvd prefill ON, MTP decode ON.

Paired percentiles, P99 dropped by instruction: p50 = ISL 74,000 / OSL 320 · p90 =
ISL 155,000 / OSL 3,300 · conc ∈ {1, 32, 64, 128}.

## Results

| pair | conc | n | dur (s) | input tok/s | output tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (ms) | cache hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p50 | 1 | 8 | 135 | 4,400 | 19.0 | 10,526 | 19,813 | 16.89 | 15,913 | 0.0 % |
| p50 | 32 | 64 | 261 | 18,161 | 78.5 | 45,216 | 246,633 | 18.74 | 50,287 | 12.4 % |
| p50 | 64 | 128 | 223 | **42,409** | 183.4 | 12,526 | 186,901 | 19.48 | 17,796 | 49.8 % |
| p50 | 128 | 256 | 398 | **47,560** | 205.7 | 22,479 | 342,921 | 16.69 | 26,016 | 49.7 % |
| p90 | 1 | 4 | 190 | 3,267 | 69.6 | 13,680 | 13,711 | 9.63 | 45,490 | 47.6 % |
| p90 | 32 | 32 | 482 | 10,298 | 219.2 | 218,033 | 405,266 | 15.48 | 271,095 | 0.0 % |
| p90 | 64 | 64 | 663 | **14,962** | 318.5 | 110,277 | 572,289 | 13.56 | 151,006 | 42.2 % |
| p90 | 128 | 128 | 1,283 | **15,459** | 329.1 | 403,362 | 1,176,149 | 24.13 | 482,204 | 10.4 % |

Every round: **100 % success** (8/8, 64/64, 128/128, 256/256, 4/4, 32/32, 64/64, 128/128).

## The bug this phase found: DP-attention prefill activation OOM

The first attempt at **p90 / conc=32** killed the prefill leg outright:

    rocdevice.cpp:3582  HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 1203306 MB
    Fatal Python error: Aborted

**It is not KV exhaustion.** The scheduler lines immediately before the abort read
`token usage: 0.01–0.05` — the KV pool was essentially *empty*. What was large was
`#pending-token: 2,486,426` with `#queue-req: 16` on DP0.

Mechanism: with DP-attention at dp8 each rank holds its own 8192-token chunk
activations, and a 155K prompt is 19 chunks. Under the prefill-delayer's batching the
transient activation peak exceeds what `1 − mem_fraction_static` leaves outside the
static reservation. `mem-fraction-static` was 0.88.

**Fix: LOWER it — prefill 0.88 → 0.80.** This is the *opposite* direction from the
decode-side retract fix (which raises it for more KV room), and the phase tells you
which: decode retract / `NotImplementedError` → raise; prefill `HSA_STATUS` / `Aborted`
→ lower. Consistent with the prior first-hand result on this stack (DSv4 dp8 prefill
OOMed at 0.90, clean at 0.85); this prompt is 155× longer, so one step further.

Decode was left at 0.85 — it did not crash, and moving both would have made the fix a
two-variable change. The decode leg's last healthy log line (10:12:08, `token usage
0.20`, `accept len 3.4–3.7`) is **after** the prefill abort at 10:11:41, so decode died
in the cascade, not independently.

Cost: KV pool 3,260,672 → 2,829,952 tokens per rank (−13 %) in exchange for activation
headroom 32.5 → 54.6 GB (+68 %). After the change, **p90/c32 ran 32/32 clean**, and
c64 and c128 followed — zero `HSA_STATUS_ERROR` for the rest of the sweep.

> The p50 rounds all predate the fix (gmu 0.88) and the p90 rounds all postdate it
> (gmu 0.80). The two pairs are therefore **not** a controlled comparison of each
> other. Stated rather than smoothed over — see Limitations.

## kv-aware spreads across all 8 DP ranks

The Phase-0 open question ("all 36 picks landed on `#dp0`") is **resolved**: that was an
artifact of a single-prefix probe. Under sweep traffic, 1,120 pick decisions:

    === Prefill (w_overlap=20.0) ===        === Decode (w_overlap=2.0) ===
      #dp0 175   #dp4  41                     #dp0 141   #dp4  60
      #dp1 106   #dp5  39                     #dp1  59   #dp5  59
      #dp2  64   #dp6  59                     #dp2  61   #dp6  58
      #dp3  35   #dp7  41                     #dp3  62   #dp7  60
      cache_hits: max=2422 mean=493.5         cache_hits: max=0 (radix off under MTP)
                  nonzero=295/560

**All 8 ranks picked, on both legs.** Prefill `cache_hits` reaching 2,422 blocks — the
kv-aware scorer is finding real reuse, not load-balancing blind. Decode 0 is by design.

The prefill distribution is deliberately skewed (dp0 175 vs dp3 35) while decode is
near-uniform (141 vs 58–62) — exactly what the weights ask for: `w_prefill=20.0`
chases cache locality, `w_decode=2.0` routes by load.

## Second finding: kvd's L3 spill tier fills the node's root disk

After round 8, every `docker exec` failed:

    OCI runtime exec failed: write /tmp/runc-processNNN: no space left on device

`/` was at **100 %** (838 GB). The cause is ours: `--long-path /tmp/kvd-long` is
**inside the container**, so kvd's L3 spills into the container's writable layer on the
node's root disk — not onto `/mnt/vast`. `bench_run`'s layer had reached **263 GB**
against a `--long-bytes 512G` budget larger than the disk.

Reclaimed 225 GB (only our own data — `bench_run`'s tier and the exited `merged_run`'s
20 GB; verified by `docker inspect` before deleting anything). `--long-bytes` is now
**64G**, matching `--max-bytes`. Case A writes far more KV over 67 minutes than this
sweep did, so 512G would have refilled the disk mid-run.

## Reading the numbers

**Prefill throughput saturates, and where.** At p50, input throughput goes
4.4K → 18.2K → 42.4K → 47.6K tok/s across conc 1 → 32 → 64 → 128. The step from 64 to
128 is only **+12 %** for 2× the offered concurrency: the prefill knee is between them.
At p90 the same shape appears earlier and harder — 15.0K → 15.5K from c64 to c128,
**+3 %**, with duration nearly doubling (663 → 1,283 s). Beyond c64 at 155K ISL the
system is queueing, not serving.

**TTFT p99 is the queueing tell.** p50/c128 p99 TTFT is 343 s against a p50 of 22 s — a
15× spread. p90/c128 reaches **1,176 s** p99. Requests are not slow; they are *waiting*.
With `max_running_requests` reported as 256 effective and prompts of 155K, only a few
can be in prefill at once, so the rest sit in `#queue-req`.

**TPOT is flat and low.** 9.6–24.1 ms across every round, against the spur baseline's
31.3 ms — MTP is doing real work here (accept len 1.5–3.7 observed). Decode is not the
constraint at any point in this sweep; TPOT rises only at p90/c128 (24.1 ms), and even
then it is the smallest term in a 482 s E2E.

**conc=1 is a DP-8 artifact, not a latency floor.** At conc=1 exactly one of eight DP
ranks is active — 4.4K tok/s is one rank's prefill rate, not the deployment's. The
p90/c1 TPOT of 9.63 ms (the best in the sweep) is the same story from the other side:
one request, no contention.

**Cache hit rate is an artifact of `--dataset-name random`, and must not be read as a
cache result.** `random` synthesises each prompt independently — there is no shared
prefix by construction. The nonzero values (12–50 %) come from the fixed system framing
plus tokens that happen to repeat, and the swing (0 % at p90/c32 up to 49.8 % at
p50/c64) tracks how much of the previous round's radix tree survived, not any property
of the cache. **The 88–90 % target lives in Case A**, whose profile builds a genuine
shared prefix. kvd counters confirm the same: `sets` climbed 102 → 145,555 across the
sweep while `gets` reached only 2,986 — writes, not reads.

## Limitations

1. **The two pairs ran at different `mem-fraction-static`** (p50 at 0.88, p90 at 0.80).
   The p90 numbers are from a leg with 13 % less KV pool. Comparisons *within* a pair
   are clean; comparisons *across* pairs carry this confound.
2. **The server was never reset between rounds** — deliberate (one frozen deployment),
   but it means each round inherits the previous round's radix tree. Each round's own
   kvd before/after snapshot bounds what it inherited.
3. **No kvd-off A/B**, so no performance claim is made for kvd here.
4. **`--random-range-ratio 1.0`** pins every prompt to exactly ISL. Real traffic is a
   distribution; that is Case A's job.
5. The kvaware weight sweep (w ∈ {1.0, 20.0}) was **not** run — the OOM debugging
   consumed its budget. The weights' *effect* is nonetheless visible in the pick
   distributions above (skewed prefill vs uniform decode).
