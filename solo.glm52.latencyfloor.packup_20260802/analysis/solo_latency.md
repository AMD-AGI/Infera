# Solo latency floor — Case A request shape at concurrency exactly 1

Run `solo_full/2026-08-01-16-05-43`, 2026-08-01 16:05–16:42 UTC. Ramp 400 s +
sustain 1800 s. **145 requests total, 102 in the measured window, 0 errors.**

Same deployment, same image, same hour as Case A — nothing was restarted between
the two runs. `infera/engine-sglang:merged-e` **+ `GLM52_P1V3`** (decode leg;
verified `P1V3: 3` in the *loaded* module immediately before the window opened).

## Why this run exists

Case A answers *"what does the service do under a realistic agentic load?"* It
cannot answer *"how fast can this service possibly be?"*, because every number in
it contains an unknown amount of queueing. This run removes the queue: identical
request shape, one request in flight, never two.

Any gap between the two runs is therefore **attributable to load alone** — the
workload, the seed, the cache construction, the warm-up and the deployment are
byte-identical.

## What was changed, and what was not

| | Case A | solo | |
|---|---|---|---|
| `input_tokens` p50/p90/p99 | 74K/155K/235K | **same** | request shape |
| `output_tokens` p50/p90/p99 | 320/3,300/17,000 | **same** | request shape |
| `cache_hit_rate` | 0.89 | **same** | request shape |
| `max_input_tokens` | 260,000 | **same** | request shape |
| `random_seed` | 1337 | **same** | request shape |
| `ramp_duration` | 400 | **same** | same warm cache state |
| `turns_per_session` | 3/20/103 | **1/1/1** | ← load |
| `initial_sessions` | 32 | **1** | ← load |
| `max_sessions` | 128 | **1** | ← load |
| `new_session_rate` | 0.10 | **1.0** | ← load |
| `max_inflight` | 48 | **1** | ← load |
| `sustain_duration` | 3600 | **1800** | 30 min, as specified |

### How concurrency is actually pinned

The gate is **`max_sessions=1`**, not `max_inflight`. The driver's control loop
spawns only while `active_sessions < max_sessions`, and `count_active_sessions()`
counts a session that is `in_flight` — so while a request is on the wire no second
session can be born. `max_inflight=1` is a belt-and-braces valve.

**It never fired: `grep -c 'Hit max_inflight'` = 0.** Measured `in_flight` across
all 1,795 sustain rows takes exactly two values, `{0, 1}` — never 2. The 0s are
the ≤1 s respawn tick between requests.

At `turns_per_session={1,1,1}` the sampler collapses (`vmin == vmax == 1`,
verified offline), the session retires after one turn, and it does so on the
`break` path **before** the inter-turn sleep — so `inter_turn_delay_s` never
executes. Duty cycle came out **96.4 %** (1,730 s busy / 1,795 s window).

> `inter_turn_delay_s` is retained in `solo.yaml` for provenance but is
> unreachable. `new_inter_arrival_times` is empty for the same reason — the
> driver only records it from a session's second turn onward. Not a bug.

## Measurement gap that had to be closed first

The driver **records neither per-request E2E nor per-request TPOT to disk**.
E2E is computed at `agent_throughput.py:2310`, used for a rate tracker, and
dropped; TPOT survives only as three percentiles in the final summary. Case A's
analysis had to *back-solve* E2E from `TTFT + (gen−1) × TPOT`.

For a latency study those two are the subject, so `scripts/apply_solo_metrics.py`
(`SOLO_M1`) adds `new_e2es` and `new_tpots` to the 1 Hz JSONL, index-aligned with
`new_ttfts`. Additive only, idempotent, anchored on exact source text. Verified
offline (alignment invariant, filtered-sample handling, default on untouched call
sites) and then verified in the **loaded** module on the jump host.

**Cross-check it bought:** predicted `TTFT + (gen−1)×TPOT` minus measured E2E is
**+0.0 ms at every percentile across all 102 requests**. The composition identity
holds exactly, which retroactively validates the method Case A had to rely on.

---

## TTFT — the prefill floor

| percentile | **solo (n=102)** | Case A sustain (n=2,702) | ratio |
|---|---|---|---|
| min | 689 | 419 | — |
| p10 | 707 | 1,973 | 2.79× |
| p25 | 999 | 2,966 | 2.97× |
| **p50** | **1,563** | 4,531 | **2.90×** |
| p75 | 1,985 | 6,660 | 3.36× |
| **p90** | **2,899** | 9,170 | **3.16×** |
| p95 | 3,336 | 11,182 | 3.35× |
| **p99** | **4,472** | 16,706 | **3.74×** |
| max | 5,554 | 22,915 | 4.13× |
| mean | 1,668 | 5,186 | 3.11× |

Whole-run (n=145, includes ramp): p50 1,165 / p90 2,646 / p99 4,432 ms.

**Reading: roughly 2/3 of Case A's TTFT was queueing, not compute.** But the raw
ratio overstates it, because the two runs did not draw the same inputs — 102
samples cannot reproduce a 235K-token p99. Bucketing by input length removes that
confound:

| input bucket | solo n | solo mean | Case A n | Case A mean | **load penalty** |
|---|---|---|---|---|---|
| 0–50K | 25 | 773 ms | 664 | 4,641 ms | **6.00×** |
| 50–80K | 34 | 1,272 ms | 827 | 4,911 ms | **3.86×** |
| 80–120K | 28 | 1,985 ms | 674 | 5,256 ms | **2.65×** |
| 120–160K | 10 | 2,946 ms | 303 | 5,671 ms | **1.93×** |
| 160–300K | 5 | 4,502 ms | 234 | 6,878 ms | **1.53×** |

This is the most informative table in the run. **The queueing penalty is a nearly
constant ~3.9 s additive term, not a multiplier**: Case A mean minus solo mean is
3,868 / 3,639 / 3,271 / 2,725 / 2,376 ms across the five buckets. A short request
pays the same wait as a long one, so the *relative* damage falls from 6.0× to
1.5× purely because the denominator grows.

Operationally: **under Case A load, a small request is not fast.** Its 0.8 s of
work is buried under ~3.9 s of waiting for other people's prefill. The service is
not latency-fair across request sizes.

**Solo TTFT vs input is close to linear.** 773 ms at <50K → 4,502 ms at >160K:
5.8× latency for ~5× tokens (ratio 1.17). Case A's own exponent was 1.18 — the
same mild superlinearity, so that curvature is prefill compute, not congestion.

---

## TPOT — the decode floor

| percentile | **solo (n=102)** | Case A sustain | ratio |
|---|---|---|---|
| min | 7.78 | — | |
| p25 | 9.84 | — | |
| **p50** | **10.68** | 14.8 | **1.39×** |
| **p90** | **12.75** | 17.8 | **1.40×** |
| **p99** | **14.65** | 20.9 | **1.43×** |
| max | 15.41 | — | |
| mean | 10.83 | — | |

**The decode path degrades far less than prefill: 1.4× under load vs 2.9× for
TTFT.** That is the expected signature of PD disaggregation working — the decode
leg runs a bounded batch of token-generation steps while the prefill leg absorbs
the bursty, size-variable arrival queue.

**7.78 ms is the hard floor of this deployment** — one MTP-accelerated decode step
on GLM-5.2-MXFP4 at dp8 on 8×MI355X. At 93 tok/s/request, single-stream.

**TPOT is flat-to-improving in generation length**, which is worth stating because
the naive expectation is the opposite (KV grows, attention cost grows):

| gen length | n | mean TPOT |
|---|---|---|
| 0–100 | 32 | 11.99 ms |
| 100–500 | 29 | 10.84 ms |
| 500–2,000 | 22 | 10.33 ms |
| 2,000–6,000 | 9 | **9.31 ms** |
| 6,000+ | 10 | 9.56 ms |

Short generations are *slower per token*. Two effects, both amortization rather
than attention cost: MTP's draft-model warm-up is paid once per request, and a
31-token response has too few steps to hide scheduling overhead. Growing KV does
not dominate even at 14K output tokens.

**MTP acceptance is load-independent: 2.069 solo vs 2.036 Case A (1.6 % apart).**
The speculative path is not something load takes away. Both sit far above the
YAML's assumed 1.56 and safely below the 4.00 that would signal a repetition loop.

---

## E2E — measured, not derived

| percentile | solo (ms) |
|---|---|
| min | 998 |
| p25 | 2,493 |
| **p50** | **4,124** |
| p75 | 14,090 |
| **p90** | **48,140** |
| p95 | 89,961 |
| **p99** | **121,243** |
| max | 163,725 |
| mean | 16,957 |

**This ladder is dominated by the output-length distribution, not by the server.**
The profile spans 320 → 17,000 output tokens; at a ~10.7 ms floor those alone are
3.4 s and 182 s of pure decode. Decomposing:

| gen length | n | E2E p50 | E2E p90 | max |
|---|---|---|---|---|
| 0–100 | 32 | 2.11 s | 3.40 s | 4.87 s |
| 100–500 | 29 | 3.28 s | 5.21 s | 8.40 s |
| 500–2,000 | 22 | 13.51 s | 17.64 s | 18.48 s |
| 2,000–6,000 | 9 | 30.08 s | 47.16 s | 48.28 s |
| 6,000+ | 10 | 92.59 s | 125.65 s | 163.72 s |

Within a bucket the spread is small; across buckets it is 44×. **E2E percentiles
at this workload measure the workload, not the service.** Any E2E target that does
not name an output length is unfalsifiable.

### This settles the `sla.e2e_p50_ms: 4500` question

Case A missed it by 2.5× (derived p50 ≈ 11.1 s) and the analysis left open
whether that was a deployment shortfall or a mis-specified target.

**Solo p50 E2E is 4,124 ms — inside the 4,500 ms target.** So the target *is*
achievable by this deployment, at concurrency 1 and at the profile's median
output length. It is not achievable under Case A's load. The target is best read
as a **latency-floor spec, not a capacity spec**; nothing in the YAML says which,
and `args.sla_cfg` is parsed and never consumed either way.

---

## The SLA block, evaluated at concurrency 1

| SLA | target | solo | Case A |
|---|---|---|---|
| `ttft_p90_ms` | ≤ 30,000 | **2,899** (10.3× margin) | 9,170 (3.3×) |
| `success_rate` | ≥ 0.97 | **1.000** (0 errors) | 0.988 |
| `e2e_p50_ms` | ≤ 4,500 | **4,124 ✓** | ~11,100 ✗ |

All three met at concurrency 1. Case A meets two of three.

---

## Cache and kvd

Actual cache hit **88.95 %** per-request mean (ideal 89.0 %) — the workload's
target reproduced exactly, same as Case A's 89.2 %. Run-level actual reads
**91.2 %** (efficiency 102.5 %) because the run-level figure is token-weighted
while the per-request mean is not; long prompts carry marginally better hit rates.
Eviction of the expected prefix: **0.0 %**.

kvd delta over the run:

| counter | before | after | delta |
|---|---|---|---|
| gets | 1,260 | 1,260 | **+0** |
| hits | 1,260 | 1,260 | **+0** |
| misses | 0 | 0 | +0 |
| sets | 108,754 | 109,876 | +1,122 |
| evictions | 53,576 | 54,782 | +1,206 |
| entries | 47,732 | 47,648 | −84 |

**Zero gets.** Every request nested inside a prefix the GPU radix tree already
held, so kvd was written to and never read from. This is the same asymmetry Case A
showed (72,844 sets to 452 gets), pushed to its limit by a warm cache and a single
stream. It confirms again that **Case A's shape verifies kvd correctness and does
not exercise tiering** — an open item that this run does not close either.

---

## What this run does not tell you

1. **It is not a capacity measurement.** Emergent throughput was 0.066 qps against
   Case A's 0.75. Sustain uncached TPM 29,676 (**3,710 /GPU**) against Case A's
   425,522 (53,190 /GPU) — a 14× gap. Nothing here says what the service can
   sustain; that is what Case A is for.
2. **n=102 in the measured window.** p50/p90 are solid; **p99 rests on ~1
   request** and should be read as an indication. This is intrinsic to concurrency
   1 — buying samples would have meant shortening requests, which would have
   broken shape-parity with Case A and invalidated every comparison above.
3. **Input percentiles under-sample the tail** (solo p99 184.6K vs the spec's
   235K) for the same reason. The bucketed table exists to work around this;
   the unbucketed TTFT ratios do not, and are the weaker number.
4. **The 3.9 s additive penalty is measured at one load point** (Case A's ~27
   live sessions, in-flight p50 17). Whether it is flat, linear or knee-shaped in
   concurrency needs a sweep — this is 1 point and Case A is the other.
5. **Post-patch, like everything else here.** Stock `merged-e` cannot run this
   shape at all.

## Reproduce

Full ordered steps, prerequisites and verification gates: **`../REPRODUCE.md`**.
Condensed below — `W` is the jump host's scratch dir, and `$W/scripts/` is this
kit's `scripts/` staged there.

```bash
W=/mnt/vast/c_huggingface/bench_20260801
python3 $W/scripts/apply_solo_metrics.py $W/agbench/agent/agent_throughput.py
find $W/agbench -name __pycache__ -type d -exec rm -rf {} +     # stale bytecode has burned this tree
ssh root@149.28.124.225 "cd $W && TAG=full RAMP=400 SUSTAIN=1800 setsid bash scripts/solo_run.sh < /dev/null"
python3 $W/scripts/solo_analyze.py <run>/metrics.jsonl --phase sustain
```

Every table above recomputes from this kit's own artifact — verified:

```bash
python3 scripts/solo_analyze.py results/metrics.jsonl.gz --phase sustain   # all solo ladders
python3 scripts/compare_vs_caseA.py                                        # the vs-Case-A tables
```

Nothing is read from `summary.json`.
