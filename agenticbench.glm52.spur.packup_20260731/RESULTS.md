# RESULTS — Optimus-AgenticBench Case A on infera + GLM-5.2-MXFP4 (spur)

Run: `caseA_full/2026-07-31-17-10-30`, ramp 400 s + sustain 3600 s (4007 s total),
2919 requests sent, kv-aware ON, **prefill kvd ON / decode kvd OFF**, **MTP OFF**,
`--context-length 262144`.

---

## Goal 1 — correctness

Graded by `scripts/correctness.py` through the router, not by the bench (the bench
sends synthesized filler and grades nothing).

| test | result |
|---|---|
| short factual, chat template, `temperature=0` | **4/4** |
| needle-at-depth, ~120,000-token prompt, 5 depths | **5/5** |

```
depth=  5%  OK   9.22s  prompt_tok=120047  cached=120000  want=6159362  got 6159362
depth= 25%  OK  11.52s  prompt_tok=120045  cached=120000  want=3331179  got 3331179
depth= 50%  OK  10.55s  prompt_tok=120046  cached=120000  want=5271814  got 5271814
depth= 75%  OK   8.26s  prompt_tok=120046  cached=120000  want=8251068  got 8251068
depth= 95%  OK   7.09s  prompt_tok=120047  cached=120000  want=5385227  got 5385227
```

The needle test is the one that matters here: the 4 short prompts are ~5 tokens, i.e.
a single prefill chunk, and say nothing about the multi-chunk path. Case A prompts run
9-29 chunks against an 8192-per-rank chunk size.

**Stated honestly:** this 5/5 was obtained with a **warm L3 prefix cache**
(`cached=120000` on every depth). The same suite scored 3/5 and 4/5 on earlier
cold-prefill runs at ctx=131072. That dependence on cache state is real and measured;
its mechanism is **not established** and is not asserted. See `notes/needle_resolved.md`.

## Goal 2 — classic serving metrics

Sustain phase (3600 s, 2588 completed requests, qps 0.719):

| metric | p50 | p90 | p99 |
|---|---|---|---|
| **TTFT** | 4,543 ms | 8,821 ms | 13,070 ms |
| **TPOT** | 31.3 ms | 32.6 ms | 37.9 ms |

Whole-run (4007 s):

| metric | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT | 4,983 ms | 4,445 ms | 8,652 ms | 13,012 ms |
| TPOT | 30.9 ms | 31.2 ms | 32.5 ms | 37.9 ms |

Throughput, sustain phase:

| | value |
|---|---|
| input TPM (incl. cache) | 3,668,448 |
| **uncached TPM (real prefill work)** | **407,291** (50,911 /GPU) |
| generation TPM | 34,997 |
| qps | 0.719 |

**Cache:**

| | |
|---|---|
| ideal hit rate | 0.8899 |
| **actual hit rate** | **0.8900** |
| efficiency (actual/ideal) | **1.0002** |
| eviction rate | **0.0** |
| server-reported | yes (`--enable-cache-report`) |

Hitting the configured 0.89 to four decimals with zero eviction is the guide's primary
sanity check, and it passes. `cached_tokens` is read from the server's
`usage.prompt_tokens_details`, not modelled.

**Requests:** 2919 sent, 2782 completed, **96 errors, success rate 0.953** (SLA 0.97).

All 96 errors are **client-side 240 s timeouts**, not server failures:
`aiohttp.ClientTimeout(total=240)` is hardcoded at `agent_throughput.py:928`. With
p99 input near 227K tokens plus a long generation tail, some requests legitimately
exceed 240 s. Both engine legs logged **0 GPU faults and 0 scheduler exceptions** for
the entire 67-minute window. The failure is in the load generator's timeout, not the
deployment — but the 0.953 is reported as measured rather than adjusted.

## Goal 3 — MTP acceptance rate

**N/A. MTP is off in this deployment**, by operator instruction (the DPA+PD+MTP fix is
not merged with kv-aware yet). No `--speculative-*` argument appears in either leg.

The bench's `acceptance_length` is deliberately **not quoted**: without speculative
decoding it degenerates to ~1 and would read as a measurement of a feature that is not
running. `acc_len: 1.0` / `mtp_draft_tokens: 1` were set in the YAML for the same
reason — left at the shipped 1.56/5 the driver reports a fabricated MTP-adjusted TPS.

Prior MTP measurements on this stack exist in the two spur kits (acceptance 2.79-2.93)
and may be cited **as prior measurement, clearly labelled** — they are not from this run.

## Goal 4 — session concurrency, turns, context length

**Live session concurrency** (3955 ticks from `metrics.jsonl`):

| | p50 | p90 | p99 | min | max |
|---|---|---|---|---|---|
| `num_sessions_active` | **38** | **47** | **52** | 1 | 54 |
| in-flight requests | 28 | 36 | 42 | — | 46 |

In-flight peaked at 46 against `max_inflight=48` and **never pinned**, so the workload
set the load, not backpressure. (Contrast the first attempt at rate 0.145, aborted
because it did pin — see Limitations.)

**Turns per session** (337 sessions observed):

| | p50 | p90 | p99 | mean | max |
|---|---|---|---|---|---|
| observed | 3 | 18 | 46 | 7.20 | 69 |
| spec | 3 | 20 | 103 | — | — |

p50 matches exactly; p90 is close; **p99 is truncated** — a 3600 s window cannot
reproduce a p99 session whose modelled lifetime is ~3557 s. Observed session lifetimes
ran 12 s to 3219 s (mean 684 s), i.e. the longest session nearly spans the window.

**Context length** (prompt tokens, actual):

| | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| observed | 85,172 | **73,862** | **151,526** | **226,854** |
| spec | — | 74,000 | 155,000 | 235,000 |

Within 2-4 % at every percentile. This is why the engine was moved to
`--context-length 262144`: at 131072 the clamp truncated **16.1 %** of the
distribution and the probe showed p90 and p99 both pinned at exactly 131,072. At
262144 only ~1.4 % clamps.

**Generation length:** mean 802, p50 286, p90 2283, p99 6301 tokens.

## kvd counters across the run

    before                          after
    entries        47,726           entries        47,587
    host_bytes     84,419,652,096   host_bytes     84,328,058,880
    long_bytes    185,584,508,928   long_bytes    415,332,043,776
    gets_total         47,713       gets_total         47,731
    hits_total         47,713       hits_total         47,731
    sets_total        110,469       sets_total        241,631
    misses_total            0       misses_total            0
    evictions_total    57,781       evictions_total   189,076

Case A wrote heavily (+131,162 sets, long tier to 415 GB) and evicted heavily
(+131,295), but served almost nothing from L3 (+18 gets). That is **expected and not a
defect**: Case A is one shared prefix with a 0.89 hit rate, so sglang's in-GPU radix
cache satisfies nearly every hit before L3 is consulted. The engine-reported cache hit
rate of 0.890 is the GPU tier doing its job.

kvd's read path was proven separately and cleanly — see below.

## kvd read-back, proven (separate experiment)

Restart-and-replay, the only attribution that survives scrutiny (a latency win proves
nothing when the GPU radix cache can serve a repeated prefix without touching L3):

    before                    after
    gets_total        0       gets_total   12,942     <- +12,942
    hits_total        0       hits_total   12,942     <- +12,942
    sets_total   12,942       sets_total   12,942     <- FLAT

Engine killed (all 8 GPUs polled to VRAM 0 %), kvd daemon deliberately kept alive,
byte-identical prompts replayed. 100 % hit, zero re-stores, and the server independently
reported `cached=120000` of `prompt_tok=120047` on every depth. Full write-up in
`notes/kvd_readback_proof.md`.

Combined with the GPU-fault fix, this is the **first demonstration of a full kvd L3
round-trip on spur** at 120K-token prompts — a regime no prior kit exercised.

## Limitations — what this run does NOT establish

* **No A/B against kvd-off.** kvd was ON (prefill) for the whole measured window. No
  claim is made that kvd improved any serving metric; Case A's single shared prefix
  creates no eviction pressure on the GPU tier by construction.
* **No MTP.** Goal 3 is N/A, not measured-and-poor.
* **Case B not run** (needs a 520K-context engine).
* **One run, no repeat.** No confidence interval on any percentile.
* **p99 turns-per-session is truncated** by the 3600 s window, as the guide predicts.
* **`max_sessions=128` was reached at t≈1089 s**, so new-session creation stopped for
  the remainder. The live population plateaued at 38-48 by that ceiling as much as by
  service time; a larger `max_sessions` would be needed to let the population find its
  own equilibrium.
* **Success rate 0.953 is below the 0.97 SLA**, attributable to the client's hardcoded
  240 s timeout rather than to server errors — but not corrected for, and reported as
  measured.
* **The needle 5/5 depends on a warm cache** (see Goal 1). The cause is not explained.

## First attempt, aborted — recorded because it is evidence

The first Case A launch used `new_session_rate = 0.145`, scaled linearly from the probe
(~22 live sessions at 0.10, targeting 32). It was **aborted at t=1195 s**: in-flight
pinned at the 48 cap (25 of the last 120 ticks at cap, mean 44.2) and live sessions
climbed monotonically 40 -> 57. Backpressure was setting the load, so the run was not
the configured workload.

The linear assumption was wrong: it predicts ~26 in-flight at 0.145; measurement gave
44-48. Service time grows with load, so the response is superlinear. The rerun anchored
on the known-stable measured point instead (0.10 -> in-flight 10-27) and stepped to
**0.110**, which held in-flight at p50 28 / max 46 without pinning.

Artifacts preserved at `/shared_nfs/yihou_agentbench/bench/caseA_full_ABORTED_saturated/`.
