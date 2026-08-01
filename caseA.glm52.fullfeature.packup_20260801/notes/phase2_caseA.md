# Phase 2 — Case A agentic bench

## Probe run (ramp 400 + sustain 600), 12:15–12:32 UTC

Purpose per CASE_AB_GUIDE Step 3: measure E2E so `new_session_rate` can be
re-solved. Shipped config otherwise untouched.

Artifacts: `results/caseA_probe/caseA_probe/2026-08-01-12-15-26/`
(`metrics.jsonl`, `metadata.json`), log `logs/caseA_probe.log`.

### Result — PASS, and the load knob needs no change

| metric | sustain phase | note |
|---|---|---|
| requests | 321 (592 sent, 576 completed) | success 97.3 % |
| **actual cache hit** | **88.9 %** | target 0.89 — **on spec** |
| ideal cache hit | 89.0 % | efficiency 99.0 %, eviction 1.0 % |
| TTFT p50 / p90 | 2,894 ms / 5,452 ms | |
| TPOT p50 / p90 | 14.5 ms / 18.7 ms | |
| **mean E2E** | **17.8 s** | TTFT + (genlen−1)×TPOT |
| in-flight p50 / max | 9 / **18** | cap 48 — **never bound** |
| live sessions p50 / max | 17 / 23 | target 32 |
| uncached TPM | 300,179 (37,522/GPU) | |
| peak prefill | 409,963 tok/s (51,245/GPU) | |

Realized input distribution p50 74,022 / p90 148,524 / p99 224,807 against the
spec triple 74,000 / 155,000 / 235,000 — the sampler is faithful.

### Step 3 re-solve → keep `new_session_rate: 0.10`

    rate = N_target / (E[turns] x (measured_E2E + E[delay]))
         = 32 / (9.50 x (17.8 + 18.0))
         = 0.0941 /s

The shipped value is `0.10`. Measured E2E (17.8 s) landed close to the 15 s the
config assumed, so the re-solve moves the rate by 6 % — inside noise. **The
shipped rate is confirmed by the guide's own formula; changing it is not
justified.**

### The N=17-vs-32 gap is real and is NOT a rate error — do not "fix" it by raising the rate

Little's law with the spec table predicts N = 0.10 x 9.5 x 35.8 = **34**.
Observed steady-state is **~17**, flat (by quarter: 19.5, 16.2, 14.4, 19.0 — no
downward drift, no runaway).

The deficit traces to the *realized* per-turn cycle, not the birth rate. The run
measured `Inter-Arrival Time (response + think) mean = 13.7 s` against a
predicted `E2E + E[delay] = 35.8 s`. Session lifetime is `E[turns] x E[cycle]`,
so a 2.6x shorter cycle gives a proportionally smaller standing population at
the same birth rate.

Why the cycle is short: the inter-turn delay triple is p50 4 / p90 31 / **p99
240** s, and the probe measured p99 158.7 s. A 1,000 s window cannot realize a
240 s tail, so the mean is censored downward. The same censoring truncates the
p99 turn budget (103 turns x ~34 s = ~3,400 s, i.e. 3.4x the whole probe).

**Consequence for the full run:** at 4,000 s the tail is far less censored, so
E[cycle] and E[turns] both realize closer to spec and N should rise toward the
target on its own — at the unchanged rate. Raising the rate to force N=32 from a
censored short probe would over-drive the full window. Empirically N scales
~linearly with rate here (N=17 at 0.0955 → N=32 would need ~0.18), but the guide
warns response is **superlinear**, and at 2x load in-flight would sit near the
48 cap that the probe peaked at 18 against. That is precisely the failure mode
the spur run hit: `max_inflight` binds and backpressure, not the config, sets
the load.

So: **full run at the shipped rate.** Report the achieved N honestly rather than
chase the nominal 32.

### The 4 errors (0.7 %) are client-side timeouts, not server faults

    Request 36 timed out    (t=292s)
    Request 232 timed out   (t=582s)
    Request 238 timed out   (t=597s)
    Request 258 timed out   (t=627s)

`agent_throughput.py:3107` — `asyncio.TimeoutError` on the driver's own
`aiohttp.ClientTimeout(total=240)`. Both engine logs are clean over the same
window: zero `abort`, zero `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, zero traceback,
`#retracted-req: 0`. These are the p99 output tail (17,000 tokens x ~15 ms/tok =
255 s) exceeding a 240 s client deadline — the workload's own tail crossing the
driver's fixed cap, not a deployment failure.

### kvd is genuinely serving under Case A — unlike the fixlen sweep

| counter | before | after |
|---|---|---|
| entries | 0 | 26,542 |
| gets | 0 | 800 |
| **hits** | 0 | **800** |
| misses | 0 | **0** |
| sets | 0 | 32,702 |
| host_bytes | 0 | 46.9 GB |

800 gets / 800 hits / **0 misses**. This is the signal Phase 1 could not produce:
`--dataset-name random` has no shared prefix, so the sweep's `cached_host_tok`
was ~0 and its cache-hit numbers were GPU-radix residue. Case A's 89 % shared
prefix is what actually exercises the L2/L3 tier.

Note `long_bytes` reached 45.2 GiB of the 64 G cap during a 1,000 s probe, so the
4,000 s run will hit the cap and begin evicting. That is correct behaviour, not a
fault — the cap is what keeps the container layer off the node's root disk
(patch 0003).

---

## Full run — attempts 1 and 2 CRASHED on an engine bug; attempt 3 PASSED after a patch

| attempt | leg TAG | outcome |
|---|---|---|
| 1 | p3 decode (stock) | **crashed at 125 s** — decode leg died, 114 client errors |
| 2 | p5 decode (stock + `SGLANG_DEBUG_DSA_ROWS=1`) | **crashed at 766 s**, same assert; root cause captured |
| 3 | p6 decode (**+ GLM52_P1V3 patch**) | **PASS — full 4,006 s, 98.8 %** |

Root cause and the fix are written up in full at repo root:
**`notes.dsa.mtp.crash.md`**, patch text in `patches/0004-*.txt`, applier
`scripts/apply_p1v3.py`. One-line version: the in-image `GLM52_P1V2` trim only
guards `real < padded`, but on a DP-attention **IDLE** rank under MTP
draft-extend the inequality inverts (`q_fp8` rows = 1, `q_offset` = `lengths` =
2), so no trim runs and `fast_topk_v2` asserts
`Expected lengths.size(0) == B`. P1V3 reconciles both sides to
`min(real, padded)`.

**The measured Case A numbers are therefore `merged-e + P1V3`, not stock
`merged-e`.** Stock merged-e cannot complete this workload at all.

### Result (run `caseA_full/2026-08-01-13-34-46`, ramp 400 + sustain 3600)

| metric | sustain | whole run |
|---|---|---|
| requests | 2,702 | 2,988 sent / 2,952 completed |
| **success rate** | — | **98.8 %** (18 errors) |
| qps (emergent) | 0.75 | 0.75 |
| **actual cache hit** | **89.0 %** | **89.2 %** |
| cache efficiency | — | **100.3 %**, eviction **0.0 %** |
| TTFT p50 / p90 / p99 | 4,533 / 9,172 ms | 4,378 / 8,940 / 16,492 ms |
| TPOT p50 / p90 / p99 | 14.8 / 17.8 ms | 14.9 / 17.9 / 20.9 ms |
| uncached TPM | 425,522 (53,190/GPU) | — |
| peak prefill | — | 686,052 tok/s (85,756/GPU) |
| in-flight p50 / max | 17 / **30** | cap 48 — **never bound** |
| live sessions p50 / p90 | 27 / 37 | target 32 |

vs SLA: `ttft_p90_ms ≤ 30,000` → **9,172 ms, 3.3x margin**.
`success_rate ≥ 0.97` → **0.988, met** (the spur run's 0.953 missed it).

Realized input p50 74,559 / p90 153,312 / p99 233,656 against spec 74,000 /
155,000 / 235,000 — within 1.1 % at every percentile.

### The population prediction held

The probe's N≈17 was diagnosed above as tail-censoring, not a rate error, with
the prediction that a 4,000 s window would let N rise toward target at the
unchanged rate. It did: live sessions by quarter **22.1 → 22.3 → 27.6 → 36.0**,
p50 27 / p90 37, straddling the nominal 32. Max session lifetime 3,978 s (vs the
probe's 1,005 s ceiling) confirms the tail was previously truncated.

`new_session_rate` was left at the shipped `0.10` throughout. **Not raising it
was correct** — had the probe's N=17 been "fixed" by roughly doubling the rate,
the full run would have landed near N≈70 and in-flight would have pinned at 48.

### MTP under Case A

`accept len` over 19,225 decode-batch lines: **mean 2.80**, min 1.20, max 4.00.
The driver's own per-request acceptance averaged **2.04** across 2,702 sustain
requests — well above the workload's declared `acc_len: 1.56`, so the shipped
"56 % acceptance @ 5 draft tokens" assumption is conservative for this stack.

431 lines read exactly `4.00`. Per CLAUDE.md principle 2 a sustained 4.00 is the
repetition-loop tell, but 431/19,225 = 2.2 % of batches, transient, against a
healthy 2.80 mean — this is small-batch saturation, not a degenerate loop.
`#retracted-req` was 0 for the entire run.

### kvd under the full run

| counter | before | after | delta |
|---|---|---|---|
| entries | 29,750 | 47,732 | +17,982 |
| gets | 808 | 1,260 | +452 |
| **hits** | 808 | 1,260 | **+452** |
| **misses** | 0 | 0 | **+0** |
| sets | 35,910 | 108,754 | +72,844 |
| evictions | 0 | 53,576 | +53,576 |
| host_bytes | 52.6 GB | 84.4 GB | +31.9 GB |

452 gets, 452 hits, **0 misses** — a perfect hit rate again. Evictions started
(53,576) as predicted once the L3 tier reached its 64 G cap; the run's own cache
efficiency stayed at 100.3 % with 0.0 % of the *expected* prefix evicted, so the
eviction pressure fell on cold entries only.

Note the asymmetry: 72,844 sets against 452 gets. This is the guide's documented
property — every request nests inside the **same** shared prefix, so the radix
tree holds one hot path and the GPU-side cache serves almost everything. kvd is
proven correct here, but Case A does not stress cache *tiering*; a
growing-prefix workload would be needed for that.
