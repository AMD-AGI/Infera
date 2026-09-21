# P8D8 concurrency sweep — running results

GLM-5.2 MXFP4 · 1P1D P8D8 TP8/DP8 · 137 prefill / 136 decode · MI355X x8 each
Image `infera-sglang:v0519-yihou-0917-nextnfix-hicache` (NextN fusion fix + PR #37152)
MTP EAGLE 5/6/1 with **simulated acceptance 3.61** — the config knob is
`DECODE_SIMULATE_ACC_LEN`, which `engine.sh:151` maps onto the engine env
`SGLANG_SIMULATE_ACC_LEN`; both names appear in logs and refer to the same setting
`--max-running-requests 256`, prefill HiCache ratio 1.5, decode HiCache off
**`SGLANG_DSA_FUSE_TOPK=0` on decode** — workaround for issue.md 3.3, see below
All points run against ONE deployment launched 2026-09-19 06:49 UTC.

| point | dur (s) | tok/s/chip | input/chip | output/chip | total tok/s | TTFT p50 | TTFT p90 | ITL p50 | ITL p90 | intvty p50 | profiled | errors | rails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c080** | 3629.7 | **19,575** | 19,420 | 155.1 | 313,202 | 5.76 s | 23.84 s | 14.67 ms | 21.53 ms | 68.2 | 9,480 | 5 | 0/0 |
| **c112** | 3629.5 | **18,180** | 18,028 | 152.0 | 290,883 | 11.47 s | 70.59 s | 15.92 ms | 22.22 ms | 62.8 | 9,240 | 13 | 0/0 |
| c144 | running since 10:23:33 | | | | | | | | | | | | |
| **c192** | 3629.6 | **7,409** | 7,338 | 71.6 | 118,547 | 132.82 s | 443.83 s | 11.23 ms | 15.75 ms | 89.0 | 4,339 | 185 | 0/0 |
| **c256** | 3627.3 | **5,329** | 5,274 | 55.1 | 85,262 | 159.51 s | 510.12 s | 10.18 ms | 13.82 ms | 98.2 | 3,659 | 487 | 0/0 |

`errors` = `records_error_dropped`; all are `InvalidInferenceResultError`.
`rails` = `transport retry counter exceeded` + `wqe is not posted`, counted before/after.

## Reference points, for orientation only

| run | shape | accept | tok/s/chip | ITL p50 |
|---|---|---|---|---|
| reference c40 | P4D4 TP4 | simulated 3.61 | 20,711 | 12.38 ms |
| reference c32 | P4D4 TP4 | simulated 3.61 | 12,448 | 11.16 ms |
| phase-1 c72 | P8D8 TP8 | **real 2.57** | 19,327 | 24.01 ms |
| **this c080** | P8D8 TP8 | simulated 3.61 | **19,575** | 14.67 ms |

Not like-for-like with the references: different shape (TP8 vs TP4), different
concurrency, and this sweep additionally carries `SGLANG_DSA_FUSE_TOPK=0`, which the
references do not.

## The fused-indexer workaround did not cost throughput here

| | tok/s/chip | duration | note |
|---|---|---|---|
| fused path ON | 18,819 | 2,640 s | **truncated** by the GPU memory fault; last ~3 min on 7 decode ranks |
| **fused path OFF** | **19,575** | 3,630 s | clean, 0 rail faults |

4 % higher with the workaround. **This is not a clean A/B** — different durations, and
the first run was degraded at the end. The defensible statement is only that the feared
cost did not show up, not that disabling the fused indexer is faster.

## Open, not chased mid-sweep

**5 `InvalidInferenceResultError` appeared in BOTH runs, identical count.** They are
therefore not caused by the DSA fault (the clean run has no crash) and not by the
phase-1 cross-rail wedge (0 rail faults, same-rail pinning verified). An independent,
low-rate failure path, currently unexplained. Recorded; not investigated during the
sweep.


## The curve has already turned: peak throughput is at or below CONC 80

| | c080 | c112 | change |
|---|---|---|---|
| per-GPU throughput | 19,575 | **18,180** | **-7.1 %** |
| TTFT p50 | 5.76 s | 11.47 s | 2.0x worse |
| **TTFT p90** | 23.84 s | **70.59 s** | **3.0x worse** |
| ITL p50 | 14.67 ms | 15.92 ms | 8.5 % worse |
| interactivity p50 | 68.2 | 62.8 | -7.9 % |
| errors | 5 | 13 | 2.6x |

Adding concurrency bought **no throughput and tripled tail latency**. That is the
signature of a system already past saturation at 80: extra requests queue rather than
execute, so latency grows roughly linearly with the excess while throughput flattens
and then declines.

**The peak therefore lies at or below the sweep's lowest point.** The remaining points
(144 / 192 / 256) will map the declining branch and test whether 256 hits the
out-of-memory end state the mission accepts — both useful — but **the maximum itself is
outside the measured range**. Locating it would need points below 80 (e.g. 40, 56).

Note this is NOT the `max_running_requests` ceiling: that was raised to 256 for exactly
this reason, and 112 is well inside it. The saturation is real, not an artefact of the
scheduler cap.

---

## ⚠ CORRECTNESS CAVEAT — the deployment produces GARBLED output

Discovered 2026-09-19 11:12 UTC by sending three probe requests straight at the router
while c144's warmup was stalled. All three came back degenerate:

```
'1!au!\n1.  **Analyze the Request:** The user is 1.  **Analyze the Request:** 1.  ...'
'1!au1. Identify the core request: the user wants the first 10 prime numbers...'
'1!au!\n2.  **Identify the 10th!au!au!au!au!au!au!au!au!au!au!au!au!au!au!au!au!...'
```

This is the GLM-5.2 MTP garbled-decode signature: first token roughly right, then
degenerate repetition.

**At the same moment `spec_accept_length` read 3.52–3.78 on all eight ranks.**

### The acceptance metric proves nothing here, and I had been reporting it as if it did

Simulated acceptance **forces** the engine to commit ~3.61 draft tokens per verify step
whether or not the target model would have accepted them. The gauge therefore reports
the value it was told to report. Throughout this sweep I checked "acceptance is holding
near 3.61" and treated it as a health signal — **it is not one, and that reading is
withdrawn.** It confirms only that the forcing is active.

### Why "simulation causes the garbling" does not fit the evidence

An earlier deployment — simulation **OFF**, fused DSA indexer **ON** — produced
**coherent** output, 16/16 correct, at **real acceptance 4.69**.

Real 4.69 **exceeds** the forced 3.61. Simulation is therefore committing *fewer*
tokens than the target would have accepted on its own, which is the conservative
direction and should not corrupt anything.

So the leading suspect is the one variable this deployment adds:
**`SGLANG_DSA_FUSE_TOPK=0`**, the §3.3 crash workaround, plausibly degrading the draft
path. **This is a candidate, not a finding** — no single-variable test has been run.

### What cannot be recovered

`aiperf_artifacts/profile_export.jsonl` stores only `metadata` and `metrics`; **response
text is not retained**. Whether c080 and c112 produced garbled output therefore **cannot
be determined from the artifacts**. It is not inferable from output-length statistics
either, and no such inference is made here.

### Standing of the c080 / c112 numbers

They measure the cost of committing ~3.61 tokens per verify step, which is the same
quantity the reference c32/c40 measured — and that kit **explicitly waived
correctness** for the same reason. On that footing the timings stand.

What is *not* the same as the reference: this sweep additionally carries
`SGLANG_DSA_FUSE_TOPK=0`, which the reference does not, and which is the leading
suspect for the garbling. Quote these numbers as timing under forced acceptance with
correctness waived **and** with a non-reference kernel path.


---

# RUN 2 — the configuration of record (2026-09-19, from 12:14 UTC)

The run above is **superseded**. It carried `SGLANG_DSA_FUSE_TOPK=0`, a workaround that
was withdrawn once the deployment was found to be emitting garbled text. This run
replaces it with `index_share_for_mtp_iteration=false`, which keeps the fused indexer
(verified in `dsa/utils.py::should_use_dsa_fused_topk`) and is the peer session's
evidence-backed mitigation for the issue.md 3.3 memory fault.

Deployment healthy/up 2026-09-19 12:14:08 UTC (launch.sh started 11:39:47); all points share it.
Config: IndexShare **off**, fused top-k **on**, custom all-reduce **on**,
simulated acceptance **3.61**, `max_running` 256, warmup 10/lane.

| point | dur (s) | tok/s/chip | input/chip | output/chip | total tok/s | TTFT p50 | TTFT p90 | ITL p50 | ITL p90 | intvty p50 | profiled | errors | rails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c080** | 3629.3 | **19,513** | 19,356 | 157.7 | 312,212 | 5.69 s | 22.12 s | 14.89 ms | 21.28 ms | 67.2 | 9,483 | 3 | 0/0 |
| **c112** | 3629.7 | **19,014** | 18,857 | 156.3 | 304,217 | 11.30 s | 64.47 s | 15.92 ms | 21.75 ms | 62.8 | 9,536 | 5 | 0/0 |
| **c144** | 3628.0 | **12,555** | 12,442 | 113.5 | 200,879 | 52.45 s | 209.33 s | 14.40 ms | 19.88 ms | 69.4 | 6,862 | 32 | 0/0 |
| **c192** | 3629.6 | **7,409** | 7,338 | 71.6 | 118,547 | 132.82 s | 443.83 s | 11.23 ms | 15.75 ms | 89.0 | 4,339 | 185 | 0/0 |
| **c256** | 3627.3 | **5,329** | 5,274 | 55.1 | 85,262 | 159.51 s | 510.12 s | 10.18 ms | 13.82 ms | 98.2 | 3,659 | 487 | 0/0 |

## Turning IndexShare off costs essentially nothing in throughput

| | tok/s/chip | ITL p50 | intvty p50 |
|---|---|---|---|
| IndexShare **on**, fused **off** (garbled run) | 19,575 | 14.67 ms | 68.2 |
| **IndexShare off, fused on** (this run) | **19,513** | 14.89 ms | 67.2 |

0.3 % apart — inside run-to-run noise. The mitigation is close to free at this point.
Whether that holds at higher concurrency is unmeasured.

## ⚠ Correctness is waived, by construction, exactly as in the reference kit

This run is **still garbled**, and that is now understood rather than alarming.
Simulated acceptance forces a fixed *count* of accepted draft tokens
(`generate_simulated_accept_index`, `match-expected` / `real-draft-token`) — it decides
how many, not which ones are correct. Force-accepting a token the target would reject
sends EAGLE at topk=1 down the draft's own branch and the error compounds.

The reference c32/c40 kit ran the same `DECODE_SIMULATE_ACC_LEN=3.61` and states
correctness was **explicitly waived**. So these timings sit on the same footing as the
reference's, and the acceptance gauge (3.5-3.8) carries no correctness information at
all — it reports the value it was told to report.

**Hypothesis falsified along the way:** `SGLANG_DSA_FUSE_TOPK=0` was predicted, in
writing beforehand, to be the cause of the garbling. Restoring the fused path did not
fix it. Simulated acceptance is the only variable aligned with coherent-vs-garbled
across all three deployments.


## The decline is gentler than in run 1

| 80 -> 112 | run 1 (IndexShare on, fused off) | **run 2 (IndexShare off, fused on)** |
|---|---|---|
| tok/s/chip | 19,575 -> 18,180 (**-7.1 %**) | 19,513 -> 19,014 (**-2.6 %**) |
| TTFT p90 | 23.84 -> 70.59 s (3.0x) | 22.12 -> 64.47 s (2.9x) |
| errors | 5 -> 13 | 3 -> 5 |

Throughput still falls with concurrency, so the peak remains at or below 80, but run 2
loses less of it and carries fewer errors. Two points each, one run each — suggestive,
not a controlled comparison, and the two runs differ in two settings at once.

---

## c144 — CORRECTED: it DOES complete, very slowly. Not a collapse.

**Retraction.** An earlier revision of this section said c144 "does not complete" and
called it a reproducible PD queueing collapse, and recommended stopping the sweep on
that basis. **That was wrong.** c144 drained its warmup and entered the profiling phase
at ~16:40 (1,595/1,598 returned, errors 0). The recommendation was withdrawn before it
was acted on.

What I mistook for a stall was a **crawl**: progress of ~8 requests/minute while a
fraction of requests each burn the full 1,800 s KV-wait timeout before failing and
releasing. Sampling it twice ten minutes apart showed no movement and I read that as
wedged; sampling it four minutes apart showed 1,478 -> 1,510.

**This also casts doubt on run 1's c144.** I killed that one at 1,180/1,598 believing it
hung. On this evidence it may well have drained too, given time. I cannot claim it would
have — but I can no longer claim it would not.

The mechanism below is still accurate and worth keeping; only the "does not complete"
conclusion was wrong.

## c144 — the slow-warmup mechanism (accurate)

**Two attempts, two configurations, same *slow* warmup** (the first was killed before it could finish, so "same failure" overstates it):

| attempt | config | warmup stalled at |
|---|---|---|
| run 1 | fused top-k **off**, IndexShare **on** | 1,180 / 1,598 |
| run 2 | fused top-k **on**, IndexShare **off** | 1,478 / 1,598 |

The two runs differ in both switches and fail identically, so the failure belongs to
**CONC 144 on this shape**, not to either mitigation.

### The mechanism, from the decode log

```
Some requests fail to receive KV Cache transfer done signal after bootstrapping.
...
KVTransferError(bootstrap_room=...): Request ... timed out after 1800.0s
in KVPoll.WaitingForInput
```

Requests sit in `KVPoll.WaitingForInput` for the full
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT` of **1,800 s** and are then failed. First
occurrence 15:56:34 against a c144 start of 15:24:23 — exactly one timeout period.

### State while it is stalled — this is the diagnostic signature

| check | value |
|---|---|
| dead rank / abort | **none**; all 8 GPUs at 87-88 % VRAM |
| RDMA rail faults | **0** |
| all three health endpoints | **200** |
| **prefill** | **working** — batches still logging, several ranks with queue |
| **decode** | **idle** — `num_running_reqs` 0 on every rank |
| `num_transfer_failed_reqs_total` | 24 and climbing (rank 5 alone 11) |

**Prefill is busy and decode is starving.** Requests queue at prefill far longer than
decode is willing to wait for their KV, so decode times them out; the client never gets
them back and the warmup never drains.

### Why this is expected rather than mysterious

We already measured that **throughput peaks at or below CONC 80** — 80 → 112 lost 2.6 %
of per-GPU throughput while tripling TTFT p90. CONC 144 is far past saturation, so the
prefill backlog grows without bound and every KV transfer waits behind it. The log's own
suggestion to raise `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` is the wrong direction here:
it would make requests wait longer, not make the server faster.

### Consequence for the sweep — revised

c144 completes; it just pays a long warmup while timed-out requests drain. So the sweep
continues rather than stopping here. What the episode does establish is a **cost**: past
saturation, each point's warmup is lengthened by roughly one
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT` period, so 192 and 256 should be budgeted with
that in mind rather than assumed to take as long as c080 did.

The out-of-memory end state the mission anticipates has **not** been reached at 144.


## The saturation cliff is between 112 and 144

| point | tok/s/chip | vs previous | TTFT p50 | TTFT p90 | profiled | errors |
|---|---|---|---|---|---|---|
| c080 | 19,513 | — | 5.69 s | 22.12 s | 9,483 | 3 |
| c112 | 19,014 | -2.6 % | 11.30 s | 64.47 s | 9,536 | 5 |
| **c144** | **12,555** | **-34 %** | **52.45 s** | **209.33 s** | 6,862 | 32 |

80 to 112 costs 2.6 %. 112 to 144 costs **34 %**, triples TTFT p90 again to 209 s, drops
completed requests from 9,536 to 6,862 and raises errors from 5 to 32. The knee is
between 112 and 144, much closer to a cliff than to a curve.

### The counter-intuitive detail that confirms the mechanism

**c144's ITL (14.40 ms) and interactivity (69.4) are BETTER than c112's** (15.92 ms,
62.8). That is not a contradiction — it is the signature of a starved decode. Only a
small number of requests get their KV across, and those few decode against an almost
empty decode batch, so their inter-token latency is excellent. The queue is entirely on
the prefill/KV-transfer side.

So the per-request latency metrics improve while the system as a whole collapses. Read
ITL alone at 144 and the deployment looks healthier than at 112; read throughput and
TTFT and it plainly is not. **ITL is not a saturation signal in a PD deployment.**


## The full collapse, and the metric that lies about it

| conc | tok/s/chip | vs prev | TTFT p50 | TTFT p90 | ITL p50 | intvty p50 | profiled | errors |
|---|---|---|---|---|---|---|---|---|
| 80 | **19,513** | — | 5.69 s | 22.1 s | 14.89 ms | 67.2 | 9,483 | 3 |
| 112 | **19,014** | -2.6 % | 11.30 s | 64.5 s | 15.92 ms | 62.8 | 9,536 | 5 |
| 144 | **12,555** | -34 % | 52.45 s | 209.3 s | 14.40 ms | 69.4 | 6,862 | 32 |
| 192 | **7,409** | -41 % | 132.82 s | 443.8 s | 11.23 ms | **89.0** | 4,339 | 185 |

Throughput is flat to 112 and then halves twice. TTFT p90 reaches **7 min 24 s** at 192
and completed requests fall to 4,339 while errors reach 185.

### ITL and interactivity IMPROVE as the system collapses

This is the trap in the table. **c192 has the best ITL (11.23 ms) and the best
interactivity (89.0) of all four points** — better than c080, which is the only healthy
one.

The reason is structural to PD disaggregation: the queue forms on the prefill side, so
decode is progressively starved. The few requests whose KV does arrive decode against a
nearly empty decode batch and therefore enjoy excellent inter-token latency. The metric
is measuring the survivors, not the system.

**Do not use ITL or interactivity as a saturation signal in a PD deployment.** Both move
the wrong way across the cliff. Throughput per chip, TTFT p90 and the completed-request
count all tell the truth here; ITL actively misleads.


---

# FINAL — all five points complete, 2026-09-20 01:47:15 UTC

| conc | tok/s/chip | vs prev | TTFT p50 | TTFT p90 | ITL p50 | intvty | profiled | errors | rails |
|---|---|---|---|---|---|---|---|---|---|
| **80** | **19,513** | — | 5.69 s | 22.1 s | 14.89 ms | 67.2 | 9,483 | 3 | 0/0 |
| **112** | **19,014** | -2.6 % | 11.30 s | 64.5 s | 15.92 ms | 62.8 | 9,536 | 5 | 0/0 |
| 144 | 12,555 | -34 % | 52.45 s | 209.3 s | 14.40 ms | 69.4 | 6,862 | 32 | 0/0 |
| 192 | 7,409 | -41 % | 132.82 s | 443.8 s | 11.23 ms | 89.0 | 4,339 | 185 | 0/0 |
| 256 | 5,329 | -28 % | 159.51 s | 510.1 s | 10.18 ms | 98.2 | 3,659 | 487 | 0/0 |

## The end state is queueing collapse, not out of memory

The mission anticipated the top point failing on OOM. **It did not.** All five points
completed their full 3,600 s window and the deployment never ran out of memory. What
degrades is service: at 256 the TTFT p90 is **8.5 minutes**, errors reach 487, and
completed requests fall to 3,659 — 38 % of what CONC 112 achieved.

## Peak throughput is at or below 80 — the sweep's lowest point

80 and 112 are within 2.6 % of each other; everything above collapses. The maximum is
therefore **outside the measured range**, and locating it would need points below 80.
This was flagged after c112 and is confirmed by the full curve.

## ITL and interactivity invert — do not use them to detect saturation here

They improve **monotonically** as the system collapses: ITL 14.89 -> 15.92 -> 14.40 ->
11.23 -> **10.18 ms**, interactivity 67.2 -> 62.8 -> 69.4 -> 89.0 -> **98.2**. The best
readings of both belong to the **worst** point.

In PD disaggregation the queue forms on the prefill side, so decode starves; the few
requests whose KV arrives decode against an almost empty batch. These metrics measure
the survivors, not the system. **Throughput per chip, TTFT p90 and completed-request
count are the honest signals; ITL actively misleads.**

## Stability: 13 h 33 m fault-free

Deployment up 2026-09-19 12:14:08, driver finished 2026-09-20 01:47:15.
**Zero `Memory access fault`, zero `Fatal Python error`, zero RDMA rail faults** across
all five points, with `index_share_for_mtp_iteration=false` and custom all-reduce **on**.
For contrast, the same shape with IndexShare **on** faulted at 1 h 26 m.

## What these numbers are, and are not

- **Timing under forced MTP acceptance at 3.61.** Correctness is waived by construction
  — the deployment emits garbled text, exactly as the reference c32/c40 kit discloses
  about its own numbers. These are comparable to that kit's, not to a correctness run.
- **Carrying one non-reference setting**: `index_share_for_mtp_iteration=false`, needed
  to avoid the issue.md 3.3 memory fault. Measured cost at CONC 80: 0.3 %, inside noise.
- **Warmup cost grows with saturation**: 27 min at 80, ~2 h 11 m at 192, ~3 h 10 m at
  256, because past saturation a share of requests must each burn the full 1,800 s
  `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` before failing and releasing.
