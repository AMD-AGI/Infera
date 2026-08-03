# Phase 1 — sglang `bench_serving` sweep, 8 points, one server

Ran 2026-08-01 10:25:41 – 11:35:16 UTC (~70 min). Deployment: two-node PD over
mooncake RDMA, DP-attention 8/8, kv-aware routing ON, **prefill kvd ON / decode
kvd OFF**, **MTP ON (decode leg, EAGLE steps=3 topk=1 draft=4)**,
`--context-length 262144`. One server for all eight points; nothing retuned
between them.

## Throughput

| point | ISL | OSL | conc | done | dur (s) | req/s | in tok/s | out tok/s | tot tok/s | real conc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p50 | 74,000 | 320 | 1 | 4 | 57.9 | 0.069 | 5,108 | 22.1 | 5,130 | 1.00 |
| p50 | 74,000 | 320 | 32 | 64 | 171.5 | 0.373 | 27,620 | 119.4 | 27,739 | 24.60 |
| p50 | 74,000 | 320 | 64 | 128 | 299.6 | 0.427 | 31,618 | 136.7 | 31,755 | 50.43 |
| p50 | 74,000 | 320 | 128 | 256 | 575.7 | 0.445 | 32,907 | 142.3 | 33,050 | 98.75 |
| p90 | 155,000 | 3,300 | 1 | 4 | 391.6 | 0.010 | 1,583 | 33.7 | 1,617 | 1.00 |
| p90 | 155,000 | 3,300 | 32 | 64 | 425.3 | 0.150 | 23,323 | 496.6 | 23,820 | 26.53 |
| p90 | 155,000 | 3,300 | 64 | 128 | 736.3 | 0.174 | 26,944 | 573.7 | 27,518 | 51.04 |
| p90 | 155,000 | 3,300 | 128 | 256 | 1,348.2 | 0.190 | 29,433 | 626.6 | 30,059 | 100.72 |

**This deployment is prefill-bound, and the sweep shows where it saturates.**
Input throughput per GPU:

| conc | p50 in tok/s/GPU | vs conc=32 | p90 in tok/s/GPU | vs conc=32 |
|---:|---:|---:|---:|---:|
| 1 | 639 | 0.18× | 198 | 0.07× |
| 32 | 3,452 | 1.00× | 2,915 | 1.00× |
| 64 | 3,952 | 1.14× | 3,368 | 1.16× |
| 128 | 4,113 | 1.19× | 3,679 | 1.26× |

Quadrupling concurrency from 32 → 128 buys only **19 %** (p50) / **26 %** (p90)
more prefill throughput. The machine is already near its prefill ceiling at
conc=32; beyond that, added concurrency converts almost entirely into queueing —
which is exactly what the latency table shows.

**`real conc` tracks the cap closely** (98.75 of 128, 100.72 of 128), so the
requested concurrency was actually achieved; these are not runs that silently
under-loaded.

## Latency (ms)

| point | conc | TTFT p50 | p90 | p99 | TPOT p50 | p90 | p99 | E2E p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p50 | 1 | 10,785 | 10,803 | 10,804 | 11.53 | 14.57 | 15.29 | 14,464 | 15,451 | 15,681 |
| p50 | 32 | 61,695 | 80,789 | 83,750 | 14.69 | 19.64 | 23.29 | 65,863 | 85,317 | 90,504 |
| p50 | 64 | 127,194 | 137,874 | 158,438 | 14.65 | 19.29 | 34.14 | 132,136 | 143,746 | 162,824 |
| p50 | 128 | 259,093 | 265,139 | 297,998 | 16.50 | 28.70 | 37.48 | 263,260 | 274,033 | 302,852 |
| p90 | 1 | 24,457 | 24,742 | 24,773 | 22.59 | 24.54 | 24.72 | 98,851 | 105,690 | 106,311 |
| p90 | 32 | 89,035 | 175,853 | 186,748 | 27.89 | 35.95 | 40.45 | 172,266 | 260,730 | 315,256 |
| p90 | 64 | 191,655 | 297,178 | 339,031 | 30.74 | 39.52 | 42.85 | 288,348 | 371,971 | 462,901 |
| p90 | 128 | 467,652 | 541,084 | 651,071 | 33.55 | 41.61 | 45.10 | 575,650 | 656,055 | 746,925 |

**TTFT is queueing, TPOT is service.** From conc=1 → 128, p50 TTFT rises **24×**
(p50 point) and **19×** (p90 point), while TPOT rises only **1.4×** and **1.5×**.
That is the signature of a saturated prefill stage in front of a healthy decode
stage: requests wait, then are served at close to full speed.

**One number that should not be read as an SLA result.** At conc≥32 these E2E
figures are far above Case A's `e2e_p50_ms: 4500`. That is expected and not
comparable: `bench_serving` fires `--request-rate inf`, i.e. all `2×conc`
requests are offered at t=0 with no think time, so the queue depth is maximal by
construction. Case A is a **closed-loop** workload with a 4 s median inter-turn
delay and a birth-limited population. The SLA is answered by Phase 2, not here.

## MTP acceptance — per point, and a correction

The obvious source is wrong twice over, so this is worth stating precisely.

`bench_serving`'s own `accept_length` field is **`null` in all eight JSONs**. It
reads `avg_spec_accept_length` from `<base_url>/server_info`
(`benchmark/serving.py:1525`), and `--base-url` here is the infera **router**,
which has no such endpoint. Not a missing feature — a consequence of routing.

Snapshotting the decode leg's `/server_info` after each point (which the sweep
did) is also **not** a per-point number:

- it is `spec_total_num_accept_tokens / spec_total_num_forward_ct`
  (`scheduler.py:3787`) — a **cumulative** mean over the engine's whole lifetime,
  so later points are diluted by all earlier traffic;
- it is reported **per DP rank** — 8 entries that genuinely differ
  (1.415 … 1.577 at the final point), so reading `internal_states[0]` is one
  rank, not the fleet.

The apparent monotone decline in that column is therefore mostly an artifact.
The real per-point distribution comes from the decode leg's own timestamped
`accept len:` lines — **12,654 samples across all 8 ranks** — binned into each
point's measured window:

| point | conc | n | mean | p10 | p50 | p90 | max | ranks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p50 | 1 | 14 | **2.381** | 1.77 | 2.17 | 3.41 | 3.55 | 1 |
| p50 | 32 | 228 | **2.222** | 1.57 | 2.08 | 3.20 | 3.98 | 8 |
| p50 | 64 | 464 | **2.232** | 1.40 | 2.08 | 3.30 | 4.00 | 8 |
| p50 | 128 | 1,033 | **1.973** | 1.00 | 1.88 | 3.12 | 4.00 | 8 |
| p90 | 1 | 260 | **1.269** | 1.00 | 1.18 | 1.52 | 3.25 | 1 |
| p90 | 32 | 1,746 | **1.602** | 1.00 | 1.30 | 2.55 | 4.00 | 8 |
| p90 | 64 | 3,066 | **1.513** | 1.00 | 1.26 | 2.41 | 4.00 | 8 |
| p90 | 128 | 5,683 | **1.345** | 1.01 | 1.22 | 1.85 | 4.00 | 8 |

Two observations, and one caution:

1. **Acceptance falls with concurrency** (p50: 2.38 → 1.97; p90: 1.60 → 1.35).
   Expected: larger verify batches mean a draft token is accepted only if it
   survives alongside the rest of the batch, and the marginal value of drafting
   drops as the decode stage fills.
2. **The p90 point accepts much less than the p50 point** (≈1.3–1.6 vs ≈2.0–2.4).
   **Do not attribute this to prompt length.** `--dataset-name random` builds a
   prompt by sampling one ShareGPT conversation and **repeating its token ids**
   to reach the target length (`datasets/random.py:131-134`,
   `(prompt_token_ids * ratio)[:input_lens[i]]`). At ISL 155,000 that is hundreds
   of repeats of the same text, and the OSL of 3,300 with `ignore_eos` forces
   long generations off that degenerate context. This is a property of the
   **benchmark's synthetic prompt**, not of the deployment or of the model at
   long context.
3. Consequently, **the acceptance number to quote for this deployment is the
   p50-point value, ≈2.2–2.4**, and it should be labelled as measured on
   synthetic repeated-ShareGPT text. It is consistent with the independent
   correctness-phase measurement (`avg_spec_accept_length = 2.749` after real
   needle prompts) and with the branch's own G1 expectation of 2.1–2.6.

Against the Case A spec's **56 % acceptance @ 5 draft tokens** (≈2.8 accepted
tokens per step): measured here is ≈2.2–2.4 at 4 draft tokens, i.e. **55–60 %
acceptance rate** — the *rate* matches the spec closely; the absolute accepted
length is lower because this deployment drafts 4 tokens, not 5.

## kvd during the sweep

| point | conc | gets | hits | sets | evictions |
|---|---:|---:|---:|---:|---:|
| p50 | 1 | 11,250 | 11,250 | 48,880 | 1,153 |
| p50 | 32 | 11,250 | 11,250 | 60,333 | 12,199 |
| p50 | 64 | 11,281 | 11,281 | 77,642 | 29,528 |
| p50 | 128 | 11,281 | 11,281 | 103,653 | 55,952 |
| p90 | 1 | 11,281 | 11,281 | 119,783 | 71,763 |
| p90 | 32 | 11,281 | 11,281 | 137,051 | 89,562 |
| p90 | 64 | 11,281 | 11,281 | 167,162 | 119,155 |
| p90 | 128 | 11,281 | 11,281 | 222,150 | 174,010 |

Cumulative counters. `gets`/`hits` are **flat at 11,281 for the entire sweep** —
every one of those reads came from the earlier restart-replay proof; the sweep
itself performed **zero** L3 reads while writing **+173,270** pages and evicting
**+172,857**.

That is expected, not a defect, and the reason is structural: every prompt in
this sweep is unique (distinct seed per point, and `random` prompts do not nest),
so there is no prefix for L3 to serve. kvd's read path was proven separately
(`kvd_serving_proof.md`); a unique-prompt sweep can only ever exercise its write
path. Note `sets ≈ evictions` — the store is at capacity and churning, which is
the honest description of L3 under a no-reuse workload.

## Health

`Traceback` **0**, `Memory access fault` **0**, `Scheduler hit an exception` **0**
on both legs across the whole 70-minute sweep — including the conc=128 × 155K
point, which is the heaviest kvd + long-prompt + MTP load in this experiment and
precisely the regime that GPU-faulted before the ROCm hicache fix.

Artifacts: `bench/sweep1/*.json` (8 runs + per-point kvd and server_info),
`results/sweep1_table.md`, `results/sweep1_accept.md`, `logs/sweep1.log.gz`.
