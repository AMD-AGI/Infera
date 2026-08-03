# MTP on vs off — what the 2.736 accepted tokens per step actually bought

This run (**MTP ON**) against `agenticbench.glm52.spur.packup_20260731`
(**MTP OFF**) — same cluster, same nodes, same model, same workload YAML, same
bench commit, same 4,007 s window.

> **This is a cross-run comparison, not a controlled A/B.** Beyond MTP the
> images differ (this one carries the three DSA patches + mooncake early-send;
> the reference is the kvaware+kvd-only build) and prefill
> `--mem-fraction-static` differs (0.80 vs 0.88). **`--context-length` is the
> same 262144 in both** — an earlier draft claimed the reference ran at 131,072;
> that was wrong (131,072 is only its script default and was overridden).
> Read the decode-side numbers as attributable and the prefill-side numbers as
> not. The rest of this file says which is which and why.

## The measurement

| metric | MTP OFF | MTP ON | ratio |
|---|---:|---:|---:|
| **TPOT p50 (ms)** | 31.2 | **17.9** | **1.74× faster** |
| TPOT mean (ms) | 30.9 | 18.9 | 1.63× |
| TPOT p90 (ms) | 32.5 | 24.9 | 1.31× |
| generation tok/s (aggregate) | 556.7 | **714.9** | 1.28× |
| completed requests | 2,782 | 2,811 | 1.01× |
| success rate | 0.9531 | **0.9757** | +2.3 pt |
| prompt p50 (tok) | 73,862 | 73,618 | 1.00× |
| TTFT p50 (ms) | 4,446 | 6,659 | **0.67× (worse)** |
| TTFT p90 (ms) | 8,652 | 19,238 | **0.45× (worse)** |

## Why the decode-side win is attributable

**The workload input is identical** — prompt p50 differs by 0.3 % (73,862 vs
73,618 tokens), the same seed drawing the same distribution. So the decode leg
was fed equivalent work in both runs.

**The speedup matches the acceptance length arithmetically.** The engine measured
2.736 accepted tokens per verify step. A perfect implementation would give
2.736×; the realized TPOT p50 ratio is 1.74×, i.e. **64 % of theoretical**. The
gap is the draft model's own forward cost plus verify overhead — EAGLE runs the
draft 3 steps to propose 4 tokens, and that work is not free. A 1.74× realized
against a 2.74 acceptance is the expected shape; a ratio at or above acceptance
would indicate a measurement error, and a ratio near 1.0 would indicate
speculation that is running but not accepted.

**Cross-check against the sibling cluster.** Vultr measured TPOT p50 14.8 ms at
acceptance 2.80 — a 2.11× ratio against its own MTP-off reference. Same regime,
different fabric; the two independently support the mechanism.

**TPOT p90 gains less than p50 (1.31× vs 1.74×)**, which is also expected:
acceptance falls in large batches where the draft model has less context per
sequence, and p90 samples the loaded moments.

## Why the TTFT regression is NOT attributable to MTP

MTP runs on the **decode leg only** (`speculative_algorithm=None` on prefill,
verified in the gate). It cannot slow prefill down. Three other things did:

1. **Concurrency roughly doubled across the window.** Live sessions ran
   22.6 → 22.3 → 34.4 → **44.1** across sustain quarters, and in-flight peaked at
   44 against the reference run's 26–30. TTFT is queueing-dominated in this
   deployment (established in the Phase-1 sweep: TTFT rose 19× from conc 1→128
   while TPOT rose 1.5×). Nearly doubling concurrency is sufficient on its own.
2. **`--mem-fraction-static` 0.88 → 0.80** cut the KV pool 3,260,992 → 2,939,264
   tokens/rank (−10 %). That was the price of not crashing; it also reduces how
   much prefill can hold concurrently.
**Not a cause: input truncation.** Both runs served the same
`--context-length 262144` and the same distribution (prompt p99 226,854 vs
225,782; max 260,013 in both). The reference was **not** measured on a truncated
workload — a claim an earlier draft of this kit made and which is retracted here.
With it removed, the TTFT regression rests on concurrency and KV pool alone, and
those two are proportionally smaller causes than the retracted one implied. The
honest statement is that the regression is **explained in direction but not
fully quantified**; a same-image MTP-off arm would settle it.

## The generation-volume difference, and why it is not double-counting

MTP-ON produced 2,864,792 generation tokens vs 2,230,606 — 1.28× more. This is
**not** the acceptance multiplier leaking into the count: `acc_len` is set to 1.0
in this run's YAML (identity), so `generation_tokens` is the raw count of tokens
actually returned to the client.

The extra volume is real and has a simple cause: faster decode means fewer
requests hit the 240 s client timeout, so longer generations survive to
completion. Mean generation length rose 801.8 → 1,019.1 tokens (1.27×) and errors
fell 96 → 39. **The 2.3-point success-rate improvement is the same effect.** MTP
did not make the model more verbose; it let the verbose requests finish.

## What this does not establish

- **No MTP-off arm was run on this image.** The clean single-variable ablation —
  same image, same ctx, same GMU, MTP toggled — was not performed. It remains the
  single highest-value follow-up, and it is cheap: one 67-minute run.
- **Acceptance was not swept.** `--speculative-num-steps 3 / topk 1 / draft 4`
  was taken from the branch's validated gate config and not varied.
- **The 4.00 tail was not investigated.** 3.0 % of decode batches read exactly
  `accept len: 4.00`. Against a 2.736 mean this reads as small-batch saturation
  rather than a repetition loop — and 0 retractions plus a 97.6 % success rate
  support that — but it was not chased down.
