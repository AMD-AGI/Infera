# Retractions

Every claim withdrawn on 2026-07-29, with the reason and the evidence that
killed it. Listed so nobody rebuilds on a foundation that has been removed.

---

## R1 — "~2 % of responses degenerate under concurrency" — the whole premise

**Claimed:** an engine defect corrupts ~1–3 % of outputs at concurrency 128.

**Withdrawn.** With the chat template applied and the model's own sampling
(`temperature=1.0, top_p=0.95`), both MXFP4 and FP8 return **0/128 degenerate**
at concurrency 128 (`results/RESULT1`, `results/RESULT2`).

**Why it looked real:** every test posted raw `text` to `/generate` (base-LM
completion, no `[gMASK]<sop>`, no `<|system|>` turn) and forced
`temperature=0`, overriding the `generation_config.json` value of 1.0. Greedy
decoding falling into repetition is documented LM behaviour, not an engine
fault. The harness manufactured the symptom.

**Directly affected documents** (in the 2026-07-28 kit):
`TRACKING_degenerate_output.md`, `RESULT_nomtp_control.md`,
`RESULT_pd_vs_mix_control.md` — all rates and all causal attributions.

---

## R2 — "MTP is not the cause: 1.04 % with MTP vs 2.86 % without"

**Withdrawn — the comparison was two-variable.**
`--disable-custom-all-reduce` lived **inside the MTP block** of
`pd_leg_spur.sh`, so `MTP=0` silently re-enabled the aiter custom all-reduce, a
kernel known to deadlock on gfx950. Confirmed by grepping the live
`server_args`:

```
MTP arm    : disable_custom_all_reduce=True
no-MTP arm : disable_custom_all_reduce=False
```

The 2.7× ratio measured MTP *and* custom-all-reduce together. Script fixed
(flag hoisted out of the MTP block; `mix_leg.sh` never had it at all and now
does). Moot anyway under R1.

---

## R3 — "The degenerate requests have a spec-decode signature (97 % of max accept length)"

**Withdrawn.** Reported from 6 samples where degenerate requests showed
`spec_accept_length ≈ 3.9/4.0` vs a normal 2.7.

Broken three ways: (a) a later degenerate request had a perfectly normal 2.667;
(b) coherent and degenerate ranges **overlap** (coherent max 3.85, degenerate
min 3.71), so it was never a criterion; (c) the no-MTP control has no draft
model at all yet still failed.

At most it was a consequence — once the model emits a periodic loop, a draft
model predicts that loop almost perfectly.

---

## R4 — "`eagle_utils.py:620`'s `or _is_hip` silently discards temperature"

**Withdrawn as stated.** The line exists, but it is in the **spec-decode verify**
path, which a no-MTP server never executes. It was presented as a general
statement about sampling on HIP; it is not.

Measured on the live FP8 server, 4 samples each:

| setting | first 50 chars |
|---|---|
| `t=1.0, p=0.95` | `' Provide examples\nQuantum Error Correction: Shor'` |
| `t=2.0` | `' Class Hours strong traveling熠.notes miserable_'` |

`temperature`, `top_p`, `top_k` all take effect.

---

## R5 — "Same prompt at `temperature=0` gives 3 different answers ⇒ engine bug"

**Withdrawn.** True but expected. sglang ships:

```
enable_deterministic_inference:
    "Enable deterministic inference mode with batch invariant ops."
    default: False
```

Its existence documents that batch composition changes reduction order and can
flip an `argmax`. All servers ran with it `False`. Non-determinism was the
default configuration, not a defect.

---

## R6 — "The divergence is always at token index 1, i.e. decode's first step"

**Withdrawn.** Based on 21 requests whose second token was `.`. Two problems:

- The degeneracy predicate scanned the whole string, so outputs that were good
  prose then collapsed scored **coherent** and were excluded from the sample.
  Re-checking stored tails: **11 of 501 (2.2 %) "coherent" outputs were looping
  at the end.** Onsets, once measured properly, range 0 → 356 chars.
- The framing was PD-specific ("token 0 is prefill's") but the single-node mix
  produces the identical shape with no prefill/decode split.

---

## R7 — "Try `--tp-size 1` to isolate TP"

**Withdrawn before running.** MXFP4 weights are **408 GB**; one MI355X has
288 GiB. TP=1 cannot load the model. Proposed without doing the arithmetic.

---

## R8 — "Variant A (`--speculative-attention-mode decode`) was tested"

**Withdrawn twice, then confirmed negative.** The first run never applied the
flag (`server_args` showed `'prefill'`) and was reported as running. A later
run did apply it and hung in warmup for 22 min with no traceback. No py-spy was
captured, so the localisation rests on behavioural similarity alone.

---

## What still stands

The **crash and deadlock** work from `glm52.mxfp4.spur.mooncake.packup_20260728/`
is untouched by all of the above. Those were hard failures with stack traces,
independent of sampling configuration:

| bug | failure | status |
|---|---|---|
| Bug 1 | `Expected lengths.size(0) == B` — HIP/aiter used DP-padded row count | fixed, verified |
| Bug 2 | PD+MTP deadlock on first routed request — D2H sync on a rank-divergent branch | fixed, verified |
| Bug 5 | `assert page_table.shape[0] == topk_indices.shape[0]` | fixed, verified |
| Bug 6 | Bug-1 slice skipped DP-idle ranks (`0 < q_offset`) | fixed, verified |

PD + DPA + MTP went from deadlocking on the first request to **640/640 HTTP 200**
at concurrency 128 across 5 rounds, 0 exceptions, 0 `KVTransferError`, 8
schedulers alive.

**Two open items unaffected by these retractions:**

1. **Bug 2b** — the draft CUDA graph decision diverges per DP rank. Worked
   around by disabling the draft graph (Variant B); no proper fix. The
   `eagle_worker_v2.py` change was deliberately **excluded** from the PR.
2. **A seventh padded-vs-real row crash** — `Expected lengths.size(0) == B` at
   `dsa_indexer.py:996`, observed **once in 500+ requests on a machine that
   already carries the Bug 1 and Bug 6 fixes** (markers verified present in the
   source at crash time). Rare, concurrency-dependent, unexplained.
   `SGLANG_DEBUG_DSA_ROWS=1` (already in `dsa_indexer.py:63`) logs the exact
   shapes at that site and should be enabled on the next long run.
