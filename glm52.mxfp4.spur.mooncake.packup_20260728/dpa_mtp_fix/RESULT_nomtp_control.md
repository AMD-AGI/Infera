> **[INVALIDATED 2026-07-29]** The premise of this document — that the
> degenerate output is an engine bug — was **falsified**. With the chat template
> applied and the model's own sampling (`temperature=1.0, top_p=0.95`), both
> MXFP4 and FP8 produce **0/128 degenerate at concurrency 128**. The symptom was
> caused by testing with raw base-LM completion at a forced `temperature=0`.
>
> Kept for the method and the raw data only. Every rate and every causal claim
> below is withdrawn. See
> `../../glm52.mxfp4.spur.mooncake.packup_20260729_degenerate_output/RETRACTIONS.md`.

# Control experiment: the degeneration is NOT caused by MTP

Date 2026-07-29. Answers "does the ~2% degenerate-output bug survive with
speculative decoding removed entirely?"  **It does — and it gets worse.**

## Setup

| | MTP arm | no-MTP control |
|---|---|---|
| jobs | 9005 prefill / 9006 decode | 11232 prefill / 11233 decode |
| nodes | m2m-244 / m2m-029 | m2m-106 / m2m-208 |
| spec-dec | EAGLE, steps=3, topk=1, draft=4 | **none** |
| source | Bug 1/2/5/6 + Variant B applied | **pristine upstream, zero patches** |
| router | :8031 | :8041 |

Everything else identical: same image, same model, DPA8+EP8, mooncake over
mlx5_0, `--temperature 0`, conc=128 × 512 tokens × 3 rounds, same prompts,
same streaming tracker.

The control was deliberately left **unpatched**. Bug 1/5/6 all crash inside
`deepseek_nextn.py` (the draft model), which a no-MTP deployment never loads,
so those patches are inapplicable; and Bug 2 touches shared code, so applying
it would have contaminated the control. Marker audit before launch confirmed
all four fixes absent and `dsa_indexer.py:965` in its pristine
`if not _is_hip and ...` form.

## Result

| | requests | degenerate | rate | decode ranks hit |
|---|---|---|---|---|
| MTP | 384 | 4 | 1.04 % | 4 of 8 |
| **no-MTP** | 384 | **11** | **2.86 %** | **7 of 8** |

The control reproduces the failure at **2.7× the rate** of the MTP arm.

Sample no-MTP outputs (`temperature=0`, so these are greedy):

```
1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20. ...
2.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3 ...
4.1.3.2.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1 ...
1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1 ...
```

Character-for-character the same failure mode as the MTP arm.

## What this kills

**Speculative decoding is not the cause.** Every draft/verify hypothesis is
dead:

- ❌ *"target and draft lock onto a degenerate loop"* — there is no draft model
  in the control.
- ❌ *"`eagle_utils.py:620`'s `or _is_hip` forces argmax and breaks verify"* —
  that code is never reached in the control.
- ❌ *"high `spec_accept_length` causes it"* — no spec stats exist in the
  control (`spec_accept_length` is `None` for all 11 failures).

The high-acceptance signature reported in `TRACKING_degenerate_output.md`
(97 % of maximum, verify_ct 130 vs 181) is therefore a **consequence**: once
the target model is emitting a periodic loop, the draft model predicts that
loop almost perfectly. It described the symptom, not the disease.

That signature had already started to crack before this control: streaming
round 1 caught a degenerate request at `spec_accept_length = 2.667`, dead
normal. The control settles it.

## What this points at

In **every** degenerate request, on both arms, the **second** token is `'.'`:

```
degenerate, no-MTP (11/11):   c0=<digit>  c1='.'
degenerate, MTP    (3/3):     c0=<digit>  c1='.'

coherent, same digit c0:      '1' -> ':'   '1' -> '\n'   '5' -> ':'
```

The first token is legitimately a digit — the prompt is `"Explain quantum
computing in detail, part 117."`, so 72 of 127 coherent requests also start
with a digit. The divergence is at **token index 1**, which in a PD
deployment is **the first token decode produces** (prefill produces index 0).

So: prefill's output is fine, and the very first decode step goes wrong.
Combined with the control, the fault is in **DPA decode under concurrency**,
upstream of anything spec-decode related.

Rank distribution is flat — 7 of 8 decode ranks produced at least one failure
while all 8 served an equal share. Not a bad rank, not a bad GPU.

## Caveats

- Prefill-side rank is **not** measured. `meta_info["dp_rank"]` comes from
  `output_streamer.py:525` (`self.ps.dp_rank`) = the *decode* scheduler.
  `disagg_prefill_dp_rank` was tried as a pin and **did not take effect**
  (8 requests pinned to rank 3 still spread over DP0/1/2/3/5/6/7 in the
  prefill log), so no prefill-side claim is made here.
- The two arms differ in patch state as well as in MTP. That asymmetry is
  unavoidable (the patches are draft-model-only) and it cuts the safe way:
  the *unpatched* arm is the one that fails *more*, so the patches cannot be
  causing the degeneration.
- 3 further no-MTP rounds were discarded: they came from a stale tracker
  binary in the container and carried `dp_rank=None` throughout. Kept in
  `track_data_stale/` rather than deleted.

## Next

The question is now "why does DPA decode corrupt output under concurrency,
without spec-decode". Single-node mix (no PD) with DPA is the next split: it
separates *DP-attention* from *PD disaggregation*.
