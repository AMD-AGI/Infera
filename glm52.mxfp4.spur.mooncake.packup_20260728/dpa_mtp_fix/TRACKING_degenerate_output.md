> **[INVALIDATED 2026-07-29]** The premise of this document — that the
> degenerate output is an engine bug — was **falsified**. With the chat template
> applied and the model's own sampling (`temperature=1.0, top_p=0.95`), both
> MXFP4 and FP8 produce **0/128 degenerate at concurrency 128**. The symptom was
> caused by testing with raw base-LM completion at a forced `temperature=0`.
>
> Kept for the method and the raw data only. Every rate and every causal claim
> below is withdrawn. See
> `../../glm52.mxfp4.spur.mooncake.packup_20260729_degenerate_output/RETRACTIONS.md`.

# Per-request tracking of the ~2% degenerate-output bug

Date: 2026-07-29. Service: job 9005 (prefill) + 9006 (decode, EAGLE MTP 3/1/4,
Variant B = draft CUDA graph disabled), all of Bug 1/2/5/6 applied.

## Method

`sglang` accepts a **client-supplied `rid`** (`GenerateReqInput.rid`, only
auto-generated when `None`) and echoes it back as `meta_info["id"]`. That makes
every request individually addressable across the client record, the prefill
log and the decode log.

`meta_info` turned out to carry far more than the id — verified live:

```
id, dp_rank, prompt_tokens, completion_tokens, cached_tokens, finish_reason,
e2e_latency, num_retractions, weight_version, response_sent_to_client_ts,
spec_accept_length, spec_accept_rate, spec_verify_ct,
spec_num_correct_drafts, spec_num_proposed_drafts,
spec_accept_histogram, spec_correct_drafts_histogram
```

So the **serving DP rank and the full spec-decode telemetry are available
per request**, with no instrumentation and no restart.

Tool: `patches/scripts_20260729/track_probe.py`. It asserts `meta_info["id"]
== rid` on every request and shouts if the echo ever fails — a correlation
built on mismatched ids would be worthless.

Data: `track_data/mtp-r{1,2,3}-*.jsonl` (3 rounds × conc=128 × 512 tok,
temperature=0, greedy).

## Result: 508 requests, 7 degenerate (1.38%)

| round | n | http 200 | degenerate |
|---|---|---|---|
| r1 | 128 | 128 | 1 |
| r2 | 128 | 128 | 4 |
| r3 | 128 | 128 | 1 |
| r4 | 128 | 124 (4 conn-fail: the leg died mid-round, see Incident) | 1 |

### The degenerate requests have a distinct spec-decode signature

| metric | coherent (n=501) | degenerate (n=7) |
|---|---|---|
| `spec_accept_length` mean | 2.72 | **3.89** |
| — as % of the 4.0 maximum | 68.1 % | **97.2 %** |
| `spec_verify_ct` median | 181 | **130** |
| `e2e_latency` median | 9.65 s | **7.04 s** |
| `num_retractions` | 0 | 0 |
| `finish_reason` | length | length |
| `completion_tokens` | 512 | 512 |

`speculative_num_draft_tokens = 4`, so 4.0 is the ceiling. The degenerate
requests sit at **97 % of the theoretical maximum acceptance** — essentially
nothing is being rejected.

The degenerate requests are the ones where the **draft model was accepted
almost every time**: accept rate 0.91–0.99 against a normal ~0.69, i.e. ~3.9
of 4 draft tokens accepted per step. Because so little is rejected they need
only ~130 verify steps for 512 tokens instead of ~181, which is exactly why
they also come back ~2.6 s *faster*.

So the failure is not "a request got stuck or was retried". It is
**"target and draft agreed on a degenerate loop and the loop ran unchecked"**.

### Separation is strong but NOT total — do not build a detector on it

```
coherent   MAX accept_len = 3.8496
degenerate MIN accept_len = 3.7101      -> ranges OVERLAP
```

20 of 378 coherent requests also exceed 3.5, the top one at 3.850 with
perfectly readable text. High acceptance alone therefore does not imply
corruption; it is a strong correlate, not a criterion.

### Output shape

`n_unique_chars` is sharply bimodal — coherent min 21, median 45; degenerate
`[2, 2, 2, 4, 7, 11]`. Typical degenerate body:

```
1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1. ...      (uniq=2)
2.1.1.2.1.3.4.5.6.7.8.9.1.0.1.1.1.2. ...      (uniq=11)
96.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1 ...      (uniq=4)
```

Several start with a plausible fragment (`5\n8.1.3.3.2...`, `96.1.1...`) and
then collapse — the request began fine and fell into the loop, rather than
being born corrupt.

### It is not a bad rank

Degenerate requests spread over dp_rank `{6:2, 0:1, 1:1, 2:1, 3:1, 7:1}` while
every rank served an equal share (16 per round). Six of the eight ranks have
produced at least one. `cached_tokens=0` and `prompt_tokens=11` for every
degenerate request — same as its coherent neighbours, so this is not a
prefix-cache artifact.

## Open, and what it implies

The tracker tells us *what* the failure looks like but not yet *why* the two
models lock together. The two candidates that remain:

1. **The KV/state the draft model reads is subtly wrong under concurrency**,
   so both models are conditioned on the same corrupted context and therefore
   agree. This would make high acceptance a *symptom*.
2. **Verify is not actually rejecting** — cf. `eagle_utils.py:620`, where
   `or _is_hip` forces `torch.argmax` and silently discards the sampling
   params. Under greedy this should still be *correct*, just deterministic,
   so it does not by itself explain corruption — but it does mean the verify
   path on HIP is not the same code CUDA runs.

The decisive experiment is the **no-MTP control** (jobs 11232/11233, clean
upstream, `MTP=0`): if degeneration reproduces without a draft model at all,
the spec-decode signature above is a consequence and the cause is upstream of
it; if it does not reproduce, the cause is in draft/verify.

## Incident during collection

Round 4 returned `503 ×128`: the decode leg had died at 14:54:48 with
`RuntimeError: Expected lengths.size(0) == B` on DP7 — the Bug-1 assertion,
on a machine that *has* the Bug 1 and Bug 6 fixes, reached through
`deepseek_nextn.py:271` (draft) → `dsa_indexer.py:996`. Both fixes were
confirmed live in the source at crash time.

This is a **seventh, distinct occurrence of the padded-vs-real row family**
and is not yet explained: the aiter branch now slices to `q_offset`
unconditionally, so `logits` should have `q_offset` rows. It fired once in
384+ requests, i.e. it is rare and concurrency-dependent.

`SGLANG_DEBUG_DSA_ROWS=1` already exists in `dsa_indexer.py:63` and logs
`q_fp8`/`q_offset`/`lengths`/`mqa_q` shapes at the exact site — that is the
instrument to enable on the next boot.
