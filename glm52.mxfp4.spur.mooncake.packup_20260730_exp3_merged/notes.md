# Notes — Exp 3 (merged)

## 1. What happened, in order

1. **Arm passed.** 4/4 probe, 32/32 conc=32, 64/64 conc=64, zero tracebacks, `acc_len`
   2.74–2.84. Against the acceptance criteria set for the day: a PASS. That result is real
   and reproducible.
2. **Graph usage was not measured** in this arm — the one gap, see §4.
3. **Split into one-variable arms** to attribute the pass to a specific port
   (e3a = #32209's 2b + our 4; e3b = our 2b + #32209's 4). This was the plan from the
   start, not a reaction to a failure here.
4. **e3b measured 0.0 % draft-graph usage** — 8 ranks, 200/200 calls refused. The patch-4
   port, as first written, was falling back to eager on every iteration.
5. **e3a crashed at conc=32** with `output tensor size must be equal to world_size times
   input tensor size`. #32209's patch-2b port does not hold up on its own.
6. **Correcting patch 4 took usage 0.0 % → 97.1 %**, and e3b passed *with the graph
   provably in use* (sibling kit `..._exp3b_patch4_32209`).

**How to read this together.** This arm works. But it works as a *combination*: within it,
patch 4's eager fallback was also masking patch 2b's defect, because the eager path
recomputes the DP padding mode per step and avoids the inconsistency that port introduces.
Correct patch 4 so the graph is genuinely used, and the 2b defect becomes reachable — which
is exactly what e3a shows.

So the combined arm passing and one of its halves being broken are **both true**, and not in
conflict. What the combined arm cannot do is tell you *which* half earned the pass; that is
what the split is for, and it is why the split was worth running rather than stopping here.

## 2. A crash "fix" I have to retract

Earlier the same day this arm hit the **same** `all_gather` size mismatch at conc=32. The
patch-2b trim was using `metadata.cache_seqlens_int32.shape[0]` — a **per-request** count —
while under MTP the query rows are **per-token** (`init_forward_metadata` expands the page
table by `repeat_interleave(..., speculative_num_draft_tokens)` for TARGET_VERIFY /
DRAFT_EXTEND_V2). The trim therefore cut `bs * num_draft_tokens` rows down to `bs`.

Changing it to `metadata.page_table_1.shape[0]` made this arm pass, and that was reported
as the fix.

**That report was wrong.** Arm e3a runs exactly the corrected code and crashes identically.
What made this arm pass was patch 4 going all-eager. The row-count change is plausibly
*necessary* — `page_table_1` is per-token and is the quantity the assert compares against —
but it is **not sufficient**, and nothing here validates it.

`patches/patch2b_32209_style.py` in this kit is therefore **known-broken**. Do not reuse it
without re-deriving the trim.

## 3. What to reuse from this kit, and what to discard

| artifact | status |
|---|---|
| `patches/patch4_32209_style.py` | **superseded, and NOT the code this arm ran** — see the warning below |
| `patches/patch2b_32209_style.py` | **known-broken** — crashes in isolation (e3a) |
| `patches/*.diff` (our 1 / 2a+2b / 3) | fine — these are the verified kit diffs |
| `scripts/instr_e3.py` | useful — probes the all-gather site itself (local/global row counts, padding mode, forward mode); written to separate "trim leaks" from "ranks disagree on `DpPaddingMode`" |
| `results/*.jsonl` | real data — a valid PASS for the *combined* configuration; not evidence about either port alone |
| `logs/` | contains both the earlier crash traces and the passing run |

> **Warning — `patches/patch4_32209_style.py` is not a faithful record of this arm.**
> The file is a single evolving script in the shared workspace, and it was **corrected
> after this arm ran** (see §1 step 6). The copy here is therefore the *corrected*
> version, which reads `future_dsa_topk_indices_available` and yields ~97 % graph usage —
> whereas this arm actually ran the earlier revision that forced eager and yielded 0 %.
> Applying this file will **not** reproduce this arm's behaviour; it will reproduce e3b's.
> The earlier logic is quoted verbatim in the e3b kit's `notes.md` §2, and the
> correction is documented inline in the script itself.
>
> The same caveat does **not** apply to `patch2b_32209_style.py`: that file is unchanged
> since this arm ran it, and remains known-broken.

## 4. The methodological lesson

This arm satisfied every stated criterion and would have been reported as a success. Two
things exposed it:

**Adding a measurement that was not in the criteria.** Counting whether the draft CUDA
graph was actually replayed. Charter criterion 5 exists precisely for this:

> the graph path is **provably taken** (marker count > 0) — not merely "no hang".

Variant B passes criteria 1–4 *by disabling the graph*. So "no hang" cannot distinguish a
fix from the workaround, and any arm without a graph-usage count is uninterpretable.

**Splitting a two-variable arm.** Two changes shipped together can pass by cancelling.
`e3a` and `e3b` each changed one thing, and each immediately exposed a defect the merged
arm had hidden.

Both lessons are instances of the same standing rule for this bug class: **a single passing
run proves nothing about a race**, and a fix must be localized by measurement, not by
"it stopped failing".

## 5. Traps on this stack (unchanged, listed for self-containment)

- **Stale `.pyc` silently reverts a patch** — verify in bytecode, with identifiers not
  comments (the compiler discards comments).
- **Logs contain binary bytes** — `grep -c` returns 0 and reads as "clean". Use `strings`
  or `grep -a`.
- **503 timing tells you what failed.** ~0.4 s = stale router circuit breaker; ~12 s = a
  real backend failure. This arm produced both on different days, and confusing them cost
  a restart cycle.
- **`/home` was 100 % full** — JIT caches were moved to `/shared_nfs`, and the image tar
  named in earlier kits no longer exists.
