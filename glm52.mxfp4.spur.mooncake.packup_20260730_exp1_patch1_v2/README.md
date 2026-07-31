# Exp 1 — patch 1 reworked in upstream PR #32762's shape, full patch set live

**Ran:** 2026-07-30 (single day), AMD spur cluster `crsuse2-m2m`, 2 × MI355X nodes.
**Author:** yihou
**Status:** **PASS** — all four acceptance criteria met, no failures observed.

## Goal

Our patch 1 (the HIP/aiter padded-row fix in `dsa_indexer.py`) is correct — it passed
2540/2540 in the `packup_20260729_bug2b_draft_graph` kit — but it is written in a shape
that **hides a shape drift instead of failing on it**: its restore step is gated on
`topk_result.shape[0] == q_offset`, so if the paged-MQA kernel ever returned a different
row count, the DP padding would silently not be restored and the caller would get a short
tensor. That is a wrong answer rather than a crash.

Upstream PR **#32762** (`[NPU] Fix DSA eager padding mismatch in PD MTP warm-up`, open as
of 2026-07-30) fixes the same bug class on NPU in a better shape: one boolean computed up
front gates both the trim and the restore, and the post-kernel row count is **asserted**
before the padding is re-attached.

This experiment reworks our patch 1 into that shape and re-runs the **full patch set**
(1v2 + 2a + 2b + 3 + 4) to confirm the rework is behaviour-preserving — i.e. that making
the guard loud did not turn a passing configuration into a failing one.

**Success criteria** (set by the user for all three arms of the 2026-07-30 three-way run):

1. 4-prompt sequential probe → 4/4, with `spec_accept_length > 1` (proves MTP is
   genuinely active and not silently bypassed);
2. conc=32 × 512 tokens → 32/32, no hang, no `KVTransferError`.

## Result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| 4-prompt probe | 4/4 | **4/4**, `acc_len` 2.18–3.00 | ✅ |
| conc=32 × 512 | 32/32 | **32/32** (run 1) | ✅ |
| conc=32 × 512, repeat | — (stability check we added) | **32/32** (run 2) | ✅ |
| conc=64 × 512 | — (headroom check we added) | **64/64** | ✅ |
| hangs / `KVTransferError` | 0 | **0** | ✅ |
| Traceback in either server log | 0 | **0** | ✅ |

Speculative-decode acceptance length, from the raw jsonl in `results/`:

| Run | ok | full 512 tok | acc_len mean | acc_len min | acc_len max |
|---|---|---|---|---|---|
| `stress_c32.jsonl` | 32/32 | 31/32 | 2.85 | 2.06 | 3.91 |
| `stress_c32_r2.jsonl` | 32/32 | 31/32 | 2.85 | 2.42 | 3.56 |
| `stress_c64.jsonl` | 64/64 | 57/64 | 2.80 | 2.23 | 3.94 |

All eight DP ranks (`dp 0..7`) served traffic in every run — the load did reach the whole
DP group, so a pass is not an artifact of one rank doing all the work.

> "full 512 tok" below the request count is **not** a failure: `max_new_tokens=512` is a
> cap, and greedy decoding hits EOS earlier on some prompts. Every request returned
> HTTP 200 with a complete response. The `ok` column is the criterion.

## What this arm does NOT establish

- It does **not** show the rework is *necessary*. v1 passed too. The assert has not yet
  been observed to fire, so its value here is preventative, not demonstrated.
- The `SGLANG_DEBUG_DSA_ROWS` comparison between `q_offset` and `num_token_non_padded_cpu`
  (the source #32762 uses) was **left disabled** in this run, so this kit contains no data
  on whether those two quantities agree. That question is still open — see `notes.md`.
- conc=128, the bar used in the bug-2b kit, was **not run here**. The user's criterion for
  this three-way comparison was conc=32; we added 64 as headroom. Do not read this kit as
  a conc=128 result.

## Folder map

- `REPRODUCE.md` — cold-start reproduction, ordered and copy-pasteable
- `environment.md` — exact hardware, fabric, image and commit the numbers came from
- `notes.md` — why patch 1 was reworked, what the rework changes, gotchas hit this run
- `patches/` — the five patches applied, with the v2 rewrite script
- `scripts/` — every script that ran, copied verbatim
- `results/` — raw per-request jsonl (the evidence behind every number above)
- `logs/` — full prefill / decode / router logs, uncompressed
