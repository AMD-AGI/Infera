# Exp 3 — both PR #32209 ports applied together

**Ran:** 2026-07-30, AMD spur cluster `crsuse2-m2m`, 2 × MI355X nodes.
**Author:** yihou
**Status:** **PASS** — all acceptance criteria met.

## Goal

Port **both** halves of upstream PR #32209 (`Fix PD decode hang with DP attention and
GLM-5.2 MTP`) onto our baseline and run them together:

- **patch 2b** in #32209's shape — trim/restore around the DSA decode call, rather than our
  approach of matching page-table rows;
- **patch 4** in #32209's shape — the draft graph/eager vote folded into the MLP-sync
  all-gather the scheduler already performs, rather than our extra gloo all-reduce.

On top of our patch 1 + 2a + 3.

**Success criteria:**

1. 4-prompt sequential probe → 4/4 with `spec_accept_length > 1`;
2. conc=32 × 512 → 32/32, no hang, no `KVTransferError`.

## Result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| 4-prompt probe | 4/4 | **4/4**, `acc_len` 1.71–3.00 | ✅ |
| conc=32 × 512 | 32/32 | **32/32**, `acc_len` mean 2.74 | ✅ |
| conc=64 × 512 | — (headroom we added) | **64/64**, `acc_len` mean 2.84 | ✅ |
| hangs / `KVTransferError` | 0 | **0** | ✅ |
| Traceback (final run) | 0 | **0** | ✅ |

All eight DP ranks served traffic. Raw per-request data in `results/`.

The deadlock this whole effort targets does **not** occur in this configuration.

## Scope — what a combined arm can and cannot tell you

This arm changed **two** things at once, so its pass is a statement about the
**combination**, not about either port individually. To attribute it, the arm was split
into two single-variable arms the same day:

| arm | patch 2b | patch 4 | conc=32 | draft graph usage |
|---|---|---|---|---|
| **e3** (this arm) | #32209 | #32209 | **32/32** ✅ | not measured |
| **e3a** | #32209 | **ours** | **0/32** ❌ | — |
| **e3b** | **ours** | #32209 | **32/32** ✅ | **97.1 %** |

What the split establishes:

- **#32209's patch 4 stands on its own** — and, with the draft CUDA graph *measured* in use
  at 97.1 %, it is a genuine fix rather than an eager-fallback workaround. See the sibling
  kit `..._exp3b_patch4_32209`.
- **#32209's patch 2b does not stand on its own** — paired with our patch 4 it crashes at
  conc=32 with `output tensor size must be equal to world_size times input tensor size` in
  `dp_gather_replicate`. Under investigation.
- **In this combined arm, patch 4 was masking that.** As first written, the patch-4 port
  drove draft-graph usage to 0 % (measured in e3b: 8 ranks, 200/200 calls refused), so
  every rank ran the eager path, which recomputes the DP padding mode per step and avoids
  the inconsistency the 2b port introduces. Correcting patch 4 so the graph is actually
  used makes the 2b defect reachable again.

So: this arm's PASS is real and reproducible. It is simply not evidence that *both* ports
are individually correct — and the split shows one of them is not.

## Known gap in this arm

Draft-graph usage was **not instrumented here**. That is the single measurement that would
have shown the masking immediately, and it is why the sibling arm adds
`instr_graph_usage.py`. Charter criterion 5 for this bug class asks that the graph path be
*provably taken* (marker count > 0), not merely that nothing hangs — because Variant B
(forcing the draft path eager) also passes criteria 1–4 while disabling the feature under
test.

## Other limits

- **conc=128 was not run** (criterion was 32; 64 added as headroom). Do not cite this kit
  as a conc=128 result.
- **No negative control** on this node pair — the patches were not reverted here to show
  the hang returns.
- An **earlier run of this arm crashed** at conc=32 with the same all-gather mismatch; the
  patch-2b trim was then changed from `cache_seqlens_int32.shape[0]` (per-request) to
  `page_table_1.shape[0]` (per-token, correct under MTP) and the arm passed. That change is
  plausibly necessary but is **not** what made the arm pass — arm e3a runs the corrected
  code and still crashes. See `notes.md`.

## Folder map

- `REPRODUCE.md` — how to re-run this arm
- `environment.md` — hardware, fabric, image, commit
- `notes.md` — the attribution story, the retraction, what to reuse
- `patches/` — both #32209-style ports + the strip helper
- `scripts/` — the scripts that ran, incl. `instr_e3.py` (all-gather-site probe)
- `results/` — raw per-request jsonl
- `logs/` — full prefill / decode / router logs
