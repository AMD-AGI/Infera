# Exp 3b — patch 4 in upstream PR #32209's shape, with the draft graph PROVEN in use

**Ran:** 2026-07-30, AMD spur cluster `crsuse2-m2m`, 2 × MI355X nodes.
**Author:** yihou
**Status:** **PASS** — and unlike every previous green run on this bug, the draft CUDA
graph is *measured* to be in use, not merely assumed.

## Goal

Our patch 4 fixes the PD + DPA + MTP deadlock by voting the draft graph/eager decision
over the TP group with an **extra** 1-element gloo all-reduce inside `draft()`.
Upstream PR **#32209** (`Fix PD decode hang with DP attention and GLM-5.2 MTP`, open) fixes
the same defect with the same strategy but a **better placement**: one more int64 slot in
the MLP-sync all-gather the scheduler *already* performs every iteration — **zero extra
collectives**.

This arm ports that placement and measures two things:

1. does it still fix the deadlock (functional);
2. **is the draft graph actually replayed** — charter criterion 5.

Point 2 is the whole reason this arm exists. Variant B (forcing the draft path eager)
already passes every functional test, so *a green run proves nothing on its own*. An arm
that silently never uses the draft graph is indistinguishable from a working fix unless
you count.

**Configuration:** our patch 1 + 2a + **our** 2b + patch 3 + **#32209-style patch 4**,
plus `scripts/instr_graph_usage.py`. Our patch 4 (`_needs_eager_local`) and the #32209-style
2b are asserted **absent** from the bytecode, so exactly one draft-graph mechanism is live.

**Success criteria:**

1. 4-prompt sequential probe → 4/4 with `spec_accept_length > 1`;
2. conc=32 × 512 → 32/32, no hang, no `KVTransferError`;
3. **the draft graph is provably taken** (usage > 0, measured).

## Result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| 4-prompt probe | 4/4 | **4/4**, `acc_len` 2.18–3.00 | ✅ |
| conc=32 × 512 | 32/32 | **32/32** | ✅ |
| conc=32 × 512, repeat | — | **32/32** | ✅ |
| conc=64 × 512 | — | **64/64** | ✅ |
| Traceback in server logs | 0 | **0** | ✅ |
| **draft graph usage** | **> 0** | **97.1 %** (777/800 calls, every rank) | ✅ |

| Run | ok | full 512 tok | acc_len mean | min | max |
|---|---|---|---|---|---|
| `stress_c32.jsonl` | 32/32 | 29/32 | 2.86 | 2.23 | 3.91 |
| `stress_c32_r2.jsonl` | 32/32 | 27/32 | 2.79 | 2.09 | 3.91 |
| `stress_c64.jsonl` | 64/64 | 61/64 | 2.75 | 2.16 | 3.94 |

### The measurement that matters

From `logs/decode.log` (`GLM52_GUSE`), after 800 `can_run_graph` calls per rank:

```
rank=0 calls=800 graph=777 (97.1%) refused_bs=0 refused_dp=0 refused_draftvote=23
rank=1 calls=800 graph=777 (97.1%) refused_bs=0 refused_dp=0 refused_draftvote=23
...
rank=7 calls=800 graph=777 (97.1%) refused_bs=0 refused_dp=0 refused_draftvote=23
```

**Identical on all 8 ranks** — that is the fix working. Now compare the *rank-local*
decision, before the vote (`GLM52_GUSE_WHY`, 600 calls each):

```
rank=0 future_seed_missing=9  (1.5%)   rank=4 future_seed_missing=10 (1.7%)
rank=1 future_seed_missing=9  (1.5%)   rank=5 future_seed_missing=9  (1.5%)
rank=2 future_seed_missing=10 (1.7%)   rank=6 future_seed_missing=10 (1.7%)
rank=3 future_seed_missing=11 (1.8%)   rank=7 future_seed_missing=8  (1.3%)
```

The local answer **diverges across ranks** (8, 9, 9, 9, 10, 10, 10, 11) while the acted-on
answer is **uniform** (777/800 everywhere). So the bug is still latent — the ranks really
do disagree — and the all-gather is provably doing the work of reconciling them. This is
the same signature our patch 4 produced (local diverged 38×, voted 0×, graph 98.4 %),
reproduced through #32209's placement at **no extra collective**.

## The bug this arm caught — and why the split was necessary

The first run of this arm passed 32/32 with `acc_len` 2.66 and looked completely healthy.
It was **0.0 % graph usage** — all 8 ranks, 200/200 calls refused, 100 % attributed to one
branch. The port had silently degenerated into Variant B.

**Cause: an error in my port, not in upstream's design.** The script asserted that
`future_dsa_topk_indices_available` "does not exist in this baseline (verified by grep)"
and fell back to *requiring eager* whenever `future_indices` was set. That claim was
**false** — the attribute is at `eagle_info.py:179` and `spec_info.py:261`, and
`eagle_disaggregation.py:71` sets it. Under overlap scheduling `future_indices` is set on
**every** decode iteration, so the fallback refused the graph 100 % of the time.

The counter also showed `seed_none = 0`: the real guard term never once fired, i.e. the
graph was available the entire time. Reading the flag the way upstream does took usage
from 0.0 % → 97.1 %.

**This is why the merged Exp-3 arm was worth splitting.** That arm passed (a real result,
sibling kit `..._exp3_merged`), but within it this patch-4 eager fallback was also masking a
genuine patch-2b defect — so its pass could not be attributed to either port on its own.

## What this arm does NOT establish

- It does **not** validate the #32209-style **patch 2b**. This arm runs *our* 2b. The
  #32209 2b port is broken — proven separately in arm e3a, which crashes at conc=32 with
  `output tensor size must be equal to world_size times input tensor size`.
- **No negative control on this node pair.** We did not revert patch 4 here to show the
  hang returns. The bug-2b kit has that control for our patch 4; this arm inherits the
  claim rather than re-demonstrating it.
- The remaining **2.9 %** eager iterations were not investigated. They correspond to the
  `future_seed_missing` cases — the seed genuinely not having arrived — which is the guard
  behaving correctly, but we did not verify that individually.
- **conc=128 was not run** (criterion was 32; 64 added as headroom). Do not cite this as a
  conc=128 result.
- `patches/patch4_32209_style.py` here is the **corrected** version. The 0.0 % run came
  from an earlier revision of the same file; that revision is not shipped, but its
  measurement is quoted above and the correction is documented inline in the script.

## Folder map

- `REPRODUCE.md` — cold-start reproduction including the graph-usage check
- `environment.md` — exact hardware, fabric, image, commit
- `notes.md` — the port, the 0 %-usage bug, why counting was mandatory
- `patches/` — the four patches applied (patch 4 in #32209 shape)
- `scripts/` — every script that ran, incl. `instr_graph_usage.py`
- `results/` — raw per-request jsonl
- `logs/` — full prefill / decode / router logs (the `GLM52_GUSE` lines live in `decode.log`)
