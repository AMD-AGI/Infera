# Bug 2b root cause — the CUDA-graph/eager switch in `draft()` is rank-divergent

Measured 2026-07-29 12:1x–12:2x on job 9006 (decode leg), GLM-5.2-MXFP4,
TP8 + DP-attention(dp_size=8) + EP8 + EAGLE MTP(steps=3, topk=1), gfx950.

This is the first time the question **"why do the DP ranks diverge?"** has been
*measured* rather than reasoned about. Three previous fixes (hoist_sync, Bug 3
broadcast, Bug 4 uniform event) all addressed *how to resynchronize after
divergence* and all failed, because none of them touched the cause.

## The defect

`python/sglang/srt/speculative/eagle_worker_v2.py:517`

```python
if (
    can_cuda_graph
    and not forward_batch.forward_mode.is_idle()   # <-- rank-divergent term
    and self.seed_dsa_topk_from_draft_extend       # True for GLM-5.2
    and draft_input.dsa_topk_indices is None
):
    can_cuda_graph = False
```

Under DP-attention each rank independently has work or is idle, so
`is_idle()` **differs across ranks in the same iteration by construction**.
This guard therefore selects the CUDA-graph path on some ranks and the eager
path on others, *within one collective step*.

`seed_dsa_topk_from_draft_extend` is True for GLM-5.2 because the model config
sets `index_share_for_mtp_iteration` and `index_topk=2048` (constructor at
eagle_worker_v2.py:274), so the guard is live on this model.

## CORRECTION (added after review) — read this before the section below

The section below states that the graph path issues **0** all-gathers while the
eager path issues 2, and concludes the collective *counts* diverge. **That
conclusion overreaches and is retracted.**

A replayed CUDA graph still performs the collectives captured into it; it simply
executes no Python, so the probe hook inside `LogitsProcessor._get_logits` never
fires. "0 all-gathers on the graph path" is a **probe blind spot, not a
measurement**. The Python-visible counts therefore cannot be used to argue that
the number of collectives differs.

What *is* measured and still stands:

* the graph/eager decision at `eagle_worker_v2.py:517` is **rank-divergent**
  (dp7 6/2 vs 8/1 on every other rank) — this is a real defect regardless of how
  the deadlock is explained;
* the Python-visible all-gather counts differ across ranks, and the rank with the
  odd graph/eager split is the rank with the odd count;
* **the hang disappears with `--disable-cuda-graph`** (independent control pair,
  6/6 requests HTTP 200 including two 512-token runs, spec-dec active) — so the
  graph path is implicated as a necessary condition.

What is **not** established: the actual mechanism. Whether it is a count
mismatch, an ordering mismatch, or a capture-time state difference has not been
measured. Two alternatives remain open: `--disable-cuda-graph` is a blunt
instrument that disables target-decode, draft, and draft-extend graphs at once
(the culprit may be any of them), and removing graphs perturbs timing enough
that it could be masking a race rather than fixing it.

The decisive experiment is the naive fix **with graphs still enabled**: make the
guard rank-uniform and see whether the hang survives. That experiment has not
yet produced a valid result — the first attempt ran stale `.pyc` bytecode and
tested nothing.

## Why a per-rank graph/eager choice deadlocks

The draft stage's LM-head vocabulary all-gather is a collective over the **full
8-way TP group**. RCCL matches collectives by **issue order**, not identity, so
every rank must issue the same number of them in the same sequence.

Instrumented counts (probe: `AG_ENTER` inside `LogitsProcessor._get_logits`,
`DRAFT_GRAPH` at the guard). Same rank, same `mode=DECODE`, same `bs=1`:

| dp7 iteration | `cg_after` | path | all-gathers in `draft` |
|---|---|---|---|
| it=1 | `False` | eager | **2** (`agc` 315 → 317) |
| it=3 | `True`  | graph | **0** visible (`agc` 319 → 319) |

The eager path runs `draft_forward`'s `for i in range(speculative_num_steps)`
loop in Python and issues one all-gather per inner step. The graph path replays
a captured graph, which executes no Python — so the probe sees zero, and more
importantly the *host-side issue sequence* differs between the two paths.

Iteration 8, across ranks, shows the divergence directly:

| rank | mode | `topk_none` | `cg_after` | all-gathers that iteration |
|---|---|---|---|---|
| dp0 | IDLE   | True | **True**  | 1 |
| dp7 | DECODE | True | **False** | 3 |

`dp7` takes eager (guard fires: not idle, seed True, topk None) while `dp0`
takes the graph (guard short-circuits on `is_idle()`). From that point the two
ranks' collective streams are permanently misaligned.

## Aggregate evidence

Per-rank totals at deadlock (`grep -c AG_ENTER`, `grep -c DRAFT_GRAPH ...`):

```
dp0 ag=977  draft_cg_true=8  draft_cg_false=1
dp1 ag=979  draft_cg_true=8  draft_cg_false=1
dp2 ag=977  draft_cg_true=8  draft_cg_false=1
dp3 ag=977  draft_cg_true=8  draft_cg_false=1
dp4 ag=977  draft_cg_true=8  draft_cg_false=1
dp5 ag=977  draft_cg_true=8  draft_cg_false=1
dp6 ag=977  draft_cg_true=8  draft_cg_false=1
dp7 ag=979  draft_cg_true=6  draft_cg_false=2   <-- different split
```

The rank with a different graph/eager split (`dp7`: 6/2) is exactly one of the
ranks with a different all-gather total (979 vs 977). The counts are frozen —
re-sampled 30 s apart with no change, so this is a deadlock, not slow progress.

An earlier run showed the same signature with a different victim rank
(651/653/651×5/652 with `dp1` and `dp7` off), consistent with a race whose
victim depends on which rank happens to hold work.

## Where the ranks are blocked (clean baseline, no Bug 4 patch)

```
7 idle ranks : process_batch_result_idle -> synchronize()  (batch_result_processor.py:623)
1 busy rank  : resolve_seq_lens_cpu      -> synchronize()  (overlap_utils.py:295)
```

Two *different* host-blocking syncs, which is the expected downstream symptom of
a misaligned collective stream rather than a defect at either sync site.

## Correction to earlier analysis in this kit

Earlier notes recorded the hang as `dp_gather_replicate` ← `prepare_mlp` ←
`deepseek_v2.py:2242`. That was accurate **before** Fix A/A2. After Fix A/A2 no
rank reaches `dsa_backend` at all and the deadlock surfaces elsewhere. The
document `evidence/pyspy_20260729_1120_lmhead_allgather.md` captures the
intermediate state.

## Correction: my own Bug 4 "fix" was actively harmful

`fix_bug4_uniform_event.py` was previously recorded only as "falsified". It is
worse than that: with it installed, all 8 ranks deadlock in
`overlap_utils.py:292` — *the wait that fix itself hoisted*. It converted a
benign state into a hard hang and contaminated the measurements taken while it
was live. It has been reverted (`git checkout` of `overlap_utils.py`, verified
zero diff against upstream). **Do not reapply it.**

While reverting I initially deleted upstream's own HIP wait block along with my
addition; `git diff` caught it (expected 0 diff, saw -7 lines) and
`git checkout` restored the pristine file. The other four patches were verified
still present by grepping their markers individually.

## Candidate fixes (not yet implemented)

The contract to restore: **the graph/eager choice must be uniform across the DP
group**, because it changes the collective issue sequence.

1. *Naive, per the project rule* — drop the `not is_idle()` term so the guard
   evaluates identically on every rank. Cost: idle ranks also fall to eager on
   those iterations, which is slower but harmless (an idle rank has no real
   work). This is the smallest change that restores the invariant and is the
   right first experiment.
2. All-reduce the decision across the DP group before acting on it. Correct in
   general but adds a collective on the hot path.
3. Make the graph path issue the same collective sequence as eager. Largest
   change, best steady-state performance.

A separate control experiment is running on a second PD pair with
`--disable-cuda-graph`: if the hang disappears when *no* rank can take the graph
path, that independently confirms this diagnosis, since the guard becomes
irrelevant when both arms are eager.
