# Notes — Exp 3b

## 1. What the port changes, and why the placement matters

**What.** Both our patch 4 and #32209 fix the same defect: the draft graph/eager decision
in `EagleDraftWorker.draft()` is made per rank from rank-dependent inputs, ranks disagree,
graph-replay and eager issue different collective sequences, deadlock.

**The difference is placement.**

| | ours (v1) | #32209 (this arm) |
|---|---|---|
| where | inside `draft()` | scheduler, folded into MLP-sync |
| cost | **extra** 1-element gloo all-reduce, every draft call | **zero** extra collectives — one more int64 slot in an all-gather that already runs |
| reduce | `ReduceOp.MAX` over need-for-eager | `min()` over may-use-graph (equivalent) |

The port is four edits mirroring #32209's own hunks, plus plumbing to carry the flag from
`ScheduleBatch` to `ForwardBatch`:

1. `eagle_worker_v2.py` — `requires_dp_attention_eager_forward(batch)`, the rank-local
   predicate lifted out of `draft()` to scheduler time;
2. `disaggregation/decode.py` — call it **just before** `maybe_prepare_mlp_sync_batch`
   (that call is the collective that spreads it);
3. `dp_attn.py` — carry it as slot 7 of the gathered tensor, `min()`-reduced like
   `can_cuda_graph` (slot 2). Idle/inactive ranks contribute `1` (permissive) so an idle
   rank can never drag the group eager;
4. `eagle_draft_cuda_graph_runner.py` — `can_run_graph` additionally requires
   `can_run_dp_draft_cuda_graph`.

**Deliberate divergences from #32209**, both recorded rather than hidden:

- #32209 also touches `_forward_trtllm` and the CUDA idle path in
  `dsa_indexer.forward_cuda`. Neither is on our HIP/tilelang path — **not ported**.
- #32209's patch-2b half is ported separately (`patch2b_32209_style.py`) and is **broken**
  — see §3.

## 2. The 0 %-graph-usage bug — the most important thing in this kit

**What happened.** The first run of this arm passed everything: 4/4 probe, 32/32 at
conc=32, `acc_len` 2.66, zero tracebacks. The counters said:

```
rank=0..7  calls=200  graph=0 (0.0%)  refused_bs=0 refused_dp=0 refused_draftvote=200
rank=0..7  total=200  future_indices=200 (100.0%)  seed_none=0 (0.0%)
```

**Zero draft-graph replays.** The port had silently become Variant B — the workaround it
was supposed to replace.

**Why.** The script contained this, with a comment asserting the attribute did not exist:

```python
# "#32209 consults future_dsa_topk_indices_available, which this baseline
#  does not have (verified by grep)" -- THIS CLAIM WAS FALSE
if getattr(draft_input, "future_indices", None) is not None:
    return True          # require eager
```

The attribute **does** exist — `eagle_info.py:179`, `spec_info.py:261` — and
`eagle_disaggregation.py:71` sets it:

```python
spec_info.future_dsa_topk_indices_available = dsa_topk_indices is not None
```

with `scheduler.py:3314` maintaining it for the next iteration and `overlap_utils.py:271`
consuming it (it fills `dsa_topk_indices` from the buffer iff the flag is set). So the flag
is precisely "will term 4 be satisfied once resolved" — the question the predicate must
answer one iteration early.

Because overlap scheduling sets `future_indices` on **every** decode iteration, the
fallback fired every time. The fix is to read the flag the way upstream does:

```python
if getattr(draft_input, "future_indices", None) is not None:
    return not draft_input.future_dsa_topk_indices_available
return getattr(draft_input, "dsa_topk_indices", None) is None
```

**Result: 0.0 % → 97.1 %.**

**Why it was invisible without counting.** `seed_none = 0` in the broken run: the real
guard term never once fired, so the graph was available the whole time and forcing eager
cost only speed. Everything worked. It just wasn't testing anything.

**Context — this is charter criterion 5.** "The graph path is provably taken (marker count
> 0)", not "no hang". Variant B passes criteria 1–4 by disabling the graph. Any arm that
does not count graph usage cannot tell a fix from that workaround. This run is the
concrete proof that the criterion is not pedantry.

**A correction to a claim I made earlier.** I wrote in the patch script that
`future_dsa_topk_indices_available` was "verified absent by grep". It was not; the grep
that produced that belief was from the patch-1 investigation, over
`forward_batch_info.py`, and does not support the claim. The wrong claim was load-bearing:
it justified a fallback that silently disabled the feature under test.

## 3. What this arm says about patch 2b — and a retraction

This arm runs **our** patch 2b, not #32209's. The #32209 2b port is **broken**: run in
isolation (arm e3a, same day, same stack) it crashes at conc=32 with

```
ValueError: output tensor size must be equal to world_size times input tensor size
  at dp_gather_replicate -> _dp_gather_via_all_gather
```

**Retraction.** Earlier the same day the merged Exp-3 arm hit that same crash. I changed
the 2b trim from `metadata.cache_seqlens_int32.shape[0]` (per-request) to
`metadata.page_table_1.shape[0]` (per-token, correct under MTP) and the arm then passed —
and I reported that as the fix. **It was not.** Arm e3a runs exactly that corrected code
and crashes identically. What actually made the merged arm pass was patch 4 forcing every
rank eager, which sidesteps the padding-mode inconsistency. The row-count change may well
be *necessary*; it is demonstrably not *sufficient*, and it is untested as a fix.

## 4. Why the merged arm had to be split

Merged Exp-3 = #32209's 2b + #32209's 4. It passed 4/4, 32/32, 64/64.

Both halves were defective, and **the patch-4 defect masked the patch-2b defect**: with
every rank eager, the eager path recomputes the DP padding mode per step, avoiding the
inconsistency the 2b port introduces. Fix patch 4 → graph gets used → 2b's defect becomes
reachable.

Two broken halves that pass together. Only the one-variable split exposed it.

## 5. Reading the counters

`GLM52_GUSE` (at `can_run_graph`, the single site that decides it):

| field | meaning |
|---|---|
| `graph=N (X%)` | draft graph actually replayed. **> 0 is the criterion.** |
| `refused_bs` | batch size / padding refused it (pre-existing behaviour) |
| `refused_dp` | `can_run_dp_cuda_graph` refused (pre-existing gate) |
| `refused_draftvote` | **patch 4's gate** refused — the group voted eager |

`GLM52_GUSE_WHY` (at the rank-local predicate, *before* the vote):

| field | meaning |
|---|---|
| `future_seed_ok` | overlap path, seed will be there → graph allowed |
| `future_seed_missing` | overlap path, seed will not be there → eager (real term 4, one iteration early) |
| `seed_none` / `graph_ok` | same, non-overlap path |

**The signature of a working fix is the two disagreeing:**

- `GLM52_GUSE` identical across ranks (777/800 everywhere) — the vote reconciles them;
- `GLM52_GUSE_WHY` **differing** across ranks (8, 9, 9, 9, 10, 10, 10, 11) — the ranks
  really do disagree locally.

If the local numbers were also identical, the vote would be inert and the arm would prove
nothing. This matches our patch 4's signature (local diverged 38×, voted 0×, graph 98.4 %).

## 6. Traps on this stack

- **Stale `.pyc` silently reverts a patch.** `apply_arm.sh` purges `__pycache__`,
  recompiles, greps bytecode. Markers must be identifiers, never `#` comments.
- **Logs contain binary bytes.** Plain `grep` reports "binary file matches" and `grep -c`
  returns 0 — reads as "no errors" when it means "grep gave up". Use `strings` / `grep -a`.
- **503 in ~0.4 s is a stale router circuit breaker,** not a dead backend. A real backend
  failure takes seconds. Restart the router and re-probe before diagnosing. (A 12 s 503,
  as arm e3a produced, *is* a real failure.)
- **`/home` was 100 % full** during this run: JIT caches moved to `/shared_nfs`, and the
  image tar named in earlier kits was gone. A freshly held node fails with
  `pull access denied`; move the image with `docker save`/`docker load`.
- **Killing the launcher + 8 scheduler PIDs frees all VRAM** — no container recreate
  needed (which would drop in-container patches). Kill by explicit PID; a broad `pkill -f`
  can match your own shell.
