# Variant B — the DRAFT cuda graph is the culprit

> **SUPERSEDED 2026-07-29 evening.** The "follow-up needed" at the bottom of this file is
> **done**: option 1 (all-reduce the decision across the group) was implemented and
> verified. See `../../glm52.mxfp4.spur.mooncake.packup_20260729_bug2b_draft_graph/`.
> Variant B's localization was correct — the draft graph is the culprit — and the proper
> fix keeps that graph, using it on 98.4% of iterations instead of disabling it.

Measured 2026-07-29 13:00–13:06, job 9006 (decode leg), graphs otherwise ON.

## What was changed

One line, in `draft()` (`eagle_worker_v2.py`, right after the guard at :517):

```python
can_cuda_graph = False   # GLM52_VARIANT_B: force the draft path eager on EVERY rank
```

Nothing else. Crucially `disable_cuda_graph=False` in the resulting
`server_args` — the **target decode graph and the draft-extend graph are still
enabled and still captured** (`Capture draft decode CUDA graph end` appears in
the log). Only the draft graph is bypassed at runtime.

## Result: the hang is gone

| test | result |
|---|---|
| PD warmup (8 concurrent, one per DP rank) | **PASSED in 10 s** (13:00:01 → 13:00:11) |
| 4 × 24-token, sequential | **4/4 HTTP 200**, 0.5–1.0 s each |
| 1 × 512-token | **HTTP 200**, 5.57 s, 512/512 tokens |

Per-request detail (`meta_info`):

| req | dp_rank | accept_len | tokens | text |
|---|---|---|---|---|
| 1 | 5 | 2.00 | 24 | ` Paris. Bordering countries include Belgium,…` |
| 2 | 6 | 1.50 | 24 | ` Paris. Distance from Paris to Paris is 0 Ki…` |
| 3 | 7 | 2.40 | 24 | ` Paris. It is one of the most beautiful citi…` |
| 4 | 0 | 1.85 | 24 | ` Paris. Distance from Paris to the nearest a…` |
| long | 5 | 2.81 | 512 | coherent essay text |

`spec_accept_length` between 1.5 and 2.8 confirms MTP/EAGLE spec-dec is
genuinely active, not silently bypassed. Requests landed on 4 different DP
ranks, so this is not one lucky rank.

Contrast with the identical configuration *without* this one line: warmup often
passed but a **single** request then deadlocked (see `WARMUP_MATRIX.md` runs
1/3/4), or warmup itself hung (runs 5/6).

Note the warmup time: **10 s here vs 1 m 56 s for the eager control** — the
other two graphs are doing their job, so this is not a disguised "everything
eager" run.

## Conclusion

The deadlock is caused specifically by the **draft (EAGLE multi-step) CUDA
graph**, not by the target decode graph and not by the draft-extend graph.
`--disable-cuda-graph` fixed the hang only because it happened to include the
draft graph.

This is a much narrower and more useful localization than "CUDA graphs cause
the hang": a real fix can keep the target and draft-extend graphs, losing only
the draft graph's speedup — and only until the underlying rank-divergence is
fixed properly.

## Why the earlier Bug 2b attempt failed

Bug 2b removed only the `not forward_batch.forward_mode.is_idle()` term from the
guard, on the theory that this term alone made the decision rank-divergent. It
did not stop the hang, and measurement showed the graph/eager split was *still*
rank-divergent with the fix applied (dp0 16/3 vs dp7 14/4). The surviving terms
— `can_cuda_graph` as returned by `prepare_for_draft`, and
`draft_input.dsa_topk_indices is None` — are themselves rank-dependent.

Variant B forces the whole decision to a constant, removing every rank-dependent
term at once, and that works. So the defect is **the rank-divergence of the
draft graph/eager decision as a whole**, not any single term in the guard.

## Follow-up needed

Variant B is a localization, not a shippable fix — it disables the draft graph
unconditionally and costs its speedup. The proper fix is to make the draft
graph/eager decision uniform across the DP group while still allowing the graph
when all ranks agree it is usable. Options:

1. all-reduce the boolean across the DP group before acting on it (correct,
   costs one small collective per draft call);
2. derive the decision only from rank-invariant quantities;
3. make the graph and eager paths issue identical collective sequences.

## Newly discovered: Bug 5 — assert under concurrency

Immediately after the above passed, a 16-way concurrent burst (128 tokens each)
**crashed** the decode leg — a crash, not a hang. All ranks:

```
sglang/srt/layers/attention/dsa_backend.py:2154  forward_decode
  -> dsa/transform_index.py:14   transform_index_page_table_decode
  -> dsa/transform_index.py:138  transform_index_page_table_decode_fast
     assert page_table.shape[0] == topk_indices.shape[0]
AssertionError
```

This is the **same family as Bug 1** (a DP-padded row count meeting an
unpadded one) but on a different code path: Bug 1 was in
`dsa_indexer.py::_get_topk_paged`, this is in `transform_index.py`. It did not
surface earlier because every previous probe was sequential; 16 concurrent
requests produce the mixed batch shapes that trigger it.

Tracked separately. It does not affect the Variant B conclusion above, which was
established on warmup + sequential + 512-token traffic before this burst.
