# Exp 3a / 3c — porting PR #32209's patch 2b: seven rounds, root cause NOT closed

**Ran:** 2026-07-30 → 2026-07-31, AMD spur cluster, 2 × MI355X nodes (three
different node pairs; the first two died `NODE_FAIL`).
**Author:** yihou
**Status:** **UNRESOLVED — negative result.** The port fails reproducibly at
conc=32. Seventeen candidate causes were measured and eliminated. The
remaining defect is localized to a narrow window but is **not identified**.

> This kit does not contain a fix and does not claim one. It is published for
> the eliminations and the reproducer, both of which are reusable.

## What this arm is, and is not

**It is not our fix.** The GLM-5.2 PD + DPA + MTP deadlock this whole effort
targets was fixed and verified on 2026-07-29 (kit
`..._20260729_bug2b_draft_graph`, 1516/1516 requests) and re-confirmed on
2026-07-30 with the draft CUDA graph *measured* in use at 97.1 % (kit
`..._20260730_exp3b_patch4_32209`). Those results are unaffected by anything
here; the four patch diffs in `patches/*.diff` are byte-identical across all
five kits (`md5 b326e523…` for patch 2).

**It is upstream-alignment research.** Upstream PR **#32209** fixes the same
defect with a better-placed patch 4 (zero extra collectives). Its *other* half
— patch 2b — reconciles DSA decode row counts by trimming q/top-k instead of
expanding the page table as ours does. This arm asks whether that half stands
on its own on our HIP/tilelang path.

**Answer: not as ported.** e3a (trim only) and e3c (trim + upstream's
`_slice_draft_output_to_local_tokens`) both crash identically.

## Result

| arm | patch 2b | patch 4 | extra | 4-prompt | conc=32 |
|---|---|---|---|---|---|
| **e3a** | #32209 trim | ours | — | 4/4 | **0/32** ×3 |
| **e3c** | #32209 trim + slice | ours | — | 4/4 | **0/32** ×4 |

Always the same failure, on three node pairs and a rebuilt image:

```
ValueError: output tensor size must be equal to world_size times input tensor size
  dp_gather_replicate -> _dp_gather -> _dp_gather_via_all_gather
  -> get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
```

503 latency ≈ 12–23 s (a real backend failure; a ~0.4 s 503 is a stale router
circuit breaker and was ruled out each time by restarting the router).

## The measured anomaly

The MLP-sync all-gather sizes its buffer from a plan every rank agrees on.
Some ranks then hand it a different row count:

```
seq=18  buffer=32 = 8 ranks × 4      plan=[4,4,4,4,4,4,4,4]
                                     orig=[2,1,3,2,2,2,2,4]
  rank 1,7        local_rows=4   ✅ matches plan
  rank 0,2,3,4,5,6 local_rows=6  ❌ ValueError
  inp_rows=4 on ALL EIGHT ranks — the current forward's own input is correct
```

Pooled over two instrumented runs, every faulting record satisfies

```
1 < orig[rank] < plan[rank]        9/9 faults
```

but that condition **also holds for 18 of 211 non-faulting records**, so it is
**necessary, not sufficient**. Observed `(orig, plan, local)` triples:
`(2,3,4)`, `(2,4,6)`, `(3,4,6)`.

## Eliminated — seventeen, all by measurement

Each was instrumented and observed, not argued from source.

| # | candidate | how it died |
|---|---|---|
| 1 | `DpPaddingMode` divergence (charter H3) | `pad_mode=1` (MAX_LEN) on all 8 ranks, all runs |
| 2 | patch 4's graph/eager vote failing | `vote graph=0` identical on all 8 ranks; 51/7 split identical per rank |
| 3 | a rank replaying the draft graph (Python probes blind) | all ranks `graph=0` on the faulting iteration |
| 4 | a rank skipping a draft step | all ranks logged both `i=0` and `i=1` |
| 5 | patch-2b trim/restore leaking rows | trim is row-neutral: ranks that trimmed 2 rows still delivered the correct count |
| 6 | draft output not sliced to local tokens | added upstream's slice hunk (e3c) — crash unchanged, and its `RuntimeError` never fired |
| 7 | `num_draft_tokens` leaking into the buffer | `ndt=1`; the bad value was 4 in one run, 6 in the next |
| 8 | TARGET_VERIFY rows contaminating decode | `fwd=2` (DECODE) on every faulting record |
| 9 | `hidden_states` stale across iterations | `draft_entry`: `hs_rows == bs` on **456/456** records |
| 10 | `merge_batch` concatenation | **0 calls** in the faulting runs |
| 11 | `merge_batch` idle-stub adoption | 0 calls |
| 12 | `filter_batch` (both arms) | 0 calls |
| 13 | upstream #31760's page-table/top-k mismatch | `pt_rows == topk_rows` on **100 %** of transform calls here — our patch 2b reconciles them first |
| 14 | `_pad_inputs_to_size` padding wrong | **188/188** exits consistent, 0 exceptions |
| 15 | `spec_info.hidden_states` pad target wrong | target always `== num_tokens`; faulting iteration measured `before=2 target=4` — correct |
| 16 | `post_forward_mlp_sync_batch` restore wrong | 68 records, `backup_rows == bs` throughout |
| 17 | arithmetic rules `local == plan+1`, `local == plan+2` | each fit one run and was refuted by the other (see `notes.md` §4) |

## Where the defect must be

Padding produces the **correct** row count (4). The all-gather receives **6**.
Both facts are measured on the same iteration, so the discrepancy is introduced
**between `_pad_inputs_to_size` and the MLP all-gather inside the draft
forward** — a window of one forward pass. Nothing narrower is established.

## Upstream state (queried with `gh` on 2026-07-31, not recalled)

Three open, unmerged PRs cover this cluster; none addresses the defect above.

| PR | title | state |
|---|---|---|
| [#32209](https://github.com/sgl-project/sglang/pull/32209) | Fix PD decode hang with DP attention and GLM-5.2 MTP | OPEN, REVIEW_REQUIRED |
| [#31760](https://github.com/sgl-project/sglang/pull/31760) | [DSA] Handle partial-DP padding in Decode page-table transform | OPEN, REVIEW_REQUIRED |
| [#32722](https://github.com/sgl-project/sglang/pull/32722) | [RED regression] Test GLM-5.2 PD + DP attention + MTP | OPEN (deliberately failing) |

#31760 is the interesting one: it independently reports **our exact numbers**
("a 3-row page table and a 4-row top-k tensor whose last row is all `-1`") and
argues the equal-rows assert is simply the wrong contract — partial-DP padding
should be masked *inside* the Triton kernel, not reconciled before it. Both our
patch 2b and #32209's reconcile beforehand. Elimination #13 shows that on our
path the two counts are already equal, so #31760's specific failure is not live
here — but its framing suggests the reconciliation approach itself may be
treating a symptom.

## Honest accounting

Five times across these rounds a mechanism was proposed from a small number of
agreeing data points and then refuted by the next batch of data (`page_table_1.
shape[0]` as the fix; "rank 4 misses a trim"; "the draft loop carries row counts
forward"; `local == plan+1`; `local == plan+2`). All five are recorded in
`notes.md` §4 with what refuted them. The eliminations in the table above are
of a different kind — each is a direct observation, and none has been
contradicted.

## Folder map

- `REPRODUCE.md` — cold-start reproduction of the failure, incl. image rebuild
- `environment.md` — hardware, image (rebuilt), commit, node pairs
- `notes.md` — the seven rounds, the five retractions, what to try next
- `patches/` — the #32209-style 2b port (both halves) + the four kit diffs
- `scripts/` — all six instrumentation probes, arm applier, boot/router
- `results/` — raw per-request jsonl, 7 runs (e3a ×3, e3c ×4), **all 0/32**
- `logs/` — full prefill/decode/router logs, gzipped, both arms, all rounds
- `logs/MANIFEST.md` — which log belongs to which round, and which probes were
  live in it (verified by marker counts in the shipped files)
