# Notes — Exp 3a / 3c

## 1. The seven rounds

| # | arm | what changed | outcome |
|---|---|---|---|
| 1 | e3a | #32209 trim, our patch 4 | 0/32. `instr_e3` present but reading a nonexistent field (`global_num_tokens`), so it logged `None` and proved nothing |
| 2 | e3a | fixed the probe field names; added trim-fire counter | 0/32. `pad_mode=1` uniform → H3 dead. Miscounted trim entries → wrong mechanism (§4.2) |
| 3 | e3a | rebuilt image, fresh nodes, `(rank, step)` probe | 0/32. Vote uniform, all steps present → hypotheses A and B dead |
| 4 | e3c | added upstream's `_slice_draft_output_to_local_tokens` | 0/32, **crash unchanged** → §4.3 refuted |
| 5 | e3c | 4-in-1 multi-hypothesis probe | 0/32. Killed H-A/H-B/H-D in one boot; H-C (stale `hidden_states`) survived |
| 6 | e3c | 5-site `hidden_states` origin trace | 0/32. `draft_entry` 456/456 consistent → H-C dead; merge/filter never called |
| 7 | e3c | 4-site padding probe | 0/32. Padding is provably correct; window narrowed to one forward |

Rounds 1–4 tested one hypothesis per boot (~25–40 min each) and three of four
guesses were wrong. Rounds 5–7 instrumented several independent sites per boot
and eliminated four, three and three candidates respectively. **The multi-site
approach was strictly better and should be the default for this bug class.**

## 2. What the port actually is

`patches/patch2b_32209_style.py` — #32209's `_trim_trtllm_decode_dp_padding` /
`_restore_trtllm_decode_dp_padding`, relocated. Upstream patches
`_forward_trtllm` (CUDA/flashinfer); our path is `forward_decode →
dsa_decode_impl == "tilelang" → _forward_tilelang`, so the trim goes into
`forward_decode` right after `_pad_topk_indices`, and the restore is a thin
`*args/**kwargs` wrapper (that method returns from ~a dozen impl-specific
branches; wrapping once is the only edit covering all of them).

One deliberate divergence, and it is load-bearing: upstream trims to
`metadata.cache_seqlens_int32.shape[0]`; we trim to
`metadata.page_table_1.shape[0]`. On upstream's CUDA path the two coincide.
Under MTP here they do not — `init_forward_metadata` expands the page table by
`repeat_interleave(..., speculative_num_draft_tokens)`, so `page_table_1` is
per-token while `cache_seqlens_int32` stays per-request.

`patches/patch2b_32209_slice.py` — the other half, added in round 4:
`_slice_draft_output_to_local_tokens`, ported verbatim. **It does not fix the
crash**, and its own `RuntimeError` never fired, meaning the tensors it slices
were never short. Kept in the kit because "we tried upstream's complete change"
is itself a result.

## 3. Reading the instrumentation

Six probes, all measurement-only, all idempotent, all verified in **bytecode**
(never source — a stale `.pyc` silently reverts a patch).

| probe | marker | answers |
|---|---|---|
| `instr_e3.py` | `GLM52_E3INSTR` | rows/plan/padding-mode at the failing all-gather |
| `instr_p2bv2_rows.py` | `GLM52_P2BROWS` | did the patch-2b trim fire, and by how much |
| `instr_draft_steps.py` | `GLM52_DSTEP` | post-vote graph/eager decision; every eager draft step |
| `instr_multi.py` | `GLM52_MULTI` | 4 hypotheses at 2 sites (gather + transform) |
| `instr_hs_origin.py` | `GLM52_HSO` | 5 `hidden_states` writer/reader sites |
| `instr_pad.py` | `GLM52_PAD` | DP padding enter/exit, spec pad, restore |

The two that carry the most weight:

```
GLM52_MULTI gather seq=18 rank=0 local_rows=6 global_rows=32
  plan=[4,4,4,4,4,4,4,4] orig=[2,1,3,2,2,2,2,4] bs=4 fwd=2 inp_rows=4
GLM52_PAD   hs_pad rank=0 before=2 target=4 bs=4 backup=2 fwd=2
```

Same rank, same iteration: padding targets 4 and is correct; the gather gets 6.

## 4. Five retracted mechanisms

Recorded because the failure mode is consistent and worth not repeating:
**a small set of agreeing numbers was treated as a mechanism.**

**4.1 — "`page_table_1.shape[0]` is the fix."** The merged Exp-3 arm crashed,
the row-count source was changed from `cache_seqlens_int32` to `page_table_1`,
the arm passed, and that was reported as the fix. Refuted: arm e3a runs exactly
that corrected code and crashes identically. What made the merged arm pass was
patch 4 driving draft-graph usage to 0 % (every rank eager). The row-count
change is plausibly *necessary*; it is demonstrably not *sufficient*.

**4.2 — "rank 4 entered the trim one fewer time, so `real` diverges."**
Refuted by full-population logging: the step counts were equal. The apparent
gap came from filtering on `pad_rows != 0`, which hides ranks whose trim was a
no-op. An artifact of the query, not of the run.

**4.3 — "the draft loop carries row counts across steps."** `step rows` matched
`orig` position-wise, which is the loop working correctly, not a defect. Acted
on it anyway by porting upstream's slice hunk (round 4): crash unchanged.

**4.4 — "`local == plan + 1`."** Fit 3/3 faults in round 3. Round 5 produced
`plan=4 → local=6`.

**4.5 — "`local == plan + 2`."** Fit 6/6 faults and 0/142 correct records in
round 7 — then 0/3 against round 3's data. Pooling both runs leaves
`(2,3,4)`, `(2,4,6)`, `(3,4,6)`, which no single arithmetic rule covers.

The eliminations in README are not of this kind: each is a direct observation
of a site, and none has since been contradicted.

## 5. What to try next

Do **not** resume by proposing another mechanism. The window is one forward
pass; brute force is now cheaper than inference:

1. **Dump row counts layer by layer** inside the draft forward, from
   `_pad_inputs_to_size` exit to `prepare_mlp`. The transition from 4 to 6 is
   in there and will be visible directly. Unglamorous and reliable.
2. **Bisect by config** rather than by code: does the fault survive
   `--speculative-num-steps 1`? with IndexShare off? at DP4? Each answer
   partitions the space without needing a mechanism.
3. **Ask upstream.** #31760's author hit the same row-count pair from a
   different direction and has context we lack. The eliminations here plus the
   reproducer are worth more to them than another guess from us.

Whoever picks this up: the seventeen eliminations are solid and need not be
redone. The `1 < orig < plan` trigger condition is necessary but **not**
sufficient (18/211 non-faulting records satisfy it) — do not build on it as if
it were a rule.

## 6. Traps hit during these rounds

- **Anchoring a patch on a bare class name.** `class EagleDraftInput` also
  appears in a comment, and `ForwardBatch` is `@dataclass`-decorated, so
  inserting before the bare `class` line split the decorator from its class →
  `SyntaxError`, twice. Anchor on `@dataclass\nclass X(...):`.
- **`apply_arm.sh`'s RESET list must contain every file a probe touches.**
  `dp_attention.py` was missing, so a corrected probe would have silently not
  applied over the previous round's copy. Caught only by checking
  `git status --porcelain` first.
- **A probe reading a nonexistent attribute fails silently.** Round 1 logged
  `global_num_tokens=None` for a whole run because the field is
  `global_num_tokens_cpu`. `getattr(..., None)` turns a typo into a wasted boot.
- **`nohup ... &` inside `spur exec` does not survive** — the exec namespace
  teardown kills it (CLAUDE.md). The image build only completed because it
  finished within the exec's lifetime.
- **Filtering log records changes what you conclude.** See §4.2.
- **NODE_FAIL takes every job at once.** Five jobs died together at 12:49 on
  2026-07-30; `spurctld` then refused connections for ~15 min. Nothing on the
  nodes survives, so the image tar must live on `/shared_nfs`.
