# sglang GLM-5.2 patches — upstream PR inventory

Scope: every patch in `deploy/docker/patches/` whose `target.library` is `sglang`
and whose problem was found on, or verified against, GLM-5.2. That is all seven
sglang patches in the tree — no sglang patch in this repo is unrelated to GLM-5.2.

Source of truth: each patch's `<prefix>.upstream.status.yaml`, re-checked against
live upstream on 2026-08-07 (`gh` for PR/issue state, `git show origin/main` for
source). vLLM / mooncake / aiter / atom patches are out of scope here even where
they mention GLM.

Upstream `main` read at `3ed2a0adf3` (2026-08-07).

## The table

| # | Patch | Component | Existing upstream PR | State (2026-08-07) | Ours? | main affected? | Needs a NEW PR |
|---|-------|-----------|----------------------|--------------------|-------|----------------|----------------|
| 1 | `sglang/patch_glm52_nextn_quark_exclude.py` | quantization | sglang#30265 | MERGED 2026-07-08 | no | **no** — fixed | **no** |
| 2 | `sglang_disagg/patch_mooncake_early_send_wait_event.py` | disaggregation | *none* | — | — | **yes** | **YES** |
| 3 | `sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | dsa | sglang#33059 | OPEN, REVIEW_REQUIRED | **yes** | yes | no — already filed |
| 4 | `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2a) | dsa | *none* | — | — | **yes** | **YES** |
| 4b | `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2b) | dsa | sglang#32209 | OPEN, REVIEW_REQUIRED | no | yes | **no** — see below |
| 5 | `sglang_dsa/draft_cuda_graph_dp_vote.diff` | speculative-decoding | sglang#32209 | OPEN, REVIEW_REQUIRED | no | yes | no — deliberate |
| 6 | `sglang_rocm/patch_hicache_rocm_host_alloc.py` | hicache | *none* | — | — | **yes** | **YES** |
| 7 | `sglang_rocm/patch_hicache_rocm_staged_write_back.py` | hicache | sglang#30350 | OPEN, CHANGES_REQUESTED | no | yes | no — nudge #30350 |

Three patches need a new upstream PR: **#2, #4 (2a half only), #6**.

## Per-patch notes

### 1 — `patch_glm52_nextn_quark_exclude.py` — no PR needed
sglang#30265 ("[AMD] Fix GLM-5.2 MTP Quark excludes", wangjiaxin99) is MERGED and
is a superset of ours: it gives GLM-5.2 a dedicated `GlmMoeDsaForCausalLMNextN`.
Our patch is a one-line backport onto the frozen `v0.5.15.post1` base, which was
cut from `release/v0.5.15` without it. Nothing to upstream — it is upstream.

Carry-forward warning (already in the record): the anchor string
`ckpt_prefix = f"model.layers.{config.num_hidden_layers}"` is STILL on main
(`deepseek_nextn.py:328`), so the patch does not self-disable on a fixed base.
Decide the drop from the base version, not from the build log.

### 2 — `patch_mooncake_early_send_wait_event.py` — NEEDS A PR
No upstream PR, and no upstream issue that establishes the root cause (#25583 is
the same corruption shape on an *aggregated* GLM-5-FP8 server, auto-closed
inactive — suggestive only).

Re-verified on main today:
- `disaggregation/mooncake/conn.py` — `wait_event` occurs **0 times**, while
  `disaggregation/mori/conn.py` has **6**. The barrier exists only for mori.
- `disaggregation/prefill.py:1113` records `_early_send_wait_event` on the
  radix early-send path; the overlap non-final-chunk send at line 812 records
  nothing at all.

So main is affected on both halves. This is a **silent** correctness bug — no
crash, no log line, output partially wrong past the first prefill chunk. Best
candidate in the tree for an upstream PR.

Before opening: the record's own open action asks for the cost of the added
`synchronize()` on prefill throughput. Not measured. Will be stated as an open
question in the PR rather than claimed either way.

### 3 — `patch_dsa_indexer_hip_dp_padded_rows.py` — already filed
Ours: sglang#33059, OPEN, `REVIEW_REQUIRED`, `MERGEABLE`/`BLOCKED`, last touched
2026-08-07. No action beyond review chasing.

Note the file moved upstream: `layers/attention/dsa_indexer.py` →
`layers/attention/dsa/dsa_indexer.py`. #33059 is already against the new path.

### 4 — `dsa_backend_dp_sync_and_page_table_rows.diff` — SPLIT
This diff bundles two independent defects. They upstream differently.

**2a — DP host-sync deadlock. NEEDS A PR.** No upstream issue, no upstream PR,
ours or anyone's. Confirmed live on main:
- `dsa_backend.py:794` — `max_seqlen_k = int(forward_batch.seq_lens.max().item())`,
  a blocking D2H sync on a branch only some DP ranks take.
- `dsa_backend.py:854-861` — the two further unconditional `.cpu()` mirrors
  (2a2) are still there. With 2a alone the hang persists, so both go together.

**2b — page-table row mismatch. NO new PR.** sglang#32209 (HZY-Wade) is OPEN and
already addresses this row mismatch, by TRIMMING q/top-k where we EXPAND the page
table. Our record documents a negative result: porting #32209's trimming approach
onto this HIP/tilelang path fails reproducibly at concurrency 32 (0/32 across
seven runs), with the root cause **not identified**. Opening a competing PR on an
unexplained failure would be noise. Correct move is to finish that investigation
first; keep 2b local until then.

### 5 — `draft_cuda_graph_dp_vote.diff` — no PR, deliberate
sglang#32209 carries this exact fix with the same strategy (group decision rather
than per-rank), and our diff adopts its placement verbatim so the two converge.
sglang#32527 (Xavier1994) reports the same deadlock independently on 8× Blackwell
— not ROCm-specific — and is still OPEN with no activity.

The value we can add is a datapoint on those threads, not a fourth PR.

### 6 — `patch_hicache_rocm_host_alloc.py` — NEEDS A PR
No upstream issue and no upstream PR. Confirmed live on main
(`mem_cache/pool_host/common.py:177-183`):

```python
ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {"npu": alloc_with_pin_memory, "musa": alloc_with_pin_memory},
)
```

No HIP entry. The merged precedent to copy is sglang#23361 ("[MUSA][19/N] Support
HiCache with pin_memory allocator") — same one-line dispatch override, same
reason. This is the clearest missing-PR gap in the whole tree: a crash-class fix,
main affected by direct source read, and an already-merged PR of the same shape.

Anchor-collision risk when it lands: #32503 and #32792 (both OPEN, Intel XPU)
touch the same dict.

### 7 — `patch_hicache_rocm_staged_write_back.py` — no new PR
sglang#30350 (Emmanuel0612) is the upstream repair and is strictly better than
ours: it flips the three CUDA-only gates via `_is_cuda_alike`, covers the
DeepSeekV4 pools we do not, and teaches `staged_write_back.cuh` to accept
`kDLROCM`. It is OPEN with `CHANGES_REQUESTED` (HaiShaw, 2026-07-13); the
conflict was cleared the same day and no re-review has been requested.

Our MI300X datapoint is already on the thread (llying-001, 2026-08-04). Correct
action is a re-review nudge, not a competing PR. Note the inversion: the MERGED
PR here (#28534) is what *introduced* the disagreement.

## Plan

1. Open an upstream PR for **#6** (hicache ROCm allocator) — smallest, strongest
   evidence, merged precedent.
2. Open an upstream PR for **#2** (mooncake early-send wait event) — silent
   correctness bug, three files, mirrors what mori already does.
3. Open an upstream PR for **#4 / 2a** (DP host-sync deadlock) — 2a + 2a2 only.
   2b stays local.
4. All three as **drafts**, via the `open-source-pr` workflow: check upstream
   main, strip local-repo semantics (`GLM52_*` markers, the `MARKER = "applied"`
   bytecode literal, infera path references), re-validate the adapted patch
   locally for equivalence and scope, then open.
5. Record the outcome in `pr.done.md` and update each patch's
   `upstream_prs` / `open_actions` in its `.upstream.status.yaml`.

Not doing, and why: #1 is already merged upstream; #3 is already our open PR;
#5 and #7 have a better third-party PR in flight that we should support rather
than compete with; #4 / 2b rests on an unexplained concurrency-32 failure.
