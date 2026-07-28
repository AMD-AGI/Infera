# Plan — fix GLM-5.2 DSA "DPA + MTP" crash on gfx950 (Bug 1)

## Goal
Make `--enable-dp-attention` + EAGLE MTP work together for GLM-5.2-MXFP4 on MI355X,
first **single-node mix**, then **2-node mooncake PD**. Fix in-container first,
then emit a patch.

## What the source actually says (verified in the image, sglang 0.5.15.post1)

The todo file's hypothesis was **partly right and partly wrong**. Verified against the
real files inside `infera.yihou.sglang.1.0`:

| Claim in todo | Verdict |
|---|---|
| `should_use_dsa_fused_topk` disables fused top-k under PD + IndexShare | ✅ true (`dsa/utils.py:71-79`) |
| The unfused `fast_topk_v2` asserts `lengths.size(0)==B` | ✅ true (`sgl_kernel/top_k.py:41`) |
| Fix = "clamp padded rows to 0 in `seqlens_expand_triton`" (mirror CUDA #30378) | ❌ **already done** — `triton_ops/pad.py:355` already has `tl.maximum(start + offs, 0)` with a DP-padding comment. Route (2) of the todo is a no-op. |
| The bug needs PD to trigger | ❌ **not required** — see below |

### The real defect: a HIP-only row-count mismatch in the indexer

`dsa_indexer.py::_get_topk_paged` computes the indexer logits and then top-k:

```python
q_offset = sum(metadata.get_dsa_extend_len_cpu())   # REAL (unpadded) query rows
...
if self.paged_mqa_logits_backend.is_aiter():        # <-- the ROCm/gfx950 path
    logits = aiter_paged_mqa_logits(q_fp8, ..., seqlens_32, ...)
else:                                               # the CUDA path
    logits = deepgemm_paged_mqa_logits_split(..., q_offset=q_offset)

topk_result = metadata.topk_transform(logits, self.index_topk)   # lengths = seqlens_expanded

# Restore possible padding exist in the hidden states.
if not _is_hip and q_offset < q_fp8.shape[0]:       # <-- HIP EXCLUDED
    ... pad topk_result back up to q_fp8.shape[0] ...
```

Two asymmetries between the CUDA and HIP branches:

1. **The CUDA path slices to `q_offset`** (`deepgemm_paged_mqa_logits_split` does
   `q_fp8[:q_offset]`, `weights[:q_offset]`) so `logits.shape[0] == q_offset ==
   seqlens_expanded.shape[0]` — the assert holds. **The aiter/HIP path does NOT
   slice**: `aiter_paged_mqa_logits` allocates
   `logits = torch.empty((batch_size * next_n, max_seq_len))` from `q_fp8.shape[0]`,
   i.e. the **DP-padded** row count.
2. **The CUDA path pads the result back** afterwards; the HIP branch is gated off by
   `not _is_hip`, consistent with (1) — HIP never shrank, so it never re-pads.

Under DP-attention the hidden states are padded to a common per-rank token count, so
`q_fp8.shape[0] > q_offset`. Therefore on gfx950:

```
score.shape[0]  = q_fp8.shape[0]            (DP-padded)
lengths.shape[0]= seqlens_expanded.shape[0] = q_offset   (unpadded)
=> RuntimeError: Expected lengths.size(0) == B
```

This explains every observed fact:
- DPA-only passes: plain decode has `extend_len=1` per row, `q_offset == B`, no padding gap
  on that path.
- MTP-only passes: no DP padding at all, so `q_fp8.shape[0] == q_offset`.
- Only the combination crashes: MTP's draft-extend inflates rows *and* DP pads them.

**PD is not the trigger.** PD only decides *which* top-k kernel runs
(`should_use_dsa_fused_topk` → unfused). The unfused kernel is simply the one that
*checks* the row count; the fused one would read garbage rows instead. The mismatch
exists either way — which is why we can and should reproduce it **single-node** first.

## Fix (route A — primary)

Make the HIP branch obey the same contract as the CUDA branch: **top-k over the real
rows, then restore padding.** In `dsa_indexer.py::_get_topk_paged`:

- Slice the aiter logits to the real row count before `topk_transform`, so
  `score.shape[0] == lengths.shape[0]` holds on both backends.
- Drop the `not _is_hip` gate on the padding-restore so the returned `topk_result` is
  back at `q_fp8.shape[0]`, which the caller (`forward_cuda` → MLA attention) expects.

This is ~5 lines, no new kernel, and makes ROCm structurally identical to the
already-correct CUDA path. It is also PD-independent, so it fixes single-node mix and
PD decode at once.

Guard rails: only re-slice when `q_offset < logits.shape[0]` and `q_offset > 0`, so the
non-DP path is bit-identical to today (zero risk to the proven Config A / Config B runs).

### Fallback (route B) — if route A shows a residual mismatch
Pad `lengths` up to the logits row count with **0**-length rows instead of slicing
(0 = "empty row → all -1", the invariant the fused kernel's docstring states). Applied
where `dsa_seqlens_expanded` is built for `is_draft_extend_v2()`. Kept as a fallback
because slicing (A) is strictly cheaper — it avoids running top-k on padded rows at all.

Route C (implement upstream's `TODO(kpham-sgl)` to keep *fused* top-k under PD) is
explicitly **out of scope**: it's a performance optimization, not a correctness fix,
and it can't be validated until A works.

## Execution steps

1. **Baseline repro (single-node mix, node069-equivalent = crsuse2-m2m-207).**
   `DPA=1 MTP=1` on one node, no PD. Capture the exact traceback + instrumented
   `score.shape[0]` vs `lengths.shape[0]` to *prove* the numbers above. Launched already.
2. **Instrument** — temporary print in `_get_topk_paged` (shapes, q_offset, forward mode)
   so the fix is driven by measured values, not inference.
3. **Apply route A** inside the container (edit `dsa_indexer.py` in place; no rebuild).
4. **Verify single-node mix:** 4/4 correctness probe; MTP actually active (accept len
   ~2.7-3.0); DP flags took (dp_size=8, ep_size=8); short conc stress.
5. **Regression-check:** re-run DPA-only and MTP-only single-node with the patch to
   confirm no change to the already-passing configs.
6. **Then PD (2 nodes: 207 + 197):** decode leg `DPA=1 MTP=1`, prefill symmetric DP8,
   mooncake mlx5+dmabuf per the proven transport recipe. 4/4 probe + conc=128 512/512,
   0 KVTransferError, `installTransport type=rdma`.
7. **Emit patch** only after step 4 passes (user: "有实质进展后，生成patch"):
   a unified diff against the image's stock `dsa_indexer.py`, plus a drop-in copy for
   bind-mounting, into `patches/`.
8. **Document** — update this packup (RESULTS/notes/todo) + `CLAUDE.md`, and record the
   corrected root cause in memory (the current memory entry says "PD+IndexShare+DPA
   triple" and "fix = clamp seqlens", both of which this analysis supersedes).

## Success criteria
- Single-node mix DPA8 + MTP: 4/4 probe, spec-dec active (accept len ≥2.7), no crash under load.
- PD decode DPA8 + MTP: 4/4 probe, conc=128 512/512, 0 KVTransferError.
- DPA-only and MTP-only unchanged (no regression).
- Patch file reproducible from a clean image.

## Risks / open questions
- The aiter MQA-logits kernel may write garbage into padded logits rows; slicing avoids
  reading them, but if the *kernel itself* faults on padded `q_fp8` rows we may also need
  to slice its inputs (`q_fp8[:q_offset]`, `weights[:q_offset]`) — that is exactly what
  the CUDA path does, so it is a natural extension of route A.
- CUDA-graph capture of the draft-extend path may re-introduce a static padded shape;
  if so the slice must be done on a fixed-size buffer, not a dynamic view.
- Node 321 was wedged in the previous run by a hicache alloc — we are not enabling
  hicache/kvd here, so that trap is out of the path.
