# Root-cause analysis: why DPA + MTP crashes on GLM-5.2 DSA (and how expensive to fix)

Deep-dived the `RuntimeError: Expected lengths.size(0) == B` crash (Config A+B fused) with
web research + reading the image's sglang 0.5.15.post1 source. Two questions answered.

## The exact code path (from source in the image)

Crash chain: `eagle_worker_v2._draft_extend_for_decode` → DSA `Indexer.forward` →
`_get_topk_paged` → `metadata.topk_transform` → **`DSATopKBackend.topk_func` → `fast_topk_v2`**
(`sgl_kernel/top_k.py:41`), which asserts `lengths.size(0) == B` (B = `score.shape[0]`).

There are **two** top-k paths in `dsa_topk_backend.py`:
1. **Fused v2** (`_topk_transform_v2_paged`, the good one): explicitly handles DP-attention —
   its docstring says *"`lengths` entries … DP-padded / idle-companion rows … Metadata producers
   clamp padded rows to 0"*. This is the path upstream fixed for CUDA (PR #30378 "clamp padded-row
   seq_lens to >= 0", shipped v0.5.15).
2. **Unfused** (`topk_func` → `fast_topk_v2`, the crashing one): a hard `assert lengths.size(0)==B`
   with **no DP-padding handling**.

**What routes us to the crashing unfused path** — `should_use_dsa_fused_topk` (dsa/utils.py:71):
```python
pd_index_share_seed = (server_args.disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend)
# TODO(kpham-sgl): Transfer request-relative IndexShare seeds and remap them
# to decode-local KV slots so fused top-k can remain enabled under PD.
return envs.SGLANG_DSA_FUSE_TOPK.get() and not pd_index_share_seed
```
- `seed_dsa_topk_from_draft_extend` (eagle_worker_v2.py:273) = `index_share_for_mtp_iteration and
  index_topk is not None`. **GLM-5.2's config sets `index_share_for_mtp_iteration=True`** (its MTP
  "IndexShare" optimization: reuse the DSA indexer top-k from draft-extend), and with
  `--speculative-eagle-topk 1` (required for EAGLE here) it's True.
- We run **PD disaggregation** (`disaggregation_mode='decode'`).
- → `pd_index_share_seed=True` → `use_fused_topk=False` → `force_unfused=True` → the unfused
  `fast_topk_v2` with the un-clamped assert. When DP-attention pads/reshapes the batch, its
  `dsa_seqlens_expanded` (the `lengths`) no longer matches the draft-extended `logits` row count → assert fails.

So the crash is a **known-unfinished path**, flagged by upstream's own `TODO(kpham-sgl)`: under
**PD + GLM-IndexShare-MTP**, sglang deliberately disables the DP-safe fused top-k and falls back to a
top-k that isn't DP-safe. It's the intersection of three features (PD + MTP-with-IndexShare + DPA),
not MTP+DPA alone.

## Q1: general bug or AMD/gfx950-specific?

**A general sglang bug class, landing on a path that is explicitly unsupported/unvalidated on AMD.**
- The `topk_v2` + DP-attn padded-row mismatch is **CUDA-general in origin**: upstream fixed the same
  class on NVIDIA (PR #30378/#30427, v0.5.15) and it still recurs even on NVIDIA Blackwell
  (issues #25704/#27384 on B300; #22757 GLM5/V3.2 DSA + dp-attention + EAGLE segfault on H20/B300).
- But **EAGLE/MTP is officially disabled for AMD on GLM-5/5.1/5.2** — the sglang GLM-5.2 cookbook
  states MTP is off in the AMD deploy panel because "the spec-decode draft kernel is not yet
  validated on gfx950." So MTP+DPA+PD on gfx950 is a combination AMD hasn't validated.
- Not a fundamental hardware limitation, and not an AMD-only bug: it's a general metadata-bookkeeping
  gap that AMD simply hasn't been carried across / re-enabled for.

## Q2: fix cost — dimension bookkeeping, reshape, or missing operator?

**Localized metadata bookkeeping — NOT a missing operator. Cheap-to-medium.**
- The kernel that must run (`fast_topk` / the fused `deepseek_v4_topk_transform_512`) **exists and
  runs on gfx950** — it's exercised in Config A (DPA-only) and Config B (MTP-only), both PASS. Nothing
  is unimplemented at the operator level.
- The bug is purely that, on the PD+IndexShare+DPA path, `lengths` (`dsa_seqlens_expanded`) is built
  over one batch decomposition while `logits` rows are the DP-padded draft-extended count → the two
  disagree. The fix is to make `lengths` cover every (DP-padded) row and clamp padded rows to 0 —
  exactly what the fused v2 path already does and what CUDA PR #30378 did with a one-line clamp.
- Two concrete fix routes (increasing effort):
  1. **Cheapest workaround (no code):** avoid the fused config — run DPA and MTP separately (what we
     did; each passes conc=128), OR drop PD for the MTP case (single-node MTP dodges
     `pd_index_share_seed`), OR disable GLM IndexShare so `seed_dsa_topk_from_draft_extend=False`
     (keeps the DP-safe fused path on) at a small accept-length cost.
  2. **Proper fix (medium):** implement the `TODO(kpham-sgl)` — remap the request-relative IndexShare
     seeds to decode-local KV slots so fused top-k stays enabled under PD, i.e. make the unfused
     `fast_topk_v2` (or its `lengths` producer) DP-padding-aware, mirroring #30378. This is
     dimension/metadata code in `dsa_backend.py`'s `seqlens_expanded` construction + the topk call,
     no new GPU kernel.

## Bottom line
- **It's a localized, fixable bug (metadata/dim bookkeeping), not a missing AMD operator.** The needed
  kernels already run on gfx950.
- **General bug class** (also hits NVIDIA), but you land on it because **MTP+DPA+PD on GLM-5.2 DSA is
  an AMD-unvalidated combo** — upstream disables MTP on AMD for exactly this reason.
- **Fix cost: cheap** if you just avoid the triple-combo (our approach) or toggle IndexShare off;
  **medium** to properly re-enable fused DP-safe top-k under PD (the upstream TODO), which is a
  reshape/bookkeeping change, not a new operator.

## Key upstream references
- PR #30378 / #30427 — CUDA fix "clamp padded-row seq_lens to >=0", fused topk_v2 for MTP (v0.5.15).
- Issue #22757 — GLM5/V3.2 DSA + `--enable-dp-attention` + EAGLE segfault (H20/B300).
- Issue #25704 / #27384 — DSA indexer + DP-attn + topk_v2 crashes (B300).
- Issue #20404 — MI355X DPA + MTP errors (disagg).
- sglang GLM-5.2 cookbook — MTP disabled on the AMD deploy panel (draft kernel unvalidated on gfx950).
- In-image code: `dsa/utils.py:71 should_use_dsa_fused_topk` (the `TODO(kpham-sgl)` gate),
  `dsa_topk_backend.py:_topk_transform_v2_paged` (DP-padding clamp) vs `fast_topk_v2` (un-clamped assert),
  `eagle_worker_v2.py:273 seed_dsa_topk_from_draft_extend`.
