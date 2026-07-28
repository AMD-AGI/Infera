# TODO — fix GLM-5.2 DSA: DP-attention + MTP + PD-disaggregation crash

**Audience:** a code agent tasked with FIXING this in sglang. Self-contained; read top to bottom.
**Status:** root-caused, not yet fixed. Workaround in place (run DPA and MTP separately).
**Engine:** sglang `0.5.15.post1` (as bundled in `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`).
**Hardware:** AMD MI355X (gfx950), ROCm 7.2.0. **Model:** GLM-5.2-MXFP4 (`GlmMoeDsaForCausalLM`,
DeepSeek-style **DSA** = sparse attention with an indexer selecting top-k=2048 KV per query).

---

## 1. Symptom (how to reproduce the crash)

Bring up a 2-node mooncake PD where the **decode leg** runs BOTH:
- EAGLE MTP: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
- DP-attention: `--dp-size 8 --enable-dp-attention --ep-size 8`
(prefill leg symmetric DP8; both `--disaggregation-mode {prefill,decode} --disaggregation-transfer-backend mooncake`.)

The decode leg loads fine, prints "ready to roll", then crashes on the **first real request**:
```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```
Full traceback (decode leg):
```
eagle_worker_v2.forward_batch_generation
 -> _draft_extend_for_decode                       (speculative/eagle_worker_v2.py)
 -> draft_runner.forward -> deepseek_nextn.forward -> deepseek_v2 self_attn
 -> forward_absorb_prepare -> self.indexer(...)     (models/deepseek_v2.py)
 -> dsa_indexer.forward_cuda -> _get_topk_paged -> metadata.topk_transform  (dsa/dsa_indexer.py)
 -> dsa_topk_backend.topk_transform -> topk_func -> fast_topk_v2            (dsa/dsa_topk_backend.py)
 -> torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths, row_starts) (sgl_kernel/top_k.py:41)
 => RuntimeError: Expected lengths.size(0) == B      (assert inside the compiled fast_topk kernel)
```
Each feature ALONE passes conc=128 (DPA-only ✅, MTP-only ✅). Only the combination crashes.

---

## 2. Root cause (source-confirmed; file:line are for sglang 0.5.15.post1)

It is the intersection of **three** features — PD + GLM MTP-IndexShare + DP-attention — not MTP+DPA alone.

**(a) There are two top-k paths in `srt/layers/attention/dsa/dsa_topk_backend.py`:**
- **FUSED (DP-safe):** `_topk_transform_v2_paged` (line ~237). Its docstring (lines ~264-268) says the
  kernel reads `lengths` as `uint32`, so **DP-padded / idle-companion rows must be clamped to 0**, and
  "Metadata producers clamp padded rows to 0". This is the path that tolerates DP-attention.
- **UNFUSED (NOT DP-safe):** `topk_func` (line 37) → `fast_topk_v2` (`sgl_kernel/top_k.py:16`) →
  `torch.ops.sgl_kernel.fast_topk` (top_k.py:41). The compiled kernel hard-requires
  `lengths.size(0) == score.size(0) (=B)` with **no DP-padding handling** → this is what asserts.
- Dispatch: `DSATopKBackend.topk_transform` (dsa_topk_backend.py:75). It takes the UNFUSED branch when
  `not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk` (line 88).

**(b) What forces `force_unfused_topk=True` here** — `DeepseekSparseAttnBackend.get_indexer_metadata`
(`srt/layers/attention/dsa_backend.py:2862`):
```python
force_unfused = not self.use_fused_topk or (hisparse ... decode)
```
and `self.use_fused_topk = should_use_dsa_fused_topk(server_args, seed_dsa_topk_from_draft_extend)`
(dsa_backend.py:442).

**(c) The gate — `srt/layers/attention/dsa/utils.py:71` `should_use_dsa_fused_topk`:**
```python
pd_index_share_seed = (server_args.disaggregation_mode != "null"
                       and seed_dsa_topk_from_draft_extend)
# TODO(kpham-sgl): Transfer request-relative IndexShare seeds and remap them
# to decode-local KV slots so fused top-k can remain enabled under PD.
return envs.SGLANG_DSA_FUSE_TOPK.get() and not pd_index_share_seed
```
So under **PD** (`disaggregation_mode="decode"`) **with** `seed_dsa_topk_from_draft_extend=True`, the
DP-safe fused top-k is DISABLED → the unfused, non-DP-safe kernel runs → crash. **The upstream
`TODO(kpham-sgl)` on line 77 IS this bug** — the PD IndexShare path was left unfinished.

**(d) Why `seed_dsa_topk_from_draft_extend=True` for GLM-5.2** —
`srt/speculative/eagle_worker_v2.py:266-275`:
```python
self.index_share_for_mtp_iteration = getattr(hf_config, "index_share_for_mtp_iteration", False) and self.topk == 1
...
self.seed_dsa_topk_from_draft_extend = self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
```
GLM-5.2's HF config sets `index_share_for_mtp_iteration=True` (its MTP "IndexShare" optimization:
reuse the DSA indexer top-k from the draft-extend step), and `--speculative-eagle-topk 1` makes
`self.topk==1` → both True.

**(e) The actual shape mismatch:** on the draft-extend path
(`dsa_backend.py:805 is_draft_extend_v2()`), `lengths` = `dsa_seqlens_expanded` built by
`seqlens_expand_triton` (line ~832) over the draft-extended token batch, while under DP-attention the
`score`/`logits` rows are the **DP-padded** row count. The unfused kernel compares the two directly →
`lengths.size(0) != B`.

---

## 3. Is it general or AMD-specific? (context for prioritization)

- **General bug class.** The same `topk_v2` + DP-attn padded-row mismatch was fixed on NVIDIA CUDA
  (sglang PR **#30378** / cherry-pick **#30427**, "clamp padded-row seq_lens to >= 0", shipped v0.5.15)
  and still recurs on NVIDIA Blackwell (issues **#22757** GLM5/V3.2 DSA+dp-attention+EAGLE segfault
  on H20/B300; **#25704**, **#27384**). Related AMD: **#20404** (MI355X DPA+MTP), **#29375**
  (GLM-5.2 DSA MI355X), **#16027** (GLM-4.7 EAGLE draft-extend ROCm).
- **But MTP is officially disabled for AMD on GLM-5/5.1/5.2** (sglang GLM-5.2 cookbook: the AMD deploy
  panel turns MTP off because "the spec-decode draft kernel is not yet validated on gfx950"). So this
  triple-combo is an AMD-unvalidated path; the CUDA-side padded-row fix was never carried across.

---

## 4. Fix cost & guidance

**Localized metadata/dimension bookkeeping — NOT a missing operator.** The required kernels already run
on gfx950 (DPA-only and MTP-only both pass), so nothing needs a new GPU kernel; only the `lengths`↔row
bookkeeping on the PD+IndexShare+DPA path is wrong.

Three fix routes, increasing effort:

1. **Config workaround (already applied, zero code):** don't run the triple-combo. Run DPA *or* MTP
   separately (each passes conc=128); OR run MTP single-node (no PD → `pd_index_share_seed` is False,
   fused path stays on); OR disable GLM IndexShare so `seed_dsa_topk_from_draft_extend=False` (small
   accept-length cost, keeps the DP-safe fused path). Good enough if you don't need MTP+DPA+PD together.

2. **Make the unfused path DP-safe (medium, mirrors CUDA #30378):** where `lengths`
   (`dsa_seqlens_expanded`) is produced for `is_draft_extend_v2()` under DP-attention
   (`dsa_backend.py` ~805-851 + `seqlens_expand_triton`), ensure it covers **every DP-padded row** and
   **clamp padded/idle rows to 0** (same invariant the fused `_topk_transform_v2_paged` relies on), so
   `lengths.size(0) == score.size(0)`. Then the `fast_topk_v2` assert holds. Verify the compiled
   `sgl_kernel.fast_topk` treats a 0-length row as "empty → all -1" (the fused kernel does).

3. **Proper: resolve the `TODO(kpham-sgl)` (medium):** in `should_use_dsa_fused_topk`
   (dsa/utils.py:71), instead of disabling fused top-k under PD, transfer the request-relative
   IndexShare seeds and remap them to decode-local KV slots so the **fused** (already DP-safe) path
   stays enabled under PD. This is the upstream-intended fix and keeps the fast path.

**Recommended:** attempt (2) first (smallest, matches an already-merged CUDA patch); fall back to (3)
if IndexShare-seed remapping is required for correctness. Do NOT write a new kernel.

---

## 5. How to verify a fix

1. Bring up 2-node mooncake PD with DPA8 + MTP(EAGLE steps=3, eagle-topk=1) on the decode leg
   (repro command in this packup's `REPRODUCE.md` §B, set `DPA=1 MTP=1` on decode).
2. Correctness probe must return 4/4 (see `scripts/probe.py`).
3. Spec-dec must be active on decode (`accept len ~2.7-3.0` in the decode log) AND DP flags took
   (`enable_dp_attention=True, dp_size=8, ep_size=8`).
4. conc=128 stress (`scripts/sweep_dpa.sh`, ISL/OSL 1k/1k, 512 prompts) must complete 512/512 with
   0 `KVTransferError` / 0 crash — matching what DPA-only (512/512, 7218 tok/s) and MTP-only
   (512/512, 8990 tok/s) each already achieve. Combined should beat DPA-only throughput (DP scaling)
   while keeping MTP's lower TPOT.

To instrument the mismatch while fixing: print `score.shape[0]` vs `lengths.shape[0]` right before
`torch.ops.sgl_kernel.fast_topk` in `sgl_kernel/top_k.py:41` on the decode leg's first request.

---

## 6. Exact source anchors (sglang 0.5.15.post1)

| file | line | what |
|------|------|------|
| `srt/layers/attention/dsa/utils.py` | 71-79 | `should_use_dsa_fused_topk` — the gate + `TODO(kpham-sgl)` |
| `srt/speculative/eagle_worker_v2.py` | 266-275 | sets `index_share_for_mtp_iteration` / `seed_dsa_topk_from_draft_extend` (True for GLM-5.2 + eagle-topk=1) |
| `srt/layers/attention/dsa_backend.py` | 442 | `self.use_fused_topk = should_use_dsa_fused_topk(...)` |
| `srt/layers/attention/dsa_backend.py` | 2862 | `force_unfused = not self.use_fused_topk or ...` |
| `srt/layers/attention/dsa_backend.py` | 805-851 | draft-extend-v2 `seqlens_expanded` (the `lengths`) construction |
| `srt/layers/attention/dsa/dsa_topk_backend.py` | 88 | dispatch: unfused when `not FUSE_TOPK or force_unfused_topk` |
| `srt/layers/attention/dsa/dsa_topk_backend.py` | 237-303 | `_topk_transform_v2_paged` — the DP-safe FUSED path (clamps padded rows) |
| `sgl_kernel/python/sgl_kernel/top_k.py` | 41 | `torch.ops.sgl_kernel.fast_topk(...)` — the un-clamped `lengths.size(0)==B` kernel |

Upstream refs: PR #30378, #30427 (CUDA fix); issues #22757, #25704, #27384, #20404.
Deeper narrative: this packup's `notes_dpa_mtp_rootcause.md`.
