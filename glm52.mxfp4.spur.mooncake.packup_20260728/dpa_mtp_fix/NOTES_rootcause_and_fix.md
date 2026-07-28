# FIX — GLM-5.2 DSA "DP-attention + MTP" crash on gfx950 (Bug 1)

**Status:** fixed and verified on single-node mix. Engine sglang `0.5.15.post1`
(image `infera.yihou.sglang.1.0`), MI355X gfx950, ROCm 7.2.0, GLM-5.2-MXFP4.

Patch: `patches/dsa_indexer_hip_dp_rows.patch` (applier: `patches/apply_fix.py`,
drop-in file: `patches/dsa_indexer.patched.py`).

---

## 1. The crash

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```
Reproduced on a **single node with NO PD** (`DPA=1 MTP=1`, `scripts/mix_leg.sh`):

```
eagle_worker_v2.py:1267 forward_batch_generation
 -> eagle_worker_v2.py:965  _draft_extend_for_decode -> draft_runner.forward
 -> eager_runner.py:409     _execute_idle                      <-- IDLE batch
 -> deepseek_nextn.py:271 -> deepseek_v2.py:1856 -> forward_mla.py:413 self.indexer(...)
 -> dsa_indexer.py:1911 forward_cuda -> dsa_indexer.py:963 _get_topk_paged
 -> dsa_backend.py:314 topk_transform -> dsa_topk_backend.py:130 fast_topk_transform_fused
 -> sgl_kernel/top_k.py:77  torch.ops.sgl_kernel.fast_topk_transform_fused  => assert
```

## 2. What the previous analysis got wrong

`glm5.2_dpa_mtp.todo.md` and `notes_dpa_mtp_rootcause.md` were written from source
reading alone. Running it corrected three claims (and confirmed a fourth):

| Earlier claim | Reality (measured) |
|---|---|
| Needs the **PD + IndexShare + DPA** triple to trigger | **False.** Single node, `disaggregation_mode=null`, still crashes. PD is irrelevant to the defect. |
| Crashes in the **unfused** `fast_topk_v2` | **False.** Crashes in the **fused** `fast_topk_transform_fused` (`dsa_topk_backend.py:130`). Both variants carry the same row-count check; which one runs doesn't matter. |
| Fix = clamp DP-padded rows to 0 in `seqlens_expand_triton` (mirror CUDA PR #30378) | **Already present.** `triton_ops/pad.py:355` has `values = tl.maximum(start + offs, 0)` with an explicit DP-padding comment. Not the gap. |
| Trigger is the draft-extend path | **Confirmed.** `DRAFT_EXTEND_V2` accounts for 26 of the 27 distinct padded shapes observed; `IDLE` is the extreme case. `TARGET_VERIFY` is never padded. |

> Note on method: an earlier revision of this file claimed the trigger was **only**
> `IDLE`. That came from a boot-time-only sample of the debug log, before any load had
> been driven. The full conc=64 dataset (`evidence/dsa_rows_measured.txt`) shows
> `DRAFT_EXTEND_V2` dominating. Corrected here — sample under load before concluding.

## 3. Actual root cause — a HIP-only row-count asymmetry

`dsa_indexer.py::_get_topk_paged` builds the indexer logits then runs top-k. `q_offset`
= `sum(metadata.get_dsa_extend_len_cpu())` is the **real (unpadded)** row count, and is
exactly what `lengths` (`dsa_seqlens_expanded`) is sized to. The two backends disagree:

* **CUDA** — `deepgemm_paged_mqa_logits_split` (`jit_kernel/dsa/paged_mqa_logits.py:38`)
  slices its inputs: `q_fp8[:q_offset]`, `weights[:q_offset]`. So
  `logits.shape[0] == q_offset == lengths.shape[0]`. It then re-pads the top-k result
  back to `q_fp8.shape[0]`.
* **aiter / HIP** — `aiter_paged_mqa_logits` (same file, :63) does **not** slice. It
  allocates `logits = torch.empty((batch_size * next_n, max_seq_len))` sized from
  `q_fp8.shape[0]`, i.e. the **DP-padded** row count. And the re-pad afterwards was
  gated `if not _is_hip and ...`, i.e. disabled on HIP — self-consistent with never
  having shrunk, but it means HIP had no path that produced matching shapes.

Under DP-attention the hidden states are padded to a common per-rank token count, so
`q_fp8.shape[0] > q_offset` and the top-k sees `score.shape[0] != lengths.shape[0]`.

**Measured proof** (from `SGLANG_DEBUG_DSA_ROWS=1`, added by the patch; counts are
occurrences over a full 4/4 probe + conc=64 / 256-prompt run — full table in
`evidence/dsa_rows_measured.txt`):

```
 902  mode=ForwardMode.DRAFT_EXTEND_V2  q_fp8=(36, 32, 128)  q_offset=32  lengths=(32,)  -> mqa_q=(32, ...)   PADDED
 602  mode=ForwardMode.DRAFT_EXTEND_V2  q_fp8=(40, 32, 128)  q_offset=32  lengths=(32,)  -> mqa_q=(32, ...)   PADDED
 555  mode=ForwardMode.DRAFT_EXTEND_V2  q_fp8=(40, 32, 128)  q_offset=28  lengths=(28,)  -> mqa_q=(28, ...)   PADDED
 159  mode=ForwardMode.IDLE             q_fp8=(4,  32, 128)  q_offset=1   lengths=(1,)   -> mqa_q=(1,  ...)   PADDED
4830  mode=ForwardMode.DRAFT_EXTEND_V2  q_fp8=(32, 32, 128)  q_offset=32  lengths=(32,)  -> mqa_q=(32, ...)   not padded
 504  mode=ForwardMode.TARGET_VERIFY    q_fp8=(96, 32, 128)  q_offset=96  lengths=(96,)  -> mqa_q=(96, ...)   not padded
```

Breakdown by mode: **27 distinct padded shapes — 26 `DRAFT_EXTEND_V2`, 1 `IDLE`**;
31 unpadded shapes (11 `DRAFT_EXTEND_V2`, 20 `TARGET_VERIFY`). So:

* `DRAFT_EXTEND_V2` is the workhorse trigger. Its row count is
  `bs * speculative_num_draft_tokens`, but under DP-attention the hidden states are padded
  to the DP group's common token width, so `q_fp8` gains extra rows (e.g. 36 or 40 rows for
  32 real ones) while `lengths` stays at the true count.
* `IDLE` is the extreme case (4 padded rows vs 1 real): a DP rank with no real request in
  the MTP draft-extend step still runs an idle-companion batch padded to the group width.
* `TARGET_VERIFY` is never padded (`q_offset == q_fp8.shape[0]` in all 20 shapes), which is
  why the verify step alone never tripped the assert.

Why each config behaves as observed:
- **DPA-only passes** — plain decode has one query row per request, `q_offset == B`.
- **MTP-only passes** — no DP padding, `q_fp8.shape[0] == q_offset`.
- **Only fused crashes** — MTP creates the idle draft-extend batch *and* DP pads it.

## 4. The fix

Make HIP obey the same contract as CUDA (`patches/dsa_indexer_hip_dp_rows.patch`):

1. Slice the aiter MQA-logits inputs to the real rows when padding is present:
   `if 0 < q_offset < q_fp8.shape[0]: _q_mqa, _w_mqa = q_fp8[:q_offset], weights[:q_offset]`
2. Let the padding-restore fire on HIP too, guarded on what actually happened:
   `if q_offset < q_fp8.shape[0] and topk_result.shape[0] == q_offset:`
3. Add an opt-in `SGLANG_DEBUG_DSA_ROWS=1` log of the row bookkeeping.

No new kernel; ~5 functional lines. The non-DP path is untouched (both guards are false
when nothing was padded), so the previously-passing configs are bit-identical.

## 5. Verification (single-node mix, node crsuse2-m2m-207)

| Check | Before | After |
|---|---|---|
| DPA8 + MTP boot + first request | crash `lengths.size(0) == B` | **ready to roll**, no crash |
| Correctness probe | n/a (crashed) | **4/4** |
| DP flags took | — | `enable_dp_attention=True, dp_size=8, ep_size=8` |
| Spec-dec active | — | **accept len median 3.86** of 4 (n=251, min 2.35, max 4.00) |
| conc=64, ISL/OSL 1k/1k, 256 prompts | n/a | **256/256, 0 failed**; 6365 tok/s total, 3183 tok/s out, median TPOT **17.2 ms**, median TTFT 864 ms |
| Regression: DPA-only (`DPA=1 MTP=0`) | 4/4 | **4/4** (unregressed) |
| Regression: MTP-only (`DPA=0 MTP=1`) | 4/4 | **4/4**, accept len 2.50-3.52 n=4 (unregressed) |

Median TPOT 17.2 ms with DPA+MTP fused beats both prior single-feature PD numbers
(DPA-only 31.3 ms, MTP-only 19.2 ms) — the speculative decode and DP scaling compose.

## 6. Two operational traps hit while validating (not defects in the fix)

**(a) `pkill -f launch_server` orphans the scheduler tree and leaks GPU memory.**
Killing the launcher leaves `sglang::scheduler_DP*` children defunct-but-resident; VRAM
stays at ~82% and the next server can't allocate, so it wedges partway through boot with
all 8 ranks alive but the detokenizer never heartbeating. `pkill -9 -f sglang` does not
reliably reap them either. **Do:** `docker rm -f <ctr>` and recreate the container
between config changes — VRAM then reads 0% and boot is clean. Verify with
`rocm-smi --showmemuse` before relaunching.

**(b) Cold Inductor cache + 8 DP ranks can deadlock the PD decode warmup.**
On a freshly recreated container (empty `/tmp/torchinductor_root`) the 8 DP schedulers
each spawn their own Inductor subproc pool — 264 compile workers on 236 cores. The ranks
then diverge inside `@torch.compile`'d `select_top_k_tokens` (`spec_utils.py:274`, the
EAGLE draft path):

```
DP0-2: synchronize (torch/cuda/__init__.py:1083)   <- inside Inductor _make_launchers
DP3-7: broadcast   (torch/distributed/...:2841)    <- already at the collective
```

DP0-2 never reach the collective, DP3-7 never leave it → warmup hangs forever with GPUs
at 100% and zero new `.so` artifacts. **This is unrelated to the DSA fix** (the stalled
frame is the EAGLE top-k helper, not the indexer). **Do:** cap Inductor parallelism
(`TORCHINDUCTOR_COMPILE_THREADS=4`) on a cold cache, or pre-warm the cache. Diagnose with
`py-spy dump --pid <scheduler pid>` across all ranks — a clean split between
`synchronize` and `broadcast` is the signature.

## 7. Upstream relevance

The defect is **ROCm-specific** (the CUDA path was always correct), so it is *not*
covered by PR #30378/#30427, which fixed the CUDA-side padded-row *values*
(`seqlens_expand_kernel` clamp) rather than the HIP-side row *count*. It is a genuine
gap in the aiter paged-MQA integration and is worth filing upstream.

The `TODO(kpham-sgl)` in `dsa/utils.py::should_use_dsa_fused_topk` (fused top-k disabled
under PD + IndexShare) is a **separate, performance-only** issue and is untouched here.
