# sglang DSA patches

Patches against the sglang source tree bundled in the ROCm engine images
(`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` and derivatives, where sglang is an
editable checkout at `/sgl-workspace/sglang`).

## `dsa_indexer_hip_dp_padded_rows.diff`

**Enables DP-attention and EAGLE MTP to run together for GLM-5.2 DSA on gfx950.**
Without it, the combination crashes on the first batch:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

The DSA indexer's aiter/HIP paged-MQA path sizes its `logits` output from the
**DP-padded** row count, while the CUDA path slices to the real row count (`q_offset`) —
which is what `lengths` is sized to. The patch makes HIP follow the CUDA contract: run
top-k over the real rows, then restore the padding. Two guards keep the non-DP path
bit-identical. Full rationale is in the patch header.

Applies to `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, exact context, no
fuzz:

```bash
cd /sgl-workspace/sglang
patch -p1 --fuzz=0 < dsa_indexer_hip_dp_padded_rows.diff
```

Or bind-mount the patched file over the container's copy.

### Verified

8× MI355X (gfx950), ROCm 7.2.0, sglang 0.5.15.post1, GLM-5.2-MXFP4, single node
`--dp-size 8 --enable-dp-attention --ep-size 8` + `--speculative-algorithm EAGLE
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`:

| Check | Result |
|---|---|
| Correctness probe | 4/4 |
| Spec-dec accept length | median 3.86 / 4 (n=251) |
| conc=64, ISL/OSL 1k/1k, 256 prompts | 256/256, 0 failed, median TPOT 17.2 ms |
| Regression: DP-attention only | 4/4, unchanged |
| Regression: MTP only | 4/4, unchanged |

Full reproduction kit (environment, commands, raw logs, evidence, root-cause notes) lives
in the `infera.yihou.glm5.2.mxfp4` workspace under
`glm52.mxfp4.spur.mooncake.packup_20260728/dpa_mtp_fix/`.

> **Updated 2026-07-29.** This diff was revised: its slice guard was
> `if 0 < q_offset < q_fp8.shape[0]`, and the `0 <` lower bound made the slice a
> no-op on DP-**idle** ranks (where `q_offset == 0` because the rank holds no
> requests), reproducing the very same assert on those ranks. It only surfaces
> once some ranks are idle while others are busy, i.e. under concurrency, which
> the original single-node validation never exercised. The bound is now removed.
> `q_fp8[:0]` is a legal empty slice and is what an idle rank should pass; the
> CUDA path has no such bound either.

## `dsa_backend_dp_sync_and_page_table_rows.diff`

Two fixes to `python/sglang/srt/layers/attention/dsa_backend.py`, both required
for **PD disaggregation with MTP on the decode leg**.

**1 — rank-divergent GPU→CPU syncs (fixes the PD+MTP deadlock).**
`max_seqlen_k = int(forward_batch.seq_lens.max().item())` is a host sync sitting on a
branch that only *some* DP ranks take; idle peers take the cheap arm, so the DP
collectives desynchronize and the group deadlocks. Replaced with the sync-free
`self.req_to_token.shape[1]`, the same idiom already used by
`_graph_page_table_width()` and the graph-capture paths. Over-allocating columns is
safe: the page table is only indexed through top-k, which masks per row by
`cache_seqlens`. Two further unconditional `.cpu()` syncs are removed.

**2 — page-table rows vs top-k rows (fixes a crash under concurrency).**
`metadata.page_table_1` is `req_to_token[req_pool_indices]`, one row per **request**,
while `topk_indices` has just been padded to `q.shape[0]`, one row per **token**. Those
are equal for plain decode but not under MTP, where the draft model runs several tokens
per request — so `transform_index_page_table_decode_fast` trips
`assert page_table.shape[0] == topk_indices.shape[0]` and every rank dies:

```
deepseek_nextn.py:271 -> dsa_backend.py:2154 forward_decode
  -> dsa/transform_index.py:138  assert page_table.shape[0] == topk_indices.shape[0]
```

A helper expands the page table to token rows via `repeat_interleave`. Rows added by
`_pad_topk_indices` hold all `-1` and are masked out by the triton kernel
(`valid_topk_mask = mask & (loaded_topk_indices >= 0)`), so only the row *count* has to
match. Applied at **both** decode call sites — the traceback named only the first, but
the second pairs the same unpadded page table with a padded `topk_indices`. The prefill
sibling already solves this with an explicit `output_num_tokens` argument; the decode
entry point had no equivalent.

## `deepseek_nextn_glm52_mtp_bf16.diff`

One line in `python/sglang/srt/models/deepseek_nextn.py`. **Required for GLM-5.2 MTP**,
whose nextn layer is bf16 while the rest of the model is quantized — without it the
weight load dies with a `3072 vs 6144` shape mismatch.

## Applying

All three apply with exact context, no fuzz, against sglang commit
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (v0.5.15.post1):

```bash
cd /sgl-workspace/sglang
for d in dsa_indexer_hip_dp_padded_rows.diff \
         dsa_backend_dp_sync_and_page_table_rows.diff \
         deepseek_nextn_glm52_mtp_bf16.diff; do
  git apply --check "$d" && git apply "$d"
done
```

Verified with `git apply --check` against a pristine checkout.

## PD + MTP status (2026-07-29)

The deadlock described as a known limitation in the original version of this file is
**fixed** by the `dsa_backend` sync change above. Current state on 2× 8×MI355X, GLM-5.2-MXFP4,
`--dp-size 8 --enable-dp-attention --ep-size 8`, EAGLE MTP on the decode leg, mooncake
RDMA over mlx5 + dma-buf:

| Check | Result |
|---|---|
| PD warmup (8 concurrent, one per DP rank) | passes in ~10 s |
| conc=128 × 512 tokens, 5 rounds | **640/640 HTTP 200** |
| Scheduler exceptions / KVTransferError | **0 / 0** |
| Spec-dec | active, mean accept length 2.73, all 8 `dp_rank`s serving |

**Two caveats, both outside these three diffs.**

1. This was measured with the EAGLE **draft** CUDA graph disabled. The graph/eager
   decision in `eagle_worker_v2.py::draft()` is rank-divergent under DP-attention
   (`not forward_batch.forward_mode.is_idle()` differs per rank), so some ranks replay a
   graph while others run eager within the same collective step, and the group hangs.
   Disabling that one graph avoids it; a proper fix — making the decision uniform across
   the DP group — is not written yet, so no `eagle_worker_v2` patch is included here.
2. About **2 % of responses under concurrency are degenerate** (`1.1.1.1...`): HTTP 200,
   correct token count, wrong content. Root cause not established. Related and separate:
   `eagle_utils.py:620` forces `argmax` on HIP (`or _is_hip`), so user-supplied
   `temperature`/`top_p` are silently discarded during MTP verify.

See `RESULTS_20260729.md` in the reproduction kit for the measurements behind both.

### Upstream

This is ROCm-specific; the CUDA path was always correct. It is **not** the same issue as
sglang PR #30378 / #30427 ("clamp padded-row seq_lens to >= 0" in
`triton_ops/pad.py::seqlens_expand_kernel`), which fixes padded row *values* and is
already present in this image. This one fixes the HIP-side row *count*.
