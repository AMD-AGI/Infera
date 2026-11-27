#!/usr/bin/env python3
"""Build the Triton PREFILL top-k overlay (VPD_TOPK_PREFILL=1; default off = unchanged behaviour) of the two files whose
prefill ends in aiter's one-block-per-row top-k (`radix_topk_one_block_{decode,prefill}_mixed_k2048_b1024`):

  FP8 indexer (the MTP draft layer 78 with S_IDXFP4=1, every layer without it):
      vllm/v1/attention/ops/rocm_aiter_mla_sparse.py, rocm_aiter_sparse_attn_indexer, per prefill chunk:
      rocm_fp8_mqa_logits -> logits [chunk rows, chunk K tokens] fp32, rows bounded by cu_seqlen_ks/ke ->
      aiter top_k_per_row_prefill (max_seqlen_k >= 32K, else vLLM's top_k_per_row_prefill) + _localize.
      The hook runs first: positions relative to cu_seqlen_ks, i.e. what the localized aiter result holds.
  MXFP4 indexer (#61985 backport, target layers):
      vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py, _dense_prefill, per prefill chunk:
      paged_mxfp4_mqa_logits -> logits [chunk rows, widest context] fp32, row r valid on [0, plan.row_ends[r]) ->
      `_topk` -> aiter top_k_per_row_decode (one-block at > 32 rows).  The hook runs instead of that `_topk`.
Both hooks take vpd_triton_topk_prefill.triton_top_k_per_row_prefill (mounted next to vllm, with
vpd_triton_topk_decode.py it imports) only for an uncompressed, candidate-free layer, fp32 unit-stride logits, int32 row
bounds (one per row), int32 unit-stride out of width k, k a power of two <= 2048; otherwise the original code runs.
VPD_TOPK_PREFILL_SCOPE=all|target|draft (layer >= VPD_TOPK_DRAFT_MIN_LAYER, 78) limits the FP8 hook (the MXFP4 path
serves target layers only).  VPD_TOPK_PREFILL_CHECK=N compares every N-th call with torch.topk and logs counters
("[vpd-topk-prefill]" in the prefill log; host sync on those calls).

Chaining: the prefill already mounts the idxfp4 overlay's rocm_paged_mxfp4_indexer_ops.py (S_IDXFP4=1), so MXFP4_ORIG
must be THAT file and MXFP4_OUT is mounted in its place (same destination, never twice).  The prefill mounts no overlay
of ops/rocm_aiter_mla_sparse.py (the decode's v3comm / tritontopk ones are decode-only), so OPS_ORIG is the image's own
file.  Each edit is an exact string that must occur exactly once; the helper is appended at the end of the module.
  make_tritontopk_prefill_overlay.py OPS_ORIG OPS_OUT [MXFP4_ORIG MXFP4_OUT]
"""
import sys

HELPER = '''

# --- vpd: Triton prefill top-k (scripts/make_tritontopk_prefill_overlay.py) ---
import os as _vpd_pf_os

_VPD_TOPK_PREFILL = _vpd_pf_os.environ.get("VPD_TOPK_PREFILL") == "1"
_VPD_TOPK_PREFILL_SCOPE = _vpd_pf_os.environ.get("VPD_TOPK_PREFILL_SCOPE", "all")  # all | target | draft
_VPD_PF_DRAFT_MIN_LAYER = int(_vpd_pf_os.environ.get("VPD_TOPK_DRAFT_MIN_LAYER", "78"))
if _VPD_TOPK_PREFILL:
    import vpd_triton_topk_prefill as _vpd_tkp


def _vpd_pf_scope_ok(prefix) -> bool:
    if _VPD_TOPK_PREFILL_SCOPE == "all":
        return True
    import re as _re

    m = _re.search(r"layers\\.(\\d+)\\.", str(prefix))
    draft = m is not None and int(m.group(1)) >= _VPD_PF_DRAFT_MIN_LAYER
    return draft if _VPD_TOPK_PREFILL_SCOPE == "draft" else not draft


def _vpd_prefill_topk(
    logits, row_starts, row_ends, out, topk_tokens, compress_ratio, candidates, prefix, where
) -> bool:
    """The prefill top-k of one chunk through the Triton kernel (VPD_TOPK_PREFILL=1). True: done (out holds
    positions relative to the row start, ascending, -1 padded). False: run the original code."""
    if not _VPD_TOPK_PREFILL or compress_ratio != 1 or candidates is not None:
        return False
    rows = logits.shape[0]
    if not (
        _vpd_tkp.supported(logits, topk_tokens)
        and out.dtype == torch.int32
        and out.stride(1) == 1
        and out.shape[0] == rows
        and out.shape[1] == topk_tokens
        and row_ends.dtype == torch.int32
        and row_ends.numel() == rows
        and row_ends.is_contiguous()
        and (
            row_starts is None
            or (
                row_starts.dtype == torch.int32
                and row_starts.numel() == rows
                and row_starts.is_contiguous()
            )
        )
        and _vpd_pf_scope_ok(prefix)
    ):
        return False
    _vpd_tkp.hook_call(logits, row_starts, row_ends, out, topk_tokens, where)
    return True
'''

FP8_OLD = (
    "            num_rows = logits.shape[0]\n"
    "\n"
    "            if rocm_aiter_ops.is_indexer_top_k_supported(\n"
    "                is_prefill=True,\n"
)
FP8_NEW = (
    "            num_rows = logits.shape[0]\n"
    "\n"
    "            if _vpd_prefill_topk(\n"
    "                logits,\n"
    "                chunk.cu_seqlen_ks,\n"
    "                chunk.cu_seqlen_ke,\n"
    "                topk_indices,\n"
    "                topk_tokens,\n"
    "                compress_ratio,\n"
    "                candidate_blocks,\n"
    "                k_cache_prefix,\n"
    "                \"fp8\",\n"
    "            ):\n"
    "                pass\n"
    "            elif rocm_aiter_ops.is_indexer_top_k_supported(\n"
    "                is_prefill=True,\n"
)

MXFP4_OLD = (
    "    _topk(\n"
    "        logits,\n"
    "        plan.row_ends,\n"
    "        layer.topk_buffer[t0:t1, : layer.topk_tokens],\n"
    "        layer.topk_tokens,\n"
    "    )\n"
)
MXFP4_NEW = (
    "    if _vpd_prefill_topk(\n"
    "        logits,\n"
    "        None,\n"
    "        plan.row_ends,\n"
    "        layer.topk_buffer[t0:t1, : layer.topk_tokens],\n"
    "        layer.topk_tokens,\n"
    "        layer.compress_ratio,\n"
    "        layer.candidates,\n"
    "        \"\",\n"
    "        \"mxfp4\",\n"
    "    ):\n"
    "        return\n"
    + MXFP4_OLD
)

# Markers that tell the inputs apart (a wrong file order fails loudly instead of patching the wrong module).
FP8_MARK = "def rocm_aiter_sparse_attn_indexer(\n"
MXFP4_MARK = "def _dense_prefill(\n"
MXFP4_IDX_MARK = "def _vpd_decode_topk(\n"  # the idxfp4 overlay's decode hook: MXFP4_ORIG must be its output


def patch(src: str, out: str, edits, mark: str, extra_mark: str | None = None) -> None:
    s = open(src).read()
    for m in (mark, extra_mark):
        if m is not None and s.count(m) != 1:
            sys.exit(f"{src}: expected exactly one {m.strip()!r}, found {s.count(m)} (wrong input file?)")
    if "_vpd_prefill_topk" in s:
        sys.exit(f"{src} already has the prefill top-k hook")
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"expected exactly one {old.strip()[:70]!r} in {src}, found {n}")
        s = s.replace(old, new)
    s = s.rstrip("\n") + "\n" + HELPER
    open(out, "w").write(s)
    print(f"wrote {out} (from {src})")


def main():
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    patch(sys.argv[1], sys.argv[2], [(FP8_OLD, FP8_NEW)], FP8_MARK)
    if len(sys.argv) == 5:
        patch(sys.argv[3], sys.argv[4], [(MXFP4_OLD, MXFP4_NEW)], MXFP4_MARK, MXFP4_IDX_MARK)


if __name__ == "__main__":
    main()
