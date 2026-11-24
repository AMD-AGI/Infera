#!/usr/bin/env python3
"""Build the Triton decode top-k overlay of vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (v0.34.0).

GLM-5.2's DSA indexer (no compression, k = 2048) runs aiter's one-block-per-row `top_k_per_row_decode` on the decode;
in P4D4 AgentX a rank's longest request is often 500K-800K tokens and the call takes as long as its longest row (a8
trace: 143.6 us, 22 calls per step). With VPD_TRITON_TOPK=1 the decode top-k uses the sglang #47872 single-pass port
(vpd_triton_topk_decode.py, mounted next to vllm: est per row, ONE chunked pass over all rows on a persistent grid,
exact select), when it applies: uncompressed indexer, fp32 logits with unit column stride, <= 128 rows, k a power of
two <= 2048, per-row lengths given (seq_lens has one entry per logits row, the same lengths aiter is given). Otherwise
the original choice. VPD_TOPK_IMPL=native instead always runs vLLM's in-tree torch.ops._C.top_k_per_row_decode
(exact, deterministic; faster than aiter's one-block kernel on long rows in bench_topk_prod.py). VPD_TOPK_SCOPE=target|draft
limits the Triton path to the target model's / the MTP draft layer's calls (layer >= VPD_TOPK_DRAFT_MIN_LAYER, 78).
VPD_TOPK_SHADOW=1 also compares every Triton call with vLLM's in-tree top-k and aiter's (vpd_triton_topk_decode.shadow_compare;
counters in the decode log). The P4D4 decode (D_MODE=v3comm) already mounts an overlay of this file (U12 gluon paged logits),
so ORIG must be that overlay ($R/overlay-v3comm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py), mounted in its place.
  make_tritontopk_overlay.py ORIG OUT
"""
import sys

HEAD_OLD = "import torch\n"
HEAD_NEW = (
    "import os as _vpd_os\n"
    "\n"
    "import torch\n"
    "\n"
    "_VPD_TOPK_IMPL = _vpd_os.environ.get(\"VPD_TOPK_IMPL\") or (\n"
    "    \"triton\" if _vpd_os.environ.get(\"VPD_TRITON_TOPK\") == \"1\" else \"\"\n"
    ")\n"
    "_VPD_TRITON_TOPK = _VPD_TOPK_IMPL == \"triton\"\n"
    "_VPD_TOPK_SCOPE = _vpd_os.environ.get(\"VPD_TOPK_SCOPE\", \"all\")  # all | target | draft\n"
    "_VPD_DRAFT_MIN_LAYER = int(_vpd_os.environ.get(\"VPD_TOPK_DRAFT_MIN_LAYER\", \"78\"))\n"
    "_VPD_TOPK_SHADOW = _vpd_os.environ.get(\"VPD_TOPK_SHADOW\") == \"1\"\n"
    "\n"
    "\n"
    "def _vpd_is_draft(prefix) -> bool:\n"
    "    import re as _re\n"
    "\n"
    "    m = _re.search(r\"layers\\.(\\d+)\\.\", str(prefix))\n"
    "    return m is not None and int(m.group(1)) >= _VPD_DRAFT_MIN_LAYER\n"
    "\n"
    "\n"
    "def _vpd_scope_ok(prefix) -> bool:\n"
    "    if _VPD_TOPK_SCOPE == \"all\":\n"
    "        return True\n"
    "    draft = _vpd_is_draft(prefix)\n"
    "    return draft if _VPD_TOPK_SCOPE == \"draft\" else not draft\n"
    "\n"
    "\n"
    "if _VPD_TRITON_TOPK:\n"
    "    import vpd_triton_topk_decode as _vpd_tk\n"
)

CALL_OLD = (
    "        if rocm_aiter_ops.is_indexer_top_k_supported(\n"
    "            is_prefill=False,\n"
)
CALL_NEW = (
    "        _vpd_lens = decode_metadata.seq_lens.reshape(-1)\n"
    "        if (\n"
    "            _VPD_TRITON_TOPK\n"
    "            and compress_ratio == 1\n"
    "            and _vpd_lens.numel() == num_rows\n"
    "            and _vpd_lens.dtype == torch.int32\n"
    "            and topk_indices.dtype == torch.int32\n"
    "            and topk_indices.stride(1) == 1\n"
    "            and _vpd_tk.supported(logits, topk_tokens)\n"
    "            and _vpd_scope_ok(k_cache_prefix)\n"
    "        ):\n"
    "            _vpd_tk.triton_top_k_per_row_decode(\n"
    "                logits, _vpd_lens, topk_indices, topk_tokens\n"
    "            )\n"
    "            if _VPD_TOPK_SHADOW:\n"
    "                _vpd_tk.shadow_compare(\n"
    "                    logits,\n"
    "                    _vpd_lens,\n"
    "                    topk_indices,\n"
    "                    topk_tokens,\n"
    "                    _vpd_is_draft(k_cache_prefix),\n"
    "                    lambda _o: torch.ops._C.top_k_per_row_decode(\n"
    "                        logits, next_n, decode_metadata.seq_lens, _o, num_rows,\n"
    "                        logits.stride(0), logits.stride(1), topk_tokens,\n"
    "                    ),\n"
    "                    (\n"
    "                        lambda _o: rocm_aiter_ops.indexer_top_k_decode(\n"
    "                            logits, 1, decode_metadata.seq_lens.reshape(-1), _o, topk_tokens\n"
    "                        )\n"
    "                    )\n"
    "                    if rocm_aiter_ops.is_indexer_top_k_supported(\n"
    "                        is_prefill=False,\n"
    "                        compress_ratio=compress_ratio,\n"
    "                        num_rows=num_rows,\n"
    "                        max_valid_seq_len=max_compressed_seq_len,\n"
    "                        topk_tokens=topk_tokens,\n"
    "                    )\n"
    "                    and not rocm_aiter_ops.dsv4_indexer_prefers_native_top_k(\n"
    "                        num_rows=num_rows,\n"
    "                        num_columns=logits.shape[1],\n"
    "                        topk_tokens=topk_tokens,\n"
    "                    )\n"
    "                    else None,\n"
    "                )\n"
    "        elif _VPD_TOPK_IMPL != \"native\" and rocm_aiter_ops.is_indexer_top_k_supported(\n"
    "            is_prefill=False,\n"
)


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    for old, new in ((HEAD_OLD, HEAD_NEW), (CALL_OLD, CALL_NEW)):
        n = s.count(old)
        if n != 1:
            sys.exit(f"expected exactly one {old.strip()[:60]!r} in {src}, found {n}")
        s = s.replace(old, new)
    open(out, "w").write(s)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
