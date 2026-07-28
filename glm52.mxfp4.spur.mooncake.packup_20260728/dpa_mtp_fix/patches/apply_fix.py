#!/usr/bin/env python3
"""Patch sglang's DSA indexer so the HIP/aiter paged-MQA path obeys the same
(unpadded) row-count contract as the CUDA path.

Bug: under DP-attention the hidden states are padded, so q_fp8.shape[0] is the
DP-padded row count while `lengths` (dsa_seqlens_expanded) is the real one.
  * CUDA (deepgemm_paged_mqa_logits_split) slices q_fp8[:q_offset] / weights[:q_offset],
    so logits.shape[0] == q_offset == lengths.shape[0]; it then re-pads the topk result.
  * aiter/HIP does NOT slice: it allocates logits with q_fp8.shape[0] rows, and the
    re-pad is gated off by `not _is_hip`.
=> on gfx950, top-k sees score.shape[0] != lengths.shape[0]:
   "RuntimeError: Expected lengths.size(0) == B to be true, but got false."

Fix: slice the aiter inputs to the real rows (mirroring CUDA), and make the
padding-restore fire whenever the result actually came back short.

Idempotent; verifies each anchor matches exactly once.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

OLD_AITER = """        if self.paged_mqa_logits_backend.is_aiter():
            logits = aiter_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32,
                block_tables,
                max_seq_len,
                preshuffle=_use_aiter_preshuffle,
                kv_block_size=block_kv,
            )
"""

NEW_AITER = """        if self.paged_mqa_logits_backend.is_aiter():
            # DP-attention pads the hidden states, so q_fp8 can carry more rows than
            # the batch really has; `q_offset` is the real (unpadded) count and is
            # what `lengths` (dsa_seqlens_expanded) is sized to. The CUDA path
            # (deepgemm_paged_mqa_logits_split) slices its q/weights to q_offset for
            # exactly this reason. aiter instead sizes its `logits` output from
            # q_fp8.shape[0], so without the same slice the top-k below sees
            # score.shape[0] != lengths.shape[0] and asserts. Slice here so both
            # backends feed top-k the same shape; the padding is restored after.
            _q_mqa, _w_mqa = q_fp8, weights
            if 0 < q_offset < q_fp8.shape[0]:
                _q_mqa = q_fp8[:q_offset]
                _w_mqa = weights[:q_offset]
            if _DSA_DEBUG_ROWS:
                logger.info(
                    "[dsa-rows] mode=%s q_fp8=%s q_offset=%s lengths=%s -> mqa_q=%s",
                    forward_batch.forward_mode,
                    tuple(q_fp8.shape),
                    q_offset,
                    tuple(metadata.get_seqlens_expanded().shape),
                    tuple(_q_mqa.shape),
                )
            logits = aiter_paged_mqa_logits(
                _q_mqa,
                kv_cache_fp8,
                weights=_w_mqa,
                seq_lens=seqlens_32,
                block_tables=block_tables,
                max_seq_len=max_seq_len,
                preshuffle=_use_aiter_preshuffle,
                kv_block_size=block_kv,
            )
"""

OLD_RESTORE = """        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
"""

NEW_RESTORE = """        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        # Restore possible padding exist in the hidden states.
        # Fire whenever the top-k actually ran on the unpadded rows -- true on CUDA
        # (which always slices) and now on HIP/aiter too (sliced just above). The
        # `topk_result.shape[0] == q_offset` guard keeps the non-DP path (where
        # nothing was sliced) bit-identical to before.
        if q_offset < q_fp8.shape[0] and topk_result.shape[0] == q_offset:
"""

# aiter_paged_mqa_logits takes weights/seq_lens/block_tables/max_seq_len positionally
# in this build; keep the call positional to avoid a TypeError on older signatures.
NEW_AITER = NEW_AITER.replace("weights=_w_mqa,", "_w_mqa,")
NEW_AITER = NEW_AITER.replace("seq_lens=seqlens_32,", "seqlens_32,")
NEW_AITER = NEW_AITER.replace("block_tables=block_tables,", "block_tables,")
NEW_AITER = NEW_AITER.replace("max_seq_len=max_seq_len,", "max_seq_len,")

DEBUG_FLAG = """_is_hip = is_hip()"""
DEBUG_FLAG_NEW = """_is_hip = is_hip()
# Set SGLANG_DEBUG_DSA_ROWS=1 to log the indexer row bookkeeping (padded vs real).
_DSA_DEBUG_ROWS = os.environ.get("SGLANG_DEBUG_DSA_ROWS", "0") == "1"
"""


def sub_once(src: str, old: str, new: str, name: str) -> str:
    n = src.count(old)
    if n == 0:
        if new.split("\n")[0] in src or "[dsa-rows]" in src:
            print(f"  {name}: already patched, skipping")
            return src
        print(f"ERROR: anchor {name!r} not found", file=sys.stderr)
        sys.exit(1)
    if n != 1:
        print(f"ERROR: anchor {name!r} matched {n} times (want 1)", file=sys.stderr)
        sys.exit(1)
    print(f"  {name}: patched")
    return src.replace(old, new)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else TARGET
    with open(path) as f:
        src = f.read()

    if "[dsa-rows]" in src:
        print("Already patched; nothing to do.")
        return

    orig = src
    if "import os" not in src.split("\n\n")[0] and "\nimport os\n" not in src:
        src = src.replace("import torch\n", "import os\n\nimport torch\n", 1)
        print("  added `import os`")
    src = sub_once(src, DEBUG_FLAG, DEBUG_FLAG_NEW, "debug flag")
    src = sub_once(src, OLD_AITER, NEW_AITER, "aiter input slice")
    src = sub_once(src, OLD_RESTORE, NEW_RESTORE, "padding restore gate")

    with open(path + ".orig", "w") as f:
        f.write(orig)
    with open(path, "w") as f:
        f.write(src)
    print(f"OK: patched {path} (backup at {path}.orig)")


if __name__ == "__main__":
    main()
