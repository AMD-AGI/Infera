#!/usr/bin/env python3
"""Patch 2b, reimplemented in the shape of upstream PR #32209.

The defect (ours, measured): `metadata.page_table_1` is
`req_to_token[req_pool_indices, :max_seqlen_k]`, so it has one row per
REQUEST, while `topk_indices` has just been widened by `_pad_topk_indices` to
`q.shape[0]`, one row per TOKEN.  Under MTP those differ and
`transform_index_page_table_decode` trips
`assert page_table.shape[0] == topk_indices.shape[0]`.

Our v1 fix (`_glm52_match_page_table_rows`) reconciles them by EXPANDING the
page table with `repeat_interleave`, plus two fallbacks (trim / edge-pad) for
a non-integral ratio that were never observed to fire and never instrumented.
It works, but it grows the smaller tensor to match padding rows that are then
masked out anyway.

PR #32209 goes the other way, and that direction is the defensible one: eager
DP-attention padding is not real work, so TRIM q and top-k down to the rows the
metadata was planned for, run attention on the real prefix, and pad the OUTPUT
back before the following MLP collectives.  In its own words: "Attention has no
DP collective, so it should run only on the real prefix."

  _trim_trtllm_decode_dp_padding(q_all, topk_indices, real_batch_size)
      -> (q_all[:real], topk[:real], num_padding_rows)
  _restore_trtllm_decode_dp_padding(out, num_padding_rows)
      -> cat([out, zeros(num_padding_rows, ...)])

with `real_batch_size = metadata.cache_seqlens_int32.shape[0]`, and asserts
that the real count does not exceed either physical count.

PORTING NOTE -- the site differs.  #32209 patches `_forward_trtllm`, which is
CUDA/flashinfer and not our path; ours is
`forward_decode -> dsa_decode_impl == "tilelang" -> _forward_tilelang`.  The
trim therefore goes into `forward_decode`, right after `_pad_topk_indices` and
before the page-table build -- the same relative position #32209 uses.

The restore is applied by a thin `*args/**kwargs` wrapper around
`forward_decode`.  `forward_decode` returns from roughly a dozen
impl-specific branches; wrapping once is the only edit that covers all of
them, and `*args/**kwargs` means the wrapper cannot drift from the wrapped
signature.

WHAT THIS CHANGES SEMANTICALLY vs v1.  v1 ran attention on the PADDED rows and
relied on the triton kernel's `loaded_topk_indices >= 0` mask to make them
harmless.  This version does not run them at all.  Strictly less work, and it
removes the reliance on the mask -- but the padded rows of the returned tensor
are now ZEROS rather than whatever the masked kernel produced.  #32209 holds
that this is the correct contract (the padding exists only so the following MLP
all-gather sees uniform shapes).  If any consumer actually reads those rows,
this arm is where it will show.

Requires v1's `_glm52_match_page_table_rows` to be ABSENT: the two are
alternatives, not a stack.  The script refuses to run otherwise.

Idempotent.  Anchors asserted to match exactly once.  Invalidates the .pyc.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
MARK = "GLM52_P2BV2"

HELPERS = '''def _p2bv2_trim_decode_dp_padding(q_all, topk_indices, real_batch_size):
    """GLM52_P2BV2: drop eager DP-attention padding before DSA decode.

    Ported from upstream PR #32209 (`_trim_trtllm_decode_dp_padding`).

    Eager DP-attention pads activations to the largest token count across
    ranks, but the DSA metadata (page table, cache_seqlens) was planned before
    that padding, so it carries `real_batch_size` rows while q / top-k carry
    the larger physical count. Attention has no DP collective, so it should run
    only on the real prefix; the output is padded back afterwards.

    Returns (q_all, topk_indices, num_padding_rows).
    """
    physical = q_all.shape[0]
    assert real_batch_size <= physical, (
        f"GLM52_P2BV2: DSA metadata batch size ({real_batch_size}) exceeds q "
        f"batch size ({physical})"
    )
    if topk_indices is not None:
        assert real_batch_size <= topk_indices.shape[0], (
            f"GLM52_P2BV2: DSA metadata batch size ({real_batch_size}) exceeds "
            f"topk batch size ({topk_indices.shape[0]})"
        )
    num_padding_rows = physical - real_batch_size
    if num_padding_rows == 0:
        return q_all, topk_indices, 0
    return (
        q_all[:real_batch_size],
        topk_indices[:real_batch_size] if topk_indices is not None else None,
        num_padding_rows,
    )


def _p2bv2_restore_decode_dp_padding(output, num_padding_rows):
    """GLM52_P2BV2: re-pad the attention output with zero rows (PR #32209)."""
    if num_padding_rows == 0:
        return output
    return torch.cat(
        [output, output.new_zeros((num_padding_rows, *output.shape[1:]))], dim=0
    )


'''

HELPER_ANCHOR = """@dataclass(frozen=True)
class DSAFlashMLAMetadata:"""

# --- the trim, inserted where v1 called _glm52_match_page_table_rows ---------
FD_ANCHOR = """        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])
"""

FD_REPL = '''        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # GLM52_P2BV2: trim eager DP-attention padding down to the rows the DSA
        # metadata was planned for, per upstream PR #32209. Without this,
        # metadata.page_table_1 (one row per REQUEST) and topk_indices (one row
        # per TOKEN after the pad above) disagree under MTP and
        # transform_index_page_table_decode asserts. The output is re-padded by
        # the forward_decode wrapper.
        #
        # q_nope / q_rope are views into q_all when the caller passed a
        # concatenated q; slicing all three keeps them consistent either way.
        #
        # The real row count is `metadata.page_table_1.shape[0]`, NOT
        # `cache_seqlens_int32.shape[0]`. #32209 uses the latter because on its
        # CUDA/trtllm path the two coincide; here they do not. Under MTP,
        # init_forward_metadata expands the page table by
        # `repeat_interleave(..., speculative_num_draft_tokens)` for
        # TARGET_VERIFY / DRAFT_EXTEND_V2, so page_table_1 is per-TOKEN while
        # cache_seqlens_int32 stays per-REQUEST. Trimming to the request count
        # cut bs*num_draft_tokens rows down to bs, which desynchronized the
        # following dp_gather_replicate all-gather:
        #   ValueError: output tensor size must be equal to world_size times
        #   input tensor size            (measured, arm E3, conc=32, 2026-07-30)
        # page_table_1 is also exactly the quantity the assert we are fixing
        # compares topk_indices against, so it is the right target by
        # construction.
        # `dsa_drop_wide_page_table` nulls page_table_1 on some graph paths;
        # there is nothing to reconcile against then, so skip the trim.
        _p2bv2_pad_rows = 0
        if metadata.page_table_1 is not None:
            _p2bv2_real = metadata.page_table_1.shape[0]
            _p2bv2_probe = q_all if q_all is not None else q_nope
            _, topk_indices, _p2bv2_pad_rows = _p2bv2_trim_decode_dp_padding(
                _p2bv2_probe, topk_indices, _p2bv2_real
            )
            if _p2bv2_pad_rows:
                if q_all is not None:
                    q_all = q_all[:_p2bv2_real]
                q_nope = q_nope[:_p2bv2_real]
                q_rope = q_rope[:_p2bv2_real]
        # Published for the wrapper installed around forward_decode; see the
        # module docstring of patch2b_32209_style.py.
        self._p2bv2_pad_rows = _p2bv2_pad_rows
'''

# --- the wrapper -------------------------------------------------------------
FD_DEF = """    def forward_decode(
        self,
        q: torch.Tensor,"""

FD_DEF_REPL = '''    def forward_decode(self, *args, **kwargs) -> torch.Tensor:
        """GLM52_P2BV2: restore the DP padding the inner call trimmed.

        The inner method returns from ~a dozen impl-specific branches, so
        wrapping once here is the only edit that covers all of them.
        `*args/**kwargs` keeps the wrapper from drifting from the wrapped
        signature. `_p2bv2_pad_rows` is reset here and set by the trim inside;
        branches that return before the trim (e.g. the trtllm dispatch) leave
        it at 0 and are passed through untouched.
        """
        self._p2bv2_pad_rows = 0
        out = self._p2bv2_forward_decode_inner(*args, **kwargs)
        return _p2bv2_restore_decode_dp_padding(out, self._p2bv2_pad_rows)

    def _p2bv2_forward_decode_inner(
        self,
        q: torch.Tensor,'''


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = open(TARGET).read()

    if MARK in src:
        print(f"already patched ({MARK} present); nothing to do")
        return
    if "_glm52_match_page_table_rows" in src:
        die("v1 patch 2b (_glm52_match_page_table_rows) is still present; "
            "revert it first -- the two are alternatives, not a stack")

    for name, anchor in (
        ("helper insertion point", HELPER_ANCHOR),
        ("forward_decode body", FD_ANCHOR),
        ("forward_decode def", FD_DEF),
    ):
        n = src.count(anchor)
        if n != 1:
            die(f"{name} anchor matched {n}x, want 1")

    src = src.replace(HELPER_ANCHOR, HELPERS + HELPER_ANCHOR, 1)
    src = src.replace(FD_DEF, FD_DEF_REPL, 1)
    src = src.replace(FD_ANCHOR, FD_REPL, 1)

    open(TARGET, "w").write(src)
    os.utime(TARGET, None)
    pc = os.path.join(os.path.dirname(TARGET), "__pycache__")
    if os.path.isdir(pc):
        for f in os.listdir(pc):
            if f.startswith("dsa_backend."):
                os.remove(os.path.join(pc, f))
    print("patched:", TARGET)
    print("marker count:", src.count(MARK))


if __name__ == "__main__":
    main()
