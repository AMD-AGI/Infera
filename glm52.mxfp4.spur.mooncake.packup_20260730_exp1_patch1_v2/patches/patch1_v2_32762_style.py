#!/usr/bin/env python3
"""Patch 1 v2 -- rework the HIP/aiter padded-row fix in the shape of PR #32762.

PR #32762 (sgl-project/sglang, NPU, open as of 2026-07-29) fixes the same bug
class: eager DP-attention padding reaching a DSA kernel that was planned for
the unpadded row count.  Three structural things it does that v1 does not:

  1. compute ONE boolean (`trim_eager_padding`) up front and gate BOTH the trim
     and the restore on it, instead of re-deriving the condition at each site;
  2. name the source of the real row count explicitly
     (`_original_num_tokens` / `num_token_non_padded_cpu`);
  3. ASSERT the post-kernel row count before restoring the padding, so a shape
     drift is a loud failure and not a silent short tensor.

What v1 (`apply_fix.py` + `fix_bug6_idle_qoffset.py`) does today.  v1 is
CORRECT -- it passed 2540/2540 -- so this is a robustness rework, not a bug
fix:

  * trim condition   : `q_offset < q_fp8.shape[0]`
  * restore condition: `q_offset < q_fp8.shape[0] and topk_result.shape[0] == q_offset`

    The second conjunct is doing an assert's job as a silent guard.  If the
    kernel ever returned a different row count, the padding would simply not
    be restored and the caller would receive a short tensor -- a wrong answer
    instead of a crash.  That is the single worst property of v1 and item (3)
    is the fix for it.

On the SOURCE of the real row count (item 2), this patch deliberately does
NOT follow #32762, and the reason is evidence:

  * `_original_num_tokens` does not exist in this baseline's ForwardBatch
    (verified: absent from src_spec/model_executor/forward_batch_info.py).
  * `num_token_non_padded_cpu` does exist (line 463) -- but I have NOT
    established that it equals the indexer's real q-row count on the
    draft-extend / target-verify paths, where MTP contributes several rows per
    request and `adjust_num_token_non_padded_for_attn_tp` rewrites the tensor
    (but not obviously the _cpu mirror).
  * `q_offset = sum(metadata.get_dsa_extend_len_cpu())` IS the quantity the
    kernel contract is written against: `lengths` (dsa_seqlens_expanded) is
    sized to it, and the CUDA path slices to it.

So `q_offset` stays the source, and `num_token_non_padded_cpu` is read only to
LOG a disagreement under SGLANG_DEBUG_DSA_ROWS.  Asserting equality here would
stake the run on a belief I have no measurement for; logging turns it into
data.  If the log shows they always agree, a later revision can promote the
comparison to an assert and adopt #32762's source verbatim.

Idempotent.  Verifies each anchor matches exactly once.  Invalidates the .pyc
(CLAUDE.md: a stale .pyc silently reverts the patch).
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

AITER_HEAD = "        if self.paged_mqa_logits_backend.is_aiter():\n"
AITER_END = "\n        elif use_cute_dsl:"

NEW_AITER = '''        # GLM52_P1V2: bound before the branch so the restore below can gate on
        # it without a locals() lookup. Only the aiter branch sets it True;
        # the CUDA branches slice internally and re-pad themselves.
        _p1v2_trim = False
        _p1v2_real = q_offset
        _p1v2_padded = q_fp8.shape[0]
        if self.paged_mqa_logits_backend.is_aiter():
            # Trim eager DP-attention padding before the aiter paged-MQA call,
            # in the shape of upstream PR #32762 (NPU, same bug class).
            #
            # Under DP-attention the hidden states are padded to the largest
            # token count across ranks, so q_fp8 carries more rows than this
            # rank's batch really has, while `lengths` (dsa_seqlens_expanded)
            # is sized to the REAL count. The CUDA path
            # (deepgemm_paged_mqa_logits_split) slices q/weights for exactly
            # this reason; aiter instead sizes its `logits` output from
            # q_fp8.shape[0], so without the same slice the top-k below sees
            # score.shape[0] != lengths.shape[0] and asserts
            # ("Expected lengths.size(0) == B to be true").
            #
            # Trim whenever there IS padding, including _p1v2_real == 0: a
            # DP-IDLE rank has no requests, so the real count is 0 while q_fp8
            # still carries padding. q_fp8[:0] is a legal empty slice and is
            # what an idle rank should pass -- the CUDA path likewise slices
            # unconditionally. (An earlier revision had a `0 < q_offset` lower
            # bound here; it made fast_topk_v2 assert on idle ranks.)
            _p1v2_trim = _p1v2_real < _p1v2_padded
            _q_mqa, _w_mqa = q_fp8, weights
            if _p1v2_trim:
                _q_mqa = q_fp8[:_p1v2_real]
                _w_mqa = weights[:_p1v2_real]
            if _DSA_DEBUG_ROWS:
                # Cross-check against the padding bookkeeping #32762 uses as
                # its source. NOT an assert: it has never been measured to
                # agree with q_offset on the MTP draft-extend path. Logged so
                # a later revision can promote it if the data supports it.
                _p1v2_ntnp = getattr(forward_batch, "_original_num_tokens", None)
                if _p1v2_ntnp is None:
                    _p1v2_ntnp = forward_batch.num_token_non_padded_cpu
                logger.info(
                    "[dsa-rows] mode=%s q_fp8=%s q_offset=%s ntnp=%s agree=%s "
                    "lengths=%s -> mqa_q=%s",
                    forward_batch.forward_mode,
                    tuple(q_fp8.shape),
                    _p1v2_real,
                    _p1v2_ntnp,
                    _p1v2_ntnp == _p1v2_real,
                    tuple(metadata.get_seqlens_expanded().shape),
                    tuple(_q_mqa.shape),
                )
            logits = aiter_paged_mqa_logits(
                _q_mqa,
                kv_cache_fp8,
                _w_mqa,
                seqlens_32,
                block_tables,
                max_seq_len,
                preshuffle=_use_aiter_preshuffle,
                kv_block_size=block_kv,
            )
'''

RESTORE_ANCHOR = """        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
"""

NEW_RESTORE = '''        # GLM52_P1V2: restore the padding the trim above removed.
        #
        # #32762's shape: gate on the SAME boolean that drove the trim, and
        # ASSERT the kernel returned exactly the trimmed row count before
        # concatenating. The previous revision re-derived the condition as
        # `q_offset < q_fp8.shape[0] and topk_result.shape[0] == q_offset`,
        # whose second conjunct silently skipped the restore on a shape drift
        # and handed the caller a short tensor -- a wrong answer rather than a
        # crash.
        if _p1v2_trim:
            assert topk_result.shape[0] == _p1v2_real, (
                "GLM52_P1V2: paged-MQA top-k returned "
                f"{topk_result.shape[0]} rows for {_p1v2_real} trimmed query "
                f"rows (padded={_p1v2_padded}); refusing to pad a mis-shaped "
                "result."
            )
            padding = torch.full(
                (_p1v2_padded - _p1v2_real, topk_result.shape[1]),
                -1,
                dtype=topk_result.dtype,
                device=topk_result.device,
            )
            topk_result = torch.cat([topk_result, padding], dim=0)
        return topk_result
'''


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = open(TARGET).read()

    if "GLM52_P1V2" in src:
        print("already patched (GLM52_P1V2 present); nothing to do")
        return

    if src.count(AITER_HEAD) != 1:
        die(f"aiter branch head matched {src.count(AITER_HEAD)} times, want 1")
    i = src.find(AITER_HEAD)
    j = src.find(AITER_END, i)
    if j < 0:
        die("end of aiter branch (elif use_cute_dsl) not found")
    src = src[:i] + NEW_AITER + src[j + 1 :]

    if src.count(RESTORE_ANCHOR) != 1:
        die(f"topk_transform anchor matched {src.count(RESTORE_ANCHOR)} times, want 1")
    k = src.find(RESTORE_ANCHOR)
    end = src.find("        return topk_result\n", k)
    if end < 0:
        die("`return topk_result` after the restore block not found")
    end += len("        return topk_result\n")
    src = src[:k] + RESTORE_ANCHOR + NEW_RESTORE + src[end:]

    open(TARGET, "w").write(src)
    os.utime(TARGET, None)
    pc = os.path.join(os.path.dirname(TARGET), "__pycache__")
    if os.path.isdir(pc):
        for f in os.listdir(pc):
            if f.startswith("dsa_indexer."):
                os.remove(os.path.join(pc, f))
    print("patched:", TARGET)
    print("marker count in source:", src.count("GLM52_P1V2"))


if __name__ == "__main__":
    main()
