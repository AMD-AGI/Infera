#!/usr/bin/env python3
"""GLM52_P1V3: handle the reversed padding case in the aiter paged-MQA trim.

Runs INSIDE the decode container. Idempotent: re-running is a no-op.

See work.bench_20260801/patches/0004-*.txt and notes.dsa.mtp.crash.md.

The existing GLM52_P1V2 trim only handles `real < padded` (q_fp8 carries DP
padding). On an IDLE rank under MTP draft-extend the inequality inverts --
q_offset overcounts the rows actually in q_fp8 -- so no trim runs and
fast_topk_v2 asserts on score.shape[0] != lengths.shape[0].
"""
import re
import sys

P = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

src = open(P).read()
if "GLM52_P1V3" in src:
    print("already patched (GLM52_P1V3 present) - no-op")
    sys.exit(0)

OLD_TRIM = """            _p1v2_trim = _p1v2_real < _p1v2_padded
            _q_mqa, _w_mqa = q_fp8, weights
            if _p1v2_trim:
                _q_mqa = q_fp8[:_p1v2_real]
                _w_mqa = weights[:_p1v2_real]"""

NEW_TRIM = """            # GLM52_P1V3: the padding can go BOTH ways.
            #   real < padded : the #32762 case -- q_fp8 carries DP padding;
            #                   trim q/weights down to the real count.
            #   real > padded : IDLE ranks under MTP draft-extend. q_offset
            #                   (sum of dsa_extend_len_cpu) OVERCOUNTS the rows
            #                   actually materialized in q_fp8, so there is
            #                   nothing to trim -- the LENGTHS must instead be
            #                   clipped to the rows present, else fast_topk_v2
            #                   asserts. Observed live as
            #                   mode=IDLE q_fp8=(1,..) q_offset=2 lengths=(2,).
            # _p1v2_rows is the single source of truth for both sides.
            _p1v2_rows = min(_p1v2_real, _p1v2_padded)
            _p1v2_trim = _p1v2_rows < _p1v2_padded
            _p1v2_clip = _p1v2_rows < _p1v2_real
            _q_mqa, _w_mqa = q_fp8, weights
            if _p1v2_trim:
                _q_mqa = q_fp8[:_p1v2_rows]
                _w_mqa = weights[:_p1v2_rows]"""

if OLD_TRIM not in src:
    sys.exit("FAIL: trim block not found verbatim - image differs from expected")
src = src.replace(OLD_TRIM, NEW_TRIM, 1)

OLD_CALL = """        topk_result = metadata.topk_transform(logits, self.index_topk)"""
NEW_CALL = """        # GLM52_P1V3: when rows were clipped (real > padded), the lengths the
        # transform indexes must be clipped to match. Passed via the existing
        # ke_offset extension point rather than mutating shared/graph-captured
        # metadata.
        if _p1v2_clip:
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ke_offset=metadata.get_seqlens_expanded()[:_p1v2_rows],
            )
        else:
            topk_result = metadata.topk_transform(logits, self.index_topk)"""

if src.count(OLD_CALL) != 1:
    sys.exit(f"FAIL: topk_transform call site count={src.count(OLD_CALL)}, want 1")
src = src.replace(OLD_CALL, NEW_CALL, 1)

# The restore-padding block must now key off _p1v2_rows, not _p1v2_real.
OLD_ASSERT = """            assert topk_result.shape[0] == _p1v2_real, (
                "GLM52_P1V2: paged-MQA top-k returned "
                f"{topk_result.shape[0]} rows for {_p1v2_real} trimmed query "
                f"rows (padded={_p1v2_padded}); refusing to pad a mis-shaped "
                "result."
            )
            padding = torch.full(
                (_p1v2_padded - _p1v2_real, topk_result.shape[1]),"""
NEW_ASSERT = """            assert topk_result.shape[0] == _p1v2_rows, (
                "GLM52_P1V3: paged-MQA top-k returned "
                f"{topk_result.shape[0]} rows for {_p1v2_rows} trimmed query "
                f"rows (real={_p1v2_real}, padded={_p1v2_padded}); refusing to "
                "pad a mis-shaped result."
            )
            padding = torch.full(
                (_p1v2_padded - _p1v2_rows, topk_result.shape[1]),"""

if OLD_ASSERT not in src:
    sys.exit("FAIL: restore/assert block not found verbatim")
src = src.replace(OLD_ASSERT, NEW_ASSERT, 1)

open(P, "w").write(src)
print(f"patched OK - GLM52_P1V3 occurrences: {src.count('GLM52_P1V3')}")
