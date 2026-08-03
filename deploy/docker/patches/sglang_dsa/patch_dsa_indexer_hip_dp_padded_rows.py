#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""PATCH 01 -- dsa_indexer HIP/aiter DP-padded rows.

WHAT: the aiter (HIP) paged-MQA branch sizes its `logits` output from the
DP-PADDED row count while `lengths` is sized to the REAL count, so top-k dies
with "Expected lengths.size(0) == B to be true, but got false". Slice q/weights
to the real count (the contract every CUDA backend already honours), then
restore the padding after `topk_transform`.

WHY: without it, DP-attention + EAGLE MTP for GLM-5.2 DSA crashes as soon as
more than one request is in flight -- on gfx950 at the very first batch, and on
gfx942 at conc > 1. CUDA was never affected: those backends take `q_offset=` and
slice internally. ROCm-specific.

WHY A SCRIPT AND NOT A DIFF. Patches 02 and 04 are `--fuzz=0` context diffs
pinned to v0.5.15.post1, the mi35x base. This fix is needed on BOTH engine
bases, and `dsa_indexer.py` drifts between them: v0.5.16 adds `_is_xpu` beside
the platform flags and calls `_mask_init_and_local_tokens` right before
`topk_transform`, which fails 2 of the diff's 4 hunks. Both of this patch's own
edit sites are byte-identical across the two releases, so anchoring on source
text instead of line numbers covers both from one source of truth. On
v0.5.15.post1 the result is byte-identical to the diff this replaces.

UPSTREAM STATUS (queried with `gh` on 2026-08-01, re-read from the web UI on
2026-08-03 -- all still OPEN, none merged)
  issue          NONE. No upstream issue reports this. Searched sgl-project/
                 sglang for "Expected lengths.size", "DSA padding DP attention"
                 and the GLM-5.2 MTP crash text: no match. Weak evidence --
                 `gh search` matches titles and bodies, not diff content.
  third-party PR #32762 "[NPU] Fix DSA eager padding mismatch in PD MTP warm-up"
                 (stellaxcpeng, OPEN). SAME BUG CLASS on a different platform,
                 and this patch is written in its shape: one boolean gates both
                 trim and restore, and the returned row count is asserted before
                 the padding is re-attached. It does not touch the aiter branch,
                 so it does not fix HIP.
                 #31683 "[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP" (long10024070,
                 OPEN). ADJACENT, NOT THE SAME FIX -- same function, but it
                 widens forward_cuda's guard from `seq_lens.numel() == 0` to
                 `is_idle() or ...`. Different symptom; verified against its diff.
                 #30378 / #30427 (both MERGED, in both bases) clamp padded-row
                 seq_lens VALUES in triton_ops/pad.py. This fixes the HIP-side
                 row COUNT -- a different defect in the same area.
  own PR         #33059 "Fix DSA indexer aiter (HIP) padding mismatch under
                 DP-attention" (dorado269, OPEN, REVIEW_REQUIRED, against main,
                 1 file +16/-4). CI is red only at `pr-gate`, which blocks on a
                 missing `run-ci` label, not on code.

THIS PATCH vs OUR OWN PR #33059. Same defect, same two edits. They differ
deliberately:
  base       here: whatever base carries the anchors (v0.5.15.post1 and v0.5.16
             verified). #33059: sglang `main`, whose `_get_topk_paged` has
             drifted further (cutedsl / dg_native branches,
             `_mask_init_and_local_tokens`).
  form       here: explicit `_p1v2_trim` / `_p1v2_real` / `_p1v2_padded` locals
             with a post-kernel row-count ASSERT, following #32762's shape.
             #33059: the minimal `q_fp8[:q_offset]` / `weights[:q_offset]` slice
             and a plain `if q_offset < q_fp8.shape[0]` restore.
  local-only Stripped for upstream: the `GLM52_P1V2` markers (the build script
             greps them out of the BYTECODE to prove the patch landed), the
             `_p1v2_*` naming, and the `SGLANG_DEBUG_DSA_ROWS` logging block.
             All three are scaffolding for this repo's verification loop and
             carry no upstream value.
  trade-off  The upstream form is smaller and easier to review. Ours fails
             LOUDER: on a shape drift the assert crashes instead of silently
             returning a short tensor. Prefer upstream's once it merges and the
             base carries it -- then delete this script.
  validation Ours: 4/4 probe, conc=64 1k/1k 256/256, plus 2540/2540 in the PD
             set, all on gfx950 / v0.5.15.post1. #33059's port to `main` was NOT
             re-run on hardware.

RELATION TO THE INDEXSHARE WORKAROUND. NOT substituted.
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'` removes
the rank divergence that patch 04 and patch 02b address, but this is an
independent bug -- present regardless of IndexShare, and the arm that validated
the override had THIS patch applied.

Self-locating and idempotent. All edits or none: an anchor that is missing or no
longer unique writes NOTHING and fails (exit 1), because a half-applied fix
crashes in the same place the unpatched tree does.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[dsa-indexer-rows]"

_REL = "layers/attention/dsa/dsa_indexer.py"

# (anchor, anchor + our edit). Each anchor must occur exactly once; the
# replacement doubles as the already-applied marker.
_EDITS: list[tuple[str, str]] = [
    # `os` for the debug-logging env switch below.
    (
        """from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
""",
        """from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import os

import torch
""",
    ),
    # Anchored on the two platform flags rather than the whole block: v0.5.16
    # adds `_is_xpu` on the line after `_is_npu`.
    (
        """_is_cuda = is_cuda()
_is_hip = is_hip()
""",
        """_is_cuda = is_cuda()
_is_hip = is_hip()
# Set SGLANG_DEBUG_DSA_ROWS=1 to log the indexer row bookkeeping (padded vs real).
_DSA_DEBUG_ROWS = os.environ.get("SGLANG_DEBUG_DSA_ROWS", "0") == "1"

""",
    ),
    # Trim q/weights to the real row count on the aiter branch only.
    (
        """        weights = weights.squeeze(2)

        if self.paged_mqa_logits_backend.is_aiter():
            logits = aiter_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
""",
        """        weights = weights.squeeze(2)

        # GLM52_P1V2: bound before the branch so the restore gates on the same
        # boolean. Only aiter sets it; the CUDA branches slice internally.
        _p1v2_trim = False
        _p1v2_real = q_offset
        _p1v2_padded = q_fp8.shape[0]
        if self.paged_mqa_logits_backend.is_aiter():
            # WHY: DP-attention pads q_fp8 past `lengths` and aiter (unlike CUDA)
            # sizes logits from q_fp8.shape[0] -> "Expected lengths.size(0) == B".
            # HOW: slice to the real count -- the contract CUDA already honours.

            # Trim at _p1v2_real == 0 too: q_fp8[:0] is what a DP-idle rank should
            # pass. An earlier `0 < q_offset` bound asserted on exactly those ranks.
            _p1v2_trim = _p1v2_real < _p1v2_padded
            _q_mqa, _w_mqa = q_fp8, weights
            if _p1v2_trim:
                _q_mqa = q_fp8[:_p1v2_real]
                _w_mqa = weights[:_p1v2_real]
            if _DSA_DEBUG_ROWS:
                # Cross-check against the padding bookkeeping #32762 keys off.
                # Logged, not asserted: it has never been measured to agree with
                # q_offset on the MTP draft-extend path.
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
""",
    ),
    # Restore the trimmed padding, gated on the SAME boolean.
    (
        """        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
""",
        """        # GLM52_P1V2: restore the trimmed padding -- gate on the SAME boolean and
        # assert the row count. BEFORE, a re-derived condition skipped the restore
        # on a shape drift, returning a short tensor instead of crashing.
        if _p1v2_trim:
            assert topk_result.shape[0] == _p1v2_real, (
                "GLM52_P1V2: paged-MQA top-k returned "
                f"{topk_result.shape[0]} rows for {_p1v2_real} trimmed query "
                f"rows (padded={_p1v2_padded}); refusing to pad a mis-shaped "
                "result."
            )
            padding = torch.full(
                (_p1v2_padded - _p1v2_real, topk_result.shape[1]),
""",
    ),
]


def _srt_dir():
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "srt"
    return d if d.is_dir() else None


def main():
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    f = srt / _REL
    if not f.is_file():
        print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
        return 1

    src = out = f.read_text()
    for old, new in _EDITS:
        if new in out:
            continue  # this edit is already in the tree
        found = out.count(old)
        if found != 1:
            where = "absent" if found == 0 else f"{found}x ambiguous"
            print(f"{_TAG} anchor {where} in {_REL}: {old.splitlines()[0]!r}")
            print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
            return 1
        out = out.replace(old, new, 1)

    if out == src:
        print(f"{_TAG} already present — skipping")
        return 0
    f.write_text(out)
    print(f"{_TAG} patched {f}")
    print(f"{_TAG} aiter paged-MQA now sees the real row count under DP-attention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
