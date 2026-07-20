#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Idempotent, self-locating, STRICT-match ATOM source patch for MiniMax-M2.

Two minimal edits to ``atom/models/minimax_m2.py`` so the stock model file loads
and runs (the base image's version errors out):

  A. drop the ``dtype=`` kwarg from the ``get_rope(...)`` call — the installed
     rope API doesn't accept it (TypeError on load).
  B. guard the TP fused QK-norm all-reduce kernel with ``qkv.shape[0] <= 256``
     so large batches fall back to the manual fp32 QK-norm path.

(NOTE: these make MiniMax *start*; correct output additionally needs the
triton/unified attention backend, selected at launch via
``ATOM_USE_UNIFIED_ATTN=1`` + ``--block-size 64`` in models/minimax_m27.conf —
not here.)

STRICT MATCHING: each hunk is applied ONLY if its exact, unique pre-image is
present. If the source has changed (e.g. an ATOM/base-image upgrade) so a hunk's
pre-image is absent or no longer unique, that hunk is SKIPPED with a loud WARNING
and the build continues — the patch never edits code it doesn't recognise.
Already-patched hunks are detected and left alone (no-op).
"""

import importlib.util
import os
import sys

REL = "models/minimax_m2.py"

# (name, exact pre-image, post-image). Pre-images are verbatim from the stock
# rocm/atom base image; chosen multi-line so each is unique in the file.
HUNKS = [
    (
        "rope-drop-dtype-kwarg",
        "            base=rope_theta,\n"
        "            rope_scaling=rope_scaling,\n"
        '            dtype=getattr(quant_config, "torch_dtype", None),\n'
        "        )\n",
        "            base=rope_theta,\n            rope_scaling=rope_scaling,\n        )\n",
    ),
    (
        "qknorm-fused-allreduce-batch-guard",
        "            if self.tp_size > 1:\n"
        "                q, k, v = tensor_model_parallel_fused_qknorm_allreduce(\n",
        "            if qkv.shape[0] <= 256 and self.tp_size > 1:\n"
        "                q, k, v = tensor_model_parallel_fused_qknorm_allreduce(\n",
    ),
]


def _find_target():
    spec = importlib.util.find_spec("atom")
    roots = (
        list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    )
    roots.append("/app/ATOM/atom")
    for root in roots:
        cand = os.path.join(root, REL)
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    tag = "atom-patch minimax-m2-qknorm-rope"
    target = _find_target()
    if not target:
        print(f"[{tag}] WARNING: target {REL} not found — MiniMax fix NOT applied")
        return 0

    src = open(target).read()
    changed = False
    skipped = []
    for name, old, new in HUNKS:
        if new in src and old not in src:
            print(f"[{tag}] {name}: already applied")
            continue
        n = src.count(old)
        if n == 1:
            src = src.replace(old, new, 1)
            changed = True
            print(f"[{tag}] {name}: patched")
        elif n == 0:
            skipped.append(name)
            print(
                f"[{tag}] WARNING: {name}: pre-image NOT found — ATOM source "
                f"likely changed; hunk SKIPPED (re-derive the patch)"
            )
        else:
            skipped.append(name)
            print(
                f"[{tag}] WARNING: {name}: pre-image not unique ({n}x) — "
                f"hunk SKIPPED to avoid wrong edit"
            )

    if changed:
        open(target, "w").write(src)
        print(f"[{tag}] wrote {target}")
    if skipped:
        print(
            f"[{tag}] WARNING: {len(skipped)} hunk(s) not applied "
            f"({', '.join(skipped)}); MiniMax may fail to load — verify against "
            f"the new ATOM source."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
