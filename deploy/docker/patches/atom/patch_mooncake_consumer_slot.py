#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Idempotent, self-locating ATOM engine source patch.

Bug: in the Mooncake PD KV-transfer consumer, ``start_load_kv`` references
``consumer_staging_pool_idx`` unconditionally when recording a pending recv
slot, but only binds it inside the ``if self._has_slot_regions:`` branch.
Hybrid models that use cache-group slot state but DON'T register slot_regions
(e.g. Qwen3.5's GDN linear-attention layout: ``local_slot_index >= 0`` while
``_has_slot_regions`` is False) take the else path and hit:

    UnboundLocalError: cannot access local variable 'consumer_staging_pool_idx'

which crashes the decode worker on the first PD request. Fix: initialise
``consumer_staging_pool_idx = -1`` at the top of the per-request loop (the
WRITE_DONE release path is already guarded by ``if pool_idx >= 0:``, so a -1
is safe and simply skips the staging scatter/release for the block-only path).

Self-locating + idempotent + no-op if the anchor is absent (so an ATOM version
bump degrades gracefully instead of failing the image build).
"""

import importlib.util
import os
import sys

REL = "kv_transfer/disaggregation/mooncake/mooncake_connector.py"


def _find_target() -> str | None:
    spec = importlib.util.find_spec("atom")
    roots = []
    if spec and spec.submodule_search_locations:
        roots.extend(spec.submodule_search_locations)
    roots.append("/app/ATOM/atom")
    for root in roots:
        cand = os.path.join(root, REL)
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    target = _find_target()
    if not target:
        print("[atom-patch mooncake-consumer-slot] target not found — skip")
        return 0

    src = open(target).read()
    if "task5: bind consumer_staging_pool_idx" in src:
        print("[atom-patch mooncake-consumer-slot] already applied")
        return 0

    anchor = "        for req_id, meta in metadata.reqs_to_recv.items():\n"
    if anchor not in src:
        print("[atom-patch mooncake-consumer-slot] anchor not found — skip")
        return 0
    if src.count(anchor) != 1:
        print("[atom-patch mooncake-consumer-slot] anchor not unique — skip")
        return 0

    inject = (
        anchor
        + "            # task5: bind consumer_staging_pool_idx for the non-slot-regions\n"
        + "            # (block-only) path too — hybrid GDN models (e.g. Qwen3.5) hit\n"
        + "            # `local_slot_index >= 0` without registering slot_regions.\n"
        + "            consumer_staging_pool_idx = -1\n"
    )
    src = src.replace(anchor, inject, 1)
    open(target, "w").write(src)
    print(f"[atom-patch mooncake-consumer-slot] patched {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
