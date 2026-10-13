#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Backport LMCache#5098: a blocking retrieve() leaves lookup pins to lookup_unpin().

ATOM's offload worker pins a request's CPU chunks with ``lookup(pin=True)``,
loads them with ``retrieve()`` and then calls ``lookup_unpin()``. The cleanup
loop of ``LMCacheEngine.retrieve`` also unpins every pinned chunk it read, so
each lookup pin is released twice. With one owner the count goes negative
(clamped to 0, with a warning); with two requests sharing a prefix, the first
to finish also drops the second's pin and the chunk becomes evictable before
the second loads it. Measured on the experiment image, 4 GPUs: shared
512-token prefix, pins 2 -> 0 and the second request recovered 256 tokens.

Upstream (LMCache/LMCache#5098, bdb66b8) unpins there only for async prefetch
or passive ranks, which own those pins; this is its cache_engine.py hunk.
Written against lmcache 0.5.5rc3 in rocm/atom-dev:nightly_202609221542.
Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

PATCH = {
    "v1/cache_engine.py": (
        "# Blocking gets acquire references, not ownership of lookup pins.",
        [
            (
                "            else:\n"
                "                if memory_obj.is_pinned:\n"
                "                    memory_obj.unpin()\n"
                "                memory_obj.ref_count_down()\n",
                "            else:\n"
                "                # Blocking gets acquire references, not ownership of lookup pins.\n"
                "                # The lookup owner releases those pins through lookup_unpin().\n"
                "                if (self.async_loading or self._is_passive()) and memory_obj.is_pinned:\n"
                "                    memory_obj.unpin()\n"
                "                memory_obj.ref_count_down()\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("lmcache-patch lookup-pin", PATCH, package="lmcache"))
