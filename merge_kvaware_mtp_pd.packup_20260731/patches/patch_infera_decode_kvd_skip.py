#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Do not wire kvd on a PD decode leg -- it is write-only there.

WHAT: SGLang never issues a storage prefetch on a PD decode worker. The only
caller of ``tree_cache.prefetch_from_storage(...)`` is ``_prefetch_kvcache``, and
``_add_request_to_queue`` calls it on the NULL (aggregated) and PREFILL branches
only -- the DECODE branch goes straight to the prealloc queue:

    sglang/srt/managers/scheduler.py:2309
        if   disaggregation_mode == NULL:    self._prefetch_kvcache(req)
        elif disaggregation_mode == PREFILL: self._prefetch_kvcache(req)
        elif disaggregation_mode == DECODE:  self.disagg_decode_prealloc_queue.add(...)

The backup path still runs, so L3 fills and is never read. Measured on the
merge-validation G0 run (kvd on both legs, MTP off, so the decode leg had a full
HiRadixCache -- the most favourable case): prefill 102 sets / **102 gets**, decode
180 sets / **0 gets**, 318 MB of host memory and the D2H bandwidth to move it, for
zero reads.

This is a property of PD decode, not of MTP. The merge hit it from the MTP side
only because a *different* restriction crashes the leg outright there: SGLang
forbids ``--disaggregation-decode-enable-radix-cache`` under speculative decoding,
which forces ``disable_radix_cache=True``, which then collides with the
``--enable-hierarchical-cache`` this wiring always appends. Fixing only that
(patch_infera_decode_radix_vs_mtp.py, superseded by this one) stops the crash but
keeps the pointless traffic.

FIX: skip kvd wiring entirely when ``disaggregation_mode == "decode"``, and say so
at INFO. The decode leg then runs with no hicache flags at all, so the
speculative-decoding collision above cannot arise either.

CAVEAT, deliberately not hidden: ``--disaggregation-decode-enable-offload-kvcache``
drives a separate decode-side mechanism (``DecodeKVCacheOffloadManager``) that
requires a hicache storage backend. We never enable it and have NOT checked
whether it reads back from L3. Skipping the wiring disables that path too. If you
need it, read that manager first and re-scope this patch. See
notes/decode_leg_kvd_is_write_only.md.

BELONGS AS A SOURCE COMMIT, not a build-time patch: this edits infera's own code.
It ships as a script only so running experiment containers pick it up without a
rebuild.

Self-locating and idempotent.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[decode-kvd-skip]"

_GUARD = '''def _skip_kvd_on_decode_leg(args: Any) -> bool:
    """True if this is a PD decode leg, where kvd would be write-only.

    SGLang only prefetches from hicache storage on the aggregated and PREFILL
    branches of ``Scheduler._add_request_to_queue``; the DECODE branch has no
    ``_prefetch_kvcache`` call, and that method is the sole caller of
    ``prefetch_from_storage``. The backup path still runs, so a decode leg fills
    L3 and never reads it -- measured at 180 sets / 0 gets against a prefill
    leg's 102 sets / 102 gets on the same run.
    """
    mode = getattr(getattr(args, "server_args", None), "disaggregation_mode", None)
    if str(mode) != "decode":
        return False
    logger.info(
        "PD decode leg: not wiring infera-kvd. SGLang issues no storage prefetch "
        "on the decode branch (scheduler._add_request_to_queue), so L3 here would "
        "be write-only -- host memory and D2H bandwidth for zero reads. kvd stays "
        "on the prefill leg, which is where prefix reuse is decided."
    )
    return True


'''

_EDITS: list[tuple[str, str]] = [
    # Insert the guard helper just above the async entry point.
    (
        "async def awire_infera_kvd_backend(args: Any) -> None:\n",
        _GUARD + "async def awire_infera_kvd_backend(args: Any) -> None:\n",
    ),
    # Apply it in the async entry point.
    (
        '''    """Async variant for callers already inside an event loop."""
    socket_path = args.infera_kvd_socket
    if not socket_path:
        return
''',
        '''    """Async variant for callers already inside an event loop."""
    socket_path = args.infera_kvd_socket
    if not socket_path:
        return
    if _skip_kvd_on_decode_leg(args):
        return
''',
    ),
]

# Only the async entry point needs the guard: the sync one
# (``wire_infera_kvd_backend``) just delegates to it via ``asyncio.run``.


def _infera_dirs() -> list[Path]:
    """Every infera package copy that could be imported.

    The engine image carries two: site-packages and the WORKDIR source tree at
    /opt/infera/infera, which shadows it for any process started there.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(d: Path) -> None:
        d = d.resolve()
        if d.is_dir() and (d / "engine" / "sglang" / "kvd_wiring.py").is_file() and d not in seen:
            seen.add(d)
            roots.append(d)

    spec = importlib.util.find_spec("infera")
    if spec and spec.origin:
        _add(Path(spec.origin).parent)
    _add(Path("/opt/infera/infera"))
    return roots


def _patch_file(f: Path) -> int:
    src = out = f.read_text()
    if "_skip_kvd_on_decode_leg" in src:
        print(f"{_TAG} {f}: already present — skipping")
        return 0

    for old, new in _EDITS:
        if out.count(old) != 1:
            print(f"{_TAG} anchor not found exactly once in {f}: {old.splitlines()[0]!r}")
            print(f"{_TAG} infera drifted — re-cut the patch, nothing written")
            return 1
        out = out.replace(old, new, 1)

    f.write_text(out)
    print(f"{_TAG} patched {f}")
    return 0


def main() -> int:
    roots = _infera_dirs()
    if not roots:
        print(f"{_TAG} no infera package found — cannot patch")
        return 1
    for root in roots:
        rc = _patch_file(root / "engine" / "sglang" / "kvd_wiring.py")
        if rc != 0:
            return rc
    print(f"{_TAG} kvd is no longer wired on PD decode legs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
