###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""kvd must not be wired on a PD decode leg.

infera appends ``--enable-hierarchical-cache`` whenever kvd is on. On a decode
leg that is wrong twice over:

* It crashes. With EAGLE/MTP, SGLang refuses
  ``--disaggregation-decode-enable-radix-cache`` (which infera also appends, on
  the separate kv-events gate), which forces ``disable_radix_cache = True``,
  which SGLang then refuses to combine with the hierarchical cache.
* Even when it does not crash, kvd is **write-only** there in every
  configuration: ``_prefetch_kvcache`` is the sole caller of
  ``prefetch_from_storage``, and ``_add_request_to_queue`` only calls it on the
  NULL and PREFILL branches. Measured at 180 sets / 0 gets against a prefill
  leg's 102 sets / 102 gets on the same run.

The companion gate -- not appending the decode radix cache under speculative
decoding -- is a different flag on a different condition and is covered by
``test_decode_radix_vs_speculative.py``, which needs sglang importable.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from infera.engine.sglang import kvd_wiring


def _args(mode: str | None):
    return SimpleNamespace(
        infera_kvd_socket="/tmp/kvd/kvd.sock",
        sglang_argv=["--model-path", "/nonexistent"],
        server_args=SimpleNamespace(
            disaggregation_mode=mode,
            enable_hierarchical_cache=False,
            hicache_storage_backend=None,
        ),
    )


def _wire(args, monkeypatch) -> list[str]:
    """Run the real wiring entry point and return the forwarded argv.

    Tests the OBSERVABLE effect -- which flags reach the sglang subprocess --
    rather than the presence of a helper. A test that imports the helper by name
    turns into a collection error when the fix is reverted, and a collection
    error proves nothing about behaviour.

    The kvd daemon probe and the backend registration both need a live process,
    so both are stubbed; neither is what this test is about.
    """
    monkeypatch.setattr(kvd_wiring, "_probe_kvd", _noop_probe)
    monkeypatch.setattr(
        kvd_wiring, "_finish_wiring",
        lambda a, sock: kvd_wiring._append_sglang_hicache_argv(a),
    )
    asyncio.run(kvd_wiring.awire_infera_kvd_backend(args))
    return list(args.sglang_argv)


async def _noop_probe(socket_path: str) -> None:
    return None


# ----------------------------------------------------------------------
# kvd wiring vs the decode leg
# ----------------------------------------------------------------------


def test_decode_leg_gets_no_hicache_flags(monkeypatch):
    """The load-bearing assertion: nothing kvd-related reaches the decode leg.

    Fails on the pre-fix code, where --enable-hierarchical-cache is appended
    unconditionally and SGLang then refuses to combine it with the
    disable_radix_cache that a speculative decode leg is forced into.
    """
    argv = _wire(_args("decode"), monkeypatch)
    assert "--enable-hierarchical-cache" not in argv
    assert "--hicache-storage-backend" not in argv


def test_prefill_leg_still_gets_them(monkeypatch):
    """kvd must keep working where it actually serves reads."""
    argv = _wire(_args("prefill"), monkeypatch)
    assert "--enable-hierarchical-cache" in argv
    assert "--hicache-storage-backend" in argv


def test_aggregated_server_still_gets_them(monkeypatch):
    """No disaggregation mode at all -- the colocated case must be untouched."""
    argv = _wire(_args(None), monkeypatch)
    assert "--enable-hierarchical-cache" in argv


def test_missing_server_args_does_not_raise(monkeypatch):
    """Defensive: the guard runs before SGLang args are fully materialised on
    some call paths, and must not turn a missing attribute into a crash."""
    args = SimpleNamespace(infera_kvd_socket="/x", sglang_argv=[])
    argv = _wire(args, monkeypatch)
    assert "--enable-hierarchical-cache" in argv
