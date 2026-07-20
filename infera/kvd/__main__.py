###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""python -m infera.kvd entry point.

Two modes:

  - ``python -m infera.kvd`` (no args / standard CLI args) → start
    the kvd daemon. This is the historical behavior.
  - ``python -m infera.kvd classify <path> [<path> ...]`` →
    storage-classifier inspection tool. Prints what
    ``infera.kvd.storage_classify.pick_io_mode`` would pick for each
    path and exits. Used to validate
    the mount layout matches expectations.
  - ``python -m infera.kvd l4-validate <config.yaml>`` → connect to a
    distributed L4 backend (Mooncake or LMCache, auto-detected from the
    config keys), put/get/delete a probe key, report latency. No daemon
    spin-up. Used to verify cluster config before launching kvd.

The classify / l4-validate subcommands are intentionally light — no
daemon spin-up, no kvd state — so operators can run them under any user
against any mount point without elevated privileges.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _classify_main(paths: list[str]) -> int:
    from infera.kvd.storage_classify import format_decision

    if not paths:
        print("usage: python -m infera.kvd classify <path> [<path> ...]", file=sys.stderr)
        return 2
    for p in paths:
        print(format_decision(Path(p)))
        print()
    return 0


def _l4_validate_main(argv: list[str]) -> int:
    """Connect to an L4 backend from a YAML config, round-trip a probe
    key, report latency. Auto-detects mooncake vs lmcache from config
    keys (master_address → mooncake, remote_url → lmcache)."""
    import time

    if not argv:
        print(
            "usage: python -m infera.kvd l4-validate <config.yaml>",
            file=sys.stderr,
        )
        return 2
    cfg_path = argv[0]
    from infera.kvd.server import _build_distributed_long_region, _load_yaml_config

    raw = _load_yaml_config(cfg_path)
    if raw.get("master_address"):
        backend = "mooncake"
    elif raw.get("remote_url"):
        backend = "lmcache"
    else:
        print(
            f"l4-validate: cannot tell backend from {cfg_path} — need "
            "'master_address' (mooncake) or 'remote_url' (lmcache)",
            file=sys.stderr,
        )
        return 2

    class _Args:
        mooncake_config = cfg_path
        lmcache_config = cfg_path
        long_path = None
        long_paths = None

    region = _build_distributed_long_region(backend, _Args())
    print(f"l4-validate: backend={backend} config={cfg_path}")
    try:
        t0 = time.perf_counter()
        region.start()
        print(f"  connect: {(time.perf_counter() - t0) * 1e3:.1f} ms")
    except Exception as exc:
        print(f"  CONNECT FAILED: {exc}", file=sys.stderr)
        return 1

    probe_key = b"__l4_validate_probe__"
    probe_val = b"\xab" * 4096
    try:
        t0 = time.perf_counter()
        ok, reason = region.put(
            probe_key,
            probe_val,
            retention="long",
            model="_probe",
            compat_key="_probe",
            metadata={},
        )
        put_ms = (time.perf_counter() - t0) * 1e3
        print(f"  put:     {put_ms:.1f} ms  ok={ok} reason={reason}")

        t0 = time.perf_counter()
        got = region.get_bytes(probe_key, model="_probe", compat_key="_probe")
        get_ms = (time.perf_counter() - t0) * 1e3
        match = got == probe_val
        print(f"  get:     {get_ms:.1f} ms  match={match}")

        stats = region.stats()
        print(f"  backend_name: {stats.get('backend_name')}")
        if not (ok and match):
            print("  RESULT: FAIL — put/get round-trip did not verify", file=sys.stderr)
            return 1
        print("  RESULT: OK")
        return 0
    finally:
        region.shutdown()


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "classify":
        raise SystemExit(_classify_main(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "l4-validate":
        raise SystemExit(_l4_validate_main(sys.argv[2:]))
    from infera.kvd.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
