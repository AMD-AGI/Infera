#!/usr/bin/env python3
"""MVP for the free_tcp_port_block collision — no GPU, no cluster, ~1 second.

Three things, in the order they were established on 2026-07-30:

  1. REPRODUCE   the pre-fix code returns the SAME base to every caller.
  2. CONFIRM     the post-fix code spreads the picks.
  3. REJECT      the two more-obvious fixes, on evidence, so nobody re-proposes
                 them:
                   (a) hold the reservation until the child binds
                   (b) reserve on 0.0.0.0 with SO_REUSEADDR

The pre-fix implementation is inlined below rather than fetched from git, so
this file runs standalone against any checkout. It is a verbatim transcription
of the original loop body; the only difference from the fixed version is the
scan start.

Usage:
    python3 mvp_port_block.py                 # all three sections
    python3 mvp_port_block.py --section 1     # just the reproduction
"""

from __future__ import annotations

import argparse
import socket
import sys

COUNT = 4  # dp_size=4 in the MVP: sglang needs base..base+3


def _ephemeral_low() -> int:
    try:
        return int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
    except (OSError, ValueError, IndexError):
        return 32768


def free_tcp_port_block_PREFIX(count: int) -> int:
    """The buggy version: fixed scan start at (low - count), descending.

    Transcribed from infera/common/net.py before the fix. The probe sockets are
    closed in `finally` BEFORE the value is returned, so nothing is held — two
    callers therefore see the identical free block.
    """
    if count <= 1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    low = _ephemeral_low()
    for base in range(low - count, 1024, -1):
        socks: list[socket.socket] = []
        try:
            for off in range(count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", base + off))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError(f"could not find {count} contiguous free TCP ports")


def section1_reproduce() -> bool:
    print("=" * 72)
    print("1. REPRODUCE — pre-fix free_tcp_port_block is deterministic")
    print("=" * 72)
    low = _ephemeral_low()
    print(f"   ip_local_port_range low = {low}  -> fixed scan start = {low - COUNT}")
    bases = [free_tcp_port_block_PREFIX(COUNT) for _ in range(10)]
    print(f"   OLD bases: {bases}")
    print(f"   distinct: {len(set(bases))}")
    bug = len(set(bases)) == 1
    print(f"   -> {'FAIL (bug reproduced, as expected)' if bug else 'did NOT reproduce'}")
    if bug:
        b = bases[0]
        print(f"\n   Consequence with dp_size=4: BOTH legs get base {b}, and sglang")
        print(f"   binds one publisher per DP rank at base+rank, i.e.")
        print(f"     leg A -> {b}, {b+1}, {b+2}, {b+3}")
        print(f"     leg B -> {b}, {b+1}, {b+2}, {b+3}   <- every one collides")
        print(f"   The observed crash was on base+1.")
    return bug


def section2_confirm_fix() -> bool:
    print()
    print("=" * 72)
    print("2. CONFIRM — the shipped fix spreads the picks")
    print("=" * 72)
    try:
        from infera.common.net import free_tcp_port_block
    except ImportError as exc:
        print(f"   SKIPPED: cannot import infera.common.net ({exc})")
        print("   Run this inside the container, or from an infera checkout.")
        return True
    bases = [free_tcp_port_block(COUNT) for _ in range(10)]
    print(f"   NEW bases: {bases}")
    print(f"   distinct: {len(set(bases))}")
    low = _ephemeral_low()
    in_range = all(1024 <= b and b + COUNT - 1 < low for b in bases)
    ok = len(set(bases)) > 1 and in_range
    print(f"   all below the ephemeral range ({low}): {in_range}")
    print(f"   -> {'PASS' if ok else 'FAIL — this checkout may be pre-fix'}")
    return ok


def section3_rejected() -> bool:
    print()
    print("=" * 72)
    print("3. REJECT — the two obvious alternative fixes, on evidence")
    print("=" * 72)
    print()
    print("   (a) Hold the 127.0.0.1 reservation until the child binds.")
    print("       Rejected: the probe binds 127.0.0.1:P but the sglang child")
    print("       binds 0.0.0.0:P (zmq 'tcp://*'). Holding locks out the very")
    print("       process the block was reserved FOR.")
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        held.bind(("127.0.0.1", 0))
        p = held.getsockname()[1]
        print(f"       reserved 127.0.0.1:{p}")
        child = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            child.bind(("0.0.0.0", p))
            a_result = "OK  <-- unexpected; holding would have been viable here"
            a_ok = False
        except OSError as exc:
            a_result = f"BLOCKED errno={exc.errno}  <-- holding would break our own child"
            a_ok = exc.errno == 98
        finally:
            child.close()
        print(f"       child bind 0.0.0.0:{p} -> {a_result}")
    finally:
        held.close()

    print()
    print("   (b) Reserve on 0.0.0.0 with SO_REUSEADDR so the child can take over.")
    print("       Rejected: then the reservation is not exclusive either — a")
    print("       second probe takes the same port and the collision is back.")
    res = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    res.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        res.bind(("0.0.0.0", 0))  # no listen()
        p = res.getsockname()[1]
        print(f"       reserved 0.0.0.0:{p} (SO_REUSEADDR, no listen)")
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            other.bind(("0.0.0.0", p))
            b_other = "OK   <-- BAD, collision still possible"
            b_ok = True
        except OSError as exc:
            b_other = f"BLOCKED errno={exc.errno}  <-- unexpected on this kernel"
            b_ok = False
        finally:
            other.close()
        print(f"         other leg probe  -> {b_other}")
        child = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        child.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            child.bind(("0.0.0.0", p))
            print("         our child        -> OK")
        except OSError as exc:
            print(f"         our child        -> BLOCKED errno={exc.errno}")
        finally:
            child.close()
    finally:
        res.close()

    print()
    print("   CONCLUSION: exclusivity and let-the-child-take-over are mutually")
    print("   incompatible here, so the reservation MUST be released before")
    print("   returning. What remains fixable is the DETERMINISM of the scan")
    print("   start — hence the randomise fix.")
    return a_ok and b_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, choices=[1, 2, 3], default=None)
    args = ap.parse_args()

    results = {}
    if args.section in (None, 1):
        results["reproduce"] = section1_reproduce()
    if args.section in (None, 2):
        results["fix"] = section2_confirm_fix()
    if args.section in (None, 3):
        results["rejected"] = section3_rejected()

    print()
    print("=" * 72)
    for k, v in results.items():
        print(f"   {k:12s} {'as expected' if v else 'UNEXPECTED — investigate'}")
    print("=" * 72)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
