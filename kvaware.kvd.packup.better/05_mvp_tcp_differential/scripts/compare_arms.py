#!/usr/bin/env python3
"""Compare the two arms of the differential and state what it does — and does
NOT — license you to conclude.

Reads the two JSON files written by probe.py --json and applies the decision
table below. It deliberately refuses to say "PASS" for the case this experiment
actually hit, because that case is a no-regression observation, not a
correctness result, and the distinction is the entire value of the round.

Usage:
    python3 compare_arms.py /tmp/armA.json /tmp/armB.json
"""

from __future__ import annotations

import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarise(d: dict) -> str:
    return (
        f"arm {d.get('arm','?')}: {d['ok']}/{d['ok']+d['wrong']+d['errored']} correct, "
        f"{d['wrong']} wrong-content, {d['errored']} errored"
        + (f", {d['kvcache_failures']} KV-transfer" if d.get("kvcache_failures") else "")
    )


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])

    print("=" * 72)
    print("DIFFERENTIAL: kvaware+kvd ON (arm A) vs OFF (arm B)")
    print("=" * 72)
    print(f"  {summarise(a)}")
    print(f"  {summarise(b)}")
    print()

    # Per-prompt side by side. The opening tokens are the most diagnostic field.
    print("--- per prompt ---")
    for ra, rb in zip(a["records"], b["records"]):
        print(f"  {ra['prompt']!r}")
        print(f"    A [{ra['status']:5s}] {ra.get('text', ra.get('detail'))!r}")
        print(f"    B [{rb['status']:5s}] {rb.get('text', rb.get('detail'))!r}")
        fa, fb = ra.get("first_16"), rb.get("first_16")
        if fa is not None and fb is not None:
            common = ""
            for ca, cb in zip(fa, fb):
                if ca != cb:
                    break
                common += ca
            if common.strip():
                print(f"    ^ both arms open with the same prefix: {common!r}")
    print()

    a_ok, b_ok = a["ok"] >= 3, b["ok"] >= 3
    a_err, b_err = a["errored"] > 0, b["errored"] > 0

    print("--- verdict ---")
    if a_err or b_err:
        print("  INCONCLUSIVE — at least one arm errored rather than answering.")
        print("  An arm that never completed a request says nothing about content.")
        print("  Fix the transport first, then re-run both arms.")
        return 3

    if a_ok and b_ok:
        print("  BOTH ARMS CORRECT.")
        print("  This is a no-regression result: the features did not break")
        print("  anything. It is NOT evidence that they DID anything — for that")
        print("  you need a workload that exercises them (e.g. a long shared")
        print("  prefix) plus the engine's own counters.")
        return 0

    if a_ok and not b_ok:
        print("  ARM A CORRECT, ARM B WRONG. Unexpected: the baseline is broken.")
        print("  Investigate the baseline before drawing any conclusion about A.")
        return 4

    if b_ok and not a_ok:
        print("  ARM A WRONG, ARM B CORRECT.")
        print("  *** THIS is the shape that would indict kvaware/kvd. *** Same")
        print("  substrate, same everything, only the switches differ, and only")
        print("  the switched-on arm is broken.")
        return 5

    # Neither arm correct — the case this experiment actually hit.
    print("  NEITHER ARM CORRECT.")
    print()
    print("  What this DOES establish:")
    print("    The failure is NOT caused by kvaware or kvd. With both features")
    print("    off, the output is just as wrong. Same node, same transport, same")
    print("    model, same ports — the only difference was the switches, and it")
    print("    made no difference to the outcome.")
    print()
    print("  What this does NOT establish:")
    print("    That kvaware/kvd are CORRECT. Nothing here was correct. This is a")
    print("    NO-REGRESSION observation on a broken substrate, not a correctness")
    print("    pass. You cannot cite it as 'kvaware+kvd verified'.")
    print()
    print("  What to do next:")
    print("    Remove the shared substrate from the equation. Here that meant")
    print("    moving to two nodes with real RDMA, which eliminates both the")
    print("    same-host loopback AND the MC_FORCE_TCP path in one step.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
