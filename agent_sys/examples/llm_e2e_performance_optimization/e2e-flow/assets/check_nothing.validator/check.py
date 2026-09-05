#!/usr/bin/env python3
"""`check_nothing` — admits every handoff without looking at it.

**This validator exists so that a chain can be made to *walk* before any verdict
is argued about.** The framework will not let a kind go unvalidated —

    handoff kind 'stock.measurement' names no validator.
    A kind with no validator cannot be admitted

— so "turn the validators off" cannot be expressed as an empty list. It is
expressed as this: one validator that reports `true` for every declared handoff
and reads none of them.

**It is never named in a `validators:` list in this package.** It is injected by
`assets/lib/make_debug_package.py` into a *generated copy* of the tree, which
`--package` selects per run. If you find `check_nothing` in a hand-written
`steps/*.yaml`, that is a mistake: it would silently disarm the strict package
that every other chain shares.

### What a green run against this validator establishes, and what it does not

It establishes **reachability**: that the graph's edges resolve, that every body
starts, produces the files its kind declares, and seals; and that stage *n+1*
can be handed what stage *n* wrote. That is the thing five stages chained
together have never yet demonstrated, and it is worth demonstrating on its own.

It establishes **nothing whatever about correctness**. Every verdict this file
writes is `true` by construction, so a green here is the weakest possible
evidence — weaker even than the trivially-passing validators this project has
already been caught by, because those at least opened the file. Do not report a
run against a generated package as "validators pass". The honest sentence is
*"the chain walked all five stages with validation disabled"*.

**`strength` is not this body's to set** — it comes from the `validator:` spec,
and `make_debug_package.py` does not write one, so whatever the framework
defaults to is what a reader will see. That is a real hazard of this file: a
`strong` verdict that read nothing looks exactly like a `strong` verdict that
read everything, and the only thing distinguishing them is the validator's
name. Hence the name.

### Why it still enumerates rather than writing a constant

`zone.write_verdict` wants *one entry per declared handoff* and
`zone.py:133` says a missing one "raises at `PhaseRunner`'s seam rather than
folding as falsy". So the enumeration is load-bearing even though the value is
not: getting the key set wrong would fail the phase for a reason that has
nothing to do with the chain under test, which is precisely the noise this file
is meant to remove.
"""
from __future__ import annotations

import os
import pathlib
import sys

_PKG = pathlib.Path(os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE", ""))
sys.path.insert(0, str(_PKG / "assets" / "lib"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402


def main() -> int:
    hids = zone.inputs()
    # Say so on stderr. A validator that passes in silence is indistinguishable
    # from one that was never asked, and this package has already spent a day
    # on that exact ambiguity.
    print(
        f"check_nothing: admitting {len(hids)} handoff(s) unread — "
        f"validation is DISABLED for this run: {', '.join(hids) or '(none)'}",
        file=sys.stderr,
    )
    zone.write_verdict({hid: True for hid in hids})
    return 0


if __name__ == "__main__":
    sys.exit(main())
