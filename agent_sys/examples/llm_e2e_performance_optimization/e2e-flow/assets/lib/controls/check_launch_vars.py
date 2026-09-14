#!/usr/bin/env python3
"""Every `--var NAME=` on a launch line must be a name the package reads.

**`agent-sys` accepts an unrecognised `--var` silently.** Measured 2026-09-04 by
m2 and confirmed by the leader against the real package:

    --var totally_made_up_var=xyz --var m2_agent=runner --var measrue_gpu=4
      ->  6 tasks in the graph; nothing was dispatched
      ->  rc=0, zero warnings anywhere

So a launch-line var is **indistinguishable from a working knob** whether or not
the package has ever heard the name. That is CONTRACT section 4.4's sixth face
with the volume up: rung 2 of that ladder assumes the var works and is merely
invisible to its owner; this one need not work at all.

**The failure it produces is the nasty shape.** `--var measrue_gpu=4` -- one
transposition -- does nothing, and the body then refuses with
`FIX: pass --var measure_gpu=<n>`, which the operator reads as *"but I did pass
it"*. The refusal is correct, the instruction is correct, and they do not meet.

Audited the leader's own rung-0 line the day this was written: **20 vars, one
bogus** -- `m2_agent`, which exists in no yaml because m2's three leaves are
literally `agent: runner`. Harmless only because that line also named m2 in
`mock_stages`, so the intended effect happened for an unintended reason.

    python3 check_launch_vars.py <package-dir> NAME=VALUE [NAME=VALUE ...]
    python3 check_launch_vars.py <package-dir> --from-line '<a full command>'

Exit 0 all known, 1 at least one unknown, 2 nothing to check -- *cannot judge*
kept apart from *judged and clean*.

**What it does NOT check.** That the value is sane, that the var reaches the
closure you meant, or that a var you *omitted* was needed -- `transport_env` is
consumed by the runner and appears in no yaml at all, so this tool is structurally
blind to it, which is why it cost three rung-0 runs. Names only.
"""
from __future__ import annotations

import glob
import os
import re
import sys

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:}]")
_ONCMD = re.compile(r"--var\s+([A-Za-z_][A-Za-z0-9_]*)=")


def declared(pkg: str) -> set[str]:
    files = glob.glob(os.path.join(pkg, "*.yaml")) + glob.glob(os.path.join(pkg, "steps", "*.yaml"))
    text = "".join(open(f, errors="replace").read() for f in files)
    return set(_VAR.findall(text))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-6], file=sys.stderr)
        return 2
    pkg, rest = argv[0], argv[1:]
    if rest[0] == "--from-line":
        used = _ONCMD.findall(" ".join(rest[1:]))
    else:
        used = [a.split("=", 1)[0] for a in rest if "=" in a]
    known = declared(pkg)
    if not known:
        print(f"no ${{var}} references under {pkg} -- CANNOT JUDGE, not clean", file=sys.stderr)
        return 2
    if not used:
        print("no --var on the line -- CANNOT JUDGE, not clean", file=sys.stderr)
        return 2
    bad = [v for v in used if v not in known]
    print(f"{len(used)} var(s) against {len(known)} declared in {pkg}")
    for v in used:
        print(f"  {v:18} {'ok' if v in known else 'UNKNOWN -- the package never reads this'}")
    if bad:
        print(f"\n{len(bad)} unknown: {', '.join(bad)}")
        print("agent-sys will accept these silently. Check for a typo before spending a hold.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
