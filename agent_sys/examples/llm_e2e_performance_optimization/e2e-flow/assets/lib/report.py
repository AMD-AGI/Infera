#!/usr/bin/env python3
"""Render a probe sweep's rows into a verdict table.

Separate from the driver so a sweep can be **re-read without being re-run**.
The rows carry every fact the verdict uses, so a change of mind about what
disqualifies a node costs a re-render and not another 57 jobs — which is the
whole reason the payload emits JSON instead of prose.

    ./report.py results/<stamp>/rows.jsonl [--need 4] [--disk 200] [--asked a,b,c]

**Three tiers, not two, and the middle one is a correction.** The first version
scored "no local base carrying m1's anchor" as `NO`, which put nodes like
`crsuse2-m2m-179` — eight free cards, 22 T of disk, simply no sglang image
pulled yet — in the same bucket as a node with eight cards under another
tenant. Those are not the same answer: one is unusable, the other costs an image
pull. Reported as one, the table would have hidden most of the usable capacity
on the cluster behind a reason that is not a blocker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: `unknown` means the grep did not run — a private registry image that will not
#: start, or a base with no `serving_responses.py`. It is not `no`, and folding
#: it into `no` would report an unmeasured thing as a measured failure.
ANCHOR_OK = "yes"


def verdict(r: dict, need: int, need_disk: int, need_root: int) -> tuple[str, str]:
    """One tier and one reason, the *first* disqualifying fact.

    Ordered by what costs most to discover late: cards decide whether the node
    is usable at all, disk decides whether an image can even land, and the base
    only refines a node that already passed both.
    """
    if r["cards_free"] < need:
        return "NO", f"{r['cards_free']}/{r['cards_total']} cards free — {r['busy'] or 'n/a'}"
    if r["disk_gb"] < need_disk:
        return "NO", f"{r['disk_gb']}G on /mnt/m2m_nobackup"
    # **`/` is the one that stopped a build.** `crsuse2-m2m-186` had the right
    # base and 3.4 G here; docker builds on `/`, so a node can pass the big
    # filesystem and still be unbuildable. A separate, smaller bar, because this
    # one holds a build tree rather than an image store.
    if r.get("root_gb", 0) < need_root:
        return "NO", f"{r.get('root_gb', 0)}G on / — docker builds there"
    # The daemon's authorization plugin, measured on 243. Its refusal names
    # neither docker's usual vocabulary nor the variable at fault, so a node it
    # would reject is worth knowing about before the hold, not during rung 3.
    if r.get("mounts") == "denied":
        return "NO", "spur-authz refuses this flow's mounts"
    ok = [b["image"] for b in r["bases"] if b["anchor"] == ANCHOR_OK]
    if ok:
        return "READY", f"cards {r['free']} free, {r['disk_gb']}G, base {ok[0]}"
    seen = ", ".join(f"{b['image'].split('/')[-1]}={b['anchor']}" for b in r["bases"])
    why = seen or "no sglang image local"
    return "USABLE", f"cards {r['free']} free, {r['disk_gb']}G, needs a base pulled ({why})"


ORDER = {"READY": 0, "USABLE": 1, "NO": 2, "?": 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--need", type=int, default=4)
    ap.add_argument("--disk", type=int, default=200)
    ap.add_argument("--root", type=int, default=20,
                    help="free GB required on / — docker builds there; 186 had 3.4")
    ap.add_argument("--asked", default="",
                    help="comma-separated nodes that were probed, so ones that "
                         "produced no row are reported rather than dropped")
    a = ap.parse_args()

    rows: dict[str, dict] = {}
    p = Path(a.rows)
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["node"]] = r

    asked = [n for n in a.asked.split(",") if n] or sorted(rows)
    table = []
    for n in asked:
        r = rows.get(n)
        if r is None:
            # Not the same as unsuitable, and only one of the two is worth
            # re-checking later.
            table.append((n, "?", "no row — probe did not run (queue, or the node refused the job)"))
        else:
            v, why = verdict(r, a.need, a.disk, a.root)
            table.append((n, v, why))
    table.sort(key=lambda t: (ORDER[t[1]], t[0]))

    print(f"{'='*100}\nevery node checked, and what each one is\n{'='*100}")
    print(f"  {'node':<22} {'verdict':<8} reason")
    print(f"  {'-'*22} {'-'*8} {'-'*62}")
    for n, v, why in table:
        print(f"  {n:<22} {v:<8} {why}")

    for tier, blurb in (("READY", "free half, disk, and a local base carrying m1's anchor"),
                        ("USABLE", "free half and disk — costs one image pull")):
        got = [n for n, v, _ in table if v == tier]
        print(f"\n  {tier}: {len(got)} — {blurb}")
        if got:
            print("    " + ", ".join(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
