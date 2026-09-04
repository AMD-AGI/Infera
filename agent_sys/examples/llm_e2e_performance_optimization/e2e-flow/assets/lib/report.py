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
    # **The tier that cost m5 a hold.** They took 037 on a SERVABLE verdict and
    # released it: the node has no `/shared_nfs`. A servable image is a fact
    # about the node's docker; the mount is a fact about the node, and this
    # package runs bodies on the node by absolute path
    # (`remote.sh:170 require_visible_on_node`), so a missing shared filesystem
    # fails three layers from its symptom. Ordered above the image checks
    # because no image makes up for it.
    if r.get("shared_nfs") is False:
        return "NO", "no /shared_nfs on the node — the weights and the run root live there"
    if r.get("model_path") is False:
        return "NO", "/shared_nfs is mounted but the model path is not on it"
    if r.get("mounts") == "denied":
        return "NO", "spur-authz refuses this flow's mounts"
    # **`SERVABLE` outranks `BUILDABLE` and the gap is a four-minute build.**
    # An image whose `infera.engine.sglang` and `infera.server` import can be
    # brought up as-is; 006 and 037 were both in that state and neither needed
    # a build. Reported as its own tier because the caller's next action
    # differs, which is the only thing a tier is for.
    serve = [b for b in r["bases"] if b.get("servable")]
    if serve:
        # The config class is reported beside the image, never gated on. m5's
        # check answers "can this image read *these* weights", which is
        # independent of "can this image serve": measured on 217,
        # `rocm/atom-dev:sglang-latest` reads `Qwen3_5Config` fine and has no
        # `infera` at all, so a config check alone would have promoted an image
        # that cannot serve at any version.
        cfg = serve[0].get("model_config") or "config unread"
        return "SERVABLE", (f"cards {r['free']} free, {r['disk_gb']}G, "
                            f"serves now from {serve[0]['image']} ({cfg})")
    ok = [b["image"] for b in r["bases"] if b["anchor"] == ANCHOR_OK]
    if ok:
        # **`BUILDABLE`, not `READY`** — m1's correction, 2026-09-04, and it
        # changes what a caller may promise. A base carrying the anchor means
        # `Dockerfile.sglang` *will build* here; it does **not** mean a servable
        # image exists. `006` had one only because a co-tenant left it behind.
        # The old label invited "ready to launch", which is a four-minute build
        # away from true.
        return "BUILDABLE", f"cards {r['free']} free, {r['disk_gb']}G, builds from {ok[0]}"
    seen = ", ".join(f"{b['image'].split('/')[-1]}={b['anchor']}" for b in r["bases"])
    why = seen or "no sglang image local"
    return "USABLE", f"cards {r['free']} free, {r['disk_gb']}G, needs a base pulled ({why})"


ORDER = {"SERVABLE": 0, "BUILDABLE": 1, "USABLE": 2, "NO": 3, "?": 4}


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

    # **Timestamp the table, because a reader acts on it minutes later.**
    # The leader picked 235 over 006 on a reading that was true when taken and
    # false four minutes on: a co-tenant took all eight cards. A node is clean
    # at the moment you measure it and not afterwards, so the table says when.
    import datetime as _dt
    rows_mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S") if p.is_file() else "?"
    print(f"{'='*100}\nevery node checked, and what each one is"
          f"   — last row written {rows_mtime}, "
          f"rendered {_dt.datetime.now().strftime('%H:%M:%S')}\n{'='*100}")
    print("  A node is clean at the moment it was measured and not afterwards.")
    print(f"  {'node':<22} {'verdict':<8} reason")
    print(f"  {'-'*22} {'-'*8} {'-'*62}")
    for n, v, why in table:
        print(f"  {n:<22} {v:<8} {why}")

    # **The tiers are costs, not grades**, which is why the recovery price is on
    # each. m5 measured it on 047: `docker load` of
    # `/shared_nfs/yihou/images/infera-sglang-local.tar` (27 G) took **4m44s**
    # and the node passed all three checks afterwards. So a node with no infera
    # image is not disqualified — it is five minutes away, and knowing that
    # before binding is worth those five minutes of planning rather than a
    # failed bring-up.
    for tier, blurb in (("SERVABLE", "brings up as-is — no build, no load. 006 was in this state; "
                                     "037 also was, and was still useless for want of /shared_nfs"),
                        ("BUILDABLE", "free half, disk, and a base to build from — NOT a servable "
                                      "image; ~4m44s to load infera-sglang-local.tar, then build"),
                        ("USABLE", "free half and disk — costs one image pull or load, then a build")):
        got = [n for n, v, _ in table if v == tier]
        print(f"\n  {tier}: {len(got)} — {blurb}")
        if got:
            print("    " + ", ".join(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
