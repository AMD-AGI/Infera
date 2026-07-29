#!/usr/bin/env python3
"""Verify the two properties the fix must have.

  (A) `voted` is UNIFORM across ranks on every iteration where all ranks logged.
      This is the invariant whose violation caused the deadlock.
  (B) The graph is still actually used -- otherwise the fix is just Variant B
      with extra steps. Measured as: iterations where voted=False on all ranks
      (nobody needs eager -> everyone replays the draft graph).
"""
import collections
import re
import sys

R = re.compile(
    r"GLM52_VOTE dp=(?P<dp>\d+) it=(?P<it>\d+) local=(?P<local>\w+) voted=(?P<voted>\w+)"
)


def main(path):
    recs = []
    with open(path, "rb") as f:
        for raw in f:
            m = R.search(raw.decode("utf-8", "replace"))
            if m:
                recs.append(m.groupdict())
    if not recs:
        print("NO VOTE RECORDS")
        return 1

    by_it = collections.defaultdict(dict)
    for r in recs:
        by_it[int(r["it"])][int(r["dp"])] = r
    ranks = {int(r["dp"]) for r in recs}
    nranks = len(ranks)

    print(f"records={len(recs)} ranks={sorted(ranks)} iterations={len(by_it)}")

    complete = {it: row for it, row in by_it.items() if len(row) == nranks}
    print(f"iterations with all {nranks} ranks logging: {len(complete)}")

    # (A) uniformity of the acted-on value
    bad = []
    local_div = 0
    for it, row in sorted(complete.items()):
        if len({row[dp]["voted"] for dp in row}) > 1:
            bad.append(it)
        if len({row[dp]["local"] for dp in row}) > 1:
            local_div += 1

    print(f"\n(A) iterations where LOCAL diverges (the latent bug): {local_div}")
    print(f"(A) iterations where VOTED diverges (must be 0)     : {len(bad)}")
    if bad:
        print(f"    VIOLATIONS at: {bad[:10]}")
        for it in bad[:2]:
            print(f"    it={it}: " + str({dp: complete[it][dp]["voted"] for dp in complete[it]}))
    else:
        print("    -> the acted-on decision is uniform on every complete iteration")

    # (B) is the graph still used
    all_graph = sum(
        1 for row in complete.values() if all(row[dp]["voted"] == "False" for dp in row)
    )
    all_eager = sum(
        1 for row in complete.values() if all(row[dp]["voted"] == "True" for dp in row)
    )
    print(f"\n(B) all-ranks-GRAPH iterations : {all_graph}")
    print(f"(B) all-ranks-EAGER iterations : {all_eager}")
    if all_graph == 0:
        print("    WARNING: graph never used -- this is Variant B in disguise, NOT a fix")
    else:
        pct = 100.0 * all_graph / max(1, len(complete))
        print(f"    -> draft graph used on {pct:.1f}% of complete iterations")

    flips = sum(1 for r in recs if r["local"] == "False" and r["voted"] == "True")
    print(f"\nvote changed a rank's decision {flips} times")
    print("  (each one is an iteration that would have split graph/eager and deadlocked)")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
