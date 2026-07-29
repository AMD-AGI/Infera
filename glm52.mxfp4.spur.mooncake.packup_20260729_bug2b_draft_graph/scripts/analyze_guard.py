#!/usr/bin/env python3
"""Diff the R1 guard-decision records across DP ranks, per iteration.

Reads a server log (binary-safe) and answers, in order:

  H2  is `t1_cangraph` rank-divergent?  -> the all-gathered vote is broken (worse bug)
  H1  is `t4_topknone` rank-divergent while t1..t3 agree?  -> the guard's term 4
  H3  is `padmode` rank-divergent, or does it differ from the captured MAX_LEN?

An iteration index is per-rank (each rank counts its own draft() calls), so
iterations only line up if all ranks entered draft() the same number of times.
That itself is diagnostic: a rank that stops incrementing is the one that blocked.
"""
import collections
import re
import sys

REC = re.compile(
    r"GLM52_R1 dp=(?P<dp>\d+) it=(?P<it>\d+) mode=(?P<mode>\w+) bs=(?P<bs>\S+) "
    r"nreq=(?P<nreq>\S+) t1_cangraph=(?P<t1>\w+) t2_notidle=(?P<t2>\w+) "
    r"t3_seed=(?P<t3>\w+) t4_topknone=(?P<t4>\w+) final=(?P<final>\w+) "
    r"gnt=(?P<gnt>\[[^\]]*\]|None) can_dp_cg=(?P<cdg>\w+) padmode=(?P<pm>\w+)"
)


def main(path):
    recs = []
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace")
            m = REC.search(line)
            if m:
                recs.append(m.groupdict())

    if not recs:
        print("NO RECORDS -- probe did not fire (check GLM52_R1_PROBE=1 and the .pyc)")
        return 1

    print(f"total records: {len(recs)}")
    by_rank = collections.defaultdict(list)
    for r in recs:
        by_rank[int(r["dp"])].append(r)

    print("\n=== per-rank totals (a rank stuck below the others is the blocker) ===")
    for dp in sorted(by_rank):
        rs = by_rank[dp]
        ngraph = sum(1 for r in rs if r["final"] == "True")
        neager = len(rs) - ngraph
        print(
            f"dp{dp}: draft_calls={len(rs):5d} graph={ngraph:5d} eager={neager:5d} "
            f"last_it={rs[-1]['it']}"
        )

    # Align by iteration index and look for divergence
    by_it = collections.defaultdict(dict)
    for r in recs:
        by_it[int(r["it"])][int(r["dp"])] = r

    ranks = sorted(by_rank)
    print(f"\n=== divergence scan over {len(by_it)} iterations, {len(ranks)} ranks ===")

    div = collections.Counter()
    examples = {}
    for it in sorted(by_it):
        row = by_it[it]
        if len(row) < len(ranks):
            continue  # incomplete iteration -- skip for the field scan
        for field in ("t1", "t2", "t3", "t4", "final", "cdg", "pm", "gnt"):
            vals = {row[dp][field] for dp in row}
            if len(vals) > 1:
                div[field] += 1
                examples.setdefault(field, (it, {dp: row[dp][field] for dp in row}))

    if not div:
        print("no field diverges on any complete iteration")
    for field, n in div.most_common():
        it, ex = examples[field]
        print(f"  {field:6s} diverges on {n:5d} iterations; first at it={it}: {ex}")

    print("\n=== verdict ===")
    if div.get("t1"):
        print("H2 SUPPORTED: can_run_graph itself is rank-divergent.")
        print("  The all-gathered vote (dp_attn.py:111 min over tp0_info[:,2]) is not")
        print("  holding. Investigate before anything else -- this is the deeper bug.")
    elif div.get("t4") and not (div.get("t2") or div.get("t3")):
        print("H1 SUPPORTED: term 4 (dsa_topk_indices is None) is the sole divergence.")
        print("  Fix: make term 4 a DP-group decision.")
    elif div.get("t4") and div.get("t2"):
        print("H1 PARTIAL: both term 2 (is_idle) and term 4 diverge.")
        print("  Removing either alone will not make the guard uniform -- matches the")
        print("  failed 'drop not is_idle()' attempt. The decision as a whole must be voted.")
    elif div.get("pm"):
        print("H3 SUPPORTED: dp_padding_mode is rank-divergent")
        print("  -> all_gather_into_tensor vs all_reduce mismatch between ranks.")
    elif div.get("final"):
        print("final decision diverges but no single logged term does --")
        print("  look for a term evaluated inside can_run_graph that is not logged.")
    else:
        print("NO DIVERGENCE FOUND on complete iterations.")
        print("  Either the hang is not caused by a divergent decision, or the")
        print("  divergence happens on an iteration where some rank never logged")
        print("  (check the per-rank totals above -- unequal counts ARE the signal).")

    # padmode distribution regardless of divergence -- H3 needs this even if uniform,
    # because uniform-SUM_LEN eager still mismatches the captured MAX_LEN graph.
    print("\n=== padmode distribution (captured graphs use MAX_LEN) ===")
    pm = collections.Counter(r["pm"] for r in recs)
    for k, v in pm.most_common():
        print(f"  {k}: {v}")
    pm_by_final = collections.Counter((r["final"], r["pm"]) for r in recs)
    print("  (final_decision, padmode):")
    for k, v in pm_by_final.most_common():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "decode.log"))
