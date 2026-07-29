import json, glob, os
from collections import Counter

D = "/home/yihou/glm52_fix/track_data"

ARMS = [
    ("MTP  + PD  (patched, custom-AR OFF)", "st-r*.jsonl"),
    ("noMTP+ PD  (clean,  custom-AR ON )", "nomtp-r*.jsonl"),
    ("noMTP+ PD  (clean,  custom-AR OFF)", "exp1-r*.jsonl"),
    ("noMTP+ MIX (clean,  custom-AR OFF)", "exp2-r*.jsonl"),
]

print(f"{'arm':38s} {'reqs':>5s} {'bad':>4s} {'rate':>6s}  ranks  2nd-token-of-bad")
print("-" * 96)
for label, pat in ARMS:
    tot = bad = 0
    ranks, badranks, c1 = Counter(), Counter(), Counter()
    atzero = nbi = 0
    for f in sorted(glob.glob(os.path.join(D, pat))):
        rows = [json.loads(l) for l in open(f)]
        ok = [r for r in rows if r.get("http") == 200]
        if not ok:
            continue
        # the stale no-MTP runs carry dp_rank=None everywhere; drop them so the
        # rank column means something
        if all(r.get("dp_rank_decode") is None for r in ok):
            continue
        b = [r for r in ok if r.get("degenerate")]
        tot += len(ok); bad += len(b)
        for r in ok:
            ranks[r.get("dp_rank_decode")] += 1
        for r in b:
            badranks[r.get("dp_rank_decode")] += 1
            ac = r.get("all_chunks") or []
            if len(ac) > 1:
                c1[ac[1]] += 1
            fb = r.get("first_bad_chunk")
            if fb is not None:
                nbi += 1
                if fb == 0:
                    atzero += 1
    if not tot:
        print(f"{label:38s}  (no usable data)")
        continue
    print(f"{label:38s} {tot:5d} {bad:4d} {100*bad/tot:5.2f}%  "
          f"{len(badranks)}/{len(ranks)}    {dict(c1.most_common(3))}")
    if nbi:
        print(f"{'':38s}        first_bad_chunk==0 in {atzero}/{nbi} of the ones "
              f"the loop-detector could locate")
