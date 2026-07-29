import json, glob, os
from collections import Counter

def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join("/home/yihou/glm52_fix/track_data", pat))):
        rows = [json.loads(l) for l in open(f)]
        if not any(r.get("http") == 200 for r in rows):
            continue          # a round that hit a dead server proves nothing
        out.append((os.path.basename(f), rows))
    return out

for label, pat in (("MTP  (spec-dec ON )", "st-r*.jsonl"),
                   ("noMTP(spec-dec OFF)", "nomtp-r*.jsonl")):
    groups = load(pat)
    tot = bad = 0
    ranks = Counter(); badranks = Counter(); c1 = Counter()
    for f, rows in groups:
        ok = [r for r in rows if r.get("http") == 200]
        b = [r for r in ok if r.get("degenerate")]
        tot += len(ok); bad += len(b)
        for r in ok:
            ranks[r.get("dp_rank_decode")] += 1
        for r in b:
            badranks[r.get("dp_rank_decode")] += 1
            ac = r.get("all_chunks") or []
            if len(ac) > 1:
                c1[ac[1]] += 1
    print(f"\n=== {label} ===")
    print(f"  rounds={len(groups)} requests={tot} degenerate={bad} "
          f"({100*bad/tot:.2f}%)" if tot else "  no data")
    key = lambda k: (k is None, k)
    print(f"  decode ranks affected: {sorted(badranks, key=key)} "
          f"({len(badranks)} of {len(ranks)})")
    print(f"  per-rank degenerate  : {dict(sorted(badranks.items(), key=lambda x:(x[0] is None,x[0])))}")
    print(f"  2nd token of bad reqs: {dict(c1.most_common(6))}")
