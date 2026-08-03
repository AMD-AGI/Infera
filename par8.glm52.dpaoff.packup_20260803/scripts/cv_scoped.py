import re, sys
from collections import Counter, defaultdict
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS   = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})")
PICK = re.compile(r'role=(?P<role>\w+).*?picked=(?P<picked>\S+).*?'
                  r'cache_hits=(?P<hits>\d+).*?request_blocks=(?P<blocks>\d+)')
SINCE = sys.argv[2] if len(sys.argv) > 2 else None
picks = defaultdict(Counter); hits = defaultdict(list); first=None; last=None; n=0; skipped=0
for line in open(sys.argv[1], errors="replace"):
    s = ANSI.sub("", line)
    m = PICK.search(s)
    if not m: continue
    t = TS.search(s)
    ts = t.group(1) if t else None
    if SINCE and ts and ts < SINCE:
        skipped += 1; continue
    n += 1
    if ts:
        first = first or ts; last = ts
    picks[m.group("role")][m.group("picked")] += 1
    hits[m.group("role")].append(int(m.group("hits")))
print(f"{n} picks in window (skipped {skipped} before {SINCE}); ts range {first} .. {last}")
for role in sorted(picks):
    h = hits[role]
    print(f"\n=== {role} ===  ({sum(picks[role].values())} picks)")
    tot = sum(picks[role].values())
    for tgt, c in sorted(picks[role].items()):
        print(f"  {tgt:34s} {c:5d}  {100*c/tot:5.1f}%")
    print(f"  cache_hits max={max(h)} mean={sum(h)/len(h):.1f} nonzero={sum(1 for x in h if x>0)}/{len(h)}")
