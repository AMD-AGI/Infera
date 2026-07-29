import json, glob, re

bad, ok, missing = [], 0, 0
for f in sorted(glob.glob("/tmp/sr_*.json")):
    try:
        s = json.load(open(f))["text"].strip()
    except Exception:
        missing += 1
        continue
    if len(set(s)) < 12 or re.search(r"(.)\1{30,}", s) or s.count("!") > len(s) * 0.3:
        bad.append((f.split("/")[-1], repr(s)[:50]))
    else:
        ok += 1
print(f"coherent={ok} degenerate={len(bad)} unreadable={missing}")
for b in bad[:6]:
    print("     BAD", b[0], b[1])
