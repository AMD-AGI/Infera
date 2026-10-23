# xsub (task dcpx9/10 2026-10-22): sum pack_sub(index,launch,sync,n) of [yihou-xfer] lines after HH:MM:SS.
import re, sys
f, t0 = sys.argv[1], sys.argv[2]; s = [0.0] * 4
for line in open(f, errors="replace"):
    if "[yihou-xfer]" not in line: continue
    m = re.search(r"\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d) DP", line)
    if not m or m.group(1) < t0: continue
    p = re.search(r"pack_sub\(index,launch,sync,n\)=\[([^\]]*)\]", line)
    if p and p.group(1).strip():
        s = [a + float(b) for a, b in zip(s, p.group(1).split(","))]
n = max(s[3], 1)
print(f"index {s[0]:.1f}s launch {s[1]:.1f}s sync {s[2]:.1f}s n={s[3]:.0f} -> ms/pack index {1e3*s[0]/n:.2f} launch {1e3*s[1]/n:.2f} sync {1e3*s[2]/n:.2f}")
