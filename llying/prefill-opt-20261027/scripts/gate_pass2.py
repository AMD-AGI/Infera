"""Pass / fail of one quick_correct.sh run (full suite: smoke, probe, long-context lines of any n) relative to the current
stack, which has a pre-existing ~0.4 % bad-start rate in the probe (10-28: E1 path not exercised, 5 / 1319).
Pass = smoke 6/6; probe correct >= --probe-min and garbled <= --probe-garbled-max; every long-context line garbled 0,
needle >= n - max(1, n // 32), long-prefix GSM8K >= n - max(2, n // 16); at least --passes long-context lines.
  python3 gate_pass2.py DRIVER_OUT LABEL_PREFIX [--passes 2] [--probe-min 0.955] [--probe-garbled-max 6]"""

import argparse
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("label")
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--probe-min", type=float, default=0.955)
ap.add_argument("--probe-garbled-max", type=int, default=6)
a = ap.parse_args()
lines = [l for l in open(a.out, errors="replace") if f" {a.label}" in l]
why = []
smoke = [l for l in lines if " smoke: " in l]
if not smoke or "6/6 checks passed" not in smoke[-1]:
    why.append(f"smoke: {smoke[-1].strip() if smoke else 'missing'}")
probe = [l for l in lines if " probe: " in l]
m = re.search(r"correct=([\d.]+) garbled=(\d+)", probe[-1]) if probe else None
if not m or float(m.group(1)) < a.probe_min or int(m.group(2)) > a.probe_garbled_max:
    why.append(f"probe: {probe[-1].strip() if probe else 'missing'}")
lc = [l for l in lines if " longctx " in l]
if len(lc) < a.passes:
    why.append(f"long-context passes {len(lc)} < {a.passes}")
for l in lc:
    nd = re.search(r"needle: n=(\d+) correct=(\d+) garbled=(\d+)", l)
    gs = re.search(r"gsm8k: n=(\d+) correct=(\d+) garbled=(\d+)", l)
    if not nd or not gs:
        why.append(f"longctx unparsed: {l.strip()}")
        continue
    n1, c1, g1 = map(int, nd.groups())
    n2, c2, g2 = map(int, gs.groups())
    if g1 or g2 or c1 < n1 - max(1, n1 // 32) or c2 < n2 - max(2, n2 // 16):
        why.append(f"longctx: {l.strip()}")
print("PASS" if not why else "FAIL: " + " | ".join(why))
sys.exit(0 if not why else 1)
