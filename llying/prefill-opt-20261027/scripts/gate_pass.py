"""Pass / fail of one quick_correct.sh run from its summary lines (the driver output), against the full llr35 + v3
reference (smoke 6/6; 1319-probe 0.972, garbled 0; 16k x 128: needle 64/64, long-prefix GSM8K 63/64).
Pass = smoke 6/6, probe garbled 0 and correct >= --probe-min, every long-context pass garbled 0 with needle >= --needle-min
and GSM8K >= --gsm8k-min (of 64), and at least --passes long-context lines. Prints the verdict; exit 0 = pass.
  python3 gate_pass.py DRIVER_OUT LABEL_PREFIX [--passes 2]"""

import argparse
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("label")
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--probe-min", type=float, default=0.955)
ap.add_argument("--needle-min", type=int, default=62)
ap.add_argument("--gsm8k-min", type=int, default=60)
a = ap.parse_args()
lines = [l for l in open(a.out, errors="replace") if f" {a.label}" in l]
why = []
smoke = [l for l in lines if " smoke: " in l]
if not smoke or "6/6 checks passed" not in smoke[-1]:
    why.append(f"smoke: {smoke[-1].strip() if smoke else 'missing'}")
probe = [l for l in lines if " probe: " in l]
m = re.search(r"correct=([\d.]+) garbled=(\d+)", probe[-1]) if probe else None
if not m or float(m.group(1)) < a.probe_min or int(m.group(2)) != 0:
    why.append(f"probe: {probe[-1].strip() if probe else 'missing'}")
lc = [l for l in lines if " longctx " in l]
if len(lc) < a.passes:
    why.append(f"long-context passes {len(lc)} < {a.passes}")
for l in lc:
    n = re.search(r"needle: n=\d+ correct=(\d+) garbled=(\d+)", l)
    g = re.search(r"gsm8k: n=\d+ correct=(\d+) garbled=(\d+)", l)
    if not n or not g or int(n.group(1)) < a.needle_min or int(g.group(1)) < a.gsm8k_min or n.group(2) != "0" or g.group(2) != "0":
        why.append(f"longctx: {l.strip()}")
print("PASS" if not why else "FAIL: " + " | ".join(why))
sys.exit(0 if not why else 1)
