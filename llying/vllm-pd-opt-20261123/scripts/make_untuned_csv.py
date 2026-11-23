#!/usr/bin/env python3
"""Untuned-GEMM input for aiter's gemm_a16w16_tune.py from a list of M,N,K lines, with the column layout of the image's
own configs/bf16_untuned_gemm.csv (columns we do not set take that file's first data row's values).
  make_untuned_csv.py SHAPES_TXT TEMPLATE_CSV OUT_CSV
"""
import csv
import sys

shapes, template, out = sys.argv[1:4]
with open(template) as f:
    rows = list(csv.reader(f))
hdr = rows[0]
base = dict(zip(hdr, rows[1])) if len(rows) > 1 else {}
fixed = {"bias": "False", "dtype": "torch.bfloat16", "outdtype": "torch.bfloat16", "scaleAB": "False", "bpreshuffle": "False"}
seen = set()
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for line in open(shapes):
        line = line.strip()
        if not line:
            continue
        m, n, k = (int(x) for x in line.split(","))
        if (m, n, k) in seen:
            continue
        seen.add((m, n, k))
        r = dict(base)
        r.update(fixed)
        r.update({"M": str(m), "N": str(n), "K": str(k)})
        w.writerow([r.get(c, "") for c in hdr])
print(f"{len(seen)} shapes, columns {hdr}")
