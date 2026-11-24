#!/usr/bin/env python3
"""Merge aiter BF16 tuned-GEMM tables: per (M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle) keep the row with the
lowest measured `us` (both tables were tuned on the same image / GPU model). Prints what changed.
  merge_tuned_gemm.py BASE.csv NEW.csv OUT.csv
"""
import csv
import sys

KEY = ("M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle")


def load(p):
    with open(p) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def us(row):
    try:
        return float(row["us"])
    except (KeyError, ValueError):
        return float("inf")


def main():
    fb, base = load(sys.argv[1])
    _, new = load(sys.argv[2])
    best = {tuple(r[k] for k in KEY): r for r in base}
    changed = added = 0
    for r in new:
        k = tuple(r[c] for c in KEY)
        old = best.get(k)
        if old is None:
            best[k] = r; added += 1
            print(f"add  M={r['M']} N={r['N']} K={r['K']}: {r['libtype']} {us(r):.1f} us")
        elif us(r) < us(old):
            print(f"take M={r['M']} N={r['N']} K={r['K']}: {old['libtype']} {us(old):.1f} -> {r['libtype']} {us(r):.1f} us")
            best[k] = r; changed += 1
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fb, extrasaction="ignore")
        w.writeheader()
        for r in sorted(best.values(), key=lambda r: (int(r["N"]), int(r["K"]), int(r["M"]))):
            w.writerow(r)
    print(f"{len(best)} rows: {changed} replaced by faster ones, {added} added")


if __name__ == "__main__":
    main()
