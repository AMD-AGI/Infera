#!/usr/bin/env python3
"""Fold newly tuned aiter BF16 GEMM rows into the tables the engine reads, without creating duplicate shapes.

aiter merges configs/bf16_tuned_gemm.csv with every model_configs/*bf16_tuned_gemm*.csv; a shape present in two files
makes it rewrite the sources (keeping the lowest `us`) and raise, which fails on read-only mounts. So, per new row:
- if no image table has the shape, it goes into OUR table (the glm52 table, merged with the rows already there,
  lowest `us` wins);
- if an image table has it and the new row is faster, a modified copy of that table is written (row replaced, same
  key) to be mounted over the original; otherwise the row is dropped.
Run inside the vLLM image (it reads the image's tables).
  merge_tuned_gemm2.py OUR_TABLE NEW_ROWS OUT_DIR
Writes OUT_DIR/glm52_bf16_tuned_gemm.csv, OUT_DIR/img/<name>.csv for each modified image table, and OUT_DIR/mounts.txt
("src:dst" per line, relative to OUT_DIR for src).
"""
import csv
import glob
import os
import sys

A = "/usr/local/lib/python3.12/dist-packages/aiter/configs"
KEY = ("cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle")


def load(p):
    with open(p) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def key(r):
    return tuple(str(r.get(k, "")).strip() for k in KEY)


def us(r):
    try:
        return float(r["us"])
    except (KeyError, ValueError):
        return float("inf")


def write(p, fields, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ours_p, new_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
    of, ours = load(ours_p)
    _, new = load(new_p)
    image = [p for p in [f"{A}/bf16_tuned_gemm.csv"] + sorted(glob.glob(f"{A}/model_configs/*bf16_tuned_gemm*.csv"))
             if os.path.basename(p) != "glm52_bf16_tuned_gemm.csv"]
    tables = {p: load(p) for p in image}
    where = {}
    for p, (_, rows) in tables.items():
        for i, r in enumerate(rows):
            where.setdefault(key(r), (p, i))
    ours_by = {key(r): r for r in ours}
    touched = set()
    for r in new:
        k = key(r)
        if k in where:
            p, i = where[k]
            old = tables[p][1][i]
            if us(r) < us(old):
                print(f"image {os.path.basename(p)} M={r['M']} N={r['N']} K={r['K']}: {old.get('libtype')} {us(old):.1f} -> {r['libtype']} {us(r):.1f} us")
                tables[p][1][i] = {**old, **{c: r[c] for c in r if c in tables[p][0]}}
                touched.add(p)
            else:
                print(f"keep  {os.path.basename(p)} M={r['M']} N={r['N']} K={r['K']}: {us(old):.1f} <= new {us(r):.1f} us")
        else:
            old = ours_by.get(k)
            if old is None or us(r) < us(old):
                print(f"ours  M={r['M']} N={r['N']} K={r['K']}: {(old or {}).get('libtype', '-')} {us(old) if old else float('nan'):.1f} -> {r['libtype']} {us(r):.1f} us")
                ours_by[k] = r
    write(f"{out}/glm52_bf16_tuned_gemm.csv", of, sorted(ours_by.values(), key=lambda r: (int(r["N"]), int(r["K"]), int(r["M"]))))
    with open(f"{out}/mounts.txt", "w") as f:
        f.write(f"glm52_bf16_tuned_gemm.csv:{A}/model_configs/glm52_bf16_tuned_gemm.csv\n")
        for p in sorted(touched):
            name = os.path.basename(p)
            write(f"{out}/img/{name}", tables[p][0], tables[p][1])
            f.write(f"img/{name}:{p}\n")
    print(f"ours: {len(ours_by)} rows; modified image tables: {[os.path.basename(p) for p in sorted(touched)]}")


if __name__ == "__main__":
    main()
