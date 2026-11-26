#!/usr/bin/env python3
"""Turn aiter's a8w8 bpreshuffle tuner output into a table the engine can mount next to the image's own tables.

aiter (0.1.28.post1, jit/core.py get_config_file) builds AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE by
merging configs/a8w8_bpreshuffle_tuned_gemm.csv with every model_configs/*a8w8_bpreshuffle_tuned_gemm*.csv (name without
"untuned") into /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv. A shape key (gfx, cu_num, M, N, K, q_dtype_w)
present twice makes it rewrite the sources and RAISE, so our file must not repeat a key any image table has. vLLM's
"is this (N, K) tuned" check (_aiter_ops._ck_gemm_shape_is_tuned) and aiter's runtime lookup both read that merged
file, so one extra file under model_configs/ covers both.

Per (N, K), all or nothing: vLLM switches the WHOLE (N, K) to CK bpreshuffle as soon as ONE M is tuned, and an M with
no row (after aiter's padded-M walk) runs the untuned default kernel. So an (N, K) whose required Ms are not all
tuned is left out (it stays on hipBLASLt), unless --allow-partial.

Run inside the vLLM image (reads the image's tables).
  merge_tuned_fp8_bpreshuffle.py NEW_TUNED_CSV OUT_DIR --ms 1,2,4,... [--allow-partial]
Writes OUT_DIR/a8w8_bpreshuffle_tuned_gemm_glm52_ptpc.csv (ours only), OUT_DIR/merged-preview.csv (image tables + ours,
the union aiter will see), OUT_DIR/mounts.txt (the CSV line only; the job appends the rebuilt JIT modules) and prints
a per-(N, K) coverage report.
"""
import argparse
import csv
import glob
import math
import os
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
A = f"{SP}/aiter/configs"
NAME = "a8w8_bpreshuffle_tuned_gemm_glm52_ptpc.csv"
KEY = ("gfx", "cu_num", "M", "N", "K", "q_dtype_w")
LIBS = ("ck", "cktile", "flydsl")   # what gemm_op_a8w8._gemm_a8w8_bpreshuffle_into dispatches; an "asm" row is ignored


def load(p):
    with open(p) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def key(r):
    out = []
    for k in KEY:
        v = str(r.get(k, "")).strip()
        try:   # 256 and 256.0 are one key, as in aiter's own duplicate check
            v = str(int(float(v))) if k in ("cu_num", "M", "N", "K") else v
        except ValueError:
            pass
        out.append(v)
    return tuple(out)


def us(r):
    try:
        v = float(r["us"])
        return v if math.isfinite(v) and v > 0 else math.inf
    except (KeyError, ValueError):
        return math.inf


def write(p, fields, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--ms", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    want = sorted({int(x) for x in a.ms.split(",")})
    base = f"{A}/a8w8_bpreshuffle_tuned_gemm.csv"
    image = [base] + sorted(p for p in glob.glob(f"{A}/model_configs/*a8w8_bpreshuffle_tuned_gemm*.csv")
                            if "untuned" not in os.path.basename(p) and os.path.basename(p) != NAME)
    fields, _ = load(base)
    taken = {}
    for p in image:
        for r in load(p)[1]:
            taken.setdefault(key(r), os.path.basename(p))
    _, new = load(a.new)
    best = {}
    for r in new:
        lib = str(r.get("libtype", "")).strip()
        if lib not in LIBS or us(r) == math.inf or str(r.get("kernelName", "None")).strip() in ("", "None"):
            print(f"drop  invalid row M={r.get('M')} N={r.get('N')} K={r.get('K')} libtype={lib} us={r.get('us')}")
            continue
        k = key(r)
        if k in taken:
            print(f"drop  M={r['M']} N={r['N']} K={r['K']}: key already in image table {taken[k]} (a duplicate makes aiter raise)")
            continue
        if k not in best or us(r) < us(best[k]):
            best[k] = r
    by_nk = {}
    for k, r in best.items():
        by_nk.setdefault((int(k[3]), int(k[4])), {})[int(k[2])] = r
    keep, ok = [], True
    for (n, k_), rows in sorted(by_nk.items()):
        missing = [m for m in want if m not in rows]
        libs = {}
        for r in rows.values():
            libs[r["libtype"]] = libs.get(r["libtype"], 0) + 1
        status = "complete" if not missing else f"MISSING M {missing}"
        take = not missing or a.allow_partial
        print(f"N={n} K={k_}: {len(rows)} tuned M ({status}); libtypes {libs}; -> {'kept' if take else 'LEFT OUT (stays on hipBLASLt)'}")
        for m in sorted(rows):
            r = rows[m]
            print(f"    M={m:>5} {r['libtype']:>6} splitK={r.get('splitK')} {us(r):9.2f} us err={r.get('errRatio')} {r['kernelName']}")
        if take:
            keep += [rows[m] for m in sorted(rows)]
        ok = ok and take
    os.makedirs(a.out, exist_ok=True)
    write(f"{a.out}/{NAME}", fields, keep)
    allrows = []
    for p in image:
        allrows += load(p)[1]
    write(f"{a.out}/merged-preview.csv", fields, allrows + keep)
    with open(f"{a.out}/mounts.txt", "w") as f:
        if keep:
            f.write(f"{NAME}:{A}/model_configs/{NAME}\n")
    print(f"ours: {len(keep)} rows in {a.out}/{NAME}; image tables {len(image)} ({len(allrows)} rows); "
          f"merged preview {len(allrows) + len(keep)} rows")
    sys.exit(0 if keep and ok else (2 if keep else 1))


if __name__ == "__main__":
    main()
