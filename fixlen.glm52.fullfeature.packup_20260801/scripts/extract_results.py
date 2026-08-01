#!/usr/bin/env python3
"""Flatten the eight bench_serving JSON artifacts into one CSV.

Reads results/raw/fixlen_<pair>_c<C>.jsonl[.gz] and writes
results/fixlen_summary.csv -- the table every claim in README.md is drawn from.

The .jsonl files are single JSON objects (not line-delimited, despite the name)
and carry BOTH the aggregates and the raw per-request arrays (`ttfts`, `itls`,
`input_lens`, `output_lens`), which is what makes a full percentile ladder
recomputable later. Handles gzip transparently so the packed kit works as-is.

    python3 scripts/extract_results.py [results_dir]
"""
import csv
import gzip
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
RAW = os.path.join(ROOT, "raw")
PAIRS = (("p50", 74000, 320), ("p90", 155000, 3300))
CONCS = (1, 32, 64, 128)


def load(path_noext):
    for p in (path_noext + ".gz", path_noext):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return json.load(f)
    return None


rows = []
for pair, isl, osl in PAIRS:
    for c in CONCS:
        d = load(os.path.join(RAW, f"fixlen_{pair}_c{c}.jsonl"))
        if d is None:
            print(f"  MISSING: fixlen_{pair}_c{c}.jsonl", file=sys.stderr)
            continue
        cr = d.get("cache_report") or {}
        rows.append(
            dict(
                pair=pair, isl=isl, osl=osl, conc=c,
                completed=d["completed"],
                duration_s=round(d["duration"], 2),
                req_per_s=round(d["request_throughput"], 4),
                input_tok_s=round(d["input_throughput"], 1),
                output_tok_s=round(d["output_throughput"], 1),
                total_tok_s=round(d["total_throughput"], 1),
                ttft_p50_ms=round(d["median_ttft_ms"], 1),
                ttft_p90_ms=round(d["p90_ttft_ms"], 1),
                ttft_p99_ms=round(d["p99_ttft_ms"], 1),
                tpot_p50_ms=round(d["median_tpot_ms"], 2),
                tpot_p99_ms=round(d["p99_tpot_ms"], 2),
                e2e_p50_ms=round(d["median_e2e_latency_ms"], 1),
                e2e_p99_ms=round(d["p99_e2e_latency_ms"], 1),
                cache_hit_pct=cr.get("cache_hit_rate_pct"),
                # device vs host vs storage separates the GPU radix cache from
                # kvd's L2/L3. host~0 across this sweep is the evidence that the
                # nonzero hit rates are radix residue, NOT kvd reads.
                cached_device_tok=cr.get("device_cached_tokens"),
                cached_host_tok=cr.get("host_cached_tokens"),
                cached_storage_tok=cr.get("storage_cached_tokens"),
            )
        )

if not rows:
    sys.exit("no result files found")

out = os.path.join(ROOT, "fixlen_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rounds -> {out}")

hdr = f"{'round':<10}{'in tok/s':>10}{'ttft_p50':>10}{'ttft_p99':>11}{'tpot':>7}{'hit%':>7}"
print(hdr)
for r in rows:
    hit = "n/a" if r["cache_hit_pct"] is None else f"{r['cache_hit_pct']:.2f}"
    print(
        f"{r['pair'] + ' c' + str(r['conc']):<10}{r['input_tok_s']:>10.0f}"
        f"{r['ttft_p50_ms']:>10.0f}{r['ttft_p99_ms']:>11.0f}"
        f"{r['tpot_p50_ms']:>7.2f}{hit:>7}"
    )
