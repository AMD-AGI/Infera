#!/usr/bin/env python3
"""Collapse the fixlen sweep's per-run jsonl into one CSV + a markdown table.

bench_serving writes ONE json object per run into --output-file (append mode), so a
re-run of the same arm leaves two objects in the file. We take the LAST object per
file, which is the most recent run of that arm.

Usage:  python3 summarize_fixlen.py <results/fixlen dir> [out.csv]
"""
import json
import sys
from pathlib import Path

FIELDS = [
    ("concurrency", "max_concurrency"),
    ("completed", "completed"),
    ("duration_s", "duration"),
    ("req_throughput", "request_throughput"),
    ("out_tok_per_s", "output_throughput"),
    ("total_tok_per_s", "total_token_throughput"),
    ("ttft_p50_ms", "median_ttft_ms"),
    ("ttft_p90_ms", "p90_ttft_ms"),
    ("ttft_p99_ms", "p99_ttft_ms"),
    ("tpot_p50_ms", "median_tpot_ms"),
    ("tpot_p90_ms", "p90_tpot_ms"),
    ("tpot_p99_ms", "p99_tpot_ms"),
    ("itl_p50_ms", "median_itl_ms"),
    ("e2e_p50_ms", "median_e2e_latency_ms"),
    ("e2e_p90_ms", "p90_e2e_latency_ms"),
    ("e2e_p99_ms", "p99_e2e_latency_ms"),
    ("cache_hit_rate", "cache_hit_rate"),
]


def load_last(path: Path) -> dict | None:
    """Last json object in the file — bench_serving appends, so earlier ones are stale."""
    last = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            pass
    return last


def main() -> int:
    d = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else d / "summary.csv"
    rows = []
    for f in sorted(d.glob("*.jsonl")):
        r = load_last(f)
        if r is None:
            print(f"  WARN: no parsable json in {f.name}", file=sys.stderr)
            continue
        # tag looks like <prefix>_<arm>_isl<N>_osl<N>_c<N>
        parts = f.stem.split("_")
        arm = parts[1] if len(parts) > 1 else "?"
        rows.append({
            "tag": f.stem, "arm": arm,
            "isl": r.get("random_input_len") or r.get("input_len"),
            "osl": r.get("random_output_len") or r.get("output_len"),
            **{k: r.get(src) for k, src in FIELDS},
        })
    if not rows:
        print("no results", file=sys.stderr)
        return 1
    # sort by arm order then concurrency, not lexically (c1 < c16 < c24 < c8 lexically)
    order = {"p50": 0, "p90": 1, "p99": 2}
    rows.sort(key=lambda x: (order.get(x["arm"], 9), x["concurrency"] or 0))

    cols = ["tag", "arm", "isl", "osl"] + [k for k, _ in FIELDS]
    with out.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else str(r[c]) for c in cols) + "\n")
    print(f"wrote {out} ({len(rows)} rows)\n")

    show = ["arm", "isl", "osl", "concurrency", "completed", "req_throughput",
            "out_tok_per_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_ms"]
    print("| " + " | ".join(show) + " |")
    print("|" + "---|" * len(show))
    for r in rows:
        cells = []
        for c in show:
            v = r.get(c)
            cells.append("" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
