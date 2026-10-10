#!/usr/bin/env python3
"""Reduce AIPerf's export to the handful of numbers anything downstream reads.

`profile_export_aiperf.csv` is 63 rows of `Metric,avg,min,max,sum,p1,...,std`,
most of which describe AIPerf's own HTTP client. Parsing it in the validator and
again in the analysis stage would be two places to get the row names wrong, so it
is parsed once, here, and published beside the CSV.

The CSV is kept. This is a convenience, not a replacement: a number that only
exists in a summary is a number nobody can check.

    summarise.py <profile_export_aiperf.csv> <summary.json>
"""

import csv
import json
import re
import sys
from pathlib import Path

#: A trailing unit in parentheses. **Stripped before matching, and that is a fix
#: rather than a convenience.** AIPerf's console tables print
#: `Request Count (requests)` while the CSV row is bare `Request Count`; most
#: other rows carry the unit in both. Keying on the printed form silently found
#: ten of eleven metrics and dropped the request count — which is the one metric
#: `check_aiperf_report` exists to check.
_UNIT = re.compile(r"\s*\([^)]*\)\s*$")

#: Metric name with its unit stripped -> key in the summary. Only rows that
#: describe the SERVICE, not AIPerf's own HTTP client.
#:
#: Matched exactly on the stripped name, not by substring: `Output Token
#: Throughput`, `Output Token Throughput Per User` and `E2E Output Token
#: Throughput Per User` are three different measurements and a substring match
#: would conflate them. All eleven below were checked to be distinct after
#: stripping.
WANTED = {
    "Request Count": "request_count",
    "Request Throughput": "request_throughput_rps",
    "Output Token Throughput": "output_token_throughput_tps",
    "Input Token Throughput": "input_token_throughput_tps",
    "Time to First Token": "ttft_ms",
    "Inter Token Latency": "inter_token_latency_ms",
    "Request Latency": "request_latency_ms",
    "Output Sequence Length": "output_sequence_length",
    "Input Sequence Length": "input_sequence_length",
    "Output Token Throughput Per User": "output_tps_per_user",
    "Effective Concurrency": "effective_concurrency",
}

#: Percentile columns worth keeping. AIPerf leaves them empty for aggregate rows
#: (a throughput has no p50), and an empty cell becomes an absent key rather than
#: a zero -- the two are not the same and DeepEval's unreached-node bug is what
#: happens when they are conflated.
COLUMNS = ("avg", "min", "max", "p50", "p90", "p99", "std")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: summarise.py <profile_export_aiperf.csv> <summary.json>", file=sys.stderr)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    if not src.is_file():
        print(f"summarise: {src} is missing", file=sys.stderr)
        return 1

    rows = {}
    with src.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = _UNIT.sub("", (row.get("Metric") or "").strip())
            if name not in WANTED:
                continue
            values = {}
            for column in COLUMNS:
                raw = (row.get(column) or "").strip()
                if not raw or raw.upper() == "N/A":
                    continue
                try:
                    values[column] = float(raw.replace(",", ""))
                except ValueError:
                    continue
            if values:
                rows[WANTED[name]] = values

    missing = sorted(set(WANTED.values()) - set(rows))
    summary = {"metrics": rows, "missing": missing, "source": src.name}
    dst.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    got = rows.get("request_count", {}).get("avg")
    print(f"summarise: {len(rows)} metric(s), request_count={got}, missing={missing or 'none'}")
    # A missing row is reported, not fatal: `check_aiperf_report` decides whether
    # the set that arrived is enough, and it has the args to say so.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
