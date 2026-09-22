#!/usr/bin/env python3
"""Save a direct prefill metrics snapshot and sum selected metrics across ranks."""
import argparse
import json
import re
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://10.235.192.136:29001/metrics")
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
raw = urllib.request.urlopen(args.url, timeout=30).read().decode()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(raw)
selected = {
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_backup_bytes_total",
    "sglang:hicache_backup_tokens_total",
    "sglang:hicache_backup_duration_seconds_sum",
    "sglang:load_back_bytes_total",
    "sglang:load_back_tokens_total",
    "sglang:load_back_duration_seconds_sum",
    "sglang:hicache_backup_dropped_tokens_total",
    "sglang:hicache_dropped_tokens_total",
}
totals = {}
for line in raw.splitlines():
    match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)", line)
    if match and match[1] in selected:
        totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
summary = {"url": args.url, "totals_across_ranks": totals}
args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
