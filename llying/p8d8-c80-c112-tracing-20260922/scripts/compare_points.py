#!/usr/bin/env python3
"""Compare C80/C112 request stages, cache tiers, and headline metrics."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def request_stats(path: Path) -> dict[str, Any]:
    durations: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    by_rank: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    cache = collections.Counter()
    records = 0
    for row in jsonl(path):
        if row.get("record_type") != "request_time_stats":
            continue
        records += 1
        role = str(row.get("source_role"))
        header = row.get("header") or {}
        rank = str(header.get("routed_dp_rank", "unlabelled"))
        for name, raw in (row.get("durations") or {}).items():
            value = number(raw)
            if value is not None and isinstance(raw, dict) and raw.get("unit") == "ms":
                durations[(role, name)].append(value)
                by_rank[(role, name, rank)].append(value)
        if role == "prefill":
            input_len = number(header.get("input_len")) or 0.0
            device = number(header.get("cached_device")) or 0.0
            host = number(header.get("cached_host")) or 0.0
            storage = number(header.get("cached_storage")) or 0.0
            cache["input_tokens"] += input_len
            cache["device_hit_tokens"] += device
            cache["host_hit_tokens"] += host
            cache["storage_hit_tokens"] += storage
            cache["miss_tokens"] += max(0.0, input_len - device - host - storage)
    summary: dict[str, Any] = {}
    for (role, name), values in sorted(durations.items()):
        summary.setdefault(role, {})[name] = stats(values)
    rank_summary: dict[str, Any] = {}
    for (role, name, rank), values in sorted(by_rank.items()):
        rank_summary.setdefault(role, {}).setdefault(name, {})[rank] = stats(values)
    denominator = cache["input_tokens"]
    cache_result = dict(cache)
    if denominator:
        for name in (
            "device_hit_tokens",
            "host_hit_tokens",
            "storage_hit_tokens",
            "miss_tokens",
        ):
            cache_result[name.replace("_tokens", "_rate")] = cache[name] / denominator
    return {
        "records": records,
        "durations": summary,
        "durations_by_rank": rank_summary,
        "cache": cache_result,
    }


def headline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload["request_metrics"]
    server = payload["server_metrics"]
    return {
        "successful_requests": payload["request_accounting"]["records_profiled"],
        "error_records": payload["request_accounting"]["records_error_dropped"],
        "qps": request["qps"],
        "latency": request["latency"],
        "throughput": request["throughput"],
        "theoretical_cache_hit_rate": request["cache"][
            "theoretical_cache_hit_rate"
        ],
        "server_cache": server["cache"],
        "server_kv_cache": server["kv_cache"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.run / "analysis/c80-c112-comparison.json"
    if output.exists() and not args.force:
        parser.error(f"output exists: {output}; pass --force")
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run": str(args.run.resolve()),
        "points": {},
    }
    for concurrency in (80, 112):
        point = args.run / f"c{concurrency:03d}"
        result["points"][str(concurrency)] = {
            "headline": headline(point / f"bench/agentx_conc{concurrency}.json"),
            "request_time_stats": request_stats(point / "request-time-stats.jsonl"),
            "trace_stage_summary": json.loads(
                (point / "trace-analysis/stage-summary.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
