#!/usr/bin/env python3
"""Sample rank-labelled SGLang metrics without saving multi-GiB raw scrapes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TARGET_METRICS = {
    "sglang:cache_hit_rate",
    "sglang:cuda_graph_passes",
    "sglang:cuda_graph_passes_total",
    "sglang:full_token_usage",
    "sglang:gen_throughput",
    "sglang:generation_tokens_total",
    "sglang:graph_memory_usage_gb",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_host_used_tokens",
    "sglang:is_cuda_graph",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
    "sglang:mamba_available_tokens",
    "sglang:mamba_evictable_tokens",
    "sglang:mamba_usage",
    "sglang:mamba_used_tokens",
    "sglang:max_total_num_tokens",
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_prefill_prealloc_queue_reqs",
    "sglang:num_prefill_retries",
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
    "sglang:num_used_tokens",
    "sglang:pending_prealloc_token_usage",
    "sglang:prefill_effective_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:realtime_tokens_total",
    "sglang:scheduler_idle_seconds_total",
    "sglang:startup_cuda_graph_time_seconds",
    "sglang:swa_available_tokens",
    "sglang:swa_evictable_tokens",
    "sglang:swa_token_usage",
    "sglang:swa_used_tokens",
    "sglang:token_usage",
}
TARGET_PREFIXES = (
    "dynamo_",
    "infera_",
    "sglang:hicache_",
    "sglang:kv_cache_",
    "sglang:prefix_cache_",
)
HISTOGRAM_SUM_COUNT_PREFIXES = (
    "sglang:kv_transfer_alloc_ms_",
    "sglang:kv_transfer_bootstrap_ms_",
    "sglang:queue_time_seconds_",
)
SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>[^\s]+)(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"')
running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_endpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("endpoint must be NAME=URL")
    name, url = value.split("=", 1)
    name = name.strip()
    url = url.strip().rstrip("/")
    if not name or not url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("endpoint must be NAME=http[s]://HOST:PORT")
    if not url.endswith("/metrics"):
        url += "/metrics"
    return name, url


def decode_label(raw: str) -> str:
    # Prometheus quoted label escapes are compatible with this JSON subset.
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        match.group("name"): decode_label(match.group("value"))
        for match in LABEL_RE.finditer(raw)
    }


def parse_value(raw: str) -> float | str:
    try:
        value = float(raw)
    except ValueError:
        return raw
    return value if math.isfinite(value) else raw


def selected(name: str) -> bool:
    histogram_summary = (
        name.startswith(HISTOGRAM_SUM_COUNT_PREFIXES)
        and name.endswith(("_sum", "_count"))
    )
    return (
        name in TARGET_METRICS
        or name.startswith(TARGET_PREFIXES)
        or histogram_summary
    )


def parse_metrics(body: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    series: list[dict[str, Any]] = []
    metric_types: dict[str, str] = {}
    for line in body.splitlines():
        if line.startswith("# TYPE "):
            parts = line.split()
            if len(parts) >= 4 and selected(parts[2]):
                metric_types[parts[2]] = parts[3]
            continue
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if match is None or not selected(match.group("name")):
            continue
        series.append(
            {
                "metric": match.group("name"),
                "labels": parse_labels(match.group("labels")),
                "value": parse_value(match.group("value")),
            }
        )
    series.sort(
        key=lambda row: (
            str(row["metric"]),
            json.dumps(row["labels"], sort_keys=True),
        )
    )
    return series, metric_types


def scrape(url: str, timeout: float) -> tuple[list[dict[str, Any]], dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/plain; version=0.0.4"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    return parse_metrics(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        action="append",
        type=parse_endpoint,
        required=True,
        help="repeatable NAME=http://HOST:PORT[/metrics]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1.5)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds; zero samples until interrupted",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0 or args.duration < 0:
        parser.error("interval/timeout must be positive and duration non-negative")
    endpoints = dict(args.endpoint)
    if len(endpoints) != len(args.endpoint):
        parser.error("endpoint names must be unique")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    header = {
        "record_type": "metadata",
        "schema_version": 1,
        "started_at": utc_now(),
        "interval_s": args.interval,
        "timeout_s": args.timeout,
        "endpoints": endpoints,
        "target_metrics": sorted(TARGET_METRICS),
        "target_prefixes": list(TARGET_PREFIXES),
        "histogram_sum_count_prefixes": list(HISTOGRAM_SUM_COUNT_PREFIXES),
    }

    with args.output.open("x", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        sequence = 0
        while running and (deadline is None or time.monotonic() < deadline):
            cycle_started = time.monotonic()
            captured_at = utc_now()
            for endpoint, url in endpoints.items():
                scrape_started = time.monotonic()
                record: dict[str, Any] = {
                    "record_type": "sample",
                    "schema_version": 1,
                    "sequence": sequence,
                    "captured_at": captured_at,
                    "elapsed_s": cycle_started - started,
                    "endpoint": endpoint,
                    "url": url,
                }
                try:
                    series, metric_types = scrape(url, args.timeout)
                    record["scrape_ms"] = (
                        time.monotonic() - scrape_started
                    ) * 1000.0
                    record["series_count"] = len(series)
                    record["metric_types"] = metric_types
                    record["series"] = series
                    if not series:
                        record["warning"] = "no selected metrics present"
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                    ValueError,
                ) as exc:
                    record["scrape_ms"] = (
                        time.monotonic() - scrape_started
                    ) * 1000.0
                    record["error"] = f"{type(exc).__name__}: {exc}"
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            sequence += 1
            if args.once:
                break
            delay = args.interval - (time.monotonic() - cycle_started)
            if delay > 0 and running:
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
