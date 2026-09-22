#!/usr/bin/env python3
"""Build auditable statistics for the externally interrupted first C144 run."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

from analyze_runtime import jsonl, number, parse_time, percentile, stats


GAUGES = {
    "sglang:cache_hit_rate",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_host_used_tokens",
    "sglang:is_cuda_graph",
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
    "sglang:token_usage",
}
COUNTERS = {
    "sglang:cuda_graph_passes_total",
    "sglang:generation_tokens_total",
    "sglang:hicache_dropped_tokens_total",
    "sglang:prefill_effective_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:realtime_tokens_total",
    "sglang:scheduler_idle_seconds_total",
}
HCA_COUNTERS = {
    "req_tx_retry_excd_err",
    "rx_rdma_ucast_bytes",
    "tx_rdma_ack_timeout",
    "tx_rdma_ucast_bytes",
}


def metric(record: dict[str, Any], name: str) -> float | None:
    value = (record.get("metrics") or {}).get(name)
    if not isinstance(value, dict):
        return None
    return number(value.get("value"))


def scalar_stats(values: list[float]) -> dict[str, Any]:
    result = stats(values)
    result["p75"] = percentile(values, 0.75)
    result["p99"] = percentile(values, 0.99)
    return result


def weighted_step_stats(
    intervals: list[tuple[float, float]],
    window_start: float,
    window_end: float,
) -> dict[str, float | int | None]:
    events = []
    for start, end in intervals:
        start = max(start, window_start)
        end = min(end, window_end)
        if end > start:
            events.append((start, 1))
            events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    value = 0
    previous = window_start
    segments: list[tuple[int, float]] = []
    for timestamp, delta in events:
        if timestamp > previous:
            segments.append((value, timestamp - previous))
            previous = timestamp
        value += delta
    if window_end > previous:
        segments.append((value, window_end - previous))
    duration = sum(weight for _, weight in segments)
    if duration <= 0:
        return {
            "duration_s": 0.0,
            "avg": None,
            "p50": None,
            "p90": None,
            "max": None,
        }

    def weighted_percentile(q: float) -> float:
        target = duration * q
        cumulative = 0.0
        for sample, weight in sorted(segments):
            cumulative += weight
            if cumulative >= target:
                return float(sample)
        return float(segments[-1][0])

    return {
        "duration_s": duration / 1e9,
        "avg": sum(sample * weight for sample, weight in segments) / duration,
        "p50": weighted_percentile(0.50),
        "p90": weighted_percentile(0.90),
        "max": float(max(sample for sample, _ in segments)),
    }


def analyze_records(path: Path) -> dict[str, Any]:
    phase_counts: collections.Counter[str] = collections.Counter()
    cancelled: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    profiling: list[dict[str, Any]] = []
    for record in jsonl(path):
        metadata = record.get("metadata") or {}
        phase = str(metadata.get("benchmark_phase", "unknown"))
        phase_counts[phase] += 1
        if metadata.get("was_cancelled"):
            cancelled[phase] += 1
        if record.get("error"):
            errors[phase] += 1
        if phase == "profiling":
            profiling.append(record)

    successful = [
        record
        for record in profiling
        if not (record.get("metadata") or {}).get("was_cancelled")
        and not record.get("error")
        and metric(record, "request_latency") is not None
    ]
    starts = [
        float(record["metadata"]["request_start_ns"])
        for record in successful
        if record.get("metadata", {}).get("request_start_ns") is not None
    ]
    ends = [
        float(record["metadata"]["request_end_ns"])
        for record in successful
        if record.get("metadata", {}).get("request_end_ns") is not None
    ]
    window_start = min(starts)
    window_end = max(ends)
    duration_s = (window_end - window_start) / 1e9

    metric_names = (
        "time_to_first_token",
        "request_latency",
        "full_response_inter_token_latency",
        "input_sequence_length",
        "output_sequence_length",
        "prefill_throughput_per_user",
        "full_response_output_token_throughput_per_user",
    )
    metric_summary = {}
    for name in metric_names:
        values = [
            value
            for record in successful
            if (value := metric(record, name)) is not None
        ]
        metric_summary[name] = scalar_stats(values)

    input_tokens = sum(
        metric(record, "usage_prompt_tokens") or 0.0
        for record in successful
    )
    output_tokens = sum(
        metric(record, "usage_completion_tokens") or 0.0
        for record in successful
    )
    total_intervals = []
    prefill_intervals = []
    decode_intervals = []
    for record in profiling:
        metadata = record.get("metadata") or {}
        start = number(metadata.get("request_start_ns"))
        end = number(metadata.get("request_end_ns"))
        ttft_ms = metric(record, "time_to_first_token")
        if start is None or end is None or end <= start:
            continue
        total_intervals.append((start, end))
        if ttft_ms is not None:
            generation_start = min(end, start + ttft_ms * 1e6)
            prefill_intervals.append((start, generation_start))
            if end > generation_start:
                decode_intervals.append((generation_start, end))

    mismatch = [
        abs(value)
        for record in successful
        if (value := metric(record, "osl_mismatch_diff_pct")) is not None
    ]
    return {
        "records": {
            "phase_counts": dict(phase_counts),
            "cancelled": dict(cancelled),
            "errors": dict(errors),
            "successful_partial_profiling": len(successful),
        },
        "window": {
            "start_ns": window_start,
            "end_ns": window_end,
            "duration_s": duration_s,
        },
        "request_metrics": metric_summary,
        "throughput": {
            "qps": len(successful) / duration_s,
            "input_tokens_per_second": input_tokens / duration_s,
            "output_tokens_per_second": output_tokens / duration_s,
            "total_tokens_per_second": (input_tokens + output_tokens) / duration_s,
            "total_tokens_per_second_per_gpu": (
                (input_tokens + output_tokens) / duration_s / 16.0
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "effective_concurrency": {
            "total": weighted_step_stats(total_intervals, window_start, window_end),
            "prefill": weighted_step_stats(
                prefill_intervals, window_start, window_end
            ),
            "decode": weighted_step_stats(decode_intervals, window_start, window_end),
        },
        "osl_mismatch": {
            "abs_diff_pct": scalar_stats(mismatch),
            "count_over_50pct": sum(value > 50 for value in mismatch),
        },
    }


def in_window(timestamp: str, start: dt.datetime, end: dt.datetime) -> bool:
    parsed = parse_time(timestamp)
    return start <= parsed <= end


def analyze_engine_window(
    path: Path,
    start: dt.datetime,
    end: dt.datetime,
) -> dict[str, Any]:
    gauge_values: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    rank_values: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    counter_edges: dict[tuple[str, str, str], list[Any]] = {}
    scrape_counts: collections.Counter[str] = collections.Counter()
    scrape_errors: collections.Counter[str] = collections.Counter()

    for row in jsonl(path):
        if row.get("record_type") != "sample":
            continue
        captured_at = str(row.get("captured_at"))
        if not in_window(captured_at, start, end):
            continue
        endpoint = str(row.get("endpoint"))
        scrape_counts[endpoint] += 1
        if row.get("error"):
            scrape_errors[endpoint] += 1
            continue
        grouped: dict[str, dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )
        metric_types = row.get("metric_types") or {}
        for item in row.get("series") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("metric"))
            value = number(item.get("value"))
            labels = item.get("labels") or {}
            if value is None:
                continue
            if name in GAUGES:
                dp_rank = str(labels.get("dp_rank", "unlabelled"))
                grouped[name][dp_rank] += value
            if name in COUNTERS or metric_types.get(name) == "counter":
                key = (
                    endpoint,
                    name,
                    json.dumps(labels, sort_keys=True, separators=(",", ":")),
                )
                edge = counter_edges.setdefault(key, [value, value, labels])
                edge[1] = value
        for name, values_by_rank in grouped.items():
            gauge_values[(endpoint, name)].append(sum(values_by_rank.values()))
            for dp_rank, value in values_by_rank.items():
                rank_values[(endpoint, name, dp_rank)].append(value)

    counters = []
    graph: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    effective_tokens: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    hicache_drops: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for (endpoint, name, _), (first, last, labels) in sorted(counter_edges.items()):
        delta = last - first
        if delta < 0:
            continue
        counters.append(
            {
                "endpoint": endpoint,
                "metric": name,
                "labels": labels,
                "delta": delta,
            }
        )
        mode = str(labels.get("mode", "unlabelled"))
        if name == "sglang:cuda_graph_passes_total":
            graph[endpoint][mode] += delta
        elif name == "sglang:prefill_effective_tokens_total":
            effective_tokens[endpoint][mode] += delta
        elif name == "sglang:hicache_dropped_tokens_total":
            hicache_drops[endpoint][str(labels.get("reason", "unlabelled"))] += delta

    graph_summary = {}
    for endpoint, modes in graph.items():
        captured = sum(
            value for mode, value in modes.items() if mode.endswith("_cuda_graph")
        )
        eager = sum(value for mode, value in modes.items() if mode.endswith("_none"))
        graph_summary[endpoint] = {
            "modes": dict(sorted(modes.items())),
            "captured_passes": captured,
            "eager_passes": eager,
            "coverage": captured / (captured + eager) if captured + eager else None,
        }

    gauge_summary: dict[str, Any] = {}
    for (endpoint, name), values in sorted(gauge_values.items()):
        gauge_summary.setdefault(endpoint, {})[name] = scalar_stats(values)
    rank_summary: dict[str, Any] = {}
    for (endpoint, name, dp_rank), values in sorted(rank_values.items()):
        rank_summary.setdefault(endpoint, {}).setdefault(name, {})[
            dp_rank
        ] = scalar_stats(values)
    return {
        "scrapes": {
            "total": dict(scrape_counts),
            "errors": dict(scrape_errors),
        },
        "gauges_rank_sum": gauge_summary,
        "per_rank": rank_summary,
        "counter_deltas": counters,
        "cuda_graph": graph_summary,
        "prefill_effective_tokens": {
            endpoint: dict(sorted(modes.items()))
            for endpoint, modes in effective_tokens.items()
        },
        "hicache_dropped_tokens": {
            endpoint: dict(sorted(reasons.items()))
            for endpoint, reasons in hicache_drops.items()
        },
    }


def analyze_rails(
    path: Path,
    start: dt.datetime,
    end: dt.datetime,
) -> dict[str, Any]:
    edges: dict[tuple[str, str, str], list[float]] = {}
    errors = []
    for row in jsonl(path):
        if row.get("record_type") != "sample":
            continue
        if not in_window(str(row.get("captured_at")), start, end):
            continue
        if row.get("error"):
            errors.append(row)
            continue
        role = str(row.get("role"))
        sample = row.get("sample") or {}
        timestamp = number(sample.get("remote_time_ns"))
        if timestamp is None:
            continue
        for hca, values in (sample.get("hcas") or {}).items():
            for name, raw in (values.get("hw_counters") or {}).items():
                if name not in HCA_COUNTERS:
                    continue
                value = number(raw)
                if value is None:
                    continue
                key = (role, str(hca), str(name))
                edge = edges.setdefault(key, [timestamp, value, timestamp, value])
                edge[2:] = [timestamp, value]
    result: dict[str, Any] = {}
    for (role, hca, name), (first_ns, first, last_ns, last) in sorted(edges.items()):
        delta = last - first
        elapsed = (last_ns - first_ns) / 1e9
        result.setdefault(role, {}).setdefault(hca, {})[name] = {
            "delta": delta,
            "rate_per_second": delta / elapsed if elapsed > 0 and delta >= 0 else None,
        }
    return {"hcas": result, "errors": len(errors)}


def timeout_summary(
    path: Path,
    start: dt.datetime,
    end: dt.datetime,
) -> dict[str, Any]:
    pattern = re.compile(
        r"^(?P<time>\S+).*rank=(?P<rank>\d+).*"
        r"timed out after 1800\.0s in KVPoll\.WaitingForInput"
    )
    ranks: collections.Counter[str] = collections.Counter()
    timestamps = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if not match:
                continue
            timestamp = parse_time(match.group("time"))
            if start <= timestamp <= end:
                ranks[match.group("rank")] += 1
                timestamps.append(timestamp)
    return {
        "count": sum(ranks.values()),
        "by_rank": dict(sorted(ranks.items())),
        "first": min(timestamps).isoformat() if timestamps else None,
        "last": max(timestamps).isoformat() if timestamps else None,
    }


def parse_completed_at(path: Path) -> dt.datetime:
    raw = path.read_text(encoding="utf-8").strip().replace(",", ".")
    return parse_time(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.run / "analysis/partial-run-data.json"
    if output.exists() and not args.force:
        parser.error(f"output exists: {output}; pass --force")

    warmup_start = parse_time("2026-09-22T07:17:52.384+00:00")
    warmup_end = parse_time("2026-09-22T08:34:03.503+00:00")
    profiling_start = parse_time("2026-09-22T08:34:03.627+00:00")
    run_end = parse_completed_at(args.run / "sampling/sampling-completed-at.txt")
    engine_path = args.run / "sampling/live/engine-metrics-v2.jsonl"
    node_path = args.run / "sampling/live/node-runtime.jsonl"

    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run": str(args.run.resolve()),
        "limitations": [
            "The client was externally terminated before the planned 3600s profiling window.",
            "Request-level profiling statistics are partial and composition-dependent.",
            "The engine v2 sampler continued briefly after client termination; all analyses are clipped to sampling-completed-at.",
        ],
        "phase_windows": {
            "warmup": {
                "start": warmup_start.isoformat(),
                "end": warmup_end.isoformat(),
                "duration_s": (warmup_end - warmup_start).total_seconds(),
            },
            "partial_profiling": {
                "start": profiling_start.isoformat(),
                "end": run_end.isoformat(),
                "duration_s": (run_end - profiling_start).total_seconds(),
            },
        },
        "records": analyze_records(
            args.run / "bench/aiperf_artifacts/profile_export.jsonl"
        ),
        "warmup": {
            "engine": analyze_engine_window(engine_path, warmup_start, warmup_end),
            "rails": analyze_rails(node_path, warmup_start, warmup_end),
            "waiting_for_input_timeouts": timeout_summary(
                args.run / "launch/server-logs/decode-0.log",
                warmup_start,
                warmup_end,
            ),
        },
        "partial_profiling": {
            "engine": analyze_engine_window(
                engine_path, profiling_start, run_end
            ),
            "rails": analyze_rails(node_path, profiling_start, run_end),
            "waiting_for_input_timeouts": timeout_summary(
                args.run / "launch/server-logs/decode-0.log",
                profiling_start,
                run_end,
            ),
        },
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
