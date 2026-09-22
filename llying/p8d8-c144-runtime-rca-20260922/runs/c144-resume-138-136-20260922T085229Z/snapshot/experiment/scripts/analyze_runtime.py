#!/usr/bin/env python3
"""Summarize per-rank scheduling, graph, cache, GPU, and rail samples."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


GAUGES = {
    "sglang:cache_hit_rate",
    "sglang:full_token_usage",
    "sglang:gen_throughput",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_host_used_tokens",
    "sglang:is_cuda_graph",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
    "sglang:mamba_usage",
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
    "sglang:pending_prealloc_token_usage",
    "sglang:swa_token_usage",
    "sglang:token_usage",
}
WORK_GAUGES = {
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_running_reqs",
}
TRACKED_HCA = {
    "req_rx_cqe_err",
    "req_rx_pkt_seq_err",
    "req_tx_retry_excd_err",
    "rx_rdma_ucast_bytes",
    "rx_rdma_ucast_pkts",
    "tx_rdma_ack_timeout",
    "tx_rdma_retx_bytes",
    "tx_rdma_retx_pkts",
    "tx_rdma_ucast_bytes",
    "tx_rdma_ucast_pkts",
}


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def label_key(labels: dict[str, Any]) -> str:
    return json.dumps(labels, sort_keys=True, separators=(",", ":"))


def rank(item: dict[str, Any]) -> str | None:
    labels = item.get("labels")
    if not isinstance(labels, dict):
        return None
    value = labels.get("dp_rank")
    return str(value) if value is not None else None


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def inspect_engine(
    path: Path,
) -> tuple[
    set[int],
    dt.datetime | None,
    dt.datetime | None,
    dict[tuple[str, str, str], list[Any]],
    list[str],
]:
    active: set[int] = set()
    active_start: dt.datetime | None = None
    active_end: dt.datetime | None = None
    counter_edges: dict[tuple[str, str, str], list[Any]] = {}
    errors: list[str] = []

    for row in jsonl(path):
        if row.get("record_type") != "sample":
            continue
        if row.get("error"):
            errors.append(
                f"{row.get('captured_at')} {row.get('endpoint')}: {row['error']}"
            )
            continue
        endpoint = str(row.get("endpoint"))
        sequence = int(row.get("sequence", -1))
        work = 0.0
        metric_types = row.get("metric_types") or {}
        for item in row.get("series") or []:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric"))
            value = number(item.get("value"))
            if value is None:
                continue
            if endpoint in {"prefill", "decode"} and metric in WORK_GAUGES:
                work += value
            if metric_types.get(metric) != "counter":
                continue
            labels = item.get("labels") or {}
            key = (endpoint, metric, label_key(labels))
            edge = counter_edges.setdefault(
                key,
                [value, value, labels],
            )
            edge[1] = value
        if endpoint in {"prefill", "decode"} and work > 0:
            active.add(sequence)
            captured = parse_time(str(row["captured_at"]))
            active_start = (
                captured if active_start is None else min(active_start, captured)
            )
            active_end = captured if active_end is None else max(active_end, captured)
    return active, active_start, active_end, counter_edges, errors


def summarize_engine(
    path: Path,
    active: set[int],
    counter_edges: dict[tuple[str, str, str], list[Any]],
) -> dict[str, Any]:
    per_rank: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    imbalance: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    active_samples: collections.Counter[str] = collections.Counter()

    for row in jsonl(path):
        if (
            row.get("record_type") != "sample"
            or row.get("error")
            or int(row.get("sequence", -1)) not in active
        ):
            continue
        endpoint = str(row.get("endpoint"))
        if endpoint not in {"prefill", "decode"}:
            continue
        active_samples[endpoint] += 1
        grouped: dict[str, dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )
        for item in row.get("series") or []:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric"))
            if metric not in GAUGES:
                continue
            dp_rank = rank(item)
            value = number(item.get("value"))
            if dp_rank is None or value is None:
                continue
            grouped[metric][dp_rank] += value
        for metric, values_by_rank in grouped.items():
            values = list(values_by_rank.values())
            for dp_rank, value in values_by_rank.items():
                per_rank[(endpoint, metric, dp_rank)].append(value)
            total = sum(values)
            mean = statistics.fmean(values)
            target = imbalance[(endpoint, metric)]
            target["total"].append(total)
            target["spread"].append(max(values) - min(values))
            if mean > 0:
                target["max_over_mean"].append(max(values) / mean)
                target["cv"].append(
                    statistics.pstdev(values) / mean if len(values) > 1 else 0.0
                )
            if metric == "sglang:num_running_reqs":
                target["ranks_at_32"].append(
                    float(sum(value >= 32 for value in values))
                )
                target["any_rank_at_32"].append(float(any(value >= 32 for value in values)))
                target["total_at_256"].append(float(total >= 256))

    rank_summary: dict[str, Any] = {}
    for (endpoint, metric, dp_rank), values in sorted(per_rank.items()):
        rank_summary.setdefault(endpoint, {}).setdefault(metric, {})[dp_rank] = stats(
            values
        )
    imbalance_summary: dict[str, Any] = {}
    for (endpoint, metric), fields in sorted(imbalance.items()):
        imbalance_summary.setdefault(endpoint, {})[metric] = {
            name: stats(values)
            for name, values in sorted(fields.items())
        }

    counter_rows = []
    resets = []
    for (endpoint, metric, _), (first, last, labels) in sorted(counter_edges.items()):
        delta = last - first
        if delta < 0:
            resets.append({"endpoint": endpoint, "metric": metric, "labels": labels})
            continue
        counter_rows.append(
            {
                "endpoint": endpoint,
                "metric": metric,
                "labels": labels,
                "first": first,
                "last": last,
                "delta": delta,
            }
        )

    graph: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    effective_tokens: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    hicache_drops: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for row in counter_rows:
        labels = row["labels"]
        mode = str(labels.get("mode", "unlabelled"))
        if row["metric"] == "sglang:cuda_graph_passes_total":
            graph[row["endpoint"]][mode] += row["delta"]
        elif row["metric"] == "sglang:prefill_effective_tokens_total":
            effective_tokens[row["endpoint"]][mode] += row["delta"]
        elif row["metric"] == "sglang:hicache_dropped_tokens_total":
            reason = str(labels.get("reason", "unlabelled"))
            hicache_drops[row["endpoint"]][reason] += row["delta"]

    graph_coverage = {}
    for endpoint, modes in graph.items():
        graph_passes = sum(
            value for mode, value in modes.items() if mode.endswith("_cuda_graph")
        )
        eager_passes = sum(
            value for mode, value in modes.items() if mode.endswith("_none")
        )
        denominator = graph_passes + eager_passes
        graph_coverage[endpoint] = {
            "modes": dict(sorted(modes.items())),
            "graph_passes": graph_passes,
            "eager_passes": eager_passes,
            "coverage": graph_passes / denominator if denominator else None,
        }

    return {
        "active_samples": dict(active_samples),
        "per_rank": rank_summary,
        "imbalance": imbalance_summary,
        "counter_deltas": counter_rows,
        "counter_resets": resets,
        "cuda_graph": graph_coverage,
        "prefill_effective_tokens": {
            endpoint: dict(sorted(modes.items()))
            for endpoint, modes in effective_tokens.items()
        },
        "hicache_dropped_tokens": {
            endpoint: dict(sorted(reasons.items()))
            for endpoint, reasons in hicache_drops.items()
        },
    }


def summarize_nodes(
    path: Path,
    active_start: dt.datetime | None,
    active_end: dt.datetime | None,
) -> dict[str, Any]:
    edges: dict[tuple[str, str, str], list[Any]] = {}
    gpu_values: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    errors: list[str] = []

    for row in jsonl(path):
        if row.get("record_type") != "sample":
            continue
        captured = parse_time(str(row["captured_at"]))
        if active_start is not None and captured < active_start:
            continue
        if active_end is not None and captured > active_end:
            continue
        role = str(row.get("role"))
        if row.get("error"):
            errors.append(f"{row.get('captured_at')} {role}: {row['error']}")
            continue
        sample = row.get("sample") or {}
        gpu = (sample.get("gpu") or {}).get("json") or {}
        for card, values in gpu.items():
            if not isinstance(values, dict):
                continue
            for field in (
                "GPU use (%)",
                "GPU Memory Allocated (VRAM%)",
                "Current Socket Graphics Package Power (W)",
            ):
                value = number(values.get(field))
                if value is not None:
                    gpu_values[(role, str(card), field)].append(value)
        remote_ns = number(sample.get("remote_time_ns"))
        for hca, values in (sample.get("hcas") or {}).items():
            if not isinstance(values, dict):
                continue
            for name, raw in (values.get("hw_counters") or {}).items():
                if name not in TRACKED_HCA:
                    continue
                value = number(raw)
                if value is None or remote_ns is None:
                    continue
                key = (role, str(hca), str(name))
                edge = edges.setdefault(key, [remote_ns, value, remote_ns, value])
                edge[2:] = [remote_ns, value]

    hcas: dict[str, Any] = {}
    resets = []
    for (role, hca, name), (first_ns, first, last_ns, last) in sorted(edges.items()):
        delta = last - first
        elapsed = (last_ns - first_ns) / 1e9
        if delta < 0:
            resets.append({"role": role, "hca": hca, "counter": name})
            continue
        record = hcas.setdefault(role, {}).setdefault(hca, {})
        record[name] = {
            "delta": delta,
            "rate_per_second": delta / elapsed if elapsed > 0 else None,
        }

    gpu_summary: dict[str, Any] = {}
    for (role, card, field), values in sorted(gpu_values.items()):
        gpu_summary.setdefault(role, {}).setdefault(card, {})[field] = stats(values)
    return {
        "gpu": gpu_summary,
        "hcas": hcas,
        "counter_resets": resets,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    engine_v2 = args.run / "sampling/live/engine-metrics-v2.jsonl"
    engine = (
        engine_v2
        if engine_v2.exists()
        else args.run / "sampling/live/engine-metrics.jsonl"
    )
    nodes = args.run / "sampling/live/node-runtime.jsonl"
    output = args.run / "analysis/runtime-summary.json"
    if output.exists() and not args.force:
        parser.error(f"output already exists: {output}; pass --force to replace")
    if not engine.exists() or not nodes.exists():
        parser.error("run does not contain engine and node sample JSONL files")

    active, active_start, active_end, counter_edges, engine_errors = inspect_engine(
        engine
    )
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run": str(args.run.resolve()),
        "engine_input": str(engine),
        "node_input": str(nodes),
        "active_sequence_count": len(active),
        "active_start": active_start.isoformat() if active_start else None,
        "active_end": active_end.isoformat() if active_end else None,
        "engine_errors": engine_errors,
        "engine": summarize_engine(engine, active, counter_edges),
        "nodes": summarize_nodes(nodes, active_start, active_end),
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
