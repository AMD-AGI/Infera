#!/usr/bin/env python3
"""Normalize OTLP spans and summarize request-stage latency by role/rank."""

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
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def role_for(span: dict[str, Any], inherited: dict[str, Any]) -> str:
    role = inherited.get("role")
    if role:
        return str(role)
    name = str(span.get("name", "")).lower()
    service = str(span.get("service_name", "")).lower()
    for role_name in ("prefill", "decode"):
        if name.startswith(role_name + " ") or role_name in service:
            return role_name
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    traces: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in jsonl(args.input):
        if row.get("record_type") == "span":
            traces[str(row.get("trace_id", ""))].append(row)

    stage_values: dict[tuple[str, str, str, str], list[float]] = (
        collections.defaultdict(list)
    )
    completeness: collections.Counter[str] = collections.Counter()
    timeline_path = args.output_dir / "request-timelines.jsonl"
    with timeline_path.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "schema_version": 1,
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "point": args.point,
                    "input": str(args.input),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for trace_id, spans in sorted(traces.items()):
            by_id = {
                str(span.get("span_id", "")): span
                for span in spans
                if span.get("span_id")
            }
            memo: dict[str, dict[str, Any]] = {}

            def inherited(span: dict[str, Any]) -> dict[str, Any]:
                span_id = str(span.get("span_id", ""))
                if span_id in memo:
                    return memo[span_id]
                parent = by_id.get(str(span.get("parent_span_id", "")))
                values = inherited(parent).copy() if parent is not None else {}
                attrs = span.get("attributes") or {}
                for key in ("rid", "bootstrap_room", "dp_rank", "tp_rank", "pp_rank"):
                    if key in attrs:
                        values[key] = attrs[key]
                name = str(span.get("name", "")).lower()
                for role_name in ("prefill", "decode"):
                    if name.startswith(role_name + " "):
                        values["role"] = role_name
                memo[span_id] = values
                return values

            normalized = []
            cache_events = []
            roots = []
            stages_seen = set()
            for span in sorted(spans, key=lambda item: int(item["start_time_ns"])):
                context = inherited(span)
                role = role_for(span, context)
                dp_rank = str(context.get("dp_rank", "unlabelled"))
                duration_ms = (
                    int(span["end_time_ns"]) - int(span["start_time_ns"])
                ) / 1e6
                name = str(span.get("name", ""))
                if duration_ms >= 0:
                    stage_values[(role, name, dp_rank, str(span.get("service_name", "")))].append(
                        duration_ms
                    )
                stages_seen.add(name)
                if not span.get("parent_span_id"):
                    roots.append(
                        {
                            "name": name,
                            "service_name": span.get("service_name"),
                            "rid": context.get("rid"),
                            "bootstrap_room": context.get("bootstrap_room"),
                        }
                    )
                for event in span.get("events") or []:
                    if event.get("name") == "prefill_cache_lookup":
                        cache_events.append(
                            {
                                "time_ns": event.get("time_ns"),
                                "role": role,
                                "dp_rank": context.get("dp_rank"),
                                "rid": context.get("rid"),
                                "attributes": event.get("attributes") or {},
                            }
                        )
                normalized.append(
                    {
                        "name": name,
                        "service_name": span.get("service_name"),
                        "role": role,
                        "dp_rank": context.get("dp_rank"),
                        "rid": context.get("rid"),
                        "bootstrap_room": context.get("bootstrap_room"),
                        "start_time_ns": span.get("start_time_ns"),
                        "end_time_ns": span.get("end_time_ns"),
                        "duration_ms": duration_ms,
                        "attributes": span.get("attributes") or {},
                    }
                )
            for required in (
                "prefill_waiting",
                "prefill_forward",
                "prefill_transfer_kv_cache",
                "decode_bootstrap",
                "decode_transferred",
                "decode_waiting",
                "decode_forward",
                "mooncake_send",
                "mooncake_recv",
            ):
                if required in stages_seen:
                    completeness[required] += 1
            output.write(
                json.dumps(
                    {
                        "record_type": "trace",
                        "schema_version": 1,
                        "point": args.point,
                        "trace_id": trace_id,
                        "roots": roots,
                        "cache_events": cache_events,
                        "spans": normalized,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    summary: dict[str, Any] = {}
    for (role, name, dp_rank, service), values in sorted(stage_values.items()):
        summary.setdefault(role, {}).setdefault(name, {}).setdefault(service, {})[
            dp_rank
        ] = stats(values)
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "point": args.point,
        "input": str(args.input),
        "trace_count": len(traces),
        "stage_summary": summary,
        "trace_completeness": dict(completeness),
        "timeline_jsonl": str(timeline_path),
    }
    (args.output_dir / "stage-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "stage-summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
