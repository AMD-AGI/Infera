#!/usr/bin/env python3
"""Analyze one raw GLM-5.2 projection scan without modifying it.

Usage (run from ``bench/glm5p2_pd``):

    ./analyze_projection_sweep.py results/projection-1p1d \
      --output-dir results/projection-1p1d/analysis/baseline

    ./analyze_projection_sweep.py results/projection-1p1d \
      --interactivity-floor 144 \
      --output-dir results/projection-1p1d/analysis/interactivity-144

Omit ``--output-dir`` for a timestamped analysis directory. The analyzer also
works while scanning: scheduled but unfinished points are marked ``pending``.
It never writes under ``RUN_DIR/raw``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
HIGH_CONCURRENCY_PROBES = (96, 128, 192, 256, 512)

CSV_FIELDS = [
    "point_id",
    "status",
    "error",
    "raw_attempt",
    "projection_source",
    "calibration",
    "total_gpus",
    "prefill_gpus",
    "prefill_gpus_per_replica",
    "prefill_replicas",
    "decode_gpus",
    "decode_gpus_per_replica",
    "decode_replicas",
    "prefill_mode",
    "prefill_tp",
    "prefill_ep",
    "prefill_attention_dp",
    "decode_mode",
    "decode_tp",
    "decode_ep",
    "decode_attention_dp",
    "concurrency",
    "input_len",
    "output_len",
    "prefix_cache_hit_rate",
    "draft_cost_factor",
    "ttft_ms",
    "prefill_compute_ttft_ms",
    "kv_transfer_ms",
    "decode_total_ms",
    "itl_ms",
    "interactivity_tok_s_user",
    "request_latency_ms",
    "decode_step_latency_ms",
    "prefill_throughput_tps",
    "prefill_capacity_req_s",
    "decode_capacity_tps",
    "decode_capacity_req_s",
    "system_decode_throughput_tps",
    "system_request_throughput_req_s",
    "total_throughput_tps",
    "total_throughput_tps_per_gpu",
    "decode_throughput_tps_per_decode_gpu",
    "bottleneck",
    "prefill_occupancy",
    "speculative_tokens_per_step",
    "prefill_comm_ms",
    "decode_comm_ms",
    "prefill_weight_gb",
    "prefill_kv_cache_gb",
    "prefill_activation_gb",
    "prefill_memory_per_gpu_gb",
    "decode_weight_gb",
    "decode_kv_cache_gb",
    "decode_activation_gb",
    "decode_memory_per_gpu_gb",
    "usable_hbm_per_gpu_gb",
    "prefill_max_concurrent_sequences_per_replica",
    "decode_max_concurrent_sequences_per_replica",
    "prefill_max_concurrent_sequences",
    "decode_max_concurrent_sequences",
    "prefill_memory_fits",
    "decode_memory_fits",
    "concurrency_over_capacity",
    "projection_seconds",
]

MEMORY_COMBINATION_FIELDS = [
    "combination_id",
    "total_gpus",
    "prefill_gpus",
    "prefill_gpus_per_replica",
    "prefill_replicas",
    "decode_gpus",
    "decode_gpus_per_replica",
    "decode_replicas",
    "prefill_mode",
    "prefill_tp",
    "prefill_ep",
    "prefill_attention_dp",
    "decode_mode",
    "decode_tp",
    "decode_ep",
    "decode_attention_dp",
    "prefill_max_concurrency",
    "decode_max_concurrency",
    "memory_max_concurrency",
    "scanned_concurrencies",
    "keep_concurrencies",
    "cut_concurrencies",
    "supplement_concurrencies",
    "recommendation",
]

MEMORY_ACTION_FIELDS = [
    "action",
    "point_id",
    "combination_id",
    "total_gpus",
    "prefill_mode",
    "decode_mode",
    "concurrency",
    "memory_max_concurrency",
    "reason",
]

MEMORY_GROUP_FIELDS = (
    "total_gpus",
    "prefill_gpus",
    "prefill_gpus_per_replica",
    "prefill_replicas",
    "decode_gpus",
    "decode_gpus_per_replica",
    "decode_replicas",
    "prefill_mode",
    "prefill_tp",
    "prefill_ep",
    "prefill_attention_dp",
    "decode_mode",
    "decode_tp",
    "decode_ep",
    "decode_attention_dp",
    "input_len",
    "output_len",
    "prefix_cache_hit_rate",
    "draft_cost_factor",
)


def read_jsonl_snapshot(
    path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Read and hash the same bytes so a live scan has a reproducible snapshot."""
    if not path.is_file():
        return [], None
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if record.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(f"{path}:{line_number}: unsupported schema version")
        records.append(record)
    return records, digest


def latest_projection_records(
    schedule: Sequence[dict[str, Any]],
    attempts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = {record["point_id"]: record for record in attempts}
    return [latest.get(record["point_id"], record) for record in schedule]


def flatten_record(
    record: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    spec = record["spec"]
    prefill = spec["prefill"]
    decode = spec["decode"]
    workload = spec["workload"]
    projection = record.get("projection") or {}
    row = {
        "point_id": record["point_id"],
        "status": record.get("status", "pending"),
        "error": record.get("error", ""),
        "raw_attempt": record.get("attempt", 0),
        "projection_source": projection.get("source", ""),
        "calibration": projection.get("calibration", ""),
        "total_gpus": spec["total_gpus"],
        "prefill_gpus": prefill["gpus"],
        "prefill_gpus_per_replica": prefill.get(
            "gpus_per_replica",
            prefill["gpus"],
        ),
        "prefill_replicas": prefill.get("replicas", 1),
        "decode_gpus": decode["gpus"],
        "decode_gpus_per_replica": decode.get(
            "gpus_per_replica",
            decode["gpus"],
        ),
        "decode_replicas": decode.get("replicas", 1),
        "prefill_mode": prefill["mode_label"],
        "prefill_tp": prefill["tp"],
        "prefill_ep": prefill["ep"],
        "prefill_attention_dp": prefill["attention_dp"],
        "decode_mode": decode["mode_label"],
        "decode_tp": decode["tp"],
        "decode_ep": decode["ep"],
        "decode_attention_dp": decode["attention_dp"],
        "concurrency": spec["concurrency"],
        "input_len": workload["input_len"],
        "output_len": workload["output_len"],
        "prefix_cache_hit_rate": settings["prefix_cache_hit_rate"],
        "draft_cost_factor": spec["draft_cost_factor"],
        "projection_seconds": record.get("elapsed_seconds", ""),
    }
    performance = projection.get("performance")
    prefill_memory = projection.get("prefill_memory")
    decode_memory = projection.get("decode_memory")
    if not performance or not prefill_memory or not decode_memory:
        return row

    extras = performance.get("extras") or {}
    input_len = max(1, int(workload["input_len"]))
    output_len = max(1, int(workload["output_len"]))
    concurrency = int(spec["concurrency"])
    prefill_replicas = max(1, int(prefill.get("replicas", 1)))
    decode_replicas = max(1, int(decode.get("replicas", 1)))
    decode_batch = max(1, concurrency // decode_replicas)
    spec_tokens = float(extras.get("speculative_tokens_per_step", 1.0))
    step_ms = float(performance["decode_step_latency_ms"])
    decode_capacity_tps = (
        decode_batch * spec_tokens * 1000.0 / step_ms * decode_replicas
        if step_ms > 0
        else 0.0
    )
    prefill_capacity = float(performance["prefill_throughput_tps"]) / input_len
    decode_capacity = decode_capacity_tps / output_len
    system_decode_tps = float(performance["decode_throughput_tps"])
    system_request_tps = system_decode_tps / output_len
    total_tps = system_request_tps * (input_len + output_len)
    if prefill_capacity < decode_capacity * 0.99:
        bottleneck = "prefill"
    elif decode_capacity < prefill_capacity * 0.99:
        bottleneck = "decode"
    else:
        bottleneck = "balanced"

    hbm_bytes = float(decode_memory.get("hbm_capacity_bytes") or 0)
    usable_hbm = (
        hbm_bytes
        * float(settings["kv_cache_memory_fraction"])
        / (1024.0**3)
        if hbm_bytes
        else None
    )
    prefill_max_per_replica = prefill_memory.get("max_concurrent_sequences")
    decode_max_per_replica = decode_memory.get("max_concurrent_sequences")
    prefill_max = (
        int(prefill_max_per_replica) * prefill_replicas
        if prefill_max_per_replica is not None
        else None
    )
    decode_max = (
        int(decode_max_per_replica) * decode_replicas
        if decode_max_per_replica is not None
        else None
    )
    over_capacity = (
        (prefill_max is not None and concurrency > int(prefill_max))
        or (decode_max is not None and concurrency > int(decode_max))
    )
    row.update(
        {
            "ttft_ms": float(performance["ttft_ms"]),
            "prefill_compute_ttft_ms": float(
                extras.get("prefill_compute_ttft_ms", 0.0)
            ),
            "kv_transfer_ms": float(performance.get("kv_transfer_ms", 0.0)),
            "decode_total_ms": float(performance["decode_total_ms"]),
            "itl_ms": float(performance["itl_ms"]),
            "interactivity_tok_s_user": float(
                performance["per_request_decode_tps"]
            ),
            "request_latency_ms": float(performance["request_latency_ms"]),
            "decode_step_latency_ms": step_ms,
            "prefill_throughput_tps": float(
                performance["prefill_throughput_tps"]
            ),
            "prefill_capacity_req_s": prefill_capacity,
            "decode_capacity_tps": decode_capacity_tps,
            "decode_capacity_req_s": decode_capacity,
            "system_decode_throughput_tps": system_decode_tps,
            "system_request_throughput_req_s": system_request_tps,
            "total_throughput_tps": total_tps,
            "total_throughput_tps_per_gpu": total_tps
            / int(spec["total_gpus"]),
            "decode_throughput_tps_per_decode_gpu": system_decode_tps
            / int(decode["gpus"]),
            "bottleneck": bottleneck,
            "prefill_occupancy": extras.get("prefill_occupancy", 0),
            "speculative_tokens_per_step": spec_tokens,
            "prefill_comm_ms": extras.get("comm_prefill_total_ms", 0.0),
            "decode_comm_ms": extras.get("comm_decode_total_ms", 0.0),
            "prefill_weight_gb": _bytes_to_gb(prefill_memory["weight_bytes"]),
            "prefill_kv_cache_gb": _bytes_to_gb(
                prefill_memory["kv_cache_bytes"]
            ),
            "prefill_activation_gb": _bytes_to_gb(
                prefill_memory["activation_bytes"]
            ),
            "prefill_memory_per_gpu_gb": _bytes_to_gb(
                prefill_memory["total_bytes"]
            ),
            "decode_weight_gb": _bytes_to_gb(decode_memory["weight_bytes"]),
            "decode_kv_cache_gb": _bytes_to_gb(
                decode_memory["kv_cache_bytes"]
            ),
            "decode_activation_gb": _bytes_to_gb(
                decode_memory["activation_bytes"]
            ),
            "decode_memory_per_gpu_gb": _bytes_to_gb(
                decode_memory["total_bytes"]
            ),
            "usable_hbm_per_gpu_gb": usable_hbm,
            "prefill_max_concurrent_sequences_per_replica": (
                prefill_max_per_replica
            ),
            "decode_max_concurrent_sequences_per_replica": (
                decode_max_per_replica
            ),
            "prefill_max_concurrent_sequences": prefill_max,
            "decode_max_concurrent_sequences": decode_max,
            "prefill_memory_fits": prefill_memory.get("fits"),
            "decode_memory_fits": decode_memory.get("fits"),
            "concurrency_over_capacity": over_capacity,
        }
    )
    return row


def _bytes_to_gb(value: Any) -> float:
    return float(value) / (1024.0**3)


def pareto_front(
    rows: Sequence[dict[str, Any]],
    *,
    x: str = "interactivity_tok_s_user",
    y: str = "total_throughput_tps_per_gpu",
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["status"] == "ok" and _finite(row.get(x)) and _finite(row.get(y))
    ]
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        cx, cy = float(candidate[x]), float(candidate[y])
        dominated = any(
            float(other[x]) >= cx
            and float(other[y]) >= cy
            and (float(other[x]) > cx or float(other[y]) > cy)
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (float(row[x]), float(row[y])))


def grouped_pareto(
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        output.extend(pareto_front(groups[key]))
    return output


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def shortlist(
    rows: Sequence[dict[str, Any]],
    interactivity_floor: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (row["total_gpus"], row["concurrency"], row["draft_cost_factor"])
        ].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        for row in pareto_front(group):
            selected[row["point_id"]] = row
        eligible = [
            row
            for row in group
            if float(row["interactivity_tok_s_user"]) >= interactivity_floor
        ]
        if eligible:
            winner = max(
                eligible,
                key=lambda row: float(row["total_throughput_tps_per_gpu"]),
            )
            selected[winner["point_id"]] = winner
    return sorted(selected.values(), key=_row_sort_key)


def memory_concurrency_plan(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return combination limits and explicit cut/supplement point actions."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in MEMORY_GROUP_FIELDS)].append(row)

    combinations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for group in groups.values():
        first = group[0]
        prefill_limits = {
            int(row["prefill_max_concurrent_sequences"])
            for row in group
            if row.get("prefill_max_concurrent_sequences") not in (None, "")
        }
        decode_limits = {
            int(row["decode_max_concurrent_sequences"])
            for row in group
            if row.get("decode_max_concurrent_sequences") not in (None, "")
        }
        if not prefill_limits or not decode_limits:
            continue
        if len(prefill_limits) != 1 or len(decode_limits) != 1:
            raise ValueError(
                f"inconsistent memory limits for {_memory_combination_id(first)}"
            )
        prefill_max = next(iter(prefill_limits))
        decode_max = next(iter(decode_limits))
        memory_max = min(prefill_max, decode_max)
        scanned = sorted({int(row["concurrency"]) for row in group})
        keep = [value for value in scanned if value <= memory_max]
        cut = [value for value in scanned if value > memory_max]
        supplements = _supplement_concurrencies(memory_max, scanned)
        if cut and supplements:
            recommendation = "cut_above_limit_and_test_boundary"
        elif cut:
            recommendation = "cut_above_limit"
        elif supplements:
            recommendation = "keep_grid_and_test_high_concurrency"
        else:
            recommendation = "keep_grid"
        combination_id = _memory_combination_id(first)
        combination = {
            "combination_id": combination_id,
            "total_gpus": first["total_gpus"],
            "prefill_gpus": first["prefill_gpus"],
            "prefill_gpus_per_replica": first["prefill_gpus_per_replica"],
            "prefill_replicas": first["prefill_replicas"],
            "decode_gpus": first["decode_gpus"],
            "decode_gpus_per_replica": first["decode_gpus_per_replica"],
            "decode_replicas": first["decode_replicas"],
            "prefill_mode": first["prefill_mode"],
            "prefill_tp": first["prefill_tp"],
            "prefill_ep": first["prefill_ep"],
            "prefill_attention_dp": first["prefill_attention_dp"],
            "decode_mode": first["decode_mode"],
            "decode_tp": first["decode_tp"],
            "decode_ep": first["decode_ep"],
            "decode_attention_dp": first["decode_attention_dp"],
            "prefill_max_concurrency": prefill_max,
            "decode_max_concurrency": decode_max,
            "memory_max_concurrency": memory_max,
            "scanned_concurrencies": _join_ints(scanned),
            "keep_concurrencies": _join_ints(keep),
            "cut_concurrencies": _join_ints(cut),
            "supplement_concurrencies": _join_ints(supplements),
            "recommendation": recommendation,
        }
        combinations.append(combination)
        for row in group:
            concurrency = int(row["concurrency"])
            if concurrency > memory_max:
                actions.append(
                    _memory_action(
                        "CUT",
                        row["point_id"],
                        combination,
                        concurrency,
                        f"requested {concurrency} > memory max {memory_max}",
                    )
                )
        for supplement in supplements:
            reason = (
                f"test exact memory boundary {memory_max}"
                if supplement == memory_max
                else (
                    f"intermediate high-concurrency probe below "
                    f"memory max {memory_max}"
                )
            )
            actions.append(
                _memory_action(
                    "SUPPLEMENT",
                    "",
                    combination,
                    supplement,
                    reason,
                )
            )
    combinations.sort(
        key=lambda row: (
            int(row["total_gpus"]),
            int(row["prefill_gpus"]),
            int(row["prefill_replicas"]),
            str(row["prefill_mode"]),
            int(row["decode_gpus"]),
            int(row["decode_replicas"]),
            str(row["decode_mode"]),
        )
    )
    action_order = {"CUT": 0, "SUPPLEMENT": 1}
    actions.sort(
        key=lambda row: (
            int(row["total_gpus"]),
            str(row["combination_id"]),
            action_order[str(row["action"])],
            int(row["concurrency"]),
        )
    )
    return combinations, actions


def _memory_combination_id(row: Mapping[str, Any]) -> str:
    prefill_replicas = int(row.get("prefill_replicas", 1))
    decode_replicas = int(row.get("decode_replicas", 1))
    replica_tag = (
        ""
        if prefill_replicas == 1 and decode_replicas == 1
        else f"_np{prefill_replicas}_nd{decode_replicas}"
    )
    return (
        f"g{row['total_gpus']}_"
        f"p{row['prefill_tp']}-{_mode_id(row['prefill_mode'])}_"
        f"d{row['decode_tp']}-{_mode_id(row['decode_mode'])}"
        f"{replica_tag}"
    )


def _mode_id(mode: Any) -> str:
    return str(mode).lower().replace("+", "_")


def _join_ints(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _supplement_concurrencies(
    memory_max: int,
    scanned: Sequence[int],
) -> list[int]:
    if memory_max <= max(scanned):
        return [] if memory_max in scanned else [memory_max]
    probes = [value for value in HIGH_CONCURRENCY_PROBES if value < memory_max]
    return list(dict.fromkeys([*probes, memory_max]))


def _memory_action(
    action: str,
    point_id: str,
    combination: Mapping[str, Any],
    concurrency: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "point_id": point_id,
        "combination_id": combination["combination_id"],
        "total_gpus": combination["total_gpus"],
        "prefill_mode": combination["prefill_mode"],
        "decode_mode": combination["decode_mode"],
        "concurrency": concurrency,
        "memory_max_concurrency": combination["memory_max_concurrency"],
        "reason": reason,
    }


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("total_gpus", 0)),
        int(row.get("prefill_gpus", 0)),
        int(row.get("prefill_replicas", 1)),
        str(row.get("prefill_mode", "")),
        int(row.get("decode_gpus", 0)),
        int(row.get("decode_replicas", 1)),
        str(row.get("decode_mode", "")),
        int(row.get("concurrency", 0)),
        float(row.get("draft_cost_factor", 0.0)),
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_dict_csv(path, CSV_FIELDS, rows)


def write_dict_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_summary(
    rows: Sequence[dict[str, Any]],
    pareto_budget: Sequence[dict[str, Any]],
    interactivity_floor: float,
) -> str:
    lines = [
        "GLM-5.2 P/D projection analysis",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        f"scheduled projections: {len(rows)}",
        f"status: {dict(sorted(Counter(row['status'] for row in rows).items()))}",
        "Pareto x-axis: modeled mean interactivity (not P90)",
        f"modeled interactivity floor: {interactivity_floor:g} tok/s/user",
        "",
    ]
    feasible = [row for row in rows if row["status"] == "ok"]
    for budget in sorted({int(row["total_gpus"]) for row in rows}):
        group = [row for row in feasible if int(row["total_gpus"]) == budget]
        front = [row for row in pareto_budget if int(row["total_gpus"]) == budget]
        lines.append(f"{budget} GPU")
        lines.append(f"  feasible: {len(group)}; Pareto: {len(front)}")
        if group:
            efficient = max(
                group,
                key=lambda row: float(row["total_throughput_tps_per_gpu"]),
            )
            interactive = max(
                group,
                key=lambda row: float(row["interactivity_tok_s_user"]),
            )
            lines.append(f"  max throughput/GPU: {_summary_point(efficient)}")
            lines.append(f"  max interactivity: {_summary_point(interactive)}")
        lines.append("")
    return "\n".join(lines)


def _summary_point(row: Mapping[str, Any]) -> str:
    return (
        f"{row['point_id']} "
        f"total={float(row['total_throughput_tps_per_gpu']):.1f} tok/s/GPU, "
        f"interactivity={float(row['interactivity_tok_s_user']):.1f} tok/s/user, "
        f"TTFT={float(row['ttft_ms']):.1f} ms, bottleneck={row['bottleneck']}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: RUN_DIR/analysis/TIMESTAMP",
    )
    parser.add_argument(
        "--interactivity-floor",
        type=float,
        default=0.0,
        help="modeled mean interactivity threshold for shortlist selection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_config.json"
    schedule_path = run_dir / "raw" / "schedule.jsonl"
    projections_path = run_dir / "raw" / "projections.jsonl"
    if not manifest_path.is_file() or not schedule_path.is_file():
        parser.error(f"not a projection sweep run directory: {run_dir}")
    if args.interactivity_floor < 0:
        parser.error("--interactivity-floor cannot be negative")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        parser.error("unsupported run schema version")
    schedule, schedule_sha256 = read_jsonl_snapshot(schedule_path)
    attempts, projections_sha256 = read_jsonl_snapshot(projections_path)
    latest = latest_projection_records(schedule, attempts)
    rows = sorted(
        [
            flatten_record(record, manifest["settings"])
            for record in latest
        ],
        key=_row_sort_key,
    )
    feasible = [row for row in rows if row["status"] == "ok"]
    pareto_global = grouped_pareto(
        feasible,
        ("input_len", "output_len", "prefix_cache_hit_rate", "draft_cost_factor"),
    )
    pareto_budget = grouped_pareto(
        feasible,
        (
            "total_gpus",
            "input_len",
            "output_len",
            "prefix_cache_hit_rate",
            "draft_cost_factor",
        ),
    )
    memory_combinations, memory_actions = memory_concurrency_plan(rows)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"analysis output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "feasible.csv", feasible)
    write_csv(output_dir / "pareto_global.csv", pareto_global)
    write_csv(output_dir / "pareto_by_budget.csv", pareto_budget)
    write_dict_csv(
        output_dir / "memory_limits_by_combination.csv",
        MEMORY_COMBINATION_FIELDS,
        memory_combinations,
    )
    write_dict_csv(
        output_dir / "memory_point_actions.csv",
        MEMORY_ACTION_FIELDS,
        memory_actions,
    )
    write_csv(
        output_dir / "shortlist.csv",
        shortlist(feasible, args.interactivity_floor),
    )
    (output_dir / "summary.txt").write_text(
        render_summary(rows, pareto_budget, args.interactivity_floor),
        encoding="utf-8",
    )
    analysis_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "interactivity_floor": args.interactivity_floor,
        "schedule_sha256": schedule_sha256,
        "projections_sha256": projections_sha256,
        "raw_attempt_records": len(attempts),
        "latest_completed_points": sum(
            record.get("status") is not None for record in latest
        ),
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"analysis directory: {output_dir}")
    print(f"raw records read: {len(attempts)}; scheduled: {len(schedule)}")
    print(f"results: {output_dir / 'results.csv'}")
    print(f"Pareto:  {output_dir / 'pareto_by_budget.csv'}")
    print(f"memory:  {output_dir / 'memory_limits_by_combination.csv'}")
    print(f"actions: {output_dir / 'memory_point_actions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
