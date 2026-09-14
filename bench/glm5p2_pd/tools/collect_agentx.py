#!/usr/bin/env python3
"""Collect AgentX result JSON files into a multi-P/D-aware results.csv."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


COLUMNS = [
    "Point", "Model", "Hardware", "Framework", "Topology",
    "Prefill Workers", "Prefill TP", "Prefill EP", "Prefill GPUs",
    "Decode Workers", "Decode TP", "Decode EP", "Decode GPUs",
    "Total GPUs", "DP Attention", "Concurrency",
    "Token Throughput per Chip (tok/s/chip)",
    "P90 Interactivity (tok/s/user)",
    "Output Throughput/Chip (tok/s)",
    "Input Throughput/Chip (tok/s)",
    "Median ITL (s)", "P90 ITL (s)", "Median TTFT (s)",
    "Records Profiled", "Records Total", "Records Errored",
]


def integer(record: dict[str, Any], name: str, default: int = 0) -> int:
    value = record.get(name)
    return default if value in (None, "") else int(value)


def enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def gpu_shape(record: dict[str, Any]) -> dict[str, int]:
    prefill_workers = integer(record, "prefill_num_workers")
    decode_workers = integer(record, "decode_num_workers")
    prefill_tp = integer(record, "prefill_tp")
    decode_tp = integer(record, "decode_tp")
    prefill_ep = integer(record, "prefill_ep", 1)
    decode_ep = integer(record, "decode_ep", 1)
    prefill_gpus = integer(record, "num_prefill_gpu")
    decode_gpus = integer(record, "num_decode_gpu")

    if enabled(record.get("disagg")):
        if prefill_gpus <= 0:
            prefill_gpus = prefill_workers * prefill_tp
        if decode_gpus <= 0:
            decode_gpus = decode_workers * decode_tp
        total_gpus = prefill_gpus + decode_gpus
        if min(prefill_workers, decode_workers, prefill_tp, decode_tp, total_gpus) <= 0:
            raise ValueError("disaggregated result has incomplete P/D GPU metadata")
    else:
        tp = integer(record, "tp")
        pp = integer(record, "pp", 1)
        pcp = integer(record, "pcp_size", 1)
        total_gpus = tp * pp * pcp
        prefill_workers = prefill_workers or 1
        prefill_tp = prefill_tp or tp
        prefill_gpus = total_gpus
        decode_workers = decode_workers or 0
        decode_tp = decode_tp or 0
        decode_gpus = 0
    if total_gpus <= 0:
        raise ValueError("result has no usable GPU count")

    return {
        "prefill_workers": prefill_workers,
        "prefill_tp": prefill_tp,
        "prefill_ep": prefill_ep,
        "prefill_gpus": prefill_gpus,
        "decode_workers": decode_workers,
        "decode_tp": decode_tp,
        "decode_ep": decode_ep,
        "decode_gpus": decode_gpus,
        "total_gpus": total_gpus,
    }


def topology_label(shape: dict[str, int]) -> str:
    if shape["decode_workers"]:
        return (
            f"{shape['prefill_workers']}P{shape['decode_workers']}D "
            f"pTP{shape['prefill_tp']}/EP{shape['prefill_ep']} "
            f"dTP{shape['decode_tp']}/EP{shape['decode_ep']}"
        )
    return f"TP{shape['prefill_tp']}/EP{shape['prefill_ep']}"


def one(directory: Path, point: str) -> dict[str, Any] | None:
    results = sorted(directory.glob("agentx_conc*.json"))
    if not results:
        return None
    if len(results) != 1:
        raise ValueError("directory contains more than one AgentX aggregate JSON")
    record = json.loads(results[0].read_text(encoding="utf-8"))
    shape = gpu_shape(record)
    throughput = record["request_metrics"]["throughput"]
    latency = record["request_metrics"]["latency"]
    total_tput = float(throughput["total"]["tokens_per_second"])
    per_chip = total_tput / shape["total_gpus"]
    claimed = throughput.get("per_gpu", {}).get("total_tput_tps")
    if claimed is not None and not math.isclose(
        float(claimed), per_chip, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            f"aggregate per-GPU throughput uses a different GPU count "
            f"(expected {shape['total_gpus']})"
        )
    full_itl = latency["full_response_itl"]
    p90_itl = round(float(full_itl["p90"]), 5)
    dp_attention = enabled(record.get("prefill_dp_attention")) or enabled(
        record.get("decode_dp_attention") or record.get("dp_attention")
    )
    topology = topology_label(shape) + (" DPA" if dp_attention else "")
    accounting = record.get("request_accounting") or {}
    return {
        "Point": point,
        "Model": str(record["model"]).rstrip("/").rsplit("/", 1)[-1],
        "Hardware": record.get("hw") or "unknown",
        "Framework": record.get("framework") or "unknown",
        "Topology": topology,
        "Prefill Workers": shape["prefill_workers"],
        "Prefill TP": shape["prefill_tp"],
        "Prefill EP": shape["prefill_ep"],
        "Prefill GPUs": shape["prefill_gpus"],
        "Decode Workers": shape["decode_workers"],
        "Decode TP": shape["decode_tp"],
        "Decode EP": shape["decode_ep"],
        "Decode GPUs": shape["decode_gpus"],
        "Total GPUs": shape["total_gpus"],
        "DP Attention": str(dp_attention).lower(),
        "Concurrency": record.get("conc"),
        "Token Throughput per Chip (tok/s/chip)": per_chip,
        "P90 Interactivity (tok/s/user)": 1 / p90_itl if p90_itl else "",
        "Output Throughput/Chip (tok/s)": (
            float(throughput["output"]["tokens_per_second"]) / shape["total_gpus"]
        ),
        "Input Throughput/Chip (tok/s)": (
            float(throughput["input"]["tokens_per_second"]) / shape["total_gpus"]
        ),
        "Median ITL (s)": full_itl.get("p50"),
        "P90 ITL (s)": p90_itl,
        "Median TTFT (s)": latency.get("ttft", {}).get("p50"),
        "Records Profiled": record.get("num_requests_successful"),
        "Records Total": record.get("num_requests_total"),
        "Records Errored": accounting.get("records_error_dropped"),
    }


def collect(root: Path) -> Path:
    rows = []
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories):
        point = str(directory.relative_to(root)) if directory != root else root.name
        try:
            row = one(directory, point)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{directory}: {exc}") from exc
        if row:
            rows.append(row)
    if not rows:
        raise SystemExit(f"no agentx_conc*.json under {root}")

    output = root / "results.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['Point']:<24} {row['Topology']:<28} "
            f"c={row['Concurrency']:<4} "
            f"Y={row['Token Throughput per Chip (tok/s/chip)']:>9.1f} "
            f"X={row['P90 Interactivity (tok/s/user)']:>7.1f}"
        )
    print(f"\n{output} ({len(rows)} points)")
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: collect_agentx.py <result directory>")
    collect(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
