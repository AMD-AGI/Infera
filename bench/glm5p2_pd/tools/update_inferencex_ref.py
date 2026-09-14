#!/usr/bin/env python3
"""Download the current official InferenceX GLM-5.2 AgentX reference data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://inferencex.semianalysis.com/api/v1/benchmarks"
MODEL = "GLM-5.2"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "ref"
    / "InferenceX_GLM-5.2_interactivity.csv"
)
COLUMNS = [
    "Model", "ISL", "OSL", "Hardware", "Hardware Key", "Framework",
    "Precision", "TP", "Concurrency", "Date",
    "Token Throughput per Chip (tok/s/chip)",
    "P90 Interactivity (tok/s/user)", "Throughput/Chip (tok/s)",
    "Output Throughput/Chip (tok/s)", "Input Throughput/Chip (tok/s)",
    "Mean TTFT (s)", "Median TTFT (s)", "P99 TTFT (s)", "Std TTFT (s)",
    "Mean TPOT (s)", "Median TPOT (s)", "P99 TPOT (s)", "Std TPOT (s)",
    "Mean Interactivity (tok/s/user)", "Median Interactivity (tok/s/user)",
    "P99 Interactivity (tok/s/user)", "Std Interactivity (tok/s/user)",
    "Mean ITL (s)", "Median ITL (s)", "P99 ITL (s)", "Std ITL (s)",
    "Mean E2E Latency (s)", "Median E2E Latency (s)",
    "P99 E2E Latency (s)", "Std E2E Latency (s)",
    "Disaggregated", "Num Prefill Chips", "Num Decode Chips",
    "Spec Decoding", "EP", "DP Attention", "Is Multinode", "Run URL",
]


def boolean(value: Any) -> str:
    return str(bool(value)).lower()


def reference_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    if record.get("benchmark_type") != "agentic_traces" or not isinstance(metrics, dict):
        raise ValueError("expected an AgentX record with metrics")

    disagg = bool(record.get("disagg"))
    prefill_gpus = int(record.get("num_prefill_gpu") or 0)
    decode_gpus = int(record.get("num_decode_gpu") or 0)
    tp = (
        prefill_gpus + decode_gpus
        if disagg
        else int(record.get("prefill_tp") or prefill_gpus)
    )
    if tp <= 0:
        raise ValueError(f"record {record.get('id')} has no usable chip count")

    def metric(name: str) -> Any:
        value = metrics.get(name)
        return "" if value is None else value

    hardware = str(record.get("hardware") or "unknown")
    framework = str(record.get("framework") or "unknown")
    return {
        "Model": MODEL,
        "ISL": record.get("isl") or "",
        "OSL": record.get("osl") or "",
        "Hardware": hardware,
        "Hardware Key": f"{hardware}_{framework}",
        "Framework": framework,
        "Precision": record.get("precision") or "",
        "TP": tp,
        "Concurrency": record.get("conc") or "",
        "Date": record.get("date") or "",
        "Token Throughput per Chip (tok/s/chip)": metric("tput_per_gpu"),
        "P90 Interactivity (tok/s/user)": metric("p90_intvty"),
        "Throughput/Chip (tok/s)": metric("tput_per_gpu"),
        "Output Throughput/Chip (tok/s)": metric("output_tput_per_gpu"),
        "Input Throughput/Chip (tok/s)": metric("input_tput_per_gpu"),
        "Mean TTFT (s)": metric("mean_ttft"),
        "Median TTFT (s)": metric("median_ttft"),
        "P99 TTFT (s)": "",
        "Std TTFT (s)": metric("std_ttft"),
        "Mean TPOT (s)": metric("mean_tpot"),
        "Median TPOT (s)": metric("median_tpot"),
        "P99 TPOT (s)": "",
        "Std TPOT (s)": metric("std_tpot"),
        "Mean Interactivity (tok/s/user)": metric("mean_intvty"),
        "Median Interactivity (tok/s/user)": metric("median_intvty"),
        "P99 Interactivity (tok/s/user)": "",
        "Std Interactivity (tok/s/user)": metric("std_intvty"),
        "Mean ITL (s)": metric("mean_itl"),
        "Median ITL (s)": metric("median_itl"),
        "P99 ITL (s)": "",
        "Std ITL (s)": metric("std_itl"),
        "Mean E2E Latency (s)": metric("mean_e2el"),
        "Median E2E Latency (s)": metric("median_e2el"),
        "P99 E2E Latency (s)": "",
        "Std E2E Latency (s)": metric("std_e2el"),
        "Disaggregated": boolean(disagg),
        "Num Prefill Chips": prefill_gpus if disagg else "",
        "Num Decode Chips": decode_gpus if disagg else "",
        "Spec Decoding": record.get("spec_method") or "none",
        "EP": max(
            int(record.get("prefill_ep") or 1),
            int(record.get("decode_ep") or 1),
        ),
        "DP Attention": boolean(
            record.get("prefill_dp_attention") or record.get("decode_dp_attention")
        ),
        "Is Multinode": boolean(record.get("is_multinode")),
        "Run URL": record.get("run_url") or "",
    }


def fetch_records() -> list[dict[str, Any]]:
    url = f"{API}?{urlencode({'model': MODEL})}"
    request = Request(url, headers={"User-Agent": "glm5p2-pd-reference-updater/1"})
    with urlopen(request, timeout=60) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip" or body.startswith(b"\x1f\x8b"):
            body = gzip.decompress(body)
        payload = json.loads(body)
    if not isinstance(payload, list) or not payload:
        raise ValueError("InferenceX returned no benchmark records")
    return payload


def write_reference(output: Path, records: list[dict[str, Any]]) -> None:
    rows = [reference_row(record) for record in records]
    rows.sort(
        key=lambda row: (
            row["Hardware"],
            row["Framework"],
            row["Disaggregated"],
            row["TP"],
            row["Concurrency"],
        )
    )
    latest = max(str(row["Date"]) for row in rows)
    retrieved = datetime.now(timezone.utc).date().isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "# Licensed under Apache License 2.0 — "
            "https://www.apache.org/licenses/LICENSE-2.0\n"
        )
        stream.write(
            "# Copyright 2026 SemiAnalysis LLC. Data from InferenceX "
            "(https://inferencex.semianalysis.com/inference/glm-5-3).\n"
        )
        stream.write(
            f"# Official API snapshot retrieved {retrieved}; "
            f"latest included benchmark date {latest}; {len(rows)} rows.\n"
        )
        stream.write("# Attribution to InferenceX is required for derivative work.\n")
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} rows, latest benchmark date {latest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_reference(args.output.resolve(), fetch_records())


if __name__ == "__main__":
    main()
