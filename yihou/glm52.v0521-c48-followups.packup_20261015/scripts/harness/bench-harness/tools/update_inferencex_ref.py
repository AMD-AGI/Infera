#!/usr/bin/env python3
"""Purpose: Refresh the official InferenceX GLM-5.2 AgentX reference snapshot.
Usage:
    python3 tools/update_inferencex_ref.py [--output PATH]
Artifacts:
    Reference CSV downloaded from the public InferenceX API.
Artifact paths:
    --output, default tools/ref/InferenceX_GLM-5.2_interactivity.csv.
"""

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://inferencex.semianalysis.com/api/v1/benchmarks"
MODEL = "GLM-5.2"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "ref"
    / "InferenceX_GLM-5.2_interactivity.csv"
)
X = "P90 Interactivity (tok/s/user)"
Y = "Token Throughput per Chip (tok/s/chip)"
COLUMNS = [
    "Model",
    "Hardware",
    "Framework",
    "Precision",
    "Disaggregated",
    "Prefill GPUs",
    "Decode GPUs",
    "Prefill TP",
    "Decode TP",
    "Prefill EP",
    "Decode EP",
    "Prefill DPA",
    "Decode DPA",
    "Concurrency",
    "Date",
    X,
    Y,
    "Run URL",
]


def text_bool(value) -> str:
    return str(bool(value)).lower()


def row(record: dict) -> dict | None:
    metrics = record.get("metrics")
    if record.get("benchmark_type") != "agentic_traces" or not isinstance(metrics, dict):
        return None
    if metrics.get("p90_intvty") is None or metrics.get("tput_per_gpu") is None:
        return None

    disagg = bool(record.get("disagg"))
    prefill_tp = int(record.get("prefill_tp") or record.get("tp") or 0)
    decode_tp = int(record.get("decode_tp") or 0) if disagg else 0
    prefill_gpus = int(record.get("num_prefill_gpu") or prefill_tp)
    decode_gpus = int(record.get("num_decode_gpu") or 0) if disagg else 0
    return {
        "Model": MODEL,
        "Hardware": record.get("hardware") or "unknown",
        "Framework": record.get("framework") or "unknown",
        "Precision": record.get("precision") or "",
        "Disaggregated": text_bool(disagg),
        "Prefill GPUs": prefill_gpus,
        "Decode GPUs": decode_gpus,
        "Prefill TP": prefill_tp,
        "Decode TP": decode_tp,
        "Prefill EP": int(record.get("prefill_ep") or 1),
        "Decode EP": int(record.get("decode_ep") or 1) if disagg else 0,
        "Prefill DPA": text_bool(record.get("prefill_dp_attention")),
        "Decode DPA": text_bool(record.get("decode_dp_attention")) if disagg else "false",
        "Concurrency": record.get("conc") or "",
        "Date": record.get("date") or "",
        X: metrics["p90_intvty"],
        Y: metrics["tput_per_gpu"],
        "Run URL": record.get("run_url") or "",
    }


def fetch() -> list[dict]:
    url = f"{API}?{urlencode({'model': MODEL})}"
    request = Request(url, headers={"User-Agent": "infera-glm5p2-reference/1"})
    with urlopen(request, timeout=60) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip" or body.startswith(b"\x1f\x8b"):
            body = gzip.decompress(body)
    records = json.loads(body)
    if not isinstance(records, list):
        raise ValueError("InferenceX API did not return a list")
    rows = [item for record in records if (item := row(record)) is not None]
    if not rows:
        raise ValueError("InferenceX API returned no usable GLM-5.2 AgentX rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = fetch()
    rows.sort(
        key=lambda item: (
            item["Hardware"],
            item["Framework"],
            item["Precision"],
            item["Disaggregated"],
            item["Prefill GPUs"],
            item["Decode GPUs"],
            item["Prefill TP"],
            item["Decode TP"],
            item["Prefill EP"],
            item["Decode EP"],
            item["Prefill DPA"],
            item["Decode DPA"],
            int(item["Concurrency"]),
        )
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    latest = max(str(item["Date"]) for item in rows)
    retrieved = datetime.now(timezone.utc).date().isoformat()
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "# Licensed under Apache License 2.0 — "
            "https://www.apache.org/licenses/LICENSE-2.0\n"
        )
        stream.write(
            "# Copyright 2026 SemiAnalysis LLC. Data from "
            "https://inferencex.semianalysis.com/.\n"
        )
        stream.write(
            f"# Official API snapshot retrieved {retrieved}; "
            f"latest benchmark {latest}; {len(rows)} rows.\n"
        )
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} rows, latest benchmark {latest}")


if __name__ == "__main__":
    main()
