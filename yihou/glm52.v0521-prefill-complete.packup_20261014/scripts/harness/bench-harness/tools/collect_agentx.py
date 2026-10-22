#!/usr/bin/env python3
"""Purpose: Collect AgentX aggregate JSON files into one multi-P/D results table.
Usage:
    python3 tools/collect_agentx.py RESULT_DIRECTORY
Artifacts:
    One aggregated CSV.
Artifact paths:
    RESULT_DIRECTORY/results.csv.

Collect AgentX aggregate JSON files into a multi-P/D-aware results.csv.

Usage: collect_agentx.py RESULT_DIRECTORY
"""

import csv
import json
import math
import sys
from pathlib import Path


COLUMNS = [
    "Point", "Model", "Hardware", "Framework", "Topology",
    "Prefill Workers", "Prefill TP", "Prefill EP", "Prefill GPUs",
    "Decode Workers", "Decode TP", "Decode EP", "Decode GPUs",
    "Total GPUs", "DP Attention", "Concurrency",
    "Token Throughput per Chip (tok/s/chip)",
    "P90 Interactivity (tok/s/user)",
    "Output Throughput/Chip (tok/s)", "Input Throughput/Chip (tok/s)",
    "Median ITL (s)", "P90 ITL (s)", "Median TTFT (s)",
    "Records Profiled", "Records Total", "Records Errored",
]


def integer(record: dict, name: str, default: int = 0) -> int:
    value = record.get(name)
    return default if value in (None, "") else int(value)


def enabled(value) -> bool:
    return value.lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def shape(record: dict) -> dict[str, int]:
    result = {
        "prefill_workers": integer(record, "prefill_num_workers"),
        "prefill_tp": integer(record, "prefill_tp"),
        "prefill_ep": integer(record, "prefill_ep", 1),
        "prefill_gpus": integer(record, "num_prefill_gpu"),
        "decode_workers": integer(record, "decode_num_workers"),
        "decode_tp": integer(record, "decode_tp"),
        "decode_ep": integer(record, "decode_ep", 1),
        "decode_gpus": integer(record, "num_decode_gpu"),
    }
    if enabled(record.get("disagg")):
        result["prefill_gpus"] = result["prefill_gpus"] or result["prefill_workers"] * result["prefill_tp"]
        result["decode_gpus"] = result["decode_gpus"] or result["decode_workers"] * result["decode_tp"]
    result["total_gpus"] = result["prefill_gpus"] + result["decode_gpus"]
    if min(
        result["prefill_workers"], result["decode_workers"],
        result["prefill_tp"], result["decode_tp"], result["total_gpus"],
    ) <= 0:
        raise ValueError("result has incomplete P/D GPU metadata")
    return result


def label(value: dict[str, int], dpa: bool) -> str:
    text = (
        f"{value['prefill_workers']}P{value['decode_workers']}D "
        f"pTP{value['prefill_tp']}/EP{value['prefill_ep']} "
        f"dTP{value['decode_tp']}/EP{value['decode_ep']}"
    )
    return text + (" DPA" if dpa else "")


def one(directory: Path, point: str):
    files = sorted(directory.glob("agentx_conc*.json"))
    if not files:
        return None
    if len(files) != 1:
        raise ValueError("directory contains more than one AgentX aggregate")
    runner_log = directory / "runner.log"
    if runner_log.is_file() and "ERROR: agentic trace replay exited with code" in (
        runner_log.read_text(encoding="utf-8", errors="replace")
    ):
        print(f"skipping failed AgentX point: {directory}", file=sys.stderr)
        return None
    record = json.loads(files[0].read_text(encoding="utf-8"))
    gpu = shape(record)
    throughput = record["request_metrics"]["throughput"]
    latency = record["request_metrics"]["latency"]
    per_chip = float(throughput["total"]["tokens_per_second"]) / gpu["total_gpus"]
    claimed = throughput.get("per_gpu", {}).get("total_tput_tps")
    if claimed is not None and not math.isclose(float(claimed), per_chip, rel_tol=1e-5):
        raise ValueError("aggregate per-GPU throughput uses a different GPU count")
    full_itl = latency["full_response_itl"]
    p90_itl = round(float(full_itl["p90"]), 5)
    dpa = enabled(record.get("prefill_dp_attention")) or enabled(
        record.get("decode_dp_attention") or record.get("dp_attention")
    )
    accounting = record.get("request_accounting") or {}
    return {
        "Point": point,
        "Model": str(record["model"]).rstrip("/").rsplit("/", 1)[-1],
        "Hardware": record.get("hw") or "unknown",
        "Framework": record.get("framework") or "unknown",
        "Topology": label(gpu, dpa),
        "Prefill Workers": gpu["prefill_workers"], "Prefill TP": gpu["prefill_tp"],
        "Prefill EP": gpu["prefill_ep"], "Prefill GPUs": gpu["prefill_gpus"],
        "Decode Workers": gpu["decode_workers"], "Decode TP": gpu["decode_tp"],
        "Decode EP": gpu["decode_ep"], "Decode GPUs": gpu["decode_gpus"],
        "Total GPUs": gpu["total_gpus"], "DP Attention": str(dpa).lower(),
        "Concurrency": record.get("conc"),
        "Token Throughput per Chip (tok/s/chip)": per_chip,
        "P90 Interactivity (tok/s/user)": 1 / p90_itl if p90_itl else "",
        "Output Throughput/Chip (tok/s)": float(throughput["output"]["tokens_per_second"]) / gpu["total_gpus"],
        "Input Throughput/Chip (tok/s)": float(throughput["input"]["tokens_per_second"]) / gpu["total_gpus"],
        "Median ITL (s)": full_itl.get("p50"), "P90 ITL (s)": p90_itl,
        "Median TTFT (s)": latency.get("ttft", {}).get("p50"),
        "Records Profiled": record.get("num_requests_successful"),
        "Records Total": record.get("num_requests_total"),
        "Records Errored": accounting.get("records_error_dropped"),
    }


def collect(root: Path) -> Path:
    rows = []
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories):
        try:
            row = one(directory, str(directory.relative_to(root)) if directory != root else root.name)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
            f"{row['Point']:<24} {row['Topology']:<28} c={row['Concurrency']:<4} "
            f"Y={row['Token Throughput per Chip (tok/s/chip)']:>9.1f} "
            f"X={row['P90 Interactivity (tok/s/user)']:>7.1f}"
        )
    print(f"{output} ({len(rows)} points)")
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: collect_agentx.py RESULT_DIRECTORY")
    collect(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
