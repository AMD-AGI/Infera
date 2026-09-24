#!/usr/bin/env python3
"""Tabulate AgentX aggregates from run directories into results/summary.{md,csv}.

Usage: scripts/summarize.py RUN_DIR [RUN_DIR ...]
Each RUN_DIR holds agentx/agentx_conc<N>.json; the JSON is also copied to
results/c<NNN>/. Per-GPU values divide by the GPU count recorded in the JSON.
"""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
COLUMNS = {
    "conc": "conc", "gpus": "GPUs", "total_tok_s_gpu": "total tok/s/GPU",
    "output_tok_s_gpu": "output tok/s/GPU", "ttft_p50_s": "TTFT p50 s",
    "ttft_p90_s": "TTFT p90 s", "itl_p50_ms": "ITL p50 ms", "intvty_p50": "intvty p50",
    "profiled": "profiled", "errors": "error dropped", "duration_s": "duration s",
    "run": "run",
}


def row(path: Path) -> dict:
    r = json.loads(path.read_text())
    tput, lat = r["request_metrics"]["throughput"], r["request_metrics"]["latency"]
    gpus = r["num_prefill_gpu"] + r["num_decode_gpu"]
    return {
        "conc": r["conc"], "gpus": gpus,
        "total_tok_s_gpu": round(tput["total"]["tokens_per_second"] / gpus, 2),
        "output_tok_s_gpu": round(tput["output"]["tokens_per_second"] / gpus, 2),
        "ttft_p50_s": round(lat["ttft"]["p50"], 3),
        "ttft_p90_s": round(lat["ttft"]["p90"], 3),
        "itl_p50_ms": round(lat["full_response_itl"]["p50"] * 1000, 3),
        "intvty_p50": round(lat["full_response_intvty"]["p50"], 2),
        "profiled": r["request_accounting"]["records_profiled"],
        "errors": r["request_accounting"]["records_error_dropped"],
        "duration_s": round(tput["duration_seconds"]),
        "run": path.parent.parent.name,
    }


def main() -> None:
    paths = sorted(p for d in sys.argv[1:] for p in Path(d).glob("agentx/agentx_conc*.json"))
    rows = sorted((row(p) for p in paths), key=lambda r: r["conc"])
    for p in paths:
        dest = RESULTS / f"c{json.loads(p.read_text())['conc']:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / p.name)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(COLUMNS.values()) + " |",
             "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(str(r[k]) for k in COLUMNS) + " |" for r in rows]
    (RESULTS / "summary.md").write_text(
        "# GLM-5.2 MXFP4 8P4D (ATOM + Infera) AgentX\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
