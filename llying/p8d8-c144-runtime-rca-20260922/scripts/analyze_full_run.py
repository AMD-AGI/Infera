#!/usr/bin/env python3
"""Build phase-separated statistics for a completed C144 replacement run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from analyze_partial_run import (
    analyze_engine_window,
    analyze_rails,
    analyze_records,
    parse_completed_at,
    timeout_summary,
)


PHASE_RE = {
    "warmup_start": re.compile(r"^(\d\d:\d\d:\d\d\.\d+).*Phase warmup .* started"),
    "warmup_end": re.compile(r"^(\d\d:\d\d:\d\d\.\d+).*Phase warmup .* complete "),
    "profiling_start": re.compile(
        r"^(\d\d:\d\d:\d\d\.\d+).*Phase profiling .* started"
    ),
    "profiling_end": re.compile(
        r"^(\d\d:\d\d:\d\d\.\d+).*Phase profiling .* complete "
    ),
}


def phase_times(log: Path, date: dt.date) -> dict[str, dt.datetime]:
    result = {}
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for name, pattern in PHASE_RE.items():
                match = pattern.search(line)
                if match:
                    clock = dt.time.fromisoformat(match.group(1))
                    result[name] = dt.datetime.combine(
                        date,
                        clock,
                        tzinfo=dt.timezone.utc,
                    )
    missing = sorted(set(PHASE_RE) - set(result))
    if missing:
        raise ValueError(f"missing phase timestamps in {log}: {missing}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.run / "analysis/full-run-data.json"
    if output.exists() and not args.force:
        parser.error(f"output exists: {output}; pass --force")

    completed_at = parse_completed_at(args.run / "completed-at.txt")
    phases = phase_times(
        args.run / "logs/bench-console.log",
        completed_at.date(),
    )
    source_run_path = args.run / "reused-service-from.txt"
    source_run = Path(source_run_path.read_text(encoding="utf-8").strip())
    server_log = source_run / "launch/server-logs/decode-0.log"
    engine = args.run / "sampling/live/engine-metrics.jsonl"
    nodes = args.run / "sampling/live/node-runtime.jsonl"
    aggregate = json.loads(
        (args.run / "bench/agentx_conc144.json").read_text(encoding="utf-8")
    )

    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run": str(args.run.resolve()),
        "source_service_run": str(source_run),
        "completed_at": completed_at.isoformat(),
        "phase_windows": {
            "warmup": {
                "start": phases["warmup_start"].isoformat(),
                "end": phases["warmup_end"].isoformat(),
                "duration_s": (
                    phases["warmup_end"] - phases["warmup_start"]
                ).total_seconds(),
            },
            "profiling": {
                "start": phases["profiling_start"].isoformat(),
                "end": phases["profiling_end"].isoformat(),
                "duration_s": (
                    phases["profiling_end"] - phases["profiling_start"]
                ).total_seconds(),
            },
        },
        "aggregate": {
            "requests": aggregate["request_accounting"],
            "request_metrics": aggregate["request_metrics"],
            "server_metrics": aggregate["server_metrics"],
        },
        "records": analyze_records(
            args.run / "bench/aiperf_artifacts/profile_export.jsonl"
        ),
        "warmup": {
            "engine": analyze_engine_window(
                engine,
                phases["warmup_start"],
                phases["warmup_end"],
            ),
            "rails": analyze_rails(
                nodes,
                phases["warmup_start"],
                phases["warmup_end"],
            ),
            "waiting_for_input_timeouts": timeout_summary(
                server_log,
                phases["warmup_start"],
                phases["warmup_end"],
            ),
        },
        "profiling": {
            "engine": analyze_engine_window(
                engine,
                phases["profiling_start"],
                phases["profiling_end"],
            ),
            "rails": analyze_rails(
                nodes,
                phases["profiling_start"],
                phases["profiling_end"],
            ),
            "waiting_for_input_timeouts": timeout_summary(
                server_log,
                phases["profiling_start"],
                phases["profiling_end"],
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
