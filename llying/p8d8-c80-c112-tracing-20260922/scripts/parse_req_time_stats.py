#!/usr/bin/env python3
"""Parse SGLang ReqTimeStats lines into stage-duration JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


LINE_RE = re.compile(r"ReqTimeStats\((?P<header>.*?)\): (?P<body>.*)$")
FIELD_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^,]+)")
ISO_RE = re.compile(r"^(?P<timestamp>\S+)")


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("log must be ROLE=PATH")
    role, path = value.split("=", 1)
    if not role or not path:
        raise argparse.ArgumentTypeError("log must be ROLE=PATH")
    return role, Path(path)


def fields(text: str) -> dict[str, str]:
    return {
        match.group("name"): match.group("value").strip()
        for match in FIELD_RE.finditer(text)
    }


def scalar(raw: str) -> Any:
    for suffix, scale, unit in (
        ("ms", 1.0, "ms"),
        (" GB/s", 1.0, "GB/s"),
        (" MB", 1.0, "MB"),
    ):
        if raw.endswith(suffix):
            try:
                return {"value": float(raw[: -len(suffix)]), "unit": unit, "scale": scale}
            except ValueError:
                return raw
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_time(raw: str) -> dt.datetime:
    return dt.datetime.fromisoformat(raw.replace("Z", "+00:00").replace(",", "."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="inclusive ISO-8601 log timestamp")
    parser.add_argument("--end", help="inclusive ISO-8601 log timestamp")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else None

    count = 0
    with args.output.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "schema_version": 1,
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "inputs": {
                        role: str(path)
                        for role, path in args.log
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        for role, path in args.log:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    match = LINE_RE.search(line)
                    if match is None:
                        continue
                    timestamp = ISO_RE.match(line)
                    logged_at = timestamp.group("timestamp") if timestamp else None
                    logged_time = parse_time(logged_at) if logged_at else None
                    if start and (logged_time is None or logged_time < start):
                        continue
                    if end and (logged_time is None or logged_time > end):
                        continue
                    header = {
                        name: scalar(value)
                        for name, value in fields(match.group("header")).items()
                    }
                    body = {
                        name: scalar(value)
                        for name, value in fields(match.group("body")).items()
                    }
                    row = {
                        "record_type": "request_time_stats",
                        "schema_version": 1,
                        "source_role": role,
                        "source_path": str(path),
                        "source_line": line_number,
                        "logged_at": logged_at,
                        "rid": str(header.get("rid", "")),
                        "bootstrap_room": header.get("bootstrap_room"),
                        "header": header,
                        "durations": body,
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
    print(json.dumps({"records": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
