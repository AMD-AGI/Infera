#!/usr/bin/env python3
"""ARG CHECK: every flag given to run_decode.sh must appear in config_yihou.json (server_cli or benchmark args).

GLM-5.2 handoff trap #10: a multi-line argument string inside ``ssh "..."`` dropped four flags while every
gate passed. Printing ``ARG CHECK: ok`` after each run makes that visible.
"""
import json
import sys
from pathlib import Path


def missing_flags(config_payload, given):
    cli = config_payload.get("server_cli", [])
    bench = config_payload.get("benchmark", {})
    missing = []
    for item in given:
        if not item.startswith("--"):
            continue
        name = item.split("=", 1)[0]
        if name in cli or name[2:].replace("-", "_") in bench:
            continue
        missing.append(name)
    return missing


def main(argv):
    out, given = Path(argv[0]), argv[1:]
    config = out / "config_yihou.json"
    if not config.is_file():
        print(f"ARG CHECK: skipped ({config} absent)")
        return 0
    missing = missing_flags(json.loads(config.read_text()), given)
    if missing:
        print(f"ARG CHECK: FAILED, not in config: {missing}")
        return 1
    print("ARG CHECK: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
