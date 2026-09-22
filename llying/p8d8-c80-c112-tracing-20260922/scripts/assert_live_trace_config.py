#!/usr/bin/env python3
"""Fail unless both live workers have the fixed tracing configuration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_IMAGE = (ROOT / "image-id.txt").read_text(encoding="utf-8").strip()
PREFIX = "glm52-pd-c80-c112-trace"
WORKERS = (
    ("crsuse2-m2m-138", f"{PREFIX}-prefill-0", "prefill"),
    ("crsuse2-m2m-136", f"{PREFIX}-decode-0", "decode"),
)
TRACE_FLAGS = {
    "--enable-trace",
    "--enable-request-time-stats-logging",
}


def inspect(node: str, name: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", node, "docker", "inspect", name],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)[0]


def option(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"missing {name}") from exc


def env_map(container: dict[str, Any]) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in (container.get("Config") or {}).get("Env", [])
        if isinstance(item, str) and "=" in item
    }


def main() -> int:
    result = {"workers": {}}
    for node, name, role in WORKERS:
        container = inspect(node, name)
        argv = [str(item) for item in (container.get("Config") or {}).get("Cmd", [])]
        env = env_map(container)
        if container.get("Image") != EXPECTED_IMAGE:
            raise SystemExit(f"{name}: image mismatch")
        if not (container.get("State") or {}).get("Running"):
            raise SystemExit(f"{name}: not running")
        missing = sorted(TRACE_FLAGS - set(argv))
        if missing:
            raise SystemExit(f"{name}: missing trace flags {missing}")
        if option(argv, "--trace-modules") != "request,mooncake":
            raise SystemExit(f"{name}: wrong trace modules")
        if option(argv, "--otlp-traces-endpoint") != "10.245.157.237:4317":
            raise SystemExit(f"{name}: wrong OTLP endpoint")
        if env.get("SGLANG_TRACE_ASYNC") != "1":
            raise SystemExit(f"{name}: async tracing disabled")
        if option(argv, "--max-running-requests") != "256":
            raise SystemExit(f"{name}: max-running drift")
        if role == "prefill" and "--enable-hierarchical-cache" not in argv:
            raise SystemExit("prefill HiCache disabled")
        if role == "decode" and "--enable-hierarchical-cache" in argv:
            raise SystemExit("decode HiCache unexpectedly enabled")
        result["workers"][name] = {
            "node": node,
            "role": role,
            "image": container.get("Image"),
            "trace_modules": option(argv, "--trace-modules"),
            "trace_async": env.get("SGLANG_TRACE_ASYNC"),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
