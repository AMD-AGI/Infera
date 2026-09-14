#!/usr/bin/env python3
"""Wait for explicit remote container health targets."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Target:
    name: str
    node: str
    container: str
    health_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "NODE", "CONTAINER", "HEALTH_URL"),
    )
    parser.add_argument("--ssh-options", default="-o BatchMode=yes -o ConnectTimeout=10")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--probe-timeout", type=float, default=5)
    parser.add_argument("--failure-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def remote(
    ssh_options: list[str], target: Target, command: list[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *ssh_options, target.node, shlex.join(command)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def inspect(
    ssh_options: list[str], target: Target, timeout: float
) -> tuple[dict[str, Any] | None, str]:
    try:
        result = remote(
            ssh_options, target, ["docker", "inspect", target.container], timeout
        )
    except subprocess.TimeoutExpired:
        return None, "docker inspect timed out"
    if result.returncode:
        return None, (result.stderr or result.stdout).strip()
    try:
        return json.loads(result.stdout)[0], ""
    except (ValueError, IndexError, TypeError) as exc:
        return None, f"invalid docker inspect output: {exc}"


def save_failure(
    ssh_options: list[str],
    target: Target,
    directory: Path,
    reason: str,
    container_info: dict[str, Any] | None,
    health_error: str,
    timeout: float,
) -> None:
    output = directory / target.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                **asdict(target),
                "reason": reason,
                "last_health_error": health_error,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if container_info is not None:
        (output / "inspect.json").write_text(
            json.dumps(container_info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    try:
        result = remote(
            ssh_options,
            target,
            ["docker", "logs", "--timestamps", target.container],
            max(timeout, 30),
        )
        logs = result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        logs = f"docker logs timed out: {exc}\n"
    (output / "container.log").write_text(logs[-4_000_000:], encoding="utf-8")


def wait_one(
    ssh_options: list[str],
    target: Target,
    timeout: float,
    interval: float,
    probe_timeout: float,
    failure_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    last_health_error = ""
    container_info: dict[str, Any] | None = None
    reason = ""
    while time.monotonic() - started < timeout:
        container_info, inspect_error = inspect(
            ssh_options, target, probe_timeout + 10
        )
        if container_info is None:
            reason = inspect_error or "docker inspect failed"
            break
        state = container_info.get("State") or {}
        if not state.get("Running"):
            reason = (
                f"container stopped: status={state.get('Status')} "
                f"exit={state.get('ExitCode')} oom={state.get('OOMKilled')}"
            )
            break
        try:
            probe = remote(
                ssh_options,
                target,
                [
                    "docker",
                    "exec",
                    target.container,
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    str(probe_timeout),
                    target.health_url,
                ],
                probe_timeout + 10,
            )
            if probe.returncode == 0:
                return {
                    **asdict(target),
                    "healthy": True,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            last_health_error = (probe.stderr or probe.stdout).strip()
        except subprocess.TimeoutExpired:
            last_health_error = "health request timed out"
        time.sleep(interval)
    else:
        reason = f"health timeout after {timeout:.0f}s"

    save_failure(
        ssh_options,
        target,
        failure_dir,
        reason,
        container_info,
        last_health_error,
        probe_timeout + 10,
    )
    return {
        **asdict(target),
        "healthy": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "reason": reason,
        "last_health_error": last_health_error,
    }


def main() -> int:
    args = parse_args()
    if min(args.timeout, args.interval, args.probe_timeout) <= 0:
        raise SystemExit("timeouts and interval must be positive")
    targets = [Target(*values) for values in args.target]
    if len({target.name for target in targets}) != len(targets):
        raise SystemExit("target names must be unique")
    args.failure_dir.mkdir(parents=True, exist_ok=True)
    ssh_options = shlex.split(args.ssh_options)
    results = []
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = {
            executor.submit(
                wait_one,
                ssh_options,
                target,
                args.timeout,
                args.interval,
                args.probe_timeout,
                args.failure_dir,
            ): target
            for target in targets
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            state = "healthy" if result["healthy"] else f"FAILED: {result['reason']}"
            print(f"[wait] {result['name']} on {result['node']}: {state}", flush=True)
    results.sort(key=lambda item: item["name"])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps({"targets": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if all(result["healthy"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
