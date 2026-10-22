#!/usr/bin/env python3
"""Purpose: Wait for remote container health targets and capture failure evidence.
Usage:
    python3 tools/wait_healthy.py --target NAME NODE CONTAINER URL [--target ...]
    --failure-dir DIR --summary FILE [timeout options]
Artifacts:
    Health summary JSON; failed targets also get inspect and container-log files.
Artifact paths:
    --summary and FAILURE_DIR/<target>/{summary.json,inspect.json,container.log}.

Wait for remote containers to become healthy and save failure evidence.

Usage: wait_healthy.py --target NAME NODE CONTAINER URL [--target ...]
"""

import argparse
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def remote(options: list[str], target: list[str], command: list[str], timeout: float):
    return subprocess.run(
        ["ssh", *options, target[1], shlex.join(command)],
        check=False, capture_output=True, text=True, errors="replace", timeout=timeout,
    )


def wait_one(
    options: list[str], target: list[str], timeout: float, interval: float,
    probe_timeout: float, failure_dir: Path,
) -> dict[str, object]:
    name, node, container, url = target
    started = time.monotonic()
    reason = ""
    last_health_error = ""
    inspect_payload = None
    inspect_timeouts = 0
    while time.monotonic() - started < timeout:
        try:
            inspected = remote(
                options, target, ["docker", "inspect", container], probe_timeout + 10
            )
        except subprocess.TimeoutExpired:
            # TRANSIENT, not fatal. The docker daemon is heavily loaded during a
            # bring-up -- four ranks each mmap and load ~31 GiB of weights -- and a
            # single `docker inspect` can exceed its budget while the container is
            # perfectly healthy. Treating that as fatal killed round r003-c64 at
            # 31.8 s into a gate with minutes of budget left, and the container it
            # abandoned went on to serve normally.
            #
            # The health probe below has always treated its own timeout as transient
            # (see `last_health_error`); this path was simply inconsistent with it.
            # A container that is genuinely gone is caught by the `returncode` branch
            # ("No such object") or by the not-Running branch, both of which are still
            # fatal. If the condition persists, the enclosing `while` runs out and we
            # report a health timeout, which is the honest description.
            inspect_timeouts += 1
            last_health_error = (
                f"docker inspect timed out {inspect_timeouts}x (transient; retrying)"
            )
            time.sleep(interval)
            continue
        if inspected.returncode:
            reason = (inspected.stderr or inspected.stdout).strip()
            break
        try:
            inspect_payload = json.loads(inspected.stdout)[0]
        except (ValueError, IndexError, TypeError) as exc:
            reason = f"invalid docker inspect output: {exc}"
            break
        state = inspect_payload.get("State") or {}
        if not state.get("Running"):
            reason = (
                f"container stopped: status={state.get('Status')} "
                f"exit={state.get('ExitCode')} oom={state.get('OOMKilled')}"
            )
            break
        try:
            health = remote(
                options, target,
                ["docker", "exec", container, "curl", "--fail", "--silent",
                 "--show-error", "--max-time", str(probe_timeout), url],
                probe_timeout + 10,
            )
            if health.returncode == 0:
                return {
                    "name": name, "node": node, "container": container,
                    "health_url": url, "healthy": True,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            last_health_error = (health.stderr or health.stdout).strip()
        except subprocess.TimeoutExpired:
            last_health_error = "health request timed out"
        time.sleep(interval)
    else:
        reason = f"health timeout after {timeout:.0f}s"

    output = failure_dir / name
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": name, "node": node, "container": container, "health_url": url,
        "healthy": False, "reason": reason,
        "last_health_error": last_health_error,
        "inspect_timeouts": inspect_timeouts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if inspect_payload is not None:
        (output / "inspect.json").write_text(
            json.dumps(inspect_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    try:
        logs = remote(
            options, target, ["docker", "logs", "--timestamps", container],
            max(probe_timeout + 10, 30),
        )
        text = logs.stdout + logs.stderr
    except subprocess.TimeoutExpired as exc:
        text = f"docker logs timed out: {exc}\n"
    (output / "container.log").write_text(text[-4_000_000:], encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", nargs=4, action="append", required=True)
    parser.add_argument("--ssh-options", default="-o BatchMode=yes -o ConnectTimeout=10")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--probe-timeout", type=float, default=5)
    parser.add_argument("--failure-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    if min(args.timeout, args.interval, args.probe_timeout) <= 0:
        raise SystemExit("timeouts and interval must be positive")
    if len({target[0] for target in args.target}) != len(args.target):
        raise SystemExit("target names must be unique")
    args.failure_dir.mkdir(parents=True, exist_ok=True)
    options = shlex.split(args.ssh_options)
    results = []
    with ThreadPoolExecutor(max_workers=len(args.target)) as pool:
        futures = [
            pool.submit(
                wait_one, options, target, args.timeout, args.interval,
                args.probe_timeout, args.failure_dir,
            )
            for target in args.target
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            state = "healthy" if result["healthy"] else f"FAILED: {result['reason']}"
            print(f"{result['name']} on {result['node']}: {state}", flush=True)
    results.sort(key=lambda item: str(item["name"]))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps({"targets": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if all(result["healthy"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
