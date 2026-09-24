#!/usr/bin/env python3
"""Sample GPU, host-memory, and per-rail counters from remote nodes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


REMOTE = r"""
import glob
import json
import os
import pathlib
import subprocess
import time

def read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None

def number(value):
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def command(argv):
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
        result = {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        try:
            result["json"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
        return result
    except Exception as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}

result = {
    "remote_time_ns": time.time_ns(),
    "loadavg": read("/proc/loadavg"),
    "meminfo": {},
    "gpu": command(
        ["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--json"]
    ),
    "hcas": {},
}
for line in (read("/proc/meminfo") or "").splitlines():
    if ":" not in line:
        continue
    name, value = line.split(":", 1)
    if name in {
        "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached",
        "SwapTotal", "SwapFree", "Slab", "SReclaimable",
    }:
        result["meminfo"][name] = value.strip()

for ib_path in sorted(glob.glob("/sys/class/infiniband/ionic_*")):
    ib = os.path.basename(ib_path)
    port = os.path.join(ib_path, "ports", "1")
    device = os.path.realpath(os.path.join(ib_path, "device"))
    netdevs = sorted(os.listdir(os.path.join(device, "net")))
    record = {
        "bdf": os.path.basename(device),
        "numa_node": number(read(os.path.join(device, "numa_node"))),
        "state": read(os.path.join(port, "state")),
        "counters": {},
        "hw_counters": {},
        "netdevs": {},
    }
    for directory, key in (
        (os.path.join(port, "counters"), "counters"),
        (os.path.join(port, "hw_counters"), "hw_counters"),
    ):
        for counter in sorted(glob.glob(os.path.join(directory, "*"))):
            record[key][os.path.basename(counter)] = number(read(counter))
    for netdev in netdevs:
        stats = {}
        for name in (
            "rx_bytes", "tx_bytes", "rx_packets", "tx_packets",
            "rx_dropped", "tx_dropped", "rx_errors", "tx_errors",
        ):
            stats[name] = number(
                read(f"/sys/class/net/{netdev}/statistics/{name}")
            )
        record["netdevs"][netdev] = stats
    result["hcas"][ib] = record

print(json.dumps(result, sort_keys=True))
"""
running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_node(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("node must be ROLE=HOST")
    role, host = (part.strip() for part in value.split("=", 1))
    if not role or not host:
        raise argparse.ArgumentTypeError("node must be ROLE=HOST")
    return role, host


def sample(host: str, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ssh", "-F", "/dev/null", "-o", "UserKnownHostsFile=/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924/config/known_hosts",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(timeout))}",
            host,
            shlex.join(["python3", "-c", REMOTE]),
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node",
        action="append",
        type=parse_node,
        required=True,
        help="repeatable ROLE=HOST",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds; zero samples until interrupted",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0 or args.duration < 0:
        parser.error("interval/timeout must be positive and duration non-negative")
    nodes = dict(args.node)
    if len(nodes) != len(args.node):
        parser.error("node roles must be unique")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    header = {
        "record_type": "metadata",
        "schema_version": 1,
        "started_at": utc_now(),
        "interval_s": args.interval,
        "timeout_s": args.timeout,
        "nodes": nodes,
    }

    with args.output.open("x", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        sequence = 0
        while running and (deadline is None or time.monotonic() < deadline):
            cycle_started = time.monotonic()
            captured_at = utc_now()
            for role, host in nodes.items():
                sample_started = time.monotonic()
                record: dict[str, Any] = {
                    "record_type": "sample",
                    "schema_version": 1,
                    "sequence": sequence,
                    "captured_at": captured_at,
                    "elapsed_s": cycle_started - started,
                    "role": role,
                    "host": host,
                }
                try:
                    record["sample"] = sample(host, args.timeout)
                except (
                    OSError,
                    subprocess.SubprocessError,
                    json.JSONDecodeError,
                ) as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["sample_ms"] = (
                    time.monotonic() - sample_started
                ) * 1000.0
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            sequence += 1
            if args.once:
                break
            delay = args.interval - (time.monotonic() - cycle_started)
            if delay > 0 and running:
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
