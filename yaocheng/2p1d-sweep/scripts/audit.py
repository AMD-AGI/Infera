#!/usr/bin/env python3
"""Capture health, container identity and cumulative faults for all three workers.

Usage: audit.py OUTPUT [--baseline deployment.json] [--before point-before.json]
Requires the exported config/topology environment from common.sh.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench-harness/tools"))
from topology import load

COUNT_LOGS = '''import json, sys
counts = {"rail": 0, "memory_fault": 0, "fatal_python": 0, "oom": 0}
for line in sys.stdin:
    text = line.lower()
    counts["rail"] += int("transport retry counter exceeded" in text or "wqe is not posted" in text)
    counts["memory_fault"] += int("memory access fault" in text)
    counts["fatal_python"] += int("fatal python error" in text)
    counts["oom"] += int("out of memory" in text or "outofmemoryerror" in text)
print(json.dumps(counts))
'''


def ssh(node, argv):
    result = subprocess.run(
        ["ssh", "-n", *shlex.split(os.environ["SSH_OPTS"]), node, shlex.join(argv)],
        capture_output=True, text=True, timeout=180, check=True,
    )
    return result.stdout


def compare(current, baseline, before=None):
    """A restarted container is a new deployment, even if its name is unchanged."""
    if set(current["containers"]) != set(baseline["containers"]):
        raise ValueError("deployment container set changed")
    for name, item in current["containers"].items():
        old = baseline["containers"][name]
        if (item["id"], item["started_at"], item["image"]) != (
            old["id"], old["started_at"], old["image"]
        ):
            raise ValueError(f"{name}: container changed/restarted; use a new sweep")
    if before:
        for name, counts in current["faults"].items():
            previous = before["faults"][name]
            if counts != previous:
                raise ValueError(f"{name}: fault counters changed: {previous} -> {counts}")
    elif any(any(counts.values()) for counts in current["faults"].values()):
        raise ValueError("deployment already has rail/GPU/fatal/OOM faults")


def capture():
    env = os.environ
    rows = load(env["TOPOLOGY"])
    targets = [(r["instance"], r["node"],
                f'http://{r["data_ip"]}:{int(env["ENGINE_PORT_BASE"]) + int(r["index"])}')
               for r in rows]
    targets.append(("router", env["CONTROL_NODE"], f'http://{env["CONTROL_IP"]}:{env["ROUTER_PORT"]}'))
    result = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "containers": {}, "faults": {}}
    for instance, node, url in targets:
        with urllib.request.urlopen(url + "/health", timeout=15) as response:
            if response.status != 200:
                raise ValueError(f"{instance}: unhealthy HTTP {response.status}")
        name = f'{env["CONTAINER_PREFIX"]}-{instance}'
        container = json.loads(ssh(node, ["docker", "inspect", name]))[0]
        state = container["State"]
        if not state["Running"] or state.get("Restarting") or state.get("OOMKilled"):
            raise ValueError(f"{instance}: unhealthy container state: {state}")
        result["containers"][instance] = {
            "node": node, "id": container["Id"], "started_at": state["StartedAt"],
            "image": container["Image"], "url": url,
        }
        if instance == "router":
            variables = dict(item.split("=", 1) for item in container["Config"]["Env"] if "=" in item)
            if variables.get("INFERA_PD_DP_RANK_AFFINITY") != "true":
                raise ValueError("router must enable same-rank P/D affinity")
            continue
        pipeline = (shlex.join(["docker", "logs", name]) + " 2>&1 | "
                    + shlex.join(["python3", "-c", COUNT_LOGS]))
        result["faults"][instance] = json.loads(ssh(node, ["bash", "-o", "pipefail", "-c", pipeline]))
    # Include etcd identity so a registry restart cannot silently join two runs.
    container = json.loads(ssh(env["CONTROL_NODE"], ["docker", "inspect", f'{env["CONTAINER_PREFIX"]}-etcd']))[0]
    if not container["State"]["Running"]:
        raise ValueError("etcd is not running")
    result["containers"]["etcd"] = {"id": container["Id"], "image": container["Image"],
                                      "started_at": container["State"]["StartedAt"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--before", type=Path)
    args = parser.parse_args()
    try:
        current = capture()
        # Write evidence even when comparison finds a fault.
        args.output.write_text(json.dumps(current, indent=2) + "\n")
        baseline = json.loads(args.baseline.read_text()) if args.baseline else current
        before = json.loads(args.before.read_text()) if args.before else None
        compare(current, baseline, before)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"audit: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
