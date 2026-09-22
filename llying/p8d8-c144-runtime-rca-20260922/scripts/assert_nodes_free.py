#!/usr/bin/env python3
"""Fail unless both nodes are idle and safe for an exclusive P8D8 launch."""

from __future__ import annotations

import json
import shlex
import subprocess
from typing import Any


EXPECTED_IMAGE = (
    "sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35"
)
IMAGE = "infera-sglang:v0519-yihou-0917-nextnfix-hicache"
PREFIX = "glm52-pd-c144-runtime-rca"
NODES = (
    ("crsuse2-m2m-138", "10.245.157.237"),
    ("crsuse2-m2m-136", "10.245.154.168"),
)
REMOTE = r"""
import glob
import json
import pathlib
import subprocess

def command(argv):
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, errors="replace"
    ).stdout

gpu = json.loads(command(["rocm-smi", "--showmemuse", "--json"]))
ids = command(["docker", "ps", "-aq"]).split()
containers = json.loads(command(["docker", "inspect", *ids])) if ids else []
active_hcas = []
for path in sorted(glob.glob("/sys/class/infiniband/ionic_*")):
    state = pathlib.Path(path, "ports", "1", "state").read_text().strip()
    if state.endswith("ACTIVE"):
        active_hcas.append(path.rsplit("/", 1)[-1])
print(json.dumps({
    "gpu": gpu,
    "containers": containers,
    "active_hcas": active_hcas,
}, sort_keys=True))
"""


def ssh(node: str, *command: str) -> str:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            node,
            shlex.join(command),
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    ).stdout


def user_gpu_containers(containers: list[dict[str, Any]]) -> list[str]:
    result = []
    for container in containers:
        name = str(container.get("Name", "")).lstrip("/")
        config = container.get("Config") or {}
        host = container.get("HostConfig") or {}
        devices = host.get("Devices") or []
        image = str(config.get("Image", ""))
        running = bool((container.get("State") or {}).get("Running"))
        has_kfd = any(
            item.get("PathOnHost") == "/dev/kfd"
            for item in devices
            if isinstance(item, dict)
        )
        system = name.startswith(("crusoe-vector", "crusoe-amd-"))
        if running and not system and (has_kfd or "sglang" in image.lower()):
            result.append(name)
    return sorted(result)


def main() -> int:
    summary: dict[str, Any] = {"nodes": {}}
    errors: list[str] = []
    for node, expected_ip in NODES:
        actual_image = ssh(
            node,
            "docker",
            "image",
            "inspect",
            IMAGE,
            "--format",
            "{{.Id}}",
        ).strip()
        if actual_image != EXPECTED_IMAGE:
            errors.append(f"{node}: image={actual_image}, expected={EXPECTED_IMAGE}")

        payload = json.loads(
            ssh(node, "python3", "-c", REMOTE)
        )
        busy_gpus = {
            card: values.get("GPU Memory Allocated (VRAM%)")
            for card, values in payload["gpu"].items()
            if str(values.get("GPU Memory Allocated (VRAM%)")) != "0"
        }
        if busy_gpus:
            errors.append(f"{node}: nonzero VRAM: {busy_gpus}")
        blockers = user_gpu_containers(payload["containers"])
        if blockers:
            errors.append(f"{node}: running GPU containers: {blockers}")
        name_conflicts = sorted(
            str(item.get("Name", "")).lstrip("/")
            for item in payload["containers"]
            if str(item.get("Name", "")).lstrip("/").startswith(PREFIX)
        )
        if name_conflicts:
            errors.append(f"{node}: stale experiment containers: {name_conflicts}")
        if len(payload["active_hcas"]) != 8:
            errors.append(
                f"{node}: active ionic HCAs={payload['active_hcas']}, expected 8"
            )
        addresses = ssh(node, "ip", "-o", "-4", "addr", "show")
        if expected_ip not in addresses:
            errors.append(f"{node}: missing data IP {expected_ip}")
        summary["nodes"][node] = {
            "image": actual_image,
            "busy_gpus": busy_gpus,
            "blocking_containers": blockers,
            "active_hcas": payload["active_hcas"],
            "data_ip": expected_ip,
        }

    summary["passed"] = not errors
    summary["errors"] = errors
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
