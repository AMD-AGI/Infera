# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Collect observed environment facts without importing accelerator frameworks."""

from __future__ import annotations

import csv
import importlib.metadata
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _probe(argv: list[str]) -> dict:
    """Retain a bounded diagnostic when a read-only command is unavailable."""
    result = {"command": argv, "stdout": "", "stderr": "", "returncode": None}
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        return result
    result.update(
        stdout=completed.stdout[:65536],
        stderr=completed.stderr[:4096],
        returncode=completed.returncode,
    )
    if len(completed.stdout) > 65536:
        result["error"] = "Probe output truncated at 65536 characters"
    return result


def _output(probe: dict) -> str:
    return probe["stdout"].strip() if probe["returncode"] == 0 and "error" not in probe else ""


def _remote_url(value: str) -> str | None:
    # An authenticated HTTPS remote may embed credentials; they are not provenance.
    if not value or value.startswith(("/", ".", "file:")):
        return None
    if "://" not in value:
        match = re.fullmatch(r"(?:[^/@\s]+@)?([^/:\s]+):(.+)", value)
        if not match:
            return None
        value = f"ssh://{match[1]}/{match[2]}"
    if "://" in value:
        scheme, rest = value.split("://", 1)
        authority, separator, path = rest.partition("/")
        value = scheme + "://" + authority.rsplit("@", 1)[-1] + separator + path
    return value.split("?", 1)[0].split("#", 1)[0]


def capture_environment(name: str, repos=(), image=None, image_digest=None) -> dict:
    """Return a portable environment note; unavailable facts stay explicitly missing.

    Run this inside the environment under test. Image identity is supplied by the
    caller because a process cannot reliably discover its own container digest.
    Dirty repository patches must be retained separately by the caller.
    """
    if not name.strip():
        raise ValueError("Environment name must not be empty")
    missing = []
    packages = {}
    for distribution in importlib.metadata.distributions():
        package_name = distribution.metadata.get("Name")
        if package_name:
            package_name = re.sub(r"[-_.]+", "-", package_name).lower()
            version = distribution.version or None
            if package_name in packages and packages[package_name] != version:
                missing.append(f"packages.{package_name}: conflicting installed versions")
            else:
                packages[package_name] = version
            if version is None:
                missing.append(f"packages.{package_name}: version unavailable")
    if not packages:
        missing.append("packages: no installed distribution metadata found")

    probes = {
        "rocminfo": _probe(["rocminfo"]),
        "rocm_driver": _probe(["rocm-smi", "--showdriverversion"]),
        "nvidia": _probe(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    hardware: dict[str, str | int | None] = dict.fromkeys(
        ("device", "architecture", "count", "driver")
    )
    rocm = _output(probes["rocminfo"])
    architectures = re.findall(r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", rocm, re.MULTILINE)
    if architectures:
        marketing = []
        for section in re.split(r"(?m)^[ \t]*Name:[ \t]*", rocm)[1:]:
            if re.fullmatch(r"gfx[0-9a-z]+", section.splitlines()[0].strip()):
                marketing.extend(re.findall(r"(?m)^[ \t]*Marketing Name:[ \t]*([^\n]*)", section))
        hardware.update(
            device=", ".join(sorted(set(x.strip() for x in marketing if x.strip()))) or None,
            architecture=", ".join(sorted(set(architectures))),
            count=len(architectures),
        )
        driver = re.search(
            r"Driver version:\s*([^\n]+)", _output(probes["rocm_driver"]), re.IGNORECASE
        )
        hardware["driver"] = driver.group(1).strip() if driver else None
    nvidia_rows = list(csv.reader(_output(probes["nvidia"]).splitlines()))
    if nvidia_rows and all(len(row) == 3 for row in nvidia_rows):
        if architectures:
            missing.append("hardware: mixed AMD/NVIDIA inventory needs manual description")
            hardware = dict.fromkeys(hardware)
        else:
            for index, key in enumerate(("device", "architecture", "driver")):
                values = {row[index].strip() for row in nvidia_rows}
                if values and not any(value in {"", "N/A", "[N/A]"} for value in values):
                    hardware[key] = ", ".join(sorted(values))
            hardware["count"] = len(nvidia_rows)
    missing.extend(
        f"hardware.{key}: not observed" for key, value in hardware.items() if value is None
    )

    repositories = []
    for index, source in enumerate(repos, 1):
        label = f"source-{index}"
        path = Path(source).resolve()
        prefix = ["git", "--no-optional-locks", "-C", str(path)]
        commit = _output(_probe([*prefix, "rev-parse", "HEAD"]))
        commit = commit if re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", commit) else None
        status = _probe([*prefix, "status", "--porcelain=v1", "--untracked-files=all"])
        dirty = bool(status["stdout"]) if status["returncode"] == 0 else None
        remote = _remote_url(_output(_probe([*prefix, "remote", "get-url", "origin"])))
        repositories.append(
            {
                "name": label,
                "url": remote,
                "commit": commit,
                "dirty": dirty,
                "patch": None,
            }
        )
        if not remote:
            missing.append(f"repositories.{label}: upstream URL unavailable")
        if commit is None:
            missing.append(f"repositories.{label}: commit unavailable")
        if dirty is not False:
            missing.append(f"repositories.{label}: clean state or retained dirty patch required")
    if not repositories:
        missing.append("repositories: source checkouts were not supplied")
    if image is None:
        missing.append("container.image: not supplied")
    if image_digest is None:
        missing.append("container.digest: not supplied")

    try:
        os_name = platform.freedesktop_os_release().get("PRETTY_NAME") or platform.system()
    except OSError:
        os_name = platform.system()
    return {
        "schema_version": 1,
        "kind": "environment",
        "name": name,
        "description": "Observed process environment; missing facts require separate evidence.",
        "hardware": hardware,
        "system": {
            "os": os_name,
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "container": {"image": image, "digest": image_digest},
        "packages": dict(sorted(packages.items())),
        "repositories": repositories,
        "evidence": [],
        "missing": sorted(set(missing)),
        "provenance": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "package_inventory": "importlib.metadata of the current Python environment",
            "container_identity": "caller-provided; not verified by the capture process",
            "accelerator_probes": probes,
        },
    }
