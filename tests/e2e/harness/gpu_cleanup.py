###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reclaim stale ROCm processes after acquiring an exclusive GPU node.

This helper runs on the compute-node host, before the test container starts.
It deliberately signals only processes owned by the current uid.  An exclusive
SLURM allocation makes same-uid GPU users stale by definition, but it does not
make an unmanaged process owned by another user safe to kill.

``INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS=1`` adds one opt-in exception: a
``docker run`` container from inside a job sits outside the scheduler's cgroup
and outlives it, so the node returns to ``idle`` still holding every GPU and the
Spur hooks that would sweep it are pinned off here.  Sound only because the
allocation is exclusive -- nothing else can hold this node, so such a container
belongs to a job that already ended.  ``stop``, not ``rm``, so it can restart.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

GPU_DIRTY_MARKER = "INFERA_E2E_GPU_NODE_DIRTY"
GPU_DIRTY_NODE_PREFIX = f"{GPU_DIRTY_MARKER}_NODE="
_VRAM_BUSY_FRACTION = 0.05
_ROCM_SMI_TIMEOUT = 5
_NO_JSON_WARNING = "No JSON data to report"
# Sweeps below the dirty threshold: _VRAM_BUSY_FRACTION asks "is this node
# unusable", this asks "is anything still here at all".
_RECLAIM_VRAM_FRACTION = 0.01
_DOCKER_TIMEOUT = 15
# SIGTERM then SIGKILL: a rushed kill of a large allocation can wedge the GPU.
_DOCKER_STOP_TIMEOUT = 30
# Kubernetes pods, and our own containers -- the disagg tier preflights one node
# while the pair's other node may already be serving this very run.
_PROTECTED_CONTAINER_NAME_PREFIXES = ("k8s_", "infera-e2e-", "infera-utest-")
_PROTECTED_CONTAINER_LABEL_PREFIXES = (
    "io.kubernetes.",
    "annotation.io.kubernetes.",
    "infera.e2e.job_tag",
)
_SYSTEM_PROCESS_NAMES = frozenset(
    {
        "amd-smi",
        "containerd",
        "dockerd",
        "gpuagent",
        "rocm-smi",
        "rsmi",
        "slurmd",
        "slurmstepd",
        "sshd",
        "systemd",
    }
)


class GpuCleanupError(RuntimeError):
    """The exclusive node still has GPU state this job cannot safely reclaim."""


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    uids: tuple[int, int, int, int]
    name: str
    cmdline: str
    start_time: str
    gpus: tuple[int, ...] = ()

    @property
    def uid(self) -> int:
        return self.uids[0]


def dirty_message(message: str, *, node: str | None = None) -> str:
    node = node or os.environ.get("INFERA_E2E_SLURM_NODE")
    node = node or socket.gethostname().split(".", 1)[0]
    return f"{GPU_DIRTY_NODE_PREFIX}{node} {message}"


def _rocm_smi_json(*args: str) -> object | None:
    try:
        done = subprocess.run(
            ["rocm-smi", *args, "--json"],
            capture_output=True,
            text=True,
            timeout=_ROCM_SMI_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    # rocm-smi exits 0 and writes no JSON when a query has nothing to report:
    # `--showpids` on a node no process holds does exactly that. Empty, not unreadable.
    stdout = done.stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        # Some builds put that warning on stdout rather than stderr.
        return {} if _NO_JSON_WARNING in stdout else None


def _gpu_process_names() -> dict[int, str] | None:
    """Return host GPU PIDs and names, or ``None`` when rocm-smi is unreadable."""
    data = _rocm_smi_json("--showpids")
    if not isinstance(data, dict):
        return None

    processes: dict[int, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                match = re.fullmatch(r"PID(\d+)", str(key))
                if match:
                    processes[int(match.group(1))] = str(child).split(",", 1)[0].strip()
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return processes


def _process_gpu_map(pids: list[int]) -> dict[int, tuple[int, ...]]:
    """Best-effort PID-to-DRM-card mapping.

    ROCm SMI 4.0 emits no JSON for ``--showpidgpus``, so this one command uses
    its stable human-readable ``PID N ...:\\n0 1`` form.
    """
    if not pids:
        return {}
    try:
        done = subprocess.run(
            ["rocm-smi", "--showpidgpus", *map(str, pids)],
            capture_output=True,
            text=True,
            timeout=_ROCM_SMI_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if done.returncode != 0:
        return {}
    result: dict[int, tuple[int, ...]] = {}
    for pid in pids:
        match = re.search(
            rf"PID\s+{pid}\s+is using \d+ DRM device\(s\)(?::\s*([0-9 ]+))?",
            done.stdout,
        )
        result[pid] = (
            tuple(int(gpu) for gpu in match.group(1).split()) if match and match.group(1) else ()
        )
    return result


def _wait_for_process_names(timeout: float) -> dict[int, str] | None:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        processes = _gpu_process_names()
        if processes is not None or time.monotonic() >= deadline:
            return processes
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _gpu_vram() -> dict[int, tuple[int, int]] | None:
    """Return ``{gpu: (used, total)}``, or ``None`` when it cannot be measured."""
    data = _rocm_smi_json("--showmeminfo", "vram")
    if not isinstance(data, dict):
        return None
    vram: dict[int, tuple[int, int]] = {}
    for card, info in data.items():
        if not str(card).startswith("card") or not isinstance(info, dict):
            continue
        try:
            index = int(str(card)[len("card") :])
            used = int(info["VRAM Total Used Memory (B)"])
            total = int(info["VRAM Total Memory (B)"])
        except (KeyError, TypeError, ValueError):
            continue
        if total > 0:
            vram[index] = (used, total)
    return vram or None


def _busy_gpus(
    vram: dict[int, tuple[int, int]] | None, fraction: float = _VRAM_BUSY_FRACTION
) -> dict[int, tuple[int, int]]:
    if not vram:
        return {}
    return {gpu: values for gpu, values in vram.items() if values[0] >= fraction * values[1]}


def _reclaim_containers_enabled() -> bool:
    return os.environ.get("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS") == "1"


def _reclaim_vram_fraction() -> float:
    return float(os.environ.get("INFERA_E2E_RECLAIM_VRAM_FRACTION", str(_RECLAIM_VRAM_FRACTION)))


def _docker(*args: str, timeout: int = _DOCKER_TIMEOUT) -> str | None:
    """Run a docker command, returning stdout, or ``None`` when it cannot run."""
    try:
        done = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def _container_is_protected(name: str, labels: str) -> bool:
    if name.startswith(_PROTECTED_CONTAINER_NAME_PREFIXES):
        return True
    if any(prefix in labels for prefix in _PROTECTED_CONTAINER_LABEL_PREFIXES):
        return True
    # Escape hatch for whatever this fleet grows next, without a code change.
    keep = os.environ.get("INFERA_E2E_RECLAIM_KEEP", "")
    return any(token and token in name for token in (part.strip() for part in keep.split(",")))


def _reclaimable_containers() -> list[tuple[str, str]] | None:
    """Running containers this job may stop, as ``[(name, image)]``.

    ``None`` means docker could not be queried -- not that the node is empty.
    """
    listing = _docker("ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Labels}}")
    if listing is None:
        return None
    containers: list[tuple[str, str]] = []
    for line in listing.splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0]:
            continue
        name, image = fields[0], fields[1]
        labels = fields[2] if len(fields) > 2 else ""
        if _container_is_protected(name, labels):
            print(f"[gpu-cleanup] protected container left running: {name}", file=sys.stderr)
            continue
        containers.append((name, image))
    return containers


def _stop_foreign_containers() -> list[str]:
    """``docker stop`` every reclaimable container. Returns the names stopped.

    Best-effort: the VRAM re-check, not this sweep, is the verdict.
    """
    containers = _reclaimable_containers()
    if containers is None:
        print(
            "[gpu-cleanup] docker could not be queried — skipping container reclaim",
            file=sys.stderr,
            flush=True,
        )
        return []
    stopped: list[str] = []
    for name, image in containers:
        print(f"[gpu-cleanup] stopping foreign container {name} (image {image})", file=sys.stderr)
        if (
            _docker(
                "stop",
                "--time",
                str(_DOCKER_STOP_TIMEOUT),
                name,
                timeout=_DOCKER_STOP_TIMEOUT + _DOCKER_TIMEOUT,
            )
            is None
        ):
            print(f"[gpu-cleanup] could not stop {name}", file=sys.stderr, flush=True)
            continue
        stopped.append(name)
    return stopped


def _expected_gpu_ids() -> set[int]:
    count = int(os.environ.get("INFERA_E2E_EXPECTED_GPU_COUNT", "8"))
    if count <= 0:
        raise ValueError("INFERA_E2E_EXPECTED_GPU_COUNT must be positive")
    return set(range(count))


def _start_time(stat: str) -> str | None:
    close_paren = stat.rfind(")")
    fields = stat[close_paren + 1 :].split() if close_paren >= 0 else []
    return fields[19] if len(fields) > 19 else None


def _read_process(pid: int, reported_name: str = "") -> GpuProcess | None:
    proc = Path("/proc") / str(pid)
    try:
        stat_before = (proc / "stat").read_text()
        status = (proc / "status").read_text()
    except OSError:
        return None
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        cmdline = ""
    try:
        stat_after = (proc / "stat").read_text()
    except OSError:
        return None

    uid_match = re.search(r"^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", status, re.MULTILINE)
    start_time = _start_time(stat_before)
    if not uid_match or not start_time or start_time != _start_time(stat_after):
        return None

    if not cmdline:
        comm_match = re.match(r"^\d+\s+\((.*)\)", stat_after)
        cmdline = comm_match.group(1) if comm_match else reported_name
    name = reported_name or os.path.basename(cmdline.split(maxsplit=1)[0])
    return GpuProcess(
        pid=pid,
        uids=tuple(map(int, uid_match.groups())),
        name=name,
        cmdline=cmdline.strip(),
        start_time=start_time,
    )


def _is_system_process(process: GpuProcess) -> bool:
    name = os.path.basename(process.name).lower()
    command = os.path.basename(process.cmdline.split(maxsplit=1)[0]).lower()
    return name in _SYSTEM_PROCESS_NAMES or command in _SYSTEM_PROCESS_NAMES


def _same_process(process: GpuProcess) -> bool:
    current = _read_process(process.pid, process.name)
    return bool(
        current and current.uids == process.uids and current.start_time == process.start_time
    )


def _remaining_targets(targets: list[GpuProcess]) -> list[GpuProcess]:
    gpu_pids = _gpu_process_names()
    if gpu_pids is None:
        return [process for process in targets if _same_process(process)]
    return [process for process in targets if process.pid in gpu_pids and _same_process(process)]


def _wait_for_targets(targets: list[GpuProcess], timeout: float) -> list[GpuProcess]:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        remaining = _remaining_targets(targets)
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _wait_for_vram(timeout: float) -> dict[int, tuple[int, int]] | None:
    deadline = time.monotonic() + max(0.0, timeout)
    last_busy: dict[int, tuple[int, int]] | None = None
    expected = _expected_gpu_ids()
    while True:
        vram = _gpu_vram()
        if vram is not None and expected.issubset(vram):
            busy = _busy_gpus(vram)
            if not busy:
                return {}
            last_busy = busy
        if time.monotonic() >= deadline:
            return last_busy
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _signal(processes: list[GpuProcess], sig: signal.Signals) -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise GpuCleanupError(
            dirty_message("this Python/kernel lacks race-safe pidfd process signalling")
        )
    opened: list[tuple[GpuProcess, int]] = []
    for process in processes:
        try:
            opened.append((process, os.pidfd_open(process.pid)))
        except ProcessLookupError:
            pass
    if not opened:
        return

    try:
        gpu_pids = _gpu_process_names()
        if gpu_pids is None:
            raise GpuCleanupError(
                dirty_message("rocm-smi process data became unavailable before signalling")
            )
        for process, pidfd in opened:
            if process.pid not in gpu_pids or not _same_process(process):
                continue
            try:
                signal.pidfd_send_signal(pidfd, sig)
                print(
                    f"[gpu-cleanup] {sig.name} {_process_detail(process)}",
                    file=sys.stderr,
                    flush=True,
                )
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(
                    f"[gpu-cleanup] could not signal pid={process.pid}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for _process, pidfd in opened:
            os.close(pidfd)


def _format_busy(busy: dict[int, tuple[int, int]]) -> str:
    return ", ".join(
        f"gpu{gpu}={used / (1024**3):.1f}/{total / (1024**3):.1f}GiB"
        for gpu, (used, total) in sorted(busy.items())
    )


def _gpu_label(process: GpuProcess) -> str:
    return ",".join(map(str, process.gpus)) if process.gpus else "unknown"


def _owned_by_ci(process: GpuProcess, current_uid: int) -> bool:
    return (
        current_uid != 0
        and all(uid == current_uid for uid in process.uids)
        and not _is_system_process(process)
    )


def _process_detail(process: GpuProcess, current_uid: int | None = None) -> str:
    owner = f"uid={process.uid} uids={'/'.join(map(str, process.uids))}"
    if current_uid is not None:
        if 0 in process.uids:
            owner += " class=root"
        elif _is_system_process(process):
            owner += " class=system"
        elif _owned_by_ci(process, current_uid):
            owner += " class=same-uid"
        else:
            owner += " class=foreign"
    return (
        f"gpus={_gpu_label(process)} pid={process.pid} {owner} "
        f"name={process.name} cmd={process.cmdline}"
    )


def _discover_processes(names: dict[int, str]) -> list[GpuProcess]:
    raw = [
        process
        for pid, name in names.items()
        if pid != os.getpid() and (process := _read_process(pid, name)) is not None
    ]
    gpu_map = _process_gpu_map([process.pid for process in raw])
    return [replace(process, gpus=gpu_map.get(process.pid, ())) for process in raw]


def cleanup_exclusive_gpu_processes(
    *,
    term_timeout: float = 25.0,
    kill_timeout: float = 10.0,
    vram_timeout: float = 30.0,
    max_rounds: int = 3,
) -> int:
    """Kill same-uid GPU processes and verify that the exclusive node is idle.

    Returns the number of processes targeted.  Raises :class:`GpuCleanupError`
    when meaningful VRAM remains occupied after all safe cleanup actions.
    """
    if os.environ.get("INFERA_E2E_EXCLUSIVE") != "1":
        return 0
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    current_uid = os.getuid()
    targeted: dict[tuple[int, str], GpuProcess] = {}
    current_processes: list[GpuProcess] = []

    for cleanup_round in range(max_rounds):
        names = (
            _wait_for_process_names(kill_timeout) if cleanup_round == 0 else _gpu_process_names()
        )
        if names is None:
            raise GpuCleanupError(
                dirty_message("rocm-smi process data is unavailable; cleanup cannot be verified")
            )
        current_processes = _discover_processes(names)
        targets = [process for process in current_processes if _owned_by_ci(process, current_uid)]
        if not targets:
            break
        for process in targets:
            targeted[(process.pid, process.start_time)] = process
        _signal(targets, signal.SIGTERM)
        remaining = _wait_for_targets(targets, term_timeout)
        if remaining:
            _signal(remaining, signal.SIGKILL)
            remaining = _wait_for_targets(remaining, kill_timeout)
        if remaining:
            detail = "; ".join(_process_detail(process, current_uid) for process in remaining)
            raise GpuCleanupError(
                dirty_message(f"same-uid GPU processes survived cleanup: {detail}")
            )

    names = _gpu_process_names()
    if names is None:
        raise GpuCleanupError(dirty_message("rocm-smi process data is unavailable after cleanup"))
    current_processes = _discover_processes(names)
    targets = [process for process in current_processes if _owned_by_ci(process, current_uid)]
    if targets:
        detail = "; ".join(_process_detail(process, current_uid) for process in targets)
        raise GpuCleanupError(
            dirty_message(
                f"same-uid GPU processes kept appearing after {max_rounds} cleanup rounds: {detail}"
            )
        )

    # VRAM still held after a same-uid-only sweep belongs to a container no
    # signal of ours reaches. Stop those; the re-check below is the verdict.
    if _reclaim_containers_enabled():
        occupied = _busy_gpus(_gpu_vram(), _reclaim_vram_fraction())
        if occupied:
            print(
                f"[gpu-cleanup] VRAM still held after same-uid cleanup "
                f"({_format_busy(occupied)}) — reclaiming foreign containers",
                file=sys.stderr,
                flush=True,
            )
            stopped = _stop_foreign_containers()
            print(
                f"[gpu-cleanup] stopped {len(stopped)} foreign container(s)"
                + (f": {', '.join(stopped)}" if stopped else ""),
                file=sys.stderr,
                flush=True,
            )

    busy = _wait_for_vram(vram_timeout)
    if busy is None:
        raise GpuCleanupError(
            dirty_message("rocm-smi VRAM data is unavailable; cleanup cannot be verified")
        )
    if busy:
        current_names = _gpu_process_names()
        current_processes = _discover_processes(current_names) if current_names is not None else []
        owners = (
            "; ".join(_process_detail(process, current_uid) for process in current_processes)
            or "no owning PID reported"
        )
        raise GpuCleanupError(
            dirty_message(f"VRAM remains busy after safe cleanup ({_format_busy(busy)}); {owners}")
        )

    if targeted:
        print(
            f"[gpu-cleanup] reclaimed {len(targeted)} stale same-uid GPU process(es)",
            file=sys.stderr,
            flush=True,
        )
    elif current_processes:
        print(
            "[gpu-cleanup] protected GPU processes are below the busy threshold; "
            "leaving them intact",
            file=sys.stderr,
            flush=True,
        )
    return len(targeted)


def main() -> int:
    try:
        cleanup_exclusive_gpu_processes(
            term_timeout=float(os.environ.get("INFERA_E2E_GPU_TERM_TIMEOUT", "25")),
            kill_timeout=float(os.environ.get("INFERA_E2E_GPU_KILL_TIMEOUT", "10")),
            vram_timeout=float(os.environ.get("INFERA_E2E_GPU_VRAM_TIMEOUT", "30")),
            max_rounds=int(os.environ.get("INFERA_E2E_GPU_CLEANUP_ROUNDS", "3")),
        )
    except GpuCleanupError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 75
    except ValueError as exc:
        print(dirty_message(f"invalid GPU cleanup setting: {exc}"), file=sys.stderr, flush=True)
        return 75
    except Exception as exc:
        print(dirty_message(f"unexpected GPU cleanup failure: {exc}"), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
