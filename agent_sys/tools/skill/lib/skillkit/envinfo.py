"""Snapshot the machine and software a result came from.

Every probe is optional and reports `unavailable: <why>` rather than guessing:
a reproduction kit that says what it could not capture is more useful than one
that quietly omits it. Values only — no secret is read (environment variables
are listed by NAME for the ones that matter, never by value).
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 20) -> str:
    if not shutil.which(cmd[0]):
        return f"unavailable: {cmd[0]} not on PATH"
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    out = (p.stdout or p.stderr).strip()
    return out if p.returncode == 0 else f"unavailable: exit {p.returncode}: {out[:300]}"


def capture(repo: str | None = None, *, verbose: bool = False) -> dict:
    info: dict = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "in_container": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
    }
    try:
        mem = Path("/proc/meminfo").read_text().splitlines()[0]
        info["memory"] = mem.split(":", 1)[1].strip()
    except OSError:
        info["memory"] = "unavailable"
    info["captured_note"] = "capture time, not necessarily when the work ran — state the gap"
    if shutil.which("rocm-smi") or shutil.which("amd-smi"):
        info["gpu"] = (
            _run(["rocm-smi", "--showproductname"])
            if shutil.which("rocm-smi")
            else _run(["amd-smi", "list"])
        )
        info["rocm_version"] = _first_line(Path("/opt/rocm/.info/version"))
    elif shutil.which("nvidia-smi"):
        info["gpu"] = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
        )
    else:
        info["gpu"] = "unavailable: no rocm-smi/amd-smi/nvidia-smi"
    if repo:
        status = _run(["git", "status", "--porcelain"], cwd=repo)
        info["git"] = {
            "repo": repo,
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo),
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo),
            "uncommitted_paths": status
            if status.startswith("unavailable")
            else len(status.splitlines()),
        }
    if not verbose and isinstance(info.get("gpu"), str):
        info["gpu"] = _gpu_summary(info["gpu"])
    names = sorted(
        k
        for k in os.environ
        if k.startswith(("HIP_", "ROCM_", "CUDA_", "HSA_", "NCCL_", "RCCL_", "SGLANG_", "VLLM_"))
    )
    info["relevant_env_var_names"] = names
    return info


def _first_line(p: Path) -> str:
    try:
        return p.read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return f"unavailable: {p}"


def render_md(info: dict) -> str:
    lines = [
        "# Environment",
        "",
        f"Captured {info['captured_at']} by agent_sys/tools/skill (skillkit.envinfo).",
        "",
    ]
    for k, v in info.items():
        if k == "captured_at":
            continue
        if isinstance(v, dict):
            lines.append(f"- **{k}**:")
            lines += [f"  - {kk}: `{vv}`" for kk, vv in v.items()]
        elif isinstance(v, str) and "\n" in v:
            lines += [f"- **{k}**:", "", "```", v, "```", ""]
        else:
            lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines) + "\n"


def _gpu_summary(raw: str) -> str:
    """`8x AMD Instinct MI300X (gfx942)` from rocm-smi or nvidia-smi csv output."""
    if raw.startswith("unavailable"):
        return raw
    series = re.findall(r"Card Series:\s*(.+)", raw)
    if not series:  # nvidia-smi --format=csv: a header, then "name, driver, memory"
        series = [ln.split(",")[0].strip() for ln in raw.splitlines()[1:] if "," in ln]
    if not series:
        return raw.splitlines()[0][:200]
    gfx = re.findall(r"GFX Version:\s*(\w+)", raw)
    tail = f" ({gfx[0]})" if gfx else ""
    counts: dict[str, int] = {}
    for name in series:
        counts[name.strip()] = counts.get(name.strip(), 0) + 1
    return ", ".join(f"{n}x {name}{tail}" for name, n in counts.items())
