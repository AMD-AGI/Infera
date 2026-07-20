###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Small shared helpers for the probes. Everything here is best-effort: a missing
tool or a non-zero exit must never raise — probes degrade to a warn/info finding.
"""

from __future__ import annotations

import shutil
import subprocess


def have(cmd: str) -> bool:
    """True if ``cmd`` is on PATH."""
    return shutil.which(cmd) is not None


def run(cmd: list[str], timeout: float = 5.0, merge_stderr: bool = True) -> tuple[int | None, str]:
    """Run a command, capturing output.

    Returns ``(returncode, output)``. On missing binary or timeout the return
    code is ``None`` and the output carries a short reason. Set
    ``merge_stderr=False`` when parsing structured stdout (e.g. JSON) so tool
    warnings on stderr don't corrupt the payload.
    """
    if not have(cmd[0]):
        return None, f"<{cmd[0]} not found>"
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return None, f"<{cmd[0]} timed out after {timeout}s>"
    except Exception as e:  # noqa: BLE001 - best-effort probe
        return None, f"<{cmd[0]} failed: {e}>"


def read_text(path: str) -> str | None:
    """Read a file, returning None on any error (missing/permission)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:  # noqa: BLE001 - best-effort probe
        return None
