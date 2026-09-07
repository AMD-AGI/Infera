###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regression tests for the CI runner's real SLURM submission ceiling."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN_TESTS = REPO / "tests" / "run_tests.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _runner_env(tmp_path: Path, mock_bin: Path, count_file: Path) -> dict[str, str]:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = os.environ.copy()
    for name in (
        "CI",
        "GITHUB_ACTIONS",
        "INFERA_E2E_RESERVATION",
        "INFERA_E2E_SITE",
        "INFERA_E2E_SLURM_ACCOUNT_QOS_PAIRS",
        "SLURM_JOB_ID",
        "SLURM_JOBID",
    ):
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "HOME": str(tmp_path),
            "TMPDIR": str(scratch),
            "COUNT_FILE": str(count_file),
            "INFERA_E2E_GFX_ARCH": "gfx950",
            "INFERA_E2E_LOCAL": "0",
            "INFERA_E2E_SLURM_PARTITION": "test",
            "INFERA_E2E_SLURM_TIME": "00:01:00",
            "INFERA_E2E_SLURM_MAX_ATTEMPTS": "5",
        }
    )
    return env


def _run_runner(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUN_TESTS), *args],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_srun_resubmissions_stop_after_five(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    count_file = tmp_path / "srun-count"
    count_file.write_text("0\n")
    _executable(mock_bin / "sleep", "exit 0\n")
    _executable(
        mock_bin / "srun",
        """
n=$(cat "$COUNT_FILE")
echo $((n + 1)) > "$COUNT_FILE"
echo "srun: error: service is currently unavailable" >&2
exit 1
""",
    )

    result = _run_runner(_runner_env(tmp_path, mock_bin, count_file), "engine")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "5"
    assert "SLURM submission limit reached (5 attempts)" in result.stderr


def test_disagg_hold_sbatch_submissions_stop_after_five(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    count_file = tmp_path / "sbatch-count"
    count_file.write_text("0\n")
    _executable(mock_bin / "sleep", "exit 0\n")
    _executable(mock_bin / "srun", "exit 1\n")
    _executable(mock_bin / "squeue", "exit 0\n")
    _executable(mock_bin / "sinfo", "printf 'node-a\\nnode-b\\nnode-c\\nnode-d\\n'\n")
    _executable(
        mock_bin / "scontrol",
        """
if [ "$1 $2" = "show node" ]; then
  echo "NodeName=$3 State=IDLE CPUAlloc=0 AllocMem=0 AllocTRES="
fi
exit 0
""",
    )
    _executable(
        mock_bin / "sbatch",
        """
n=$(cat "$COUNT_FILE")
echo $((n + 1)) > "$COUNT_FILE"
echo "sbatch: submission temporarily unavailable" >&2
exit 1
""",
    )

    result = _run_runner(_runner_env(tmp_path, mock_bin, count_file), "e2e", "sglang", "disag")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "5"
    assert "SLURM hold submission limit reached (5 attempts)" in result.stderr
