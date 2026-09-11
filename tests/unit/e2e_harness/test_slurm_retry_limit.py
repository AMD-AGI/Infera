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


def test_dirty_gpu_node_is_excluded_and_retried_within_limit(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    count_file = tmp_path / "srun-count"
    args_file = tmp_path / "srun-args"
    count_file.write_text("0\n")
    _executable(mock_bin / "sleep", "exit 0\n")
    _executable(
        mock_bin / "srun",
        """
n=$(cat "$COUNT_FILE")
n=$((n + 1))
echo "$n" > "$COUNT_FILE"
printf '%s\n' "$*" >> "$ARGS_FILE"
echo "INFERA_E2E_SLURM_NODE=node-$n"
echo "INFERA_E2E_GPU_NODE_DIRTY_NODE=node-$n mock foreign GPU owner" >&2
echo "Uvicorn running on wrong-node" >&2
exit 75
""",
    )
    env = _runner_env(tmp_path, mock_bin, count_file)
    env["ARGS_FILE"] = str(args_file)

    result = _run_runner(env, "engine")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "5"
    attempts = args_file.read_text().splitlines()
    assert "-x node-1" in attempts[1]
    assert "-x node-1,node-2" in attempts[2]
    assert "still have GPU owners after exclusive cleanup" in result.stderr
    assert "SLURM submission limit reached (5 attempts)" in result.stderr


def test_marker_text_inside_test_failure_is_not_retried(tmp_path):
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
echo "INFERA_E2E_SLURM_NODE=node-a"
echo "AssertionError: expected INFERA_E2E_GPU_NODE_DIRTY marker" >&2
exit 1
""",
    )

    result = _run_runner(_runner_env(tmp_path, mock_bin, count_file), "engine")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "1"
    assert "still have GPU owners after exclusive cleanup" not in result.stderr


def test_dirty_marker_in_shared_worker_log_excludes_exact_node(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    count_file = tmp_path / "srun-count"
    args_file = tmp_path / "srun-args"
    count_file.write_text("0\n")
    _executable(mock_bin / "sleep", "exit 0\n")
    _executable(
        mock_bin / "srun",
        """
n=$(cat "$COUNT_FILE")
n=$((n + 1))
echo "$n" > "$COUNT_FILE"
printf '%s\n' "$*" >> "$ARGS_FILE"
for arg in "$@"; do
  case "$arg" in
    */dispatch-engine-*.log)
      echo "INFERA_E2E_GPU_NODE_DIRTY_NODE=node-$n mock owner" > "$arg"
      ;;
  esac
done
exit 75
""",
    )
    env = _runner_env(tmp_path, mock_bin, count_file)
    env.update({"ARGS_FILE": str(args_file), "CI": "true"})

    result = _run_runner(env, "engine")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "5"
    attempts = args_file.read_text().splitlines()
    assert "-x node-1" in attempts[1]
    assert "-x node-1,node-2" in attempts[2]


def test_dirty_node_is_not_resubmitted_inside_one_node_allocation(tmp_path):
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
echo "INFERA_E2E_GPU_NODE_DIRTY_NODE=node-a mock owner" >&2
exit 75
""",
    )
    env = _runner_env(tmp_path, mock_bin, count_file)
    env.update({"SLURM_JOB_ID": "42", "SLURM_JOB_NUM_NODES": "1"})

    result = _run_runner(env, "engine")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "1"
    assert "this one-node allocation cannot reselect" in result.stderr
    assert "SLURM submission limit reached" not in result.stderr


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


# Spur's refusal when the QoS's per-user ceiling on submitted jobs is full. It
# creates no job, so it is not one of the five real submissions the ceiling
# above counts -- the tier waits for one of its own sibling legs to free a slot,
# bounded by INFERA_E2E_SLURM_SLOT_WAIT rather than by that counter.
_SUBMIT_LIMIT_REFUSAL = (
    "Error: job submission failed\n\n"
    "Caused by:\n"
    "    code: 'Client specified an invalid argument', message: \"you have reached "
    'the QOS limit on submitted jobs per user (QOSMaxSubmitJobPerUserLimit)"\n'
)


def _slot_wait_env(tmp_path, mock_bin, count_file) -> dict[str, str]:
    env = _runner_env(tmp_path, mock_bin, count_file)
    # Two waits, then the budget is spent -- enough to prove both halves.
    env.update(
        {
            "INFERA_E2E_SLURM_SLOT_WAIT": "60",
            "INFERA_E2E_SLURM_SLOT_POLL": "30",
        }
    )
    return env


def test_per_user_submit_ceiling_is_waited_out_not_counted_as_an_attempt(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    count_file = tmp_path / "srun-count"
    count_file.write_text("0\n")
    _executable(mock_bin / "sleep", "exit 0\n")
    _executable(mock_bin / "squeue", "echo '101 infera-ci-mixed-1-vllm'\nexit 0\n")
    _executable(
        mock_bin / "srun",
        f"""
n=$(cat "$COUNT_FILE")
echo $((n + 1)) > "$COUNT_FILE"
cat >&2 <<'EOF'
{_SUBMIT_LIMIT_REFUSAL}
EOF
exit 1
""",
    )

    result = _run_runner(_slot_wait_env(tmp_path, mock_bin, count_file), "engine")

    assert result.returncode == 1
    # Two waits of 30s, then give up: the five-submission ceiling never applies,
    # because a refused submission never reached the queue.
    assert count_file.read_text().strip() == "3"
    assert "waiting 30s for one (30/60s)" in result.stderr
    assert "no submit slot in this QoS after 60s" in result.stderr
    assert "SLURM submission limit reached" not in result.stderr


def test_disagg_hold_waits_for_a_submit_slot_and_names_that_wall(tmp_path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    count_file = tmp_path / "sbatch-count"
    count_file.write_text("0\n")
    _executable(mock_bin / "sleep", "exit 0\n")
    _executable(mock_bin / "srun", "exit 1\n")
    _executable(mock_bin / "squeue", "echo '101 infera-ci-hold-1-vllm-disag'\nexit 0\n")
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
        f"""
n=$(cat "$COUNT_FILE")
echo $((n + 1)) > "$COUNT_FILE"
cat >&2 <<'EOF'
{_SUBMIT_LIMIT_REFUSAL}
EOF
exit 1
""",
    )

    result = _run_runner(_slot_wait_env(tmp_path, mock_bin, count_file), "e2e", "sglang", "disag")

    assert result.returncode == 1
    assert count_file.read_text().strip() == "3"
    assert "no submit slot in this QoS after 60s" in result.stderr
    # The hold ceiling is a different wall and must not be the one reported.
    assert "SLURM hold submission limit reached" not in result.stderr
    assert "no SLURM submit slot for a node hold within 60s" in result.stderr
