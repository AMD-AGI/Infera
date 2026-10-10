###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Run the SLURM preflight launcher with srun/docker recorders."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "infera/tools/preflight/run_preflight_slurm.sh"
_FORWARDED = {
    "INFERA_PREFLIGHT_MOONCAKE_OPCODE": "write",
    "INFERA_PREFLIGHT_RUN_ID": "write-check-001",
    "INFERA_PREFLIGHT_RDMA_DEVICE": "ionic_0,ionic_1",
}


def test_slurm_launcher_forwards_mooncake_selection_into_containers(tmp_path):
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/bash\nwhile [[ "$1" == -* ]]; do shift; done\nexec "$@"\n')
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "seen = {k: v for k, v in os.environ.items() if k.startswith('INFERA_PREFLIGHT_')}\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'env': seen}) + '\\n')\n"
    )
    for stub in (srun, docker):
        stub.chmod(0o755)
    log = tmp_path / "docker.jsonl"
    env = {
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "COMMAND_LOG": str(log),
        "NODES": "node1,node2",
        "PARTITION": "test",
        "IMAGE": "example:test",
        "STORAGE_PATH": "",
        "DUMP_PATH": str(tmp_path / "dump"),
        **_FORWARDED,
    }

    subprocess.run(["bash", str(_SCRIPT)], env=env, check=True, capture_output=True)

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    run = next(call for call in calls if call["argv"][:1] == ["run"])
    for name, value in _FORWARDED.items():
        # A bare `-e NAME` passes the invoking shell's value and leaves an unset one unset.
        assert run["argv"][run["argv"].index(name) - 1] == "-e"
        assert run["env"][name] == value
