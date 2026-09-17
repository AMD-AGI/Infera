###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Run the example launchers with command recorders, without SSH or Docker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_KIT = Path(__file__).resolve().parents[3] / "examples/sglang_1p1d_glm5.2"


@pytest.fixture
def launch_env(tmp_path):
    recorder = tmp_path / "record"
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    recorder.chmod(0o755)
    (tmp_path / "docker").symlink_to(recorder)
    env = {
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "COMMAND_LOG": str(tmp_path / "commands.jsonl"),
        "SSH_CMD": str(recorder),
        "PREFILL_NODE": "node-a",
        "DECODE_NODE": "node-b",
        "PREFILL_IP": "192.0.2.1",
        "DECODE_IP": "192.0.2.2",
        "MY_IP": "192.0.2.1",
        "ETCD_IP": "192.0.2.1",
        "KIT_DIR": str(_KIT),
        "INFERA_IMAGE": "example:test",
        "MODEL": "/models/test",
        "MODEL_MOUNT": "/models",
        "RDMA_IB_DEVICES": "mlx5_0",
        "MC_GID_INDEX": "3",
        "NIC": "eth0",
        "PREFILL_KVD": "0",
        "DECODE_KVD": "0",
        "KVD": "0",
        "ROLE": "prefill",
        "TP": "1",
    }
    return env


def commands(env):
    return [json.loads(line) for line in Path(env["COMMAND_LOG"]).read_text().splitlines()]


def test_up_forwards_startup_settings_to_both_hosts(launch_env):
    launch_env.update(READY_TIMEOUT="5400", HOST_RDMA_MOUNT="/host-rdma/custom.so")
    subprocess.run(
        ["bash", str(_KIT / "engine/up.sh")], env=launch_env, check=True, capture_output=True
    )
    calls = commands(launch_env)
    for node in ("node-a", "node-b"):
        startup = [args[1] for args in calls if args[0] == node and "start_container" in args[1]]
        assert len(startup) == 1
        assert "HOST_RDMA_MOUNT=/host-rdma/custom.so" in startup[0]
        legs = [args[1] for args in calls if args[0] == node and "engine/leg.sh" in args[1]]
        assert len(legs) == 1
        assert "READY_TIMEOUT=5400" in legs[0]


@pytest.mark.parametrize("timeout, expected", [(None, "3600"), ("5400", "5400")])
def test_leg_passes_ready_timeout_to_container(launch_env, timeout, expected):
    if timeout is not None:
        launch_env["READY_TIMEOUT"] = timeout
    subprocess.run(
        ["bash", str(_KIT / "engine/leg.sh")], env=launch_env, check=True, capture_output=True
    )
    launches = [args for args in commands(launch_env) if args[:2] == ["exec", "-d"]]
    assert len(launches) == 1
    assert f"INFERA_ENGINE_READY_TIMEOUT={expected}" in launches[0]
    assert not any(arg.startswith("INFERA_SGLANG_READY_TIMEOUT=") for arg in launches[0])
