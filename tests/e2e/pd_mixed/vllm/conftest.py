###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM engine adapter + the shared ``worker`` fixture.

Workers are spawned as subprocesses via ``python -m infera.engine.vllm`` (which
spawns ``vllm serve``); only the argv mapping is vLLM-specific.
"""

from __future__ import annotations

import os

import pytest_asyncio

from ...harness import EngineAdapter, EngineParams
from ...harness.fixtures import make_worker_fixture


class VllmAdapter(EngineAdapter):
    engine = "vllm"
    module = "infera.engine.vllm"

    def gpus_per_worker(self, params: EngineParams) -> int:
        return max(1, params.tensor_parallel_size)

    def build_argv(
        self,
        params: EngineParams,
        *,
        port: int,
        host: str,
        server_ctx: dict,
        gpu_ids: list[int],
    ) -> list[str]:
        argv = [
            "python3",
            "-m",
            self.module,
            "--model",  # vLLM uses --model (not --model-path)
            params.model,
            "--port",
            str(port),
            "--host",
            host,
            # Same infera discovery/transport pins as the SGLang worker.
            "--discovery-backend",
            "etcd",
            "--etcd-endpoint",
            server_ctx["etcd_endpoint"],
            "--etcd-prefix",
            server_ctx["etcd_prefix"],
            "--request-transport",
            "http",
            "--kv-event-transport",
            "zmq",
            "--trust-remote-code",
        ]

        tp = max(1, params.tensor_parallel_size)
        if tp > 1:
            argv += ["--tensor-parallel-size", str(tp)]
        if params.expert_parallel:
            argv += ["--enable-expert-parallel"]

        argv += list(params.extra_args)
        return argv


worker = make_worker_fixture(VllmAdapter)


@pytest_asyncio.fixture
async def kvd_daemon(tmp_path):
    """Start one ``infera.kvd`` daemon (RAM + local-disk L3) for the kvd-offload
    test, and yield a small handle: the socket path a worker's InferaKvdConnector
    connects to, plus a ``stats()`` reader over ``infera.kvd.statctl``.

    RAM tier kept small (``--max-bytes``) so the hot set spills to the disk (L3)
    tier under ``tmp_path``; both are torn down with the daemon.
    """
    import json
    import subprocess
    import time

    sock = str(tmp_path / "kvd.sock")
    long_dir = tmp_path / "l3"
    long_dir.mkdir()
    proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "infera.kvd",
            "--socket",
            sock,
            "--max-bytes",
            "4G",
            "--long-path",
            str(long_dir),
            "--long-bytes",
            "64G",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    def _stats() -> dict:
        try:
            out = subprocess.run(
                ["python3", "-m", "infera.kvd.statctl", "--socket", sock],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout
            return json.loads(out)
        except Exception:
            return {}

    # Wait for the daemon's unix socket to appear (fail-fast if it never boots).
    for _ in range(100):
        if os.path.exists(sock):
            break
        if proc.poll() is not None:
            raise RuntimeError(f"infera.kvd daemon exited early (rc={proc.returncode})")
        time.sleep(0.1)

    try:
        yield {"socket": sock, "long_dir": str(long_dir), "stats": _stats}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
