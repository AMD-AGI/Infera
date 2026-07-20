###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM engine adapter + the shared ``worker`` fixture.

Workers are spawned as subprocesses via ``python -m infera.engine.atom`` (which
spawns ``atom.entrypoints.openai_server``); only the argv mapping is ATOM-specific.

ATOM quirks vs sglang/vllm:
- The OpenAI HTTP port infera routes to is ``--server-port``. ATOM's ``--port``
  is the internal torch-distributed MASTER_PORT (auto-allocated by the launcher),
  so we must NOT pass ``--port`` here.
- The launcher has no ``--request-transport`` / ``--discovery-backend`` flags;
  ATOM registers over HTTP by default, which is what the test server routes over.
"""

from __future__ import annotations

from ...harness import EngineAdapter, EngineParams
from ...harness.fixtures import make_worker_fixture


class AtomAdapter(EngineAdapter):
    engine = "atom"
    module = "infera.engine.atom"

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
            "--model",
            params.model,
            "--server-port",  # ATOM's OpenAI HTTP port (NOT --port, see module docstring)
            str(port),
            "--host",
            host,
            "--etcd-endpoint",
            server_ctx["etcd_endpoint"],
            "--etcd-prefix",
            server_ctx["etcd_prefix"],
        ]

        tp = max(1, params.tensor_parallel_size)
        if tp > 1:
            argv += ["--tensor-parallel-size", str(tp)]
        if params.expert_parallel:
            argv += ["--enable-expert-parallel"]
        argv += list(params.extra_args)
        return argv


worker = make_worker_fixture(AtomAdapter)
