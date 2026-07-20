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
