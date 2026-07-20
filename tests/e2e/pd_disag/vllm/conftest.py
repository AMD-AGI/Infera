###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM PD-disaggregated adapter + the shared ``disagg_stack`` fixture.

Two cross-node workers over Mooncake RDMA: a prefill (``kv_producer``) on node 0
and a decode (``kv_consumer``) on node 1, KV transferred P->D. The launch recipe
mirrors .claude/regression/04-two-node-vllm-pd.md (the ``kv-transfer-config``,
``--advertise-host``, ``--no-enable-kv-events`` flags + the ``VLLM_HOST_IP`` /
``MC_GID_INDEX`` / bootstrap-port env), but placement/topology and image build
are handled generically by the harness (launcher + cluster), so this file only
maps :class:`EngineParams` + role -> argv/env.

Only the argv/env mapping is vLLM-specific; the node placement, image build,
readiness poll and teardown are shared (see
:mod:`tests.e2e.harness.disagg_fixtures`).
"""

from __future__ import annotations

import json

from ...harness import EngineAdapter, EngineParams
from ...harness.disagg_fixtures import make_disagg_stack_fixture
from ...harness.params import DisaggRole

# Same image/Dockerfile tag run_tests.sh builds for the vLLM engine.
IMAGE = "infera/engine-vllm:test-local"
DOCKERFILE = "deploy/docker/Dockerfile.vllm"

# Mooncake bootstrap TCP port (prefill's BootstrapServer; see vllm/args.py
# _compute_disagg_meta). Both roles set it; they're on different nodes.
_BOOTSTRAP_PORT = "8998"


class VllmDisaggAdapter(EngineAdapter):
    engine = "vllm"
    module = "infera.engine.vllm"
    supports_disagg = True

    def gpus_per_worker(self, params: EngineParams) -> int:
        return max(1, params.tensor_parallel_size)

    def build_argv(self, params, *, port, host, server_ctx, gpu_ids):
        # This adapter is PD-disaggregated only; PD-mixed lives in
        # tests/e2e/pd_mixed/vllm/. Kept explicit so a mis-wire fails loudly.
        raise NotImplementedError(
            "VllmDisaggAdapter is PD-disaggregated only; use pd_mixed for mixed"
        )

    def build_disagg_argv(
        self,
        params: EngineParams,
        role: DisaggRole,
        *,
        port: int,
        host: str,
        server_ctx: dict,
        advertise_host: str,
        gpu_ids: list[int],
    ) -> list[str]:
        argv = [
            "python3",
            "-m",
            self.module,
            "--model",
            params.model,
            "--port",
            str(port),
            "--host",
            host,
            # Routable address peers use: the router reaches this worker here and
            # Mooncake publishes it as the bootstrap addr (vllm/args.py).
            "--advertise-host",
            advertise_host,
            "--discovery-backend",
            "etcd",
            "--etcd-endpoint",
            server_ctx["etcd_endpoint"],
            "--etcd-prefix",
            server_ctx["etcd_prefix"],
            "--request-transport",
            "http",
            # PD KV moves over the RDMA connector, not the KV-event bus.
            "--no-enable-kv-events",
            "--trust-remote-code",
        ]

        tp = max(1, params.tensor_parallel_size)
        if tp > 1:
            argv += ["--tensor-parallel-size", str(tp)]
        argv += ["--max-num-seqs", "64"]
        argv += [
            "--kv-transfer-config",
            json.dumps({"kv_connector": "MooncakeConnector", "kv_role": role.kv_role()}),
        ]
        if params.expert_parallel:
            argv += ["--enable-expert-parallel"]
        argv += list(params.extra_args)
        return argv

    def disagg_worker_env(
        self,
        params: EngineParams,
        role: DisaggRole,
        *,
        advertise_host: str,
        gpu_ids: list[int],
        gid_index: str,
    ) -> dict[str, str]:
        env = {
            "HIP_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids),
            # Mooncake/vLLM must bind the data-plane IP (not the default NIC).
            "VLLM_HOST_IP": advertise_host,
            "MC_GID_INDEX": gid_index,
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": _BOOTSTRAP_PORT,
        }
        env.update(dict(params.extra_env))
        return env


disagg_stack = make_disagg_stack_fixture(VllmDisaggAdapter, image=IMAGE, dockerfile=DOCKERFILE)
