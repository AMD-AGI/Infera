###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM PD-disaggregated adapter + the shared ``disagg_stack`` fixture.

Two cross-node workers over Mooncake RDMA: a prefill (``kv_producer``) on node 0
and a decode (``kv_consumer``) on node 1, KV transferred P->D. The launch recipe
mirrors .claude/regression/10-two-node-atom-pd.md (the ``--kv-transfer-config``
mooncake JSON + ``ATOM_HOST_IP`` / ``MC_*`` env); placement/topology, image
build, readiness poll and teardown are shared (see
:mod:`tests.e2e.harness.disagg_fixtures`), so this file only maps
:class:`EngineParams` + role -> argv/env.

ATOM quirks vs vllm/sglang:
- OpenAI HTTP port is ``--server-port`` (``--port`` is the torch MASTER_PORT).
- PD requests only work over ``/v1/completions`` (the shared disagg correctness
  uses the counting probe, which is completions-based — see disagg_suite).
- Cross-node RDMA needs both roles' ``ib_device`` on the same subnet/rail; it
  follows the cluster's ``MC_TE_FILTERS`` and ``INFERA_E2E_ATOM_IB_DEVICE``
  overrides that.
"""

from __future__ import annotations

import json
import os

from ...harness import EngineAdapter, EngineParams
from ...harness.cluster import kv_transport_env
from ...harness.disagg_fixtures import make_disagg_stack_fixture
from ...harness.images import engine_image
from ...harness.params import DisaggRole

# Same image/Dockerfile run_tests.sh builds, for this run's target arch.
IMAGE, DOCKERFILE = engine_image("atom")

# Per-role Mooncake handshake ports (distinct so co-located debugging is safe;
# each role is on its own node here). http_port == the engine's --server-port.
_HANDSHAKE_PORT = {DisaggRole.PREFILL: 6301, DisaggRole.DECODE: 6311}


class AtomDisaggAdapter(EngineAdapter):
    engine = "atom"
    module = "infera.engine.atom"
    supports_disagg = True

    def gpus_per_worker(self, params: EngineParams) -> int:
        return max(1, params.tensor_parallel_size)

    def build_argv(self, params, *, port, host, server_ctx, gpu_ids):
        raise NotImplementedError(
            "AtomDisaggAdapter is PD-disaggregated only; use pd_mixed for mixed"
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
        kv_cfg = {
            "kv_role": role.kv_role(),  # kv_producer | kv_consumer
            "kv_connector": "mooncake",
            "handshake_port": _HANDSHAKE_PORT[role],
            "http_port": port,
            "proxy_ip": advertise_host,
        }
        # ATOM names its own Mooncake NIC and guesses the MI300X "rdma<N>"; where
        # devices are named otherwise that discovers 0 HCAs and Mooncake drops to
        # TCP. Default to the rail the cluster pinned for every engine.
        ib_device = (
            os.environ.get("INFERA_E2E_ATOM_IB_DEVICE")
            or kv_transport_env().get("MC_TE_FILTERS", "").split(",")[0]
        )
        if ib_device:
            kv_cfg["ib_device"] = ib_device

        argv = [
            "python3",
            "-m",
            self.module,
            "--model",
            params.model,
            # ATOM's OpenAI HTTP port (NOT --port, which is the torch MASTER_PORT).
            "--server-port",
            str(port),
            "--host",
            host,
            "--advertise-host",
            advertise_host,
            "--trust-remote-code",
            "--etcd-endpoint",
            server_ctx["etcd_endpoint"],
            "--etcd-prefix",
            server_ctx["etcd_prefix"],
            "--kv-transfer-config",
            json.dumps(kv_cfg),
        ]
        tp = max(1, params.tensor_parallel_size)
        if tp > 1:
            argv += ["--tensor-parallel-size", str(tp)]
        # --max-model-len / --gpu-memory-utilization set per-case in matrix.py.
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
            "OMP_NUM_THREADS": "1",
            # Pin Mooncake engine/handshake to the peer-reachable shared subnet.
            "ATOM_HOST_IP": advertise_host,
            "MC_DISABLE_HIP_TRANSPORT": "1",  # force RDMA (not HIP P2P)
            "RDMAV_FORK_SAFE": "1",
            "MC_GID_INDEX": gid_index,
        }
        env.update(dict(params.extra_env))
        return env


disagg_stack = make_disagg_stack_fixture(AtomDisaggAdapter, image=IMAGE, dockerfile=DOCKERFILE)
