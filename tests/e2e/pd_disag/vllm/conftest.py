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

# Per-case KV connector selector. A case opts INTO MoRIIO by setting
# extra_env={_KV_CONNECTOR_ENV: "MoRIIOConnector"} in its matrix row; absent it,
# the connector defaults to Mooncake so every existing case is byte-for-byte
# unchanged. The key is consumed here (to shape argv/env) and stripped from the
# worker's actual env — the worker never sees it.
_KV_CONNECTOR_ENV = "INFERA_E2E_KV_CONNECTOR"

# MoRIIO fixed control ports (host-networked, one PD stack per node pair; freed by
# the RDMA teardown between runs). host_ip/http_port/proxy_ip are derived per-role.
_MORIIO_PROXY_PING_PORT = 36000
_MORIIO_HANDSHAKE_PORT = 36100
_MORIIO_NOTIFY_PORT = 36200


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

        connector = dict(params.extra_env).get(_KV_CONNECTOR_ENV, "MooncakeConnector")
        if connector == "MoRIIOConnector":
            # MoRIIO carries all its coordination in extra_config (the router forges
            # the request_id from the workers' handshake/notify ports it reads back
            # from etcd — see infera/router/disagg_protocols/vllm_moriio.py). proxy_ip
            # is the router/prefill node; http_port is this worker's own HTTP port.
            proxy_ip = server_ctx["etcd_endpoint"].split(":")[0]
            kv_cfg = {
                "kv_connector": "MoRIIOConnector",
                "kv_role": role.kv_role(),
                "kv_connector_extra_config": {
                    "host_ip": advertise_host,
                    "proxy_ip": proxy_ip,
                    "proxy_ping_port": _MORIIO_PROXY_PING_PORT,
                    "http_port": port,
                    "handshake_port": _MORIIO_HANDSHAKE_PORT,
                    "notify_port": _MORIIO_NOTIFY_PORT,
                    "backend": "rdma",
                    "tp_size": str(tp),
                },
            }
        else:
            kv_cfg = {"kv_connector": connector, "kv_role": role.kv_role()}
        argv += ["--kv-transfer-config", json.dumps(kv_cfg)]

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
        case_env = dict(params.extra_env)
        # MoRIIO reads its GID rail from MORI_IB_GID_INDEX; only set on the MoRIIO
        # path so the default Mooncake env is untouched.
        if case_env.get(_KV_CONNECTOR_ENV) == "MoRIIOConnector":
            env["MORI_IB_GID_INDEX"] = gid_index
        # The connector selector is consumed by build_disagg_argv, not the worker.
        case_env.pop(_KV_CONNECTOR_ENV, None)
        env.update(case_env)
        return env


disagg_stack = make_disagg_stack_fixture(VllmDisaggAdapter, image=IMAGE, dockerfile=DOCKERFILE)
