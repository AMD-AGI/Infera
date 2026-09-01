###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang PD-disaggregated adapter + the shared ``disagg_stack`` fixture.

Two cross-node workers over Mooncake RDMA: a prefill (``--disaggregation-mode
prefill``) on node 0 and a decode on node 1, KV transferred P->D. The launch
recipe mirrors .claude/regression/03-two-node-sglang-pd.md (``--disaggregation-*``
flags + ``MC_GID_INDEX`` env); placement/topology, image build, readiness poll
and teardown are shared (see :mod:`tests.e2e.harness.disagg_fixtures`), so this
file only maps :class:`EngineParams` + role -> argv/env.
"""

from __future__ import annotations

from ...harness import EngineAdapter, EngineParams
from ...harness.disagg_fixtures import make_disagg_stack_fixture
from ...harness.images import engine_image
from ...harness.params import DisaggRole

# Same image/Dockerfile run_tests.sh builds, for this run's target arch.
IMAGE, DOCKERFILE = engine_image("sglang")

# Prefill's Mooncake bootstrap port (decode connects here to fetch KV).
_BOOTSTRAP_PORT = "8998"


class SglangDisaggAdapter(EngineAdapter):
    engine = "sglang"
    module = "infera.engine.sglang"
    supports_disagg = True

    def gpus_per_worker(self, params: EngineParams) -> int:
        return max(1, params.tensor_parallel_size)

    def build_argv(self, params, *, port, host, server_ctx, gpu_ids):
        raise NotImplementedError(
            "SglangDisaggAdapter is PD-disaggregated only; use pd_mixed for mixed"
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
            "--model-path",
            params.model,
            "--port",
            str(port),
            "--host",
            host,
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
            "--no-enable-kv-events",
            "--trust-remote-code",
            # PD role + RDMA KV transport (SGLang bootstrap protocol over Mooncake).
            "--disaggregation-mode",
            role.value,  # "prefill" | "decode"
            "--disaggregation-transfer-backend",
            "mooncake",
        ]
        tp = max(1, params.tensor_parallel_size)
        if tp > 1:
            argv += ["--tp-size", str(tp)]
        # Same mapping as the PD-mixed adapter, so a row's axes mean one thing in
        # both tiers. Both legs get it from the same params, which is what SGLang
        # requires: it rejects a PD pair whose attention parallelism disagrees.
        if params.dp_attention:
            argv += ["--enable-dp-attention", "--dp-size", str(tp)]
        if params.expert_parallel:
            argv += ["--ep-size", str(tp)]
        if role.is_prefill:
            argv += ["--disaggregation-bootstrap-port", _BOOTSTRAP_PORT]
        # --mem-fraction-static is set per-case in matrix.py (Mooncake pins the KV
        # for RDMA, so it's bounded by the ionic ibv_reg_mr ceiling — see matrix).
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
        env = super().disagg_worker_env(
            params,
            role,
            advertise_host=advertise_host,
            gpu_ids=gpu_ids,
            gid_index=gid_index,
        )
        env["MC_GID_INDEX"] = gid_index
        return env


disagg_stack = make_disagg_stack_fixture(SglangDisaggAdapter, image=IMAGE, dockerfile=DOCKERFILE)
