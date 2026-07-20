###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared pytest fixture for the PD-disaggregated per-engine suites.

``make_disagg_stack_fixture(adapter_factory, image=..., dockerfile=...)`` builds
the ``disagg_stack`` factory fixture each engine's conftest binds to its own
:class:`~.adapter.EngineAdapter`. It stands up the ENTIRE stack in containers on
the allocated nodes — etcd + infera router + prefill on node 0, decode on node 1
— so nothing infera runs on the driver host (which only issues srun/docker and
runs the HTTP correctness probes). The engine only supplies role-aware argv/env
via ``build_disagg_argv`` / ``disagg_worker_env``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable

import pytest
import pytest_asyncio

from . import cluster
from .adapter import EngineAdapter
from .launcher import (
    ROUTER_PORT,
    SrunDockerLauncher,
    wait_url_ok,
    wait_workers_active,
)
from .params import DisaggRole, EngineParams

# Fixed per-role engine HTTP ports (each container is host-networked on its own
# node; removed by name before launch, so a fixed port never lingers).
_PREFILL_PORT = 30001
_DECODE_PORT = 30002


def require_disagg_env() -> None:
    """Skip unless we can actually place a 2-node PD stack (SLURM + allocation +
    >=2 nodes + resolvable node IPs)."""
    if not cluster.have_slurm():
        pytest.skip("PD-disaggregation e2e needs SLURM (srun/scontrol) on PATH")
    # An explicit node pin (INFERA_E2E_NODES, set by run_tests.sh's disagg
    # dispatcher) means we place the stack ourselves via per-node `srun` and need
    # no held allocation — the Spur scheduler has no salloc/sbatch to sit in.
    if not os.environ.get("INFERA_E2E_NODES") and not cluster.in_allocation():
        pytest.skip("PD-disaggregation e2e needs INFERA_E2E_NODES or a live SLURM allocation")
    pair = cluster.pd_nodes()
    if pair is None:
        pytest.skip(
            f"PD-disaggregation e2e needs >=2 nodes; allocation has "
            f"{cluster.allocated_nodes() or 'none'}"
        )
    for node in pair:
        if cluster.node_ip(node) is None:
            pytest.skip(
                f"could not resolve a routable IP for node {node} (set INFERA_E2E_NODE_IPS)"
            )


def make_disagg_stack_fixture(
    adapter_factory: Callable[[], EngineAdapter],
    *,
    image: str,
    dockerfile: str,
    shell_entrypoint: bool = False,
):
    """Return a function-scoped ``disagg_stack`` fixture bound to ``adapter_factory``.

    The fixture yields an async factory ``await disagg_stack(params)`` that, on
    the allocated nodes: builds the image on both, starts etcd + the infera
    router (node 0), launches a prefill worker (node 0) + a decode worker
    (node 1), waits for both active, and returns the server context
    (``{"url", "etcd_endpoint", "etcd_prefix"}``). Every container is torn down
    after the test (workers, then router, then etcd), freeing both nodes' GPUs.
    """

    @pytest_asyncio.fixture
    async def disagg_stack():
        adapter = adapter_factory()
        launcher = SrunDockerLauncher(
            image=image, dockerfile=dockerfile, shell_entrypoint=shell_entrypoint
        )
        handles: list = []  # torn down in reverse order

        async def _up(params: EngineParams | None = None) -> dict:
            params = params or EngineParams()
            if not adapter.supports_disagg:
                pytest.skip(f"{adapter.engine} has no PD-disaggregated adapter yet")
            require_disagg_env()

            prefill_node, decode_node = cluster.pd_nodes()  # type: ignore[misc]
            prefill_ip = cluster.node_ip(prefill_node)
            decode_ip = cluster.node_ip(decode_node)
            gid = cluster.gid_index()
            tag = uuid.uuid4().hex[:8]
            tp = max(1, params.tensor_parallel_size)
            gpu_ids = list(range(tp))

            launcher.cleanup_stale([prefill_node, decode_node])
            launcher.ensure_images([prefill_node, decode_node])

            # etcd + router on node 0 (control plane).
            etcd = launcher.start_etcd(
                node=prefill_node, container=f"infera-e2e-etcd-{tag}", advertise_host=prefill_ip
            )
            handles.append(etcd)
            etcd_endpoint = f"{prefill_ip}:{etcd.port}"
            await wait_url_ok(
                f"http://{etcd_endpoint}/health", timeout=60, launcher=launcher, handle=etcd
            )

            etcd_prefix = f"/infera/e2e-{tag}/"
            router = launcher.start_router(
                node=prefill_node,
                container=f"infera-e2e-router-{tag}",
                advertise_host=prefill_ip,
                etcd_endpoint=etcd_endpoint,
                etcd_prefix=etcd_prefix,
                model=params.model,
            )
            handles.append(router)
            server_ctx = {
                "url": f"http://{prefill_ip}:{ROUTER_PORT}",
                "etcd_endpoint": etcd_endpoint,
                "etcd_prefix": etcd_prefix,
            }
            await wait_url_ok(
                f"{server_ctx['url']}/health", timeout=120, launcher=launcher, handle=router
            )

            # prefill on node 0, decode on node 1.
            workers = []
            for role, node, ip, port in (
                (DisaggRole.PREFILL, prefill_node, prefill_ip, _PREFILL_PORT),
                (DisaggRole.DECODE, decode_node, decode_ip, _DECODE_PORT),
            ):
                argv = adapter.build_disagg_argv(
                    params,
                    role,
                    port=port,
                    host="0.0.0.0",
                    server_ctx=server_ctx,
                    advertise_host=ip,
                    gpu_ids=gpu_ids,
                )
                env = adapter.disagg_worker_env(
                    params, role, advertise_host=ip, gpu_ids=gpu_ids, gid_index=gid
                )
                h = launcher.start(
                    node=node,
                    argv=argv,
                    env=env,
                    container=f"infera-e2e-{adapter.engine}-{role.value}-{tag}",
                    advertise_host=ip,
                    port=port,
                    role=role.value,
                )
                handles.append(h)
                workers.append(h)

            await wait_workers_active(
                launcher, server_ctx["url"], workers, timeout=params.server_ready_timeout
            )
            return server_ctx

        yield _up

        for h in reversed(handles):
            launcher.stop(h)

    return disagg_stack
