###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The connector contract, and the factory that picks one from the CLI args."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from infera.planner.decision import ScalingDecision

if TYPE_CHECKING:
    from infera.planner.args import PlannerArgs

logger = logging.getLogger(__name__)


@runtime_checkable
class PlannerConnector(Protocol):
    """Applies a scaling decision to whatever runs the engine replicas.

    Deliberately narrow: the planner decides *how many* replicas each pool
    needs and nothing else, so a connector only has to make the desired counts
    true. Counts are absolute rather than deltas, so a connector that drops a
    decision self-corrects on the next interval instead of drifting.
    """

    async def apply(self, decision: ScalingDecision) -> None:
        """Drive the deployment toward ``decision``."""
        ...

    async def aclose(self) -> None:
        """Release any transport resources."""
        ...


class NoOperationConnector:
    """Logs decisions without touching anything (``--no-operation``).

    Worth running for an interval or two after profiling: it shows what the
    planner *would* do against real traffic, which is how you catch profiling
    data that no longer matches the deployment before it costs you replicas.
    """

    async def apply(self, decision: ScalingDecision) -> None:
        logger.info("no-operation: would scale %s", decision.summary())

    async def aclose(self) -> None:
        return


def build_connector(args: PlannerArgs) -> PlannerConnector:
    """Construct the connector named by ``args``, honouring ``--no-operation``."""
    if args.no_operation:
        return NoOperationConnector()
    if args.connector == "kubernetes":
        from infera.planner.connectors.kubernetes import KubernetesConnector

        return KubernetesConnector(
            deployment_name=args.deployment_name,
            namespace=args.k8s_namespace,
            prefill_service=args.prefill_service,
            decode_service=args.decode_service,
        )
    if args.connector == "virtual":
        from infera.planner.connectors.virtual import VirtualConnector

        return VirtualConnector(etcd_endpoint=args.etcd_endpoint, prefix=args.etcd_prefix)
    raise ValueError(f"unknown connector {args.connector!r}")
