###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Resize the engine pools by patching the InferaDeployment.

The operator already reconciles ``spec.services[<name>].replicas`` into the
underlying Deployment or LeaderWorkerSet, so the planner only has to change the
desired count in the custom resource and let the existing control loop do the
rest -- no operator changes, and the CR stays the single source of truth for
what the deployment should look like.

A JSON merge patch touches only the two replica fields, so it will not clobber
concurrent edits to anything else in the spec. The planner needs ``get`` and
``patch`` on ``inferadeployments`` in the deployment's namespace.
"""

from __future__ import annotations

import logging

import httpx

from infera.common import k8s_client
from infera.planner.decision import ScalingDecision

logger = logging.getLogger(__name__)

CRD_GROUP = "infera.amd.com"
CRD_VERSION = "v1alpha1"
CRD_PLURAL = "inferadeployments"

_MERGE_PATCH = "application/merge-patch+json"


class KubernetesConnector:
    """Patches replica counts on an ``InferaDeployment``."""

    def __init__(
        self,
        *,
        deployment_name: str,
        namespace: str | None = None,
        prefill_service: str = "prefill",
        decode_service: str = "decode",
        timeout: float = 10.0,
    ) -> None:
        if not deployment_name:
            raise ValueError("KubernetesConnector needs the InferaDeployment name")
        self.deployment_name = deployment_name
        self.namespace = namespace or k8s_client.in_cluster_namespace()
        self.prefill_service = prefill_service
        self.decode_service = decode_service
        self._http = k8s_client.make_client(timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def resource_path(self) -> str:
        return (
            f"/apis/{CRD_GROUP}/{CRD_VERSION}/namespaces/{self.namespace}"
            f"/{CRD_PLURAL}/{self.deployment_name}"
        )

    def patch_for(self, decision: ScalingDecision) -> dict:
        return {
            "spec": {
                "services": {
                    self.prefill_service: {"replicas": decision.num_prefill},
                    self.decode_service: {"replicas": decision.num_decode},
                }
            }
        }

    async def apply(self, decision: ScalingDecision) -> None:
        try:
            resp = await self._http.patch(
                self.resource_path,
                json=self.patch_for(decision),
                headers={"Content-Type": _MERGE_PATCH},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # 404 usually means --deployment-name or the namespace is wrong;
            # 403 means the planner's ServiceAccount is missing the RBAC rule.
            logger.warning(
                "patching %s failed with %d: %s",
                self.resource_path,
                exc.response.status_code,
                exc.response.text[:500],
            )
            return
        except httpx.HTTPError as exc:
            logger.warning(
                "could not reach the API server to patch %s: %s: %s",
                self.resource_path,
                type(exc).__name__,
                exc,
            )
            return
        logger.info("patched %s: %s", self.deployment_name, decision.summary())
