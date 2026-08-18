###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Changing pool sizes through the router.

The supported way to resize a pool is to edit `spec.services.<name>.replicas` on
the InferaDeployment, which the operator then reconciles. That is a
``kubectl patch`` -- fine for a human, awkward for the orchestration layer that
usually decides these things, which then needs cluster credentials and has to
know the CR's shape.

This puts that same edit behind the router's admin API. It writes the CR and
nothing else: the generated Deployment and LeaderWorkerSet are derived state,
rewritten on every reconcile pass, so scaling those directly succeeds, reports
no error, and is silently undone a few seconds later.

**Not an autoscaler, and not a step toward one.** It is a way to issue a
decision that has already been made. Two things make the automated version a
different problem: a worker takes around two minutes to become useful, which is
longer than most bursts last, and Kubernetes offers no way to choose *which*
replica to remove, so a scale-down is as likely to take the worker holding the
warmest cache as any other.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from infera.common.k8s_client import in_cluster_namespace, make_client

logger = logging.getLogger(__name__)

# Written by the operator onto every workload it generates; the router reads its
# own to find the deployment it belongs to, rather than being told separately.
LABEL_DEPLOYMENT = "infera.amd.com/deployment"
LABEL_SERVICE = "infera.amd.com/service"

_CR_GROUP = "infera.amd.com"
_CR_VERSION = "v1alpha1"
_CR_PLURAL = "inferadeployments"

# Container states a Pod does not recover from on its own. Distinguished from
# the ordinary ones (ContainerCreating, PodInitializing) because those resolve
# in seconds and reporting them would make every scale-up look stuck.
_TERMINAL_WAITS = frozenset(
    {"ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff", "CreateContainerConfigError"}
)


def _one_line(reason: str | None, message: str | None) -> str:
    """A reason and its detail on one line, short enough to log.

    Scheduler messages enumerate every node and run to several hundred
    characters; the head of one carries the verdict.
    """
    text = " ".join((message or "").split())
    if len(text) > 300:
        text = text[:297] + "..."
    if reason and text:
        return f"{reason}: {text}"
    return reason or text or "unknown"


class ScalingError(Exception):
    """A scale request that cannot be honoured, with a reason to return."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class DeploymentScaler:
    """Reads and resizes the pools of one InferaDeployment."""

    def __init__(self, *, namespace: str | None = None, pod_name: str | None = None) -> None:
        self._namespace = namespace or in_cluster_namespace()
        self._pod_name = pod_name or os.environ.get("POD_NAME", "")
        self._deployment: str | None = None

    async def _resolve_deployment(self, client: httpx.AsyncClient) -> str:
        """Find the InferaDeployment this router was created by.

        Read from the router's own Pod labels rather than configured, so the
        two cannot disagree: a name passed by flag would keep pointing at the
        old deployment after a rename, and resize a fleet nobody asked about.
        """
        if self._deployment:
            return self._deployment
        if not self._pod_name:
            raise ScalingError(
                "scaling needs POD_NAME (downward API) to find its own deployment",
                status=503,
            )
        resp = await client.get(f"/api/v1/namespaces/{self._namespace}/pods/{self._pod_name}")
        if resp.status_code == 403:
            raise ScalingError(
                "the router's ServiceAccount cannot read its own Pod; re-apply the operator RBAC",
                status=403,
            )
        resp.raise_for_status()
        labels = (resp.json().get("metadata") or {}).get("labels") or {}
        name = labels.get(LABEL_DEPLOYMENT)
        if not name:
            raise ScalingError(
                "this router was not created by the operator "
                f"(no {LABEL_DEPLOYMENT} label), so there is no deployment to scale",
                status=409,
            )
        self._deployment = name
        return name

    def _cr_path(self, name: str) -> str:
        return f"/apis/{_CR_GROUP}/{_CR_VERSION}/namespaces/{self._namespace}/{_CR_PLURAL}/{name}"

    async def _fetch(self, client: httpx.AsyncClient, name: str) -> dict:
        resp = await client.get(self._cr_path(name))
        if resp.status_code == 403:
            raise ScalingError(
                "the router's ServiceAccount cannot read InferaDeployments; "
                "re-apply the operator RBAC",
                status=403,
            )
        if resp.status_code == 404:
            raise ScalingError(f"InferaDeployment {name!r} not found", status=404)
        resp.raise_for_status()
        return resp.json()

    async def snapshot(self) -> dict:
        """Every pool's requested and observed size.

        Both, because they answer different questions: the spec is what was
        asked for, the status is what the cluster has managed so far, and a
        caller deciding whether to scale again needs to know a previous request
        is still landing.
        """
        async with make_client() as client:
            name = await self._resolve_deployment(client)
            cr = await self._fetch(client, name)
            spec = (cr.get("spec") or {}).get("services") or {}
            status = (cr.get("status") or {}).get("services") or {}
            pools = {}
            for svc, cfg in sorted(spec.items()):
                observed = status.get(svc) or {}
                want = int(cfg.get("replicas", 1))
                pools[svc] = pool = {
                    "role": cfg.get("role") or "mixed",
                    "replicas": want,
                    "current_replicas": int(observed.get("replicas", 0)),
                    "ready_replicas": int(observed.get("readyReplicas", 0)),
                    "nodes_per_replica": int(cfg.get("numberOfNodes", 1)),
                }
                if pool["ready_replicas"] < want:
                    # Short of what was asked for. Whether that is a scale-up
                    # still landing or one that never will is not visible in the
                    # counts -- a replica the scheduler could not place is
                    # missing from both -- so the reason comes from the Pods.
                    blocked = await self._blocked_reason(client, name, svc)
                    if blocked:
                        pool["blocked"] = blocked

        return {
            "deployment": name,
            "namespace": self._namespace,
            # The operator's own verdict, which compares ready replicas against
            # the spec rather than against the ones that exist. That is the
            # comparison a caller wants and the easy one to get wrong.
            "state": (cr.get("status") or {}).get("state") or "unknown",
            "services": pools,
        }

    async def _blocked_reason(self, client: httpx.AsyncClient, name: str, svc: str) -> str | None:
        """Why this pool's Pods are not running, if the cluster says so.

        Reported rather than diagnosed: the message is the scheduler's or the
        kubelet's, so a caller polling this endpoint can tell "still starting"
        from "will never start" without reaching for kubectl. Best-effort --
        a pool short of replicas is worth reporting even when the reason is not
        available.
        """
        selector = f"{LABEL_DEPLOYMENT}={name},{LABEL_SERVICE}={svc}"
        try:
            resp = await client.get(
                f"/api/v1/namespaces/{self._namespace}/pods",
                params={"labelSelector": selector},
            )
            resp.raise_for_status()
            pods = resp.json().get("items") or []
        except Exception as exc:  # noqa: BLE001 - a missing reason is not an error
            logger.debug("could not read Pods for %s/%s: %s", name, svc, exc)
            return None

        for pod in pods:
            status = pod.get("status") or {}
            if status.get("phase") != "Pending":
                continue
            for cond in status.get("conditions") or []:
                if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
                    return _one_line(cond.get("reason"), cond.get("message"))
            for cs in status.get("containerStatuses") or []:
                waiting = (cs.get("state") or {}).get("waiting") or {}
                # ImagePullBackOff and friends: scheduled, but never going to
                # run, which reads the same as a slow start from the counts.
                if waiting.get("reason") in _TERMINAL_WAITS:
                    return _one_line(waiting.get("reason"), waiting.get("message"))
        return None

    async def scale(self, requested: dict[str, int]) -> dict:
        """Resize the named pools, all of them or none.

        One patch carries the whole request, so a caller rebalancing prefill
        against decode cannot end up with one applied and the other rejected --
        a state neither the caller nor the cluster asked for, and one that could
        leave a PD deployment lopsided until someone noticed.
        """
        if not requested:
            raise ScalingError("no services named")

        async with make_client() as client:
            name = await self._resolve_deployment(client)
            cr = await self._fetch(client, name)
            known = (cr.get("spec") or {}).get("services") or {}
            _validate(requested, known)

            patch = {"spec": {"services": {s: {"replicas": n} for s, n in requested.items()}}}
            resp = await client.patch(
                self._cr_path(name),
                content=json.dumps(patch),
                headers={"Content-Type": "application/merge-patch+json"},
            )
            if resp.status_code == 403:
                raise ScalingError(
                    "the router's ServiceAccount cannot patch InferaDeployments; "
                    "re-apply the operator RBAC",
                    status=403,
                )
            resp.raise_for_status()

        logger.info(
            "scaled %s: %s",
            name,
            ", ".join(f"{s}={n}" for s, n in sorted(requested.items())),
        )
        return await self.snapshot()


def _validate(requested: dict[str, int], known: dict) -> None:
    """Reject what the cluster would accept but nobody wants.

    The API server would take any of these -- the CR's own validation only
    bounds `replicas` at zero -- and the damage would show up later as workers
    that never arrive or a pool that stops serving.
    """
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ScalingError(
            f"no such service(s): {', '.join(unknown)}. "
            f"This deployment has: {', '.join(sorted(known))}"
        )

    for svc, count in sorted(requested.items()):
        if not isinstance(count, int) or isinstance(count, bool):
            raise ScalingError(f"{svc}: replicas must be an integer, got {count!r}")
        if count < 1:
            # Zero is a valid CR value and a valid thing to want -- for a pool
            # being retired. It is not something to reach through this API by
            # accident, and for a PD role it takes the whole deployment down:
            # dispatch fails closed when either side is empty, so emptying one
            # returns 503 for every request, not just the ones that would have
            # landed there.
            raise ScalingError(
                f"{svc}: refusing to scale to {count}. A pool at zero stops serving, "
                "and in a PD deployment an empty prefill or decode pool fails every "
                "request. Edit the InferaDeployment directly if that is the intent."
            )
