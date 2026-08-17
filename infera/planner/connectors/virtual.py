###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Publish scaling decisions to etcd for an external executor to carry out.

The planner knows how many replicas each pool needs; it does not have to be the
thing that starts them. This connector writes the decision to a well-known etcd
key and stops there, which covers the deployments Infera runs outside
Kubernetes -- bare metal, Docker, a bespoke scheduler -- and makes the planner
testable against nothing more than an etcd.

The consumer watches ``<prefix>/planner/decision`` and reads::

    {
      "num_prefill_workers": 3,
      "num_decode_workers": 5,
      "decision_id": 42,
      "observed_prefill": 2,
      "observed_decode": 5,
      "gpu_budget_exceeded": false
    }

``decision_id`` increments on every write, so a consumer can tell a fresh
decision from a re-read of the same one. Counts are absolute: applying only the
newest decision is always correct, and a consumer that misses one has nothing
to catch up on.

etcd is reached through its v3 JSON gateway over httpx, matching
:mod:`infera.common.registration` -- no extra client library.
"""

from __future__ import annotations

import json
import logging

import httpx

from infera.common.discovery import _b64, _normalize_endpoint
from infera.planner.decision import ScalingDecision

logger = logging.getLogger(__name__)

DECISION_SUFFIX = "planner/decision"


class VirtualConnector:
    """Writes the decision to etcd; something else does the scaling."""

    def __init__(
        self,
        *,
        etcd_endpoint: str,
        prefix: str = "/infera/workers",
        timeout: float = 10.0,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=_normalize_endpoint(etcd_endpoint), timeout=timeout)
        base = prefix if prefix.endswith("/") else prefix + "/"
        self.key = base + DECISION_SUFFIX
        self._decision_id = 0

    async def aclose(self) -> None:
        await self._http.aclose()

    def payload_for(self, decision: ScalingDecision, decision_id: int) -> dict:
        return {
            "num_prefill_workers": decision.num_prefill,
            "num_decode_workers": decision.num_decode,
            "decision_id": decision_id,
            "observed_prefill": decision.observed_prefill,
            "observed_decode": decision.observed_decode,
            "gpu_budget_exceeded": decision.gpu_budget_exceeded,
        }

    async def apply(self, decision: ScalingDecision) -> None:
        self._decision_id += 1
        payload = self.payload_for(decision, self._decision_id)
        try:
            resp = await self._http.post(
                "/v3/kv/put",
                json={"key": _b64(self.key), "value": _b64(json.dumps(payload))},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Leave _decision_id advanced: the next successful write must look
            # newer than anything a consumer may have already seen.
            logger.warning(
                "could not publish decision %d to %s: %s: %s",
                self._decision_id,
                self.key,
                type(exc).__name__,
                exc,
            )
            return
        logger.info(
            "published decision %d to %s: %s", self._decision_id, self.key, decision.summary()
        )
