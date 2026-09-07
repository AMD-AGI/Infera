###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The server-side template defaults a worker was launched with.

``--default-chat-template-kwargs`` is the one input to the engine's render that
the router is never told about. The client does not send it, discovery does not
carry it, and it is applied *before* the template runs -- so a router that does
not model it renders a different preamble than the worker, for every request,
forever. We have shipped exactly that: on ``infera-glm53-pd-1p1d-stable`` role1
the engine held ``{"reasoning_effort": "high"}`` and the router rendered
``Max``, diverging at token 8 of 13.

Its signature is nastier than a flat zero hit rate. The merge is a
``setdefault``, so a request that *does* carry ``reasoning_effort`` agrees with
the engine and hits normally -- only the requests that omit it diverge. The
symptom is a hit rate that is merely lower than it should be, which is why this
one hid for as long as it did.

A variant is not per worker, it is per launch configuration: every worker in a
role shares one. A fleet with a single configuration has a single variant,
renders once, and pays nothing for this machinery.

Mirror of ``rust/router/src/render_variant.rs``. Two ports of one contract; the
merge semantics below are the same six lines of ``serving_chat.py`` in both.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from xxhash import xxh3_64_intdigest

logger = logging.getLogger(__name__)


class RenderVariant:
    """The template-scope defaults one group of workers renders with.

    The default instance is the empty variant: no server-side defaults, the body
    renders as the client sent it. That is what every worker gets until
    something tells the router otherwise, and it is exactly today's behaviour.
    """

    __slots__ = ("_kwargs", "_id")

    def __init__(self, kwargs: dict[str, Any] | None = None) -> None:
        # Sorted at construction so identity does not depend on how the engine
        # happened to serialise its dict.
        self._kwargs: dict[str, Any] = dict(sorted((kwargs or {}).items()))
        self._id = self._compute_id()

    @classmethod
    def from_default_chat_template_kwargs(cls, value: Any) -> RenderVariant:
        """From ``--default-chat-template-kwargs`` as the engine reports it.

        A non-dict (or an absent value) is the empty variant -- sglang's own
        ``server_args`` validation rejects a non-dict, so there is nothing here
        to be strict about that the engine has not already refused.
        """
        return cls(value if isinstance(value, dict) else None)

    def _compute_id(self) -> int:
        if not self._kwargs:
            return 0
        buf = bytearray()
        for k, v in self._kwargs.items():
            buf += k.encode()
            buf += b"\0"
            buf += json.dumps(v, separators=(",", ":"), sort_keys=True).encode()
            buf += b"\0"
        # Never collide with the empty variant, whatever the digest says.
        #
        # This is a process-local cache key, not a wire value: nothing compares
        # it against the Rust router's id or persists it, so the two ports are
        # free to disagree on a nested-object kwarg's serialisation.
        return xxh3_64_intdigest(bytes(buf)) | 1

    @property
    def id(self) -> int:
        """Stable id, so the policy can hash a request once per *variant*
        rather than once per worker. The empty variant is 0 by construction: a
        fleet where nothing is configured collapses to a single key and the hash
        cache behaves exactly as it did before variants existed.
        """
        return self._id

    def is_empty(self) -> bool:
        return not self._kwargs

    def label(self) -> str:
        """A one-line rendering for logs and metrics labels."""
        if not self._kwargs:
            return "default"
        return ",".join(
            f"{k}={json.dumps(v, separators=(',', ':'), sort_keys=True)}"
            for k, v in self._kwargs.items()
        )

    def apply(self, body: dict) -> dict:
        """The body the engine will actually template, given these defaults.

        Mirrors ``serving_chat.py:1026-1032`` exactly, and the two details that
        look incidental are the whole contract:

        * ``setdefault``, not assignment -- an explicit ``chat_template_kwargs``
          from the client wins over the server default. Overwriting instead
          would make the router diverge on precisely the requests that agree
          today.
        * the merged ``reasoning_effort`` is promoted back onto the top-level
          field when the client left it unset, because downstream (the
          ``effort_kwarg`` remap, ``extra_template_kwargs``) reads it from
          there. That promotion happens BEFORE the low/medium/high handling, so
          applying the variant late would silently skip it.

        Rewrites the body rather than patching the template context, so the
        native encoders -- which model ``chat_template_kwargs`` themselves and
        never see that context -- get the merge too.

        Returns `body` itself for the empty variant, which is the common path.
        """
        if not self._kwargs or not isinstance(body, dict):
            return body
        ctk = body.get("chat_template_kwargs")
        ctk = dict(ctk) if isinstance(ctk, dict) else {}
        for k, v in self._kwargs.items():
            ctk.setdefault(k, v)
        out = dict(body)
        if out.get("reasoning_effort") is None and ctk.get("reasoning_effort") is not None:
            out["reasoning_effort"] = ctk["reasoning_effort"]
        out["chat_template_kwargs"] = ctk
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RenderVariant) and other._kwargs == self._kwargs

    def __hash__(self) -> int:
        return self._id

    def __repr__(self) -> str:
        return f"RenderVariant({self.label()})"


EMPTY_VARIANT = RenderVariant()


class VariantRegistry:
    """Which variant each worker renders with.

    Two tiers on purpose. ``fleet`` comes from
    ``--kv-default-chat-template-kwargs`` and applies to everything;
    ``per_worker`` is what the worker itself reported from ``/get_server_info``
    and wins where we have it. The fallback direction matters: a worker we have
    not been able to ask keeps rendering the way the router rendered for it
    before any of this existed, so nothing regresses on an engine without that
    endpoint.

    Populated by the render probe, which already visits every worker once at
    registration.
    """

    def __init__(self, fleet: RenderVariant | None = None, *, enabled: bool = True) -> None:
        self._fleet = fleet or EMPTY_VARIANT
        self._per_worker: dict[str, RenderVariant] = {}
        self._enabled = enabled
        self._lock = threading.Lock()

    @property
    def per_worker_enabled(self) -> bool:
        return self._enabled

    @property
    def fleet(self) -> RenderVariant:
        return self._fleet

    def for_worker(self, worker_id: str) -> RenderVariant:
        """This worker's variant: its own if we know it, the fleet default if
        not."""
        if not self._enabled:
            return self._fleet
        with self._lock:
            return self._per_worker.get(worker_id, self._fleet)

    def record(self, worker_id: str, variant: RenderVariant) -> None:
        """Record what a worker reported.

        Logs only when it changes the answer, because the interesting event is a
        fleet that is *not* uniform -- a fleet that is uniform should be silent.
        """
        if variant != self._fleet:
            logger.warning(
                "kv-aware: worker %s renders with server-side template defaults (%s) that "
                "differ from the router's (%s). Requests for it are now hashed its way; if "
                "--kv-per-worker-template-kwargs is off, or this router is older than the "
                "worker, every lookup for it misses instead",
                worker_id,
                variant.label(),
                self._fleet.label(),
            )
        with self._lock:
            self._per_worker[worker_id] = variant

    def forget(self, worker_id: str) -> None:
        """Drop a recorded per-worker variant so the next probe re-reads it."""
        with self._lock:
            self._per_worker.pop(worker_id, None)

    def retain(self, alive) -> None:
        """Forget workers that left the fleet."""
        with self._lock:
            for wid in [w for w in self._per_worker if not alive(w)]:
                del self._per_worker[wid]

    # How many distinct variants the fleet runs is the number worth watching
    # before trusting any of this -- 1 means the per-worker tier was never
    # needed, 2+ means no single fleet-wide flag could have been right. It is
    # the label cardinality of `infera_router_render_variant`, so it is read
    # off the metric rather than counted here.
