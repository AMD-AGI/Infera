###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Policy registry: add a builder, add it to _BUILDERS, done.

Builders take ``**kwargs`` and pull only what they need so the server can
pass policy-specific knobs uniformly via ``build_policy(name, **kwargs)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.render_variant import RenderVariant, VariantRegistry
from infera.router.policy.base import Policy
from infera.router.policy.kv_event_aware import KvEventAwarePolicy
from infera.router.policy.round_robin import RoundRobinPolicy


def _build_round_robin(**_: Any) -> Policy:
    return RoundRobinPolicy()


def _build_kv_aware(
    *,
    overlap_weight: float = 1.0,
    prefill_overlap_weight: float | None = None,
    decode_overlap_weight: float | None = None,
    tokenizer_path: str | None = None,
    kv_event_transport: str = "zmq",
    nats_server: str | None = None,
    default_chat_template_kwargs: str | None = None,
    per_worker_template_kwargs: bool = True,
    **_: Any,
) -> Policy:
    # Transport for the per-worker KV-event view: direct ZMQ (one SUB per
    # worker) or a shared NATS broker subscription. Both feed the identical
    # KvEventAwarePolicy cache-view logic.
    if kv_event_transport == "nats":
        from infera.router.kv_event.nats_client import NatsKvEventClient

        kv_client = NatsKvEventClient(nats_server)
    else:
        kv_client = KvEventClient()
    return KvEventAwarePolicy(
        kv_client,
        BlockHasher(tokenizer_path=tokenizer_path),
        overlap_weight=overlap_weight,
        prefill_overlap_weight=prefill_overlap_weight,
        decode_overlap_weight=decode_overlap_weight,
        variants=VariantRegistry(
            _parse_template_kwargs(default_chat_template_kwargs),
            enabled=per_worker_template_kwargs,
        ),
    )


def _parse_template_kwargs(raw: str | None) -> RenderVariant:
    """``--kv-default-chat-template-kwargs`` as the engine would take it.

    Rejects rather than shrugs. A typo here does not fail anything at runtime:
    the router would just keep rendering the preamble the workers do not render,
    which is the exact silent failure this flag exists to fix -- so it is worth
    refusing to start over.
    """
    if not raw:
        return RenderVariant()
    import json

    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"--kv-default-chat-template-kwargs is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "--kv-default-chat-template-kwargs must be a JSON object (sglang's own "
            f"--default-chat-template-kwargs is a dict), got {type(parsed).__name__}"
        )
    return RenderVariant(parsed)


_BUILDERS: dict[str, Callable[..., Policy]] = {
    "round-robin": _build_round_robin,
    "kv-aware": _build_kv_aware,
}

POLICY_NAMES = sorted(_BUILDERS.keys())


def build_policy(name: str, **kwargs: Any) -> Policy:
    if name not in _BUILDERS:
        raise ValueError(f"unknown router policy {name!r}; available: {POLICY_NAMES}")
    return _BUILDERS[name](**kwargs)
