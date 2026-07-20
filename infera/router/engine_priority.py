###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Translate `CacheHints` to engine-specific priority + retention fields.

Different engines accept different fields:

- **SGLang**: `priority: int` on the request body (see
  `sglang/srt/managers/io_struct.py:225`). With
  `--radix-eviction-policy priority`, the radix cache evicts
  lower-priority blocks first. Higher = retained longer. We ALSO
  stash a `infera_retention` string in the body — SGLang ignores
  it, but a worker-side middleware (see `engine/sglang/kvd_adapter.
  set_request_retention_hint`) can extract it and set the per-
  request ContextVar so the kvd write inherits the right tier.
- **vLLM**: passes through `kv_transfer_params: dict[str, Any]` via
  ``SamplingParams.extra_args`` (see ``vllm/v1/request.py``). Our
  connector reads ``kv_transfer_params["infera_retention"]`` in
  ``build_connector_meta`` and stamps the value into the
  ``packed_blocks_to_save`` retention slot — that's the seam that
  actually drives long vs short SET behavior in kvd.

## Asymmetry note (PR #9 review fix P0-3/4)

vLLM has a natural in-band channel (`kv_transfer_params`) that the
engine plumbs to our connector. SGLang doesn't — `HiCacheStorage.set`
sees no request context. Until upstream populates
`batch_set_v1.extra_info["infera_retention"]`, the SGLang side
needs an explicit worker-side middleware to bridge body →
ContextVar → adapter. We stash the retention in the body here; the
middleware is the SGLang worker's responsibility.

The mutation is idempotent and additive — we only set the field if
it isn't already present, so a deliberate per-request override from
the client wins over our default.

We pick integer priority values that fit SGLang's expected range
(default 0; higher = keep longer). The exact numbers are tunable —
production may want to differentiate further (e.g. tools vs. system
prompt) but for v1 three buckets are enough.

For vLLM the JSON value is the retention STRING (``"long"`` /
``"short"`` / ``"none"``) — matches `infera.kvd.wire.RETENTION_*`
constants so the connector can hand it straight to ``kvd.put``.
"""

from __future__ import annotations

from typing import Any

from infera.common.worker_pool import EngineType
from infera.router.cache_control import CacheHints, Retention, effective_retention

# Priority levels (higher = retained longer under PriorityStrategy).
# Spread numerically so future per-block tweaks have headroom.
_SGLANG_PRIORITY = {
    Retention.NONE: 0,
    Retention.SHORT: 50,
    Retention.LONG: 100,
}


def inject_engine_priority(
    body: dict[str, Any],
    hints: CacheHints,
    engine: EngineType,
) -> dict[str, Any]:
    """Return a NEW body dict with engine-specific priority field set
    from `hints`. Does not mutate `body` in place — caller can swap
    references safely. The dict is shallow-copied; nested message
    arrays are NOT copied (engines don't mutate them).

    A client-provided `priority` field always wins over our default.
    """
    out = dict(body)

    # Apply the router default policy: a request with no explicit cache_control
    # (retention == NONE) is stamped with the configured default (long) instead
    # of 'none', so uncontrolled traffic still caches. Explicit hints pass
    # through. retention is now an eviction-priority hint (single disk tier).
    retention = effective_retention(hints.retention)

    if engine == EngineType.SGLANG:
        if "priority" not in out:
            out["priority"] = _SGLANG_PRIORITY[retention]
        # Stash the retention string too — body field that any worker-
        # side middleware can read to bridge to the kvd ContextVar.
        # See `engine/sglang/kvd_adapter.set_request_retention_hint`.
        if "infera_retention" not in out:
            out["infera_retention"] = retention.value
    elif engine == EngineType.VLLM:
        # vLLM in-band: stash retention under `kv_transfer_params`
        # (which the engine plumbs to Request.kv_transfer_params, our
        # connector reads it in build_connector_meta).
        existing = out.get("kv_transfer_params") or {}
        if not isinstance(existing, dict):
            existing = {}
        if "infera_retention" not in existing:
            existing = {**existing, "infera_retention": retention.value}
            out["kv_transfer_params"] = existing
        # Legacy compat: the previous shape was a top-level
        # `_infera_retention` field. Some test harnesses still look
        # for it. Keep until next refactor.
        if "_infera_retention" not in out:
            out["_infera_retention"] = retention.value
    # Other engines: no-op until we add support.

    return out
