###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Parse client cache-control hints from a chat / completion request body.

We accept two shapes:

1. **Anthropic-style** (`messages[].content[].cache_control`,
   `system[].cache_control`, `tools[].cache_control`)::

       {
         "model": "...",
         "system": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
         "messages": [{"role": "user", "content": [{"type": "text", "text": "...",
                                                    "cache_control": {"type": "ephemeral"}}]}]
       }

   - `cache_control.type == "ephemeral"` with `ttl == "1h"` → ``long``
   - `cache_control.type == "ephemeral"` without ttl (default 5 min) → ``short``
   - No `cache_control` field → ``none``

2. **OpenAI-style** (`prompt_cache_key`, `prompt_cache_retention`)::

       {
         "prompt_cache_key": "<session_id>",
         "prompt_cache_retention": "24h"
       }

   - `prompt_cache_retention` ∈ {``"24h"``, ``"1h"``} → ``long``
   - `prompt_cache_key` present, retention unset or ``"5m"`` → ``short``
   - Both absent → ``none``

The overall request retention is the **maximum** across all blocks
that carry a hint. The session key (Anthropic doesn't have one
directly; OpenAI's is `prompt_cache_key`) is returned separately so
the router can implement sticky routing.

Why parse two formats?
- Most clients today (Anthropic SDK, OpenClaw, Claude Code) emit
  Anthropic-style.
- vLLM and SGLang OpenAI-compat endpoints see OpenAI-style.
- A single router serving both inference endpoints should respect
  whichever the client used.

We do NOT mutate the body in this module — that's the caller's job
(see `engine_priority.py` for translating retention → SGLang's
`priority` field).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Retention(str, Enum):
    """Three retention levels matching OpenClaw's production usage.

    Ordering: NONE < SHORT < LONG (so `max(...)` over a list gives
    the strongest signal, which is what we want).
    """

    NONE = "none"
    SHORT = "short"
    LONG = "long"

    def rank(self) -> int:
        return {"none": 0, "short": 1, "long": 2}[self.value]


@dataclass(frozen=True)
class CacheHints:
    """Result of parsing one request body.

    Fields are read by `KvEventAwarePolicy.pick` (for retention-aware
    weight adjustment) and by the engine adapter (for `priority` field
    injection on SGLang, similar on vLLM).
    """

    retention: Retention
    session_id: str | None
    # Were ANY blocks explicitly tagged? Helps distinguish "implicit short"
    # (no hint at all) from "client said short" (deliberate ephemeral).
    explicit_hint_seen: bool
    # True if the body contains image/audio/video blocks. The router-side
    # block hasher today is text-only (matches what SGLang/vLLM engines
    # emit), so MM requests with the same surrounding tokens but different
    # images would collide → silent KV reuse from the wrong image.
    # When this is True the policy MUST NOT trust cache locality; it falls
    # back to pure load balance.
    has_multimodal_content: bool = False


_NONE = Retention.NONE
_SHORT = Retention.SHORT
_LONG = Retention.LONG

# Default retention for requests carrying NO cache_control hint. Historically
# 'none' (lowest eviction priority → evicted first), which made every request
# without an explicit hint effectively uncacheable — i.e. all traffic through
# the disagg router and all synthetic benchmarks. With the single disk tier,
# retention is only an eviction-priority hint (not a tier selector), so default
# to a retained class and cache uncontrolled traffic like everything else.
# Override with INFERA_ROUTER_DEFAULT_RETENTION={none,short,long}.
try:
    _DEFAULT_RETENTION = Retention(
        os.environ.get("INFERA_ROUTER_DEFAULT_RETENTION", "long").strip().lower()
    )
except ValueError:
    _DEFAULT_RETENTION = _LONG


def effective_retention(retention: Retention) -> Retention:
    """Apply the router's default-retention policy. ``parse_cache_hints``
    reports what the client asked (``NONE`` == no cache_control); this maps
    "no ask" to ``_DEFAULT_RETENTION`` (default ``long``) so uncontrolled
    traffic — everything through the disagg router, all plain benchmarks —
    still caches instead of being stamped 'none' (evict-first). Explicit
    client hints pass through unchanged.
    """
    return retention if retention != _NONE else _DEFAULT_RETENTION


def parse_cache_hints(body: dict[str, Any]) -> CacheHints:
    """Inspect a chat completion / completion request body for cache
    hints. Return the strongest retention level the client asked for
    and a session ID if one is provided.

    Safe to call on bodies that don't carry any hint — returns
    `CacheHints(retention=NONE, session_id=None, explicit_hint_seen=False)`.

    Never raises on malformed bodies (we treat them as no-hint).
    """
    if not isinstance(body, dict):
        return CacheHints(_NONE, None, False, has_multimodal_content=False)

    # OpenAI-style — explicit and cheapest to check.
    openai_retention = _parse_openai_retention(body)
    openai_session = body.get("prompt_cache_key") or body.get("session_id")

    # Anthropic-style — scan messages + system + tools.
    anthropic_retention, anthropic_seen = _scan_anthropic(body)

    # MM detection runs unconditionally — independent of cache_control —
    # because vision/audio inputs taint cache locality regardless of
    # whether the client asked for caching.
    has_mm = _detect_multimodal(body)

    # Take the strongest signal across both encodings. If a request
    # somehow carries both (unusual, but legal — a proxy might add the
    # OpenAI fields to an Anthropic body), the higher wins.
    candidates = []
    if openai_retention is not None:
        candidates.append(openai_retention)
    if anthropic_retention is not None:
        candidates.append(anthropic_retention)

    if not candidates:
        return CacheHints(
            _NONE,
            _coerce_session_id(openai_session),
            False,
            has_multimodal_content=has_mm,
        )

    retention = max(candidates, key=lambda r: r.rank())
    return CacheHints(
        retention=retention,
        session_id=_coerce_session_id(openai_session),
        explicit_hint_seen=anthropic_seen or openai_retention is not None,
        has_multimodal_content=has_mm,
    )


def _coerce_session_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_openai_retention(body: dict[str, Any]) -> Retention | None:
    """OpenAI's `prompt_cache_retention` field. Returns None if absent."""
    retention = body.get("prompt_cache_retention")
    if isinstance(retention, str):
        retention = retention.strip().lower()
        # OpenAI accepts e.g. "24h", "1h"; "5m" is the default-short.
        if retention in {"24h", "1h", "long"}:
            return _LONG
        if retention in {"5m", "short"}:
            return _SHORT
        if retention in {"none", "off", "disabled"}:
            return _NONE
        # Unrecognized value: warn once but don't crash.
        logger.debug("unknown prompt_cache_retention value: %r", retention)
        return None

    # `prompt_cache_key` alone (no retention specified) → implicit short.
    if body.get("prompt_cache_key"):
        return _SHORT

    return None


def _scan_anthropic(body: dict[str, Any]) -> tuple[Retention | None, bool]:
    """Find any `cache_control` block in `messages`, `system`, `tools`.

    Returns (retention, explicit_hint_seen) where retention is the max
    across all blocks (None if no block carries cache_control).
    """
    found: list[Retention] = []

    for block in _iter_blocks_with_cache_control(body):
        retention = _retention_for_block(block)
        if retention is not None:
            found.append(retention)

    if not found:
        return None, False
    return max(found, key=lambda r: r.rank()), True


def _iter_blocks_with_cache_control(body: dict[str, Any]):
    """Yield every dict that has a `cache_control` key, walking:
    - `system` (array of content blocks, or a plain string skipped)
    - `tools` (array of tool defs; cache_control at top level)
    - `messages[].content` (array of content blocks; cache_control per block)

    Doesn't recurse into nested content arbitrarily — Anthropic's schema
    is shallow.
    """
    # system blocks
    system = body.get("system")
    if isinstance(system, list):
        for entry in system:
            if isinstance(entry, dict) and "cache_control" in entry:
                yield entry

    # tools
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and "cache_control" in tool:
                yield tool

    # message content blocks
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        yield block


# ----------------------------------------------------------------------
# Multimodal detection
# ----------------------------------------------------------------------

# Content-block types that signal non-text input. We're permissive: a
# request that mentions any of these is treated as MM, even if the
# block is malformed. False positives only cost a cache-locality
# downgrade (load-balance fallback); false negatives risk silent
# corruption from the same-text-different-image cache collision.
_MM_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        # Anthropic / Claude
        "image",
        # OpenAI vision
        "image_url",
        # OpenAI audio input
        "input_audio",
        # Anthropic document / OpenAI file content
        "document",
        "file",
        # Forward-compat: anything starting with these prefixes is MM
        # — Anthropic adds new types as they ship vision/audio.
        "video",
        "audio",
    }
)


def _detect_multimodal(body: dict[str, Any]) -> bool:
    """Return True if the request carries any non-text content block.

    Why: the router-side block hasher (`infera.router.kv_event.hasher`)
    is text-only — it hashes post-tokenize integer IDs. SGLang and vLLM
    today emit matching text-only hashes from their workers. For a
    vision request, the image-placeholder token (`<|image|>`) is the
    SAME id regardless of which image was sent — so two requests with
    the same surrounding tokens but DIFFERENT images produce the same
    block hash and would silently share KV. That's wrong KV reuse, the
    most insidious caching bug.

    Until both router and engine adopt Dynamo-style MM-aware hashing,
    the safe play is: detect MM at the router edge and
    route by load only, ignoring cache locality. Documented as
    Phase 4.7(b) in PR #9.

    Detection is intentionally permissive — we'd rather over-detect
    and lose a cache hit than miss a vision request and serve wrong KV.
    """
    # Anthropic shape: messages[].content[] is a list of blocks.
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if isinstance(btype, str) and btype in _MM_BLOCK_TYPES:
                    return True

    # OpenAI shape (legacy): top-level `images` or `audio` field on the body.
    if isinstance(body.get("images"), list) and body["images"]:
        return True
    if isinstance(body.get("audio"), (dict, list)) and body["audio"]:
        return True

    # Anthropic top-level `system` may also contain images on newer models.
    system = body.get("system")
    if isinstance(system, list):
        for entry in system:
            if not isinstance(entry, dict):
                continue
            btype = entry.get("type")
            if isinstance(btype, str) and btype in _MM_BLOCK_TYPES:
                return True

    return False


def _retention_for_block(block: dict[str, Any]) -> Retention | None:
    """Translate one Anthropic-style `cache_control` value to a retention
    level. Block types `thinking` / `redacted_thinking` are NEVER cached
    even if some buggy client attached cache_control to them — matches
    OpenClaw's defensive stripping (see openclaw
    src/agents/anthropic-payload-policy.ts).
    """
    block_type = block.get("type")
    if block_type in {"thinking", "redacted_thinking"}:
        return _NONE

    cc = block.get("cache_control")
    if not isinstance(cc, dict):
        return None

    cc_type = cc.get("type")
    if cc_type != "ephemeral":
        # Anthropic only defines "ephemeral" today; anything else is
        # unknown — don't speculate.
        return None

    ttl = cc.get("ttl")
    if isinstance(ttl, str):
        ttl_norm = ttl.strip().lower()
        if ttl_norm in {"1h", "1hr", "60m", "3600s", "long"}:
            return _LONG
        # "5m" or any short TTL → short
        if ttl_norm in {"5m", "5min", "300s", "short"}:
            return _SHORT

    # ephemeral with no ttl: Anthropic's default is 5 min → short.
    return _SHORT
