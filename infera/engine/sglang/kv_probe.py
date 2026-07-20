###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang-specific KV cache probe wiring.

Hooks SGLang's `RadixCache.insert` and `RadixCache.evict` at the
**node level** (not at the per-page `_record_store_event` callbacks).
This is what makes `--page-size 1` (ROCm + AITER) efficient: each
RadixCache.insert is one Python dispatch carrying all the tokens of
the inserted node; the probe iterates inside that call.

The actual SGLang hook is a thin monkey-patch of the RadixCache
instance. The patch reads the inserted/evicted node's full prefix
tokens by walking up the trie, then hands them to a `KvEventProbe`.
Tests against a `MockRadixCache` (see `tests/unit/kv/test_probe.py`)
validate the wiring without requiring SGLang to be installed.
"""

from __future__ import annotations

import logging
from typing import Any

from infera.kv.probe import KvEventProbe

logger = logging.getLogger(__name__)


def _walk_full_prefix_tokens(node: Any) -> list[int]:
    """Walk up the RadixCache trie from `node` to the root, collecting
    tokens. Returns the full token sequence representing the path.

    Expects `node` to have:
      - `node.key` — the token id list for this node's segment
      - `node.parent` — the parent node, or None at root

    SGLang's `TreeNode` matches this shape. If a different engine uses
    different attribute names, write a wrapper or subclass.
    """
    chain: list[list[int]] = []
    cur = node
    while cur is not None and getattr(cur, "key", None):
        chain.append(list(cur.key))
        cur = getattr(cur, "parent", None)
    # We collected leaf-to-root, so reverse to get root-to-leaf order.
    out: list[int] = []
    for segment in reversed(chain):
        out.extend(segment)
    return out


def attach_to_radix_cache(probe: KvEventProbe, radix_cache: Any) -> dict[str, Any]:
    """Monkey-patch `radix_cache` to invoke `probe.on_node_inserted` /
    `on_node_evicted` after the matching operations.

    Returns a dict of the originals so callers can `detach_from_radix_cache`
    to restore. The probe and cache must outlive this attachment.

    Hook strategy:
      - We wrap `insert(...)` (or whichever method actually inserts).
        The wrapped function calls the original, then walks the trie
        to compute the full prefix tokens of the new node and fires
        the probe.
      - Same shape for eviction.

    Real-world note: SGLang's RadixCache API evolves. The exact method
    names and the way to obtain the just-inserted node from the return
    value may need adjustment per SGLang version. The function below is
    structured so the SGLang-specific parts are isolated and easy to
    update without touching the probe code.
    """
    originals: dict[str, Any] = {}

    if hasattr(radix_cache, "insert"):
        original_insert = radix_cache.insert

        def wrapped_insert(*args, **kwargs):
            result = original_insert(*args, **kwargs)
            try:
                # SGLang `RadixCache.insert(token_ids, ...)` may return the
                # inserted node, a (node, prefix_len) tuple, or None depending
                # on version. The simplest portable behavior is to recover the
                # full token sequence from the *call argument*, which is
                # always the token_ids being inserted.
                token_ids = _extract_insert_tokens(args, kwargs)
                if token_ids:
                    probe.on_node_inserted(full_prefix_tokens=list(token_ids))
            except Exception:
                logger.exception("KvEventProbe.on_node_inserted hook raised")
            return result

        radix_cache.insert = wrapped_insert
        originals["insert"] = original_insert

    if hasattr(radix_cache, "evict"):
        original_evict = radix_cache.evict

        def wrapped_evict(*args, **kwargs):
            result = original_evict(*args, **kwargs)
            try:
                # SGLang `RadixCache.evict` typically returns the evicted
                # node or a list of evicted nodes. Walk each one to get the
                # full prefix tokens.
                for node in _iter_evicted_nodes(result):
                    tokens = _walk_full_prefix_tokens(node)
                    if tokens:
                        probe.on_node_evicted(full_prefix_tokens=tokens)
            except Exception:
                logger.exception("KvEventProbe.on_node_evicted hook raised")
            return result

        radix_cache.evict = wrapped_evict
        originals["evict"] = original_evict

    if hasattr(radix_cache, "reset"):
        # Some SGLang versions have `reset()` to clear the whole cache.
        original_reset = radix_cache.reset

        def wrapped_reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            try:
                probe.on_clear()
            except Exception:
                logger.exception("KvEventProbe.on_clear hook raised")
            return result

        radix_cache.reset = wrapped_reset
        originals["reset"] = original_reset

    return originals


def detach_from_radix_cache(originals: dict[str, Any], radix_cache: Any) -> None:
    """Restore the original RadixCache methods."""
    for name, fn in originals.items():
        if hasattr(radix_cache, name):
            setattr(radix_cache, name, fn)


def _extract_insert_tokens(args, kwargs) -> list[int]:
    """SGLang's `RadixCache.insert(token_ids, value=None, ...)` —
    extract the first positional arg or the `token_ids` kwarg.
    """
    if args:
        candidate = args[0]
    else:
        candidate = kwargs.get("token_ids") or kwargs.get("key")
    if candidate is None:
        return []
    return list(candidate)


def _iter_evicted_nodes(evict_result) -> list[Any]:
    """Normalize the return of `radix_cache.evict(...)` to an iterable
    of node objects. SGLang versions differ: some return a single node,
    some a list, some an int count and require side-effect tracking.
    """
    if evict_result is None:
        return []
    if isinstance(evict_result, list):
        # List of nodes (or list of (node, ...) tuples).
        nodes: list[Any] = []
        for item in evict_result:
            if isinstance(item, tuple) and item:
                nodes.append(item[0])
            elif hasattr(item, "key"):
                nodes.append(item)
        return nodes
    if hasattr(evict_result, "key"):
        return [evict_result]
    return []
