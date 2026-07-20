###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM KV cache event publishing for KV-aware routing.

ATOM has no native KV cache event stream (unlike vLLM's
``--kv-events-config`` or SGLang's RadixCache events). But its
``BlockManager`` already maintains a chained-hash prefix-cache index:

* ``hash_blocks()`` registers a content hash for every block whose KV the
  forward pass just finalized — this is exactly a *BlockStored*.
* ``_allocate_block()`` evicts a stale ``hash_to_block_id`` entry when it
  reuses a block for new content — this is exactly a *BlockRemoved*.

This module taps those two points by monkey-patching ``BlockManager`` and
re-publishes them on a ZMQ PUB socket in the **same** msgpack wire format
the infera router already consumes from vLLM/SGLang
(:mod:`infera.router.kv_event.events`). The router's ``KvEventClient``
re-hashes the reported ``token_ids`` into its own hash space and uses the
engine ``block_hashes`` only as opaque translation keys, so no router-side
change is needed — ATOM workers light up KV-aware routing exactly like
vLLM/SGLang ones.

The hooks are installed inside the ATOM ``EngineCore`` subprocess (which is
where ``BlockManager`` actually lives) via
:mod:`infera.engine.atom.hooks.kv_event_bootstrap`. The PUB socket binds
lazily on the first ``BlockManager`` instantiation, so only the process that
owns the block manager binds the port.
"""

from __future__ import annotations

import importlib.machinery
import inspect
import logging
import os
import sys
import time
import traceback

logger = logging.getLogger(__name__)

_TOPIC = b"kv-events"
_BLOCK_MANAGER_MODULE = "atom.model_engine.block_manager"

# Private ATOM symbols the hooks depend on. If an ATOM image bump renames or
# drops any of these, we detect it at install time and degrade loudly instead
# of corrupting the event stream. Keep this list in sync with the hook bodies.
_REQUIRED_METHODS = ("__init__", "hash_blocks", "_allocate_block")
# Set INFERA_ATOM_KV_EVENTS_STRICT=1 to RAISE on API drift (fail the worker)
# instead of degrading to round-robin — useful in CI / image-promotion smoke
# tests to catch an incompatible ATOM bump before it ships.
_STRICT = os.environ.get("INFERA_ATOM_KV_EVENTS_STRICT", "") not in ("", "0", "false")


def _atom_version() -> str:
    """Best-effort ATOM version string for logs (helps post-mortem an upgrade)."""
    try:
        import atom

        v = getattr(atom, "__version__", None)
        if v:
            return str(v)
        from atom import _version  # type: ignore

        return str(getattr(_version, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _check_block_manager_compat(BlockManager) -> str | None:
    """Verify the ATOM ``BlockManager`` API matches what the hooks expect.

    Returns ``None`` when compatible, else a short human-readable reason. This
    is the single chokepoint that makes an ATOM image bump safe: the hooks
    read private attributes (``blocks``, ``hash_to_block_id``, ``block.hash``,
    ``seq.num_cached_tokens`` ...) that can't all be checked statically, but
    the *method surface* and its signature can, which catches the common
    rename/refactor breakages early and clearly.
    """
    for name in _REQUIRED_METHODS:
        if not callable(getattr(BlockManager, name, None)):
            return f"BlockManager.{name} missing or not callable"
    # hash_blocks(self, seq, num_new_tokens): expect self + 2 positional params.
    try:
        params = [
            p
            for p in inspect.signature(BlockManager.hash_blocks).parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return "BlockManager.hash_blocks signature not introspectable"
    if len(params) < 3:  # self, seq, num_new_tokens
        names = [p.name for p in params]
        return f"BlockManager.hash_blocks signature changed (params={names})"
    return None


class _Publisher:
    """Thin ZMQ PUB wrapper that encodes batches with the router's msgspec
    structs (guaranteeing byte-for-byte wire compatibility) and binds lazily.

    All publishing happens from the single EngineCore scheduler thread that
    drives ``BlockManager``, so a plain (non-async, non-thread-safe) PUB
    socket is safe.
    """

    def __init__(self, bind_endpoint: str) -> None:
        self._bind_endpoint = bind_endpoint
        self._sock = None
        self._encoder = None
        self._events_mod = None

    def ensure_bound(self) -> None:
        if self._sock is not None:
            return
        import msgspec.msgpack
        import zmq

        from infera.router.kv_event import events as events_mod

        self._events_mod = events_mod
        self._encoder = msgspec.msgpack.Encoder()
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        # Mirror vLLM's ZmqEventPublisher: a small send HWM is fine because
        # the router rebuilds its view from a fresh subscription if it ever
        # falls behind; we never want a slow subscriber to block the engine.
        sock.setsockopt(zmq.SNDHWM, 100_000)
        sock.bind(self._bind_endpoint)
        self._sock = sock
        logger.info("ATOM kv-events: PUB bound at %s", self._bind_endpoint)

    def _send_batch(self, events: list) -> None:
        if self._sock is None or not events:
            return
        batch = self._events_mod.KVEventBatch(ts=time.time(), events=events)
        try:
            payload = self._encoder.encode(batch)
            self._sock.send_multipart([_TOPIC, payload])
        except Exception:
            logger.exception("ATOM kv-events: failed to publish batch")

    def block_stored(
        self, blocks: list[tuple[int, int | None, list[int]]], block_size: int
    ) -> None:
        """``blocks`` is a list of ``(block_hash, parent_hash, token_ids)``;
        ``parent_hash`` is ``None`` for the first block in a chain."""
        if self._sock is None:
            return
        BlockStored = self._events_mod.BlockStored
        evs = [
            BlockStored(
                block_hashes=[bh],
                parent_block_hash=ph,
                token_ids=list(tokens),
                block_size=block_size,
                lora_id=None,
            )
            for (bh, ph, tokens) in blocks
        ]
        self._send_batch(evs)

    def block_removed(self, block_hash: int) -> None:
        if self._sock is None:
            return
        BlockRemoved = self._events_mod.BlockRemoved
        self._send_batch([BlockRemoved(block_hashes=[block_hash])])


def _patch_block_manager(BlockManager, publisher: _Publisher) -> None:
    """Wrap ``BlockManager`` methods to publish KV cache events.

    Idempotent. The PUB socket only binds when a ``BlockManager`` is actually
    instantiated (i.e. inside the EngineCore subprocess), so this is safe to
    apply in any process that imports the class.
    """
    if getattr(BlockManager, "_infera_kv_patched", False):
        return

    # --- Compatibility preflight: an ATOM image bump that changes the private
    # BlockManager API must NOT silently corrupt the event stream. Detect drift
    # here and degrade to round-robin (or fail-fast under STRICT) with a clear,
    # single message telling the operator to update this module.
    version = _atom_version()
    reason = _check_block_manager_compat(BlockManager)
    if reason is not None:
        msg = (
            f"ATOM kv-events DISABLED: BlockManager API drift on atom=={version} "
            f"({reason}). KV-aware routing falls back to round-robin for ATOM "
            f"workers. Update infera/engine/atom/hooks/kv_events.py to match "
            f"this ATOM version."
        )
        if _STRICT:
            raise RuntimeError(msg)
        logger.error(msg)
        return
    logger.info("ATOM kv-events: BlockManager API compatible (atom==%s)", version)

    # One-shot runtime guard: if a hook raises on a live request (e.g. an
    # attribute rename the static preflight can't see), disable publishing once
    # with a clear message instead of spamming a traceback every request.
    state = {"runtime_failed": False}

    def _disable(where: str) -> None:
        state["runtime_failed"] = True
        logger.exception(
            "ATOM kv-events: %s hook failed on a live request (atom==%s) — "
            "disabling KV event publishing for this worker; router falls back "
            "to round-robin. Likely ATOM BlockManager API drift.",
            where,
            version,
        )

    orig_init = BlockManager.__init__
    orig_hash_blocks = BlockManager.hash_blocks
    orig_allocate_block = BlockManager._allocate_block

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Bind in the process that owns a BlockManager (the EngineCore).
        publisher.ensure_bound()
        self._infera_kv_pub = publisher

    def patched_hash_blocks(self, seq, num_new_tokens):
        # Never let the hook change ATOM's own behaviour: run the original
        # first, then (best-effort) publish. On the first failure, disable.
        if state["runtime_failed"] or not getattr(self, "enable_prefix_caching", False):
            return orig_hash_blocks(self, seq, num_new_tokens)
        # Compute the [start, end) block range exactly as the original does,
        # BEFORE calling it (the original mutates block hashes but not
        # seq.num_cached_tokens — the scheduler advances that afterwards).
        try:
            bs = self.block_size
            start = seq.num_cached_tokens // bs
            end = (seq.num_cached_tokens + num_new_tokens) // bs
        except Exception:
            ret = orig_hash_blocks(self, seq, num_new_tokens)
            _disable("block_stored(range)")
            return ret
        ret = orig_hash_blocks(self, seq, num_new_tokens)
        if start >= end:
            return ret
        try:
            blocks = []
            for i in range(start, end):
                block = self.blocks[seq.block_table[i]]
                # Parent is the previous block's (already-stored) hash; None
                # for the very first block so the router seeds from ROUTER_SEED.
                parent = self.blocks[seq.block_table[i - 1]].hash if i > 0 else None
                blocks.append((block.hash, parent, block.token_ids))
            publisher.block_stored(blocks, bs)
        except Exception:
            _disable("block_stored")
        return ret

    def patched_allocate_block(self, block_id):
        if state["runtime_failed"]:
            return orig_allocate_block(self, block_id)
        # Mirror the original's eviction condition so we report the hash that
        # is about to be dropped from hash_to_block_id.
        try:
            block = self.blocks[block_id]
            evicted = (
                block.hash
                if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id
                else None
            )
        except Exception:
            ret = orig_allocate_block(self, block_id)
            _disable("block_removed(detect)")
            return ret
        ret = orig_allocate_block(self, block_id)
        if evicted is not None:
            try:
                publisher.block_removed(evicted)
            except Exception:
                _disable("block_removed")
        return ret

    BlockManager.__init__ = patched_init
    BlockManager.hash_blocks = patched_hash_blocks
    BlockManager._allocate_block = patched_allocate_block
    BlockManager._infera_kv_patched = True
    logger.info("ATOM kv-events: BlockManager hooks installed (bind=%s)", publisher._bind_endpoint)


class _DeferredPatchFinder:
    """``sys.meta_path`` finder that patches ``BlockManager`` the moment ATOM
    imports its module — without importing ATOM ourselves.

    We cannot import ``atom`` (→ torch → aiter) from a site ``.pth`` /
    ``sitecustomize`` hook: doing heavy imports during interpreter ``site``
    initialization deadlocks. So instead of importing the block manager at
    startup, we install this dormant finder; it fires only when the ATOM
    EngineCore naturally imports ``atom.model_engine.block_manager``, by which
    point ``site`` init is long done.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._done = False

    def find_spec(self, fullname, path, target=None):
        if self._done or fullname != _BLOCK_MANAGER_MODULE:
            return None
        # Locate the real module using the parent package's path (no atom
        # re-import), then wrap its loader to patch right after exec.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        self._done = True
        endpoint = self._endpoint
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module, _orig=orig_exec, _ep=endpoint):
            _orig(module)
            try:
                _patch_block_manager(module.BlockManager, _Publisher(_ep))
            except Exception:
                traceback.print_exc()

        loader.exec_module = exec_module
        return spec


def arm_kv_event_hooks(bind_endpoint: str) -> None:
    """Arrange for ATOM's ``BlockManager`` to publish KV cache events.

    If the block-manager module is already imported, patch immediately;
    otherwise install a deferred ``sys.meta_path`` finder (the common case
    from the site-startup bootstrap, where importing ATOM eagerly would
    deadlock). Idempotent.
    """
    mod = sys.modules.get(_BLOCK_MANAGER_MODULE)
    if mod is not None:
        _patch_block_manager(mod.BlockManager, _Publisher(bind_endpoint))
        return
    if any(isinstance(f, _DeferredPatchFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _DeferredPatchFinder(bind_endpoint))
    logger.info("ATOM kv-events: armed deferred BlockManager patch (bind=%s)", bind_endpoint)
