###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import logging
from collections import OrderedDict

from infera.common.worker_pool import WorkerInfo
from infera.router.cache_control import (
    CacheHints,
    Retention,
    extract_image_keys,
    parse_cache_hints,
)
from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.client import KvEventClient
from infera.router.policy.base import Policy
from infera.router.policy.target import RouteTarget, expand_targets
from infera.server import metrics

logger = logging.getLogger(__name__)


# Multiplier applied to the per-role overlap_weight when the client
# explicitly asked for long retention via cache_control. The idea:
# long-retention requests are exactly where cache locality pays off
# most (stable system prompts that will be reused for hours), so we
# bias the cost function more aggressively toward the worker that
# already has the prefix.
_LONG_RETENTION_AMPLIFIER = 2.0
# Conversely, "client said NONE" means they don't want caching — we
# fall back to load balance with a tiny overlap weight to break ties.
_NONE_RETENTION_DAMPENER = 0.1

# Multimodal image-affinity (twin of the Rust router). Per-worker cap on
# remembered image keys (LRU); small because a handful of hot images dominate
# agentic VLM traffic. Each held image is worth this many "equivalent prefill
# blocks" of cache value — an image expands to hundreds of vision tokens on the
# engine, so a hit outweighs small load differences without a separate knob.
_MM_AFFINITY_CAP = 256
_MM_IMAGE_BLOCK_WEIGHT = 48.0


class KvEventAwarePolicy(Policy):
    """Pick the worker minimising

        cost(w) = overlap_weight * (request_blocks - hits(w)) + active_blocks(w)

    where ``active_blocks(w)`` is a refcounted set of distinct in-flight
    block hashes (deduped across requests sharing prefixes), not a sum of
    prompt lengths.

    For PD-disaggregated routing, the disagg-bootstrap router passes
    ``role_hint="prefill"`` or ``role_hint="decode"`` to ``pick``. The
    policy then applies a role-specific overlap weight:

    - **Prefill workers** are compute-bound. A cache hit lets the worker
      skip an entire prefill pass, which is the dominant cost. Weight
      cache locality aggressively (default 20.0).
    - **Decode workers** are memory-bound on KV. A prefill-time cache
      hit doesn't help the decode loop itself; routing decode by load
      keeps latency consistent. Weight cache locality weakly (default 2.0).
    - **Mixed pool** (``role_hint=None``) uses the original
      ``overlap_weight`` for backward compatibility.
    """

    def __init__(
        self,
        kv_client: KvEventClient,
        block_hasher: BlockHasher,
        *,
        overlap_weight: float = 1.0,
        prefill_overlap_weight: float | None = None,
        decode_overlap_weight: float | None = None,
    ) -> None:
        self._kv = kv_client
        self._hasher = block_hasher
        self._w = overlap_weight
        # If unset, fall back to the global overlap_weight so a policy
        # built without PD knobs keeps acting like before.
        self._w_prefill = (
            prefill_overlap_weight if prefill_overlap_weight is not None else overlap_weight
        )
        self._w_decode = (
            decode_overlap_weight if decode_overlap_weight is not None else overlap_weight
        )
        # worker_id -> {block_hash -> refcount}; len() is the load term.
        self._active_block_refs: dict[str, dict[int, int]] = {}
        # route_key -> ordered set of recent image keys (MRU last, bounded LRU).
        # Multimodal affinity: a request whose image a worker already holds
        # costs less there, co-locating repeat images onto the warm vision
        # cache. OrderedDict-as-set: move_to_end = touch, popitem(last=False) =
        # evict oldest.
        self._mm_affinity: dict[str, OrderedDict[int, None]] = {}

    def _base_weight_for(self, role_hint: str | None) -> float:
        if role_hint == "prefill":
            return self._w_prefill
        if role_hint == "decode":
            return self._w_decode
        return self._w

    def _retention_amplifier(self, hints: CacheHints) -> float:
        """Multiplier on overlap_weight based on client retention hint.

        Three semantic cases (not two — implicit-NONE differs from explicit-NONE):
        - LONG → amplify (cache locality matters a lot for this request)
        - explicit NONE → dampen (client opted out of caching; load balance)
        - SHORT / implicit-NONE / no hint → neutral (1.0)
        """
        if hints.retention == Retention.LONG:
            return _LONG_RETENTION_AMPLIFIER
        if hints.retention == Retention.NONE and hints.explicit_hint_seen:
            return _NONE_RETENTION_DAMPENER
        return 1.0

    def pick(
        self,
        candidates: list[WorkerInfo],
        request: dict,
        *,
        role_hint: str | None = None,
    ) -> tuple[RouteTarget, list[int]]:
        # Rank-multiplexed workers (SGLang --dp-size) fan out to one target
        # per DP rank so we can score and steer each rank's cache separately.
        targets = expand_targets(candidates)

        # Hash once per distinct (engine, block_size): different engines can
        # tokenize the same model differently (fast vs slow), so the query
        # hashes must be computed with the engine's own tokenizer to match the
        # ids it reports in kv-events. Typically there's just one (engine, bs).
        keys = {
            (t.worker.engine, t.worker.kv_block_size) for t in targets if t.worker.kv_block_size
        }
        hashes_for: dict[tuple, list[int]] = {
            k: self._hasher.hash_for(request, block_size=k[1], engine=k[0]) for k in keys
        }

        # Cache-control hints from the request body (Anthropic / OpenAI).
        # Server may have parsed these already and attached the result;
        # otherwise we parse here. Cheap on already-parsed bodies.
        cached_hints = request.get("_infera_cache_hints")
        hints: CacheHints = (
            cached_hints if isinstance(cached_hints, CacheHints) else parse_cache_hints(request)
        )

        amplified = self._base_weight_for(role_hint) * self._retention_amplifier(hints)

        # Multimodal requests: the text hasher can't reproduce the engine's
        # image blocks (SGLang substitutes pad-values, vLLM folds in extra-keys)
        # and the image placeholder token is the SAME id regardless of which
        # image — so trusting text overlap would collide a new image onto a
        # different image's cached KV (silent corruption). Drop text overlap
        # (w_overlap=0) and steer by IMAGE AFFINITY instead: keys the router's
        # own image→worker map, so one path serves SGLang, vLLM and ATOM alike.
        # (We still compute text hashes for `picked_blocks` — the refcount hooks
        # use them.)
        if hints.has_multimodal_content:
            w_overlap = 0.0
            mm_keys = extract_image_keys(request)
            metrics.cache_locality_skipped_total.labels(reason="multimodal").inc()
        else:
            w_overlap = amplified
            mm_keys = []
        w_mm = amplified * _MM_IMAGE_BLOCK_WEIGHT

        def active(t: RouteTarget) -> int:
            return len(self._active_block_refs.get(t.route_key, {}))

        def cost(t: RouteTarget) -> float:
            total = len(hashes_for.get((t.worker.engine, t.worker.kv_block_size), []))
            hits = self._cache_hits(t, hashes_for)
            # Image miss term: images this worker does NOT already hold cost
            # w_mm each; the worker with the warm vision cache pays 0 → wins.
            mm_miss = len(mm_keys) - self._mm_hits(t.route_key, mm_keys)
            return w_overlap * (total - hits) + w_mm * mm_miss + active(t)

        # Tie-break by lower active so equal-cost candidates fall back to
        # least-loaded.
        picked = min(targets, key=lambda t: (cost(t), active(t)))
        picked_blocks = list(
            hashes_for.get((picked.worker.engine, picked.worker.kv_block_size), [])
        )
        # Mark the chosen worker as now holding this request's images, so the
        # next request for the same image is drawn back to its warm cache.
        mm_affinity_hits = self._mm_hits(picked.route_key, mm_keys)
        self._record_mm(picked.route_key, mm_keys)

        # Telemetry: record the pick decision per role + target, plus
        # the retention bucket we observed on this request.
        cache_hits = self._cache_hits(picked, hashes_for)
        metrics.record_pick(
            role=role_hint or "mixed",
            worker_id=picked.route_key,
            cache_hits=cache_hits,
            request_blocks=len(picked_blocks),
        )
        # policy_active_blocks is NOT published here: at this point the pick's
        # own blocks have not been refcounted yet (on_request_started does that,
        # after pick returns), so writing it here reports a one-request-stale
        # count. It is published from _publish_active on every refcount change.
        metrics.cache_control_seen_total.labels(retention=hints.retention.value).inc()

        # Structured log: ops can grep `policy=kv-aware role=...` and correlate
        # with the X-Infera-Request-Id the server emits.
        logger.info(
            "pick policy=kv-aware role=%s retention=%s request_id=%s picked=%s "
            "cache_hits=%d request_blocks=%d active_blocks=%d w_overlap=%.2f "
            "mm_images=%d mm_affinity_hits=%d",
            role_hint or "mixed",
            hints.retention.value,
            request.get("_infera_request_id", "-"),
            picked.route_key,
            cache_hits,
            len(picked_blocks),
            active(picked),
            w_overlap,
            len(mm_keys),
            mm_affinity_hits,
        )

        return picked, picked_blocks

    # --- Lifecycle hooks ---

    def on_worker_added(self, worker: WorkerInfo) -> None:
        self._kv.on_worker_added(worker)

    def on_worker_removed(self, worker_id: str) -> None:
        self._kv.on_worker_removed(worker_id)
        # Drop the worker's own key plus any per-rank keys ("<id>#dpN").
        prefix = f"{worker_id}#dp"
        for key in [k for k in self._active_block_refs if k == worker_id or k.startswith(prefix)]:
            self._active_block_refs.pop(key, None)
            # Drop the series too: a removed worker that keeps exporting its
            # last in-flight count reads as a permanently loaded worker.
            try:
                metrics.policy_active_blocks.remove(key)
            except KeyError:
                pass
        for key in [k for k in self._mm_affinity if k == worker_id or k.startswith(prefix)]:
            self._mm_affinity.pop(key, None)

    def _publish_active(self, route_key: str) -> None:
        """Mirror ``route_key``'s in-flight block count into the gauge.

        Must be called after every mutation of ``_active_block_refs``. Setting
        it only where the refcounts change is the point: publishing from
        ``pick()`` reports the count from *before* this request's blocks are
        added, so the gauge trails by one request and reads a flat 0 whenever
        requests finish before the next one is picked.
        """
        metrics.policy_active_blocks.labels(worker_id=route_key).set(
            len(self._active_block_refs.get(route_key, {}))
        )

    def on_request_started(self, route_key: str, blocks: list[int] | None = None) -> None:
        if not blocks:
            return
        refs = self._active_block_refs.setdefault(route_key, {})
        for h in blocks:
            refs[h] = refs.get(h, 0) + 1
        self._publish_active(route_key)

    def on_request_finished(self, route_key: str, blocks: list[int] | None = None) -> None:
        if not blocks:
            return
        refs = self._active_block_refs.get(route_key)
        if refs is None:
            return
        for h in blocks:
            rc = refs.get(h, 0) - 1
            if rc <= 0:
                refs.pop(h, None)
            else:
                refs[h] = rc
        self._publish_active(route_key)

    async def aclose(self) -> None:
        await self._kv.aclose()

    @property
    def kv_client(self) -> KvEventClient:
        return self._kv

    def _mm_hits(self, route_key: str, keys: list[int]) -> int:
        """How many of ``keys`` this worker is recorded as holding."""
        if not keys:
            return 0
        aff = self._mm_affinity.get(route_key)
        if not aff:
            return 0
        return sum(1 for k in keys if k in aff)

    def _record_mm(self, route_key: str, keys: list[int]) -> None:
        """Record that ``route_key`` now holds ``keys`` (MRU, deduped, capped)."""
        if not keys:
            return
        aff = self._mm_affinity.setdefault(route_key, OrderedDict())
        for k in keys:
            if k in aff:
                aff.move_to_end(k)
            else:
                aff[k] = None
        while len(aff) > _MM_AFFINITY_CAP:
            aff.popitem(last=False)

    def _cache_hits(self, t: RouteTarget, hashes_for: dict[tuple, list[int]]) -> int:
        if not t.worker.kv_block_size:
            return 0
        request_hashes = hashes_for.get((t.worker.engine, t.worker.kv_block_size), [])
        if not request_hashes:
            return 0
        view = self._kv.cache_view(t.worker.worker_id, t.dp_rank)
        n = 0
        for h in request_hashes:
            if h not in view:
                break
            n += 1
        return n
