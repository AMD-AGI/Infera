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
from infera.router.kv_event import responses_input
from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.render_probe import ProbeResult, spawn_probe
from infera.router.kv_event.render_variant import VariantRegistry
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

# Per-pick decay on each worker's recent-dispatch total (half-life ~23 picks).
#
# The load half of the cost function counts blocks that are *in flight*, which
# is 0 for every worker whenever a request finishes before the next one is
# picked. Session-paced agent traffic has exactly that shape, so on that
# workload the cost function loses its load term entirely: a cold fleet ties,
# the tie goes to whichever candidate is enumerated first, and the cache that
# winner picks up re-elects it on every later request. The result is a
# permanent 100/0 split across symmetric workers at any overlap weight.
#
# `recent` keeps the load signal alive across requests that never overlap in
# time. A pick is charged the blocks the winner MISSED, not the blocks the
# request contains: prefill work is proportional to what the worker has to
# compute, and a block already in its cache costs it nothing. Charging misses
# rather than totals is what lets the term coexist with cache affinity --
# a worker serving a fully-cached prompt accrues no load, so it keeps winning
# that prompt, while a worker handed a cold prompt accrues the whole thing and
# the next cold prompt goes elsewhere.
#
# Misses are in blocks, the same unit as in-flight load, so the two sum without
# a conversion factor and the term scales with request size. Charging one point
# per request instead would cap the term at 1/(1 - decay) no matter how large
# the requests were, which a single block of cache edge outvotes outright at
# any overlap weight above that cap -- leaving the split unfixed on exactly the
# prefill-weighted pools (typical production weight: 20.0) that most need
# spreading.
#
# The decay sets how long a worker carries what it was handed. Too fast and the
# term cannot accumulate enough to displace an incumbent between one session
# and the next; measured against the field trace, 0.9 still left w=20 fully
# pinned while anything from 0.95 up balanced it.
_RECENT_DECAY = 0.97

# Load charged for a pick we have no block information about: the hasher
# returned nothing, because the prompt is shorter than the index block size
# (768 tokens by default) or tokenisation failed. Such a request still costs
# the worker something, and charging 0 would hold the load term at 0 for every
# candidate -- restoring the permanent tie, and with it the 100/0 split, on any
# workload whose prompts are short. One block is the smallest honest estimate:
# enough to break the tie and rotate workers, small enough that a real
# multi-block request still outweighs it.
_UNKNOWN_COST_BLOCKS = 1.0

# Sentinel for "never probed", distinct from a recorded `None` verdict ("probed,
# could not reach the engine"). The two must not collapse: the first has to be
# probed, the second has to be RE-probed, and a settled True/False must not be.
_UNPROBED = object()


class KvEventAwarePolicy(Policy):
    """Pick the worker minimising

        cost(w) = overlap_weight * (request_blocks - hits(w)) + load(w)
        load(w) = active_blocks(w) + recent_blocks(w)

    where ``active_blocks(w)`` is a refcounted set of distinct in-flight
    block hashes (deduped across requests sharing prefixes), not a sum of
    prompt lengths, and ``recent_blocks(w)`` is a decayed sum of the block
    counts recently dispatched to ``w``.

    Both halves are load-bearing. ``active_blocks`` alone is 0 for every
    worker under traffic paced so that a request finishes before the next is
    picked -- the normal shape of multi-turn agent sessions -- which drops the
    load term out of the comparison and pins the whole fleet's traffic onto
    one worker. See ``_RECENT_DECAY``.

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
        variants: VariantRegistry | None = None,
    ) -> None:
        self._kv = kv_client
        self._hasher = block_hasher
        # What each worker's engine merges into every render before the
        # template runs. Empty by default, which is exactly the behaviour that
        # predates it: no merge, hash the body as the client sent it.
        self._variants = variants or VariantRegistry()
        self._w = overlap_weight
        # If unset, fall back to the global overlap_weight so a policy
        # built without PD knobs keeps acting like before.
        self._w_prefill = (
            prefill_overlap_weight if prefill_overlap_weight is not None else overlap_weight
        )
        self._w_decode = (
            decode_overlap_weight if decode_overlap_weight is not None else overlap_weight
        )
        # worker_id -> {block_hash -> refcount}; len() is the in-flight load.
        self._active_block_refs: dict[str, dict[int, int]] = {}
        # route_key -> decayed sum of recently dispatched block counts. Carries
        # the load signal across requests that never overlap in time.
        self._recent_blocks: dict[str, float] = {}
        # route_key -> ordered set of recent image keys (MRU last, bounded LRU).
        # Multimodal affinity: a request whose image a worker already holds
        # costs less there, co-locating repeat images onto the warm vision
        # cache. OrderedDict-as-set: move_to_end = touch, popitem(last=False) =
        # evict oldest.
        self._mm_affinity: dict[str, OrderedDict[int, None]] = {}
        # Render-probe bookkeeping. `_parity_pending` holds workers with a probe
        # in flight: a verdict is only reported once all four bodies have
        # round-tripped, which is longer than the gap between discovery
        # snapshots, so the gauge alone cannot answer "have we already asked
        # this one". `_parity_labels` remembers each worker's exact gauge labels
        # so removal does not have to read prometheus_client's internals.
        # worker_id -> the epoch of the probe reserved for it. Keyed by probe,
        # not by worker: a bare set cannot tell one probe from the next, so a
        # probe still in flight when its worker left and rejoined consumed the
        # REPLACEMENT probe's reservation -- writing the dead instance's
        # verdict, dropping the fresh one, and refusing every later claim.
        self._parity_pending: dict[str, int] = {}
        self._parity_labels: dict[str, str] = {}
        # worker_id -> whether its settled verdict was conclusive. `None` (could
        # not reach the engine) is not final: a worker that was merely slow to
        # start would otherwise export -1 for the life of the router.
        self._parity_verdict: dict[str, bool | None] = {}
        self._parity_epoch = 0

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

        # Hash once per distinct (engine, block_size, variant): each of the
        # three changes the ids the engine will report in its kv-events, so a
        # query hashed under the wrong one matches nothing.
        #
        #   engine      -- different engines tokenize the same model
        #                  differently (fast vs slow tokenizer).
        #   block_size  -- the blocking itself.
        #   variant     -- the server-side --default-chat-template-kwargs this
        #                  worker was launched with, merged into the body
        #                  before the template runs. See render_variant: the
        #                  router is never told about it, and a fleet that is
        #                  uniform (the normal case) still yields exactly one
        #                  key here and hashes exactly once.
        #
        # Normalised BEFORE the variant is applied, and once for the whole
        # fleet: the engine turns a `/v1/responses` body into a chat body
        # (`_make_request`) and only then merges its server-side template
        # defaults (`_process_messages`). Applying the variant first writes
        # `chat_template_kwargs` onto a body the normalisation rebuilds from
        # scratch, which drops the variant for `/v1/responses` alone.
        base = responses_input.normalised(request)
        variant_of = {t.route_key: self._variants.for_worker(t.worker.worker_id) for t in targets}
        key_of: dict[str, tuple] = {
            t.route_key: (t.worker.engine, t.worker.kv_block_size, variant_of[t.route_key].id)
            for t in targets
            if t.worker.kv_block_size
        }
        hashes_for: dict[tuple, list[int]] = {}
        for t in targets:
            k = key_of.get(t.route_key)
            if k is None or k in hashes_for:
                continue
            hashes_for[k] = self._hasher.hash_for(
                variant_of[t.route_key].apply(base), block_size=k[1], engine=k[0]
            )

        def blocks(t: RouteTarget) -> list[int]:
            return hashes_for.get(key_of.get(t.route_key), [])

        # Cache-control hints from the request body (Anthropic / OpenAI).
        # Server may have parsed these already and attached the result;
        # otherwise we parse here. Cheap on already-parsed bodies.
        #
        # Read from `base`, not `request`: both this and `extract_image_keys`
        # below key off `messages`/`system`/`images`, none of which a Responses
        # body has -- it carries `input`. Reading the raw body reports every
        # `/v1/responses` request as text-only, which was harmless while such a
        # body hashed to nothing and is not any more. See the multimodal note
        # below for what that costs.
        cached_hints = request.get("_infera_cache_hints")
        hints: CacheHints = (
            cached_hints if isinstance(cached_hints, CacheHints) else parse_cache_hints(base)
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
            mm_keys = extract_image_keys(base)
            metrics.cache_locality_skipped_total.labels(reason="multimodal").inc()
        else:
            w_overlap = amplified
            mm_keys = []
        w_mm = amplified * _MM_IMAGE_BLOCK_WEIGHT

        def active(t: RouteTarget) -> int:
            return len(self._active_block_refs.get(t.route_key, {}))

        def load(t: RouteTarget) -> float:
            """Blocks in flight now, plus blocks dispatched recently."""
            return active(t) + self._recent_blocks.get(t.route_key, 0.0)

        def cost(t: RouteTarget) -> float:
            hits = self._cache_hits(t, blocks(t))
            # Image miss term: images this worker does NOT already hold cost
            # w_mm each; the worker with the warm vision cache pays 0 → wins.
            mm_miss = len(mm_keys) - self._mm_hits(t.route_key, mm_keys)
            # Credit hits rather than charging misses. The two used to be the
            # same ranking -- `w*(total - hits)` differs from `-w*hits` by
            # `w*total`, a constant that cancels in a min() as long as every
            # candidate has the same `total`. Variants break that premise: two
            # workers rendering different preambles produce block lists of
            # different lengths, and charging misses would then penalise the
            # worker whose preamble is merely longer, independently of what it
            # actually holds. Crediting hits scores only the cache.
            return -w_overlap * hits + w_mm * mm_miss + load(t)

        # Tie-break by lower load so equal-cost candidates fall back to
        # least-loaded.
        picked = min(targets, key=lambda t: (cost(t), load(t)))
        picked_blocks = list(blocks(picked))
        # Mark the chosen worker as now holding this request's images, so the
        # next request for the same image is drawn back to its warm cache.
        mm_affinity_hits = self._mm_hits(picked.route_key, mm_keys)
        self._record_mm(picked.route_key, mm_keys)

        cache_hits = self._cache_hits(picked, picked_blocks)
        # Charge the winner for the blocks it will have to compute. Done here
        # rather than in on_request_started because the hooks run on the
        # dispatch path, which skips them on the failure routes -- and a pick
        # that goes uncharged is invisible to the next one.
        self._record_dispatch(picked.route_key, len(picked_blocks) - cache_hits, len(picked_blocks))

        # Telemetry: record the pick decision per role + target, plus
        # the retention bucket we observed on this request.
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
        # Confirm that what we render is what this worker renders. See
        # render_probe: a divergence here is the one kv-aware failure that
        # produces no error anywhere, so it has to be actively looked for.
        #
        # Discovery now calls this again when a worker's record changes in
        # place, so a worker restarted in the same Pod -- possibly with a
        # different chat template or --default-chat-template-kwargs -- is
        # re-read instead of keeping a verdict and a variant earned by the
        # process that used to be there.
        wid = worker.worker_id
        if wid in self._parity_pending:
            return
        settled = self._parity_verdict.get(wid, _UNPROBED)
        if settled is not _UNPROBED and settled is not None:
            return
        self._parity_epoch += 1
        self._parity_pending[wid] = self._parity_epoch
        spawn_probe(
            self._hasher,
            worker,
            report=self._report_render_parity,
            variants=self._variants,
            epoch=self._parity_epoch,
            release=self._release_parity,
        )

    def _release_parity(self, worker_id: str, epoch: int) -> None:
        """Hand a reservation back unused.

        Claiming happens here and recording only inside the probe task, so
        every path that abandons the probe in between -- an exception building
        the client, a cancelled task at shutdown, or `spawn_probe` returning
        early because the engine is not sglang or there is no running loop --
        has to come through here. Without it those workers sit in
        `_parity_pending` forever: never re-probed, and never given a gauge.
        """
        if self._parity_pending.get(worker_id) == epoch:
            del self._parity_pending[worker_id]

    def _forget_parity(self, worker_id: str) -> None:
        """Drop a settled verdict so the next registration re-probes."""
        self._parity_verdict.pop(worker_id, None)

    def _report_render_parity(
        self, worker: WorkerInfo, result: ProbeResult, epoch: int | None = None
    ) -> None:
        if epoch is not None and self._parity_pending.get(worker.worker_id) != epoch:
            # The worker left the fleet while this probe was in flight, or a
            # later probe has since taken the reservation. Setting the gauge
            # now would resurrect a series `on_worker_removed` just dropped, or
            # overwrite a fresher verdict with a dead instance's.
            logger.debug(
                "kv-aware: worker %s probe reservation gone (left the fleet, or superseded); "
                "verdict dropped",
                worker.worker_id,
            )
            return
        self._parity_pending.pop(worker.worker_id, None)
        self._parity_verdict[worker.worker_id] = result.ok
        if result.ok is False:
            logger.error(
                "kv-aware: worker %s (%s) does NOT render prompts the way this router "
                "does, so cache lookups for it will always miss and it will be routed "
                "on load only -- %s. Most likely causes: the engine was started with "
                "--default-chat-template-kwargs (the router is never told), or its "
                "model directory has a different chat template than --tokenizer-path.",
                worker.worker_id,
                worker.model_name,
                result.detail,
            )
            metrics.render_parity_diverged_total.labels(model=worker.model_name).inc()
        elif result.ok is None:
            logger.info(
                "kv-aware: could not verify prompt rendering against worker %s (%s) -- %s",
                worker.worker_id,
                worker.model_name,
                result.detail,
            )
        else:
            logger.info(
                "kv-aware: worker %s (%s) render verified -- %s",
                worker.worker_id,
                worker.model_name,
                result.detail,
            )
        self._parity_labels[worker.worker_id] = worker.model_name
        # The variant the probe read back and judged parity under. Labelled by
        # worker, but the number worth alerting on is how many DISTINCT labels
        # the fleet exports: 1 means a single fleet-wide flag would have done,
        # 2+ means it could not have been right for everyone.
        metrics.render_variant.labels(
            worker_id=worker.worker_id,
            variant=self._variants.for_worker(worker.worker_id).label(),
        ).set(1)
        metrics.render_parity.labels(worker_id=worker.worker_id, model=worker.model_name).set(
            {True: 1, False: 0, None: -1}[result.ok]
        )

    def on_worker_removed(self, worker_id: str) -> None:
        self._kv.on_worker_removed(worker_id)
        # A variant that outlives its worker keeps a hash-cache key alive for
        # something that is not running, and keeps its gauge exported. Read the
        # label before dropping it -- it is the gauge's own label value, and
        # there is no other way back to it once forgotten.
        variant_label = self._variants.for_worker(worker_id).label()
        self._variants.retain(lambda wid: wid != worker_id)
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
        # Iterated separately from _active_block_refs: a worker can carry a
        # recent total with nothing in flight, which is the state this term
        # exists to represent.
        for key in [k for k in self._recent_blocks if k == worker_id or k.startswith(prefix)]:
            self._recent_blocks.pop(key, None)
        # The parity gauge is labelled by worker_id, so a removed worker would
        # otherwise keep exporting its last verdict -- possibly a 0 -- forever.
        # The labels come from our own bookkeeping rather than from
        # prometheus_client's private `_metrics`, which is not API and whose
        # absence would raise here and abandon the rest of this method.
        self._parity_pending.pop(worker_id, None)
        self._parity_verdict.pop(worker_id, None)
        model = self._parity_labels.pop(worker_id, None)
        if model is not None:
            try:
                metrics.render_parity.remove(worker_id, model)
            except KeyError:
                pass
            try:
                metrics.render_variant.remove(worker_id, variant_label)
            except KeyError:
                pass

    def _record_dispatch(self, route_key: str, missed_blocks: int, request_blocks: int) -> None:
        """Decay every worker's recent total, then charge for the pick.

        Decaying on each pick rather than on a wall-clock timer keeps routing a
        pure function of the request sequence: same requests in, same decisions
        out, which is what makes the behaviour testable. Totals that decay to
        nothing are dropped so an idle worker returns to a clean 0.

        Two different situations both arrive here with zero misses, and they
        must not be charged the same way:

        - ``request_blocks > 0`` and every one of them hit. The worker has no
          prefill to do, so it takes on no load and stays the right answer for
          that prompt. Charge nothing.
        - ``request_blocks == 0``. The hasher produced no blocks at all -- the
          prompt is shorter than the index block size, or tokenisation failed.
          We know nothing about this request's cost, but the worker still has
          to serve it. Charging nothing here leaves the load term at 0 for
          every candidate forever, which is the exact tie that sends every
          request to whichever worker sorts first: the 100/0 split this term
          exists to prevent, reappearing on short prompts. Charge the floor.
        """
        for key in list(self._recent_blocks):
            decayed = self._recent_blocks[key] * _RECENT_DECAY
            if decayed < 1e-3:
                del self._recent_blocks[key]
            else:
                self._recent_blocks[key] = decayed
        charge = float(missed_blocks)
        if request_blocks <= 0:
            charge = _UNKNOWN_COST_BLOCKS
        if charge <= 0:
            return
        self._recent_blocks[route_key] = self._recent_blocks.get(route_key, 0.0) + charge

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

    def _cache_hits(self, t: RouteTarget, request_hashes: list[int]) -> int:
        if not t.worker.kv_block_size:
            return 0
        if not request_hashes:
            return 0
        view = self._kv.cache_view(t.worker.worker_id, t.dp_rank)
        n = 0
        for h in request_hashes:
            if h not in view:
                break
            n += 1
        return n
