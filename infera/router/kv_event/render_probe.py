###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Ask a worker what it would actually tokenise, and check we agree.

kv-aware routing rests on an assumption the router cannot verify on its own:
that the prompt it renders is the prompt the engine renders, byte for byte. When
that stops being true nothing fails. The block hashes simply never match, every
lookup misses, the policy quietly degrades to load balancing, and every health
signal -- readiness, kv event flow, cache view size -- stays green. We have
found these by noticing a flat 0% hit rate days later.

sglang serves `/v1/tokenize`, and for a `messages` body it runs the real
`OpenAIServingChat._process_messages`. That makes it ground truth rather than a
second opinion: it sees the engine's own template, its own tokenizer, and the
server-side merges the router is never told about -- `--default-chat-template-
kwargs` above all, which has broken kv-aware here before and is invisible from
the router by construction.

So on worker registration we render a few small probe bodies both ways and
compare token ids. A mismatch is reported once, loudly, with the index of the
first differing token; the render-parity gauge then carries it for alerting.
This is best-effort and never blocks a worker from serving: a worker whose
render we cannot confirm still takes traffic, it just cannot be trusted to hit
cache, and now we know that at startup instead of at post-mortem.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass

from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.kv_event import responses_input
from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.render_variant import EMPTY_VARIANT, RenderVariant, VariantRegistry

logger = logging.getLogger(__name__)

# Deliberately tiny, and deliberately not the unit-test corpus: this runs
# against production workers at registration, so it must cost the engine
# almost nothing (tokenisation only -- /v1/tokenize never schedules a forward
# pass). Each body earns its place by covering a divergence we have actually
# shipped:
#
#   plain      -- the template preamble. GLM-5.3 opens with a
#                 "Reasoning Effort: High" system line, so a router that gets
#                 the defaults wrong diverges in the FIRST block of every
#                 prompt, and everything chained after it.
#   effort     -- an explicit reasoning_effort, which the engine forwards into
#                 template scope and (for some templates) remaps onto a boolean
#                 toggle instead. Catches both the plumbing and the remap.
#   tools      -- tools rendered from a full pydantic Tool.model_dump(), whose
#                 materialised defaults the client never sent.
#   tool_call  -- an assistant turn carrying tool_calls with `arguments` in the
#                 OpenAI string form. Templates disagree about parsing it; get
#                 it wrong and every agentic conversation loses its cache from
#                 the second turn on, which is nearly the whole workload.
_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

PROBE_BODIES: dict[str, dict] = {
    "plain": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    "effort": {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "low",
    },
    "tools": {
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": _PROBE_TOOLS,
    },
    "tool_call": {
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C, clear"},
        ],
        "tools": _PROBE_TOOLS,
    },
    # The live failure this file exists for: a `/v1/responses` body hashes
    # to nothing unless `responses_input` rebuilds the chat request the
    # engine's `_make_request` would. Chat-only probes stay green through
    # that outage. Tokenise the normalised chat body, not `input` --
    # `/v1/tokenize` runs `_process_messages`, which is the post-conversion
    # path.
    "responses": {"input": "What is 2+2?"},
}


@dataclass(frozen=True)
class ProbeResult:
    """`ok` is a three-state, and the third state matters.

    True  -- every probe body matched; kv-aware is known good on this worker.
    False -- at least one diverged; kv-aware is known broken on this worker.
    None  -- we could not tell (endpoint absent, worker unreachable, request
             refused). Reporting that as a failure would page on every engine
             that simply does not serve /v1/tokenize, and reporting it as a
             pass would claim a guarantee we never obtained.
    """

    ok: bool | None
    detail: str


async def engine_render_variant(client, worker: WorkerInfo) -> RenderVariant | None:
    """What the worker says it was launched with.

    ``--default-chat-template-kwargs`` is merged into every request *before* the
    template runs and is invisible from everywhere else the router looks, so
    this endpoint is the only way to know a worker renders a different preamble
    than we do. ``None`` means we could not ask -- older engine, non-sglang,
    unreachable -- which is not the same as "it has none", so the caller keeps
    its existing assumption rather than recording an empty variant.
    """
    for path in ("/get_server_info", "/v1/server_info"):
        try:
            resp = await client.get(f"{worker.url}{path}")
        except Exception as exc:  # noqa: BLE001 - a probe must never break startup
            logger.debug("render probe: %s%s unreachable: %s", worker.url, path, exc)
            return None
        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            return None
        try:
            parsed = resp.json()
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(parsed, dict):
            return None
        # sglang has served this both flat and nested under `server_args`.
        args = parsed.get("server_args") if isinstance(parsed.get("server_args"), dict) else {}
        if "default_chat_template_kwargs" in parsed:
            return RenderVariant.from_default_chat_template_kwargs(
                parsed["default_chat_template_kwargs"]
            )
        if "default_chat_template_kwargs" in args:
            return RenderVariant.from_default_chat_template_kwargs(
                args["default_chat_template_kwargs"]
            )
        # Answered, but does not carry the field at all -- an engine older than
        # the flag, a renamed key, a proxy that trims the payload. That is "we
        # could not ask", not "it has none": returning the empty variant here
        # would be RECORDED against this worker, and `for_worker` prefers a
        # recorded entry over the fleet default, so an operator who set
        # --kv-default-chat-template-kwargs to match their fleet would have it
        # silently discarded for every worker and every request.
        logger.debug(
            "render probe: %s answered %s without default_chat_template_kwargs; "
            "keeping the router's existing assumption for this worker",
            worker.url,
            path,
        )
        return None
    return None


async def probe_worker(
    hasher: BlockHasher,
    worker: WorkerInfo,
    *,
    timeout: float = 10.0,
    bodies: dict[str, dict] | None = None,
    client=None,
    variant: RenderVariant | None = None,
) -> ProbeResult:
    """Compare our render against the worker's for every probe body.

    ``variant`` is the server-side template default this worker was found to
    render with; the probe applies it exactly as the policy will, so a pass here
    means the policy's hashes are the worker's hashes -- and a mistake in
    modelling the variant shows up as a reported divergence rather than as a hit
    rate nobody is watching.
    """
    import contextlib

    import httpx

    bodies = bodies if bodies is not None else PROBE_BODIES
    variant = variant if variant is not None else EMPTY_VARIANT
    unknown: list[str] = []
    declined: list[str] = []
    skipped: list[str] = []
    mismatches: list[str] = []
    # An injected client is the caller's to close.
    owned = client is None
    client = client if client is not None else httpx.AsyncClient(timeout=timeout)

    try:
        async with client if owned else contextlib.nullcontext(client):
            for name, template in bodies.items():
                body = {**template, "model": worker.model_name}
                # Normalise before applying the variant, in the engine's order:
                # `_make_request` first, `_process_messages` (which merges
                # --default-chat-template-kwargs) second. The other way round,
                # a Responses probe body silently loses the variant and the
                # probe reports parity the live path does not have.
                base = responses_input.normalised(body)
                # Still a Responses body means we could not borrow
                # `_make_request` (no sglang, or too old). Chat hashing on
                # this host still works. Counting that as a decline marks the
                # worker Diverged and pages the "alert on 0" rule, for a
                # converter the probe never had. Skip; do not fold into
                # `unknown` either -- that would turn a matching chat corpus
                # into Unknown.
                if responses_input.is_responses_body(base):
                    skipped.append(f"{name}: no Responses converter on this router")
                    continue
                # Off the event loop. The first call for a model can trigger a
                # synchronous `AutoTokenizer.from_pretrained` / sglang
                # `get_tokenizer` -- filesystem, possibly the HF hub -- and this
                # runs from the discovery hook, so during a rollout the router
                # would dispatch nothing for the seconds that takes. The
                # per-body `apply_chat_template` + `encode` are cheaper but add
                # up over four bodies per worker.
                ours = await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        hasher.token_ids_for,
                        variant.apply(base),
                        engine=worker.engine,
                    ),
                )
                if ours is None:
                    # NOT folded in with "engine did not answer". The two look
                    # alike -- neither produced a comparison -- and mean
                    # opposite things. An engine that cannot be asked is a
                    # limit on the probe. A body the ROUTER declined is a
                    # kv-aware outage for that request shape, already decided,
                    # on this worker: those requests hash to nothing and route
                    # on load. Reporting it as a gap and then returning
                    # Confirmed is how a worker whose every tool-carrying
                    # prompt is unhashable exports the same green 1 as a worker
                    # that matched on all four.
                    declined.append(f"{name}: router declined to render")
                    continue
                # Ask the engine about the hashed shape. A raw Responses
                # `input` body is not what `/v1/tokenize` runs through
                # `_process_messages`; the converted chat body is.
                theirs = await _engine_token_ids(client, worker, base)
                if theirs is None:
                    unknown.append(f"{name}: engine did not answer")
                    continue
                if theirs != ours:
                    mismatches.append(_describe(name, ours, theirs))
    except Exception as exc:  # noqa: BLE001 - a probe must never break startup
        return ProbeResult(None, f"probe failed: {exc}")

    compared = len(bodies) - len(skipped)
    if mismatches:
        return ProbeResult(False, "; ".join(mismatches))
    if declined and len(declined) < compared:
        # PARTIAL decline is the dangerous shape, and `False` is the honest
        # verdict: the router demonstrably renders for this model, just not for
        # these request shapes, so those requests hash to nothing and route on
        # load while the rest look fine. Folded into `unknown` it returned
        # Confirmed, and under the documented "alert on 0" rule a worker whose
        # every tool-carrying prompt was unhashable read as verified-good.
        return ProbeResult(False, "; ".join(declined + unknown))
    if compared == 0 or declined or unknown:
        # Nothing compared (no converter and no other bodies), nothing
        # rendered at all (a router with no tokenizer configured), or the
        # engine never answered. Neither is a statement about this worker's
        # render, so neither may claim one.
        return ProbeResult(None, "; ".join(declined + unknown + skipped))
    return ProbeResult(True, f"matched on {compared} probe bodies")


async def _engine_token_ids(client, worker: WorkerInfo, body: dict) -> list[int] | None:
    """`/v1/tokenize` for a chat body, or None if this worker cannot answer.

    `add_special_tokens=False` mirrors what the chat path does with an
    already-templated string; for a `messages` body sglang ignores the flag and
    returns `_process_messages`' ids directly, but pinning it keeps the request
    honest against engines that route the two the same way.
    """
    for path in ("/v1/tokenize", "/tokenize"):
        try:
            resp = await client.post(
                f"{worker.url}{path}", json={**body, "add_special_tokens": False}
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("render probe: %s%s unreachable: %s", worker.url, path, exc)
            return None
        if resp.status_code == 404:
            continue  # older engine, or a non-sglang one; try the alias
        if resp.status_code != 200:
            logger.debug(
                "render probe: %s%s returned %s: %s",
                worker.url,
                path,
                resp.status_code,
                resp.text[:200],
            )
            return None
        tokens = resp.json().get("tokens")
        if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
            return tokens
        return None
    return None


def _describe(name: str, ours: list[int], theirs: list[int]) -> str:
    at = next(
        (i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b),
        min(len(ours), len(theirs)),
    )
    lo = max(0, at - 4)
    return (
        f"{name}: diverges at token {at} of {len(theirs)} "
        f"(router {ours[lo : at + 4]} vs engine {theirs[lo : at + 4]})"
    )


def spawn_probe(
    hasher: BlockHasher,
    worker: WorkerInfo,
    *,
    report,
    variants: VariantRegistry | None = None,
    epoch: int | None = None,
    release=None,
) -> None:
    """Fire-and-forget `probe_worker`, if there is a loop to fire it on.

    Called from the registry's synchronous worker-added hook. Outside a running
    loop (unit tests, sync embedders) this does nothing rather than spinning up
    an event loop behind the caller's back.

    `release(worker_id, epoch)` hands the caller's reservation back on every
    path that does not reach `report` -- the engine is not sglang, the model is
    already ruled out, there is no loop, or the task raised. The caller claims
    before spawning (it has to: a probe takes longer than the gap between
    discovery snapshots), so a path that neither reports nor releases strands
    the worker as permanently in-flight: never re-probed, and never given a
    gauge series at all.
    """

    def _release() -> None:
        if release is not None and epoch is not None:
            release(worker.worker_id, epoch)

    if worker.engine != EngineType.SGLANG:
        # /v1/tokenize is sglang's. vLLM's equivalent differs in both path and
        # payload; probing it blindly would just produce 404 noise.
        _release()
        return
    if not hasher.can_render(worker.model_name, worker.engine):
        # This model's tokenizer has already been ruled out, so every body would
        # come back "router declined to render" and the verdict would be Unknown
        # before we asked anything. The comparison is worth nothing here; the
        # /get_server_info round trip is all that would be left of it. Mirrors
        # the Rust probe's `hasher.is_enabled()` gate. Not a guarantee in the
        # other direction: a model nothing has tried yet still gets probed.
        _release()
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _release()
        return

    async def _run() -> None:
        import httpx

        # Everything between the caller's claim and `report` runs under this:
        # `probe_worker` self-insures, but the client construction, the variant
        # read and `report` itself do not, and an exception (or a cancellation
        # at loop shutdown) escaping here would leave the reservation held for
        # the life of the router.
        try:
            variant = None
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Ask what it renders with BEFORE checking whether we agree, so
                # the verdict is about the render we will actually route on.
                if variants is not None and variants.per_worker_enabled:
                    variant = await engine_render_variant(client, worker)
                    if variant is not None:
                        variants.record(worker.worker_id, variant)
                if variant is None and variants is not None:
                    variant = variants.for_worker(worker.worker_id)
                result = await probe_worker(hasher, worker, client=client, variant=variant)
            if epoch is None:
                report(worker, result)
            else:
                report(worker, result, epoch)
        except asyncio.CancelledError:
            _release()
            raise
        except Exception:
            logger.exception("render probe: task failed for %s", worker.worker_id)
            _release()

    task = loop.create_task(_run())
    # Hold a reference: a bare create_task can be garbage-collected mid-flight.
    _IN_FLIGHT.add(task)
    task.add_done_callback(_IN_FLIGHT.discard)


_IN_FLIGHT: set[asyncio.Task] = set()
