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
        field = parsed.get("default_chat_template_kwargs")
        if field is None and isinstance(parsed.get("server_args"), dict):
            field = parsed["server_args"].get("default_chat_template_kwargs")
        return RenderVariant.from_default_chat_template_kwargs(field)
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
                ours = hasher.token_ids_for(
                    variant.apply(responses_input.normalised(body)), engine=worker.engine
                )
                if ours is None:
                    # Not a divergence: the router already knows it cannot
                    # reproduce this body and routes it on load. Silent
                    # WRONGNESS is the thing this probe exists to find.
                    unknown.append(f"{name}: router declined to render")
                    continue
                theirs = await _engine_token_ids(client, worker, body)
                if theirs is None:
                    unknown.append(f"{name}: engine did not answer")
                    continue
                if theirs != ours:
                    mismatches.append(_describe(name, ours, theirs))
    except Exception as exc:  # noqa: BLE001 - a probe must never break startup
        return ProbeResult(None, f"probe failed: {exc}")

    if mismatches:
        return ProbeResult(False, "; ".join(mismatches))
    if unknown and len(unknown) == len(bodies):
        return ProbeResult(None, "; ".join(unknown))
    if unknown:
        return ProbeResult(True, f"matched, with gaps: {'; '.join(unknown)}")
    return ProbeResult(True, f"matched on {len(bodies)} probe bodies")


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
) -> None:
    """Fire-and-forget `probe_worker`, if there is a loop to fire it on.

    Called from the registry's synchronous worker-added hook. Outside a running
    loop (unit tests, sync embedders) this does nothing rather than spinning up
    an event loop behind the caller's back.
    """
    if worker.engine != EngineType.SGLANG:
        # /v1/tokenize is sglang's. vLLM's equivalent differs in both path and
        # payload; probing it blindly would just produce 404 noise.
        return
    if not hasher.can_render(worker.model_name, worker.engine):
        # This model's tokenizer has already been ruled out, so every body would
        # come back "router declined to render" and the verdict would be Unknown
        # before we asked anything. The comparison is worth nothing here; the
        # /get_server_info round trip is all that would be left of it. Mirrors
        # the Rust probe's `hasher.is_enabled()` gate. Not a guarantee in the
        # other direction: a model nothing has tried yet still gets probed.
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        import httpx

        variant = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Ask what it renders with BEFORE checking whether we agree, so the
            # verdict is about the render we will actually route on.
            if variants is not None and variants.per_worker_enabled:
                variant = await engine_render_variant(client, worker)
                if variant is not None:
                    variants.record(worker.worker_id, variant)
            if variant is None and variants is not None:
                variant = variants.for_worker(worker.worker_id)
            result = await probe_worker(hasher, worker, client=client, variant=variant)
        report(worker, result)

    task = loop.create_task(_run())
    # Hold a reference: a bare create_task can be garbage-collected mid-flight.
    _IN_FLIGHT.add(task)
    task.add_done_callback(_IN_FLIGHT.discard)


_IN_FLIGHT: set[asyncio.Task] = set()
