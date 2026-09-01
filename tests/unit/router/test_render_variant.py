###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The server-side chat-template defaults, and the policy keying off them.

Mirror of the Rust `render_variant` unit tests. The merge semantics asserted
here are `serving_chat.py`'s six lines, and getting any of them backwards
reproduces the failure the module exists to prevent -- a hit rate that is merely
lower than it should be, with nothing anywhere reporting an error.
"""

from __future__ import annotations

import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo, WorkerStatus
from infera.router.kv_event.render_variant import RenderVariant, VariantRegistry
from infera.router.policy.factory import _parse_template_kwargs
from infera.router.policy.kv_event_aware import KvEventAwarePolicy


def _v(d: dict) -> RenderVariant:
    return RenderVariant.from_default_chat_template_kwargs(d)


# ----------------------------------------------------------------------
# RenderVariant — the merge
# ----------------------------------------------------------------------


def test_the_empty_variant_is_a_no_op():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    v = RenderVariant()
    assert v.apply(body) is body, "the common path must not copy"
    assert v.id == 0
    assert v.label() == "default"


def test_a_server_default_reaches_both_the_kwargs_and_the_top_level_field():
    # This is the divergence measured on role1: the engine renders the preamble
    # for "high" and, without this, the router renders the template's fallback.
    out = _v({"reasoning_effort": "high"}).apply({"messages": []})
    assert out["reasoning_effort"] == "high"
    assert out["chat_template_kwargs"]["reasoning_effort"] == "high"


def test_the_request_wins_over_the_server_default():
    # `setdefault`. Getting this backwards would break the requests that are
    # correct today -- the ones that send an explicit effort.
    out = _v({"reasoning_effort": "high", "enable_thinking": True}).apply(
        {"messages": [], "chat_template_kwargs": {"reasoning_effort": "low"}}
    )
    assert out["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert out["reasoning_effort"] == "low", "promotion takes the merged value"


def test_an_explicit_top_level_effort_is_not_overwritten():
    out = _v({"reasoning_effort": "high"}).apply({"messages": [], "reasoning_effort": "low"})
    assert out["reasoning_effort"] == "low"
    assert out["chat_template_kwargs"]["reasoning_effort"] == "high", (
        "the engine leaves the merged dict alone; only the field is guarded"
    )


def test_a_default_that_is_not_an_effort_does_not_invent_one():
    out = _v({"enable_thinking": False}).apply({"messages": []})
    assert "reasoning_effort" not in out
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_apply_does_not_mutate_the_caller_s_body():
    body = {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}
    _v({"reasoning_effort": "high"}).apply(body)
    assert body == {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}, (
        "the request body is shared with the dispatch path; the variant is a "
        "hashing concern and must never change what the client actually gets"
    )


def test_ids_are_stable_across_key_order_and_distinct_across_values():
    a = _v({"reasoning_effort": "high", "enable_thinking": True})
    b = _v({"enable_thinking": True, "reasoning_effort": "high"})
    c = _v({"reasoning_effort": "low", "enable_thinking": True})
    assert a.id == b.id, "serialisation order is the engine's business"
    assert a.id != c.id
    assert a.id != 0, "only the empty variant is 0"


@pytest.mark.parametrize("value", [None, "high", ["high"], 3])
def test_a_missing_or_non_object_report_is_the_empty_variant(value):
    # sglang rejects a non-dict itself; we do not guess at one.
    assert RenderVariant.from_default_chat_template_kwargs(value).is_empty()


# ----------------------------------------------------------------------
# VariantRegistry — the two tiers
# ----------------------------------------------------------------------


def test_an_unasked_worker_gets_the_fleet_default():
    reg = VariantRegistry(_v({"reasoning_effort": "high"}))
    assert reg.for_worker("never-probed").label() == 'reasoning_effort="high"'


def test_a_workers_own_report_wins_over_the_flag():
    # The flag is a guess about the fleet; /get_server_info is the fleet.
    reg = VariantRegistry(_v({"reasoning_effort": "high"}))
    reg.record("w1", _v({"reasoning_effort": "low"}))
    assert reg.for_worker("w1").label() == 'reasoning_effort="low"'
    assert reg.for_worker("w2").label() == 'reasoning_effort="high"'


def test_the_kill_switch_pins_everything_to_the_flag():
    reg = VariantRegistry(_v({"reasoning_effort": "high"}), enabled=False)
    reg.record("w1", _v({"reasoning_effort": "low"}))
    assert reg.for_worker("w1").label() == 'reasoning_effort="high"', (
        "off means off, even for a worker that answered"
    )


def test_a_departed_worker_is_forgotten():
    """Asserted through `for_worker`, the only accessor the policy uses: a
    variant that outlives its worker keeps a hash-cache key alive for a worker
    that will never be asked about again."""
    reg = VariantRegistry()
    reg.record("w1", _v({"reasoning_effort": "low"}))
    reg.record("w2", _v({"reasoning_effort": "low"}))
    reg.retain(lambda wid: wid == "w2")
    assert reg.for_worker("w1").is_empty(), "back to the fleet default"
    assert reg.for_worker("w2").label() == 'reasoning_effort="low"'


# ----------------------------------------------------------------------
# The flag
# ----------------------------------------------------------------------


def test_the_flag_parses_an_object_and_refuses_anything_else():
    assert _parse_template_kwargs('{"reasoning_effort": "high"}').label() == (
        'reasoning_effort="high"'
    )
    assert _parse_template_kwargs(None).is_empty()
    assert _parse_template_kwargs("").is_empty()
    # A typo here fails nothing at runtime -- it just makes the router render
    # the preamble the workers do not. Refusing to start is the cheaper outcome.
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_template_kwargs('{"reasoning_effort": high}')
    with pytest.raises(ValueError, match="must be a JSON object"):
        _parse_template_kwargs('"high"')


# ----------------------------------------------------------------------
# The policy — one hash per variant, and the right one per worker
# ----------------------------------------------------------------------


class _StubKvClient:
    def __init__(self, views: dict[str, set[int]]) -> None:
        self._views = views

    def cache_view(self, worker_id: str, dp_rank=None) -> set[int]:
        return self._views.get(worker_id, set())

    def on_worker_added(self, worker) -> None:
        pass

    def on_worker_removed(self, worker_id: str) -> None:
        pass


class _RecordingHasher:
    """Hashes by the effort that reaches the render, and remembers every call.

    Stands in for a tokenizer: what matters to the policy is only that two
    different renders produce two different block lists, which is exactly what
    a real template does with two different preambles.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    # BlockHasher's gate for `spawn_probe`; these doubles always render.
    def can_render(self, model_id, engine=None) -> bool:
        return True

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        self.calls.append(body)
        effort = body.get("reasoning_effort", "unset")
        return [hash(("preamble", effort)) & 0xFFFF, 4242]


def _worker(worker_id: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        url=f"http://{worker_id}",
        model_name="test/m",
        engine=EngineType.SGLANG,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=f"tcp://{worker_id}:5557",
        kv_block_size=4,
    )


def test_a_uniform_fleet_still_hashes_exactly_once():
    """The whole design rests on this: nobody pays for machinery they do not use."""
    hasher = _RecordingHasher()
    policy = KvEventAwarePolicy(_StubKvClient({}), hasher)  # type: ignore[arg-type]
    policy.pick([_worker("w1"), _worker("w2"), _worker("w3")], {"model": "m"})
    assert len(hasher.calls) == 1


def test_two_variants_hash_twice_and_each_worker_is_scored_on_its_own():
    hasher = _RecordingHasher()
    variants = VariantRegistry()
    variants.record("w1", _v({"reasoning_effort": "high"}))
    variants.record("w2", _v({"reasoning_effort": "low"}))
    hi = hasher.hash_for({"reasoning_effort": "high"}, block_size=4)
    lo = hasher.hash_for({"reasoning_effort": "low"}, block_size=4)
    assert hi != lo
    hasher.calls.clear()

    # w2 holds the blocks for ITS OWN render. Keyed the old way -- one hash for
    # the whole fleet -- those blocks are unreachable and w2 scores zero hits.
    client = _StubKvClient({"w1": set(), "w2": set(lo)})
    policy = KvEventAwarePolicy(client, hasher, variants=variants)  # type: ignore[arg-type]

    picked, blocks = policy.pick([_worker("w1"), _worker("w2")], {"model": "m"})
    assert len(hasher.calls) == 2, "one render per distinct variant, not per worker"
    assert picked.worker.worker_id == "w2"
    assert blocks == lo, "the returned blocks must be the ones w2 will actually report"


def test_a_longer_preamble_is_not_punished_for_being_longer():
    """Cost credits hits; it must not charge misses.

    Two workers, neither holding anything. Their variants render to block lists
    of different lengths. Under `w * (total - hits)` the shorter one wins on
    length alone, which is a property of the template, not of the cache.
    """

    class _LengthHasher:
        # BlockHasher's gate for `spawn_probe`; these doubles always render.
        def can_render(self, model_id, engine=None) -> bool:
            return True

        def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
            return [1, 2, 3, 4] if body.get("reasoning_effort") == "high" else [9]

    def _pick(order):
        # A fresh policy per ordering: `pick` charges the winner's blocks to its
        # recent-load total, so reusing one would let the first call decide the
        # second.
        variants = VariantRegistry()
        variants.record("w1", _v({"reasoning_effort": "high"}))
        variants.record("w2", _v({"reasoning_effort": "low"}))
        policy = KvEventAwarePolicy(
            _StubKvClient({}),
            _LengthHasher(),  # type: ignore[arg-type]
            overlap_weight=20.0,
            variants=variants,
        )
        return policy.pick([_worker(w) for w in order], {"model": "m"})[0].worker.worker_id

    # Both orderings, so this asserts a genuine tie rather than list order:
    # min() keeps the first of equal costs, so a real tie picks whichever came
    # first, and a bias picks the same worker both times. Under the old
    # `w * (total - hits)` the four-block preamble costs 80 against the
    # one-block preamble's 20, and w2 wins from either position.
    assert _pick(["w1", "w2"]) == "w1"
    assert _pick(["w2", "w1"]) == "w2"


def test_a_responses_body_is_normalised_before_the_variant_is_applied():
    """The policy must run the engine's order: `_make_request` first, then
    `_process_messages` (which merges `--default-chat-template-kwargs`).

    Applied the other way round the variant lands on `input`-shaped fields that
    `to_chat_body` rebuilds from scratch, and `/v1/responses` -- alone -- loses
    the variant entirely. The chat-only render-probe corpus reports parity
    while it happens.
    """
    pytest.importorskip("sglang.srt.entrypoints.openai.serving_responses")
    hasher = _RecordingHasher()
    variants = VariantRegistry()
    variants.record("w1", _v({"reasoning_effort": "high"}))
    policy = KvEventAwarePolicy(_StubKvClient({}), hasher, variants=variants)  # type: ignore[arg-type]

    policy.pick([_worker("w1")], {"model": "m", "input": "hi"})

    (rendered,) = hasher.calls
    assert rendered["messages"] == [{"role": "user", "content": "hi"}], (
        "the hasher must be handed a chat body; a Responses body hashes to nothing"
    )
    assert rendered["reasoning_effort"] == "high", (
        "the worker's server-side template default must reach the render"
    )


def test_a_responses_image_request_still_trips_the_multimodal_guard():
    """The guard reads the body the hasher hashed, not the one the client sent.

    `parse_cache_hints` / `extract_image_keys` key off `messages`/`images`,
    which a Responses body does not have -- it carries `input`. Against the raw
    body every `/v1/responses` request reports text-only, so `w_overlap` stays
    non-zero and image affinity is skipped. That was harmless while such a body
    hashed to nothing; once it hashes a real prefix it is the silent KV
    collision the guard exists for, because the image placeholder token id is
    the same whichever image was sent.
    """
    pytest.importorskip("sglang.srt.entrypoints.openai.serving_responses")
    from infera.router.cache_control import extract_image_keys, parse_cache_hints
    from infera.router.kv_event import responses_input

    def _body(url: str) -> dict:
        return {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this"},
                        {"type": "input_image", "image_url": url},
                    ],
                }
            ],
        }

    raw = _body("https://x/cat.png")
    assert not parse_cache_hints(raw).has_multimodal_content, (
        "precondition: the raw Responses body is invisible to the guard"
    )

    base = responses_input.normalised(raw)
    assert parse_cache_hints(base).has_multimodal_content
    cat = extract_image_keys(base)
    dog = extract_image_keys(responses_input.normalised(_body("https://x/dog.png")))
    assert cat and dog and cat != dog, "affinity must distinguish the images the text hash cannot"
