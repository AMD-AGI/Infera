###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reading token ids out of engine chunks, and refusing to guess.

Two engines report ids in two different places, and the field has moved between
releases. What matters more than covering every shape is the failure mode: an
unrecognised chunk must produce None so the caller falls back to carrying text,
never a partial list that looks like the truth.
"""

from __future__ import annotations

from infera.common.worker_pool import EngineType
from infera.router.token_ids import (
    deltas_from_chunk,
    prompt_from_chunk,
    strip_token_ids,
    supports_streaming_ids,
)


def test_vllm_reports_deltas_per_choice():
    chunk = {"choices": [{"delta": {"content": "hi"}, "token_ids": [15339, 1917]}]}
    assert deltas_from_chunk(chunk) == [15339, 1917]


def test_vllm_reports_the_prompt_at_the_top_level():
    assert prompt_from_chunk({"prompt_token_ids": [1, 2, 3], "choices": []}) == [1, 2, 3]


def test_sglang_reports_deltas_under_its_own_extension():
    chunk = {"choices": [{"text": "hi"}], "sglext": {"completion_token_ids": [[4, 5]]}}
    assert deltas_from_chunk(chunk) == [4, 5]


def test_sglang_reports_the_prompt_under_the_same_extension():
    assert prompt_from_chunk({"sglext": {"prompt_token_ids": [7, 8]}}) == [7, 8]


def test_a_chunk_without_ids_reads_as_absent():
    """The ordinary case for an engine that was never asked. Absent is not an
    error -- the caller carries text instead."""
    assert deltas_from_chunk({"choices": [{"delta": {"content": "hi"}}]}) is None
    assert prompt_from_chunk({"choices": [{"delta": {"content": "hi"}}]}) is None


def test_a_malformed_id_list_is_rejected_whole():
    """Filtering out the bad entries would leave a plausible prefix, and
    resuming from a prefix drops output the client already read."""
    assert deltas_from_chunk({"choices": [{"token_ids": [1, "two", 3]}]}) is None
    assert deltas_from_chunk({"choices": [{"token_ids": [1, None]}]}) is None
    assert deltas_from_chunk({"choices": [{"token_ids": "12"}]}) is None


def test_booleans_are_not_token_ids():
    # bool is an int in Python; a JSON true here means the field is not ids.
    assert deltas_from_chunk({"choices": [{"token_ids": [True, False]}]}) is None


def test_nonsense_input_is_survived():
    for junk in (None, [], "text", 42, {"choices": "no"}, {"choices": [None]}):
        assert deltas_from_chunk(junk) is None
        assert prompt_from_chunk(junk) is None


def test_an_empty_delta_list_is_not_absent():
    """A chunk that generated nothing is different from one that did not say --
    the first still means the engine is reporting ids."""
    assert deltas_from_chunk({"choices": [{"token_ids": []}]}) == []


def test_stripping_leaves_the_response_as_the_client_expects():
    chunk = {
        "choices": [{"delta": {"content": "hi"}, "token_ids": [1], "prompt_token_ids": [2]}],
        "prompt_token_ids": [2, 3],
        "sglext": {"completion_token_ids": [[1]]},
    }
    assert strip_token_ids(chunk) is True
    assert chunk == {"choices": [{"delta": {"content": "hi"}}]}


def test_stripping_reports_when_there_was_nothing_to_strip():
    """Lets the caller skip re-serialising a chunk it did not change."""
    chunk = {"choices": [{"delta": {"content": "hi"}}]}
    assert strip_token_ids(chunk) is False
    assert chunk == {"choices": [{"delta": {"content": "hi"}}]}


def test_sglang_is_not_asked_for_ids_on_streaming_chat():
    """It rejects the request rather than ignoring the field, so asking would
    turn every migratable chat request into a 400."""
    assert supports_streaming_ids(EngineType.SGLANG, "/v1/chat/completions") is False
    assert supports_streaming_ids(EngineType.SGLANG, "/v1/completions") is True


def test_vllm_is_asked_on_both_endpoints():
    assert supports_streaming_ids(EngineType.VLLM, "/v1/chat/completions") is True
    assert supports_streaming_ids(EngineType.VLLM, "/v1/completions") is True


def test_an_unknown_engine_is_not_asked():
    """An engine that errors on an unknown parameter would fail requests that
    work today, so silence is the safe default."""
    assert supports_streaming_ids(EngineType.ATOM, "/v1/completions") is False
    assert supports_streaming_ids(None, "/v1/completions") is False
