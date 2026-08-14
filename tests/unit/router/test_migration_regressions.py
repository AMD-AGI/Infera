###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Requests whose shape the carried prefix cannot represent.

Every case here produced output that was wrong rather than absent, which is the
worse failure: a severed stream is visibly a failure, while a continuation built
from the wrong prefix reads as the model losing the plot.
"""

from __future__ import annotations

import json

from infera.router.migration import MigrationState


def completion_chunk(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'text': text}]})}\n\n".encode()


def exact_chunk(content: str, ids: list[int], prompt_ids: list[int] | None = None) -> bytes:
    obj = {"choices": [{"delta": {"content": content}, "token_ids": ids}]}
    if prompt_ids is not None:
        obj["prompt_token_ids"] = prompt_ids
    return f"data: {json.dumps(obj)}\n\n".encode()


def test_a_pre_tokenized_prompt_is_never_carried_as_text():
    """`prompt` may be a token array, which engines accept verbatim. Formatting
    one into a string yields its Python repr -- the next worker would be asked
    to continue the literal text "[1, 2, 3]hello"."""
    st = MigrationState({"prompt": [1, 2, 3]}, limit=1, path="/v1/completions")
    st.observe(completion_chunk("hello"))  # no ids: the text path is all there is

    assert not st.can_migrate(), "a token array cannot be extended with text"


def test_a_batch_prompt_is_never_migrated():
    """A list of prompts is several generations at once. There is no single
    prefix to carry, and formatting the list would produce its repr."""
    st = MigrationState({"prompt": ["a", "b"]}, limit=1, path="/v1/completions")
    st.observe(completion_chunk("hello"))

    assert not st.can_migrate()


def test_a_pre_tokenized_prompt_still_migrates_exactly():
    """The id path builds `prompt_ids + output_ids`, which is correct for a
    token array -- it is only the text path that cannot express one."""
    st = MigrationState({"prompt": [1, 2, 3]}, limit=1, path="/v1/completions")
    st.observe(exact_chunk("hello", [42], prompt_ids=[1, 2, 3]))

    assert st.can_migrate()
    cont = st.next_continuation()
    assert cont.exact
    assert cont.body["prompt"] == [1, 2, 3, 42]


def test_more_than_one_choice_is_never_migrated():
    """Only the first choice is accumulated, so every other one would resume
    from a prefix belonging to the first: not a gap, but the wrong content."""
    st = MigrationState({"prompt": "hi", "n": 2}, limit=1, path="/v1/completions")
    st.observe(completion_chunk("hello"))

    assert not st.can_migrate()


def test_several_choices_in_a_chunk_are_noticed_even_without_n():
    """`best_of`, or an engine that returns several choices for its own
    reasons, reaches the same place by a different route."""
    obj = {"choices": [{"index": 0, "text": "AAA"}, {"index": 1, "text": "BBB"}]}
    st = MigrationState({"prompt": "hi"}, limit=1, path="/v1/completions")
    st.observe(f"data: {json.dumps(obj)}\n\n".encode())

    assert not st.can_migrate()
    assert st.produced_text == "", "a prefix covering one choice of several is not a prefix"


def test_several_completions_are_rejected_on_the_exact_path_too():
    """Exact ids do not help here: they are accumulated for one choice, so a
    second one would resume from the first one's tokens."""
    st = MigrationState({"prompt": "hi", "n": 2}, limit=1, path="/v1/completions")
    obj = {"choices": [{"text": "A", "token_ids": [1]}], "prompt_token_ids": [9]}
    st.observe(f"data: {json.dumps(obj)}\n\n".encode())

    assert not st.can_migrate()


def test_a_batch_of_token_arrays_is_still_a_batch():
    st = MigrationState({"prompt": [[1, 2], [3, 4]]}, limit=1, path="/v1/completions")
    assert not st.can_migrate()


def test_an_empty_prompt_list_is_not_read_as_a_token_array():
    """Nothing sensible can be carried, and treating it as a single generation
    would put it back on the text path it cannot take."""
    st = MigrationState({"prompt": []}, limit=1, path="/v1/completions")
    assert not st.can_migrate()


def test_best_of_is_rejected_like_n():
    st = MigrationState({"prompt": "hi", "best_of": 3}, limit=1, path="/v1/completions")
    assert not st.can_migrate()


def test_ids_are_still_stripped_after_migration_is_ruled_out():
    """The ids are requested by the router and the client never asked for them.
    Whether a request is still migratable is the router's business; it must not
    change the shape of what the caller receives."""
    st = MigrationState({"messages": []}, limit=1, path="/v1/chat/completions")
    st.observe(exact_chunk("hi", [1], prompt_ids=[0]))

    tool = {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}
    st.observe(f"data: {json.dumps(tool)}\n\n".encode())
    assert st.poisoned

    after = st.observe(exact_chunk("more", [12]))
    assert b"token_ids" not in after, "ids leaked once the request stopped being migratable"
    assert b"more" in after


def test_ids_are_stripped_even_after_an_unparseable_chunk():
    """The same holds for the other way a request stops being migratable."""
    st = MigrationState({"messages": []}, limit=1, path="/v1/chat/completions")
    st.observe(b"data: {not json\n\n")
    assert st.poisoned

    after = st.observe(exact_chunk("more", [12]))
    assert b"token_ids" not in after


def test_an_unparseable_chunk_reaches_the_client_unchanged():
    """What the router cannot read, it must not rewrite."""
    st = MigrationState({"messages": []}, limit=1, path="/v1/chat/completions")
    raw = b"data: {not json\n\n"
    assert st.observe(raw) == raw
