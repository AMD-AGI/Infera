###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Carrying a generation to another worker without the client noticing.

The whole feature rests on one property: the text handed to the next worker is
exactly what the client already received. If it is short, the client reads a
gap; if it is long, it reads a repetition. Both are worse than the failure
migration exists to hide, so most of this file is about that equality.
"""

from __future__ import annotations

import json

from infera.router.migration import MigrationState


def chat_chunk(content: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n".encode()


def completion_chunk(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'text': text}]})}\n\n".encode()


def test_the_carried_text_is_what_the_client_received():
    st = MigrationState({"messages": [{"role": "user", "content": "hi"}]}, limit=1)
    for piece in ("Hello", ",", " world"):
        st.observe(chat_chunk(piece))
    assert st.produced_text == "Hello, world"


def test_completions_carry_the_same_way():
    st = MigrationState({"prompt": "once"}, limit=1)
    st.observe(completion_chunk(" upon"))
    st.observe(completion_chunk(" a time"))
    assert st.produced_text == " upon a time"


def test_several_chunks_in_one_frame_are_all_counted():
    # A slow reader coalesces frames; the transport is free to deliver them
    # together and the accumulated text must not depend on that.
    st = MigrationState({"prompt": ""}, limit=1)
    st.observe(completion_chunk("a") + completion_chunk("b") + completion_chunk("c"))
    assert st.produced_text == "abc"
    assert st.produced_tokens == 3


def test_the_done_sentinel_is_not_carried():
    """`[DONE]` belongs to the stream the client is reading, not to the
    generation. Carried into a prompt it would become model input."""
    st = MigrationState({"prompt": ""}, limit=1)
    st.observe(completion_chunk("x"))
    st.observe(b"data: [DONE]\n\n")
    assert st.produced_text == "x"


def test_a_chat_continuation_appends_an_assistant_turn():
    body = {"messages": [{"role": "user", "content": "count"}], "max_tokens": 10}
    st = MigrationState(body, limit=1)
    for piece in ("one", " two"):
        st.observe(chat_chunk(piece))

    nxt = st.next_continuation().body
    assert nxt["messages"][-1] == {"role": "assistant", "content": "one two"}
    # Two tokens spent, eight left: the next worker finishes the answer rather
    # than producing a second one of full length.
    assert nxt["max_tokens"] == 8
    # The original is untouched -- a failed migration must be able to fall back.
    assert body["max_tokens"] == 10
    assert len(body["messages"]) == 1


def test_a_completion_continuation_extends_the_prompt():
    st = MigrationState({"prompt": "once", "max_tokens": 5}, limit=1)
    st.observe(completion_chunk(" upon"))
    nxt = st.next_continuation().body
    assert nxt["prompt"] == "once upon"
    assert nxt["max_tokens"] == 4


def test_the_budget_never_reaches_zero():
    """A request for zero tokens is rejected, which would turn a migration the
    client cannot see into an error it can."""
    st = MigrationState({"prompt": "", "max_tokens": 2}, limit=1)
    for _ in range(5):
        st.observe(completion_chunk("x"))
    assert st.next_continuation().body["max_tokens"] == 1


def test_a_request_without_a_budget_stays_without_one():
    st = MigrationState({"prompt": ""}, limit=1)
    st.observe(completion_chunk("x"))
    assert "max_tokens" not in st.next_continuation().body


def test_an_unparseable_chunk_disables_migration():
    """Continuing from a prefix that could not be fully reconstructed would
    drop output the client already read -- worse than not migrating."""
    st = MigrationState({"prompt": ""}, limit=1)
    st.observe(completion_chunk("good"))
    st.observe(b"data: {not json\n\n")
    assert st.poisoned
    assert not st.can_migrate()


def test_chunks_after_poisoning_are_ignored():
    st = MigrationState({"prompt": ""}, limit=1)
    st.observe(b"data: {not json\n\n")
    st.observe(completion_chunk("later"))
    assert st.produced_text == ""


def test_the_migration_limit_is_enforced():
    st = MigrationState({"prompt": ""}, limit=2)
    assert st.can_migrate()
    st.next_continuation()
    assert st.can_migrate()
    st.next_continuation()
    assert not st.can_migrate(), "a request must not migrate forever"


def test_a_zero_limit_disables_migration():
    st = MigrationState({"prompt": ""}, limit=0)
    assert not st.can_migrate()


def exact_chunk(content: str, ids: list[int], prompt_ids: list[int] | None = None) -> bytes:
    """A chunk from an engine that was asked to report token ids."""
    obj = {"choices": [{"delta": {"content": content}, "token_ids": ids}]}
    if prompt_ids is not None:
        obj["prompt_token_ids"] = prompt_ids
    return f"data: {json.dumps(obj)}\n\n".encode()


def test_exact_ids_are_carried_instead_of_text():
    """The point of the whole id path: the next worker resumes from the
    sequence the model sampled, not from a re-encoding of the words."""
    st = MigrationState({"prompt": "hi", "max_tokens": 10}, limit=1, path="/v1/completions")
    st.observe(exact_chunk("Hel", [100], prompt_ids=[1, 2]))
    st.observe(exact_chunk("lo", [200]))

    assert st.is_exact()
    cont = st.next_continuation()
    assert cont.exact
    assert cont.body["prompt"] == [1, 2, 100, 200]
    assert cont.body["max_tokens"] == 8, "two real tokens, not two chunks"


def test_the_token_count_is_the_real_one_when_ids_are_known():
    """Without ids this is chunks-with-text, which is only a good guess. A
    chunk carrying several tokens makes the two differ."""
    st = MigrationState({"prompt": "", "max_tokens": 20}, limit=1, path="/v1/completions")
    st.observe(exact_chunk("a b c", [1, 2, 3], prompt_ids=[9]))
    assert st.produced_tokens == 3
    assert st.next_continuation().body["max_tokens"] == 17


def test_ids_without_a_prompt_are_not_enough():
    """Appending exact output ids to a re-encoded prompt just moves the
    ambiguity to the other end of the sequence."""
    st = MigrationState({"prompt": "hi"}, limit=1, path="/v1/completions")
    st.observe(exact_chunk("out", [5]))  # engine never sent prompt ids
    assert not st.is_exact()
    assert st.next_continuation().exact is False


def test_text_the_ids_do_not_cover_abandons_the_id_path():
    """Half the output accounted for is worse than none: continuing from those
    ids would drop the text they omit, which the client has already read."""
    st = MigrationState({"prompt": ""}, limit=1, path="/v1/completions")
    st.observe(exact_chunk("counted", [1], prompt_ids=[0]))
    st.observe(completion_chunk("unaccounted"))
    assert not st.is_exact()
    cont = st.next_continuation()
    assert cont.exact is False
    assert cont.body["prompt"] == "countedunaccounted", "falls back to the full text"


def test_an_exact_chat_continuation_moves_to_the_completions_path():
    """Chat has no pre-tokenized entry, so exactness costs a change of
    endpoint; the caller is told so it can convert the replies back."""
    st = MigrationState(
        {"messages": [{"role": "user", "content": "hi"}]},
        limit=1,
        path="/v1/chat/completions",
    )
    st.observe(exact_chunk("part", [7], prompt_ids=[1, 2]))

    cont = st.next_continuation()
    assert cont.exact
    assert cont.path == "/v1/completions"
    assert cont.body["prompt"] == [1, 2, 7]
    assert "messages" not in cont.body


def test_a_chat_request_with_tools_keeps_the_text_path():
    """A completions request cannot emit a tool call. Losing that halfway
    through an answer is worse than a token boundary that might differ."""
    st = MigrationState(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        },
        limit=1,
        path="/v1/chat/completions",
    )
    st.observe(exact_chunk("part", [7], prompt_ids=[1, 2]))

    cont = st.next_continuation()
    assert cont.exact is False
    assert cont.path == "/v1/chat/completions"
    assert cont.body["messages"][-1]["content"] == "part"


def test_structured_output_also_keeps_the_text_path():
    st = MigrationState(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
        limit=1,
        path="/v1/chat/completions",
    )
    st.observe(exact_chunk("{", [7], prompt_ids=[1]))
    assert st.next_continuation().exact is False


def test_the_ids_are_taken_out_before_the_client_sees_them():
    """Enabling migration must not change the shape of the response."""
    st = MigrationState({"prompt": ""}, limit=1, path="/v1/completions")
    out = st.observe(exact_chunk("hi", [42], prompt_ids=[1]))

    assert b"token_ids" not in out
    assert b"prompt_token_ids" not in out
    assert json.loads(out.split(b"data: ")[1])["choices"][0]["delta"]["content"] == "hi"
    assert st.produced_text == "hi", "still recorded, just not forwarded"


def test_a_chunk_without_ids_is_forwarded_untouched():
    """No ids means nothing to strip, and the engine's own bytes go through
    without a re-serialisation that could perturb them."""
    st = MigrationState({"prompt": ""}, limit=1, path="/v1/completions")
    original = completion_chunk("hi")
    assert st.observe(original) is original


def test_the_router_only_field_is_not_passed_on():
    st = MigrationState({"prompt": "", "return_token_ids": True}, limit=1, path="/v1/completions")
    st.observe(exact_chunk("hi", [1], prompt_ids=[0]))
    assert "return_token_ids" not in st.next_continuation().body


def tool_call_chunk() -> bytes:
    delta = {"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": '{"a"'}}]}
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


def test_a_tool_call_stops_the_text_path():
    """Tool calls arrive under their own key, not as content, so carried text
    would omit them entirely -- and the client, already holding half a call,
    would be sent a whole one after the migration."""
    st = MigrationState({"messages": []}, limit=1, path="/v1/chat/completions")
    st.observe(chat_chunk("Let me check"))
    st.observe(tool_call_chunk())

    assert st.poisoned
    assert not st.can_migrate()


def test_hidden_reasoning_stops_the_text_path():
    """A reasoning parser moves the model's thinking out of `content`. Carrying
    only what the client saw would resume from an answer with its reasoning
    removed, which is missing output rather than imprecise output."""
    st = MigrationState({"messages": []}, limit=1, path="/v1/chat/completions")
    st.observe(b'data: {"choices": [{"delta": {"reasoning_content": "hmm..."}}]}\n\n')

    assert st.poisoned


def test_exact_ids_survive_a_tool_call():
    """The ids are the tokens behind whatever the parser emitted, so they carry
    the call itself. Only the text path is defeated by it."""
    st = MigrationState({"messages": []}, limit=1, path="/v1/chat/completions")
    st.observe(exact_chunk("Let me check", [10], prompt_ids=[1]))
    obj = {
        "choices": [{"delta": {"tool_calls": [{"index": 0}]}, "token_ids": [20, 21]}],
    }
    st.observe(f"data: {json.dumps(obj)}\n\n".encode())

    assert not st.poisoned
    assert st.is_exact()
    assert st.next_continuation().body["prompt"] == [1, 10, 20, 21]


def test_keepalives_and_role_frames_add_nothing():
    # An opening chat frame carries `role` and no content; comment lines are
    # heartbeats. Neither is output, and counting them would shorten the answer.
    st = MigrationState({"prompt": ""}, limit=1)
    st.observe(b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n')
    st.observe(b": keepalive\n\n")
    st.observe(completion_chunk("real"))
    assert st.produced_text == "real"
    assert st.produced_tokens == 1
