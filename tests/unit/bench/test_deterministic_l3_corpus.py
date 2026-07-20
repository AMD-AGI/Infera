###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The deterministic L3 bench's whole point is that the corpus
reproduces bit-exact across runs. Test that contract — if it ever
breaks, kvd validation is silently meaningless."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Add bench dir to sys.path so we can import the corpus module
# without making `bench/` a package the regular install picks up.
_BENCH_DIR = Path(__file__).resolve().parents[3] / "bench" / "kvcache"
sys.path.insert(0, str(_BENCH_DIR))
corpus_mod = importlib.import_module("deterministic_l3.corpus")
sys.path.remove(str(_BENCH_DIR))


def test_user_turn_is_deterministic():
    """Same (session_id, turn_idx) → byte-identical text."""
    a = corpus_mod.gen_user_turn(7, 3, 500)
    b = corpus_mod.gen_user_turn(7, 3, 500)
    assert a == b
    assert len(a) > 0


def test_user_turn_varies_by_session():
    """Different sessions must NOT collide."""
    assert corpus_mod.gen_user_turn(0, 5, 500) != corpus_mod.gen_user_turn(1, 5, 500)


def test_user_turn_varies_by_turn():
    """Different turns within a session must NOT collide."""
    assert corpus_mod.gen_user_turn(3, 0, 500) != corpus_mod.gen_user_turn(3, 1, 500)


def test_user_turn_target_token_count_is_respected_approximately():
    """We over-target by ~1.3× ratio; allow ±20% slack but the
    delivered word count must scale with target."""
    short = corpus_mod.gen_user_turn(0, 0, 500)
    long_ = corpus_mod.gen_user_turn(0, 0, 5000)
    assert len(long_.split()) > 5 * len(short.split())


def test_session_role_alternation():
    """Chat templates require strict role alternation; the final
    message must be `user` (engine generates the assistant turn)."""
    msgs = corpus_mod.gen_session(0, 4, 100)
    # Should be user, assistant, user, assistant, user, assistant, user
    # = 7 messages for 4 user turns.
    assert len(msgs) == 2 * 4 - 1
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "user"
    for i, m in enumerate(msgs):
        expected = "user" if i % 2 == 0 else "assistant"
        assert m["role"] == expected, f"message {i} role mismatch"


def test_session_growing_prefix_is_bit_identical_across_calls():
    """The growing-prefix invariant: session S's first K turns'
    bytes are a prefix of its first K+1 turns. If this isn't true,
    kvd's content-hashed block keys won't line up across turn-N and
    turn-(N+1) requests of the same session."""
    msgs_4 = corpus_mod.gen_session(11, 4, 200)
    msgs_5 = corpus_mod.gen_session(11, 5, 200)
    # First 2*K-1 entries of the K+1-turn session must match the
    # whole K-turn session including the assistant turns we record.
    assert msgs_5[: len(msgs_4)] == msgs_4


def test_session_assistant_text_is_constant():
    """Assistant turns must be a fixed function of (turn_idx)
    so re-running with the same params produces identical bytes."""
    msgs = corpus_mod.gen_session(0, 5, 100)
    asst_turns = [m for m in msgs if m["role"] == "assistant"]
    # Turn N's assistant says "OK turn N." with N 0-indexed.
    for i, m in enumerate(asst_turns):
        assert m["content"] == f"OK turn {i}."
