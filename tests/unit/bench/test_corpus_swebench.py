###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the SWE-bench-Pro trace converter.

We mock the HF dataset payload — these tests never hit the network.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parents[3] / "bench" / "kvcache"
sys.path.insert(0, str(_BENCH_DIR))
mod = importlib.import_module("deterministic_l3.corpus_swebench")
sys.path.remove(str(_BENCH_DIR))


# ----------------------------------------------------------------------
# Fixtures — synthetic trials in the same shape as the real dataset
# ----------------------------------------------------------------------


def _trial(*pairs: tuple[str, str]) -> dict:
    """Build a trial in the dataset's native shape — list of
    `{"from": ..., "value": ...}` dicts wrapped in `conversations`."""
    return {"conversations": [{"from": role, "value": text} for role, text in pairs]}


@pytest.fixture
def short_user_only_trial():
    return _trial(("human", "do thing"))


@pytest.fixture
def alternating_trial():
    return _trial(
        ("human", "request 1"),
        ("gpt", "response 1"),
        ("human", "request 2"),
        ("gpt", "response 2"),
        ("human", "request 3"),
    )


@pytest.fixture
def trailing_assistant_trial():
    # Bench should drop trailing assistant turn so request ends on user
    return _trial(
        ("human", "u1"),
        ("gpt", "a1"),
        ("human", "u2"),
        ("gpt", "a2"),
    )


# ----------------------------------------------------------------------
# trial_to_session
# ----------------------------------------------------------------------


def test_trial_to_session_maps_roles():
    """`human` → `user`, `gpt` → `assistant`. This is the only
    mapping the real dataset uses; if a future version adds new
    role strings the converter must explicitly handle them."""
    trial = _trial(("human", "hi"), ("gpt", "hello"))
    sess = mod.trial_to_session(trial, 0)
    assert sess["messages"] == [
        {"role": "user", "content": "hi"},
        # NOTE: trailing assistant trimmed
    ]


def test_trial_to_session_preserves_alternation(alternating_trial):
    sess = mod.trial_to_session(alternating_trial, 7)
    assert sess["session_id"] == 7
    roles = [m["role"] for m in sess["messages"]]
    # Trailing assistant turn (if any) is trimmed; this trial ends on user,
    # so all 5 messages survive.
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_trial_to_session_trims_trailing_assistant(trailing_assistant_trial):
    sess = mod.trial_to_session(trailing_assistant_trial, 0)
    # Last msg must be user — replay sends "next-token-predict" requests.
    assert sess["messages"][-1]["role"] == "user"
    # And the truly-trailing assistant is gone.
    roles = [m["role"] for m in sess["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_trial_to_session_rejects_empty():
    assert mod.trial_to_session({"conversations": []}, 0) is None
    assert mod.trial_to_session({}, 0) is None
    assert mod.trial_to_session({"conversations": None}, 0) is None


def test_trial_to_session_rejects_unknown_role():
    """If a future dataset version adds `system` / tool roles the
    converter must NOT silently coerce — we want a known regression
    signal, not garbage output."""
    trial = _trial(("system", "you are a"), ("human", "x"))
    assert mod.trial_to_session(trial, 0) is None


def test_trial_to_session_rejects_non_string_value():
    """The real format always has `value` as str. A multimodal /
    tool-call entry would be a dict — refuse it loudly via None."""
    trial = {
        "conversations": [
            {"from": "human", "value": {"type": "image", "data": "..."}},
        ]
    }
    assert mod.trial_to_session(trial, 0) is None


def test_trial_to_session_rejects_assistant_first():
    """A trace that opens with assistant has no user prompt to
    seed from — we can't build a `messages` list whose first entry
    is a user turn (chat templates require user-led)."""
    trial = _trial(("gpt", "I am"), ("human", "ok"))
    assert mod.trial_to_session(trial, 0) is None


# ----------------------------------------------------------------------
# cap_turns
# ----------------------------------------------------------------------


def test_cap_turns_truncates_to_user_boundary(alternating_trial):
    sess = mod.trial_to_session(alternating_trial, 0)
    capped = mod.cap_turns(sess, max_user_turns=2)
    roles = [m["role"] for m in capped["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_cap_turns_no_op_when_under_cap(alternating_trial):
    sess = mod.trial_to_session(alternating_trial, 0)
    capped = mod.cap_turns(sess, max_user_turns=100)
    assert capped == sess


def test_cap_turns_zero_is_pass_through(alternating_trial):
    """A cap of 0 (or negative) is a sentinel for "no cap" so users
    can disable the safety bound."""
    sess = mod.trial_to_session(alternating_trial, 0)
    assert mod.cap_turns(sess, max_user_turns=0) == sess


# ----------------------------------------------------------------------
# select_trials
# ----------------------------------------------------------------------


def _session_of_size(session_id: int, bytes_per_msg: int) -> dict:
    return {
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": "u" * bytes_per_msg},
            {"role": "assistant", "content": "a" * bytes_per_msg},
            {"role": "user", "content": "u" * bytes_per_msg},
        ],
    }


def test_select_smallest():
    trials = [_session_of_size(i, (i + 1) * 100) for i in range(5)]
    picked = mod.select_trials(trials, n=3, strategy="smallest", seed=0)
    # Sizes 100, 200, 300 — sessions 0, 1, 2
    assert [s["session_id"] for s in picked] == [0, 1, 2]


def test_select_largest():
    trials = [_session_of_size(i, (i + 1) * 100) for i in range(5)]
    picked = mod.select_trials(trials, n=2, strategy="largest", seed=0)
    assert [s["session_id"] for s in picked] == [4, 3]


def test_select_first_is_dataset_order():
    trials = [_session_of_size(i, (i + 1) * 100) for i in range(5)]
    picked = mod.select_trials(trials, n=3, strategy="first", seed=0)
    assert [s["session_id"] for s in picked] == [0, 1, 2]


def test_select_random_is_seeded():
    trials = [_session_of_size(i, 100) for i in range(10)]
    a = mod.select_trials(trials, n=5, strategy="random", seed=42)
    b = mod.select_trials(trials, n=5, strategy="random", seed=42)
    assert [s["session_id"] for s in a] == [s["session_id"] for s in b]


def test_select_unknown_strategy_raises():
    with pytest.raises(ValueError):
        mod.select_trials([], n=1, strategy="???", seed=0)


# ----------------------------------------------------------------------
# build_corpus — full pipeline
# ----------------------------------------------------------------------


def test_build_corpus_round_trip(alternating_trial, trailing_assistant_trial):
    raw = [alternating_trial, trailing_assistant_trial, _trial(("human", "single"))]
    corpus, stats = mod.build_corpus(
        raw_trials=raw,
        num_trials=10,
        strategy="first",
        max_turns_per_trial=100,
        seed=0,
    )
    assert corpus["config"]["source_dataset"] == mod.DATASET_REPO_ID
    assert len(corpus["sessions"]) == 3
    # session_ids re-densified
    assert [s["session_id"] for s in corpus["sessions"]] == [0, 1, 2]
    # Stats matches what `replay.py` will count
    assert stats["num_sessions"] == 3
    assert stats["num_user_turns"] == sum(
        1 for s in corpus["sessions"] for m in s["messages"] if m["role"] == "user"
    )


def test_build_corpus_drops_invalid_trials():
    raw = [
        {"conversations": []},  # empty, rejected
        _trial(("gpt", "first")),  # assistant-first, rejected
        _trial(("human", "ok")),  # kept
    ]
    corpus, stats = mod.build_corpus(
        raw_trials=raw,
        num_trials=10,
        strategy="first",
        max_turns_per_trial=100,
        seed=0,
    )
    assert stats["num_sessions"] == 1
    assert stats["skipped_invalid"] == 2
    assert corpus["config"]["raw_trials_skipped_invalid"] == 2


def test_build_corpus_respects_max_turns(alternating_trial):
    """alternating_trial has 3 user turns; cap to 2 → 2 sessions × 2 user turns each."""
    raw = [alternating_trial, alternating_trial]
    corpus, stats = mod.build_corpus(
        raw_trials=raw,
        num_trials=10,
        strategy="first",
        max_turns_per_trial=2,
        seed=0,
    )
    for sess in corpus["sessions"]:
        user_count = sum(1 for m in sess["messages"] if m["role"] == "user")
        assert user_count == 2


def test_build_corpus_num_trials_clamps_to_available():
    """Asking for more trials than the dataset contains returns
    whatever's available rather than crashing."""
    raw = [_trial(("human", "x"))]
    corpus, _ = mod.build_corpus(
        raw_trials=raw,
        num_trials=100,
        strategy="smallest",
        max_turns_per_trial=10,
        seed=0,
    )
    assert len(corpus["sessions"]) == 1


# ----------------------------------------------------------------------
# Output is `replay.py`-compatible
# ----------------------------------------------------------------------


def test_output_shape_matches_replay_expectations(alternating_trial):
    """`replay.py` walks `corpus["sessions"]` and per-session iterates
    user turns via `msgs[: turn_idx * 2 + 1]`. That only works if
    role layout is strict user/assistant/user/.../user."""
    raw = [alternating_trial]
    corpus, _ = mod.build_corpus(
        raw_trials=raw,
        num_trials=10,
        strategy="first",
        max_turns_per_trial=100,
        seed=0,
    )
    sess = corpus["sessions"][0]
    # Same invariants the deterministic-corpus tests pin for corpus.py
    assert sess["messages"][0]["role"] == "user"
    assert sess["messages"][-1]["role"] == "user"
    for i, m in enumerate(sess["messages"]):
        expected = "user" if i % 2 == 0 else "assistant"
        assert m["role"] == expected
