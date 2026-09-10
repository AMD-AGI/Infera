###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The long-context correctness probe as pure logic.

The probe decides whether an e2e case passes, so its classifier has to be right
before a node reservation depends on it.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import correctness

# --- long-context ledger ---


def test_the_ledger_is_deterministic():
    assert correctness.build_longctx_prompt() == correctness.build_longctx_prompt()


def test_the_code_appears_exactly_once_and_mid_document():
    prompt = correctness.build_longctx_prompt()
    ledger = prompt.split("\n\nQuestion:")[0]
    assert ledger.count(correctness.LONGCTX_ANSWER) == 1
    where = ledger.index(correctness.LONGCTX_ANSWER) / len(ledger)
    assert 0.3 < where < 0.8, f"needle at {where:.0%} is too close to an edge"


def test_the_ledger_spans_many_kv_blocks_but_fits_the_tightest_context():
    prompt = correctness.build_longctx_prompt()
    # ~4 chars/token: long enough to be a real prefill, short enough for --max-model-len 9472.
    assert 6000 < len(prompt) < 20000, len(prompt)


def test_the_question_is_the_last_thing_the_model_reads():
    assert correctness.build_longctx_prompt().rstrip().endswith("Answer:")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("4731", True),
        (" The archive access code is 4731.", True),
        ("code: 4731\n", True),
        ("1234", False),
        ("I could not find it in the ledger.", False),
        ("", False),
        ("   ", False),
        ("\ufffd\ufffd 4731", False),  # the digits, but the reply is decode garbage
    ],
)
def test_longctx_classifier(reply, expected):
    assert correctness.is_longctx_correct(reply) is expected


# Replies below are verbatim from real runs on MI355X (gpt-oss, Kimi-K2.6, GLM-5.1-FP8).


@pytest.mark.parametrize(
    "reply",
    [
        # answered, then continued the document or asked its own question
        "4731 --- Ledger 001: depot BRAVO shipped 47 crates of copper lugs on day 2. Ledger 002:",
        "4731 Question: according to the ledger above, how many crates of steel shims"
        " were shipped from depot CIVET on day 10?",
        "4731 We need to output digits only: 4731. The conversation: The user gave ledger"
        ' and asked: "according to the ledger above,',
        "4731 I need to find the archive access code for this quarter in the ledger."
        " Looking through the entries, I find: - Ledger 071:",
        # correct, but the tail is numeric: must not read as salad
        "4731 The answer is: 4731 2 1 0 0 1 1 0 0 0 0",
    ],
)
def test_a_coherent_reply_carrying_the_code_passes(reply):
    assert correctness.is_longctx_correct(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        # a different number that merely contains the code
        "47312 1 0.9 1,0,0,0,0,0,1,0,0,0.",
        # the code, then digit salad: pure ASCII, so looks_garbage is blind to it
        "4731.0 1. 1.0,0.0,0. needle,2.0,0.0,0.",
        "473,1,4,2,0,1,0,0,1,0,0,1,0,80,0,0,1,0,1,5,0,0,4731 1,4,0,0,0,0,0,0,0,0,0,",
        # wrong digit
        "4732 1. 1.1.1.1.3.1.0.1.0.1.0.0.",
        "4739 1. 1.5.1.1.1.1.0.0.0.0. Unlike1.5",
    ],
)
def test_a_corrupt_reply_fails_even_when_it_contains_the_digits(reply):
    assert correctness.is_longctx_correct(reply) is False


def test_the_degeneracy_check_leaves_the_counting_probe_alone():
    """Counting is legitimately almost all digits; folding this into looks_garbage breaks it."""
    counting = "6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
    assert correctness.looks_degenerate(counting) is True
    assert correctness.is_counting_correct(counting) is True


def test_a_short_reply_is_never_called_degenerate():
    """The ideal answer is the bare code, which is all digits by nature."""
    assert correctness.looks_degenerate("4731") is False
    assert correctness.is_longctx_correct("4731") is True
