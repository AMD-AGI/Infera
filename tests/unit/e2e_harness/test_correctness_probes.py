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
