###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The long-context and quicksort correctness probes, as pure logic.

Both probes decide whether an e2e case passes, so their classifiers have to be
right before a node reservation depends on them. Everything they do off the wire
— building the ledger, pulling code out of a reply, executing that code against
``sorted()`` — runs here against hand-written stand-ins for model output, good
and bad, in under a second.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import correctness

GOOD_QUICKSORT = """
def quicksort(nums):
    if len(nums) <= 1:
        return list(nums)
    pivot = nums[len(nums) // 2]
    lo = [x for x in nums if x < pivot]
    mid = [x for x in nums if x == pivot]
    hi = [x for x in nums if x > pivot]
    return quicksort(lo) + mid + quicksort(hi)
"""

# Drops duplicates of the pivot: passes on a plain permutation, fails on (5,1,5,1,5).
# The classic reason a quicksort probe must test more than one sequence.
DROPS_DUPLICATES = """
def quicksort(nums):
    if len(nums) <= 1:
        return list(nums)
    pivot = nums[0]
    lo = [x for x in nums[1:] if x < pivot]
    hi = [x for x in nums[1:] if x > pivot]
    return quicksort(lo) + [pivot] + quicksort(hi)
"""


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


# --- pulling code out of a reply ---


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("```python\ndef f():\n    pass\n```", "def f():\n    pass"),
        ("```py\nx = 1\n```", "x = 1"),
        ("```\nx = 1\n```", "x = 1"),
        ("Sure!\n```python\nx = 1\n```\nHope that helps.", "x = 1"),
        ("def f():\n    pass", "def f():\n    pass"),  # unfenced, still runnable
        ("```python\nfirst = 1\n```\n```python\nsecond = 2\n```", "first = 1"),
        ("", ""),
    ],
)
def test_extract_python_code(reply, expected):
    assert correctness.extract_python_code(reply) == expected


# --- the sandbox ---


def test_a_correct_quicksort_passes():
    ok, detail = correctness.run_quicksort_code(GOOD_QUICKSORT)
    assert ok, detail


def test_quick_sort_and_qsort_are_accepted_names():
    ok, detail = correctness.run_quicksort_code(GOOD_QUICKSORT.replace("quicksort", "quick_sort"))
    assert ok, detail


def test_a_partition_that_drops_duplicates_fails():
    ok, detail = correctness.run_quicksort_code(DROPS_DUPLICATES)
    assert not ok
    assert "5" in detail


@pytest.mark.parametrize(
    "code",
    [
        "def quicksort(nums):\n    return sorted(nums)",
        "def quicksort(nums):\n    out = list(nums)\n    out.sort()\n    return out",
        "quicksort = sorted",  # an alias, so no call syntax to grep for
        "def quicksort(nums):\n    f = sorted\n    return f(nums)",
        "import numpy\ndef quicksort(nums):\n    return list(numpy.sort(nums))",
        "from bisect import insort\ndef quicksort(nums):\n    out = []\n"
        "    for n in nums:\n        insort(out, n)\n    return out",
        "import heapq\ndef quicksort(nums):\n    h = list(nums)\n    heapq.heapify(h)\n"
        "    return [heapq.heappop(h) for _ in range(len(h))]",
    ],
)
def test_delegating_to_a_library_sort_is_not_an_implementation(code):
    """Each of these sorts correctly, so only the static check separates them from a real
    implementation — which is why that check reads the AST rather than grepping for text."""
    ok, detail = correctness.run_quicksort_code(code)
    assert not ok
    assert "instead of implementing quicksort" in detail


@pytest.mark.parametrize(
    "prose",
    [
        "# Hand-rolled, unlike list.sort() or sorted().\n",
        '"""A quicksort. Do not use sorted() or nums.sort() here."""\n',
    ],
)
def test_only_mentioning_a_library_sort_is_not_cheating(prose):
    ok, detail = correctness.run_quicksort_code(prose + GOOD_QUICKSORT)
    assert ok, detail


def test_a_model_that_names_its_function_sort_is_judged_by_running_it():
    """`sort` is a banned name, but a locally defined one is the model's own recursion —
    it fails on the real ground (no quicksort() to call), not on a false cheating charge."""
    ok, detail = correctness.run_quicksort_code(GOOD_QUICKSORT.replace("quicksort", "sort"))
    assert not ok
    assert "defines no quicksort" in detail


@pytest.mark.parametrize(
    ("code", "expect_in_detail"),
    [
        ("", "no code"),
        ("   \n  ", "no code"),
        ("def helper():\n    return 1", "defines no quicksort"),
        ("def quicksort(nums)\n    return nums", "SyntaxError"),
        ("def quicksort(nums):\n    raise ValueError('boom')", "ValueError"),
        ("def quicksort(nums):\n    nums.reverse()", "returned None"),
        ("def quicksort(nums):\n    return list(nums)", "returned"),  # returns it unsorted
    ],
)
def test_broken_replies_fail_with_a_readable_reason(code, expect_in_detail):
    ok, detail = correctness.run_quicksort_code(code)
    assert not ok
    assert expect_in_detail in detail


def test_a_hanging_implementation_is_killed_not_waited_on():
    ok, detail = correctness.run_quicksort_code(
        "def quicksort(nums):\n    while True:\n        pass", timeout=2.0
    )
    assert not ok
    assert "did not finish" in detail


def test_the_sandbox_cannot_litter_the_repo(tmp_path, monkeypatch):
    """Model code runs in a scratch cwd, so a stray open('out.txt','w') lands nowhere
    that outlives the probe."""
    monkeypatch.chdir(tmp_path)
    ok, detail = correctness.run_quicksort_code(
        "open('side-effect.txt', 'w').write('x')\n" + GOOD_QUICKSORT
    )
    assert ok, detail
    assert not (tmp_path / "side-effect.txt").exists()
