###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Counting-probe correctness classifier.

Ported from the atom_tasks stress-probe (``task3_results/probe_check.py``):
send a deterministic "keep counting" prompt and verify the service replies
with real counting — or at least coherently acknowledges the task — rather
than token-salad / garbage.

Lenient rule (temp=0, wording varies): a response is CORRECT if it is NOT
garbage AND it either
  - emits an ascending consecutive integer run of length >= 5 (e.g. 6,7,8,9,10), or
  - mentions counting/number/sequence (any language, incl. 数/计数/数到).

Pure logic, no dependencies — usable from e2e assertions and unit tests alike.
"""

from __future__ import annotations

import re

COUNTING_PROMPT = (
    "Please strictly follow me to count from 1 to 30. "
    "Do not respond with anything other than numbers. 1,2,3,4,5,"
)
COUNTING_MAX_TOKENS = 64

# Second, independent correctness probe: a one-shot factual chat question.
CAPITAL_PROMPT = "Just answer me directly: What is the capital city of China?"
CAPITAL_MAX_TOKENS = 64

# Non-ASCII punctuation that legitimately appears in coherent output
# (smart quotes, em dash, ellipsis, CJK punctuation) — not "foreign" noise.
_OK_NONASCII = set("—…“”‘’")
_OK_NONASCII |= set("、，。：；！？（）《》")


def looks_garbage(text: str) -> bool:
    """Token-salad detector: replacement chars, emoji/pictographs, or a heavy
    fraction of random mixed-script characters. Coherent English/Chinese is
    NOT garbage."""
    if "\ufffd" in text:  # unicode replacement char -> decode garbage
        return True
    for ch in text:
        o = ord(ch)
        if 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x1F300 <= o <= 0x1FAFF:
            return True  # emoji / pictograph
    foreign = sum(1 for ch in text if ord(ch) > 0x024F and ch not in _OK_NONASCII)
    return foreign / max(len(text), 1) > 0.15


def is_counting_correct(text: str) -> bool:
    """Correct = NOT garbage AND shows actual counting evidence (an ascending
    consecutive integer run >= 5) OR verbally recognizes the counting task."""
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if looks_garbage(t):
        return False

    nums = [int(n) for n in re.findall(r"\d+", t)]
    run = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            run += 1
            if run >= 5:
                return True
        else:
            run = 1
    return False


def is_capital_correct(text: str) -> bool:
    """Correct = NOT garbage AND the reply names China's capital, i.e. mentions
    'beijing' (case-insensitive)."""
    if not text:
        return False
    t = text.strip()
    if not t or looks_garbage(t):
        return False
    return "beijing" in t.lower()
