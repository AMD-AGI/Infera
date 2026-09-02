###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Semantic correctness probes: their prompts and classifiers.

Three probes, in ascending order of what they can catch:

1. counting (``/v1/completions``) — the engine emits coherent tokens at all.
   Ported from the atom_tasks stress-probe (``task3_results/probe_check.py``):
   seed "1,2,3,4,5," and require an ascending consecutive run of >= 5 integers.
2. capital (``/v1/chat/completions``) — one memorised fact survives the chat
   template.
3. long context (``/v1/completions``) — a 4-digit code buried ~55% into a ~9k-char
   ledger has to be retrieved. This is the only probe that fills more than a
   handful of KV blocks, so it is also the only one that makes the PD tier's
   prefill→decode transfer move a KV cache worth the name.

Probes 1 and 2 pass on any model that still produces fluent text; 3 does not.
That gap catches fluent-but-wrong output from broken kernels without executing
model-generated code on the test orchestrator.
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


# --- probe 3: long-context retrieval -----------------------------------------

# The needle. Four digits, and no filler value can collide with it: line numbers are
# 3-digit and crate counts stay under 200.
LONGCTX_ANSWER = "4731"

# ~9k chars ≈ 2.5k tokens: long enough to span many KV blocks, short enough for the
# tightest --max-model-len in the matrix (9472, with room for the answer).
LONGCTX_LINES = 130

# Mid-document, where neither primacy nor recency helps the model cheat.
LONGCTX_NEEDLE_FRAC = 0.55
LONGCTX_MAX_TOKENS = 32

_DEPOTS = ("ALPHA", "BRAVO", "CIVET", "DELTA", "ECHO", "FOXTROT", "GOLF")
_GOODS = (
    "grade-B cable",
    "sealed bearings",
    "ceramic fuses",
    "copper lugs",
    "anodised brackets",
    "silicone gaskets",
    "steel shims",
)


def build_longctx_prompt() -> str:
    """A deterministic depot ledger with the access code on one line ~55% in, then the
    question. Uniform lines, so finding it is retrieval rather than spotting an oddity."""
    needle_line = int(LONGCTX_LINES * LONGCTX_NEEDLE_FRAC)
    lines = []
    for i in range(1, LONGCTX_LINES + 1):
        if i == needle_line:
            lines.append(
                f"Ledger {i:03d}: the archive access code for this quarter is "
                f"{LONGCTX_ANSWER}; keep it on file."
            )
            continue
        lines.append(
            f"Ledger {i:03d}: depot {_DEPOTS[i % len(_DEPOTS)]} shipped "
            f"{40 + (i * 7) % 160} crates of {_GOODS[(i * 3) % len(_GOODS)]} "
            f"on day {1 + i % 28}."
        )
    return "\n".join(lines) + (
        "\n\nQuestion: according to the ledger above, what is the archive access code "
        "for this quarter?\nAnswer with the digits only.\nAnswer:"
    )


def is_longctx_correct(text: str) -> bool:
    """Correct = NOT garbage AND the reply carries the buried access code."""
    if not text:
        return False
    t = text.strip()
    return bool(t) and not looks_garbage(t) and LONGCTX_ANSWER in t
