###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Semantic correctness probes: their prompts, their classifiers, and the sandbox.

Four probes, in ascending order of what they can catch:

1. counting (``/v1/completions``) — the engine emits coherent tokens at all.
   Ported from the atom_tasks stress-probe (``task3_results/probe_check.py``):
   seed "1,2,3,4,5," and require an ascending consecutive run of >= 5 integers.
2. capital (``/v1/chat/completions``) — one memorised fact survives the chat
   template.
3. long context (``/v1/completions``) — a 4-digit code buried ~55% into a ~9k-char
   ledger has to be retrieved. This is the only probe that fills more than a
   handful of KV blocks, so it is also the only one that makes the PD tier's
   prefill→decode transfer move a KV cache worth the name.
4. quicksort (``/v1/chat/completions``) — generated Python is *executed* against
   ``sorted()`` on a permutation, an empty list, a singleton, duplicates,
   negatives, and sorted/reversed input.

Probes 1 and 2 pass on any model that still produces fluent text; 3 and 4 do not.
That gap is the point: fluent-but-wrong output is exactly how a new architecture's
kernels fail (MXFP4 emulated on gfx942, a mismatched attention backend), and it is
invisible to a "did it say Beijing" check.

Stdlib only, and side-effect free apart from the quicksort sandbox — which runs a
language model's code in an isolated ``python -I`` subprocess, under a timeout, in
a scratch cwd. That is sound only because the e2e containers are disposable.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile

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


# --- probe 4: generated code, executed ---------------------------------------

QUICKSORT_PROMPT = (
    "Write a Python 3 function `quicksort(nums)` that returns a new list holding the "
    "integers of `nums` in ascending order. Implement the quicksort algorithm yourself: "
    "do not call sorted(), list.sort() or any library sort. It must handle duplicates, "
    "negative numbers and the empty list. Reply with exactly one ```python code block "
    "and no other text."
)
QUICKSORT_MAX_TOKENS = 512
QUICKSORT_TIMEOUT = 15.0

# A permutation, then the six inputs a hand-rolled partition actually breaks on:
# empty, singleton, duplicates, negatives, already-sorted and reversed.
QUICKSORT_SEQUENCES = (
    (3, 6, 1, 8, 2, 9, 4, 7, 5),
    (),
    (42,),
    (5, 1, 5, 1, 5),
    (-9, 4, 0, -3, 12, -3),
    (1, 2, 3, 4, 5),
    (9, 8, 7, 6, 5),
)

# Leaning on one of these answers the prompt without implementing anything, so the probe
# would pass having tested nothing. Matched on the AST, not on substrings: `quicksort =
# sorted` has to count, while a docstring mentioning list.sort() must not.
_BANNED_NAMES = frozenset(
    {"sorted", "sort", "argsort", "msort", "insort", "insort_left", "heapify", "heappop"}
)
_BANNED_MODULES = frozenset({"heapq", "bisect", "numpy", "pandas", "torch"})

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Appended to the model's code and run as one script; __CASES__ becomes a JSON literal.
_DRIVER = """
import json as _json

_fn = next(
    (f for f in (globals().get(n) for n in ("quicksort", "quick_sort", "qsort")) if callable(f)),
    None,
)
if _fn is None:
    raise SystemExit("the reply defines no quicksort()/quick_sort()/qsort() function")
for _case in _json.loads(__CASES__):
    _got = _fn(list(_case))
    if _got is None:
        raise SystemExit("quicksort(%r) returned None; it must return the sorted list" % (_case,))
    if list(_got) != sorted(_case):
        raise SystemExit("quicksort(%r) returned %r" % (_case, _got))
print("QUICKSORT_OK")
"""


def extract_python_code(text: str) -> str:
    """The first fenced code block, else the whole reply — a model that ignores the
    fencing instruction still deserves to have its code run."""
    match = _FENCE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def _delegated_sort(code: str) -> str:
    """The library sort this code leans on, or ``""``. A locally defined name does not
    count, so a model that writes its own ``def sort`` is judged by running it instead."""
    tree = ast.parse(code)
    own = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_NAMES:
            return node.attr
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES and node.id not in own:
            return node.id
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [alias.name for alias in node.names] + [getattr(node, "module", "") or ""]
            hit = next(
                (n.split(".")[0] for n in imported if n.split(".")[0] in _BANNED_MODULES), ""
            )
            if hit:
                return hit
    return ""


def run_quicksort_code(code: str, *, timeout: float = QUICKSORT_TIMEOUT) -> tuple[bool, str]:
    """Run ``code`` against every :data:`QUICKSORT_SEQUENCES` entry and compare with
    ``sorted()``, returning (ok, one-line detail). On the sandbox, see the module docstring."""
    if not code.strip():
        return False, "the reply contained no code"
    try:
        delegated = _delegated_sort(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    if delegated:
        return False, f"the reply uses {delegated!r} instead of implementing quicksort"

    cases = json.dumps([list(seq) for seq in QUICKSORT_SEQUENCES])
    script = code + "\n" + _DRIVER.replace("__CASES__", repr(cases))
    try:
        with tempfile.TemporaryDirectory() as scratch:
            done = subprocess.run(
                [sys.executable, "-I", "-"],
                input=script,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=scratch,
            )
    except subprocess.TimeoutExpired:
        return False, f"the code did not finish within {timeout:.0f}s"
    if "QUICKSORT_OK" in done.stdout:
        return True, f"sorted all {len(QUICKSORT_SEQUENCES)} sequences correctly"
    reported = (done.stderr.strip() or done.stdout.strip()).splitlines()
    return False, reported[-1] if reported else f"exited {done.returncode} with no output"
