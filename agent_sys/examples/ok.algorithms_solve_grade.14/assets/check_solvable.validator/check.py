#!/usr/bin/env python3
"""`check_solvable` — trustworthiness, **weak**.

Every problem ships at least `min_examples` worked examples: `input` and
`output` both non-empty, the output not a copy of the input, no placeholder
text, and — where `output_format` says the answer is a single line — an output
that is a single line.

**This cannot show that a problem is solvable and does not claim to.** It has no
solver, no reference implementation and nothing that runs; it checks that a
worked answer was supplied at all and that it is shaped like an answer. A
problem with two beautifully formatted wrong outputs passes. `weak` is the most
that can honestly be claimed, and `weak` qualifies a PASS and never a failure
(`validator` spec §5.4) — when this returns False the examples really are
missing or malformed.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

#: Text that means "I did not write an answer here". Listed rather than pattern
#: matched, so the readme can quote the list and a reader can check the quote.
PLACEHOLDERS = ("...", "…", "TODO", "TBD", "<answer", "<output", "N/A")

#: Phrases in an `output_format` that promise a one-line answer. Crude on
#: purpose: a format that does not use any of these is not checked for shape at
#: all, which is a gap this states rather than papers over.
ONE_LINE_PHRASES = (
    "one line",
    "a single line",
    "single line",
    "one integer",
    "a single integer",
    "one number",
)

#: Phrases that contain a member of `ONE_LINE_PHRASES` and mean the **opposite**.
#:
#: Measured, on the first full run of this package. A `merge-intervals` problem
#: described its output as *"One line per merged interval, in increasing order
#: of left endpoint"*. That contains `"one line"`, so the shape check demanded
#: exactly one line and rejected two of the three worked examples as unsolved —
#: `check_solvable` FAILED a perfectly good artefact and the graph stalled with
#: seven tasks in `waiting_handoff`.
#:
#: **A substring match caught the phrase that negates it.** `"one line per X"`
#: promises one line *each*, which is a promise of many. Listed rather than
#: solved with a parser: the check is `weak` and a phrase list is honest about
#: being a phrase list, where a regex would look like grammar and still be one.
#: A format that says "one line per" is not shape-checked at all, which is the
#: same gap the note above already declares, one case wider.
NOT_ONE_LINE_PHRASES = (
    "one line per",
    "a line per",
    "one line for each",
    "a line for each",
    "one per line",
    "one integer per",
)

#: The plural word, and it catches what the phrase list above cannot.
#:
#: **Measured, and it stalled a run in exactly the way the note above records.**
#: A `cells-draining-to-both-oceans` problem described its output as *"Line 1: a
#: single integer k, the number of qualifying cells. Then k lines, each holding
#: a cell's row index…"*. That contains `"a single integer"`, so the shape check
#: demanded exactly one line and rejected all three worked examples — the
#: artefact was correct and `check_solvable` FAILED it, and the graph stalled
#: with seven tasks waiting.
#:
#: **The negating construction was not a `per`.** It is *"Line 1: … Then k
#: lines"*, which no amount of adding to the phrase list above would have
#: anticipated — the list enumerates ways of saying "one each", and this says
#: "one, and then more". Enumerating negations is the losing side of that game.
#:
#: So the test inverts: a format that mentions **lines**, plural, is describing
#: more than one line whatever else it says, and is not shape-checked at all.
#: That is the same gap the module already declares — *"a format that does not
#: use any of these is not checked for shape"* — reached from the other side,
#: and it fails in the safe direction: it **skips** a check rather than
#: rejecting a good artefact.
_PLURAL_LINES = re.compile(r"\blines\b")


def has_placeholder(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in PLACEHOLDERS)


def worked(example: object, output_format: str) -> tuple[bool, str]:
    """Is this one example a worked answer? Returns the verdict and the reason."""
    if not isinstance(example, dict):
        return False, "not an object"
    given, produced = example.get("input"), example.get("output")
    if not isinstance(given, str) or not given.strip():
        return False, "empty `input`"
    if not isinstance(produced, str) or not produced.strip():
        return False, "empty `output`"
    if has_placeholder(produced):
        return False, "`output` contains placeholder text"
    if produced.strip() == given.strip():
        # An answer identical to its question is the shape a filled-in template
        # takes when nobody solved anything. It is legitimate for a handful of
        # real problems — echo the input, sorted-already — and those will have
        # to phrase one example differently. A weak check trading a rare false
        # negative for catching the common template is the trade this makes,
        # and it makes it out loud.
        return False, "`output` is a verbatim copy of `input`"
    lowered = output_format.lower()
    promises_many = any(phrase in lowered for phrase in NOT_ONE_LINE_PHRASES) or bool(
        _PLURAL_LINES.search(lowered)
    )
    if not promises_many and any(phrase in lowered for phrase in ONE_LINE_PHRASES):
        lines = [line for line in produced.splitlines() if line.strip()]
        if len(lines) != 1:
            return False, f"`output_format` promises one line, `output` has {len(lines)}"
    return True, ""


def check(content: Path, minimum: int) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        print("check_solvable: no items/text.json")
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"check_solvable: items/text.json is not JSON: {exc}")
        return False
    rows = data.get("problems") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        print("check_solvable: no non-empty `problems` list")
        return False

    ok = True
    for index, row in enumerate(rows):
        where = f"problems[{index}]"
        if not isinstance(row, dict):
            print(f"check_solvable: {where} is not an object")
            ok = False
            continue
        examples = row.get("examples")
        if not isinstance(examples, list):
            print(f"check_solvable: {where} has no `examples` list")
            ok = False
            continue
        output_format = row.get("output_format")
        output_format = output_format if isinstance(output_format, str) else ""
        good = 0
        for position, example in enumerate(examples):
            verdict, reason = worked(example, output_format)
            if verdict:
                good += 1
            else:
                print(f"check_solvable: {where}.examples[{position}]: {reason}")
        if good < minimum:
            print(f"check_solvable: {where} has {good} worked examples, needs {minimum}")
            ok = False
    return ok


def main() -> int:
    minimum = int(store.args().get("min_examples", 2))
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        # No published content is not a pass — a check that could not run has
        # not found nothing. The weak label qualifies what a PASS is worth; it
        # does not soften a failure.
        results[hid] = False if content is None else check(content, minimum)
    store.write_verdict(results)
    print(f"check_solvable: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
