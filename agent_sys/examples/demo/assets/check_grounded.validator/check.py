#!/usr/bin/env python3
"""`check_grounded` — trustworthiness, strong.

Every numeral in the summary also appears in the facts it summarises.

The extraction is **deliberately crude**: `\\d+` over the raw text of both
sides, and set inclusion. A crude check that is honestly described is a strong
validator; a sophisticated one that is silently approximate is not.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

NUMERAL = re.compile(r"\d+")


def numerals(text: str) -> set:
    return set(NUMERAL.findall(text))


def read_text_tree(content: Path) -> str:
    """Every file under `content/`, concatenated. The grounding set is the whole
    artefact, not one nominated item: a number the summary quotes is grounded if
    the facts carry it anywhere."""
    parts = []
    for path in sorted(content.rglob("*")):
        if path.is_file() and not path.is_symlink():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def main() -> int:
    # **This body reads only what it was handed, and that is the whole change.**
    # It used to reach outside itself for the `facts`, because *grounded in its
    # input* compares two handoffs and an output phase stages only the one it
    # validates. Two routes out were tried and both are shut — a store scan
    # (`EACCES` from a confined body) and adding `facts` to this validator's own
    # inputs (overtaken: not the system's job). The ruling put the second
    # artefact **inside the first**, so there is nothing to reach for.
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            # No summary published. Not a pass: a check that could not run has
            # not found nothing.
            results[hid] = False
            continue

        # **The grounding set is inside the artefact**, because the `summary`
        # kind declares `items/grounding/` required — the task declaration
        # passing its own input through to its own output, per the user ruling.
        # So both halves of the comparison are in the one staged handoff and
        # this body reaches outside itself for nothing: no store root, no
        # `AGENT_SYS_INPUT_FACTS`, no scan. It works confined or unconfined.
        inside = content / "items" / "grounding"
        if inside.is_dir():
            grounding = numerals(read_text_tree(inside))
        else:
            # **Unreachable for a published summary**, and kept loud rather than
            # deleted. `grounding` is a required item, so `HandoffStore.put`
            # refuses a summary without it — this arm can only be reached by a
            # summary that never went through publication. Falling back
            # *silently* would be the worse bug of the two it guards: it would
            # hide a missing required item **and** restore the external route
            # the ruling closed.
            outside = store.declared_dir("facts") or store.latest_of_kind("facts")
            print(
                f"check_grounded: {hid}: NO items/grounding/ in the artefact — "
                f"a required item is missing and this fell back to "
                f"{'the producer input' if outside else 'nothing'}"
            )
            if outside is None:
                results[hid] = False
                continue
            grounding = numerals(read_text_tree(outside))

        claimed = numerals((content / "items" / "content").read_text(encoding="utf-8"))
        ungrounded = sorted(claimed - grounding)
        results[hid] = not ungrounded
        if ungrounded:
            print(f"check_grounded: {hid}: ungrounded numerals {ungrounded}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
