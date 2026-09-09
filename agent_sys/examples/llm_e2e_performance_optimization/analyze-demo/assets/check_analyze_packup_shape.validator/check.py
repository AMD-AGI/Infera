#!/usr/bin/env python3
"""`check_analyze_packup_shape` — completeness, strong.

The deliverable carries every mandated file, and each has substance.

Named for the stage rather than `check_packup_shape`, because `profiling-demo`
has a validator of that name over its own packup and the spec registry keys on
`name` alone. The two are different checks over different kinds — that one reads
`profile_packup` and counts content lines and commands, this one reads
`analyze_packup` and enforces a byte floor and a placeholder-phrase list — so
they want two names.

Presence alone is not enough: a `REPRODUCE.md` reading "TBD" satisfies every
existence check and helps nobody. So each mandated file clears a byte floor and
is scanned for placeholder phrases. The floor is deliberately low — a terse but
real section should pass — and the phrase list catches the stubs long enough to
clear it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402

EXECUTABLE_ITEMS = frozenset({"script", "command", "entry"})

REPRODUCIBLE_ITEMS = ("result", "env", "command", "code")


def check(content: Path, args: dict) -> tuple[bool, str]:
    for item in REPRODUCIBLE_ITEMS:
        target = content / "items" / item
        if not target.exists():
            return False, f"items/{item} is absent"
        # See `check_workset_shape`: `agent/gate.py` refuses the seal for a
        # `command` item that is not executable, and says so only after the
        # producing body has already returned.
        if item in EXECUTABLE_ITEMS and not os.access(target, os.X_OK):
            return False, (
                f"items/{item} is not executable; run `chmod +x items/{item}`"
            )

    minimum = int(args.get("min_bytes") or 400)
    forbidden = [p.lower() for p in (args.get("forbidden_phrases") or [])]

    # The README is the handoff's own, beside `items/`; the rest live inside it.
    mandated = [content / "README.md"] + [
        content / "items" / name for name in (args.get("required_files") or []) if name != "README.md"
    ]
    for path in mandated:
        if not path.is_file():
            return False, f"{path.name} is absent"
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text.encode("utf-8")) < minimum:
            return False, f"{path.name} is {len(text)} bytes, below the {minimum}-byte floor"
        for phrase in forbidden:
            if phrase in text.lower():
                return False, f"{path.name} still contains the placeholder {phrase!r}"

    for name in args.get("required_dirs") or []:
        directory = content / "items" / name
        if not directory.is_dir():
            return False, f"items/{name}/ is absent"
        if not any(directory.iterdir()):
            return False, f"items/{name}/ is empty"

    return True, f"{len(mandated)} mandated file(s) present with substance"


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_analyze_packup_shape: {hid}: no published content")
            continue
        ok, why = check(content, args)
        results[hid] = ok
        print(f"check_analyze_packup_shape: {hid}: {'PASS' if ok else 'FAIL'} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
