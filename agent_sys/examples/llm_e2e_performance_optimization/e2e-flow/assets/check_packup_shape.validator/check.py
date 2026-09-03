#!/usr/bin/env python3
"""`check_packup_shape` — completeness, strong.

Adapted from `single_real_task/assets/check_packup_shape.validator`, against the
same layout reference.

**Substance is counted, not assumed.** A presence check on a document nobody
filled in is theatre: the layout reference ships templates, so a packup that has
every mandated file and nothing in them is exactly what a lazy producer emits.
Hugging Face's live card validator returns 200 for a card whose entire prose is
`[More Information Needed]`, a string appearing in 636,321 repositories.

So each mandated document is measured in CONTENT lines — non-blank, not a
heading, not a code-fence marker — and `REPRODUCE.md`, the one file a reproducer
actually executes, is measured in command lines instead.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_FENCE = re.compile(r"^\s*(```|~~~)")
#: A line that looks like something a reproducer types. Indented code blocks and
#: fenced ones both count; prose does not. Deliberately loose — the rule is "this
#: document contains commands", not "these commands are correct".
_COMMAND = re.compile(r"^\s*(?:[a-zA-Z_./][\w./-]*\s|export\s|cd\s|\$\s|\.\s)")

#: The placeholder our own templates and the layout reference emit.
_PLACEHOLDERS = ("[more information needed]", "<...>", "tbd", "todo")


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def content_lines(text: str) -> list[str]:
    out = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not line.strip() or (not fenced and _HEADING.match(line)):
            continue
        out.append(line)
    return out


def command_lines(text: str) -> list[str]:
    out = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        stripped = line.strip()
        if not stripped or _HEADING.match(line):
            continue
        # Inside a fence, or indented by four spaces, and shaped like a command.
        if (fenced or line.startswith("    ")) and _COMMAND.match(stripped):
            out.append(stripped)
    return out


def check(content: Path, args: dict, reasons: list) -> bool:
    root = content / "items" / "codes"
    if not root.is_dir():
        return _fail(reasons, "items/codes/ is missing — the packup has no directory")

    ok = True

    for name in args.get("require_dirs") or []:
        path = root / name
        if not path.is_dir():
            ok = _fail(reasons, f"{name}/ is missing")
        elif not any(path.iterdir()):
            ok = _fail(reasons, f"{name}/ is empty")

    floors = args.get("min_content_lines") or {}
    for name in args.get("require_files") or []:
        path = root / name
        if not path.is_file():
            ok = _fail(reasons, f"{name} is missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = content_lines(text)
        floor = int(floors.get(name, 1))
        if len(lines) < floor:
            ok = _fail(reasons, f"{name} has {len(lines)} content line(s), floor is {floor}")
        lowered = " ".join(lines).casefold()
        for marker in _PLACEHOLDERS:
            if marker in lowered:
                ok = _fail(reasons, f"{name} still carries the placeholder {marker!r}")
                break

    reproduce = root / "REPRODUCE.md"
    if reproduce.is_file():
        commands = command_lines(reproduce.read_text(encoding="utf-8", errors="replace"))
        floor = int(args.get("min_command_lines", 5))
        if len(commands) < floor:
            ok = _fail(
                reasons,
                f"REPRODUCE.md carries {len(commands)} command line(s), floor is {floor} — "
                f"it is the file a reproducer executes, so prose is not enough",
            )

    results = root / "results"
    if results.is_dir():
        found = [p for p in results.iterdir() if p.is_file() and p.stat().st_size > 2]
        floor = int(args.get("min_result_files", 4))
        if len(found) < floor:
            ok = _fail(
                reasons,
                f"results/ holds {len(found)} non-empty file(s), floor is {floor} — "
                f"without evidence the packup is a narrative",
            )
    return ok


def main() -> int:
    args = zone.args()
    results = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_packup_shape: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
