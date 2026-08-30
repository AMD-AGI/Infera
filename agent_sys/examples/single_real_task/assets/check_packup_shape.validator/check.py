#!/usr/bin/env python3
"""`check_packup_shape` — completeness, **strong**.

The handoff carries exactly one `<name>.packup_<YYYYMMDD>/` directory, that
directory holds every entry this package makes mandatory, and each mandatory
document carries substance rather than a heading and a blank line.

The readme beside this file argues which entries are mandatory and what
"substance" was settled on. This module is the exact form of those rules.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

#: Mandatory files, in the order a reader meets them. See the readme for why
#: `notes.md` is in here and `patches/` and `logs/` are not.
REQUIRED_FILES = ("README.md", "REPRODUCE.md", "environment.md", "notes.md")

#: Mandatory directories. Both must hold at least one regular file — an empty
#: `scripts/` is the "hollow scaffolding" the packup layout tells an author to
#: omit, and omitting it fails the presence check instead, which is the same
#: verdict by a clearer route.
REQUIRED_DIRS = ("scripts", "results")

#: Placeholder tokens. Anything matching is a document that was templated and
#: not written. `<...>` is the packup templates' own placeholder form —
#: `<exact commands>`, `<who>`, `<YYYY-MM-DD>` — and it is matched only when the
#: angle brackets wrap something that is not a URL or an e-mail address, so a
#: legitimate `<https://...>` or `<user@host>` does not trip it.
PLACEHOLDER = re.compile(
    r"""
      \b (?: TODO | TBD | FIXME | XXX ) \b
    | to \s+ be \s+ filled \s+ in
    | < (?! [A-Za-z][A-Za-z0-9+.-]* : // ) (?! [^<>@\s]+ @ ) [^<>\n]{2,} >
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: A markdown fence, opening or closing.
FENCE = re.compile(r"\A\s*(?:```|~~~)")

#: A heading line.
HEADING = re.compile(r"\A\s*#")


def content_lines(text: str) -> list[str]:
    """Lines that carry content: non-blank, not a heading, not a fence marker.

    Headings are excluded on purpose. A document that is four `##` lines and
    nothing under them is exactly the failure this floor exists to catch, and
    counting the headings would let it pass.
    """
    out = []
    for line in text.splitlines():
        if not line.strip() or FENCE.match(line) or HEADING.match(line):
            continue
        out.append(line)
    return out


def command_lines(text: str) -> list[str]:
    """Non-blank lines inside a code block — fenced, or indented by four.

    This is what "copy-pasteable commands" reduces to when it has to be counted.
    It does not try to decide whether a line is a *valid* command: that would be
    a shell parser, and a check that guesses at shell syntax fails honest kits.
    """
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not line.strip():
            continue
        if fenced or line.startswith("    ") or line.startswith("\t"):
            out.append(line.strip())
    return out


def check_document(path: Path, floor: int) -> list[str]:
    """One mandatory markdown file. Every failure it has, not just the first."""
    faults = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path.name}: unreadable: {exc}"]
    lines = content_lines(text)
    if len(lines) < floor:
        faults.append(f"{path.name}: {len(lines)} content lines, needs {floor}")
    found = PLACEHOLDER.search(text)
    if found:
        faults.append(f"{path.name}: unfilled placeholder {found.group(0)!r}")
    return faults


def check_packup(root: Path, floors: dict, min_commands: int) -> list[str]:
    """The whole packup directory. Returns every fault; empty means pass."""
    faults: list[str] = []

    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file():
            faults.append(f"{name}: missing")
            continue
        faults += check_document(path, int(floors.get(name, 1)))

    for name in REQUIRED_DIRS:
        path = root / name
        if not path.is_dir():
            faults.append(f"{name}/: missing")
        elif not any(p.is_file() for p in path.rglob("*")):
            faults.append(f"{name}/: empty")

    # `REPRODUCE.md` is the file a reproducer executes, so prose alone is not a
    # reproduction kit however much of it there is.
    reproduce = root / "REPRODUCE.md"
    if reproduce.is_file():
        commands = command_lines(reproduce.read_text(encoding="utf-8", errors="replace"))
        if len(commands) < min_commands:
            faults.append(
                f"REPRODUCE.md: {len(commands)} command lines in code blocks, needs {min_commands}"
            )

    # `README.md` answers "did it work". The packup template makes `## Result`
    # the section that says so, and this is the one heading required by name.
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^\s*#{1,6}\s*Result\b", text, re.IGNORECASE | re.MULTILINE):
            faults.append("README.md: no `## Result` heading")

    # `environment.md` is where versions are pinned. A file with no digit
    # anywhere in it has pinned nothing — crude, exact, and it has never yet
    # been true of a real environment note.
    environment = root / "environment.md"
    if environment.is_file():
        text = environment.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\d", text):
            faults.append("environment.md: no version, digest or date anywhere in it")

    return faults


def main() -> int:
    parameters = zone.args()
    floors = parameters.get("min_content_lines") or {}
    min_commands = int(parameters.get("min_command_lines", 1))

    results: dict[str, bool] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            results[hid] = False
            print(f"check_packup_shape: {hid}: no staged content")
            continue
        packup, why = zone.find_packup(content)
        if packup is None:
            results[hid] = False
            print(f"check_packup_shape: {hid}: {why}")
            continue
        faults = check_packup(packup, floors, min_commands)
        results[hid] = not faults
        print(f"check_packup_shape: {hid}: packup {why}")
        for fault in faults:
            print(f"check_packup_shape: {hid}: FAIL: {fault}")
    zone.write_verdict(results)
    print(f"check_packup_shape: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
