#!/usr/bin/env python3
"""The seal's locality rule, runnable before the seal runs it.

`handoff/locality.py` scans every text file under a handoff's `content/` and
refuses to publish when it finds an absolute path outside a small allow-list.
The refusal arrives as `output was never delivered` **after** the body has
returned, so a body that only finds out then has already done all its work and
lost it.

This runs the same two regexes and the same allow-list against a directory, and
prints every offender with its file and line. Exit 0 means the seal will accept
it on this rule.

    python check_locality.py <dir>

Two false positives are worth knowing about, because they are what a careful
author actually trips over rather than a genuine host path.

**A placeholder followed by a separator.** The candidate regex's lookbehind is
`(?<![A-Za-z0-9._~@+-])`, which does not include `>` — so in

    <operator_id>/scripts/forge_driver.py

the run `/scripts/forge_driver.py` is two segments preceded by `>`, and it
matches. Measured: this is what refused an otherwise complete workset. Write
`scripts/forge_driver.py` on its own, or separate the placeholder from the path.

**A prose path with two or more segments.** `items/code/x` is fine because it
does not start with `/`; `/items/code` is not.

The allow-list and both regexes are duplicated from `handoff/locality.py` rather
than imported, for the reason `demo/assets/lib/store.py` gives for duplicating
the store layout: the alternative is a task package importing a framework
component. Duplication is bounded to the three constants below.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_PREFIXES = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/etc/", "/opt/",
    "/proc/", "/sys/", "/dev/", "/var/lib/", "/var/log/", "/run/", "/srv/",
    "/workspace/", "/app/",
)

CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9._~@+-])(?:[A-Za-z]:\\[^\s\"'<>|]*|(?:/[A-Za-z0-9._+@-]+){2,}/?)"
)
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S*")

#: `handoff/locality.py` skips anything larger, so rewriting it would be work
#: for no effect.
MAX_BYTES = 4 << 20


def offenders(text: str) -> list[tuple[int, str]]:
    """Every absolute path the seal would reject, as `(line, path)`."""
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = URL.sub(" ", line)
        if stripped.lstrip().startswith("#!"):
            continue  # a shebang names an interpreter, not a produced artefact
        for match in CANDIDATE.finditer(stripped):
            path = match.group(0)
            if not any(
                path == prefix.rstrip("/") or path.startswith(prefix)
                for prefix in ALLOWED_PREFIXES
            ):
                out.append((lineno, path))
    return out


def scan(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: the seal skips it too
        for lineno, hit in offenders(text):
            found.append(f"{path.relative_to(root)}:{lineno}: {hit}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_locality.py <dir>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"check_locality: {root} is not a directory", file=sys.stderr)
        return 2

    found = scan(root)
    if not found:
        print(f"check_locality: clean — the seal will accept {root.name} on this rule")
        return 0

    print(
        f"check_locality: {len(found)} path(s) the seal will refuse.\n"
        f"Rewrite each as a relative path, or as one of the ${{...}} placeholders "
        f"operator_identity carries:",
        file=sys.stderr,
    )
    # Deduplicated but not truncated: a partial list invites fixing one and
    # paying for the next the same way.
    for line in dict.fromkeys(found):
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
