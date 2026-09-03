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

**The rule this reproduces is currently disconnected in the framework, and the
reason is this file's subject.** `handoff/store.py` calls `readme.check` and
`content.check_items` at both publication points and, at `:447` and `:494`,
deliberately does not call `locality.check`:

    # **`locality.check` is not called, and criterion 17 is therefore not
    # enforced.** User-ruled 2026-08-31 after it refused a correct artefact:
    # the shape heuristic read an HTTP access-log line as a filesystem path,
    # and the brief that produced the artefact *required* that line. Measured
    # 97% false positive on a real kit.

(`handoff/protocols.py:294` still promises *"Runs the README check and the
locality check before anything is created"* — that docstring is stale.
`ROADMAP.md` §6.4 carries the rebuild.)

So this script's job has changed. It cannot be *the* oracle for the seal, since
the seal does not ask. What it can be is an honest warning to an author, and a
97%-false-positive warning is not one. It therefore reports two classes
separately and exits non-zero only for the first.

**Class 1 — a genuine absolute path.** `/shared_nfs/yihou/...` in a log,
`/data/...` in a README. Real, portable-to-nobody, worth fixing.

**Class 2 — a composition artefact.** The candidate regex's lookbehind is
`(?<![A-Za-z0-9._~@+-])`, which excludes word characters and nothing else. So a
path run that *follows a closing delimiter* matches even though the text never
named an absolute path:

    <operator_id>/scripts/forge_driver.py      preceded by `>`
    "$PACKUP"/scripts/kernel/*.py              preceded by `"`
    ${AITER_ROOT}/ops/triton/gemm.py           preceded by `}`

All three are relative paths under something the reader substitutes. **This
system does not compose across a `}` or a quote before `/`** — the same failure
appears in the seal's `@NAME@` rule and in the variable grammar's `[^}]*`, and
it has now bitten in three separate places. The first two were measured on real
deliverables: one refused a complete workset, the other was found by
`kernel-opt` on a packup that had already sealed.

Class 2 is still *printed*, because a `${VAR}/a/b` that the consumer cannot
expand is a real problem of a different kind — just not this rule's.

**A prose path with two or more segments** is class 1: `items/code/x` is fine
because it does not start with `/`; `/items/code` is not.

The allow-list and both regexes are duplicated from `handoff/locality.py` rather
than imported, for the reason `demo/assets/lib/store.py` gives for duplicating
the store layout: the alternative is a task package importing a framework
component. Duplication is bounded to the constants below.
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

#: Characters that close a placeholder, a quotation or a substitution. A path
#: run immediately following one of these is the tail of something the reader
#: substitutes, not an absolute path — see the module docstring, class 2. The
#: regex's own lookbehind excludes word characters only, so it cannot tell.
CLOSERS = ">}\"')]"


def offenders(text: str) -> list[tuple[int, str, bool]]:
    """Every path the rule matches, as `(line, path, is_composition_artefact)`.

    The third field is what separates *a host path leaked into a deliverable*
    from *a relative path following a `}` or a quote*. Both match the regex;
    only the first is this rule's business.
    """
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = URL.sub(" ", line)
        if stripped.lstrip().startswith("#!"):
            continue  # a shebang names an interpreter, not a produced artefact
        for match in CANDIDATE.finditer(stripped):
            path = match.group(0)
            if any(
                path == prefix.rstrip("/") or path.startswith(prefix)
                for prefix in ALLOWED_PREFIXES
            ):
                continue
            before = stripped[match.start() - 1] if match.start() else ""
            out.append((lineno, path, before in CLOSERS))
    return out


def scan(root: Path) -> tuple[list[str], list[str]]:
    """`(genuine, composition_artefacts)`, each `path:line: hit`."""
    genuine: list[str] = []
    artefacts: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: the seal skips it too
        for lineno, hit, is_artefact in offenders(text):
            entry = f"{path.relative_to(root)}:{lineno}: {hit}"
            (artefacts if is_artefact else genuine).append(entry)
    return genuine, artefacts


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_locality.py <dir>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"check_locality: {root} is not a directory", file=sys.stderr)
        return 2

    genuine, artefacts = scan(root)

    # Printed whichever way the exit code goes: a `${VAR}/a/b` the consumer
    # cannot expand is a real problem, just not this rule's, and staying silent
    # about it because it is not refusable would hide it entirely.
    if artefacts:
        print(
            f"check_locality: {len(artefacts)} match(es) that follow a placeholder, "
            f"quote or brace. These are relative paths under something the reader "
            f"substitutes, not host paths — reported, not counted:",
            file=sys.stderr,
        )
        for line in dict.fromkeys(artefacts):
            print(f"  ~ {line}", file=sys.stderr)

    if not genuine:
        print(f"check_locality: clean — no absolute path outside the allow-list in {root.name}")
        return 0

    print(
        f"check_locality: {len(genuine)} absolute path(s) outside the allow-list.\n"
        f"Rewrite each as a relative path, or as one of the ${{...}} placeholders "
        f"operator_identity carries:",
        file=sys.stderr,
    )
    # Deduplicated but not truncated: a partial list invites fixing one and
    # paying for the next the same way.
    for line in dict.fromkeys(genuine):
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
