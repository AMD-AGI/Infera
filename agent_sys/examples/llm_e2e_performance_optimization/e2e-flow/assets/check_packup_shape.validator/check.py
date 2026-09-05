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

import io
import contextlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import workset_io  # noqa: E402 — the shared report writer; see _report()
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

    # **Directories are counted by substance, not by presence**, for the same
    # reason the files above are counted by content line: *a presence check on a
    # document nobody filled in is theatre*, and that argument covers a directory
    # exactly as well as a README. The first version tested `any(path.iterdir())`
    # and m2 showed what it buys — `logs/` reduced from 17 files to one **empty
    # subdirectory** passed, and `scripts/` reduced from its command scripts to
    # one **zero-byte** `run.sh` passed. Both are packups a reproducer cannot
    # use, graded complete.
    #
    # `rglob` and not `iterdir` because the real kit nests: `logs/` is 17 files
    # under seven subdirectories, so a top-level count reads it as zero. m2 hit
    # that too, in their own instrument, and reported it against my kit before
    # catching it — which is the reason this counts the way `find -type f` does.
    #
    # `st_size > 2` is `min_result_files`' own test, reused rather than re-picked
    # so the two agree about what an empty file is.
    #
    # **The floor stays at 1 unless somebody measures a reason.** Real kit today:
    # results 18, logs 17, scripts 3. A floor of, say, 4 for `scripts/` would fit
    # today's kit and refuse a legitimate one that ships three — the bar has to
    # come from the contract, and the contract only says the directory must mean
    # something.
    dir_floors = args.get("min_dir_files") or {}
    for name in args.get("require_dirs") or []:
        path = root / name
        if not path.is_dir():
            ok = _fail(reasons, f"{name}/ is missing")
            continue
        found = [p for p in path.rglob("*") if p.is_file() and p.stat().st_size > 2]
        floor = int(dir_floors.get(name, 1))
        if len(found) < floor:
            ok = _fail(
                reasons,
                f"{name}/ holds {len(found)} non-empty file(s), floor is {floor} — "
                f"an empty subdirectory or a zero-byte placeholder is not the directory's contents",
            )

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
        # `rglob`, matching `require_dirs` above. It was `iterdir` and the two
        # measurements of the same directory disagreed about nesting: a
        # `results/` holding only `nested/a.json` refused with *"holds 0
        # non-empty file(s)"* when it holds one. The verdict was right and the
        # number was a lie, which is the worse half — a reader who goes to check
        # finds a file where the message says there is none and stops believing
        # the next message too. Found by m2 reading `ad6d431`.
        found = [p for p in results.rglob("*") if p.is_file() and p.stat().st_size > 2]
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
    results: dict = {}
    findings: dict = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            # Captured and re-echoed: the lines that explain a PASS go to stdout,
            # and a zone keeps no stdout at all. A person watching the run still
            # sees them; so, now, does anyone reading the zone afterwards.
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    results[hid] = check(content, args, reasons)
            except Exception as exc:  # noqa: BLE001
                # A crash is not a refusal. verdict.json cannot express the
                # difference (todo.md T29); this text is the only place it exists.
                results[hid] = False
                reasons.append(f"THIS VALIDATOR DID NOT RUN: {type(exc).__name__}: {exc}")
            sys.stdout.write(buffer.getvalue())
            notes = [ln.strip() for ln in buffer.getvalue().splitlines() if ln.strip()]
            findings[hid] = ([] if results[hid] else list(reasons),
                             notes + (list(reasons) if results[hid] else []))
        findings.setdefault(hid, (list(reasons), []))
        print(f"check_packup_shape: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before write_verdict, deliberately: a crash in the writer must not be able
    # to take the reasons with it, and the verdict is what the phase reads.
    _report(findings, results)
    zone.write_verdict(results)
    return 0


def _report(findings: dict, results: dict) -> None:
    """`workset_io.write_report`, and never a second implementation of it.

    m3 measured that 16 of 21 validators persist nothing, and seven of those are
    this stage's. That matters most here because **stage 5 has never been
    reached**: every other stage has had refusals to learn from, and m5's first
    one would otherwise arrive with the diagnostics switched off.

    `verdicts` is passed rather than letting the heading infer from `problems`
    being non-empty — these bodies keep informational lines in the same
    `reasons` list, which is the case that made the argument exist.

    Wrapped so that a failure to write the report cannot fail the validation:
    the report is evidence *about* a verdict and must never become the reason
    there is not one.
    """
    try:
        workset_io.write_report("check_packup_shape", findings, results)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print("check_packup_shape: could not write the validator report: %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
