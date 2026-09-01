#!/usr/bin/env python3
"""`check_deploy_kit` — completeness, **strong**.

The handoff carries exactly one `<name>.packup_<YYYYMMDD>/` directory, that
directory holds every entry this package makes mandatory, each mandatory
document carries substance rather than a heading and a blank line, and the kit
meets the four **standardisation** rules this stage adds.

**This file is `single_real_task/check_packup_shape/check.py` with four rules
added**, and it is a copy rather than an import because a task package's assets
are data: nothing on a validator body's path makes another package importable,
and a shared module would make two packages that must be independently loadable
into one. The inherited rules' reasoning lives in that package's readme and is
not repeated; the added rules are argued in the readme beside this file, and
each of the four is a fault that was observed in a real kit.
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

# ------------------------------------------------------- the standardisation floor
#
# The four rules below are what this validator adds over
# `single_real_task`'s `check_packup_shape`, which it is otherwise a copy of.
# Each one is a fault that was **observed** in a real kit rather than a rule
# somebody liked the sound of, and each is switched by an `args` key so a
# package can state its floor in one place.

#: `## Expected output`, at any heading depth and in either case. It is the only
#: criterion `check_deploy_reproduces` hands its reproducer; a kit without one
#: is a kit whose reproduction cannot be judged.
EXPECTED_OUTPUT = re.compile(r"^\s*#{1,6}\s*Expected\s+output\b", re.IGNORECASE | re.MULTILINE)

#: The three facts `environment.md` must name for a kit to travel, as
#: (label, pattern). **Deliberately generous**: each is satisfied by any of
#: several spellings, because the rule is "this fact is stated", not "it is
#: stated the way I would have written it".
ENVIRONMENT_FACTS = (
    ("GPU architecture", re.compile(r"gfx\d{3,4}|MI\s?\d{3}X?|CDNA\d?", re.IGNORECASE)),
    ("image", re.compile(r"^.*\b(image|docker)\b.*$", re.IGNORECASE | re.MULTILINE)),
    ("model", re.compile(r"^.*\bmodel\b.*$", re.IGNORECASE | re.MULTILINE)),
)

#: A served model name that is a filesystem path. Measured: one kit registered
#: `Qwen/Qwen3.6-27B` and the next registered `/data/<user>/…/Qwen3.6-27B`,
#: purely from `--served-model-name` being absent — and the second one bakes a
#: machine's directory layout into the one field every caller must copy.
#:
#: Anchored on the JSON field so it fires on evidence rather than on prose: a
#: sentence in `notes.md` explaining this very trap must not fail the kit.
SERVED_PATH = re.compile(r'"(?:model|served_model_name|model_name)"\s*:\s*"(/[^"]*)"')

# ------------------------------------------------------- shared-namespace identifiers

#: Flags that bind something into a namespace shared with everyone else on the
#: host: a docker container name, a published port, a bind mount, a listening
#: port. The set is small and literal on purpose — the readme argues why this is
#: a list of flags rather than an attempt to recognise "a container name" from
#: its text.
BINDING_FLAGS = ("--name", "--publish", "--volume", "--mount", "--port", "-p", "-v")

#: The two short flags mean something else in other commands — `mkdir -p` above
#: all — so they only count inside a command that mentions `docker`.
DOCKER_ONLY_FLAGS = ("-p", "-v")

#: Flags where a bare literal is a fault on its own, with no variable involved.
#: `--volume` is **not** here: a read-only mount of an input path is legitimately
#: fixed, and only the *host* side of it is shared. See the readme.
LITERAL_FORBIDDEN_FLAGS = ("--name", "--publish", "--port", "-p")

#: `${X:=v}`, `${X:-v}`, `${X:?…}` — the three forms that let a caller supply a
#: different value. Finding a name here anywhere in the kit is what makes it a
#: parameter rather than a constant.
PARAMETERISED = re.compile(r"\$\{\s*(\w+)\s*:[=\-?]")

#: `X=…`, `export X=…`, `local X=…`, `declare -r X=…`.
PLAIN_ASSIGN = re.compile(r"\A\s*(?:export\s+|local\s+|declare\s+-\w+\s+)?(\w+)=(.*)\Z")

#: `${X}`, `${X:-…}`, `$X`.
VAR_REF = re.compile(r"\$\{(\w+)[^}]*\}|\$(\w+)")

#: A flag and the token it takes, `--name x`, `--name=x`, `-v x`. **Built from
#: `BINDING_FLAGS` rather than repeating it**: a second spelling of the set is
#: how the constant a reader checks and the pattern that runs come to disagree.
#: Longest first, so `--publish` is never matched as `-p` followed by text.
FLAG_USE = re.compile(
    r"(?<![\w-])(" + "|".join(sorted(BINDING_FLAGS, key=len, reverse=True)) + r")[=\s]+(\S+)"
)

#: Whether anything survives once every variable reference is removed. A value
#: built only out of other variables and punctuation — `"${HOST}:${PORT}"` — is
#: as parameterised as the variables it is made of.
ALNUM = re.compile(r"[A-Za-z0-9]")


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations, keeping the line number of the first line.

    `docker run -d --name "${CTR_NAME}" \\` spans a dozen lines, and a scanner
    that reads them one at a time sees a flag with no argument and an argument
    with no flag.
    """
    out: list[tuple[int, str]] = []
    buffered, start = "", 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not buffered:
            start = number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffered += stripped[:-1] + " "
            continue
        out.append((start, buffered + stripped))
        buffered = ""
    if buffered:
        out.append((start, buffered))
    return out


def check_shared_identifiers(root: Path) -> list[str]:
    """Identifiers the kit freezes and then binds into a host-wide namespace.

    Two facts are collected over every file in `scripts/`, and a fault needs
    both: a name is **frozen** if it is assigned with a plain assignment whose
    value is not built purely out of other variables, and it is **bound** if it
    reaches one of `BINDING_FLAGS`. Frozen-and-bound is the shape a second copy
    of the kit cannot run beside the first.

    A name is exempt the moment it appears in a `${X:=…}` form *anywhere* in the
    kit, because that is all it takes to let a caller re-point it.
    """
    scripts = root / "scripts"
    if not scripts.is_dir():
        return []  # its absence is already a fault, reported by the caller

    files = sorted(p for p in scripts.rglob("*") if p.is_file())
    sources: list[tuple[str, str]] = []
    for path in files:
        try:
            sources.append((path.name, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue  # a binary or unreadable file in scripts/ is not our business

    parameterised = {name for _, text in sources for name in PARAMETERISED.findall(text)}

    frozen: dict[str, str] = {}  # name -> "file:line"
    bound: dict[str, str] = {}  # name -> "file:line"
    faults: list[str] = []

    for filename, text in sources:
        for number, line in logical_lines(text):
            where = f"scripts/{filename}:{number}"

            assigned = PLAIN_ASSIGN.match(line)
            if assigned:
                name, value = assigned.group(1), assigned.group(2)
                remainder = VAR_REF.sub("", value).strip("\"' \t")
                if ALNUM.search(remainder) and name not in frozen:
                    frozen[name] = where

            for flag, token in FLAG_USE.findall(line):
                if flag in DOCKER_ONLY_FLAGS and "docker" not in line:
                    continue
                referenced = [a or b for a, b in VAR_REF.findall(token)]
                for name in referenced:
                    bound.setdefault(name, where)
                if not referenced and flag in LITERAL_FORBIDDEN_FLAGS:
                    bare = token.strip("\"'")
                    if ALNUM.search(bare):
                        faults.append(
                            f"{where}: `{flag} {bare}` binds a literal into a "
                            f"host-wide namespace; take it from an environment "
                            f"variable with a default"
                        )

    for name in sorted(bound):
        if name in parameterised or name not in frozen:
            continue
        faults.append(
            f"{frozen[name]}: {name} is fixed here and reaches a binding flag at "
            f'{bound[name]}; write it as `: "${{{name}:=…}}"` so a second copy of '
            f"this kit can run beside the first"
        )
    return faults


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


def check_expected_output(root: Path) -> list[str]:
    """`REPRODUCE.md` states what success looks like."""
    path = root / "REPRODUCE.md"
    if not path.is_file():
        return []  # its absence is already reported by the caller
    text = path.read_text(encoding="utf-8", errors="replace")
    if EXPECTED_OUTPUT.search(text):
        return []
    return [
        "REPRODUCE.md: no `Expected output` section; it is the only criterion "
        "the reproduction check has, and without it a correct reproduction can "
        "be judged a failure"
    ]


def check_json_results(root: Path, floor: int) -> list[str]:
    """`results/` carries machine-readable evidence, not only prose.

    The next stage of this flow consumes results, and it consumes files rather
    than paragraphs. Counted over `rglob` so a kit is free to organise
    `results/` into subdirectories.
    """
    results = root / "results"
    if not results.is_dir():
        return []  # its absence is already reported by the caller
    found = sorted(p.name for p in results.rglob("*.json") if p.is_file() and p.stat().st_size)
    if len(found) >= floor:
        return []
    return [
        f"results/: {len(found)} non-empty .json file(s), needs {floor} "
        f"(found: {found})"
    ]


def check_environment_facts(root: Path) -> list[str]:
    """`environment.md` names the three facts a reproduction fails without."""
    path = root / "environment.md"
    if not path.is_file():
        return []  # its absence is already reported by the caller
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        f"environment.md: does not name the {label}"
        for label, pattern in ENVIRONMENT_FACTS
        if not pattern.search(text)
    ]


def check_served_name(root: Path) -> list[str]:
    """No evidence file shows the model served under a filesystem path.

    Scanned over `results/` only. That is where the evidence is, and it keeps
    the rule off `notes.md`, where a kit is *encouraged* to write about this
    trap in prose.
    """
    results = root / "results"
    if not results.is_dir():
        return []
    faults = []
    for path in sorted(p for p in results.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found = SERVED_PATH.search(text)
        if found:
            faults.append(
                f"results/{path.relative_to(results)}: the model is served as "
                f"{found.group(1)!r} — a filesystem path, not a published model "
                f"name. Set the served name explicitly"
            )
    return faults


def check_packup(root: Path, floors: dict, min_commands: int, options: dict) -> list[str]:
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

    # An identifier the kit freezes and then binds into a namespace it shares
    # with every other tenant — and with the reproducer, which starts from this
    # same kit while the run that produced it may still be up.
    faults += check_shared_identifiers(root)

    # The standardisation floor. Each is switchable because a package that
    # standardises something else should be able to say so, not fork this file.
    if options.get("require_expected_output", True):
        faults += check_expected_output(root)
    faults += check_json_results(root, int(options.get("min_json_results", 0)))
    if options.get("environment_facts", True):
        faults += check_environment_facts(root)
    if options.get("require_served_name_not_a_path", True):
        faults += check_served_name(root)

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
            print(f"check_deploy_kit: {hid}: no staged content")
            continue
        packup, why = zone.find_packup(content)
        if packup is None:
            results[hid] = False
            print(f"check_deploy_kit: {hid}: {why}")
            continue
        faults = check_packup(packup, floors, min_commands, parameters)
        results[hid] = not faults
        print(f"check_deploy_kit: {hid}: packup {why}")
        for fault in faults:
            print(f"check_deploy_kit: {hid}: FAIL: {fault}")
    zone.write_verdict(results)
    print(f"check_deploy_kit: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
