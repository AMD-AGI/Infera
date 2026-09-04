#!/usr/bin/env python3
"""`check_deploy_kit` — completeness, **strong**. Program, no model call.

An **interpreter for `assets/schemas/deploy_kit.layout.yaml`**, and that is the
whole design. Mission M1.1 asks for the kit's file and directory requirements to
be written into their own yaml and for a program validator to check against it;
so every path, floor, pattern and message this file applies is read from that
yaml at run time and none is written here. A kit author and the check that
grades the kit read one document.

What this file therefore *is*: the six primitives that yaml is expressed in —
presence, substance, placeholders, evidence predicates over `results/`, the
frozen-and-bound identifier rule over `scripts/`, and JSON-Schema validation of
`codes/environment.yaml`.

**The one replacement mission M1.1.1 asks for.** The previous stage checked the
environment with three regexes over `environment.md`
(`../deploy-demo/assets/check_deploy_kit.validator/check.py:71-80` — "does the
word `image` appear on some line"). That is gone. The record is now
`codes/environment.yaml`, validated against `environment.schema.json` through
`assets/lib/schema.py`, and `environment.md` is checked as a *rendering* of it:
the values the layout marks load-bearing must appear in it verbatim, so the two
cannot drift into disagreeing.

Everything else is carried across in substance from that file, because each rule
there was a fault that was **observed in a real kit** rather than a rule somebody
liked: `require_served_name_not_a_path`, `require_mode_readback`,
`require_completion_evidence`, `min_json_results`, `require_expected_output`, and
`check_shared_identifiers`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Re-exec under an interpreter that can validate, before anything else runs.
#
# **This exists because of a recorded framework bug that would have made this
# validator misdiagnose every correct kit.** An output-phase validator body takes
# §8.2's PRODUCER row, which **shadows** the GLOBAL row rather than merging with
# it — and the GLOBAL row is the only one carrying `AGENT_SYS_DEMO_PYTHON`
# (`kernel-opt-demo/bugs/002-validator-env-row-shadows-demo-python.md`). So the
# entry script's `"${AGENT_SYS_DEMO_PYTHON:-python3}"` resolves to
# `/usr/bin/python3` on exactly the phase this validator runs in.
#
# Measured on this host: `/usr/bin/python3` has `yaml` and `jsonschema` but
# **not `referencing`**, which `../lib/schema.py:109` imports to build its
# `$ref` registry. The failure would have arrived inside `validate()` as a
# `ModuleNotFoundError`, been caught by the broad handler around document
# loading, and been reported as *"environment.yaml: not loadable as a
# document"* — a wrong diagnosis of a correct kit, which is the exact failure
# class this package exists to prevent.
#
# So: probe for an interpreter that can import both, and re-exec into it. If
# none can, the run must **fail naming the dependency** rather than grading a
# kit with a broken checker.
# `referencing` is **preferred, not required**: it builds the cross-schema `$ref`
# registry, and `environment.schema.json` uses no `$ref` today. So the rule is
# "re-exec into an interpreter that has all three if one exists, and fall back to
# registry-free validation rather than refuse" — see `_validate` below. Refusing
# outright was the first version and it is wrong: it would block this validator
# on every output phase on this host, which is a worse failure than losing `$ref`
# support nothing currently uses.
_REQUIRED = ("jsonschema", "yaml")
_PREFERRED = ("jsonschema", "yaml")
_GUARD = "E2E_CHECK_DEPLOY_KIT_REEXEC"


def _imports(python: str, modules) -> bool:
    try:
        return subprocess.run(
            [python, "-c", "import " + ", ".join(modules)],
            capture_output=True, timeout=30,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _have(modules) -> bool:
    try:
        for module in modules:
            __import__(module)
        return True
    except ImportError:
        return False


def _reexec_if_needed() -> None:
    if os.environ.get(_GUARD) or _have(_PREFERRED):
        return  # already re-executed once, or nothing to gain

    seen: list[str] = []
    for candidate in (
        os.environ.get("AGENT_SYS_DEMO_PYTHON"),
        sys.executable,
        shutil.which("python3"),
        "/usr/bin/python3",
    ):
        if not candidate or candidate in seen:
            continue
        seen.append(candidate)
        if _imports(candidate, _PREFERRED):
            os.environ[_GUARD] = "1"
            os.execv(candidate, [candidate, str(Path(__file__).resolve()), *sys.argv[1:]])

    if not _have(_REQUIRED):
        print(
            f"check_deploy_kit: no interpreter here can import {', '.join(_REQUIRED)} "
            f"(tried: {seen}). This validator checks `codes/environment.yaml` against a "
            f"JSON Schema and cannot do that without them. Refusing rather than grading "
            f"a kit with a broken checker.",
            file=sys.stderr,
        )
        sys.exit(2)


_reexec_if_needed()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402 — the path insert above is what makes it importable
import zone  # noqa: E402

#: A markdown fence, opening or closing.
FENCE = re.compile(r"\A\s*(?:```|~~~)")

#: A heading line.
HEADING = re.compile(r"\A\s*#")

#: `${X:=v}`, `${X:-v}`, `${X:?…}` — the three forms that let a caller supply a
#: different value. Finding a name in one of them anywhere in the kit is what
#: makes it a parameter rather than a constant.
PARAMETERISED = re.compile(r"\$\{\s*(\w+)\s*:[=\-?]")

#: `X=…`, `export X=…`, `local X=…`, `declare -r X=…`.
PLAIN_ASSIGN = re.compile(r"\A\s*(?:export\s+|local\s+|declare\s+-\w+\s+)?(\w+)=(.*)\Z")

#: `${X}`, `${X:-…}`, `$X`.
VAR_REF = re.compile(r"\$\{(\w+)[^}]*\}|\$(\w+)")

#: Whether anything survives once every variable reference is removed. A value
#: built only out of other variables and punctuation — `"${HOST}:${PORT}"` — is
#: as parameterised as the variables it is made of.
ALNUM = re.compile(r"[A-Za-z0-9]")

#: `re` flag names the layout may ask for, so the yaml can say `IGNORECASE`
#: without this file evaluating a string as code.
FLAGS = {"IGNORECASE": re.IGNORECASE, "MULTILINE": re.MULTILINE, "DOTALL": re.DOTALL}


def compiled(spec, default_flags: int = 0) -> re.Pattern:
    """A pattern from the layout. A string, or `{pattern:, flags: [...]}`."""
    if isinstance(spec, dict):
        flags = default_flags
        for name in spec.get("flags") or []:
            flags |= FLAGS[name]
        return re.compile(spec["pattern"], flags)
    return re.compile(spec, default_flags)


# --------------------------------------------------------------------------- #
# substance


def content_lines(text: str) -> list[str]:
    """Lines that carry content: non-blank, not a heading, not a fence marker.

    Headings are excluded on purpose. A document that is four `##` lines with
    nothing under them is exactly the failure a floor exists to catch, and
    counting the headings would let it pass.
    """
    return [
        line
        for line in text.splitlines()
        if line.strip() and not FENCE.match(line) and not HEADING.match(line)
    ]


def command_lines(text: str) -> list[str]:
    """Non-blank lines inside a code block — fenced, or indented by four.

    What "copy-pasteable commands" reduces to when it has to be counted. It does
    not try to decide whether a line is a *valid* command: that would be a shell
    parser, and a check that guesses at shell syntax fails honest kits.
    """
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if line.strip() and (fenced or line.startswith("    ") or line.startswith("\t")):
            out.append(line.strip())
    return out


def read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _validate(name: str, document) -> None:
    """`schema_lib.validate`, and nothing else. Kept as a seam, not as logic.

    This wrapper carried a registry-free fallback for one revision, on the stated
    grounds that no schema in `assets/schemas/` uses `$ref`. **That was false** —
    `kernel_optimization` and `workset` reference `environment.schema.json` at
    three sites, so the fallback would have validated those two *without
    resolving the reference*: a silently weaker check wearing a comment claiming
    equivalence, which is the failure this package exists against.

    `schema.validate` now inlines cross-file refs, so it needs nothing beyond
    `jsonschema` and behaves identically under the run's interpreter and under a
    bare `/usr/bin/python3`. The wrapper stays because the call site reads better
    with a name, and because the next person tempted to put a fallback here
    should read the paragraph above first.
    """
    schema_lib.validate(name, document)


def dotted(document, path: str):
    """`fixed.gpu_arch` out of a loaded document, or `None` if it is not there."""
    cursor = document
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


# --------------------------------------------------------------------------- #
# the entries


def check_invariant(rule: dict, record, where: str) -> list[str]:
    """One `invariants:` row — a relation between two fields of one document.

    `count_of` names a list; `at_most` names the number it may not exceed.
    Deliberately not a general expression language: the layout is a document a
    kit author reads, and one relation spelled in words is readable where an
    embedded grammar is not. A second relation adds a key here, not a parser.

    `required_unless` is the half that makes this more than advisory. The field
    is **optional in the schema** — the leader's call and the right one, because
    a record written before the producer criterion existed is still a valid
    record and omitting the field honestly says *"this run did not record which
    devices it took"*, where `[]` would falsely claim it took none. But **a real
    bring-up knows the answer**, so absence is a fault unless the named path is
    set. That path is `runtime.replayed_from`: a replay is describing another
    day's deployment and cannot know which devices *it* took, which is the same
    distinction the rendering comparison above already draws.
    """
    values = dotted(record, rule["count_of"])
    limit = dotted(record, rule["at_most"])

    if values is None:
        if rule.get("on_absent", "fault") == "skip":
            print(
                f"check_deploy_kit: {where}: {rule['count_of']} absent, and this "
                f"invariant is declared `on_absent: skip` — not applied. "
                f"Presence is not yet required of anyone; see the layout."
            )
            return []
        exempt = rule.get("required_unless")
        if exempt and dotted(record, exempt):
            print(
                f"check_deploy_kit: {where}: {rule['count_of']} absent, and "
                f"{exempt} is set — a replay cannot know which devices it took, "
                f"so this invariant is not applied"
            )
            return []
        return [f"{where}: {rule['err']}"]

    if not isinstance(values, list):
        return [f"{where}: {rule['count_of']} is {type(values).__name__}, not a list"]
    if not isinstance(limit, int):
        # Its own schema entry reports the wrong type; comparing against it here
        # would report the same fault twice under a less useful name.
        return []
    if len(values) > limit:
        # **`err` is the *absent* case's message and must not be appended here.**
        # It reads "fixed.gpu_devices is absent", which is false when the field is
        # present and too long — so the verdict said the field was missing while
        # quoting its nine entries. Caught in the gate's own output 2026-09-04.
        return [
            f"{where}: {rule['count_of']} has {len(values)} entries "
            f"({values}) but {rule['at_most']} is {limit} — a deployment cannot "
            f"take more cards than the node has."
        ]
    return []


def check_entry(entry: dict, roots: dict[str, Path], layout: dict) -> list[str]:
    """One row of `entries:`. Every fault it has, not only the first."""
    anchor = entry["anchor"]
    where = f"{anchor}/{entry['path']}"
    target = roots[anchor] / entry["path"]
    required = entry.get("required", True)

    if entry["kind"] == "dir":
        if not target.is_dir():
            return [f"{where}/: missing"] if required else []
        faults = []
        if entry.get("non_empty") and not any(p.is_file() for p in target.rglob("*")):
            faults.append(f"{where}/: empty")
        floor = entry.get("min_files")
        if floor:
            found = sorted(
                p.name
                for p in target.rglob(floor["glob"])
                if p.is_file() and (p.stat().st_size or not floor.get("non_empty"))
            )
            if len(found) < floor["count"]:
                faults.append(
                    f"{where}/: {len(found)} file(s) matching {floor['glob']}, "
                    f"needs {floor['count']} (found: {found})"
                )
        return faults

    if not target.is_file():
        return [f"{where}: missing"] if required else []

    text = read(target)
    if text is None:
        return [f"{where}: unreadable"]

    faults: list[str] = []

    floor = entry.get("min_content_lines")
    if floor is not None:
        lines = content_lines(text)
        if len(lines) < floor:
            faults.append(f"{where}: {len(lines)} content lines, needs {floor}")

    floor = entry.get("min_command_lines")
    if floor is not None:
        commands = command_lines(text)
        if len(commands) < floor:
            faults.append(
                f"{where}: {len(commands)} command lines in code blocks, needs {floor}"
            )

    for heading in entry.get("require_headings") or []:
        if not re.search(
            rf"^\s*#{{1,6}}\s*{re.escape(heading)}\b", text, re.IGNORECASE | re.MULTILINE
        ):
            faults.append(f"{where}: no `{heading}` heading")

    # The document this handoff is actually built on. Validated against the
    # package's own schema through the loader the producer also calls, so the two
    # sides cannot be looking at different documents (mission G2).
    name = entry.get("schema")
    if name:
        try:
            document = schema_lib._read_doc(target)
        except Exception as exc:  # a malformed yaml never reaches the validator
            faults.append(f"{where}: not loadable as a document: {exc}")
        else:
            try:
                _validate(name, document)
            except schema_lib.SchemaError as exc:
                faults += [f"{where}: {line}" for line in str(exc).splitlines()]

    # Record-internal invariants (todo.md T19). **Declared here rather than in
    # `environment.schema.json` because JSON Schema relates a value to a
    # constant, and these relate two fields of the same document to each
    # other** — no `$ref` or keyword expresses `len(a) <= b`. Like every other
    # rule this validator applies, the rule lives in the layout and this is only
    # its interpreter.
    invariants = entry.get("invariants") or []
    if invariants:
        try:
            record = schema_lib._read_doc(target)
        except Exception:
            record = None  # the `schema` entry above already reported it, once
        if record is not None:
            for rule in invariants:
                faults += check_invariant(rule, record, where)

    # M1.1.1: a rendering, not the record. Every value the layout marks
    # load-bearing must appear in the rendering verbatim.
    rendering = entry.get("rendered_from")
    if rendering:
        source = roots[rendering["anchor"]] / rendering["document"]
        if not source.is_file():
            return faults  # its absence is its own entry's fault to report, once
        try:
            document = schema_lib._read_doc(source)
        except Exception as exc:
            faults.append(f"{where}: cannot read {rendering['document']} to check it renders it: {exc}")
        else:
            # **A replayed kit's packup is a document about another day.**
            # `runtime.replayed_from` is set by `mock_adapt.sh` when this handoff
            # is a deployment stand-in: `results/` is the sealed run's evidence
            # and the record describes *today's* node, because that is the
            # machine every later stage will run on. Both are true and they are
            # about different machines, so comparing them is comparing two things
            # that were never claims about the same host — the check would fail
            # for a correct artefact.
            #
            # So in replay mode the equality comparison is **not** made, and the
            # verdict says which mode it took rather than passing silently. What
            # is still checked is everything else about the rendering: it must
            # exist, meet its substance floor, and carry no placeholder.
            #
            # **The real-mode rule is unchanged and deliberately so.** With
            # `replayed_from` absent this is exactly as strict as before — it is
            # the comparison that caught a wrong `--var` reaching a real record,
            # and that is the case it exists for.
            replayed = dotted(document, "runtime.replayed_from")
            if replayed:
                print(
                    f"check_deploy_kit: {where}: replayed kit "
                    f"(runtime.replayed_from={replayed}) — checked for substance, "
                    f"NOT compared against the live record"
                )
            else:
                for field in rendering.get("must_render") or []:
                    value = dotted(document, field)
                    if value is None:
                        continue  # its absence is the schema's fault to report, not this one's
                    if str(value) not in text:
                        faults.append(
                            f"{where}: does not render {field} = {value!r} from "
                            f"{rendering['document']}; this file is a rendering of that "
                            f"record and the two must not disagree"
                        )

    return faults


# --------------------------------------------------------------------------- #
# the evidence rules


def evidence_files(root: Path, over: dict, roots: dict[str, Path]) -> list[tuple[str, str]]:
    """Every readable file under the rule's subtree, as `(relative name, text)`."""
    base = roots[over["anchor"]] / over["dir"]
    if not base.is_dir():
        return []
    out = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        text = read(path)
        if text is not None:
            out.append((str(path.relative_to(base)), text))
    return out


def check_evidence(rule: dict, roots: dict[str, Path]) -> list[str]:
    """One row of `evidence:`. Three predicate shapes, chosen by which key is set."""
    over = rule["over"]
    label = f"{over['anchor']}/{over['dir']}/"
    files = evidence_files(roots[over["anchor"]], over, roots)
    if not files:
        return []  # an absent or empty directory is already reported by its entry

    faults: list[str] = []

    # `forbid`: no file may match. The captured group, when there is one, is what
    # goes in the message — the offending value rather than the whole line.
    if "forbid" in rule:
        pattern = compiled(rule["forbid"])
        for name, text in files:
            found = pattern.search(text)
            if found:
                offending = found.group(1) if found.groups() else found.group(0)
                faults.append(f"{label}{name}: {offending!r}: {rule['message']}")

    # `require_each`: every listed pattern must be found in *some* file. The
    # files may differ — that is the point when the rule is "two independent
    # components said so".
    for want in rule.get("require_each") or []:
        pattern = compiled(want["pattern"])
        if not any(pattern.search(text) for _, text in files):
            faults.append(f"{label}: no {want['label']}: {rule['message']}")

    # `require_together`: every listed pattern must be found in **one** file.
    together = rule.get("require_together")
    if together:
        patterns = [compiled(p) for p in together]
        if not any(all(p.search(text) for p in patterns) for _, text in files):
            faults.append(f"{label}: {rule['message']}")

    return faults


# --------------------------------------------------------------------------- #
# shared-namespace identifiers


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


def check_shared_identifiers(rule: dict, roots: dict[str, Path]) -> list[str]:
    """Identifiers the kit freezes and then binds into a host-wide namespace.

    Two facts are collected over the scanned subtree and a fault needs both: a
    name is **frozen** when a plain assignment gives it a value that is not built
    purely out of other variables, and **bound** when it reaches one of the
    layout's `binding_flags`.
    """
    scan = rule["scan"]
    base = roots[scan["anchor"]] / scan["dir"]
    if not base.is_dir():
        return []  # its absence is already a fault, reported by its entry

    binding = list(rule["binding_flags"])
    docker_only = set(rule.get("docker_only_flags") or ())
    literal_forbidden = set(rule.get("literal_forbidden_flags") or ())

    # Built from the layout's own list rather than repeated: a second spelling of
    # the set is how the document a reader checks and the pattern that runs come
    # to disagree. Longest first, so `--publish` is never matched as `-p`
    # followed by text.
    flag_use = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(f) for f in sorted(binding, key=len, reverse=True))
        + r")[=\s]+(\S+)"
    )

    sources: list[tuple[str, str]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        text = read(path)
        if text is not None:  # a binary in scripts/ is not our business
            sources.append((str(path.relative_to(base)), text))

    parameterised = {name for _, text in sources for name in PARAMETERISED.findall(text)}

    frozen: dict[str, str] = {}
    bound: dict[str, str] = {}
    faults: list[str] = []

    for filename, text in sources:
        for number, line in logical_lines(text):
            where = f"{scan['dir']}/{filename}:{number}"

            assigned = PLAIN_ASSIGN.match(line)
            if assigned:
                name, value = assigned.group(1), assigned.group(2)
                remainder = VAR_REF.sub("", value).strip("\"' \t")
                if ALNUM.search(remainder) and name not in frozen:
                    frozen[name] = where

            for flag, token in flag_use.findall(line):
                if flag in docker_only and "docker" not in line:
                    continue
                referenced = [a or b for a, b in VAR_REF.findall(token)]
                for name in referenced:
                    bound.setdefault(name, where)
                if not referenced and flag in literal_forbidden:
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


# --------------------------------------------------------------------------- #
# the runtime contract


def check_runtime_contract(contract: dict, scan_rule: dict, roots: dict[str, Path]) -> list[str]:
    """Each declared parameter is honoured in a defaulting form under `scripts/`.

    The defaulting forms only — `${X:=…}`, `${X:-…}`, `${X:?…}`. A bare `$X` is
    not enough and the distinction is the whole point: a kit that reads `$X`
    without a default runs correctly for a caller who sets it and breaks for the
    reproducer who does not, which is the reader most likely to be holding this
    kit.

    Where the kit *writes* its handshake is not checkable statically — that is
    what `check_deploy_serves` is for, and it fails loudly when the file does not
    appear. What is checkable here is that the kit was written against these
    names at all, and that is what this does.
    """
    base = roots[scan_rule["anchor"]] / scan_rule["dir"]
    if not base.is_dir():
        return []

    seen: set[str] = set()
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        text = read(path)
        if text is not None:
            seen.update(PARAMETERISED.findall(text))

    faults = []
    for parameter in contract.get("parameters") or []:
        if parameter["name"] in seen:
            continue
        # **`err` is the operator-facing one-liner; `brief` is documentation.**
        # A `brief` here runs to paragraphs, and emitting it put an entire essay
        # into a run's verdict — measured 2026-09-04, on the first run after a
        # seventh parameter was added. A verdict a person has to scroll is a
        # verdict they skim. Falling back to the brief's first sentence keeps the
        # six older parameters, which carry no `err`, readable too.
        why = parameter.get("err") or " ".join(parameter["brief"].split()).split(". ")[0] + "."
        faults.append(f"{scan_rule['dir']}/: nothing reads ${{{parameter['name']}:=…}}. {' '.join(why.split())}")

    # **There is deliberately no static mount check here**, and the reason is a
    # measured false positive rather than an omission. The layout requires the
    # work root to be writable from inside the engine container
    # (`runtime_contract.writable_work_root`), and the obvious static form —
    # "`E2E_KIT_WORK_ROOT` must appear in a `--volume`" — fails the real sealed
    # kit, which mounts `"${DK_RUN_DIR}:/workdir"` where `DK_RUN_DIR` is derived
    # from the work root. A checker cannot follow that derivation without
    # evaluating the shell, and one that guesses rejects a kit that is correct.
    #
    # So the rule is enforced where it can be answered instead of guessed:
    # `check_deploy_serves` writes a file through the running container after
    # bring-up. That is the same evidence-over-inference rule the mode readback
    # follows.
    return faults


# --------------------------------------------------------------------------- #
# the whole kit


def resolve_anchors(content: Path, layout: dict) -> tuple[dict[str, Path] | None, str]:
    """The layout's anchors against one staged handoff content directory."""
    roots: dict[str, Path] = {}
    for name, spec in layout["anchors"].items():
        if "path" in spec:
            path = content / spec["path"]
            if not path.is_dir():
                return None, f"no {spec['path']} directory"
            roots[name] = path
            continue

        under = content / spec["under"]
        if not under.is_dir():
            return None, f"no {spec['under']} directory"
        pattern = re.compile(spec["name_pattern"])
        found = sorted(e for e in under.iterdir() if e.is_dir() and pattern.match(e.name))
        if not found:
            loose = sorted(e.name for e in under.iterdir())
            return None, (
                f"no directory matching {spec['name_pattern']} under "
                f"{spec['under']} (found: {loose})"
            )
        if spec.get("cardinality") == "exactly_one" and len(found) > 1:
            return None, f"more than one {name} directory: {[e.name for e in found]}"
        roots[name] = found[0]
    return roots, roots["packup"].name if "packup" in roots else "ok"


def check_kit(content: Path, layout: dict) -> tuple[list[str], str]:
    roots, why = resolve_anchors(content, layout)
    if roots is None:
        return [why], why

    faults: list[str] = []
    for entry in layout["entries"]:
        faults += check_entry(entry, roots, layout)

    placeholders = layout.get("placeholders")
    if placeholders:
        pattern = compiled(placeholders, 0)
        for entry in layout["entries"]:
            if entry["path"] not in placeholders["applies_to"]:
                continue
            target = roots[entry["anchor"]] / entry["path"]
            text = read(target) if target.is_file() else None
            if text is None:
                continue
            found = pattern.search(text)
            if found:
                faults.append(
                    f"{entry['anchor']}/{entry['path']}: unfilled placeholder {found.group(0)!r}"
                )

    contract = layout.get("runtime_contract")
    if contract:
        faults += check_runtime_contract(contract, layout["shared_identifiers"]["scan"], roots)

    for rule in layout.get("evidence") or []:
        faults += check_evidence(rule, roots)

    if layout.get("shared_identifiers"):
        faults += check_shared_identifiers(layout["shared_identifiers"], roots)

    return faults, why


def main() -> int:
    parameters = zone.args()
    name = parameters.get("layout", "deploy_kit.layout")
    path = schema_lib.package_root() / "assets" / "schemas" / f"{name}.yaml"
    try:
        import yaml

        layout = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        # A layout this body cannot read is not a kit that passes. Refusing every
        # input is the only honest verdict, and it names the file.
        print(f"check_deploy_kit: cannot load layout {path}: {exc}", file=sys.stderr)
        zone.write_verdict({hid: False for hid in zone.inputs()})
        return 1

    results: dict[str, bool] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            results[hid] = False
            print(f"check_deploy_kit: {hid}: no staged content")
            continue
        faults, why = check_kit(content, layout)
        results[hid] = not faults
        print(f"check_deploy_kit: {hid}: {why}")
        for fault in faults:
            print(f"check_deploy_kit: {hid}: FAIL: {fault}")
    zone.write_verdict(results)
    print(f"check_deploy_kit: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
