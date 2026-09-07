#!/usr/bin/env python3
"""What `harness` runs: turn every submission into ONE executable.

**This file imports nothing from `agent_sys`.** It is package data, run as a
subprocess by `agent.backends.program.ProgramExecutor`. `assets/lib/cpp.py` is
the only thing it imports that it did not bring with it, and that is package
data too.

| | |
|---|---|
| `AGENT_SYS_INPUT_PROBLEMS` | the problem set; its worked examples are copied into the manifest |
| `AGENT_SYS_INPUT_SOLUTIONS_A/B/C` | the three students' submissions |
| `AGENT_SYS_INPUT_REVIEW` | the reconciled review; its verdicts are copied into the manifest |
| `AGENT_SYS_OUTPUT_HARNESS` | where to write `README.md` and `items/` |

`readme.md` beside this file is the design; this docstring records only the two
things a reader of the code needs up front.

**One translation unit, because `cpp.compile` builds one.** The frozen helper is
`compile(src, out)` — `g++ -O2 -std=c++17 -o out src` — with no object files
and no link step, so thirty-six solutions have to become one `.cpp`. Each is
wrapped in its own namespace, which makes its globals and its `main` distinct
symbols without editing a line of what the student wrote.

**The case id arrives on `argv[1]`.** The stdin fallback is kept for a human
typing `harness < case.txt` by hand, and it is documented as unsafe rather than
removed, because a reader who finds it needs to know why not to use it.

Measured (`scratch/demo2-2026-08/probe_fgets_eats_stdin.py`, seven programs): a
solution's `std::ios::sync_with_stdio(false)` discards the position **any**
earlier stdio read left, so `fgets`, `getchar` and `getline` all starve it
equally. The function is not the variable; reading stdin at all before the
solution runs is. The first full demo2 run went through the stdin form and every
case produced empty output with return code 0 — every student scored exactly
30.0 and every validator passed.
"""

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import cpp  # noqa: E402 — the path insert above is what makes it importable

#: The three students, and the environment variable each one's submissions
#: arrive under. A list rather than a loop over `AGENT_SYS_INPUT_SOLUTIONS_*`
#: because the set is closed by the graph: `models.py`'s single-slot
#: `producer_of[kind]` is why there are three kinds and not one, so a fourth
#: student is a change to `main.yaml` and should be a change here too.
STUDENTS = ("a", "b", "c")

#: `#include` has to leave the namespace: a system header declares things at
#: global scope and nesting one changes every name it introduces.
INCLUDE_LINE = re.compile(r"^\s*#\s*include\b.*$")

#: The macro name a `#define` introduces, so it can be `#undef`ed when the
#: submission's namespace closes. Function-like and object-like macros are the
#: same shape up to the name, which is all that `#undef` takes.
DEFINE_NAME = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")

#: A submission's entry point. `signed main()` is admitted because it is half of
#: the `#define int long long` idiom and a solution using it is not malformed.
#: `auto` is admitted because `auto main() -> int` is legal and somebody writes
#: it. The captured group is the parameter list, which decides how it is called.
MAIN_SIGNATURE = re.compile(
    r"^[ \t]*(?:int|signed|auto)[ \t\r\n]+main[ \t]*\(([^)]*)\)",
    re.MULTILINE,
)

README = """# harness

## Purpose

One executable built from every submitted solution. It takes a case id
`<student>/<problem_id>`, runs that student's solution to that problem with the
test input on stdin, and writes the answer to stdout.

There is exactly **one** binary, `items/codes/bin/harness`, and it answers to
{n_cases} case ids across {n_problems} problems. `items/manifest.json` lists
them, together with the worked examples and the review verdicts that `score`
needs and does not otherwise receive.

## Interface

```
harness --list                    every case id, one per line
harness <student>/<problem_id>    run that solution; test input on stdin
harness                           read the case id from the first line of
                                  stdin, then hand the rest to the solution
```

Exit status is the solution's own, except: **2** for an unrecognised or missing
case id, with a message on stderr. That is distinguishable from a solution that
ran and answered wrongly, which exits 0.

**The stdin form is a convenience for a human and is unsafe for a solution that
desyncs.** `assets/lib/cpp.py::run` takes `argv` and `score.py` uses it; the
fallback survives only so that `harness < case.txt` does something sensible at a
prompt. It starves any solution that calls `sync_with_stdio(false)` — which is
every one this package has produced — and no choice of stdio function repairs
that.

`items/manifest.json`:

```json
{{"binary": "codes/bin/harness",
 "cases": ["a/p1", "b/p1", ...],
 "units": [{{"case_id": "a/p1", "student": "a", "problem_id": "p1",
            "namespace": "sol_a_p1", "source": "<path in the submission>"}}],
 "problems": [{{"problem_id": "p1",
               "examples": [{{"input": "...", "expected": "..."}}]}}],
 "reviews": [{{"student": "a", "problem_id": "p1", "verdict": "accept"}}],
 "excluded": [{{"case_id": "c/p4", "reason": "...g++ diagnostics..."}}],
 "build": {{"ok": true, "flags": {flags}}}}}
```

## Boundary

**What this is not.** It does not run any test, judge any output, or score
anything — `score` does all three. It does not decide whether a solution is
correct; it only decides whether it *builds*, and a solution that does not
build is dropped by name into `excluded` rather than silently omitted.

**One translation unit.** Every submission is wrapped in its own namespace and
its macros are `#undef`ed at the closing brace, so one solution's `#define`
cannot reach another. Two things this does not cover, stated rather than
hidden: an `#include` written inside an `#ifdef` is hoisted out of its guard,
and a submission whose text is not self-contained C++ will not build here even
if it built alone.

**The examples and the verdicts in this manifest are copies**, made
mechanically from `$AGENT_SYS_INPUT_PROBLEMS` and `$AGENT_SYS_INPUT_REVIEW`.
Nothing was selected and nothing was summarised. They are here because `score`
does not consume those kinds and needs both; `assets/harness.task/readme.md`
records why that route was taken and what it costs.

{excluded_note}
"""


def _required(name: str) -> str:
    """A named refusal instead of a bare `KeyError`."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. A task body reads its inputs from "
            f"AGENT_SYS_INPUT_<KIND> and writes into AGENT_SYS_OUTPUT_<KIND>, "
            f"exported per slot by env_mgr.grants at dispatch."
        )
    return value


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid JSON: {exc}") from exc


# ------------------------------------------------------------------ discovery


def submissions(root: Path, student: str) -> list[dict]:
    """Every `(problem_id, source)` under one student's `items/codes/`.

    **The problem id is the first path component below `codes/`**, which is what
    `assets/solve_a.task/readme.md` asks the students for —
    `items/codes/<problem_id>/solution.cpp` — and all that
    `scratch/demo2-2026-08/CONTRACT.md` §1 fixes: *"one dir per problem"*. The
    file name is not assumed: the search is a recursive glob and the id comes
    from the top of the relative path, so a student who nests or renames still
    lands in the right place.

    A directory holding more than one `.cpp` is reported and the first is taken,
    sorted, so the choice is at least reproducible. `check_compiles` is the
    validator that has an opinion about a submission's shape; this one only has
    to build what is there.
    """
    codes = root / "items" / "codes"
    if not codes.is_dir():
        raise SystemExit(
            f"{codes} does not exist. `content_type: code` requires an item named "
            f"'codes' (handoff/content.py:69-74), so a solutions handoff without "
            f"one would not have passed its own content check."
        )
    found: dict[str, list[Path]] = {}
    for source in sorted(codes.rglob("*.cpp")):
        parts = source.relative_to(codes).parts
        if len(parts) < 2:
            # A `.cpp` sitting directly in `codes/` names no problem. Skipped
            # rather than guessed at, and named on stderr.
            print(f"harness: {source} is not under a problem directory; skipped", file=sys.stderr)
            continue
        found.setdefault(parts[0], []).append(source)

    out: list[dict] = []
    for problem_id in sorted(found):
        sources = found[problem_id]
        if len(sources) > 1:
            print(
                f"harness: {student}/{problem_id} has {len(sources)} sources; "
                f"taking {sources[0].name}",
                file=sys.stderr,
            )
        out.append({"student": student, "problem_id": problem_id, "source": sources[0]})
    return out


def problem_examples(content: Path) -> list[dict]:
    """The worked examples, per problem, copied out of the problems artefact.

    The shape is `assets/problems.task/readme.md`'s:
    `{"problems": [{"id": ..., "examples": [{"input", "output", "note"}]}]}`.
    `CONTRACT.md` §1 fixes only that `problems` is `structured_text` with a
    `text.json`, so the shape is the *setter's* readme rather than the contract
    — and the alternative spellings below are tolerance across a boundary two
    authors own, not a second contract. They cost a `.get` each and turn a
    rename in that readme into a degraded manifest rather than a dead run.
    """
    data = _load_json(content / "items" / "text.json")
    entries = data.get("problems") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise SystemExit(
            f"{content / 'items' / 'text.json'}: expected a list of problems, or an "
            f"object with a 'problems' list"
        )
    out: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        problem_id = entry.get("id") or entry.get("problem_id") or entry.get("slug")
        raw = entry.get("examples") or entry.get("worked_examples") or entry.get("cases") or []
        examples = [
            {
                "input": str(ex.get("input") or ""),
                "expected": str(ex.get("expected") or ex.get("output") or ""),
            }
            for ex in raw
            if isinstance(ex, dict)
        ]
        out.append(
            {
                "problem_id": str(problem_id) if problem_id else "",
                "examples": [e for e in examples if e["input"] and e["expected"]],
            }
        )
    return out


def review_verdicts(content: Path) -> list[dict]:
    """The reconciled verdict per pair. A disagreed pair has no agreed verdict.

    `reconcile` writes `agreed` and `disagreed`, and only the first carries a
    single answer. A pair in `disagreed` is recorded here with a `null` verdict
    rather than being dropped, so that `score` scores it as unreviewed instead
    of silently omitting the pair. `check_reviews_agree` will already have
    failed the run over it.
    """
    data = _load_json(content / "items" / "text.json")
    if not isinstance(data, dict):
        raise SystemExit(f"{content / 'items' / 'text.json'}: expected a JSON object")
    out: list[dict] = []
    for row in data.get("agreed") or []:
        if isinstance(row, dict):
            out.append(
                {
                    "student": str(row.get("student")),
                    "problem_id": str(row.get("problem_id")),
                    "verdict": row.get("verdict"),
                }
            )
    for row in data.get("disagreed") or []:
        if isinstance(row, dict):
            out.append(
                {
                    "student": str(row.get("student")),
                    "problem_id": str(row.get("problem_id")),
                    "verdict": None,
                }
            )
    return sorted(out, key=lambda r: (r["student"], r["problem_id"]))


# ----------------------------------------------------------------- generation


def namespace_for(unit: dict, taken: set[str]) -> str:
    """`sol_<student>_<problem_id>`, sanitised, and unique within the file.

    Two problem ids differing only in punctuation would sanitise to one name and
    then silently redefine each other's `main`, so collisions are resolved with
    a counter rather than left to chance.
    """
    base = "sol_" + re.sub(r"\W", "_", f"{unit['student']}_{unit['problem_id']}")
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name


def wrap(source_text: str, namespace: str) -> tuple[list[str], str, bool]:
    """One submission, ready to paste into the combined unit.

    Returns `(includes, block, main_takes_args)`:

    - `includes` — every `#include` line, lifted out. They must be at global
      scope, and de-duplication happens at the call site because it is global to
      the whole generated file.
    - `block` — `namespace <namespace> { ...the rest verbatim... }` followed by
      an `#undef` for every macro the submission defined. The `#undef` pass is
      what keeps one solution's `#define int long long` from reaching the next,
      which is the single hazard one translation unit has that separate ones do
      not.
    - `main_takes_args` — whether its `main` was declared with parameters, which
      decides how the dispatcher calls it.

    **The submission's text is not otherwise edited.** No renaming, no
    reformatting, nothing removed. A namespace is enough to make every symbol in
    it distinct, including `main`: the rules that make `main` special apply to
    the one at global scope.
    """
    includes: list[str] = []
    defines: list[str] = []
    body: list[str] = []
    for line in source_text.splitlines():
        if INCLUDE_LINE.match(line):
            includes.append(line.strip())
            continue
        found = DEFINE_NAME.match(line)
        if found:
            defines.append(found.group(1))
        body.append(line)

    signature = MAIN_SIGNATURE.search(source_text)
    params = (signature.group(1) if signature else "").strip()
    main_takes_args = bool(params) and params != "void"

    lines = [f"namespace {namespace} {{", *body, f"}}  // namespace {namespace}"]
    lines += [f"#undef {name}" for name in dict.fromkeys(defines)]
    return includes, "\n".join(lines), main_takes_args


def generate(units: list[dict]) -> str:
    """The whole combined translation unit, as text.

    Order: every `#include` first, then one namespace per submission, then the
    dispatcher. The dispatcher is last so that it calls names already declared
    and so that every `#undef` has run before its own code is compiled.
    """
    includes: list[str] = ["#include <cstdio>", "#include <cstring>", "#include <string>"]
    blocks: list[str] = []
    for unit in units:
        unit_includes, block, takes_args = wrap(unit["text"], unit["namespace"])
        includes += unit_includes
        blocks.append(block)
        unit["main_takes_args"] = takes_args

    arms = []
    for unit in units:
        call = (
            f"{unit['namespace']}::main(1, demo2_argv)"
            if unit["main_takes_args"]
            else f"{unit['namespace']}::main()"
        )
        arms.append(f'  if (id == "{unit["case_id"]}") return static_cast<int>({call});')

    listing = "\n".join(f'  "{unit["case_id"]}",' for unit in units)

    return "\n".join(
        [
            "// GENERATED by assets/harness.task/build.py. Do not edit.",
            "//",
            "// One translation unit holding every submitted solution, each in its own",
            "// namespace, plus a dispatcher that selects one by case id. A function named",
            "// `main` inside a namespace is an ordinary function; only the global one is",
            "// special, and the dispatcher below is that one.",
            "",
            # `dict.fromkeys` rather than `sorted(set(...))`: the order a
            # submission wrote its includes in is occasionally load-bearing, and
            # first-seen order preserves it while still de-duplicating.
            *dict.fromkeys(includes),
            "",
            "namespace demo2_harness {",
            "static const char* const CASES[] = {",
            listing,
            "};",
            "static const int N_CASES = sizeof(CASES) / sizeof(CASES[0]);",
            "}  // namespace demo2_harness",
            "",
            *blocks,
            "",
            "// A parameter list for a solution whose `main` takes one. It is never the",
            "// harness's own argv: a solution must not see the case id it was selected by.",
            'static char demo2_prog[] = "harness";',
            "static char* demo2_argv[] = {demo2_prog, nullptr};",
            "",
            "int main(int argc, char** argv) {",
            "  std::string id;",
            "  if (argc > 1) {",
            "    id = argv[1];",
            "  } else {",
            "    // A convenience for a human at a prompt, and UNSAFE for a solution",
            "    // that calls sync_with_stdio(false): that discards the position any",
            "    // earlier stdio read left, so this consumes the input and the solution",
            "    // sees an empty stream. fgets, getchar and getline all do it. Drive",
            "    // the harness with argv[1]; see build.py's module docstring.",
            "    char buffer[512];",
            "    if (std::fgets(buffer, sizeof(buffer), stdin) != nullptr) {",
            "      id = buffer;",
            "    }",
            "  }",
            "  while (!id.empty() && (id.back() == '\\n' || id.back() == '\\r' ||",
            "                         id.back() == ' ' || id.back() == '\\t')) {",
            "    id.pop_back();",
            "  }",
            "",
            '  if (id == "--list") {',
            "    for (int i = 0; i < demo2_harness::N_CASES; ++i) {",
            '      std::printf("%s\\n", demo2_harness::CASES[i]);',
            "    }",
            "    return 0;",
            "  }",
            "  if (id.empty()) {",
            '    std::fprintf(stderr, "harness: no case id on argv[1] or on the first line '
            'of stdin\\n");',
            "    return 2;",
            "  }",
            "",
            *arms,
            "",
            "  std::fprintf(stderr, \"harness: unknown case id '%s'; try --list\\n\", id.c_str());",
            "  return 2;",
            "}",
            "",
        ]
    )


# ---------------------------------------------------------------------- build


def probe(unit: dict, workdir: Path) -> tuple[bool, str]:
    """Does this one submission build on its own?

    Only reached when the combined build failed. `cpp.compile` builds a whole
    program, so the probe is the submission's namespace plus a trivial global
    `main` — which is a whole program, and is the smallest one that exercises
    exactly what the combined file will do with this unit.
    """
    includes, block, _ = wrap(unit["text"], unit["namespace"])
    text = "\n".join([*dict.fromkeys(includes), "", block, "", "int main() { return 0; }", ""])
    src = workdir / f"{unit['namespace']}.cpp"
    src.write_text(text, encoding="utf-8")
    return cpp.compile(src, workdir / f"{unit['namespace']}.bin")


def build(units: list[dict], codes: Path) -> tuple[list[dict], list[dict], str]:
    """Build the one binary, dropping submissions that will not compile.

    Returns `(kept, excluded, diagnostics)`. The combined build is attempted
    first and is the only one that runs when everything is well; the per-unit
    probe costs one compile per submission and only on the failure path.
    """
    source = codes / "harness.cpp"
    source.write_text(generate(units), encoding="utf-8")
    ok, diagnostics = cpp.compile(source, codes / "bin" / "harness")
    if ok:
        return units, [], diagnostics

    print("harness: combined build failed; probing each submission", file=sys.stderr)
    workdir = Path(tempfile.mkdtemp(prefix="harness-probe-"))
    try:
        kept: list[dict] = []
        excluded: list[dict] = []
        for unit in units:
            unit_ok, unit_diagnostics = probe(unit, workdir)
            if unit_ok:
                kept.append(unit)
            else:
                excluded.append({"case_id": unit["case_id"], "reason": unit_diagnostics})
    finally:
        # The probe binaries must not survive: `check_one_binary` counts the
        # executables in the staged content and expects exactly one.
        shutil.rmtree(workdir, ignore_errors=True)

    if not excluded:
        # Every submission builds alone and the combination does not. That is a
        # fault in the generated file or an interaction between two submissions,
        # and it is not something to paper over by shipping nothing.
        raise SystemExit(
            "harness: the combined build failed but every submission builds on its "
            f"own, so the fault is in the generated unit rather than in a "
            f"submission. g++ said:\n{diagnostics}"
        )

    source.write_text(generate(kept), encoding="utf-8")
    ok, diagnostics = cpp.compile(source, codes / "bin" / "harness")
    if not ok:
        raise SystemExit(
            f"harness: build failed after excluding "
            f"{[e['case_id'] for e in excluded]}. g++ said:\n{diagnostics}"
        )
    return kept, excluded, diagnostics


# ----------------------------------------------------------------------- main


def main() -> int:
    problems_content = Path(_required("AGENT_SYS_INPUT_PROBLEMS"))
    review_content = Path(_required("AGENT_SYS_INPUT_REVIEW"))
    dst = Path(_required("AGENT_SYS_OUTPUT_HARNESS"))

    units: list[dict] = []
    taken: set[str] = set()
    for student in STUDENTS:
        root = Path(_required(f"AGENT_SYS_INPUT_SOLUTIONS_{student.upper()}"))
        for found in submissions(root, student):
            found["case_id"] = f"{found['student']}/{found['problem_id']}"
            found["namespace"] = namespace_for(found, taken)
            found["text"] = found["source"].read_text(encoding="utf-8", errors="replace")
            units.append(found)

    if not units:
        raise SystemExit("harness: no submission found under any student's items/codes/")

    codes = dst / "items" / "codes"
    codes.mkdir(parents=True, exist_ok=True)
    kept, excluded, diagnostics = build(units, codes)
    (codes / "build.log").write_text(diagnostics, encoding="utf-8")

    problems = problem_examples(problems_content)
    manifest = {
        "binary": "codes/bin/harness",
        "cases": [unit["case_id"] for unit in kept],
        "units": [
            {
                "case_id": unit["case_id"],
                "student": unit["student"],
                "problem_id": unit["problem_id"],
                "namespace": unit["namespace"],
                "source": str(unit["source"].name),
            }
            for unit in kept
        ],
        "problems": problems,
        "reviews": review_verdicts(review_content),
        "excluded": excluded,
        "build": {"ok": True, "flags": list(cpp.COMPILE_FLAGS)},
    }
    (dst / "items" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    note = (
        ""
        if not excluded
        else (
            "**{n} submission(s) were excluded** because they do not compile: "
            "{ids}. Their diagnostics are in `items/manifest.json` under "
            "`excluded`, and `items/codes/build.log` is g++'s output for the "
            "build that shipped."
        ).format(n=len(excluded), ids=", ".join(e["case_id"] for e in excluded))
    )
    (dst / "README.md").write_text(
        README.format(
            n_cases=len(kept),
            n_problems=len({unit["problem_id"] for unit in kept}),
            flags=json.dumps(list(cpp.COMPILE_FLAGS)),
            excluded_note=note,
        ),
        encoding="utf-8",
    )
    print(f"harness: {len(kept)} cases, {len(excluded)} excluded -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
