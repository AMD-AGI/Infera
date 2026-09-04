#!/usr/bin/env python3
"""Every `kind: ai` agent must declare the `E2E_*` it reads, and agree with `runner`.

**Why this exists, measured 2026-09-04 on rung 1.** `env_mgr/material.py:96` calls
`_declared_env(agent_spec)` on **the agent that is actually running**, and
`_declared_env` returns `{}` when the agent has no `env` block. `shared.yaml`
declares all 36 `E2E_*` on `runner`. So a `kind: ai` agent that declares none
receives **none** — and the survey when this was written was:

    runner                (program)  36
    e2e_deployer          (ai)        0
    workset_builder       (ai)        0
    e2e_kernel_optimizer  (ai)        1
    e2e_integrator        (ai)       32      <- the one that was right all along

Rung 1 ran with four `--var`s inert — `tp`, `image`, `instruction`,
`mock_stages` — **and looked correct**, because the agent read the sealed kit's
defaults and this node's free half happened to be the same half the sealed run
used. `tp_size: 1` was `DK_TP_SIZE:=1`; GPU 4 was `DK_GPU_ID:=4`, not the
instruction that told it to take 4–7. Right answer, wrong mechanism.

**Two checks, and the second is the one that matters.** m1's point when the
first was proposed: byte-identity catches *divergence* and cannot catch
*omission* — and omission is the failure that actually happened. An agent
declaring nothing has no mismatches, so a divergence-only check calls it clean.
**A check that would have passed the bug it was written for is this package's
recurring defect**, so:

  1. **divergence** — a name declared on a `kind: ai` agent must have exactly
     `runner`'s value, byte for byte;
  2. **omission** — a name the agent's own task assets *reference* must be
     declared on that agent.

Nobody is required to declare all 36: `e2e_deployer` genuinely does not read
m2's trace knobs, and declaring them would widen the staleness surface for no
gain. The minimum is what the code actually reads.

    python3 assets/lib/check_agent_env.py            # from the package root
    python3 assets/lib/check_agent_env.py --package <dir>

Exit 0 clean, 1 with every problem listed. It reads the yaml as text-with-
`${...}`-intact, deliberately: the point is that two *declarations* agree, not
that two substituted values happen to agree in one run.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

#: The one authority. Every other block is measured against this agent's.
AUTHORITY = "runner"

#: `(agent, variable)` pairs that differ from `runner` **on purpose**, each with
#: the reason. An exception carrying no reason is indistinguishable from drift,
#: which is the thing this file exists to prevent — so the reason is the entry.
DELIBERATE: dict[tuple[str, str], str] = {
    ("e2e_integrator", "E2E_TRACE_END_MS"): (
        "m5 measures TWICE. Two arms at the package's 180000 ms do not fit the "
        "CLI's 1800 s settle budget, and work finishing outside it is discarded. "
        "A one-armed and a two-armed stage cannot share a duration default — "
        "m5, 2026-09-04, and this is the reason rather than an exemption."
    ),
}

#: Names referenced by assets. Matches the shell and python spellings alike —
#: `$E2E_X`, `${E2E_X}`, `${E2E_X:-d}`, `os.environ["E2E_X"]`, `getenv('E2E_X')`.
_REF = re.compile(r"\bE2E_[A-Z0-9_]+")

#: Files worth grepping for a reference. A readme is included on purpose: for a
#: `kind: ai` closure the readme **is** the program, and a variable named only
#: in prose is one the agent will try to read.
_SUFFIXES = {".sh", ".py", ".md", ".yaml", ".yml"}


def _agents(doc) -> list[dict]:
    """Every agent spec in a loaded yaml document, at any depth."""
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("kind") in ("ai", "program") and "name" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    return found


def _e2e(agent: dict) -> dict[str, str]:
    env = agent.get("env") or {}
    return {k: str(v) for k, v in env.items() if k.startswith("E2E_")}


def _task_dirs(doc, agent_name: str, assets: pathlib.Path) -> list[pathlib.Path]:
    """Asset directories of the closures that run on `agent_name`.

    A closure names its agent as `'${m1_agent:-e2e_deployer}'`, so the match is
    on **containment** rather than equality — the default is the interesting
    half, since that is the agent a run uses unless someone swaps it out.

    **The assets live at `assets/<closure name>.task`, not at anything the
    closure's `task:` key says** — `task:` is a *dict* (`goal`, `version`,
    `repos`, …), not a path. The first version of this function read it as a
    path, found nothing, and reported every agent clean: **a probe that could
    not fail, written into the check whose entire purpose is catching probes
    that cannot fail.** Caught by asserting the directory count is non-zero
    before trusting a pass, which is the only reason it is not still here.
    """
    dirs: list[pathlib.Path] = []

    def walk(node):
        if isinstance(node, dict):
            agent = node.get("agent")
            closure = node.get("name")
            if isinstance(agent, str) and agent_name in agent and isinstance(closure, str):
                candidate = assets / f"{closure}.task"
                if candidate.is_dir():
                    dirs.append(candidate)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    return dirs


#: Never grepped for a `kind: ai` agent. **An ai closure does not run
#: `entry.sh`** — the mock switch works precisely because `--var mN_agent=runner`
#: swaps in the program agent that does. So a variable read only there is not a
#: dependency of the ai path, and requiring it would declare one that cannot
#: exist. m4's objection, and they were right: the first version of this check
#: demanded `E2E_MOCK_STAGES` on an agent that can never reach `mock.sh`.
_PROGRAM_ONLY = {"entry.sh"}


#: `assets/lib/<name>` mentioned by a task's readme is followed. **A readme is
#: the program for a `kind: ai` closure**, so a script it tells the agent to run
#: is part of that agent's dependency set even though it lives outside the task
#: directory. m1's counterexample: `deploy_and_prove.task/readme.md:71` invokes
#: `mock.sh` at STEP 0, and `mock.sh` reads `E2E_MOCK_STAGES` — a variable that,
#: **undeclared, makes `mock.sh` unable to tell *unset* from *not listed*, so
#: `--var mock_stages=all` runs the stage for real.** That is rung 1's own bug,
#: and without this the checker written to prevent it could not see it.
_LIB_REF = re.compile(r"\b([a-z_][a-z0-9_]*\.(?:sh|py))\b")


def _referenced(dirs: list[pathlib.Path], lib: pathlib.Path) -> dict[str, str]:
    """`E2E_*` names appearing under those directories, plus libs their readmes name."""
    seen: dict[str, str] = {}
    followed: set[pathlib.Path] = set()
    for directory in dirs:
        for readme in directory.rglob("readme.md"):
            try:
                text = readme.read_text(errors="replace")
            except OSError:
                continue
            for candidate in set(_LIB_REF.findall(text)):
                target = lib / candidate
                if target.is_file():
                    followed.add(target)
    for path in sorted(followed):
        try:
            for name in _REF.findall(path.read_text(errors="replace")):
                seen.setdefault(name, f"{path} (named by a readme)")
        except OSError:
            pass
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in _SUFFIXES:
                continue
            if path.name in _PROGRAM_ONLY:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for name in _REF.findall(_strip_comments(path, text)):
                seen.setdefault(name, str(path))
    return seen


def _strip_comments(path: pathlib.Path, text: str) -> str:
    """Drop whole-line `#` comments from code. **Not from markdown.**

    m4, 2026-09-04: their fix removed a reader of `E2E_MEASURE_GPU` and left the
    name in the comment *explaining why they no longer use it* — and this check
    still reported it. **A false problem caused by documentation**, and the
    repair they were pushed into was to reword the comment. Their words: *"the
    fix made the comment slightly worse to keep a grep happy."* A checker that
    makes people stop naming variables in comments is doing net harm.

    Markdown is deliberately exempt: for a `kind: ai` closure the readme **is**
    the program, so a variable named in prose is one the agent will try to read.
    There is no comment/code distinction there to draw.
    """
    if path.suffix == ".md":
        return text
    return "\n".join("" if line.lstrip().startswith("#") else line for line in text.splitlines())


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", default=".", help="the e2e-flow package directory")
    a = ap.parse_args(argv)
    pkg = pathlib.Path(a.package).resolve()

    shared = yaml.safe_load((pkg / "shared.yaml").read_text())
    authority: dict[str, str] = {}
    for agent in _agents(shared):
        if agent.get("name") == AUTHORITY:
            authority = _e2e(agent)
    if not authority:
        print(f"check_agent_env: no `{AUTHORITY}` agent with an E2E_ block in shared.yaml", file=sys.stderr)
        return 1

    problems: list[str] = []
    notes: list[str] = []
    checked = 0

    for step in sorted((pkg / "steps").glob("*.yaml")):
        doc = yaml.safe_load(step.read_text())
        for agent in _agents(doc):
            if agent.get("kind") != "ai":
                continue
            checked += 1
            name = str(agent.get("name"))
            declared = _e2e(agent)

            # 1. divergence — **only where both declare a name.**
            #
            # A stage-specific knob is not a violation. m5 declares eight —
            # `E2E_NEEDLE_DEPTHS`, `E2E_GSM8K_DATA` and friends — that no other
            # stage has any use for, and requiring them on `runner` would make
            # eight false failures and grow `runner` by eight names one stage
            # reads. Their question before this landed, and they were right.
            for key, value in sorted(declared.items()):
                if key not in authority:
                    continue
                if (name, key) in DELIBERATE:
                    continue
                if authority[key] != value:
                    problems.append(
                        f"{step.name}: {name} declares {key}={value!r} but `{AUTHORITY}` "
                        f"has {authority[key]!r}. Two declarations of one default is how "
                        f"they drift; make them byte-identical or say why here."
                    )

            # 2. omission — the half that catches the bug this was written for
            dirs = _task_dirs(doc, name, pkg / "assets")
            if not dirs:
                # **A clean result from an instrument that looked at nothing is
                # not a clean result.** This exact state shipped once; see
                # `_task_dirs`.
                problems.append(
                    f"{step.name}: found no task assets for {name}, so the omission "
                    f"check looked at nothing and its silence means nothing. "
                    f"Fix the closure->assets mapping before believing this agent passed."
                )
                continue
            refs = _referenced(dirs, pkg / "assets" / "lib")
            for key in sorted(refs):
                if key in declared:
                    continue
                if key not in authority:
                    # **Reported, not failed, and not silent.** A name no spec
                    # declares is usually one the body sets for itself —
                    # `E2E_KIT_RUN_TAG`, `E2E_ARM`, `E2E_OUTPUT_DIR`. But not
                    # always: `E2E_MEASURE_GPU` is m3's and m4 reads it, which
                    # is a real cross-owner gap. Failing on all of them is
                    # noise; skipping all of them silently is how the gap hides.
                    notes.append(
                        f"{step.name}: {name} reads {key}, which no spec declares "
                        f"({refs[key].split('/e2e-flow/')[-1]}). Body-set, or a gap?"
                    )
                    continue
                problems.append(
                    f"{step.name}: {name} does NOT declare {key}, which its own assets "
                    f"read ({refs[key].split('/e2e-flow/')[-1]}). A `kind: ai` agent "
                    f"receives only its own env block, so this arrives EMPTY and the "
                    f"body silently takes whatever default it wrote."
                )

    for line in notes:
        print(f"check_agent_env: note: {line}", file=sys.stderr)
    for line in problems:
        print(f"check_agent_env: {line}", file=sys.stderr)
    if problems:
        print(
            f"check_agent_env: {len(problems)} problem(s) across {checked} `kind: ai` agent(s)",
            file=sys.stderr,
        )
        return 1
    print(f"check_agent_env: {checked} `kind: ai` agent(s) agree with `{AUTHORITY}` and declare what they read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
