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


def _referenced(dirs: list[pathlib.Path]) -> dict[str, str]:
    """`E2E_*` names appearing anywhere under those directories -> where."""
    seen: dict[str, str] = {}
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in _SUFFIXES:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for name in _REF.findall(text):
                seen.setdefault(name, str(path))
    return seen


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
    checked = 0

    for step in sorted((pkg / "steps").glob("*.yaml")):
        doc = yaml.safe_load(step.read_text())
        for agent in _agents(doc):
            if agent.get("kind") != "ai":
                continue
            checked += 1
            name = str(agent.get("name"))
            declared = _e2e(agent)

            # 1. divergence
            for key, value in sorted(declared.items()):
                if key not in authority:
                    problems.append(
                        f"{step.name}: {name} declares {key} which `{AUTHORITY}` does not. "
                        f"One of the two is the authority and it is not this one."
                    )
                elif authority[key] != value:
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
            refs = _referenced(dirs)
            for key in sorted(refs):
                if key in declared:
                    continue
                if key not in authority:
                    continue  # a name nothing declares is a separate problem
                problems.append(
                    f"{step.name}: {name} does NOT declare {key}, which its own assets "
                    f"read ({refs[key].split('/e2e-flow/')[-1]}). A `kind: ai` agent "
                    f"receives only its own env block, so this arrives EMPTY and the "
                    f"body silently takes whatever default it wrote."
                )

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
