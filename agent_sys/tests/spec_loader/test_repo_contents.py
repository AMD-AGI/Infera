"""Main spec criterion 5 — this repository contains no workflow-specific spec.

*"Every handoff kind, validator, task, agent, and closure here is either a
schema, or a general spec (§4.5), or the demo package's (demo spec §1.1)."*

The rule it protects is main spec §4.3's, and it is the one easiest to lose one
file at a time: *"a `collect_trace` handoff kind living in this repository would
make every change to that workflow a change to the system."* Nothing in this
repository needs updating to add, change, or retire a workflow — and a test is
how that stays true, because the first violation always looks like a
convenience.

## The scan was aimed at a format nobody can add any more

Measured, with each plant removed in a `finally`
(`scratch/ui-yaml-2026-08/w5/probe_stray_spec_guard.py`, reproduced before this
was changed):

    baseline, nothing planted            1 passed
    a stray .jsonnet in closure/         1 FAILED    <- aimed, not dead
    a stray .yaml    in closure/         1 passed    <- the hole

So the guard was **not vacuous in general** — it still catches the format it was
written for. It was blind to the only format a package can now be written in,
which is a different defect with a different repair: **widen the scan, keep
`.jsonnet` in it.** Rewriting the test would throw away the tripwire criterion 17
asks for — *"a deletion that is 95% done is the state in which a second format
quietly comes back."*

## A `.yaml` glob would be wrong, and the measurement says so

Ten `.yaml`/`.yml` files live under `agent_sys/` with `scratch/` pruned, and
**one of them is not a spec**: `env_mgr/recipes/sglang.repo.yaml`. A blind
extension scan reports it as a stray workflow-specific spec, which is a **false
positive**, and a false positive here condemns a valid repository. The costs are
not symmetric — a missed stray is caught by the next reader, a spurious one makes
the suite lie.

So a `.yaml` is a spec source **iff it declares `module:`**, which is the
discriminator the format itself uses (`refine.task_package.define.md` §1.1.1: an
object is a validator because it says `module: validator`). One definition of
what a spec is, and it is the loader's — this test does not get a second one.

## The shape of the assertion, which is the part that rots quietly

| shape | what an empty scan means |
|---|---|
| `assert not scan()` | still works. Empty **is** the claim, and it fails the day something appears |
| `assert not [p for p in scan() if pred]` | vacuous. The claim is about each item, and there are none |

This test is the second shape and always was, so it needs the non-vacuity
assertion that `tests/env_mgr/test_imports.py:225` already carries for a
README-derived set — *"the README cites no tests at all; the mapping has been
lost"*. Copied rather than invented, because a local precedent is worth more than
an argument.

`tests/interfaces/test_import_rules.py`'s criterion-17 guard is the **first**
shape and needs no such assertion; the two look alike and behave oppositely.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spec_loader import MODULE_KEY
from spec_loader.yaml_source import read_yaml

AGENT_SYS = Path(__file__).resolve().parents[2]

#: Directory names that make a spec source legitimate, matched as a path
#: *component* rather than as a prefix.
#:
#: `general_specs` is main spec §4.5's — workflow-independent specs, the
#: degenerate case of a template whose `config` is empty. It is matched anywhere
#: on purpose: `docs/design.md` §2 projects one at the `agent_sys/` top level and
#: `validator` has already shipped three under `validator/general_specs/`. §4.5
#: asks only that they live in *their own* directory, separate from any task
#: package, and both layouts satisfy that. The criterion is about what a spec
#: *is*, not where it sits.
#:
#: `tests` holds fixtures, which are inputs to a test rather than specs the
#: system ships.
PERMITTED_COMPONENTS = {"general_specs", "tests"}

#: The demo's package roots — YAML and data, deliberately not a Python package
#: (`docs/design.md` D7), and the only workflow-specific specs this repository
#: may hold (`demo` spec §1.1, criterion 5's *"or the demo package's"*).
#:
#: **Two roots for one demo, and the second exists because of W3's own scan
#: rule.** `examples/demo-broken/` used to be `examples/demo/broken/`, inside the
#: package. `YamlPackage` scans every `*.yaml` under a root except `assets/`, so a
#: deliberately-broken document nested inside the good package would be loaded on
#: every ordinary run — `demo` criterion 13's *"two runs, no hand-editing"* gone.
#: `cli/package.py:36-42` records the move; nothing but
#: `cli.package.broken_package()` reaches it, for `--dry-run --with-broken`.
#:
#: **Enumerated, and deliberately not widened to `examples/`.** Criterion 5 says
#: *"the demo package's"*, and both of these are the demo's. A blanket
#: `examples/` prefix would let anyone add a workflow package under it forever
#: with nothing to say so, which is the pressure this test exists to apply —
#: *"the first violation always looks like a convenience"*. Enumeration means a
#: third example root has to be argued for in this file. **If the rule is meant
#: to be "any task package under `examples/`", that is a specification change and
#: `spec-author`'s**; reported rather than taken.
#:
#: **`examples/demo2/` is the third, and this is the argument the paragraph above
#: demands.** It is a *second reference package*, not a second workflow: it is
#: package data, imported by nobody and importing nothing from `agent_sys`, and
#: it ships for the same reason `examples/demo/` does — to be run. What it adds
#: is shape `examples/demo/` structurally cannot show. Demo-1 is four tasks in a
#: single flat subgraph, so it exercises no fan-out (N consumers of one produced
#: kind), no fan-in (one consumer of several distinct kinds) and no depth-2
#: nesting (a non-leaf inside a non-leaf). Those three are where `task_graph`'s
#: edge derivation is least covered by an end-to-end run, and one of them —
#: fan-in needing DISTINCT kinds, because `producer_of[kind]` is single-slot
#: (`task_graph/models.py:360`) — is a constraint no test in this repository
#: currently runs into. The list stays **closed** at three: the pressure this
#: guard applies is that a fourth root has to be argued for here too.
PERMITTED_PREFIXES = ("examples/demo/", "examples/demo-broken/", "examples/demo2/")

#: Not part of the distribution, and **pruned rather than filtered**. `scratch/`
#: is gitignored working space holding cloned third-party trees, and walking into
#: one takes minutes — `pyproject.toml` excludes it from collection for the same
#: reason. A filter that visits first and discards after would make this test the
#: slowest in the suite.
IGNORED = {"scratch", ".git", "build", "dist", "__pycache__", ".ruff_cache", ".pytest_cache"}


#: The deleted format. Kept in the scan with nothing left to find, because
#: criterion 17 asks for a tripwire and an empty result *is* the claim here: the
#: day a `.jsonnet` reappears anywhere but `scratch/`, this goes red.
_DEAD_FORMAT = (".jsonnet", ".libsonnet")

#: The live one. An extension alone is not enough — see the module docstring.
_LIVE_FORMAT = (".yaml", ".yml")


def _declares_an_object(path: Path) -> bool:
    """Does this YAML declare a spec, or is it just a YAML file?

    Read with the same parser and keyed on the same discriminator the loader
    uses, so there is one definition of "a spec" in the system and this test does
    not carry a second (`engineer_principle.md` §1). A root mapping and a root
    list are both handled, which is the two shapes a spec file may take.

    A file that does not parse answers `False`. That is a real limit and not a
    silent one: an unparseable file cannot be identified as a workflow spec by
    anything, and reporting it here as a stray would name the wrong fault. It is
    also unreachable in practice — the only malformed YAML in the tree is written
    to `tmp_path` by a test, and `tmp_path` is not under `agent_sys/`.
    """
    doc, problems = read_yaml(path, origin=str(path))
    if problems or doc is None:
        return False
    if isinstance(doc, dict):
        return MODULE_KEY in doc
    if isinstance(doc, list):
        return any(isinstance(item, dict) and MODULE_KEY in item for item in doc)
    return False


def _spec_sources() -> list[Path]:
    out = []
    for root, dirs, names in os.walk(AGENT_SYS):
        dirs[:] = [d for d in dirs if d not in IGNORED]
        for name in names:
            path = Path(root) / name
            if name.endswith(_DEAD_FORMAT) or (
                name.endswith(_LIVE_FORMAT) and _declares_an_object(path)
            ):
                out.append(path.relative_to(AGENT_SYS))
    return sorted(out)


def test_the_scan_finds_something_to_check() -> None:
    """**The non-vacuity assertion, and it is the whole repair.**

    `test_repo_holds_only_schemas_general_and_demo` is
    `assert not [p for p in scan() if pred]`: its claim is *about each item*, so
    an empty scan satisfies it while measuring nothing. That is what happened
    when the format changed under a `.jsonnet`-only scan — the guard stayed green
    and stopped guarding, and only a planted file revealed it.

    Copied from `tests/env_mgr/test_imports.py:225`, which carries the same one
    line for a README-derived set: *"the README cites no tests at all; the
    mapping has been lost"*. A local precedent rather than an invention.

    The two numbers are named so the failure says which half broke — a scan that
    finds no YAML has lost the format, and one that finds no `.jsonnet` is
    expected and is criterion 17 holding.
    """
    found = _spec_sources()

    assert found, (
        "the scan found no spec source anywhere in agent_sys/. Either the "
        "repository really holds none — it holds the demo's and the general "
        "specs' — or _spec_sources() no longer recognises the format, which is "
        "how this file's guard went green and stopped guarding once before."
    )
    live = [p for p in found if p.suffix in _LIVE_FORMAT]
    assert live, f"no spec source in the live format; scan found only {found}"


def test_repo_holds_only_schemas_general_and_demo() -> None:
    stray = [
        path
        for path in _spec_sources()
        if not (set(path.parts) & PERMITTED_COMPONENTS)
        and not str(path).startswith(PERMITTED_PREFIXES)
    ]

    assert not stray, (
        f"spec sources outside {sorted(PERMITTED_COMPONENTS)} / "
        f"{list(PERMITTED_PREFIXES)}: {[str(p) for p in stray]}. "
        f"Main spec §4.3: a workflow's specs live in a task package, outside this "
        f"repository, so nothing here changes to add, change, or retire one."
    )


def test_the_schemas_are_inside_the_installed_package() -> None:
    """`docs/design.md` D1, which is a correction to the README's projection.

    The README put `schemas/` at the `agent_sys/` top level and **that is not
    installable**: a bare directory of `.json` files has no `__init__.py`, so
    `find_packages` cannot see it and setuptools will not ship it. Anything
    reading it by relative path works from a git checkout and dies from a wheel.
    """
    assert not (AGENT_SYS / "schemas").exists(), (
        "schemas at the top level are not installable — docs/design.md D1"
    )
    bundled = sorted(p.name for p in (AGENT_SYS / "spec_loader" / "schemas").glob("*.json"))
    assert bundled == [
        "_common.schema.json",
        "agent.schema.json",
        "closure.schema.json",
        "handoff.schema.json",
        "task.schema.json",
        "validator.schema.json",
    ]


def test_the_schemas_are_readable_as_package_resources() -> None:
    """Read through `importlib.resources`, which behaves the same from a
    checkout, a wheel, and a zipimport — the whole reason for D1's move."""
    from importlib.resources import files

    from spec_loader import KINDS

    for kind in KINDS:
        resource = files("spec_loader") / "schemas" / f"{kind}.schema.json"
        assert resource.is_file(), kind


@pytest.mark.parametrize("marker", ["package-data", "spec_loader*"])
def test_pyproject_ships_the_schemas(marker: str) -> None:
    """A schema that is not packaged is a loader that works only from a checkout.

    Both halves are needed and neither is implied by the other: `packages.find`
    has to see `spec_loader`, and `package-data` has to carry the `.json` files
    inside it.
    """
    assert marker in (AGENT_SYS / "pyproject.toml").read_text()
